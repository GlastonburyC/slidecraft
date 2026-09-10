"""
Run DeepSpot-M over a whole slide on a GPU, and write a map Slidecraft can load.

This is the answer to "can I have all 19,338 genes?". In the browser you cannot:
a 1B-parameter ViT-g encoder plus a decoder whose cost scales with gene count is
minutes per patch under WASM. On a GPU it is seconds. So the heavy pass happens
here, once, and the browser gets a file.

    python scripts/predict_expression.py slide.svs --all --device cuda

Writes slide.expression.bin beside the slide. Drop the folder into Slidecraft
and each slide opens with its own map already attached — the same pairing the
tissue GeoJSON sidecars use.

The map is tied to the slide it was computed on: patch coordinates are level-0
pixels of *that* slide, and Slidecraft refuses to draw one over a different
slide rather than silently misplacing it.

Weights: https://huggingface.co/ratschlab/DeepSpotM (gated, CC-BY-NC-SA-4.0).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

MAGIC = b"SCEXPR1\x00"
DEFAULT_GENES = ["EPCAM", "PTPRC", "CD3D", "COL1A1", "MKI67", "VIM", "KRT19", "ACTA2"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("slide", help="Path to the whole-slide image")
    ap.add_argument("--repo", default="ratschlab/DeepSpotM")
    ap.add_argument("--genes", nargs="*", default=DEFAULT_GENES)
    ap.add_argument("--all", action="store_true", help="Every gene the model predicts")
    ap.add_argument("--target-mpp", type=float, default=0.5,
                    help="Magnification to read at; the model was trained near 20x")
    ap.add_argument("--patch", type=int, default=224)
    ap.add_argument("--stride", type=int, default=None, help="Defaults to no overlap")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-tissue", type=float, default=0.25,
                    help="Skip patches with less than this fraction of tissue")
    ap.add_argument("--fp32", action="store_true", help="Store fp32 rather than fp16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        import numpy as np
        import openslide
        import torch
        from PIL import Image
        from deepspotm import DeepSpotM
    except ImportError as err:
        print(
            f"Missing dependency: {err}\n"
            "  pip install torch openslide-python pillow numpy\n"
            "  pip install git+https://github.com/ratschlab/DeepSpotM",
            file=sys.stderr,
        )
        return 1

    slide_path = pathlib.Path(args.slide)
    slide = openslide.OpenSlide(str(slide_path))

    mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    mpp = float(mpp_x) if mpp_x else None
    if mpp is None:
        print("This slide reports no MPP; assuming 0.5 µm/px.", file=sys.stderr)
        mpp = 0.5

    # The level whose scale is nearest what the model was trained at. Reading a
    # 40x tile for a 20x model shows it half the tissue it expects, which
    # changes every prediction without failing.
    level = min(
        range(slide.level_count),
        key=lambda l: abs(slide.level_downsamples[l] * mpp - args.target_mpp),
    )
    downsample = slide.level_downsamples[level]
    side0 = int(round(args.patch * downsample))
    step0 = int(round((args.stride or args.patch) * downsample))
    print(f"Level {level} ({downsample:.1f}x, {downsample * mpp:.3f} µm/px), "
          f"patch {args.patch}px = {side0}px at level 0")

    print(f"Loading {args.repo} …")
    model, image_processor = DeepSpotM.from_pretrained(args.repo)
    model = model.eval().to(args.device)

    gene_names = list(model.gene_names)
    genes = gene_names if args.all else list(args.genes)
    missing = [g for g in genes if g not in gene_names]
    if missing:
        print(f"Not in this model: {', '.join(missing)}", file=sys.stderr)
        return 1
    print(f"{len(genes)} genes")

    width, height = slide.level_dimensions[0]
    thumb = np.asarray(slide.get_thumbnail((1024, 1024)).convert("RGB")).astype(np.float32)
    # Tissue by the same cue Slidecraft uses: distance from blank glass, taking
    # the stronger of saturation and darkness.
    mx = thumb.max(axis=2)
    mn = thumb.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1) * 255, 0)
    dark = 255 - (0.299 * thumb[..., 0] + 0.587 * thumb[..., 1] + 0.114 * thumb[..., 2])
    score = np.maximum(sat, dark)
    tissue = score > max(6, np.percentile(score, 55))
    th, tw = tissue.shape

    coords: list[tuple[int, int]] = []
    for y in range(0, height - side0 + 1, step0):
        for x in range(0, width - side0 + 1, step0):
            y0 = int(y / height * th)
            y1 = max(y0 + 1, int((y + side0) / height * th))
            x0 = int(x / width * tw)
            x1 = max(x0 + 1, int((x + side0) / width * tw))
            if tissue[y0:y1, x0:x1].mean() >= args.min_tissue:
                coords.append((x, y))
    if not coords:
        print("No patches passed the tissue filter.", file=sys.stderr)
        return 1
    print(f"{len(coords):,} patches on tissue")

    values = np.zeros((len(coords), len(genes)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(coords), args.batch):
            chunk = coords[start : start + args.batch]
            tiles = []
            for x, y in chunk:
                tile = slide.read_region((x, y), level, (args.patch, args.patch)).convert("RGB")
                if tile.size != (args.patch, args.patch):
                    tile = tile.resize((args.patch, args.patch), Image.BILINEAR)
                tiles.append(tile)

            batch = torch.stack([image_processor(t) for t in tiles]).to(args.device)
            if args.all:
                out, *_ = model(batch)
            else:
                out = model.predict_genes(batch, genes)
            values[start : start + len(chunk)] = out.float().cpu().numpy()

            done = start + len(chunk)
            print(f"\r{done:,}/{len(coords):,}", end="", flush=True)
    print()

    dtype = "float32" if args.fp32 else "float16"
    header = {
        "slide": slide_path.name,
        "genes": genes,
        "patches": [{"x": x, "y": y} for x, y in coords],
        "side": side0,
        "dtype": dtype,
        "model": f"DeepSpot-M ({len(genes)} genes)",
        "modelId": f"deepspot-m-{len(genes)}",
        "createdAt": __import__("datetime").datetime.now().astimezone().isoformat(),
        "mpp": mpp,
    }
    blob = json.dumps(header).encode()

    out = pathlib.Path(args.out) if args.out else slide_path.with_suffix(".expression.bin")
    with out.open("wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<I", len(blob)))
        fh.write(blob)
        fh.write(values.astype(np.float16 if dtype == "float16" else np.float32).tobytes())

    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("Drop the folder into Slidecraft; it attaches to the slide by name.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
