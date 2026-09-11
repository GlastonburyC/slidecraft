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
import os
import pathlib
import shlex
import struct
import sys

# The launcher lives beside this file, which is also how it reaches the cluster.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

MAGIC = b"SCEXPR1\x00"
DEFAULT_GENES = ["EPCAM", "PTPRC", "CD3D", "COL1A1", "MKI67", "VIM", "KRT19", "ACTA2"]

# The gene router draws its projections from one of five frozen biological
# embedding spaces -- DNA, RNA, protein, single-cell and text. The checkpoint
# carries all five and refuses to guess, so one has to be named.
SOURCES = ["evo2", "orthrus", "prott5", "scgpt", "apertus"]


def submit_to_cluster(args) -> int:
    """
    Hand this run to Slurm rather than doing it here.

    The arguments that describe *what* to predict are forwarded untouched; the
    ones that describe *where* are consumed here. Keeping that split means the
    remote run is the same command you would have run locally, which is what
    makes a failure on the node reproducible on your laptop.
    """
    from hpc import Job, submit

    slide = pathlib.Path(args.slide)
    if not args.remote_slide and not slide.exists():
        print(f"{slide} not found. Use --remote-slide for a slide already on the cluster.",
              file=sys.stderr)
        return 1

    forwarded: list[str] = ["--repo", args.repo,
                            "--source", args.source,
                            "--target-mpp", str(args.target_mpp),
                            "--patch", str(args.patch),
                            "--batch", str(args.batch),
                            "--min-tissue", str(args.min_tissue),
                            "--device", args.device]
    if args.all:
        forwarded.append("--all")
    else:
        forwarded += ["--genes", *args.genes]
    if args.stride:
        forwarded += ["--stride", str(args.stride)]
    if args.fp32:
        forwarded.append("--fp32")

    env = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        # Forwarded so the gated weights can be fetched on the node, and
        # exported inside the script rather than passed on the command line.
        env["HF_TOKEN"] = token
    else:
        print("HF_TOKEN is not set here; the node will need its own "
              "`hf auth login` or the weights already cached.", file=sys.stderr)

    job = Job(
        host=args.submit,
        remote_dir=args.remote_dir,
        partition=args.partition,
        gres=args.gres,
        time_limit=args.time_limit,
        cpus=args.cpus,
        mem=args.mem,
        python=args.remote_python,
        account=args.account,
        modules=args.module,
        env=env,
        # A value like "-S ~/.ssh/cm-host" is one argument to argparse but two
        # to ssh, and `~` has to be expanded here because there is no shell in
        # between to do it.
        ssh_options=[
            os.path.expanduser(part)
            for opt in args.ssh_option
            for part in shlex.split(opt)
        ],
    )
    return submit(job, slide, forwarded,
                  remote_slide=args.remote_slide,
                  watch=not args.no_watch,
                  poll=args.poll,
                  dry_run=args.dry_run)



def warm_up(model, image_processor, genes, all_genes, patch, device) -> None:
    """
    Run one throwaway tile before the real loop, and route around cuDNN if it
    cannot handle this GPU.

    The backbone is a DINOv2-giant whose patch embedding is a 14x14 stride-14
    convolution into 1536 channels. cuDNN 9 ships no engine for that shape on
    Volta, so the first forward dies with "GET was unable to find an engine to
    execute this computation" -- after the slide has been tiled and the weights
    loaded, which is an expensive place to discover it.

    Disabling cuDNN costs almost nothing: that convolution is the only one in
    the model, and with stride equal to kernel it is a reshape and a matrix
    multiply either way. Everything after it is attention and linear layers,
    which never touched cuDNN. So probe with the real computation, and fall
    back only if it genuinely fails.
    """
    import torch
    from PIL import Image

    probe = Image.new("RGB", (patch, patch), (200, 180, 200))
    batch = image_processor(probe).unsqueeze(0).to(device)

    def forward():
        with torch.inference_mode():
            if all_genes:
                model(batch)
            else:
                model.predict_genes(batch, genes)

    try:
        forward()
        return
    except RuntimeError as err:
        if "unable to find an engine" not in str(err):
            raise
    torch.backends.cudnn.enabled = False
    print("cuDNN has no kernel for this backbone on this GPU; running without it.",
          file=sys.stderr)
    forward()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("slide", help="Path to the whole-slide image")
    ap.add_argument("--repo", default="ratschlab/DeepSpotM")
    ap.add_argument("--source", default="scgpt", choices=SOURCES,
                    help="Which frozen gene-embedding pathway conditions the "
                         "gene router. The five are not interchangeable, so "
                         "the choice is recorded in the output header.")
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

    hpc = ap.add_argument_group(
        "cluster",
        "Run it on a Slurm cluster instead of here: copies what is needed, submits, "
        "waits, and brings the map back. Authentication is your own SSH config and "
        "agent — no password is asked for or stored.",
    )
    hpc.add_argument("--submit", metavar="HOST",
                     help="SSH host or alias to submit to, e.g. a Host entry in ~/.ssh/config")
    hpc.add_argument("--partition", default="gpuq")
    hpc.add_argument("--gres", default="gpu:1")
    hpc.add_argument("--time-limit", default="08:00:00")
    hpc.add_argument("--cpus", type=int, default=8)
    hpc.add_argument("--mem", default="64G")
    hpc.add_argument("--account", default=None)
    hpc.add_argument("--remote-dir", default="~/slidecraft",
                     help="Working directory on the cluster")
    hpc.add_argument("--remote-python", default="python",
                     help="Interpreter on the compute node; usually a venv or conda python")
    hpc.add_argument("--remote-slide", default=None,
                     help="Path to the slide already on cluster storage, so it is not copied")
    hpc.add_argument("--module", action="append", default=[], metavar="NAME",
                     help="module load NAME on the node; repeatable")
    hpc.add_argument("--no-watch", action="store_true",
                     help="Submit and exit rather than waiting and fetching")
    hpc.add_argument("--poll", type=int, default=30, help="Seconds between status checks")
    hpc.add_argument("--ssh-option", action="append", default=[], metavar="OPT",
                     help="Passed to ssh and to rsync's ssh; repeatable, and a single value may "
                          "hold several words. Use it to reuse a connection you have already "
                          "authenticated, e.g. --ssh-option='-S ~/.ssh/cm-host'")
    hpc.add_argument("--dry-run", action="store_true",
                     help="Print the batch script and every command, and send nothing")

    args = ap.parse_args()

    if args.submit:
        return submit_to_cluster(args)

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

    print(f"Loading {args.repo} (source {args.source}) …")
    model, image_processor = DeepSpotM.from_pretrained(args.repo, source=args.source)
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

    warm_up(model, image_processor, genes, args.all, args.patch, args.device)

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
        "model": f"DeepSpot-M {args.source} ({len(genes)} genes)",
        # The source is part of the identity, not a footnote: the same tile run
        # through two pathways gives two different numbers, and a map that does
        # not say which one it came from cannot be compared with another.
        "modelId": f"deepspot-m-{args.source}-{len(genes)}",
        "source": args.source,
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
