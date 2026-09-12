"""
Convert a Slidecraft expression map to AnnData, and optionally to SpatialData.

    python scripts/expression_to_anndata.py slide.expression.bin

Writes `slide.h5ad` beside it. The browser exports AnnData directly for maps it
can hold, which is most of them; this exists for the whole transcriptome, where
32,000 patches by 19,338 genes is 2.5 GB once widened to float32 and no tab
should be assembling it.

The matrix is written a block of patches at a time rather than materialised,
so the peak cost is one block, not the whole map.

    --spatialdata   also write a SpatialData zarr, with the map as a table of
                    points. Needs `pip install spatialdata`.
    --genes A B C   keep only these, which is usually what makes a whole
                    transcriptome workable downstream.

These are predictions from morphology, not measurements. That is recorded in
`uns` so it travels with the file.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

MAGIC = b"SCEXPR1\x00"
# Patches per block. 4,000 x 19,338 as float32 is about 300 MB.
BLOCK = 4000


def read_header(path: pathlib.Path):
    with path.open("rb") as fh:
        if fh.read(8) != MAGIC:
            raise SystemExit(f"{path} is not a Slidecraft expression map.")
        length = struct.unpack("<I", fh.read(4))[0]
        header = json.loads(fh.read(length))
        return header, 12 + length


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("map", help="a .expression.bin written by predict_expression.py")
    ap.add_argument("--out", default=None, help="defaults to <map>.h5ad")
    ap.add_argument("--genes", nargs="*", default=None,
                    help="keep only these genes, in this order")
    ap.add_argument("--spatialdata", action="store_true",
                    help="also write a SpatialData zarr beside the h5ad")
    args = ap.parse_args()

    try:
        import anndata
        import numpy as np
        import pandas as pd
    except ImportError as err:
        print(f"Missing dependency: {err}\n  pip install anndata", file=sys.stderr)
        return 1

    path = pathlib.Path(args.map)
    header, offset = read_header(path)
    genes = list(header["genes"])
    patches = header["patches"]
    n, g = len(patches), len(genes)
    stored = header.get("dtype", "float16")
    itemsize = {"float32": 4, "float16": 2, "uint8": 1}.get(stored)
    if itemsize is None:
        print(f"Unknown dtype in the header: {stored}", file=sys.stderr)
        return 1
    dtype = {"float32": np.float32, "float16": np.float16, "uint8": np.uint8}[stored]
    # A quantised map carries one scale and one zero per gene; see
    # scripts/subset_expression.py for why it is stored that way.
    scale = np.asarray(header.get("scale", []), dtype=np.float32)
    zero = np.asarray(header.get("zero", []), dtype=np.float32)
    if stored == "uint8" and (len(scale) != g or len(zero) != g):
        print(
            f"This uint8 map lists {len(scale)} scales and {len(zero)} zeros "
            f"for {g} genes; it cannot be decoded.",
            file=sys.stderr,
        )
        return 1

    keep = None
    if args.genes:
        missing = [x for x in args.genes if x not in genes]
        if missing:
            print(f"Not in this map: {', '.join(missing)}", file=sys.stderr)
            return 1
        keep = np.array([genes.index(x) for x in args.genes])
        genes = list(args.genes)
        if stored == "uint8":
            scale = scale[keep]
            zero = zero[keep]

    out_genes = len(genes)
    print(f"{n:,} patches x {g:,} genes"
          + (f" -> keeping {out_genes}" if keep is not None else ""))

    X = np.empty((n, out_genes), dtype=np.float32)
    with path.open("rb") as fh:
        for start in range(0, n, BLOCK):
            rows = min(BLOCK, n - start)
            fh.seek(offset + start * g * itemsize)
            block = np.frombuffer(fh.read(rows * g * itemsize), dtype=dtype).reshape(rows, g)
            if keep is not None:
                block = block[:, keep]
            if stored == "uint8":
                X[start : start + rows] = block.astype(np.float32) * scale + zero
            else:
                X[start : start + rows] = block
            print(f"\r  {min(start + rows, n):,}/{n:,}", end="", flush=True)
    print()

    xs = np.array([p["x"] for p in patches], dtype=np.int64)
    ys = np.array([p["y"] for p in patches], dtype=np.int64)
    side = header["side"]

    obs = pd.DataFrame({"x": xs, "y": ys},
                       index=pd.Index([f"{x}_{y}" for x, y in zip(xs, ys)], name="patch"))
    var = pd.DataFrame(index=pd.Index(genes, name="gene"))

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    # Centres, not corners: this is what every plotting routine expects.
    adata.obsm["spatial"] = np.column_stack([xs + side / 2, ys + side / 2]).astype(np.float32)
    adata.uns.update({
        "slide": header.get("slide", ""),
        "model": header.get("model", ""),
        "model_id": header.get("modelId", ""),
        "source": header.get("source", ""),
        "created_at": header.get("createdAt", ""),
        "patch_side_px": str(side),
        "mpp": str(header.get("mpp", "")),
        "units": "level-0 slide pixels",
        "provenance": "Predicted from H&E by Slidecraft. Not a measurement.",
    })

    out = pathlib.Path(args.out) if args.out else path.with_suffix("").with_suffix(".h5ad")
    adata.write_h5ad(out)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    if args.spatialdata:
        try:
            import spatialdata
            from spatialdata.models import PointsModel, TableModel
        except ImportError:
            print("SpatialData is not installed: pip install spatialdata", file=sys.stderr)
            return 1
        points = PointsModel.parse(
            pd.DataFrame({"x": adata.obsm["spatial"][:, 0],
                          "y": adata.obsm["spatial"][:, 1]})
        )
        # region/instance columns are how a table says which geometry it
        # annotates; without them SpatialData takes it as unattached.
        adata.obs["region"] = pd.Categorical(["patches"] * n)
        adata.obs["instance_id"] = np.arange(n)
        table = TableModel.parse(adata, region="patches",
                                 region_key="region", instance_key="instance_id")
        sd_out = out.with_suffix(".zarr")
        spatialdata.SpatialData(points={"patches": points}, tables={"expression": table}).write(sd_out)
        print(f"Wrote {sd_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
