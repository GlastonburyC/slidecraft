"""
Cut a whole-transcriptome expression map down to something a browser can hold.

    python scripts/subset_expression.py "slide.expression.bin" --out "slide.small.bin"

A whole-slide run at quarter stride is 306,587 patches by 19,338 genes: 11.9 GB
as fp16, which is ten times what a tab survives. Two things are wasted in it.

**Most of the genes are not expressed.** The decoder emits a number for every
gene in the reference, whether or not that gene is transcribed in colon — so the
map carries olfactory receptors and testis-specific antigens at the same cost
per value as COL1A1. They are not merely uninteresting; they are noise being
ranked against signal in every differential test, which is the flaw that put the
expression floor into the enrichment code in the first place. Dropping them here
is the same judgement applied earlier, where it also saves the bytes.

**fp16 is finer than the model is.** DeepSpot-M predicts log1p-normalised
expression from a 224-pixel tile; its disagreement with held-out truth is a
substantial fraction of the value. Storing that to eleven bits of mantissa
records the decoder's arithmetic, not the biology. One byte per value with a
per-gene scale keeps 255 steps across a gene's observed range, which is finer
than the colour ramp that displays it and finer than any test that ranks it.

Together they turn 11.9 GB into something that loads. Precision is a choice
here, not an assumption: `--dtype float16` keeps the original precision and
subsets only, and the tool reports the error it introduced either way.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import struct
import sys

import numpy as np

MAGIC = b"SCEXPR1\x00"
BLOCK = 4000  # patches read at once; 4000 x 19,338 fp16 is 155 MB

DTYPES = {"float16": np.float16, "float32": np.float32, "uint8": np.uint8}


def read_header(path: pathlib.Path) -> tuple[dict, int]:
    with path.open("rb") as fh:
        if fh.read(8) != MAGIC:
            raise SystemExit(f"{path} is not a Slidecraft expression map.")
        length = struct.unpack("<I", fh.read(4))[0]
        return json.loads(fh.read(length)), 12 + length


def open_values(path: pathlib.Path, header: dict, offset: int) -> np.memmap:
    """The value block, as a patches-by-genes array that is never fully read."""
    rows = len(header["patches"])
    cols = len(header["genes"])
    dtype = DTYPES[header["dtype"]]
    if header["dtype"] == "uint8":
        raise SystemExit(
            f"{path} is already quantised. Subset the fp16 map it came from, "
            "so the error is introduced once rather than compounded."
        )
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(rows, cols))


def survey(values: np.memmap, quiet: bool):
    """Per-gene mean, spread, minimum and maximum, in one pass over the file."""
    rows, cols = values.shape
    total = np.zeros(cols, np.float64)
    squares = np.zeros(cols, np.float64)
    lo = np.full(cols, np.inf, np.float32)
    hi = np.full(cols, -np.inf, np.float32)
    for start in range(0, rows, BLOCK):
        block = np.asarray(values[start : start + BLOCK], dtype=np.float32)
        total += block.sum(axis=0, dtype=np.float64)
        squares += np.square(block, dtype=np.float32).sum(axis=0, dtype=np.float64)
        lo = np.minimum(lo, block.min(axis=0))
        hi = np.maximum(hi, block.max(axis=0))
        if not quiet:
            print(f"  surveyed {min(start + BLOCK, rows):,}/{rows:,}", end="\r", flush=True)
    if not quiet:
        print(" " * 40, end="\r")
    mean = total / rows
    sd = np.sqrt(np.maximum(0.0, squares / rows - mean * mean))
    return mean.astype(np.float32), sd.astype(np.float32), lo, hi


def chosen(
    genes: list[str], mean: np.ndarray, sd: np.ndarray, args: argparse.Namespace
) -> np.ndarray:
    """The gene indices to keep, as a sorted array."""
    if args.genes:
        wanted = load_gene_list(args.genes)
        at = {g: i for i, g in enumerate(genes)}
        missing = [g for g in wanted if g not in at]
        if missing:
            print(
                f"Not in this map: {', '.join(missing[:10])}"
                + (f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""),
                file=sys.stderr,
            )
        keep = np.array(sorted(at[g] for g in wanted if g in at), dtype=np.int64)
    else:
        keep = np.flatnonzero(mean >= args.min_expression)

    if args.top and len(keep) > args.top:
        """
        Spread, not level, decides which survivors are worth the bytes.

        A gene sitting flat at 2.0 across every patch is highly expressed and
        draws a uniform field — nothing to see, and nothing for a differential
        test to find either. A gene confined to the crypt base has a lower mean
        and is the whole reason to look. Ranking by standard deviation keeps the
        second kind, in the units the colour ramp is drawn in. `--rank mean`
        asks for the loudest instead.

        The expression floor still applies first, so this cannot resurrect a
        gene that is only varying because it is noise near zero.
        """
        by = mean if args.rank == "mean" else sd
        order = keep[np.argsort(-by[keep])][: args.top]
        # Written back in map order, so the output's gene list still reads the
        # way the original did.
        keep = np.sort(order)
    return keep


def load_gene_list(spec: str) -> list[str]:
    path = pathlib.Path(spec)
    text = path.read_text() if path.exists() else spec
    return [g for g in text.replace(",", " ").split() if g]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("source", help="the .expression.bin to cut down")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--min-expression",
        type=float,
        default=0.05,
        help="drop genes whose mean across the slide is below this (default 0.05, "
        "the same floor the enrichment tests use)",
    )
    ap.add_argument(
        "--genes",
        help="keep exactly these instead: a file of symbols, or a comma-separated list",
    )
    ap.add_argument(
        "--top",
        type=int,
        help="keep only N of the survivors — use this to hit a size a browser can hold",
    )
    ap.add_argument(
        "--rank",
        choices=["variance", "mean"],
        default="variance",
        help="which survivors --top keeps: the most spatially variable (default), "
        "or the most expressed",
    )
    ap.add_argument(
        "--dtype",
        choices=["uint8", "float16"],
        default="uint8",
        help="uint8 with a per-gene scale (default), or the original fp16",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    source = pathlib.Path(args.source)
    out = pathlib.Path(args.out)
    if not source.exists():
        print(f"Not found: {source}", file=sys.stderr)
        return 1
    if out.resolve() == source.resolve():
        print("Refusing to write over the map being read.", file=sys.stderr)
        return 1

    header, offset = read_header(source)
    values = open_values(source, header, offset)
    genes = header["genes"]
    rows, cols = values.shape
    if not args.quiet:
        print(f"{source.name}: {rows:,} patches x {cols:,} genes, {header['dtype']}")

    mean, sd, lo, hi = survey(values, args.quiet)
    keep = chosen(genes, mean, sd, args)
    if len(keep) == 0:
        print(
            f"No gene has a mean of {args.min_expression} or more. "
            f"The highest is {mean.max():.4f}.",
            file=sys.stderr,
        )
        return 1

    survivors = int((mean >= args.min_expression).sum())
    kept_genes = [genes[i] for i in keep]
    quantise = args.dtype == "uint8"

    """
    A gene's own range, not a shared one.

    One scale over the whole map would spend all 255 steps on the handful of
    structural genes that reach 3 and leave everything else in the bottom two
    or three, which is exactly the dynamic range that matters. Per gene, the
    floor stays at zero where the gene never goes negative — so a patch with no
    expression still reads as exactly zero rather than half a step above it.
    """
    floor = np.minimum(0.0, lo[keep]).astype(np.float32)
    ceiling = np.maximum(hi[keep], floor + 1e-6).astype(np.float32)
    scale = ((ceiling - floor) / 255.0).astype(np.float32)

    out_header = dict(header)
    out_header["genes"] = kept_genes
    """
    The model string carries its own gene count, and it has to follow.

    Slidecraft shows it verbatim, so a subset that kept the original text says
    "(19,338 genes)" over a list of 3,000 — the file describing itself wrongly
    in the one place a reader would look to check.
    """
    out_header["model"] = re.sub(
        r"\(\s*[\d,]+\s+genes?\s*\)",
        f"({len(keep):,} genes)",
        str(header.get("model", "")),
    )
    out_header["dtype"] = args.dtype
    if quantise:
        # float() so json writes numbers, not numpy repr.
        out_header["scale"] = [float(s) for s in scale]
        out_header["zero"] = [float(z) for z in floor]
    else:
        out_header.pop("scale", None)
        out_header.pop("zero", None)
    out_header["subsetOf"] = source.name
    out_header["minExpression"] = args.min_expression if not args.genes else None
    if args.top:
        out_header["rankedBy"] = args.rank
    header_blob = json.dumps(out_header).encode()

    unit = 1 if quantise else 2
    err_max = 0.0
    err_sum = 0.0
    err_n = 0

    with out.open("wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<I", len(header_blob)))
        fh.write(header_blob)
        for start in range(0, rows, BLOCK):
            block = np.asarray(values[start : start + BLOCK], dtype=np.float32)[:, keep]
            if quantise:
                codes = np.clip(np.rint((block - floor) / scale), 0, 255).astype(np.uint8)
                back = codes.astype(np.float32) * scale + floor
                err = np.abs(back - block)
                err_max = max(err_max, float(err.max()))
                err_sum += float(err.sum())
                err_n += err.size
                fh.write(codes.tobytes())
            else:
                fh.write(block.astype(np.float16).tobytes())
            if not args.quiet:
                print(f"  wrote {min(start + BLOCK, rows):,}/{rows:,}", end="\r", flush=True)

    if not args.quiet:
        print(" " * 40, end="\r")
    before = source.stat().st_size
    after = out.stat().st_size

    def size(n: int) -> str:
        """Bytes in a unit that shows them — "0.00 GB" tells the reader nothing."""
        for unit, scale_ in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
            if n >= scale_:
                return f"{n / scale_:.2f} {unit}"
        return f"{n} B"

    if args.genes:
        print(f"Kept {len(keep):,} of {cols:,} genes, as listed")
    else:
        # Two steps, reported as two, because "dropped N below a mean" is not
        # true of the genes --top removed — those cleared the floor.
        print(
            f"{survivors:,} of {cols:,} genes have a mean of {args.min_expression} or more "
            f"({cols - survivors:,} dropped as not expressed)"
        )
        if args.top and args.top < survivors:
            by = "most expressed" if args.rank == "mean" else "most spatially variable"
            print(f"Kept the {len(keep):,} {by} of those")
        else:
            print(f"Kept all {len(keep):,}")
    print(
        f"Wrote {out.name} — {rows:,} patches x {len(keep):,} genes, {args.dtype}, "
        f"{size(after)} (was {size(before)}, {before / max(after, 1):.1f}x smaller)"
    )
    if quantise:
        print(
            f"Quantisation error: {err_sum / err_n:.6f} mean, {err_max:.6f} worst — "
            f"against a value range of {float(ceiling.max()):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
