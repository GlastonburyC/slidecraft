"""
Join the shards of a sharded prediction back into one expression map.

    python scripts/merge_shards.py slide.shard*.bin --out slide.expression.bin

`predict_expression.py --shard I/N` splits a slide's patches into N contiguous
blocks so several GPUs can work on it at once. Each writes a complete, valid map
of its own block; this concatenates them in shard order, which restores the
original patch order because the blocks partition the list in sequence.

The checks are the point. A merge that quietly accepts mismatched pieces
produces a file that opens, draws, and is wrong — patches from one stride glued
to values from another, with nothing to show for it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

MAGIC = b"SCEXPR1\x00"


def read_header(path: pathlib.Path):
    with path.open("rb") as fh:
        if fh.read(8) != MAGIC:
            raise SystemExit(f"{path} is not a Slidecraft expression map.")
        length = struct.unpack("<I", fh.read(4))[0]
        return json.loads(fh.read(length)), 12 + length


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("shards", nargs="+", help="the .bin files to join")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = [pathlib.Path(x) for x in args.shards]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("Not found: " + ", ".join(str(p) for p in missing), file=sys.stderr)
        return 1

    pieces = []
    for path in paths:
        header, offset = read_header(path)
        if "shard" not in header:
            print(f"{path} was not written with --shard.", file=sys.stderr)
            return 1
        pieces.append((header["shard"][0], header, offset, path))
    pieces.sort(key=lambda x: x[0])

    first = pieces[0][1]
    total = first["shard"][1]

    # Every shard, exactly once. A missing one leaves a hole in the slide that
    # nothing downstream would notice; a duplicate doubles part of it.
    seen = [p[0] for p in pieces]
    if seen != list(range(1, total + 1)):
        print(f"Expected shards 1..{total}, got {seen}.", file=sys.stderr)
        return 1

    for index, header, _, path in pieces:
        for key in ("genes", "side", "dtype", "slide", "modelId"):
            if header.get(key) != first.get(key):
                print(f"{path} disagrees about {key}; these are not one run.",
                      file=sys.stderr)
                return 1

    genes = first["genes"]
    itemsize = 4 if first.get("dtype") == "float32" else 2
    patches = [p for _, h, _, _ in pieces for p in h["patches"]]

    merged = dict(first)
    merged.pop("shard", None)
    merged["patches"] = patches
    blob = json.dumps(merged).encode()

    out = pathlib.Path(args.out)
    written = 0
    with out.open("wb") as dst:
        dst.write(MAGIC)
        dst.write(struct.pack("<I", len(blob)))
        dst.write(blob)
        for index, header, offset, path in pieces:
            want = len(header["patches"]) * len(genes) * itemsize
            with path.open("rb") as src:
                src.seek(offset)
                # Copied in chunks: a shard of a strided whole-transcriptome run
                # is gigabytes, and there is no reason for it to pass through
                # memory on the way.
                left = want
                while left:
                    block = src.read(min(1 << 24, left))
                    if not block:
                        print(f"{path} is short by {left} bytes.", file=sys.stderr)
                        return 1
                    dst.write(block)
                    left -= len(block)
            written += want
            print(f"  shard {index}/{total}: {len(header['patches']):,} patches")

    expected = 12 + len(blob) + written
    if out.stat().st_size != expected:
        print(f"Wrote {out.stat().st_size} bytes, expected {expected}.", file=sys.stderr)
        return 1

    print(f"Wrote {out} — {len(patches):,} patches x {len(genes):,} genes "
          f"({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
