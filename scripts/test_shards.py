"""
Checks that sharding a run and merging it back is lossless.

    .venv-export/bin/python -m pytest scripts/test_shards.py -q

Sharding splits a slide across GPUs and the merge glues the pieces back. A
mistake there is invisible in the output — a file that opens, draws, and has
somebody else's patches in the middle of it — so the round trip is checked
rather than assumed.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MAGIC = b"SCEXPR1\x00"


def write_map(path: Path, patches, genes, values, shard=None, **over):
    header = {
        "slide": "s.svs", "genes": genes, "patches": patches, "side": 100,
        "dtype": "float16", "model": "test", "modelId": "test-1",
    }
    if shard:
        header["shard"] = list(shard)
    header.update(over)
    blob = json.dumps(header).encode()
    path.write_bytes(MAGIC + struct.pack("<I", len(blob)) + blob + values.tobytes())


def read_map(path: Path):
    import numpy as np
    raw = path.read_bytes()
    n = struct.unpack("<I", raw[8:12])[0]
    header = json.loads(raw[12:12 + n])
    values = np.frombuffer(raw[12 + n:], dtype=np.float16)
    return header, values


def merge(tmp_path, shards, out):
    return subprocess.run(
        [sys.executable, str(HERE / "merge_shards.py"), *[str(s) for s in shards],
         "--out", str(out)],
        capture_output=True, text=True,
    )


@pytest.fixture
def three_shards(tmp_path):
    np = pytest.importorskip("numpy")
    genes = ["A", "B"]
    made = []
    start = 0
    for i in range(1, 4):
        patches = [{"x": (start + k) * 100, "y": 0} for k in range(3)]
        vals = np.arange(start * 2, (start + 3) * 2, dtype=np.float16)
        path = tmp_path / f"s{i}.bin"
        write_map(path, patches, genes, vals, shard=(i, 3))
        made.append(path)
        start += 3
    return made, genes


def test_merging_restores_order_and_values(tmp_path, three_shards):
    np = pytest.importorskip("numpy")
    shards, genes = three_shards
    out = tmp_path / "merged.bin"
    r = merge(tmp_path, shards, out)
    assert r.returncode == 0, r.stderr

    header, values = read_map(out)
    assert header["genes"] == genes
    assert "shard" not in header, "the merged file is not itself a shard"
    assert len(header["patches"]) == 9
    # Patch order has to survive, or values line up with the wrong positions.
    assert [p["x"] for p in header["patches"]] == [i * 100 for i in range(9)]
    np.testing.assert_array_equal(values, np.arange(18, dtype=np.float16))


def test_order_on_the_command_line_does_not_matter(tmp_path, three_shards):
    shards, _ = three_shards
    out = tmp_path / "merged.bin"
    assert merge(tmp_path, [shards[2], shards[0], shards[1]], out).returncode == 0
    header, _ = read_map(out)
    assert [p["x"] for p in header["patches"]] == [i * 100 for i in range(9)]


def test_a_missing_shard_is_refused(tmp_path, three_shards):
    shards, _ = three_shards
    r = merge(tmp_path, shards[:2], tmp_path / "out.bin")
    assert r.returncode != 0
    assert "Expected shards 1..3" in r.stderr


def test_a_duplicate_shard_is_refused(tmp_path, three_shards):
    shards, _ = three_shards
    r = merge(tmp_path, [shards[0], shards[0], shards[1]], tmp_path / "out.bin")
    assert r.returncode != 0


def test_shards_from_different_runs_are_refused(tmp_path, three_shards):
    np = pytest.importorskip("numpy")
    shards, _ = three_shards
    # Same slide, different gene set: gluing these would pair patches with
    # values that mean something else entirely.
    write_map(shards[1], [{"x": 0, "y": 0}], ["A", "B", "C"],
              np.zeros(3, dtype=np.float16), shard=(2, 3))
    r = merge(tmp_path, shards, tmp_path / "out.bin")
    assert r.returncode != 0
    assert "genes" in r.stderr


def test_an_unsharded_file_is_refused(tmp_path, three_shards):
    np = pytest.importorskip("numpy")
    shards, _ = three_shards
    write_map(shards[0], [{"x": 0, "y": 0}], ["A", "B"], np.zeros(2, dtype=np.float16))
    r = merge(tmp_path, shards, tmp_path / "out.bin")
    assert r.returncode != 0
    assert "--shard" in r.stderr
