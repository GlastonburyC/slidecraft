import { describe, expect, it } from "vitest";
import { writeFileSync } from "node:fs";
import { crc32, makeZip, zipSize } from "../io/zip";
import { annDataZipBytes, matrixBytes, toAnnDataZip } from "../io/anndata";
import type { SpatialResult } from "../ml/spatialResult";

const utf8 = new TextEncoder();
const dec = new TextDecoder();

/** Read the archive back the way a reader does: from the central directory. */
function readZip(zip: Uint8Array): Map<string, Uint8Array> {
  const view = new DataView(zip.buffer, zip.byteOffset, zip.byteLength);
  // End of central directory is the last 22 bytes when there is no comment.
  const eocd = zip.length - 22;
  expect(view.getUint32(eocd, true)).toBe(0x06054b50);
  const count = view.getUint16(eocd + 10, true);
  const dirSize = view.getUint32(eocd + 12, true);
  const dirStart = view.getUint32(eocd + 16, true);
  // The directory has to end exactly where the record begins, which is the
  // invariant a miscounted size breaks.
  expect(dirStart + dirSize).toBe(eocd);

  const out = new Map<string, Uint8Array>();
  let at = dirStart;
  for (let i = 0; i < count; i++) {
    expect(view.getUint32(at, true)).toBe(0x02014b50);
    const size = view.getUint32(at + 24, true);
    const nameLen = view.getUint16(at + 28, true);
    const offset = view.getUint32(at + 42, true);
    const name = dec.decode(zip.subarray(at + 46, at + 46 + nameLen));
    const localName = view.getUint16(offset + 26, true);
    const localExtra = view.getUint16(offset + 28, true);
    const start = offset + 30 + localName + localExtra;
    out.set(name, zip.subarray(start, start + size));
    at += 46 + nameLen;
  }
  return out;
}

describe("the zip writer", () => {
  it("computes CRC32 the way the format defines it", () => {
    // The check value every CRC-32 implementation is tested against.
    expect(crc32(utf8.encode("123456789"))).toBe(0xcbf43926);
    expect(crc32(new Uint8Array(0))).toBe(0);
  });

  it("produces exactly the size it predicted", () => {
    const entries = [
      { name: "a.txt", data: utf8.encode("hello") },
      { name: "nested/b.bin", data: new Uint8Array([1, 2, 3, 4, 5, 6]) },
    ];
    expect(makeZip(entries).length).toBe(zipSize(entries));
  });

  it("round-trips names and bytes", () => {
    const entries = [
      { name: "x/.zarray", data: utf8.encode('{"zarr_format":2}') },
      { name: "x/0.0", data: new Uint8Array([9, 8, 7]) },
    ];
    const back = readZip(makeZip(entries));
    expect([...back.keys()].sort()).toEqual(["x/.zarray", "x/0.0"]);
    expect(dec.decode(back.get("x/.zarray"))).toBe('{"zarr_format":2}');
    expect([...back.get("x/0.0")!]).toEqual([9, 8, 7]);
  });

  it("is byte-identical across runs, so an export can be diffed", () => {
    const entries = [{ name: "a", data: utf8.encode("same") }];
    expect([...makeZip(entries)]).toEqual([...makeZip(entries)]);
  });
});

const GENES = ["EPCAM", "MUC2", "COL1A1", "PTPRC"];

function demo(n = 7): SpatialResult {
  const values = new Float32Array(n * GENES.length);
  for (let i = 0; i < n; i++)
    for (let g = 0; g < GENES.length; g++) values[i * GENES.length + g] = i + g / 10;
  return {
    slide: "demo.svs", genes: GENES, values,
    patches: Array.from({ length: n }, (_, i) => ({
      x: i * 100, y: 50, index: i, col: i, row: 0, size: 100,
    })),
    side: 100, modelId: "deepspot-m-scgpt-4", modelName: "DeepSpot-M scgpt (4 genes)",
    roiId: null, ms: 0, createdAt: "2026-09-12T00:00:00Z",
  } as unknown as SpatialResult;
}

describe("exporting a map as AnnData", () => {
  it("writes the groups a reader dispatches on", () => {
    const back = readZip(toAnnDataZip(demo()));
    const root = JSON.parse(dec.decode(back.get(".zattrs")!));
    expect(root["encoding-type"]).toBe("anndata");

    const obs = JSON.parse(dec.decode(back.get("obs/.zattrs")!));
    expect(obs["encoding-type"]).toBe("dataframe");
    expect(obs._index).toBe("patch");
    expect(obs["column-order"]).toEqual(["x", "y"]);

    const varAttrs = JSON.parse(dec.decode(back.get("var/.zattrs")!));
    expect(varAttrs._index).toBe("gene");
  });

  it("lays X out row-major, float32, with the declared chunking", () => {
    const back = readZip(toAnnDataZip(demo()));
    const meta = JSON.parse(dec.decode(back.get("X/.zarray")!));
    expect(meta.shape).toEqual([7, 4]);
    expect(meta.dtype).toBe("<f4");
    expect(meta.compressor).toBeNull();

    const chunk = back.get("X/0.0")!;
    const x = new Float32Array(chunk.buffer.slice(chunk.byteOffset, chunk.byteOffset + chunk.byteLength));
    // Patch 2, gene 3 is 2 + 0.3 and lives at row-major index 2*4+3.
    expect(x[2 * 4 + 3]).toBeCloseTo(2.3, 5);
  });

  it("encodes strings the way numcodecs vlen-utf8 reads them", () => {
    const back = readZip(toAnnDataZip(demo()));
    const blob = back.get("var/gene/0")!;
    const view = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
    expect(view.getUint32(0, true)).toBe(4);
    // First element: its byte length, then its bytes.
    const len = view.getUint32(4, true);
    expect(dec.decode(blob.subarray(8, 8 + len))).toBe("EPCAM");
  });

  it("puts patch centres in obsm/spatial, not the corners", () => {
    const back = readZip(toAnnDataZip(demo()));
    const blob = back.get("obsm/spatial/0.0")!;
    const s = new Float32Array(blob.buffer.slice(blob.byteOffset, blob.byteOffset + blob.byteLength));
    // Patch 0 sits at (0, 50) with side 100, so its centre is (50, 100).
    expect([s[0], s[1]]).toEqual([50, 100]);
  });

  it("carries the tissue mask as obs.in_tissue when there is one", () => {
    const mask = new Uint8Array(7).fill(1);
    mask[0] = 0;
    const withMask = readZip(toAnnDataZip(demo(), { tissueMask: mask }));
    expect(JSON.parse(dec.decode(withMask.get("obs/.zattrs")!))["column-order"])
      .toEqual(["x", "y", "in_tissue"]);

    // A mask for a different map is ignored rather than misaligned.
    const wrong = readZip(toAnnDataZip(demo(), { tissueMask: new Uint8Array(3) }));
    expect(JSON.parse(dec.decode(wrong.get("obs/.zattrs")!))["column-order"])
      .toEqual(["x", "y"]);
  });

  it("records where the numbers came from, and that they are predicted", () => {
    const back = readZip(toAnnDataZip(demo()));
    const read = (k: string) => {
      const b = back.get(`uns/${k}/0`)!;
      const v = new DataView(b.buffer, b.byteOffset, b.byteLength);
      return dec.decode(b.subarray(8, 8 + v.getUint32(4, true)));
    };
    expect(read("model_id")).toBe("deepspot-m-scgpt-4");
    expect(read("slide")).toBe("demo.svs");
    expect(read("provenance")).toMatch(/Not a measurement/);
  });

  it("estimates its own size closely enough to gate on", () => {
    const result = demo(50);
    const actual = toAnnDataZip(result).length;
    const predicted = annDataZipBytes(result);
    expect(matrixBytes(result)).toBe(50 * 4 * 4);
    expect(predicted).toBeGreaterThanOrEqual(actual);
    expect(predicted).toBeLessThan(actual + 200_000);
  });

  /*
   * The archive is also checked against real anndata, which is the only thing
   * that proves the layout rather than restating it. Set the variable and the
   * fixture is written for `scripts/test_anndata_roundtrip.py` to read.
   */
  const out = process.env.SLIDECRAFT_ANNDATA_OUT;
  it.runIf(out)("writes a fixture for the Python round-trip", () => {
    const mask = new Uint8Array(7).fill(1);
    mask[0] = 0;
    writeFileSync(out!, toAnnDataZip(demo(), { tissueMask: mask }));
  });
});
