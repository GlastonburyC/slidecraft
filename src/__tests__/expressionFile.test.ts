import { describe, expect, it } from "vitest";
import { InvalidExpressionFile, matchesSlide, parseExpressionFile } from "../io/expressionFile";
import { geneValues } from "../ml/spatialResult";
import { resolveSlides } from "../slide/dropResolver";
import type { ResolvedFile } from "../slide/types";

const MAGIC = "SCEXPR1\0";

/** Build a file the way scripts/predict_expression.py does. */
function build(
  header: Record<string, unknown>,
  values: number[],
  dtype: "float16" | "float32" = "float32",
  magic = MAGIC,
): ArrayBuffer {
  const head = new TextEncoder().encode(JSON.stringify({ ...header, dtype }));
  const payload =
    dtype === "float32"
      ? new Uint8Array(Float32Array.from(values).buffer)
      : new Uint8Array(Uint16Array.from(values.map(toHalf)).buffer);

  const out = new Uint8Array(8 + 4 + head.length + payload.length);
  out.set(new TextEncoder().encode(magic), 0);
  new DataView(out.buffer).setUint32(8, head.length, true);
  out.set(head, 12);
  out.set(payload, 12 + head.length);
  return out.buffer;
}

/** Minimal float32 -> IEEE half, enough for the round-trip test. */
function toHalf(v: number): number {
  const f = new Float32Array([v]);
  const i = new Uint32Array(f.buffer)[0];
  const sign = (i >> 16) & 0x8000;
  const exp = ((i >> 23) & 0xff) - 127 + 15;
  const frac = (i >> 13) & 0x3ff;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | frac;
}

const header = {
  slide: "case_01.svs",
  genes: ["EPCAM", "CD3D"],
  patches: [{ x: 0, y: 0 }, { x: 224, y: 0 }, { x: 0, y: 224 }],
  side: 224,
  model: "DeepSpot-M (2 genes)",
  modelId: "deepspot-m-2",
  mpp: 0.25,
};
// Row-major by patch: patch 0 = [1,2], patch 1 = [3,4], patch 2 = [5,6].
const values = [1, 2, 3, 4, 5, 6];

describe("reading a precomputed expression map", () => {
  it("round-trips patches, genes and values", () => {
    const r = parseExpressionFile(build(header, values));
    expect(r.genes).toEqual(["EPCAM", "CD3D"]);
    expect(r.patches.length).toBe(3);
    expect(r.side).toBe(224);
    expect(r.slide).toBe("case_01.svs");
    expect(Array.from(geneValues(r, "EPCAM")!)).toEqual([1, 3, 5]);
    expect(Array.from(geneValues(r, "CD3D")!)).toEqual([2, 4, 6]);
    expect(r.patches[1].x).toBe(224);
  });

  it("decodes fp16, which is how a whole-transcriptome map is stored", () => {
    // 19k genes over 10k patches is 40 MB as fp16 and 80 as fp32.
    const r = parseExpressionFile(build(header, values, "float16"));
    expect(Array.from(geneValues(r, "EPCAM")!)).toEqual([1, 3, 5]);
  });

  it("rejects a file that is not one of ours", () => {
    expect(() => parseExpressionFile(build(header, values, "float32", "NOTMINE\0")))
      .toThrow(InvalidExpressionFile);
  });

  /**
   * A truncated payload would otherwise be read as a shorter grid, silently
   * pairing every patch with the wrong gene from partway through the file.
   */
  it("rejects a payload that does not match the header", () => {
    expect(() => parseExpressionFile(build(header, [1, 2, 3, 4])))
      .toThrow(/but the file holds/);
  });

  it("rejects a header with no genes", () => {
    expect(() => parseExpressionFile(build({ ...header, genes: [] }, values)))
      .toThrow(InvalidExpressionFile);
  });
});

describe("pairing a map to its slide", () => {
  it("matches on the stem, whatever the extension", () => {
    const r = parseExpressionFile(build(header, values));
    expect(matchesSlide(r, "case_01.svs")).toBe(true);
    expect(matchesSlide(r, "case_01.ndpi")).toBe(true);
    // The one that matters: another slide's map must not be drawn over this one.
    expect(matchesSlide(r, "case_02.svs")).toBe(false);
  });

  it("is picked up by the drop resolver alongside the slide", () => {
    const file = (path: string): ResolvedFile =>
      ({ path, file: new File(["x"], path.slice(path.lastIndexOf("/") + 1)) });
    const [slide] = resolveSlides([
      file("run/case_01.svs"),
      file("run/case_01.expression.bin"),
      file("run/case_01.geojson"),
    ]);
    expect(slide.expression?.name).toBe("case_01.expression.bin");
    expect(slide.annotations?.name).toBe("case_01.geojson");
  });

  it("leaves a slide with no map alone", () => {
    const file = (path: string): ResolvedFile =>
      ({ path, file: new File(["x"], path.slice(path.lastIndexOf("/") + 1)) });
    const [slide] = resolveSlides([file("run/case_02.svs"), file("run/case_01.expression.bin")]);
    expect(slide.expression).toBe(null);
  });
});

describe("a whole-transcriptome map", () => {
  /**
   * The reason this file does not widen or copy.
   *
   * 32,054 patches by 19,338 genes is 620 million values. Widening them to
   * fp32 on load costs 2.5 GB and the slice that used to feed it another 1.2 GB
   * on top of the 1.2 GB buffer — about 5 GB to put one gene on screen. These
   * pin the two properties that avoid it: the halves are kept as halves, and
   * the array is a view onto the original buffer rather than a copy of it.
   */
  function build(nPatches: number, nGenes: number, pad = 0) {
    const genes = Array.from({ length: nGenes }, (_, i) => `G${i}`);
    const header = {
      slide: "s.svs",
      genes,
      patches: Array.from({ length: nPatches }, (_, i) => ({ x: i * 10, y: 0 })),
      side: 10,
      dtype: "float16",
      model: "test",
      // Padding lets a test force an odd header length.
      ...(pad ? { pad: "x".repeat(pad) } : {}),
    };
    const json = new TextEncoder().encode(JSON.stringify(header));
    const count = nPatches * nGenes;
    const buf = new ArrayBuffer(12 + json.length + count * 2);
    const b = new Uint8Array(buf);
    b.set(new TextEncoder().encode("SCEXPR1\0"), 0);
    new DataView(buf).setUint32(8, json.length, true);
    b.set(json, 12);
    // Through a DataView, because a Uint16Array view cannot start on an odd
    // byte — the very constraint the parser has to cope with, so the fixture
    // must not depend on being aligned either.
    // 0x3C00 is 1.0 in IEEE half; 0x4000 is 2.0.
    const dv = new DataView(buf);
    const base = 12 + json.length;
    for (let i = 0; i < count; i++) dv.setUint16(base + i * 2, i % 2 ? 0x4000 : 0x3c00, true);
    return { buf, json };
  }

  it("keeps half precision as halves, viewing the buffer it was given", () => {
    const { buf, json } = build(4, 3);
    const result = parseExpressionFile(buf);
    expect(result.half).toBe(true);
    expect(result.values).toBeInstanceOf(Uint16Array);
    // The decisive check: a view shares the buffer, a copy does not.
    if ((12 + json.length) % 2 === 0) {
      expect(result.values.buffer).toBe(buf);
      expect(result.values.byteLength).toBe(4 * 3 * 2);
    }
  });

  it("decodes those halves correctly when a gene is read", () => {
    const { buf } = build(4, 3);
    const result = parseExpressionFile(buf);
    // Gene 0 of patch 0 is index 0 -> 1.0; gene 1 is index 1 -> 2.0.
    expect(geneValues(result, "G0")![0]).toBe(1);
    expect(geneValues(result, "G1")![0]).toBe(2);
  });

  it("copies rather than misaligning when the header length is odd", () => {
    // A Uint16Array view must start on an even byte. Find a padding that makes
    // the payload start odd, and check it still parses.
    let made: ReturnType<typeof build> | null = null;
    for (let pad = 0; pad < 8; pad++) {
      const candidate = build(4, 3, pad);
      if ((12 + candidate.json.length) % 2 === 1) { made = candidate; break; }
    }
    expect(made).not.toBeNull();
    const result = parseExpressionFile(made!.buf);
    expect(result.half).toBe(true);
    expect(geneValues(result, "G0")![0]).toBe(1);
    expect(geneValues(result, "G1")![0]).toBe(2);
  });
});
