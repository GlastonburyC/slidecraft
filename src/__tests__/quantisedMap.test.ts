import { describe, expect, it } from "vitest";
import { InvalidExpressionFile, parseExpressionFile } from "../io/expressionFile";
import { geneValues, valueAt, type SpatialResult } from "../ml/spatialResult";
import { differentialExpression } from "../ml/enrichment";
import { geneGradient } from "../ml/gradient";

const MAGIC = "SCEXPR1\0";

function pack(header: Record<string, unknown>, payload: Uint8Array): ArrayBuffer {
  const head = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(12 + head.length + payload.length);
  out.set(new TextEncoder().encode(MAGIC), 0);
  new DataView(out.buffer).setUint32(8, head.length, true);
  out.set(head, 12);
  out.set(payload, 12 + head.length);
  return out.buffer;
}

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

/**
 * Quantise a patches-by-genes matrix the way scripts/subset_expression.py does,
 * so what the reader is tested against is the writer's actual arithmetic.
 */
function quantise(matrix: number[][], genes: string[]) {
  const n = matrix.length;
  const g = genes.length;
  const scale: number[] = [];
  const zero: number[] = [];
  for (let j = 0; j < g; j++) {
    let lo = 0;
    let hi = -Infinity;
    for (let i = 0; i < n; i++) {
      lo = Math.min(lo, matrix[i][j]);
      hi = Math.max(hi, matrix[i][j]);
    }
    hi = Math.max(hi, lo + 1e-6);
    zero.push(lo);
    scale.push((hi - lo) / 255);
  }
  const codes = new Uint8Array(n * g);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < g; j++) {
      codes[i * g + j] = Math.min(255, Math.max(0, Math.round((matrix[i][j] - zero[j]) / scale[j])));
    }
  }
  return { codes, scale, zero };
}

function header(genes: string[], n: number) {
  return {
    slide: "case_01.svs",
    genes,
    patches: Array.from({ length: n }, (_, i) => ({ x: (i % 40) * 224, y: Math.floor(i / 40) * 224 })),
    side: 224,
    model: "DeepSpot-M",
    modelId: "deepspot-m",
    mpp: 0.5,
  };
}

/** Values with the shape real predictions have: mostly small, a few large, some below zero. */
function synthetic(n: number, g: number) {
  let seed = 7;
  const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: g }, (_, j) => {
      const base = j === 0 ? 3.5 : j === 1 ? 0.4 : 0.02;
      const gradient = j === 2 ? (i / n) * 1.5 : 0;
      const v = base * rand() + gradient - 0.05;
      return j === 3 ? 0 : v; // one gene that is flatly absent
    }),
  );
}

describe("quantised expression maps", () => {
  const genes = ["COL1A1", "CD3D", "AQP8", "OR4F5"];
  const matrix = synthetic(300, genes.length);
  const { codes, scale, zero } = quantise(matrix, genes);
  const buffer = pack({ ...header(genes, 300), dtype: "uint8", scale, zero }, codes);

  it("decodes to within one quantisation step of the original", () => {
    const r = parseExpressionFile(buffer);
    expect(r.values).toBeInstanceOf(Uint8Array);
    expect(r.half).toBe(false);
    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < genes.length; j++) {
        expect(valueAt(r, i * genes.length + j)).toBeCloseTo(matrix[i][j], 1);
        expect(Math.abs(valueAt(r, i * genes.length + j) - matrix[i][j])).toBeLessThanOrEqual(
          scale[j] / 2 + 1e-6,
        );
      }
    }
  });

  it("keeps an absent gene at exactly zero", () => {
    // A gene with no expression must not pick up a floor from the encoding, or
    // every sparse gene reads as faintly present everywhere.
    const r = parseExpressionFile(buffer);
    const absent = geneValues(r, "OR4F5")!;
    expect([...absent].every((v) => v === 0)).toBe(true);
  });

  it("reads negative predictions back as negative", () => {
    const r = parseExpressionFile(buffer);
    const low = geneValues(r, "AQP8")!;
    const original = matrix.map((row) => row[2]);
    expect(Math.min(...original)).toBeLessThan(0);
    expect(Math.min(...low)).toBeLessThan(0);
  });

  it("refuses a map whose scales do not cover its genes", () => {
    const bad = pack({ ...header(genes, 300), dtype: "uint8", scale: [1, 1], zero: [0, 0] }, codes);
    expect(() => parseExpressionFile(bad)).toThrow(InvalidExpressionFile);
  });

  it("refuses a map with no scales at all", () => {
    const bad = pack({ ...header(genes, 300), dtype: "uint8" }, codes);
    expect(() => parseExpressionFile(bad)).toThrow(InvalidExpressionFile);
  });

  /*
   * The counting path for quantised maps bins by the stored byte and takes the
   * group means out of the same sweep, which is a different computation from
   * both the fp16 path and the sorting path. It has to agree with a plain sort
   * over the decoded values — that comparison is what caught the negative-zero
   * bug in the fp16 path, and it is the only thing standing behind this one.
   */
  it("gives the same U, AUC and means as sorting the decoded values", () => {
    const r = parseExpressionFile(buffer);
    const decoded: SpatialResult = {
      ...r,
      values: Float32Array.from({ length: matrix.length * genes.length }, (_, k) => valueAt(r, k)),
      half: false,
      scale: undefined,
      zero: undefined,
    };
    const inside = new Set(Array.from({ length: 90 }, (_, i) => i * 3));

    const quick = differentialExpression(r, inside);
    const slow = differentialExpression(decoded, inside);
    expect(quick.genes.length).toBe(slow.genes.length);

    const by = (res: typeof quick) => new Map(res.genes.map((s) => [s.gene, s]));
    const a = by(quick);
    const b = by(slow);
    for (const [gene, s] of a) {
      const t = b.get(gene)!;
      expect(s.auc).toBeCloseTo(t.auc, 10);
      expect(s.p).toBeCloseTo(t.p, 10);
      expect(s.meanIn).toBeCloseTo(t.meanIn, 5);
      expect(s.meanOut).toBeCloseTo(t.meanOut, 5);
      expect(s.mean).toBeCloseTo(t.mean, 5);
    }
  });

  it("ranks genes along an axis the same as the fp16 map it came from", () => {
    const asHalf = pack(
      { ...header(genes, 300), dtype: "float16" },
      new Uint8Array(
        Uint16Array.from(matrix.flat().map(toHalf)).buffer,
      ),
    );
    const q = parseExpressionFile(buffer);
    const h = parseExpressionFile(asHalf);
    const axis = {
      id: "a",
      geometry: { type: "LineString", coordinates: [[0, 0], [39 * 224, 7 * 224]] },
    } as never;

    const qg = geneGradient(q, axis, 4000, 0.5);
    const hg = geneGradient(h, axis, 4000, 0.5);
    // The planted gradient is on AQP8, and quantisation must not lose it.
    expect(qg.items[0].name).toBe("AQP8");
    expect(hg.items[0].name).toBe("AQP8");
    expect(qg.items[0].rho).toBeCloseTo(hg.items[0].rho, 2);
  });
});
