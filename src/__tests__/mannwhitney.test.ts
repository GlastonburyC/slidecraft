import { describe, expect, it } from "vitest";
import { differentialExpression, differentialSignatures } from "../ml/enrichment";
import { scoreSignature } from "../ml/signatures";
import type { SpatialResult } from "../ml/spatialResult";

/**
 * The fast path has to agree with the definition, not merely look plausible.
 *
 * U is computed by sorting whichever group is smaller and searching the other
 * against it, which is a different route to the same number — so the number is
 * checked against one worked out the slow, obvious way.
 */
function bruteAuc(values: number[], inside: Set<number>): number {
  let u = 0;
  for (let i = 0; i < values.length; i++) {
    if (!inside.has(i)) continue;
    for (let j = 0; j < values.length; j++) {
      if (inside.has(j)) continue;
      if (values[i] > values[j]) u += 1;
      else if (values[i] === values[j]) u += 0.5;
    }
  }
  const nIn = inside.size;
  return u / (nIn * (values.length - nIn));
}

function mapOf(perPatch: number[][]): SpatialResult {
  const g = perPatch[0].length;
  const values = new Float32Array(perPatch.flat());
  return {
    slide: "s.svs",
    genes: Array.from({ length: g }, (_, i) => `G${i}`),
    values,
    patches: perPatch.map((_, i) => ({ x: i * 10, y: 0, index: i, col: i, row: 0, size: 10 })),
    side: 10, modelId: "t", modelName: "t", roiId: null, ms: 0, createdAt: "",
  } as unknown as SpatialResult;
}

describe("the fast Mann-Whitney", () => {
  it("matches a brute-force AUC on random data, both ways round", () => {
    const rng = (() => { let s = 7; return () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648; })();
    for (const insideFraction of [0.15, 0.5, 0.85]) {
      const n = 60;
      // Deliberately coarse, so there are plenty of exact ties.
      const col = Array.from({ length: n }, () => Math.round(rng() * 6) / 2);
      const inside = new Set<number>();
      for (let i = 0; i < n; i++) if (rng() < insideFraction) inside.add(i);
      if (inside.size < 5 || n - inside.size < 5) continue;

      const result = mapOf(col.map((v) => [v]));
      const got = differentialExpression(result, inside).genes[0].auc;
      expect(got).toBeCloseTo(bruteAuc(col, inside), 10);
    }
  });

  it("is exactly 1 when the region is strictly above the rest", () => {
    const col = [9, 9, 9, 9, 9, 1, 2, 3, 4, 5];
    const inside = new Set([0, 1, 2, 3, 4]);
    expect(differentialExpression(mapOf(col.map((v) => [v])), inside).genes[0].auc).toBe(1);
  });

  it("is exactly 0 when it is strictly below", () => {
    const col = [1, 1, 1, 1, 1, 6, 7, 8, 9, 10];
    const inside = new Set([0, 1, 2, 3, 4]);
    expect(differentialExpression(mapOf(col.map((v) => [v])), inside).genes[0].auc).toBe(0);
  });

  it("is 0.5 when every value is tied", () => {
    const col = new Array(12).fill(3);
    const inside = new Set([0, 1, 2, 3, 4]);
    const out = differentialExpression(mapOf(col.map((v) => [v])), inside).genes[0];
    expect(out.auc).toBe(0.5);
    // No ordering to compare means no evidence, whichever way it is computed.
    expect(out.p).toBe(1);
  });

  it("gives the same answer whichever group is the smaller one", () => {
    const col = [5, 1, 4, 2, 3, 9, 8, 7, 6, 0, 5, 5];
    const few = new Set([0, 1, 2]);
    const many = new Set(col.map((_, i) => i).filter((i) => !few.has(i)));
    const a = differentialExpression(mapOf(col.map((v) => [v])), few, 3).genes[0].auc;
    const b = differentialExpression(mapOf(col.map((v) => [v])), many, 3).genes[0].auc;
    // Swapping the groups mirrors the statistic about 0.5.
    expect(a).toBeCloseTo(1 - b, 10);
  });
});

/**
 * The half-precision path is a different algorithm reaching the same number.
 *
 * It bins values by bit pattern and sweeps the bins instead of sorting, which
 * is only correct if the bit order matches the numeric order — sign bits make
 * that false without an adjustment, and negatives do occur in predicted
 * expression. So it is checked against the sorting path on the same values,
 * including negative ones and heavy ties.
 */
const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
function toHalf(value: number): number {
  f32[0] = value;
  const bits = u32[0];
  const sign = (bits >>> 16) & 0x8000;
  const exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  const fraction = (bits >>> 13) & 0x3ff;
  if (exponent <= 0) return sign;
  if (exponent >= 31) return sign | 0x7c00;
  return sign | (exponent << 10) | fraction;
}

function halfMapOf(perPatch: number[][]): SpatialResult {
  const flat = perPatch.flat();
  const raw = new Uint16Array(flat.length);
  for (let i = 0; i < flat.length; i++) raw[i] = toHalf(flat[i]);
  return {
    slide: "s.svs",
    genes: Array.from({ length: perPatch[0].length }, (_, i) => `G${i}`),
    values: raw,
    half: true,
    patches: perPatch.map((_, i) => ({ x: i * 10, y: 0, index: i, col: i, row: 0, size: 10 })),
    side: 10, modelId: "t", modelName: "t", roiId: null, ms: 0, createdAt: "",
  } as unknown as SpatialResult;
}

describe("the half-precision counting path", () => {
  it("agrees with the sorting path, negatives and ties included", () => {
    const rng = (() => { let s = 99; return () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648; })();
    const n = 90;
    const genes = 6;
    const perPatch = Array.from({ length: n }, () =>
      // Coarse and straddling zero: ties in every bin, and both signs.
      Array.from({ length: genes }, () => Math.round((rng() * 6 - 1.5) * 4) / 4));

    const inside = new Set<number>();
    for (let i = 0; i < n; i++) if (rng() < 0.3) inside.add(i);

    // The fp32 map must hold exactly what fp16 rounds to, or the two are being
    // asked different questions.
    const rounded = perPatch.map((row) => row.map((v) => {
      const h = toHalf(v);
      const sign = h & 0x8000 ? -1 : 1;
      const e = (h & 0x7c00) >> 10;
      const f = h & 0x03ff;
      return e === 0 ? sign * Math.pow(2, -14) * (f / 1024)
                     : sign * Math.pow(2, e - 15) * (1 + f / 1024);
    }));

    const slow = differentialExpression(mapOf(rounded), inside).genes;
    const fast = differentialExpression(halfMapOf(perPatch), inside).genes;

    expect(fast).toHaveLength(slow.length);
    const bySlow = Object.fromEntries(slow.map((g) => [g.gene, g]));
    for (const g of fast) {
      expect(g.auc).toBeCloseTo(bySlow[g.gene].auc, 10);
      expect(g.meanIn).toBeCloseTo(bySlow[g.gene].meanIn, 5);
      expect(g.meanOut).toBeCloseTo(bySlow[g.gene].meanOut, 5);
      expect(g.p).toBeCloseTo(bySlow[g.gene].p, 8);
    }
  });

  it("orders negatives below positives, which raw bit order does not", () => {
    // Without the key adjustment the sign bit sorts negatives ABOVE everything.
    const col = [-2, -1, -0.5, -0.25, -0.125, 0.5, 1, 2, 3, 4, 5, 6];
    const inside = new Set([0, 1, 2, 3, 4]); // the five most negative
    const out = differentialExpression(halfMapOf(col.map((v) => [v])), inside).genes[0];
    expect(out.auc).toBe(0); // strictly below everything outside
  });

  it("leaves the bins clean, so one gene cannot contaminate the next", () => {
    // Two genes with disjoint value ranges: if bins were not cleared, the
    // second would inherit the first's counts and its AUC would drift.
    const perPatch = Array.from({ length: 20 }, (_, i) => [i < 10 ? 1 : 2, i < 10 ? 100 : 200]);
    const inside = new Set(Array.from({ length: 10 }, (_, i) => i));
    const got = differentialExpression(halfMapOf(perPatch), inside).genes;
    for (const g of got) expect(g.auc).toBe(0); // inside is the lower half of both
  });
});

describe("negative zero", () => {
  it("ties with positive zero rather than sorting below it", () => {
    // -0 and 0 are the same number with different bits. Ten patches, half of
    // each, split across the region boundary: if they did not tie, the AUC
    // would move off 0.5.
    const col = [-0, 0, -0, 0, -0, 0, -0, 0, -0, 0];
    const inside = new Set([0, 1, 2, 3, 4]);
    const out = differentialExpression(halfMapOf(col.map((v) => [v])), inside).genes[0];
    expect(out.auc).toBe(0.5);
  });
});

describe("filtering genes the model barely expresses", () => {
  /**
   * A rank test is scale-free, which is the trap. A gene sitting at 0.001 the
   * whole way across can separate a region perfectly on the ordering of noise
   * and outrank a real marker, so the floor is on the VALUE, not the statistic.
   */
  function twoGenes(): SpatialResult {
    const n = 20;
    const rows: number[][] = [];
    for (let i = 0; i < n; i++) {
      // REAL separates and is expressed; GHOST separates just as cleanly and is
      // three orders of magnitude smaller.
      rows.push([i < 10 ? 2.0 : 0.5, i < 10 ? 0.002 : 0.0005]);
    }
    return mapOf(rows);
  }
  const genesOf = (r: { genes: { gene: string }[] }) => r.genes.map((g) => g.gene);
  const inside = new Set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

  it("reports both when there is no floor", () => {
    const out = differentialExpression(twoGenes(), inside, 5, 0);
    expect(genesOf(out).sort()).toEqual(["G0", "G1"]);
    // Both separate perfectly, which is exactly why the floor is needed.
    expect(out.genes.every((g) => g.auc === 1)).toBe(true);
  });

  it("drops the unexpressed one once a floor is set", () => {
    const out = differentialExpression(twoGenes(), inside, 5, 0.05);
    expect(genesOf(out)).toEqual(["G0"]);
  });

  it("records each gene's mean, which is what the floor reads", () => {
    const out = differentialExpression(twoGenes(), inside, 5, 0);
    const by = Object.fromEntries(out.genes.map((g) => [g.gene, g.mean]));
    expect(by.G0).toBeCloseTo(1.25, 5);
    expect(by.G1).toBeCloseTo(0.00125, 6);
  });

  it("corrects q over the genes it reports, not the ones it dropped", () => {
    /*
     * Forty unexpressed decoys that separate nothing, alongside one real gene.
     * Benjamini-Hochberg scales the top p-value by the number of tests, so
     * leaving the decoys in multiplies the real gene's q by forty-one for no
     * reason — nothing about them was ever going to be reported.
     */
    const n = 20;
    const rng = (() => { let z = 5; return () => (z = (z * 1103515245 + 12345) % 2147483648) / 2147483648; })();
    const rows = Array.from({ length: n }, (_, i) =>
      [i < 10 ? 2.0 : 0.5, ...Array.from({ length: 40 }, () => 0.0005 + rng() * 0.001)]);
    const result = mapOf(rows);

    const withFloor = differentialExpression(result, inside, 5, 0.05);
    const without = differentialExpression(result, inside, 5, 0);
    expect(withFloor.genes).toHaveLength(1);
    expect(without.genes).toHaveLength(41);

    const real = (r: typeof withFloor) => r.genes.find((g) => g.gene === "G0")!.q;
    expect(real(withFloor)).toBeLessThan(real(without));
    // And by the factor the correction actually applies.
    expect(real(without) / real(withFloor)).toBeCloseTo(41, 0);
  });

  it("never filters cell types, which have no expression of their own", () => {
    const sig = [{ name: "Module", genes: [{ gene: "G0", weight: 1 }] }];
    const out = differentialSignatures(twoGenes(), inside, sig, scoreSignature);
    expect(out.genes.map((g) => g.gene)).toEqual(["Module"]);
  });
});
