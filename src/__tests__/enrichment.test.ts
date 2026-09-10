import { describe, expect, it } from "vitest";
import {
  benjaminiHochberg, differentialExpression, NotEnoughPatches, patchesInside,
} from "../ml/enrichment";
import type { SpatialResult } from "../ml/spatialResult";
import { makeAnnotation } from "../annotate/store";

/** A grid of patches with three genes, so a known signal can be planted. */
function fixture(
  n: number,
  fill: (patch: number, gene: number) => number,
  genes = ["CALC", "FLAT", "NOISE"],
): SpatialResult {
  const patches = Array.from({ length: n }, (_, i) => ({
    index: i, col: i % 10, row: Math.floor(i / 10),
    x: (i % 10) * 224, y: Math.floor(i / 10) * 224, size: 224,
  }));
  const values = new Float32Array(n * genes.length);
  for (let i = 0; i < n; i++) {
    for (let g = 0; g < genes.length; g++) values[i * genes.length + g] = fill(i, g);
  }
  return {
    genes, values, patches, side: 224,
    modelId: "m", modelName: "M", slide: "s.svs", roiId: null, ms: 1,
    createdAt: "2026-09-10T00:00:00.000Z",
  };
}

const noise = (i: number) => ((i * 2654435761) % 1000) / 1000;

describe("differential expression over a drawn region", () => {
  /**
   * The whole point: a gene raised only inside the region must come out top,
   * and a gene that is flat everywhere must not.
   */
  it("ranks a gene raised inside the region above one that is flat", () => {
    const inside = new Set(Array.from({ length: 20 }, (_, i) => i));
    const result = fixture(100, (i, g) => {
      if (g === 0) return inside.has(i) ? 5 + noise(i) : noise(i);  // CALC
      if (g === 1) return 3;                                        // FLAT
      return noise(i * 7);                                          // NOISE
    });

    const de = differentialExpression(result, inside);
    expect(de.genes[0].gene).toBe("CALC");
    expect(de.genes[0].auc).toBeGreaterThan(0.99);
    expect(de.genes[0].diff).toBeGreaterThan(4);
    expect(de.inside).toBe(20);
    expect(de.outside).toBe(80);

    const flat = de.genes.find((g) => g.gene === "FLAT")!;
    // Identical everywhere: no ordering to compare, so no evidence either way.
    expect(flat.auc).toBeCloseTo(0.5, 6);
    expect(flat.p).toBe(1);
  });

  it("puts a gene depleted inside at the bottom, not the top", () => {
    const inside = new Set(Array.from({ length: 20 }, (_, i) => i));
    const result = fixture(100, (i, g) =>
      g === 0 ? (inside.has(i) ? noise(i) : 5 + noise(i)) : noise(i * 3),
    );
    const de = differentialExpression(result, inside);
    const calc = de.genes.find((g) => g.gene === "CALC")!;
    expect(calc.auc).toBeLessThan(0.01);
    expect(calc.diff).toBeLessThan(0);
    // Ranking is by enrichment, so a depleted gene is last rather than first.
    expect(de.genes[de.genes.length - 1].gene).toBe("CALC");
  });

  /** AUC has a definition worth pinning: P(inside > outside). */
  it("computes AUC as the probability an inside patch exceeds an outside one", () => {
    // Inside {3,4}, outside {1,2}: every inside value beats every outside one.
    const result = fixture(4, (i) => [1, 2, 3, 4][i], ["G"]);
    const de = differentialExpression(result, new Set([2, 3]), 2);
    expect(de.genes[0].auc).toBeCloseTo(1, 6);

    // Interleaved {1,3} vs {2,4}: 1 beats none, 3 beats 2 → 1 of 4 wins... 
    const mixed = fixture(4, (i) => [1, 2, 3, 4][i], ["G"]);
    const de2 = differentialExpression(mixed, new Set([0, 2]), 2);
    expect(de2.genes[0].auc).toBeCloseTo(0.25, 6);
  });

  it("refuses a region too small to say anything about", () => {
    const result = fixture(100, (i) => noise(i), ["G"]);
    expect(() => differentialExpression(result, new Set([0, 1]))).toThrow(NotEnoughPatches);
    // And equally when nearly everything is inside, leaving no comparison group.
    const nearlyAll = new Set(Array.from({ length: 98 }, (_, i) => i));
    expect(() => differentialExpression(result, nearlyAll)).toThrow(NotEnoughPatches);
  });

  it("handles ties without producing an impossible p-value", () => {
    // Half the patches share one value, half another: heavy ties on purpose.
    const inside = new Set(Array.from({ length: 50 }, (_, i) => i));
    const result = fixture(100, (i) => (inside.has(i) ? 1 : 0), ["G"]);
    const de = differentialExpression(result, inside);
    expect(de.genes[0].p).toBeGreaterThanOrEqual(0);
    expect(de.genes[0].p).toBeLessThanOrEqual(1);
    expect(Number.isFinite(de.genes[0].p)).toBe(true);
    expect(de.genes[0].auc).toBeCloseTo(1, 6);
  });
});

describe("multiple testing", () => {
  it("leaves a single p-value alone and never exceeds 1", () => {
    expect(benjaminiHochberg([0.04])).toEqual([0.04]);
    expect(benjaminiHochberg([0.9, 0.95])[1]).toBeLessThanOrEqual(1);
  });

  it("adjusts upward and stays monotonic in the original order", () => {
    const p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.6];
    const q = benjaminiHochberg(p);
    q.forEach((v, i) => expect(v).toBeGreaterThanOrEqual(p[i]));
    const sortedByP = [...q.keys()].sort((a, b) => p[a] - p[b]).map((i) => q[i]);
    for (let i = 1; i < sortedByP.length; i++) {
      expect(sortedByP[i]).toBeGreaterThanOrEqual(sortedByP[i - 1] - 1e-12);
    }
    expect(q[0]).toBeCloseTo(0.006, 3);
  });
});

describe("finding the patches a region covers", () => {
  const square = (x: number, y: number, s: number) =>
    makeAnnotation(
      { type: "Polygon", coordinates: [[[x, y], [x + s, y], [x + s, y + s], [x, y + s], [x, y]]] },
      { classId: "calcification" },
    );

  it("takes a patch when its centre is inside, not merely its corner", () => {
    const result = fixture(100, () => 0, ["G"]);
    // Covers the first two columns' centres (centres at 112, 336).
    const inside = patchesInside(result, [square(0, 0, 448)]);
    expect(inside.size).toBe(2 * 2); // two columns by two rows
    expect(inside.has(0)).toBe(true);
    expect(inside.has(2)).toBe(false);
  });

  it("unions several regions, and ignores points and lines", () => {
    const result = fixture(100, () => 0, ["G"]);
    const point = makeAnnotation({ type: "Point", coordinates: [112, 112] }, { classId: "c" });
    const inside = patchesInside(result, [square(0, 0, 224), square(448, 0, 224), point]);
    expect(inside.has(0)).toBe(true);
    expect(inside.has(2)).toBe(true);
    expect(inside.size).toBe(2);
  });
});
