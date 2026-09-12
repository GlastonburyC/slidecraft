import { describe, expect, it } from "vitest";
import {
  latticeOf, moduleFromEnrichment, moduleFromGradient, percentileThreshold, similarRegions,
} from "../ml/findSimilar";
import type { GradientStat } from "../ml/gradient";
import { scoreSignature } from "../ml/signatures";
import type { GeneStat } from "../ml/enrichment";
import type { SpatialResult } from "../ml/spatialResult";

const stat = (gene: string, auc: number): GeneStat =>
  ({ gene, auc, mean: 1, meanIn: 0, meanOut: 0, diff: 0, p: 0, q: 0 });

describe("turning an enrichment into a module", () => {
  it("keeps both directions, signed", () => {
    const m = moduleFromEnrichment("Aggregate", [
      stat("CXCL13", 0.95), stat("MS4A1", 0.88), stat("COL1A1", 0.05),
    ]);
    const by = Object.fromEntries(m.genes.map((g) => [g.gene, g.weight]));
    expect(by.CXCL13).toBeCloseTo(0.9, 3);
    // Depleted is as descriptive as enriched, and has to score that way.
    expect(by.COL1A1).toBeCloseTo(-0.9, 3);
  });

  it("drops genes that barely separate, however many there are", () => {
    const noise = Array.from({ length: 500 }, (_, i) => stat(`N${i}`, 0.51));
    const m = moduleFromEnrichment("X", [stat("REAL", 0.9), ...noise]);
    expect(m.genes.map((g) => g.gene)).toEqual(["REAL"]);
  });

  it("caps the list, strongest first", () => {
    const stats = Array.from({ length: 80 }, (_, i) => stat(`G${i}`, 0.5 + (i + 1) / 200));
    const m = moduleFromEnrichment("X", stats, { top: 10 });
    expect(m.genes).toHaveLength(10);
    expect(m.genes[0].gene).toBe("G79");
    expect(Math.abs(m.genes[0].weight)).toBeGreaterThan(Math.abs(m.genes[9].weight));
  });

  it("scores as a signature, so it can colour the slide immediately", () => {
    const genes = ["CXCL13", "COL1A1"];
    const n = 6;
    const values = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      values[i * 2] = i < 3 ? 5 : 0.1;      // CXCL13 high in the first half
      values[i * 2 + 1] = i < 3 ? 0.1 : 5;  // COL1A1 high in the second
    }
    const result = {
      slide: "s.svs", genes, values, side: 100,
      patches: Array.from({ length: n }, (_, i) => ({ x: i * 100, y: 0, index: i, col: i, row: 0, size: 100 })),
      modelId: "t", modelName: "t", roiId: null, ms: 0, createdAt: "",
    } as unknown as SpatialResult;

    const m = moduleFromEnrichment("Aggregate", [stat("CXCL13", 0.95), stat("COL1A1", 0.05)]);
    const score = scoreSignature(result, m)!;
    // The half the module describes must score above the half it does not.
    expect(score[0]).toBeGreaterThan(score[5]);
  });
});

/** A map on a regular lattice, with `hot` marking the cells to light up. */
function lattice(cols: number, rows: number, hot: (x: number, y: number) => boolean, step = 100) {
  const patches = [];
  const values: number[] = [];
  for (let y = 0; y < rows; y++)
    for (let x = 0; x < cols; x++) {
      patches.push({ x: x * step, y: y * step, index: patches.length, col: x, row: y, size: step });
      values.push(hot(x, y) ? 10 : 0);
    }
  const result = {
    slide: "s.svs", genes: ["A"], values: new Float32Array(values), side: step,
    patches, modelId: "t", modelName: "t", roiId: null, ms: 0, createdAt: "",
  } as unknown as SpatialResult;
  return { result, field: new Float32Array(values) };
}

describe("recovering the grid a map sits on", () => {
  it("finds the spacing, the origin and the extent", () => {
    const { result } = lattice(5, 4, () => false);
    const l = latticeOf(result);
    expect([l.cols, l.rows]).toEqual([5, 4]);
    expect([l.originX, l.originY]).toEqual([0, 0]);
    expect(l.step).toBe(100);
    expect(l.cell[0]).toBe(0);
  });

  it("uses the stride, not the patch side, when patches overlap", () => {
    // Half-stride: patches are 100 wide but 50 apart.
    const patches = Array.from({ length: 6 }, (_, i) => ({
      x: i * 50, y: 0, index: i, col: i, row: 0, size: 100,
    }));
    const result = {
      slide: "s.svs", genes: ["A"], values: new Float32Array(6), side: 100,
      patches, modelId: "t", modelName: "t", roiId: null, ms: 0, createdAt: "",
    } as unknown as SpatialResult;
    const l = latticeOf(result);
    // Using the side would fold six patches into three cells and lose half.
    expect(l.step).toBe(50);
    expect(l.cols).toBe(6);
  });

  it("marks cells no patch covers, so nothing is invented there", () => {
    const { result } = lattice(3, 3, () => false);
    const sparse = { ...result, patches: result.patches.filter((_, i) => i !== 4) } as SpatialResult;
    expect(latticeOf(sparse).cell[4]).toBe(-1);
  });
});

describe("finding the regions that match", () => {
  it("traces a blob, in slide coordinates", () => {
    const { result, field } = lattice(12, 12, (x, y) => x >= 3 && x <= 7 && y >= 3 && y <= 7);
    const regions = similarRegions(field, latticeOf(result), { threshold: 5, minPatches: 4 });
    expect(regions).toHaveLength(1);

    const xs = regions[0].ring.map((p) => p[0]);
    const ys = regions[0].ring.map((p) => p[1]);
    // The blob spans cells 3..7, so 300..800 in slide pixels.
    expect(Math.min(...xs)).toBeCloseTo(300, 0);
    expect(Math.max(...xs)).toBeCloseTo(800, 0);
    expect(Math.min(...ys)).toBeCloseTo(300, 0);
    expect(regions[0].patches).toBeGreaterThan(15);
  });

  it("separates blobs that do not touch", () => {
    const { result, field } = lattice(16, 8,
      (x, y) => (x >= 1 && x <= 4 && y >= 2 && y <= 5) || (x >= 10 && x <= 13 && y >= 2 && y <= 5));
    expect(similarRegions(field, latticeOf(result), { threshold: 5, minPatches: 4 })).toHaveLength(2);
  });

  it("drops anything smaller than the floor", () => {
    const { result, field } = lattice(10, 10, (x, y) => x === 5 && y === 5);
    expect(similarRegions(field, latticeOf(result), { threshold: 5, minPatches: 4 })).toEqual([]);
  });

  it("returns nothing when the field never clears the threshold", () => {
    const { result, field } = lattice(8, 8, () => false);
    expect(similarRegions(field, latticeOf(result), { threshold: 5 })).toEqual([]);
  });

  it("will not grow a region across cells with no patch under them", () => {
    // Two hot blocks that would join if the empty column between them were
    // filled in — the map says nothing there, so they must stay apart.
    const { result, field } = lattice(9, 5, (x, y) => y >= 1 && y <= 3 && (x <= 3 || x >= 5));
    const sparse = {
      ...result,
      patches: result.patches.filter((p) => p.x / 100 !== 4),
    } as SpatialResult;
    const kept = field.filter((_, i) => result.patches[i].x / 100 !== 4);
    expect(similarRegions(kept, latticeOf(sparse), { threshold: 5, minPatches: 4 }))
      .toHaveLength(2);
  });
});

describe("choosing a threshold", () => {
  it("reads a percentile of the field", () => {
    const v = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(percentileThreshold(v, 0)).toBe(0);
    expect(percentileThreshold(v, 100)).toBe(9);
    expect(percentileThreshold(v, 50)).toBeGreaterThanOrEqual(4);
  });

  it("is what makes a threshold mean the same thing on two slides", () => {
    // Same shape, different scale — a percentile picks the same cells.
    const a = new Float32Array([1, 2, 3, 4, 100]);
    const b = new Float32Array([10, 20, 30, 40, 1000]);
    const keep = (v: Float32Array) => v.filter((x) => x > percentileThreshold(v, 75)).length;
    expect(keep(a)).toBe(keep(b));
  });
});

describe("turning a gradient into a field", () => {
  const g = (name: string, rho: number): GradientStat =>
    ({ name, rho, mean: 1, meanStart: 0, meanEnd: 0, p: 0, q: 0 });

  it("carries rho as the weight, sign and all", () => {
    const m = moduleFromGradient("Axis", [g("RISES", 0.9), g("FALLS", -0.85), g("FLAT", 0.01)]);
    const by = Object.fromEntries(m.genes.map((x) => [x.gene, x.weight]));
    expect(by.RISES).toBeCloseTo(0.9, 4);
    // A gradient is the pairing of what rises with what falls; dropping the
    // fallers would throw away half the measurement.
    expect(by.FALLS).toBeCloseTo(-0.85, 4);
    expect(by.FLAT).toBeUndefined();
  });

  it("scores as a field that increases along the axis it came from", () => {
    // Six patches left to right: one gene rising, one falling.
    const genes = ["UP", "DOWN"];
    const n = 6;
    const values = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      values[i * 2] = i;         // rises
      values[i * 2 + 1] = n - i; // falls
    }
    const result = {
      slide: "s.svs", genes, values, side: 100,
      patches: Array.from({ length: n }, (_, i) => ({ x: i * 100, y: 0, index: i, col: i, row: 0, size: 100 })),
      modelId: "t", modelName: "t", roiId: null, ms: 0, createdAt: "",
    } as unknown as SpatialResult;

    const m = moduleFromGradient("Axis", [g("UP", 0.95), g("DOWN", -0.95)]);
    const score = scoreSignature(result, m)!;
    // Monotonic along the same direction the gradient was measured in — which
    // is the whole point of drawing it.
    for (let i = 1; i < n; i++) expect(score[i]).toBeGreaterThan(score[i - 1]);
  });

  it("flips when the arrow that produced it was reversed", () => {
    const forward = moduleFromGradient("A", [g("UP", 0.9), g("DOWN", -0.9)]);
    const reversed = moduleFromGradient("A", [g("UP", -0.9), g("DOWN", 0.9)]);
    const w = (m: typeof forward, gene: string) => m.genes.find((x) => x.gene === gene)!.weight;
    expect(w(reversed, "UP")).toBeCloseTo(-w(forward, "UP"), 6);
  });
});
