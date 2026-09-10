import { describe, expect, it } from "vitest";
import { blendPredictions } from "../ml/blend";
import { buildPatchGrid } from "../ml/patchGrid";
import type { Patch } from "../ml/patchGrid";

const SIDE = 224;

/** Patches on a regular lattice at the given stride. */
function lattice(cols: number, rows: number, stride: number): Patch[] {
  const out: Patch[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      out.push({ index: out.length, col: c, row: r, x: c * stride, y: r * stride, size: SIDE });
    }
  }
  return out;
}

describe("blending overlapping patches", () => {
  it("gives a raster at the stride's resolution, not the patch's", () => {
    const stride = SIDE / 4;
    const patches = lattice(4, 4, stride);
    const probs = new Float32Array(patches.length * 2);
    for (let i = 0; i < patches.length; i++) probs[i * 2] = 1;

    const blend = blendPredictions(patches, probs, 2, SIDE, stride)!;
    expect(blend.cell).toBe(stride);
    // The lattice spans 3 strides plus a patch in each direction.
    expect(blend.cols).toBe(Math.ceil((3 * stride + SIDE) / stride));
    expect(blend.rows).toBe(blend.cols);
  });

  it("keeps every cell a probability distribution", () => {
    const stride = SIDE / 2;
    const patches = lattice(3, 3, stride);
    const probs = new Float32Array(patches.length * 3);
    for (let i = 0; i < patches.length; i++) {
      // A different class per patch, so the overlaps genuinely disagree.
      probs[i * 3 + (i % 3)] = 1;
    }

    const blend = blendPredictions(patches, probs, 3, SIDE, stride)!;
    for (let i = 0; i < blend.cols * blend.rows; i++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) sum += blend.probs[i * 3 + k];
      // Either untouched, or a distribution.
      expect(sum === 0 || Math.abs(sum - 1) < 1e-5).toBe(true);
    }
  });

  /**
   * The point of the exercise: where two patches disagree, the blend has to
   * move between them rather than jump, or the result is the same blocky map
   * with more steps in it.
   */
  it("transitions across a boundary instead of stepping", () => {
    const stride = SIDE / 4;
    // A row of patches; the left half say class 0, the right half class 1.
    const patches = lattice(8, 1, stride);
    const probs = new Float32Array(patches.length * 2);
    patches.forEach((_, i) => { probs[i * 2 + (i < 4 ? 0 : 1)] = 1; });

    const blend = blendPredictions(patches, probs, 2, SIDE, stride)!;
    const row = 0;
    const series: number[] = [];
    for (let c = 0; c < blend.cols; c++) {
      series.push(blend.probs[(row * blend.cols + c) * 2 + 1]);
    }

    // Monotone left to right, and passing through intermediate values rather
    // than only 0 and 1.
    for (let i = 1; i < series.length; i++) expect(series[i]).toBeGreaterThanOrEqual(series[i - 1] - 1e-6);
    expect(series.some((v) => v > 0.15 && v < 0.85)).toBe(true);
  });

  it("weights a patch's centre above its corners", () => {
    const stride = SIDE / 2;
    // Two patches side by side; only the left one is confident.
    const patches = lattice(2, 1, stride);
    const probs = Float32Array.from([1, 0, 0, 1]);
    const blend = blendPredictions(patches, probs, 2, SIDE, stride)!;

    // Cell 0 is reached only by the left patch; cell 1 is the overlap, where
    // the two are equidistant from their centres and cancel.
    const at = (cell: number) => blend.probs[cell * 2];
    expect(at(0)).toBeGreaterThan(at(1));
    expect(at(1)).toBeGreaterThan(at(2));
    // And the overlap is genuinely mixed rather than taking one side.
    expect(at(1)).toBeGreaterThan(0.2);
    expect(at(1)).toBeLessThan(0.8);
  });

  it("returns nothing when there is nothing to blend", () => {
    expect(blendPredictions([], new Float32Array(), 2, SIDE, 56)).toBe(null);
  });

  it("matches the grid the patcher actually builds", () => {
    const grid = buildPatchGrid({ x: 1000, y: 2000, width: 224 * 4, height: 224 * 3 }, 1, 0.5, {
      patchPx: 224,
      stridePx: 112,
    });
    expect(grid.stridePx).toBe(112);
    const probs = new Float32Array(grid.patches.length * 2);
    for (let i = 0; i < grid.patches.length; i++) probs[i * 2] = 1;

    const blend = blendPredictions(grid.patches, probs, 2, 224, 112)!;
    expect(blend.originX).toBe(1000);
    expect(blend.originY).toBe(2000);
    expect(blend.cell).toBe(112);
  });
});
