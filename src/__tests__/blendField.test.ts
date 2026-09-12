import { describe, expect, it } from "vitest";
import { blendField } from "../ml/blend";
import type { Patch } from "../ml/patchGrid";

/** A row of overlapping patches: width `side`, spaced `step` apart. */
function strided(values: number[], side: number, step: number) {
  const patches: Patch[] = values.map((_, i) => ({
    x: i * step, y: 0, index: i, col: i, row: 0, size: side,
  })) as unknown as Patch[];
  return { patches, values: new Float32Array(values) };
}

describe("blending a strided expression map", () => {
  it("produces a raster at the stride, not at the patch width", () => {
    const { patches, values } = strided([0, 1, 2, 3, 4, 5, 6, 7], 400, 100);
    const b = blendField(patches, values, null, 400, 100)!;
    expect(b.cell).toBe(100);
    // Eight patches 100 apart, each 400 wide: the covered span is 1100 px.
    expect(b.cols).toBeGreaterThan(8);
    expect(b.classes).toBe(2);
  });

  it("averages overlapping patches instead of letting the last one win", () => {
    // Two patches covering the same ground with very different values. Drawn as
    // squares one would simply cover the other; blended, the middle is between.
    const patches = [
      { x: 0, y: 0, index: 0, col: 0, row: 0, size: 200 },
      { x: 100, y: 0, index: 1, col: 1, row: 0, size: 200 },
    ] as unknown as Patch[];
    const b = blendField(patches, new Float32Array([0, 10]), null, 200, 100)!;
    const at = (i: number) => b.probs[i * 2];
    const mid = Math.floor(b.cols / 2);
    expect(at(mid)).toBeGreaterThan(0);
    expect(at(mid)).toBeLessThan(10);
  });

  it("weights by distance from a patch's centre, so edges do not dominate", () => {
    // One patch at 0 with value 10, one far right with value 0. A cell under
    // the first patch's centre must be nearer 10 than a cell at its rim.
    const { patches, values } = strided([10, 0, 0, 0, 0, 0], 400, 100);
    const b = blendField(patches, values, null, 400, 100)!;
    const at = (i: number) => b.probs[i * 2];
    // Cell 2 sits at the first patch's centre (x=200 of a patch spanning 0-400).
    expect(at(2)).toBeGreaterThan(at(5));
  });

  it("blanks cells whose weight came from off-tissue patches", () => {
    const { patches, values } = strided([5, 5, 5, 5, 5, 5], 200, 100);
    const mask = new Uint8Array([0, 0, 0, 1, 1, 1]);
    const b = blendField(patches, values, mask, 200, 100)!;
    const covered = (i: number) => b.probs[i * 2 + 1];
    // The far left is reached only by masked-out patches; the far right only by
    // kept ones. Half is the midpoint the renderer cuts at.
    expect(covered(0)).toBeLessThan(0.5);
    expect(covered(b.cols - 1)).toBeGreaterThan(0.5);
  });

  it("keeps every patch when nothing is masked", () => {
    const { patches, values } = strided([1, 2, 3, 4], 200, 100);
    const b = blendField(patches, values, null, 200, 100)!;
    for (let i = 0; i < b.cols * b.rows; i++) {
      // Every cell any patch reached is fully covered.
      if (b.probs[i * 2 + 1] > 0) expect(b.probs[i * 2 + 1]).toBeCloseTo(1, 6);
    }
  });

  it("returns nothing for an empty map rather than an empty raster", () => {
    expect(blendField([], new Float32Array(0), null, 200, 100)).toBeNull();
  });
});
