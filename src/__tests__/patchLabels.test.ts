import { describe, expect, it } from "vitest";
import { labelPatches } from "../ml/patchLabels";
import { makeAnnotation } from "../annotate/store";
import type { Patch } from "../ml/patchGrid";

const grid = (cols: number, rows: number, size = 224): Patch[] => {
  const out: Patch[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      out.push({ index: out.length, col: c, row: r, x: c * size, y: r * size, size });
    }
  }
  return out;
};

const box = (x: number, y: number, w: number, h: number, classId: string) =>
  makeAnnotation(
    { type: "Polygon", coordinates: [[[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]]] },
    { classId },
  );

describe("turning annotations into patch labels", () => {
  it("labels a patch by the region its centre falls in", () => {
    const patches = grid(4, 1);
    // Covers the centres of the first two patches (112 and 336), not the third.
    const { labelled, perClass } = labelPatches(patches, [
      { classId: "tumour", regions: [box(0, 0, 448, 224, "tumour")] },
    ]);
    expect(labelled.map((l) => l.patch.col)).toEqual([0, 1]);
    expect(perClass).toEqual([2]);
  });

  /**
   * Unlabelled tissue is not background. Treating it as a negative is how a
   * head learns that everything you have not looked at is "normal".
   */
  it("leaves patches outside every region unlabelled", () => {
    const patches = grid(4, 1);
    const { labelled } = labelPatches(patches, [
      { classId: "tumour", regions: [box(0, 0, 224, 224, "tumour")] },
    ]);
    expect(labelled.length).toBe(1);
    expect(patches.length).toBe(4);
  });

  it("keeps classes separate and counts each", () => {
    const patches = grid(4, 1);
    const { labelled, perClass, classIds } = labelPatches(patches, [
      { classId: "tumour", regions: [box(0, 0, 224, 224, "tumour")] },
      { classId: "stroma", regions: [box(448, 0, 448, 224, "stroma")] },
    ]);
    expect(classIds).toEqual(["tumour", "stroma"]);
    expect(perClass).toEqual([1, 2]);
    expect(labelled.filter((l) => l.label === 1).map((l) => l.patch.col)).toEqual([2, 3]);
  });

  /**
   * Overlapping regions have to resolve the same way every time, or a retrain
   * on unchanged annotations produces a different head.
   */
  it("resolves overlap by class order, deterministically", () => {
    const patches = grid(2, 1);
    const both = [
      { classId: "a", regions: [box(0, 0, 448, 224, "a")] },
      { classId: "b", regions: [box(0, 0, 448, 224, "b")] },
    ];
    const first = labelPatches(patches, both);
    const second = labelPatches(patches, both);
    expect(first.labelled.every((l) => l.label === 0)).toBe(true);
    expect(second.perClass).toEqual(first.perClass);

    // Reversing the class order reverses the winner, and nothing else.
    const flipped = labelPatches(patches, [both[1], both[0]]);
    expect(flipped.labelled.every((l) => l.label === 0)).toBe(true);
    expect(flipped.classIds).toEqual(["b", "a"]);
  });

  it("ignores points and lines, which enclose no patches", () => {
    const patches = grid(2, 1);
    const point = makeAnnotation({ type: "Point", coordinates: [112, 112] }, { classId: "a" });
    const { labelled } = labelPatches(patches, [{ classId: "a", regions: [point] }]);
    expect(labelled.length).toBe(0);
  });
});
