import { describe, expect, it } from "vitest";
import {
  axesIn, geneGradient, NoPatchesAlongAxis, patchesAlong, signatureGradient,
} from "../ml/gradient";
import { scoreSignature, type Signature } from "../ml/signatures";
import { makeAnnotation } from "../annotate/store";
import { AXIS_CLASS_ID } from "../annotate/types";
import type { SpatialResult } from "../ml/spatialResult";

/**
 * A row of patches left to right, with a gene that rises along it, one that
 * falls, and one that is flat. The answer is known before the test runs, which
 * is the only way a correlation test proves anything.
 */
const SIDE = 100;
function strip(n = 30): SpatialResult {
  const genes = ["RISES", "FALLS", "FLAT"];
  const values = new Float32Array(n * genes.length);
  for (let i = 0; i < n; i++) {
    values[i * 3 + 0] = i;                     // rises toward the head
    values[i * 3 + 1] = n - i;                 // falls
    values[i * 3 + 2] = 5;                     // says nothing
  }
  return {
    slide: "s.svs", genes, values,
    patches: Array.from({ length: n }, (_, i) => ({
      x: i * SIDE, y: 0, index: i, col: i, row: 0, size: SIDE,
    })),
    side: SIDE, modelId: "t", modelName: "t", roiId: null, ms: 0,
    createdAt: new Date().toISOString(),
  } as unknown as SpatialResult;
}

const axis = (x0: number, y0: number, x1: number, y1: number) =>
  makeAnnotation({ type: "LineString", coordinates: [[x0, y0], [x1, y1]] },
                 { classId: AXIS_CLASS_ID });

describe("projecting patches onto an axis", () => {
  it("orders them along the arrow, tail at 0 and head at 1", () => {
    const r = strip();
    const along = patchesAlong(r, axis(50, 50, 2950, 50), 200);
    expect(along.length).toBe(30);
    expect(along[0].t).toBeCloseTo(0, 2);
    expect(along[along.length - 1].t).toBeCloseTo(1, 2);
  });

  it("reverses when the arrow is drawn the other way", () => {
    const r = strip();
    const along = patchesAlong(r, axis(2950, 50, 50, 50), 200);
    // Patch 0 is now at the far end.
    expect(along.find((a) => a.index === 0)!.t).toBeCloseTo(1, 2);
  });

  it("drops what the arrow does not point at", () => {
    const r = strip();
    // Half the length, so only the first half of the strip is covered.
    expect(patchesAlong(r, axis(50, 50, 1500, 50), 200).length).toBeLessThan(20);
    // A narrow corridor on a line that runs past the patches catches nothing.
    expect(patchesAlong(r, axis(50, 5000, 2950, 5000), 50).length).toBe(0);
  });
});

describe("what changes along an axis", () => {
  it("signs the gradient by which way the arrow points", () => {
    const r = strip();
    const out = geneGradient(r, axis(50, 50, 2950, 50), 200, 0.5);
    const by = Object.fromEntries(out.items.map((i) => [i.name, i]));

    expect(by.RISES.rho).toBeGreaterThan(0.95);
    expect(by.FALLS.rho).toBeLessThan(-0.95);
    expect(Math.abs(by.FLAT.rho)).toBeLessThan(0.01);
    // Ordered by steepness regardless of direction: FLAT must be last.
    expect(out.items[out.items.length - 1].name).toBe("FLAT");
  });

  it("flips every sign when the arrow is reversed", () => {
    const r = strip();
    const fwd = geneGradient(r, axis(50, 50, 2950, 50), 200, 0.5);
    const rev = geneGradient(r, axis(2950, 50, 50, 50), 200, 0.5);
    const f = Object.fromEntries(fwd.items.map((i) => [i.name, i.rho]));
    const b = Object.fromEntries(rev.items.map((i) => [i.name, i.rho]));
    expect(b.RISES).toBeCloseTo(-f.RISES, 4);
    expect(b.FALLS).toBeCloseTo(-f.FALLS, 4);
  });

  it("reports the ends the way round the arrow was drawn", () => {
    const r = strip();
    const out = geneGradient(r, axis(50, 50, 2950, 50), 200, 0.5);
    const rises = out.items.find((i) => i.name === "RISES")!;
    expect(rises.meanEnd).toBeGreaterThan(rises.meanStart);
  });

  it("gives the axis length in microns when the slide has a scale", () => {
    const r = strip();
    const out = geneGradient(r, axis(50, 50, 2950, 50), 200, 0.25);
    expect(out.lengthPx).toBeCloseTo(2900, 0);
    expect(out.lengthUm).toBeCloseTo(725, 0);
    expect(geneGradient(r, axis(50, 50, 2950, 50), 200, null).lengthUm).toBeNull();
  });

  it("refuses an axis with too little under it", () => {
    const r = strip();
    expect(() => geneGradient(r, axis(50, 50, 300, 50), 200, 0.5))
      .toThrow(NoPatchesAlongAxis);
  });
});

describe("cell types along an axis", () => {
  const signatures: Signature[] = [
    { name: "climbing", genes: [{ gene: "RISES", weight: 1 }] },
    { name: "sinking", genes: [{ gene: "FALLS", weight: 1 }] },
  ];

  it("ranks the module that tracks the direction", () => {
    const r = strip();
    const out = signatureGradient(r, axis(50, 50, 2950, 50), 200, 0.5,
                                  signatures, scoreSignature);
    expect(out.kind).toBe("signature");
    const by = Object.fromEntries(out.items.map((i) => [i.name, i.rho]));
    expect(by.climbing).toBeGreaterThan(0.95);
    expect(by.sinking).toBeLessThan(-0.95);
  });

  it("leaves out a module this map cannot score", () => {
    const r = strip();
    const out = signatureGradient(r, axis(50, 50, 2950, 50), 200, 0.5, [
      ...signatures, { name: "absent", genes: [{ gene: "NOPE", weight: 1 }] },
    ], scoreSignature);
    expect(out.items.map((i) => i.name)).not.toContain("absent");
  });
});

describe("finding the axes on a slide", () => {
  it("takes only the arrows, not every line", () => {
    const arrow = axis(0, 0, 10, 0);
    const line = makeAnnotation(
      { type: "LineString", coordinates: [[0, 0], [5, 5]] }, { classId: "other" });
    const region = makeAnnotation(
      { type: "Polygon", coordinates: [[[0, 0], [1, 0], [1, 1], [0, 0]]] },
      { classId: AXIS_CLASS_ID });
    expect(axesIn([arrow, line, region]).map((a) => a.id)).toEqual([arrow.id]);
  });
});
