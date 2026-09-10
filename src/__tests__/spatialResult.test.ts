import { describe, expect, it } from "vitest";
import {
  colourFor, geneValues, robustRange, toCsv, type SpatialResult,
} from "../ml/spatialResult";
import { resampleTo } from "../ml/spatialController";

const patches = Array.from({ length: 4 }, (_, i) => ({
  index: i, col: i % 2, row: Math.floor(i / 2), x: 1000 + (i % 2) * 224, y: 2000 + Math.floor(i / 2) * 224, size: 224,
}));

const result: SpatialResult = {
  genes: ["EPCAM", "CD3D", "COL1A1"],
  // Row-major by patch: patch 0 = [1,2,3], patch 1 = [4,5,6], …
  values: Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
  patches,
  side: 224,
  modelId: "deepspot-m-3",
  modelName: "DeepSpot-M (3 genes)",
  slide: "case_01.svs",
  roiId: null,
  ms: 1200,
  createdAt: "2026-09-10T00:00:00.000Z",
};

describe("reading a prediction", () => {
  /**
   * Values are stored row-major by patch, so pulling one gene means striding.
   * Getting the stride wrong returns another gene's numbers under this gene's
   * name — plausible, wrong, and invisible.
   */
  it("pulls one gene across every patch", () => {
    expect(Array.from(geneValues(result, "EPCAM")!)).toEqual([1, 4, 7, 10]);
    expect(Array.from(geneValues(result, "CD3D")!)).toEqual([2, 5, 8, 11]);
    expect(Array.from(geneValues(result, "COL1A1")!)).toEqual([3, 6, 9, 12]);
  });

  it("returns nothing for a gene the model does not predict", () => {
    expect(geneValues(result, "MKI67")).toBe(null);
  });

  it("exports a row per patch with its coordinates", () => {
    const csv = toCsv(result).trim().split("\n");
    expect(csv[0]).toBe("patch,col,row,x,y,size,EPCAM,CD3D,COL1A1");
    expect(csv.length).toBe(1 + patches.length);
    expect(csv[1].startsWith("0,0,0,1000,2000,224,")).toBe(true);
  });
});

describe("the colour scale", () => {
  /**
   * One saturated patch — a fold, a pen mark — would otherwise set the maximum
   * and flatten everything else to the bottom of the scale, which reads as "no
   * expression anywhere" when the truth is "one outlier".
   */
  it("clips to percentiles so an outlier does not flatten the field", () => {
    const ordinary = Float32Array.from(Array.from({ length: 100 }, (_, i) => i / 99));
    const withSpike = Float32Array.from([...ordinary, 1000]);
    expect(robustRange(withSpike).max).toBeLessThan(10);
  });

  it("widens a flat field rather than dividing by zero", () => {
    const flat = Float32Array.from([5, 5, 5, 5]);
    const r = robustRange(flat);
    expect(r.max).toBeGreaterThan(r.min);
    expect(Number.isFinite(colourFor(5, r)[0])).toBe(true);
  });

  it("runs dark to light across the range, and clamps outside it", () => {
    const r = { min: 0, max: 1 };
    const low = colourFor(0, r);
    const high = colourFor(1, r);
    const sum = (c: [number, number, number]) => c[0] + c[1] + c[2];
    expect(sum(high)).toBeGreaterThan(sum(low));
    expect(colourFor(-5, r)).toEqual(low);
    expect(colourFor(50, r)).toEqual(high);
  });
});

describe("resampling a tile to the model's input", () => {
  it("fills every pixel and leaves it opaque", () => {
    const src = new Uint8ClampedArray(8 * 8 * 4).fill(120);
    const dst = new Uint8ClampedArray(4 * 4 * 4);
    resampleTo(src, 8, 8, dst, 4);
    expect(dst.every((v, i) => (i % 4 === 3 ? v === 255 : v === 120))).toBe(true);
  });

  it("keeps the corners in their corners", () => {
    const src = new Uint8ClampedArray(2 * 2 * 4);
    // Top-left red, bottom-right blue.
    src[0] = 255; src[3] = 255;
    src[12] = 0; src[14] = 255; src[15] = 255;
    const dst = new Uint8ClampedArray(4 * 4 * 4);
    resampleTo(src, 2, 2, dst, 4);
    expect(dst[0]).toBe(255);                       // top-left still red
    expect(dst[(4 * 4 - 1) * 4 + 2]).toBe(255);     // bottom-right still blue
  });
});
