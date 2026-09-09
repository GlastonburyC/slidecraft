import { describe, expect, it } from "vitest";
import { buildPatchGrid, patchKey } from "../ml/patchGrid";
import { makeAnnotation } from "../annotate/store";

const region = { x: 0, y: 0, width: 1024, height: 512 };

describe("patch grid", () => {
  it("tiles a region with the requested pixel size", () => {
    const g = buildPatchGrid(region, 1, 0.25, { patchPx: 128 });
    expect(g.cols).toBe(8);
    expect(g.rows).toBe(4);
    expect(g.patches).toHaveLength(32);
    expect(g.patches[0]).toMatchObject({ x: 0, y: 0, size: 128, col: 0, row: 0 });
    expect(g.patches.at(-1)).toMatchObject({ x: 896, y: 384 });
  });

  it("reports the physical size the pixel patch covers", () => {
    expect(buildPatchGrid(region, 1, 0.25, { patchPx: 128 }).patchUm).toBe(32);
    // Same pixel patch on a coarser level covers four times the distance.
    expect(buildPatchGrid(region, 4, 0.25, { patchPx: 128 }).patchUm).toBe(128);
    expect(buildPatchGrid(region, 1, null, { patchPx: 128 }).patchUm).toBeNull();
  });

  it("scales patch extent by the level downsample", () => {
    const g = buildPatchGrid(region, 4, 0.25, { patchPx: 128 });
    expect(g.patches[0].size).toBe(512);
    expect(g.cols).toBe(2);
    expect(g.rows).toBe(1);
  });

  it("honours a stride smaller than the patch, producing overlap", () => {
    const g = buildPatchGrid(region, 1, null, { patchPx: 128, stridePx: 64 });
    expect(g.cols).toBe(15);
    expect(g.patches[1].x - g.patches[0].x).toBe(64);
  });

  it("keeps only patches centred inside the given regions", () => {
    const left = makeAnnotation({
      type: "Polygon",
      coordinates: [[[0, 0], [256, 0], [256, 512], [0, 512], [0, 0]]],
    });
    const g = buildPatchGrid(region, 1, null, { patchPx: 128, within: [left] });
    expect(g.patches.every((p) => p.x + 64 <= 256)).toBe(true);
    expect(g.patches).toHaveLength(8);
  });

  it("drops patches that leave the scanned bounds", () => {
    const g = buildPatchGrid(region, 1, null, {
      patchPx: 128,
      bounds: { x: 0, y: 0, width: 512, height: 512 },
    });
    expect(g.patches.every((p) => p.x + p.size <= 512)).toBe(true);
    expect(g.cols).toBe(8);
    expect(g.patches).toHaveLength(16);
  });

  it("returns nothing when the region is smaller than one patch", () => {
    const g = buildPatchGrid({ x: 0, y: 0, width: 64, height: 64 }, 1, null, { patchPx: 128 });
    expect(g.patches).toHaveLength(0);
  });

  it("keys patches by position, model and level so caches cannot collide", () => {
    const g = buildPatchGrid(region, 1, null, { patchPx: 128 });
    const a = patchKey("slideA", "uni", 0, 128, g.patches[0]);
    expect(a).not.toBe(patchKey("slideB", "uni", 0, 128, g.patches[0]));
    expect(a).not.toBe(patchKey("slideA", "phikon", 0, 128, g.patches[0]));
    expect(a).not.toBe(patchKey("slideA", "uni", 1, 128, g.patches[0]));
    expect(a).not.toBe(patchKey("slideA", "uni", 0, 256, g.patches[0]));
    expect(a).not.toBe(patchKey("slideA", "uni", 0, 128, g.patches[1]));
  });
});
