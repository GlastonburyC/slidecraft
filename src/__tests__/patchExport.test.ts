import { describe, expect, it } from "vitest";
import { buildPatchGrid } from "../ml/patchGrid";
import { toPatchGeoJSON, toPatchManifest } from "../ml/patchExport";
import type { SlideMeta } from "../slide/types";

const meta: SlideMeta = {
  name: "case_01.svs", bytes: 0, vendor: "test", mppX: 0.25, mppY: 0.25,
  objectivePower: 40, bounds: null, backgroundColor: null,
  levels: [
    { level: 0, width: 20000, height: 10000, downsample: 1 },
    { level: 1, width: 5000, height: 2500, downsample: 4 },
  ],
  properties: {},
};

const region = { x: 1000, y: 2000, width: 2048, height: 1024 };

describe("exporting a patch grid", () => {
  it("describes patches in the coordinates read_region takes", () => {
    const grid = buildPatchGrid(region, 1, meta.mppX, { patchPx: 256, level: 0 });
    const m = toPatchManifest(grid, meta);

    expect(m.count).toBe(grid.patches.length);
    expect(m.patchPx).toBe(256);
    // At level 0 the level-pixel and level-0-pixel sizes agree.
    expect(m.patchPx0).toBe(256);
    expect(m.patchUm).toBeCloseTo(256 * 0.25, 6);
    // The first patch starts at the region's own origin.
    expect(m.patches[0].x).toBe(region.x);
    expect(m.patches[0].y).toBe(region.y);
    // And every patch lies inside the region that was asked for.
    for (const p of m.patches) {
      expect(p.x).toBeGreaterThanOrEqual(region.x);
      expect(p.y).toBeGreaterThanOrEqual(region.y);
      expect(p.x + m.patchPx0).toBeLessThanOrEqual(region.x + region.width);
      expect(p.y + m.patchPx0).toBeLessThanOrEqual(region.y + region.height);
    }
  });

  /**
   * A patch specified at a coarser level covers proportionally more slide.
   * Getting this wrong silently patches the wrong scale — the grid still looks
   * plausible, and every downstream feature is computed on the wrong field.
   */
  it("scales level-0 extents by the level's downsample", () => {
    const grid = buildPatchGrid(region, 4, meta.mppX, { patchPx: 256, level: 1 });
    const m = toPatchManifest(grid, meta);
    expect(m.patchPx).toBe(256);
    expect(m.patchPx0).toBe(1024);
    expect(m.patchUm).toBeCloseTo(1024 * 0.25, 6);
    expect(m.level).toBe(1);
  });

  it("emits the same squares as GeoJSON", () => {
    const grid = buildPatchGrid(region, 1, meta.mppX, { patchPx: 256, level: 0 });
    const fc = toPatchGeoJSON(grid, meta);
    expect(fc.features.length).toBe(grid.patches.length);

    const ring = fc.features[0].geometry.coordinates[0];
    expect(ring.length).toBe(5);
    expect(ring[0]).toEqual(ring[4]); // closed
    const [x0, y0] = ring[0];
    const [x2, y2] = ring[2];
    expect(x2 - x0).toBe(256);
    expect(y2 - y0).toBe(256);
  });

  it("keeps the manifest and the GeoJSON describing one grid", () => {
    const grid = buildPatchGrid(region, 4, meta.mppX, { patchPx: 128, level: 1 });
    const m = toPatchManifest(grid, meta);
    const fc = toPatchGeoJSON(grid, meta);
    expect(fc.features.length).toBe(m.count);
    for (let i = 0; i < m.count; i++) {
      expect(fc.features[i].geometry.coordinates[0][0]).toEqual([m.patches[i].x, m.patches[i].y]);
    }
  });

  it("says nothing about microns when the slide reports no scale", () => {
    const grid = buildPatchGrid(region, 1, null, { patchPx: 256, level: 0 });
    expect(toPatchManifest(grid, { ...meta, mppX: null }).patchUm).toBe(null);
  });
});

describe("patches as objects", () => {
  /**
   * A patch that becomes an object has to sit exactly where the grid drew it —
   * the preview is what the scale was judged on, and an object offset from it
   * would be classified on a different field than the one that was checked.
   */
  it("makes a square per patch, matching the exported geometry", () => {
    const grid = buildPatchGrid(region, 4, meta.mppX, { patchPx: 128, level: 1 });
    const side = grid.patchPx * grid.downsample;
    const fc = toPatchGeoJSON(grid, meta);

    for (const [i, p] of grid.patches.entries()) {
      const ring = fc.features[i].geometry.coordinates[0];
      expect(ring[0]).toEqual([p.x, p.y]);
      expect(ring[2]).toEqual([p.x + side, p.y + side]);
      // The square is the size the grid says, in level-0 pixels.
      expect(ring[2][0] - ring[0][0]).toBe(side);
    }
  });

  it("keeps patches inside the ROI they were laid over", () => {
    const grid = buildPatchGrid(region, 1, meta.mppX, { patchPx: 256, level: 0 });
    const side = grid.patchPx * grid.downsample;
    for (const p of grid.patches) {
      expect(p.x + side).toBeLessThanOrEqual(region.x + region.width);
      expect(p.y + side).toBeLessThanOrEqual(region.y + region.height);
    }
  });
});
