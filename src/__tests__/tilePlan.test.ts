import { describe, expect, it } from "vitest";
import { planTile, TILE_SIZE } from "../viewer/openslideTileSource";

/**
 * Every tile must read a region that exists.
 *
 * The mapping crosses three coordinate frames — the synthetic power-of-two grid
 * OpenSeadragon assumes, level-0 slide pixels offset by the MIRAX scan region,
 * and the chosen OpenSlide level's own pixels. A mistake in any of them does
 * not throw: it reads the wrong part of the slide, or a region off the end that
 * the reader answers with blank, which on screen is a tile that never fills in.
 */

interface Shape {
  name: string;
  /** Full canvas at level 0. */
  canvas: [number, number];
  downsamples: number[];
  /** Scan region within the canvas: MIRAX's bounds-*. */
  bounds: { x: number; y: number; width: number; height: number } | null;
}

const SHAPES: Shape[] = [
  {
    name: "MIRAX with a large offset scan region",
    canvas: [170000, 410000],
    downsamples: [1, 2, 4, 8, 16, 32, 64, 128],
    bounds: { x: 34048, y: 152832, width: 102912, height: 120832 },
  },
  {
    name: "MIRAX whose region touches the canvas edge",
    canvas: [102912 + 34048, 120832 + 152832],
    downsamples: [1, 2, 4, 8, 16, 32, 64],
    bounds: { x: 34048, y: 152832, width: 102912, height: 120832 },
  },
  {
    name: "SVS, no bounds",
    canvas: [127412, 25374],
    downsamples: [1, 4, 16, 32],
    bounds: null,
  },
  {
    name: "shallow NDPI",
    canvas: [59520, 61952],
    downsamples: [1, 2, 4, 8, 16, 32],
    bounds: null,
  },
];

function frameOf(s: Shape) {
  const width = s.bounds?.width ?? s.canvas[0];
  const height = s.bounds?.height ?? s.canvas[1];
  return {
    width,
    height,
    offsetX: s.bounds?.x ?? 0,
    offsetY: s.bounds?.y ?? 0,
    maxLevel: Math.max(0, Math.ceil(Math.log2(Math.max(width, height)))),
  };
}

function metaOf(s: Shape) {
  return {
    levels: s.downsamples.map((d) => ({
      width: Math.floor(s.canvas[0] / d),
      height: Math.floor(s.canvas[1] / d),
      downsample: d,
    })),
  };
}

describe.each(SHAPES)("$name", (shape) => {
  const meta = metaOf(shape);
  const frame = frameOf(shape);

  /** Levels worth walking: the finest, the coarsest useful, and the middle. */
  const levels = [frame.maxLevel, frame.maxLevel - 1, frame.maxLevel - 4, frame.maxLevel - 8]
    .filter((l) => l >= 0);

  it("never reads past the end of the level it chose", () => {
    for (const level of levels) {
      const scale = 2 ** (frame.maxLevel - level);
      const cols = Math.ceil(Math.ceil(frame.width / scale) / TILE_SIZE);
      const rows = Math.ceil(Math.ceil(frame.height / scale) / TILE_SIZE);
      // Corners and edges are where the arithmetic goes wrong.
      const probe = [0, 1, Math.floor(cols / 2), cols - 2, cols - 1].filter((c) => c >= 0 && c < cols);
      const probeRows = [0, 1, Math.floor(rows / 2), rows - 2, rows - 1].filter((r) => r >= 0 && r < rows);

      for (const col of probe) {
        for (const row of probeRows) {
          const plan = planTile(meta, frame, level, col, row);
          if (!plan) continue;
          const lvl = meta.levels[plan.osLevel];
          const ds = lvl.downsample;
          expect(plan.rw).toBeGreaterThan(0);
          expect(plan.rh).toBeGreaterThan(0);
          expect(Math.floor(plan.x0 / ds) + plan.rw).toBeLessThanOrEqual(lvl.width);
          expect(Math.floor(plan.y0 / ds) + plan.rh).toBeLessThanOrEqual(lvl.height);
        }
      }
    }
  });

  it("only ever downscales, never upscales", () => {
    for (const level of levels) {
      const scale = 2 ** (frame.maxLevel - level);
      const plan = planTile(meta, frame, level, 0, 0);
      if (!plan) continue;
      // The chosen level's downsample must not exceed what the tile needs, or
      // the tile is being magnified from data too coarse for it.
      expect(meta.levels[plan.osLevel].downsample).toBeLessThanOrEqual(scale + 1e-6);
    }
  });

  it("covers the scan region without a gap between tiles", () => {
    const level = frame.maxLevel;
    const a = planTile(meta, frame, level, 0, 0)!;
    const b = planTile(meta, frame, level, 1, 0)!;
    const c = planTile(meta, frame, level, 0, 1)!;
    // Adjacent tiles start exactly one tile apart in level-0 pixels.
    expect(b.x0 - a.x0).toBe(TILE_SIZE);
    expect(c.y0 - a.y0).toBe(TILE_SIZE);
  });

  it("starts at the scan region, not at the canvas origin", () => {
    const plan = planTile(meta, frame, frame.maxLevel, 0, 0)!;
    expect(plan.x0).toBe(frame.offsetX);
    expect(plan.y0).toBe(frame.offsetY);
  });

  /**
   * A tile drawn at less than its full size is stretched to fill. That is fine
   * for the last tile in a row, and wrong anywhere else — a whole-grid stretch
   * is how a slide ends up subtly misaligned with its annotations.
   */
  it("reads a full tile everywhere except the last row and column", () => {
    const level = frame.maxLevel;
    const plan = planTile(meta, frame, level, 1, 1)!;
    expect(plan.rw).toBe(plan.tw);
    expect(plan.rh).toBe(plan.th);
    expect(plan.tw).toBe(TILE_SIZE);
  });
});
