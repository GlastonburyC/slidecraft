import { describe, expect, it } from "vitest";
import { detectTissue, scoreOverview } from "../ml/tissue";
import { areaOf } from "../annotate/types";
import type { SlideSource, SlideMeta } from "../slide/types";

/**
 * A synthetic overview: white glass with painted regions.
 *
 * `strong` is well-stained H&E (saturated pink/purple); `faded` is the case
 * that broke the old detector — barely any colour left, but still measurably
 * darker than the glass around it.
 */
function fakeSlide(paint: (x: number, y: number) => [number, number, number]): SlideSource {
  const W = 400;
  const H = 300;
  const meta: SlideMeta = {
    name: "synthetic",
    bytes: 0,
    vendor: "test",
    mppX: 1,
    mppY: 1,
    objectivePower: 20,
    bounds: null,
    backgroundColor: null,
    levels: [{ level: 0, width: W, height: H, downsample: 1 }],
    properties: {},
  };
  return {
    meta,
    async readRegion(_x, _y, _level, w, h) {
      const out = new Uint8ClampedArray(w * h * 4);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const [r, g, b] = paint(Math.round((x / w) * W), Math.round((y / h) * H));
          const i = (y * w + x) * 4;
          out[i] = r; out[i + 1] = g; out[i + 2] = b; out[i + 3] = 255;
        }
      }
      return out;
    },
    async bestLevelForDownsample() { return 0; },
    async close() { /* nothing */ },
  };
}

const inBox = (x: number, y: number, bx: number, by: number, bw: number, bh: number) =>
  x >= bx && x < bx + bw && y >= by && y < by + bh;

describe("tissue detection", () => {
  it("finds strongly stained tissue", async () => {
    const slide = fakeSlide((x, y) =>
      inBox(x, y, 40, 40, 120, 120) ? [180, 90, 170] : [248, 246, 248],
    );
    const r = await detectTissue(slide, { minAreaUm2: 100 });
    expect(r.polygons.length).toBe(1);
    expect(r.coverage).toBeGreaterThan(0.1);
  });

  it("finds faded tissue that has almost no saturation left", async () => {
    // Nearly grey, only ~8% darker than the glass: saturation alone misses this.
    const slide = fakeSlide((x, y) =>
      inBox(x, y, 40, 40, 120, 120) ? [228, 224, 228] : [250, 249, 250],
    );
    const r = await detectTissue(slide, { minAreaUm2: 100 });
    expect(r.polygons.length).toBe(1);
    expect(r.coverage).toBeGreaterThan(0.08);
  });

  it("finds strong and faded regions on the same slide", async () => {
    const slide = fakeSlide((x, y) => {
      if (inBox(x, y, 30, 40, 100, 100)) return [175, 85, 165];
      if (inBox(x, y, 240, 40, 100, 100)) return [229, 225, 229];
      return [250, 249, 250];
    });
    const r = await detectTissue(slide, { minAreaUm2: 100 });
    expect(r.polygons.length).toBe(2);
  });

  it("reaches the edge of a fragment rather than shrinking inside it", async () => {
    // A 120x120 block with a soft 8px falloff, like a real section edge.
    const slide = fakeSlide((x, y) => {
      const dx = Math.max(40 - x, x - 159, 0);
      const dy = Math.max(40 - y, y - 159, 0);
      const d = Math.max(dx, dy);
      if (d > 8) return [250, 249, 250];
      const t = d / 8;
      return [175 + t * 75, 85 + t * 164, 165 + t * 84];
    });
    const r = await detectTissue(slide, { minAreaUm2: 100 });
    const area = r.polygons.reduce(
      (n, rings) => n + areaOf({ type: "Polygon", coordinates: rings }),
      0,
    );
    // The solid core is 120x120 = 14400; capturing the falloff should exceed it.
    expect(area).toBeGreaterThan(14400);
  });

  it("fills a lumen so the region is not fragmented", async () => {
    const slide = fakeSlide((x, y) => {
      const inRing = inBox(x, y, 40, 40, 160, 160) && !inBox(x, y, 90, 90, 60, 60);
      return inRing ? [180, 90, 170] : [250, 249, 250];
    });
    const r = await detectTissue(slide, { minAreaUm2: 100 });
    expect(r.polygons.length).toBe(1);
    // Hole filling means the ring encloses its centre rather than leaving it out.
    expect(r.polygons[0].length).toBe(1);
  });

  it("finds nothing on blank glass", async () => {
    const slide = fakeSlide(() => [250, 249, 250]);
    const r = await detectTissue(slide, { minAreaUm2: 100 });
    expect(r.polygons.length).toBe(0);
  });

  it("rejects a scanner seam: a long strip far thinner than any core", async () => {
    // 6px wide (6 µm at mpp 1) running the full height — aspect 50.
    const slide = fakeSlide((x, y) => (x >= 200 && x < 206 ? [200, 150, 200] : [250, 249, 250]));
    const r = await detectTissue(slide, { minAreaUm2: 100 });
    expect(r.polygons.length).toBe(0);
    expect(r.rejected).toBe(1);
  });

  it("keeps a needle core, which is legitimately long and thin", async () => {
    // 0.8 mm wide at mpp 1 => 800px, aspect ~3.5. Must survive the filter.
    const slide = fakeSlide((x, y) =>
      inBox(x, y, 100, 10, 80, 280) ? [185, 95, 175] : [250, 249, 250],
    );
    const r = await detectTissue(slide, { minAreaUm2: 100, minWidthUm: 40 });
    expect(r.polygons.length).toBe(1);
    expect(r.rejected).toBe(0);
  });

  it("keeps a thin strip when the aspect limit is raised", async () => {
    const slide = fakeSlide((x, y) => (x >= 200 && x < 206 ? [200, 150, 200] : [250, 249, 250]));
    const r = await detectTissue(slide, { minAreaUm2: 100, maxAspect: 500 });
    expect(r.polygons.length).toBe(1);
    expect(r.rejected).toBe(0);
  });

  it("keeps separate fragments separate, with no chord between them", async () => {
    const slide = fakeSlide((x, y) => {
      const a = inBox(x, y, 20, 120, 60, 60);
      const b = inBox(x, y, 320, 120, 60, 60);
      return a || b ? [180, 90, 170] : [250, 249, 250];
    });
    const r = await detectTissue(slide, { minAreaUm2: 100 });
    expect(r.polygons.length).toBe(2);
    for (const rings of r.polygons) {
      const xs = rings[0].map((p) => p[0]);
      // A chord joining the fragments would span most of the 400px width.
      expect(Math.max(...xs) - Math.min(...xs)).toBeLessThan(120);
    }
  });
});

describe("a featureless overview", () => {
  /**
   * The Python port failed exactly here, and the two have to agree. With one
   * occupied histogram bin there is no split to find, and the levels derived
   * from a span of 1 sit below the score of glass — so without a guard the
   * whole image reads as tissue.
   */
  it("has no tissue in it", () => {
    const w = 64;
    const h = 48;
    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      rgba[i * 4] = 245;
      rgba[i * 4 + 1] = 245;
      rgba[i * 4 + 2] = 245;
      rgba[i * 4 + 3] = 255;
    }
    const scored = scoreOverview({
      rgba, w, h, step: 1,
      level: { level: 0, width: w, height: h, downsample: 1 },
      originX: 0, originY: 0, spanX: w, spanY: h,
    });
    // Nothing can clear a threshold above the top of the range.
    expect(scored.relaxed).toBeGreaterThan(255);
    expect(scored.weakLevel).toBeGreaterThan(255);
  });
});
