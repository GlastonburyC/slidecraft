import { describe, expect, it } from "vitest";
import { detectTissue } from "../ml/tissue";
import type { SlideMeta, SlideSource } from "../slide/types";
import type { Ring } from "../annotate/types";

/**
 * A faithful fake slide.
 *
 * Unlike the painterly fake used for the thresholding tests, this one honours
 * the region arguments exactly as OpenSlide does — `x`/`y` in level-0 pixels,
 * `width`/`height` in the requested level's pixels. That distinction is the
 * whole point: a fake that ignores the region and simply scales the picture to
 * fit will happily pass while the real reader is cropping, which is exactly how
 * a stretched-overview bug reaches a user.
 */
function slideWithLevels(
  width: number,
  height: number,
  downsamples: number[],
  paint: (x: number, y: number) => [number, number, number],
): SlideSource {
  const meta: SlideMeta = {
    name: "faithful",
    bytes: 0,
    vendor: "test",
    mppX: 1,
    mppY: 1,
    objectivePower: 20,
    bounds: null,
    backgroundColor: null,
    levels: downsamples.map((d, i) => ({
      level: i,
      width: Math.round(width / d),
      height: Math.round(height / d),
      downsample: d,
    })),
    properties: {},
  };
  return {
    meta,
    async readRegion(x, y, level, w, h) {
      const ds = meta.levels[level].downsample;
      const out = new Uint8ClampedArray(w * h * 4);
      for (let j = 0; j < h; j++) {
        for (let i = 0; i < w; i++) {
          const [r, g, b] = paint(Math.round(x + i * ds), Math.round(y + j * ds));
          const o = (j * w + i) * 4;
          out[o] = r; out[o + 1] = g; out[o + 2] = b; out[o + 3] = 255;
        }
      }
      return out;
    },
    async bestLevelForDownsample() { return meta.levels.length - 1; },
    async close() { /* nothing */ },
  };
}

const box = (x: number, y: number, bx: number, by: number, bw: number, bh: number) =>
  x >= bx && x < bx + bw && y >= by && y < by + bh;

/**
 * Deterministic value noise, so a synthetic slide has a histogram with real
 * spread rather than a handful of spikes. Thresholding on three clean deltas
 * behaves nothing like thresholding on a scan, and a fake without noise lets a
 * threshold that sits right against the glass look perfectly healthy.
 */
const noise = (x: number, y: number) => {
  let n = (x * 374761393 + y * 668265263) >>> 0;
  n = ((n ^ (n >>> 13)) * 1274126177) >>> 0;
  return ((n >>> 16) & 255) / 255;
};

const glass = (n: number): [number, number, number] => [246 + n * 5, 245 + n * 5, 247 + n * 4];

function bboxOf(polygons: number[][][][]): [number, number, number, number] {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const rings of polygons) {
    for (const [x, y] of rings[0]) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  return [minX, minY, maxX, maxY];
}

describe("overview geometry", () => {
  it("finds tissue on the right of a wide slide whose overview exceeds the cap", async () => {
    // Coarsest level is 3981x792 — the real shape of 0024_C1_L1-3.svs, and wider
    // than the 2048 working cap. Tissue sits in the right-hand third, which a
    // cropping reader never even looks at.
    const W = 127412;
    const H = 25374;
    const slide = slideWithLevels(W, H, [1, 4, 16, 32], (x, y) =>
      box(x, y, 90000, 8000, 20000, 9000) ? [150, 70, 150] : [250, 249, 250],
    );
    const r = await detectTissue(slide, { minAreaUm2: 1e6 });
    expect(r.polygons.length).toBe(1);

    const [minX, minY, maxX, maxY] = bboxOf(r.polygons as unknown as number[][][][]);
    // Within a few overview cells of the painted box.
    expect(minX).toBeGreaterThan(85000);
    expect(maxX).toBeLessThan(115000);
    expect(minY).toBeGreaterThan(4000);
    expect(maxY).toBeLessThan(21000);
  });

  it("does not stretch: two fragments keep their separation and their places", async () => {
    const W = 120000;
    const H = 20000;
    const slide = slideWithLevels(W, H, [1, 4, 16, 32], (x, y) =>
      box(x, y, 8000, 6000, 9000, 8000) || box(x, y, 100000, 6000, 9000, 8000)
        ? [150, 70, 150]
        : [250, 249, 250],
    );
    const r = await detectTissue(slide, { minAreaUm2: 1e6 });
    expect(r.polygons.length).toBe(2);

    const centres = r.polygons
      .map((rings) => {
        const xs = rings[0].map((p) => p[0]);
        return (Math.min(...xs) + Math.max(...xs)) / 2;
      })
      .sort((a, b) => a - b);
    expect(centres[0]).toBeGreaterThan(6000);
    expect(centres[0]).toBeLessThan(19000);
    expect(centres[1]).toBeGreaterThan(98000);
    expect(centres[1]).toBeLessThan(111000);
  });

  it("covers the full height of a tall slide", async () => {
    const W = 20000;
    const H = 120000;
    const slide = slideWithLevels(W, H, [1, 4, 16, 32], (x, y) =>
      box(x, y, 6000, 100000, 8000, 9000) ? [150, 70, 150] : [250, 249, 250],
    );
    const r = await detectTissue(slide, { minAreaUm2: 1e6 });
    expect(r.polygons.length).toBe(1);
    const ys = r.polygons[0][0].map((p) => p[1]);
    expect(Math.min(...ys)).toBeGreaterThan(96000);
    expect(Math.max(...ys)).toBeLessThan(112000);
  });

  /**
   * Real coarsest-level shapes measured across the slide sets on hand. Every
   * Exeter slide has an overview wider than the 2048 working cap, so all of
   * them were being cropped and stretched — one slide's worth of testing would
   * not have caught how systematic that was.
   */
  it.each([
    ["Exeter 0015", 87596, 19102, [1, 4, 16, 32]],
    ["Exeter 0024", 127412, 25374, [1, 4, 16, 32]],
    ["Exeter 0020", 119424, 28832, [1, 4, 16, 32]],
    ["Exeter 0023", 107552, 32800, [1, 4, 16, 32]],
    ["shallow NDPI", 59520, 61952, [1, 2, 4, 8, 16, 32]],
    ["deep NDPI", 57600, 57600, [1, 2, 4, 8, 16, 32, 64, 128, 256]],
    ["tiny overview", 20000, 20000, [1, 4, 16, 64, 256]],
  ])(
    "%s: detections sit on tissue, not on glass",
    async (_name, W, H, downsamples) => {
      // Two fragments, one deliberately in the far right third — the part a
      // cropping reader never sees.
      const a = { x: Math.round(W * 0.08), y: Math.round(H * 0.3), w: Math.round(W * 0.12), h: Math.round(H * 0.35) };
      const b = { x: Math.round(W * 0.72), y: Math.round(H * 0.3), w: Math.round(W * 0.12), h: Math.round(H * 0.35) };
      const slide = slideWithLevels(W, H, downsamples, (x, y) =>
        box(x, y, a.x, a.y, a.w, a.h) || box(x, y, b.x, b.y, b.w, b.h)
          ? [150, 70, 150]
          : [250, 249, 250],
      );

      const r = await detectTissue(slide, { minAreaUm2: 1e6 });
      expect(r.polygons.length).toBe(2);

      // Each painted fragment must be matched by a detection centred on it.
      // Centres are the right measure here: a stretched overview displaces them
      // by a large fraction of the slide, while ordinary boundary wobble moves
      // them barely at all.
      const centres = r.polygons
        .map((rings) => {
          const xs = rings[0].map((p) => p[0]);
          const ys = rings[0].map((p) => p[1]);
          return [
            (Math.min(...xs) + Math.max(...xs)) / 2,
            (Math.min(...ys) + Math.max(...ys)) / 2,
          ];
        })
        .sort((p, q) => p[0] - q[0]);

      for (const [i, want] of [a, b].entries()) {
        const wantX = want.x + want.w / 2;
        const wantY = want.y + want.h / 2;
        expect(Math.abs(centres[i][0] - wantX)).toBeLessThan(W * 0.03);
        expect(Math.abs(centres[i][1] - wantY)).toBeLessThan(H * 0.05);
      }
    },
  );


  /**
   * Two fragments with clear glass between them must be two objects.
   *
   * They are 256 slide pixels apart here — eight cells of this slide's
   * overview, comfortably past `bindRadius`, which is the distance below which
   * two pieces of tissue are deliberately treated as one fragment. Merged
   * fragments are worse than a missed one, because the ROI, its patch grid and
   * every prediction downstream inherit a boundary that runs through blank
   * glass.
   */
  it("keeps fragments separate across clear glass", async () => {
    const W = 60000;
    const H = 40000;
    const left = { x: 10000, y: 12000, w: 9872, h: 16000 };
    const right = { x: 20128, y: 12000, w: 9872, h: 16000 };
    const slide = slideWithLevels(W, H, [1, 4, 16, 32], (x, y) => {
      const n = noise(x >> 5, y >> 5);
      return box(x, y, left.x, left.y, left.w, left.h) || box(x, y, right.x, right.y, right.w, right.h)
        ? [150 + n * 20, 70 + n * 20, 150 + n * 20]
        : glass(n);
    });

    const r = await detectTissue(slide, { minAreaUm2: 1e6 });
    expect(r.polygons.length).toBe(2);

    // Neither may cross the gap: each stays on its own side of x = 20000.
    const spans = r.polygons
      .map((rings) => {
        const xs = rings[0].map((p) => p[0]);
        return [Math.min(...xs), Math.max(...xs)];
      })
      .sort((a, b) => a[0] - b[0]);
    expect(spans[0][1]).toBeLessThanOrEqual(20064);
    expect(spans[1][0]).toBeGreaterThanOrEqual(20064);

    // And nothing may claim the glass: coverage stays at the painted area.
    const painted = (left.w * left.h + right.w * right.h) / (W * H);
    expect(r.coverage).toBeLessThan(painted * 1.05);
  });

  /**
   * One speckled section is one object, not forty.
   *
   * Real tissue does not threshold into a solid blob: stroma and fat fall below
   * the confident threshold while the nuclei around them clear it, so a single
   * section arrives as a scatter of islands. Identifying fragments straight
   * from that shatters the section into dozens of objects — which is the same
   * bug as welding two sections together, in the other direction, and just as
   * useless to annotate against.
   */
  it("does not shatter one speckled section into many objects", async () => {
    const W = 40000;
    const H = 30000;
    const section = { x: 12000, y: 9000, w: 16000, h: 12000 };
    const slide = slideWithLevels(W, H, [1, 4, 16, 32], (x, y) => {
      const n = noise(x >> 5, y >> 5);
      if (!box(x, y, section.x, section.y, section.w, section.h)) return glass(n);
      // Two thirds of the section is pale connective tissue and a third is
      // strongly stained, in patches a few overview cells across — the scale
      // stroma and glands actually vary at, rather than per-pixel salt.
      const patch = noise(x >> 7, y >> 7);
      return patch > 0.66
        ? [150 + n * 20, 70 + n * 20, 150 + n * 20]
        : [236 + n * 6, 230 + n * 6, 236 + n * 6];
    });

    const r = await detectTissue(slide, { minAreaUm2: 1e6 });

    // One object covering the section is the claim. A stray speck at the edge
    // is tolerable; forty pieces where a pathologist sees one section is not,
    // and the count alone would not tell those two apart.
    const areaOfBox = (rings: Ring[]) => {
      const xs = rings[0].map((p) => p[0]);
      const ys = rings[0].map((p) => p[1]);
      return (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
    };
    const biggest = Math.max(...r.polygons.map(areaOfBox));
    expect(r.polygons.length).toBeLessThanOrEqual(3);
    expect(biggest / (section.w * section.h)).toBeGreaterThan(0.8);

    // And it must cover the section, not just the stained patches in it.
    const painted = (section.w * section.h) / (W * H);
    expect(r.coverage).toBeGreaterThan(painted * 0.7);
  });

  /**
   * Two fragments whose pale halos touch are still two fragments.
   *
   * Each section here has a smear of faded tissue around it, and the smears
   * meet in the gap. Growing first and labelling the result would make that a
   * single connected region — one object with a waist, spanning both sections
   * and the glass between them. Because each fragment is identified at the
   * confident threshold and then grown into its own territory, they advance at
   * the same rate and meet at a boundary instead, which is where the edge
   * between two touching sections actually belongs.
   */
  it("keeps fragments separate when their faded halos meet", async () => {
    const W = 60000;
    const H = 40000;
    const a = { x: 10000, y: 14000, w: 8000, h: 12000 };
    const b = { x: 20000, y: 14000, w: 8000, h: 12000 };
    const near = (x: number, y: number, r: { x: number; y: number; w: number; h: number }) => {
      const dx = Math.max(r.x - x, x - (r.x + r.w), 0);
      const dy = Math.max(r.y - y, y - (r.y + r.h), 0);
      return Math.hypot(dx, dy) < 1200;
    };
    const slide = slideWithLevels(W, H, [1, 4, 16, 32], (x, y) => {
      const n = noise(x >> 5, y >> 5);
      if (box(x, y, a.x, a.y, a.w, a.h) || box(x, y, b.x, b.y, b.w, b.h))
        return [150 + n * 20, 70 + n * 20, 150 + n * 20];
      if (near(x, y, a) || near(x, y, b)) return [240 + n * 6, 234 + n * 6, 240 + n * 6];
      return glass(n);
    });

    const r = await detectTissue(slide, { minAreaUm2: 1e6 });
    expect(r.polygons.length).toBe(2);

    // The halos are genuinely reached: coverage is well past the bare cores.
    expect(r.coverage).toBeGreaterThan(0.1);

    // Each object must contain exactly one of the two cores.
    const holds = (rings: Ring[], cx: number, cy: number) => {
      let inside = false;
      const ring = rings[0];
      for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const [xi, yi] = ring[i];
        const [xj, yj] = ring[j];
        if (yi > cy !== yj > cy && cx < ((xj - xi) * (cy - yi)) / (yj - yi) + xi) inside = !inside;
      }
      return inside;
    };
    for (const centre of [a, b]) {
      const cx = centre.x + centre.w / 2;
      const cy = centre.y + centre.h / 2;
      expect(r.polygons.filter((rings) => holds(rings, cx, cy)).length).toBe(1);
    }
  });

  /**
   * A fragment whose staining fades away must come out whole.
   *
   * The threshold is pinned here so the test is about the growth rather than
   * about where Otsu happens to land: at that threshold only the stained core
   * qualifies, and the faded two-thirds of the fragment is reached only by
   * growing into it. The isolated faded fragment is the other half of the
   * claim — the same growth must not invent tissue out of pale noise that
   * touches nothing.
   */
  it("captures a fragment whose staining fades, but not faded glass alone", async () => {
    const W = 60000;
    const H = 40000;
    const core = { x: 10000, y: 14000, w: 8000, h: 12000 };
    const faded = { x: 18000, y: 14000, w: 14000, h: 12000 };
    const alone = { x: 44000, y: 14000, w: 10000, h: 12000 };
    const slide = slideWithLevels(W, H, [1, 4, 16, 32], (x, y) => {
      const n = noise(x >> 5, y >> 5);
      if (box(x, y, core.x, core.y, core.w, core.h)) return [150 + n * 20, 70 + n * 20, 150 + n * 20];
      if (box(x, y, faded.x, faded.y, faded.w, faded.h) || box(x, y, alone.x, alone.y, alone.w, alone.h))
        return [236 + n * 6, 230 + n * 6, 236 + n * 6];
      return glass(n);
    });

    const pinned = { minAreaUm2: 1e6, saturation: 100 / 255 };
    const grown = await detectTissue(slide, pinned);
    // weakBias 1 puts the growth threshold at the strict one, disabling it.
    const strictOnly = await detectTissue(slide, { ...pinned, weakBias: 1 });

    // The faded continuation is reached only by growing.
    const spanOf = (r: Awaited<ReturnType<typeof detectTissue>>) => {
      const xs = r.polygons.flatMap((rings) => rings[0].map((p) => p[0]));
      return Math.max(...xs);
    };
    expect(spanOf(strictOnly)).toBeLessThan(19000);
    expect(spanOf(grown)).toBeGreaterThan(31000);

    // The fragment that fades is one object, and the one standing alone in the
    // same pale ink is not an object at all.
    expect(grown.polygons.length).toBe(1);
  });

  /**
   * The renderer bridges a hole to its outer ring with a cut. If a polygon is
   * self-intersecting, or a hole is attached to a polygon it is not inside,
   * that bridge is drawn across the slide as a long thin sliver — which is what
   * "lines between the tissue masks" looks like on screen.
   */
  it("emits polygons a tessellator can handle: simple rings, holes inside them", async () => {
    const W = 60000;
    const H = 40000;
    // Two ring-shaped fragments, each with a lumen, far apart. Assembling holes
    // by a single-vertex test is most likely to mis-assign exactly here.
    const donut = (x: number, y: number, cx: number, cy: number) => {
      const d = Math.hypot(x - cx, y - cy);
      return d < 9000 && d > 3500;
    };
    const slide = slideWithLevels(W, H, [1, 4, 16, 32], (x, y) =>
      donut(x, y, 14000, 20000) || donut(x, y, 46000, 20000)
        ? [150, 70, 150]
        : [250, 249, 250],
    );

    const r = await detectTissue(slide, { minAreaUm2: 1e6 });
    expect(r.polygons.length).toBe(2);

    for (const rings of r.polygons) {
      const [outer, ...holes] = rings;
      const ox = outer.map((p) => p[0]);
      const oy = outer.map((p) => p[1]);
      const span = Math.max(Math.max(...ox) - Math.min(...ox), Math.max(...oy) - Math.min(...oy));
      // A bridge to a distant polygon would blow the extent far past one donut.
      expect(span).toBeLessThan(24000);

      // Every hole vertex must lie within its own outer ring's bounding box.
      for (const hole of holes) {
        for (const [hx, hy] of hole) {
          expect(hx).toBeGreaterThanOrEqual(Math.min(...ox) - 1);
          expect(hx).toBeLessThanOrEqual(Math.max(...ox) + 1);
          expect(hy).toBeGreaterThanOrEqual(Math.min(...oy) - 1);
          expect(hy).toBeLessThanOrEqual(Math.max(...oy) + 1);
        }
      }
    }
  });

  it("honours a MIRAX scan-region offset", async () => {
    const W = 100000;
    const H = 200000;
    const slide = slideWithLevels(W, H, [1, 4, 16, 32, 64], (x, y) =>
      box(x, y, 20000, 120000, 20000, 20000) ? [150, 70, 150] : [250, 249, 250],
    );
    slide.meta.bounds = { x: 0, y: 100000, width: 60000, height: 60000 };
    const r = await detectTissue(slide, { minAreaUm2: 1e6 });
    expect(r.polygons.length).toBe(1);
    const ys = r.polygons[0][0].map((p) => p[1]);
    // Must land near y=120000, not near y=20000 as it would if the offset were dropped.
    expect(Math.min(...ys)).toBeGreaterThan(112000);
    expect(Math.max(...ys)).toBeLessThan(148000);
  });
});
