import { describe, expect, it } from "vitest";
import { detectTissue, readOverviewGrid } from "../ml/tissue";
import { collectSamples, mergeSamples, summarise } from "../ml/tissueLabels";
import { NotEnoughLabels, explain, trainTissueModel } from "../ml/tissueModel";
import { computeFeatures, FEATURE_COUNT } from "../ml/tissueFeatures";
import { makeAnnotation } from "../annotate/store";
import type { Annotation } from "../annotate/types";
import type { SlideMeta, SlideSource } from "../slide/types";

const noise = (x: number, y: number) => {
  let n = (x * 374761393 + y * 668265263) >>> 0;
  n = ((n ^ (n >>> 13)) * 1274126177) >>> 0;
  return ((n >>> 16) & 255) / 255;
};

function slide(
  width: number,
  height: number,
  paint: (x: number, y: number) => [number, number, number],
): SlideSource {
  const ds = [1, 4, 16, 32];
  const meta: SlideMeta = {
    name: "classifier", bytes: 0, vendor: "test", mppX: 1, mppY: 1, objectivePower: 20,
    bounds: null, backgroundColor: null,
    levels: ds.map((d, i) => ({
      level: i, width: Math.round(width / d), height: Math.round(height / d), downsample: d,
    })),
    properties: {},
  };
  return {
    meta,
    async readRegion(x, y, level, w, h) {
      const d = meta.levels[level].downsample;
      const out = new Uint8ClampedArray(w * h * 4);
      for (let j = 0; j < h; j++) {
        for (let i = 0; i < w; i++) {
          const [r, g, b] = paint(Math.round(x + i * d), Math.round(y + j * d));
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

const inBox = (x: number, y: number, b: Box) =>
  x >= b.x && x < b.x + b.w && y >= b.y && y < b.y + b.h;

interface Box { x: number; y: number; w: number; h: number }

const rect = (b: Box): Annotation =>
  makeAnnotation(
    {
      type: "Polygon",
      coordinates: [[
        [b.x, b.y], [b.x + b.w, b.y], [b.x + b.w, b.y + b.h], [b.x, b.y + b.h], [b.x, b.y],
      ]],
    },
    { classId: "c" },
  );

const W = 30000;
const H = 20000;

/** Real tissue: pale on average, but textured at the scale of glands. */
const TISSUE: Box[] = [
  { x: 2000, y: 3000, w: 4000, h: 4000 },
  { x: 8000, y: 3000, w: 3800, h: 4000 },
  { x: 15000, y: 3000, w: 4000, h: 4000 },
  { x: 22000, y: 3000, w: 3800, h: 4000 },
];
/** Artefacts: the same paleness, but smooth — dust, bubbles, a smear. */
const ARTEFACT: Box[] = [
  { x: 5000, y: 13000, w: 1600, h: 1200 },
  { x: 10000, y: 14000, w: 1400, h: 1100 },
  { x: 16500, y: 13000, w: 1600, h: 1300 },
  { x: 22500, y: 14000, w: 1500, h: 1100 },
];

/**
 * A slide whose artefacts are exactly as pale as its faint tissue.
 *
 * This is the case the colour rule cannot win: any threshold that keeps the
 * tissue keeps the smears too, because on brightness and saturation they are
 * the same. The only thing that separates them is that tissue has structure
 * and a smear does not.
 */
const awkwardSlide = () =>
  slide(W, H, (x, y) => {
    const fine = noise(x >> 6, y >> 6);
    if (TISSUE.some((b) => inBox(x, y, b))) {
      // Pale overall, with gland-scale variation running through it.
      return fine > 0.55
        ? [206 + fine * 10, 188 + fine * 10, 206 + fine * 10]
        : [236 + fine * 6, 230 + fine * 6, 236 + fine * 6];
    }
    if (ARTEFACT.some((b) => inBox(x, y, b))) {
      // The mean of the tissue above, with none of its structure.
      return [224, 213, 224];
    }
    return [246 + noise(x >> 5, y >> 5) * 5, 245, 247];
  });

function trainOn(source: SlideSource, grid: Awaited<ReturnType<typeof readOverviewGrid>>) {
  const samples = collectSamples(grid, {
    tissue: TISSUE.map(rect),
    artefact: ARTEFACT.map(rect),
  });
  return { samples, model: trainTissueModel(samples.set, { name: "t", slides: ["s"] }) };
}

/** The same layout, restained — a second slide from a different batch. */
const otherSlide = () =>
  slide(W, H, (x, y) => {
    const fine = noise(x >> 6, y >> 6);
    if (TISSUE.some((b) => inBox(x, y, b))) {
      return fine > 0.55
        ? [198 + fine * 10, 196 + fine * 10, 214 + fine * 10]
        : [230 + fine * 6, 232 + fine * 6, 240 + fine * 6];
    }
    if (ARTEFACT.some((b) => inBox(x, y, b))) return [218, 220, 230];
    return [247 + noise(x >> 5, y >> 5) * 4, 246, 249];
  });

describe("tissue classifier", () => {
  it("refuses to train without both classes", () => {
    const set = { x: new Float32Array(FEATURE_COUNT), y: new Uint8Array([1]), group: new Int32Array([0]), count: 1 };
    expect(() => trainTissueModel(set, { name: "t", slides: [] })).toThrow(NotEnoughLabels);
  });

  it("samples cells from labelled regions, keeping regions separable", async () => {
    const source = awkwardSlide();
    const grid = await readOverviewGrid(source);
    const { samples } = trainOn(source, grid);

    expect(samples.regions.tissue).toBe(4);
    expect(samples.regions.artefact).toBe(4);
    expect(samples.set.count).toBeGreaterThan(50);
    // Both classes present, and every sample tagged with the region it came from.
    const labels = new Set(samples.set.y);
    expect([...labels].sort()).toEqual([0, 1]);
    // The eight labelled regions, plus the background sampled automatically.
    expect(new Set(samples.set.group).size).toBeGreaterThan(8);
  }, 30000);

  /**
   * The claim the whole feature set rests on: what tells a smear from faint
   * tissue is texture, so a model that cannot see texture cannot do this.
   */
  it("leans on texture, not just colour", async () => {
    const source = awkwardSlide();
    const { model } = trainOn(source, await readOverviewGrid(source));
    // The features that describe local variation rather than local colour.
    const texture = new Set(["detailFine", "detailCoarse", "saturationDetail", "localRange", "gradient"]);
    const ranked = explain(model);
    expect(texture.has(ranked[0].name)).toBe(true);

    // And it must outweigh plain colour, or the model is only doing what the
    // threshold already did.
    const strongestTexture = Math.abs(ranked.find((f) => texture.has(f.name))!.weight);
    const strongestColour = Math.abs(ranked.find((f) => !texture.has(f.name))!.weight);
    expect(strongestTexture).toBeGreaterThan(strongestColour);
  });

  it("separates held-out regions it was not fitted on", async () => {
    const source = awkwardSlide();
    const { model } = trainOn(source, await readOverviewGrid(source));
    expect(model.metrics).not.toBe(null);
    expect(model.metrics!.heldOut).toBeGreaterThan(0);
    expect(model.metrics!.f1).toBeGreaterThan(0.8);
  }, 30000);

  /**
   * End to end, and the actual complaint: the colour rule detects the smears
   * as tissue, and after training on a handful of them the detector stops.
   */
  it("drops artefacts the colour rule detects", async () => {
    const source = awkwardSlide();
    const grid = await readOverviewGrid(source);

    const before = await detectTissue(source, { grid, minAreaUm2: 1e6 });
    const { model } = trainOn(source, grid);
    const after = await detectTissue(source, { grid, minAreaUm2: 1e6, model });

    const covers = (r: Awaited<ReturnType<typeof detectTissue>>, b: Box) => {
      const cx = b.x + b.w / 2;
      const cy = b.y + b.h / 2;
      return r.polygons.some((rings) => {
        let inside = false;
        const ring = rings[0];
        for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
          const [xi, yi] = ring[i];
          const [xj, yj] = ring[j];
          if (yi > cy !== yj > cy && cx < ((xj - xi) * (cy - yi)) / (yj - yi) + xi) inside = !inside;
        }
        return inside;
      });
    };

    // The colour rule cannot tell them apart, so it takes the smears.
    expect(ARTEFACT.filter((b) => covers(before, b)).length).toBeGreaterThan(0);
    // The model can, and does not.
    expect(ARTEFACT.filter((b) => covers(after, b)).length).toBe(0);
    // While keeping the tissue it was shown.
    expect(TISSUE.every((b) => covers(after, b))).toBe(true);
    expect(after.modelId).toBe(model.id);
  }, 30000);

  /**
   * A model fitted on one slide learns that slide's stain as much as it learns
   * tissue, so labels from several slides have to fit one model. The samples
   * are kept per slide and merged, which is also what lets a slide whose
   * labels were wrong be dropped without starting again.
   */
  it("trains one model from labels on several slides", async () => {
    const a = awkwardSlide();
    const b = otherSlide();
    const parts = [
      { slide: "a", report: collectSamples(await readOverviewGrid(a), { tissue: TISSUE.map(rect), artefact: ARTEFACT.map(rect) }) },
      { slide: "b", report: collectSamples(await readOverviewGrid(b), { tissue: TISSUE.map(rect), artefact: ARTEFACT.map(rect) }) },
    ];

    const totals = summarise(parts);
    expect(totals.slides).toBe(2);
    expect(totals.tissue).toBe(8);
    expect(totals.artefact).toBe(8);

    const merged = mergeSamples(parts);
    expect(merged.count).toBe(parts[0].report.set.count + parts[1].report.set.count);
    // Region ids must not collide, or one slide leaks into the other's split.
    const aGroups = new Set<number>();
    for (let i = 0; i < parts[0].report.set.count; i++) aGroups.add(merged.group[i]);
    for (let i = parts[0].report.set.count; i < merged.count; i++) {
      expect(aGroups.has(merged.group[i])).toBe(false);
    }

    const model = trainTissueModel(merged, { name: "both", slides: ["a", "b"] });
    expect(model.slides).toEqual(["a", "b"]);

    // And it must work on both slides, not just the one it saw most of.
    for (const source of [a, b]) {
      const r = await detectTissue(source, { minAreaUm2: 1e6, model });
      const covered = (bx: Box) => {
        const cx = bx.x + bx.w / 2;
        const cy = bx.y + bx.h / 2;
        return r.polygons.some((rings) => {
          let inside = false;
          const ring = rings[0];
          for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
            const [xi, yi] = ring[i];
            const [xj, yj] = ring[j];
            if (yi > cy !== yj > cy && cx < ((xj - xi) * (cy - yi)) / (yj - yi) + xi) inside = !inside;
          }
          return inside;
        });
      };
      expect(TISSUE.every(covered)).toBe(true);
      expect(ARTEFACT.filter(covered).length).toBe(0);
    }
  }, 60000);

  it("computes one feature vector per cell", async () => {
    const source = awkwardSlide();
    const grid = await readOverviewGrid(source);
    const f = computeFeatures({ rgba: grid.rgba, w: grid.w, h: grid.h });
    expect(f.length).toBe(grid.w * grid.h * FEATURE_COUNT);
    expect(f.every(Number.isFinite)).toBe(true);
  });
});
