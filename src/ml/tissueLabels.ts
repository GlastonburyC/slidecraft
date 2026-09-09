import { containsPoint, isAreaGeometry, type Annotation } from "../annotate/types";
import { cellToSlide, scoreOverview, type OverviewGrid } from "./tissue";
import { computeFeatures, FEATURE_COUNT } from "./tissueFeatures";
import type { TrainingSet } from "./tissueModel";

/**
 * Turn labelled regions into training cells.
 *
 * The labels are ordinary annotations — whatever is in the tissue class counts
 * as tissue, whatever is in the artefact class counts as not. That matters more
 * than it sounds: correcting a detection is already the thing you would do
 * anyway, so the labels come from the work rather than from a separate chore,
 * and the fastest way to teach the model is to reclassify the blobs it got
 * wrong instead of deleting them.
 *
 * Sampling is on the detector's own grid, so a training cell and the cell the
 * model later scores are the same cell.
 */

export interface LabelSources {
  tissue: Annotation[];
  artefact: Annotation[];
}

/**
 * Cells sampled from confident glass, added to the artefact class.
 *
 * Nobody should have to paint the blank slide. It is unambiguous, it is most of
 * every slide, and without it the model is only ever shown tissue against the
 * few smears that were marked — so it never learns what plain background looks
 * like and will happily call the whole slide tissue. These come from exactly
 * the cells the colour rule would have called background, well below even the
 * level it grows into, so they are the safe part of its judgement rather than
 * the part being corrected.
 */
const BACKGROUND_SAMPLES = 3000;
/** Background is split into this many synthetic regions, for the held-out split. */
const BACKGROUND_REGIONS = 8;

export interface SampleReport {
  set: TrainingSet;
  regions: { tissue: number; artefact: number };
  /** Regions that covered no grid cell at all — too small to teach anything. */
  skipped: number;
}

/**
 * At most this many cells from any one region.
 *
 * A whole tissue section can be tens of thousands of cells while a marked
 * artefact is twenty, and letting one section supply most of the training set
 * makes the model a description of that section rather than of tissue. Regions
 * are subsampled evenly instead, so each one gets a comparable voice.
 */
const MAX_PER_REGION = 4000;

export function collectSamples(
  grid: OverviewGrid,
  sources: LabelSources,
  opts: { background?: boolean } = {},
): SampleReport {
  const features = computeFeatures({ rgba: grid.rgba, w: grid.w, h: grid.h });

  const xs: number[] = [];
  const ys: number[] = [];
  const groups: number[] = [];
  let group = 0;
  let skipped = 0;
  const counted = { tissue: 0, artefact: 0 };

  const take = (regions: Annotation[], label: 0 | 1, key: "tissue" | "artefact") => {
    for (const a of regions) {
      if (!isAreaGeometry(a.geometry)) continue;
      const [minX, minY, maxX, maxY] = a.bbox;

      // Walk only the grid cells the region's bounding box can reach.
      const gx0 = Math.max(0, Math.floor(((minX - grid.originX) / grid.spanX) * grid.w));
      const gx1 = Math.min(grid.w - 1, Math.ceil(((maxX - grid.originX) / grid.spanX) * grid.w));
      const gy0 = Math.max(0, Math.floor(((minY - grid.originY) / grid.spanY) * grid.h));
      const gy1 = Math.min(grid.h - 1, Math.ceil(((maxY - grid.originY) / grid.spanY) * grid.h));

      const inside: number[] = [];
      for (let gy = gy0; gy <= gy1; gy++) {
        for (let gx = gx0; gx <= gx1; gx++) {
          const [sx, sy] = cellToSlide(grid, gx, gy);
          if (containsPoint(a.geometry, sx, sy)) inside.push(gy * grid.w + gx);
        }
      }
      if (!inside.length) { skipped++; continue; }

      const stride = Math.max(1, Math.ceil(inside.length / MAX_PER_REGION));
      for (let k = 0; k < inside.length; k += stride) {
        xs.push(inside[k]);
        ys.push(label);
        groups.push(group);
      }
      group++;
      counted[key]++;
    }
  };

  take(sources.tissue, 1, "tissue");
  take(sources.artefact, 0, "artefact");

  if (opts.background !== false) {
    const { score, glass, weakLevel } = scoreOverview(grid);
    // Comfortably below the level the detector would grow into, so a faint
    // rim of real tissue is never handed over as a negative.
    const ceiling = Math.max(glass, Math.round((glass + weakLevel) / 2));
    const claimed = new Set(xs);
    const candidates: number[] = [];
    for (let i = 0; i < grid.w * grid.h; i++) {
      if (score[i] <= ceiling && !claimed.has(i)) candidates.push(i);
    }
    const stride = Math.max(1, Math.ceil(candidates.length / BACKGROUND_SAMPLES));
    let taken = 0;
    for (let k = 0; k < candidates.length; k += stride) {
      xs.push(candidates[k]);
      ys.push(0);
      // Spread across several groups so validation can hold some of it out.
      groups.push(group + (taken % BACKGROUND_REGIONS));
      taken++;
    }
    if (taken) group += BACKGROUND_REGIONS;
  }

  const count = xs.length;
  const x = new Float32Array(count * FEATURE_COUNT);
  const y = new Uint8Array(count);
  const g = new Int32Array(count);
  for (let i = 0; i < count; i++) {
    x.set(features.subarray(xs[i] * FEATURE_COUNT, (xs[i] + 1) * FEATURE_COUNT), i * FEATURE_COUNT);
    y[i] = ys[i];
    g[i] = groups[i];
  }

  return { set: { x, y, group: g, count }, regions: counted, skipped };
}

/**
 * Labels gathered from one slide, kept so several slides can train one model.
 *
 * A model fitted on a single slide learns that slide's stain and scanner as
 * much as it learns tissue, and the first slide from another batch undoes it.
 * Collecting labels per slide and fitting across all of them is what makes the
 * model transfer — and keeping them separate means a slide whose labels turn
 * out to be wrong can be dropped without starting over.
 */
export interface SlideSamples {
  slide: string;
  report: SampleReport;
}

/**
 * Concatenate several slides' samples into one training set.
 *
 * Region ids are renumbered so that two slides cannot share one, which would
 * otherwise put cells from different slides in the same validation group and
 * let one leak into the other's held-out score.
 */
export function mergeSamples(parts: SlideSamples[]): TrainingSet {
  const count = parts.reduce((n, p) => n + p.report.set.count, 0);
  const x = new Float32Array(count * FEATURE_COUNT);
  const y = new Uint8Array(count);
  const group = new Int32Array(count);

  let at = 0;
  let groupBase = 0;
  for (const part of parts) {
    const { set } = part.report;
    x.set(set.x.subarray(0, set.count * FEATURE_COUNT), at * FEATURE_COUNT);
    let highest = -1;
    for (let i = 0; i < set.count; i++) {
      y[at + i] = set.y[i];
      group[at + i] = groupBase + set.group[i];
      if (set.group[i] > highest) highest = set.group[i];
    }
    at += set.count;
    groupBase += highest + 1;
  }
  return { x, y, group, count };
}

/** Totals across every slide contributing labels, for the panel to report. */
export function summarise(parts: SlideSamples[]): {
  slides: number; tissue: number; artefact: number; cells: number;
} {
  return {
    slides: parts.length,
    tissue: parts.reduce((n, p) => n + p.report.regions.tissue, 0),
    artefact: parts.reduce((n, p) => n + p.report.regions.artefact, 0),
    cells: parts.reduce((n, p) => n + p.report.set.count, 0),
  };
}
