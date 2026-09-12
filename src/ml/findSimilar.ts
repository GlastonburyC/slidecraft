import { traceMask } from "./contour";
import type { Ring } from "../annotate/types";
import type { GeneStat } from "./enrichment";
import type { Signature } from "./signatures";
import type { SpatialResult } from "./spatialResult";

/**
 * From one example to every other place that looks like it.
 *
 * Draw round a thing, ask what is in it, and the answer is a ranked list of
 * genes or cell types. That list is already a description of the thing — so
 * turning it into a signature and scoring it over every patch says where else
 * on the slide the same description fits, with no model trained and nothing
 * labelled.
 *
 * What this is NOT: independent evidence. Predicted expression is a function of
 * the same pixels the morphology encoder sees, so a match here means the two
 * regions look alike to a model trained on transcriptomics — a useful and
 * interpretable way of looking alike, but still looking alike. It finds
 * candidates for you to judge, which is what the training loop wants.
 */

/** Weight from separation: +1 for perfectly enriched, -1 for perfectly depleted. */
const weightOf = (stat: GeneStat) => Math.round((stat.auc - 0.5) * 2 * 1e4) / 1e4;

export interface ModuleOptions {
  /** At most this many genes, strongest separation first. */
  top?: number;
  /**
   * Minimum |AUC - 0.5|. A gene that barely separates contributes noise in
   * proportion to its weight, and there are always thousands of them.
   */
  minSeparation?: number;
}

/**
 * Turn an enrichment result into a signature.
 *
 * Both directions are kept. A gene the region is depleted of describes it just
 * as well as one it is full of — "no collagen here" is half of what makes a
 * lymphoid aggregate look like one — and the negative weight is what lets
 * scoreSignature use it that way.
 */
export function moduleFromEnrichment(
  name: string,
  stats: GeneStat[],
  opts: ModuleOptions = {},
): Signature {
  const { top = 40, minSeparation = 0.15 } = opts;
  const genes = [...stats]
    .filter((s) => Math.abs(s.auc - 0.5) >= minSeparation)
    .sort((a, b) => Math.abs(b.auc - 0.5) - Math.abs(a.auc - 0.5))
    .slice(0, top)
    .map((s) => ({ gene: s.gene, weight: weightOf(s) }));
  return { name, genes };
}

/** The lattice a map's patches sit on. */
export interface PatchLattice {
  cols: number;
  rows: number;
  originX: number;
  originY: number;
  /** Spacing in slide pixels, which is the stride rather than the patch side. */
  step: number;
  /** Index into the map's patches for each cell, or -1 where nothing was run. */
  cell: Int32Array;
}

/**
 * Recover the grid the patches were laid on.
 *
 * A map arrives as a flat list because it was written by whatever produced it,
 * but tracing a contour needs a raster. The spacing is the smallest gap between
 * distinct coordinates rather than the patch side: a strided run overlaps, and
 * using the side would collapse several patches into one cell.
 */
export function latticeOf(result: SpatialResult): PatchLattice {
  const xs = [...new Set(result.patches.map((p) => p.x))].sort((a, b) => a - b);
  const ys = [...new Set(result.patches.map((p) => p.y))].sort((a, b) => a - b);
  const gap = (v: number[]) => {
    let g = Infinity;
    for (let i = 1; i < v.length; i++) g = Math.min(g, v[i] - v[i - 1]);
    return Number.isFinite(g) && g > 0 ? g : result.side;
  };
  const step = Math.min(gap(xs), gap(ys));

  const originX = xs[0] ?? 0;
  const originY = ys[0] ?? 0;
  const cols = Math.max(1, Math.round(((xs[xs.length - 1] ?? 0) - originX) / step) + 1);
  const rows = Math.max(1, Math.round(((ys[ys.length - 1] ?? 0) - originY) / step) + 1);

  const cell = new Int32Array(cols * rows).fill(-1);
  result.patches.forEach((p, i) => {
    const cx = Math.round((p.x - originX) / step);
    const cy = Math.round((p.y - originY) / step);
    if (cx >= 0 && cx < cols && cy >= 0 && cy < rows) cell[cy * cols + cx] = i;
  });
  return { cols, rows, originX, originY, step, cell };
}

export interface SimilarOptions {
  /** Keep cells scoring above this. */
  threshold: number;
  /** Discard regions smaller than this, in patches. */
  minPatches?: number;
  /** Douglas-Peucker tolerance, in patches. */
  simplify?: number;
}

export interface SimilarRegion {
  ring: Ring;
  /** Patches enclosed, as a rough size. */
  patches: number;
}

/**
 * Regions of the slide where a field clears a threshold, as polygons.
 *
 * Cells with no patch are left below the threshold rather than interpolated:
 * the map says nothing about them, and inventing a value there would grow a
 * region across glass the encoder never saw.
 */
export function similarRegions(
  values: Float32Array,
  lattice: PatchLattice,
  opts: SimilarOptions,
): SimilarRegion[] {
  const { threshold, minPatches = 4, simplify = 0.75 } = opts;
  const { cols, rows, cell, originX, originY, step } = lattice;

  const raster = new Float32Array(cols * rows);
  let above = 0;
  for (let i = 0; i < cell.length; i++) {
    const patch = cell[i];
    if (patch < 0) { raster[i] = threshold - 1; continue; }
    raster[i] = values[patch];
    if (values[patch] > threshold) above++;
  }
  if (above < minPatches) return [];

  const rings = traceMask(raster, cols, rows, { threshold, simplify, minArea: minPatches });
  return rings.map((ring) => ({
    /*
     * The half-cell shift matters.
     *
     * Marching squares works over cell CENTRES and puts its vertices on the
     * boundaries between them, so a vertex at 2.5 is the edge between cells 2
     * and 3 — which in slide pixels is where cell 3 starts, not where cell 2
     * does. Dropping the half leaves every region offset by half a patch,
     * which at 224px and 0.5 µm/px is 56 µm of tissue in the wrong place.
     */
    ring: ring.map(([x, y]) =>
      [originX + (x + 0.5) * step, originY + (y + 0.5) * step] as [number, number]),
    patches: Math.abs(shoelace(ring)),
  })).filter((r) => r.patches >= minPatches);
}

/** Twice the signed area, in cells. */
function shoelace(ring: Ring): number {
  let sum = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    sum += (ring[j][0] + ring[i][0]) * (ring[j][1] - ring[i][1]);
  }
  return sum / 2;
}

/**
 * A threshold at a percentile of the field.
 *
 * Absolute thresholds do not transfer: a module's score is standardised per
 * slide, so 0.8 means something different on every one. "The top 5% of this
 * slide" is a question with the same meaning everywhere.
 */
export function percentileThreshold(values: Float32Array, percentile: number): number {
  if (!values.length) return 0;
  const sorted = Float32Array.from(values).sort();
  const at = Math.min(sorted.length - 1,
                      Math.max(0, Math.round((percentile / 100) * (sorted.length - 1))));
  return sorted[at];
}
