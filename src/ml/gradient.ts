import type { Annotation } from "../annotate/types";
import { AXIS_CLASS_ID } from "../annotate/types";
import type { Signature } from "./signatures";
import { valueAt, type SpatialResult } from "./spatialResult";
import { benjaminiHochberg } from "./enrichment";

/**
 * What changes along a direction you drew.
 *
 * Enrichment asks whether a region differs from the rest, which is the right
 * question for a thing with a boundary — a lymphoid aggregate, a tumour nest.
 * A lot of what matters in mucosa has no boundary: expression varies *along*
 * an axis, crypt base to luminal surface, mucosa to muscularis. Splitting that
 * into inside and outside throws away the ordering, which is the signal.
 *
 * So a patch is projected onto the arrow, giving a position along it, and each
 * gene is rank-correlated with that position. The answer is signed: positive
 * rises toward the arrowhead.
 */

export class NoPatchesAlongAxis extends Error {}

export interface AxisPatch {
  index: number;
  /** Distance along the axis, 0 at the tail and 1 at the head. */
  t: number;
}

export interface GradientStat {
  /** The gene, or the cell type. */
  name: string;
  /** Spearman's rho against position along the axis. Positive rises. */
  rho: number;
  /** Mean in the first third of the axis, and in the last. */
  meanStart: number;
  meanEnd: number;
  p: number;
  q: number;
}

export interface GradientResult {
  items: GradientStat[];
  /** Patches that fell in the corridor. */
  used: number;
  /** Axis length in level-0 pixels, and in microns when the slide says. */
  lengthPx: number;
  lengthUm: number | null;
  kind: "gene" | "signature";
}

/** The axis annotations on the slide, most recent last. */
export function axesIn(items: Iterable<Annotation>): Annotation[] {
  return [...items].filter(
    (a) => a.classId === AXIS_CLASS_ID && a.geometry.type === "LineString"
      && a.geometry.coordinates.length >= 2,
  );
}

/**
 * Patches lying along an axis, with how far along each one is.
 *
 * `width` is the half-width of the corridor in level-0 pixels: a patch further
 * than this from the line is not on the axis and is dropped. Without it the
 * projection happily accepts the whole slide, and a gradient gets computed
 * across tissue the arrow never pointed at.
 *
 * Patches before the tail or past the head are dropped for the same reason —
 * the arrow states where the question starts and stops.
 */
export function patchesAlong(
  result: SpatialResult,
  axis: Annotation,
  width: number,
): AxisPatch[] {
  if (axis.geometry.type !== "LineString") return [];
  const pts = axis.geometry.coordinates;
  const [ax, ay] = pts[0];
  const [bx, by] = pts[pts.length - 1];
  const dx = bx - ax;
  const dy = by - ay;
  const len2 = dx * dx + dy * dy;
  if (len2 === 0) return [];

  const half = result.side / 2;
  const out: AxisPatch[] = [];
  result.patches.forEach((p, index) => {
    const px = p.x + half - ax;
    const py = p.y + half - ay;
    // Projection onto the axis, as a fraction of its length.
    const t = (px * dx + py * dy) / len2;
    if (t < 0 || t > 1) return;
    // Perpendicular distance, from the rejection of the projection.
    const perp = Math.abs(px * dy - py * dx) / Math.sqrt(len2);
    if (perp > width) return;
    out.push({ index, t });
  });
  return out;
}

/** Ranks with ties averaged. */
function ranksOf(values: Float32Array | number[]): Float64Array {
  const n = values.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => values[a] - values[b]);
  const ranks = new Float64Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && values[order[j + 1]] === values[order[i]]) j++;
    const avg = (i + j + 2) / 2;
    for (let k = i; k <= j; k++) ranks[order[k]] = avg;
    i = j + 1;
  }
  return ranks;
}

/** Normal tail, via the same erfc approximation the enrichment test uses. */
function normalSf(z: number): number {
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.5 * x);
  const tau =
    t *
    Math.exp(
      -x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
        t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
          t * (-0.82215223 + t * 0.17087277)))))))),
    );
  const erfc = z >= 0 ? tau : 2 - tau;
  return erfc / 2;
}

/** Pearson correlation of two rank vectors, which is Spearman's rho. */
function spearman(aRanks: Float64Array, bRanks: Float64Array): number {
  const n = aRanks.length;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) { ma += aRanks[i]; mb += bRanks[i]; }
  ma /= n; mb /= n;
  let num = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < n; i++) {
    const da = aRanks[i] - ma;
    const db = bRanks[i] - mb;
    num += da * db; va += da * da; vb += db * db;
  }
  // A column with no variation correlates with nothing.
  return va > 0 && vb > 0 ? num / Math.sqrt(va * vb) : 0;
}

/**
 * Correlate each column against position, one at a time.
 *
 * `column(i)` fills and returns a reusable buffer rather than a new array per
 * item: a whole transcriptome is 19,338 columns, and materialising them all
 * before correlating any costs about 90 MB for no reason — every one is read
 * once and never looked at again.
 */
function assemble(
  names: string[],
  column: (i: number) => Float32Array,
  along: AxisPatch[],
  kind: "gene" | "signature",
  lengthPx: number,
  lengthUm: number | null,
): GradientResult {
  const n = along.length;
  const tRanks = ranksOf(along.map((a) => a.t));
  // The ends, for saying which way and by how much in readable terms.
  const third = Math.max(1, Math.round(n / 3));
  const startIdx = [...along].sort((a, b) => a.t - b.t).slice(0, third);
  const endIdx = [...along].sort((a, b) => b.t - a.t).slice(0, third);
  const pos = new Map(along.map((a, i) => [a.index, i]));

  const items: GradientStat[] = [];
  const ps: number[] = [];
  for (let c = 0; c < names.length; c++) {
    const col = column(c);

    /*
     * A column that never changes correlates with nothing, and ranking it
     * would sort a thousand identical values to prove it. Worth the check: a
     * whole-transcriptome map holds thousands of genes the model predicts flat
     * across any given corridor, and the sort is what the whole pass costs.
     */
    let flat = true;
    for (let i = 1; i < col.length; i++) {
      if (col[i] !== col[0]) { flat = false; break; }
    }
    const rho = flat ? 0 : spearman(ranksOf(col), tRanks);
    // Fisher's z, the usual large-sample approximation for Spearman.
    const z = Math.abs(rho) >= 1 ? Infinity : Math.atanh(rho) * Math.sqrt(n - 3);
    const p = n > 3 && Number.isFinite(z) ? Math.min(1, 2 * normalSf(Math.abs(z))) : 1;
    // Taken now, from the buffer this column currently occupies — it is
    // overwritten by the next one.
    const mean = (set: AxisPatch[]) =>
      set.reduce((s, a) => s + col[pos.get(a.index)!], 0) / set.length;
    items.push({ name: names[c], rho, meanStart: mean(startIdx), meanEnd: mean(endIdx), p, q: 1 });
    ps.push(p);
  }

  const qs = benjaminiHochberg(ps);
  items.forEach((s, i) => { s.q = qs[i]; });
  // Steepest first, in both directions — a gene that falls along the axis is
  // as much of a finding as one that rises.
  items.sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho));
  return { items, used: n, lengthPx, lengthUm, kind };
}

function axisLength(axis: Annotation): number {
  if (axis.geometry.type !== "LineString") return 0;
  const pts = axis.geometry.coordinates;
  const [ax, ay] = pts[0];
  const [bx, by] = pts[pts.length - 1];
  return Math.hypot(bx - ax, by - ay);
}

const MIN_PATCHES = 8;

function check(along: AxisPatch[]): void {
  if (along.length < MIN_PATCHES) {
    throw new NoPatchesAlongAxis(
      `Only ${along.length} patches lie along this axis — at least ${MIN_PATCHES} are needed. ` +
        "Draw a longer arrow, widen the corridor, or predict over more of the slide.",
    );
  }
}

/** Which genes change along the axis. */
export function geneGradient(
  result: SpatialResult,
  axis: Annotation,
  width: number,
  mppX: number | null,
): GradientResult {
  const along = patchesAlong(result, axis, width);
  check(along);

  const stride = result.genes.length;
  const buffer = new Float32Array(along.length);
  const column = (g: number) => {
    for (let i = 0; i < along.length; i++) {
      buffer[i] = valueAt(result, along[i].index * stride + g);
    }
    return buffer;
  };
  const len = axisLength(axis);
  return assemble(result.genes, column, along, "gene", len, mppX ? len * mppX : null);
}

/** Which cell types change along the axis. */
export function signatureGradient(
  result: SpatialResult,
  axis: Annotation,
  width: number,
  mppX: number | null,
  signatures: Signature[],
  score: (r: SpatialResult, s: Signature) => Float32Array | null,
): GradientResult {
  const along = patchesAlong(result, axis, width);
  check(along);

  const names: string[] = [];
  const scored: Float32Array[] = [];
  for (const signature of signatures) {
    // Scored over every patch, then cut down to the corridor: the
    // standardisation inside scoreSignature has to see the whole slide, or each
    // axis would be centred on its own contents and stop being comparable.
    const full = score(result, signature);
    if (!full) continue;
    names.push(signature.name);
    scored.push(full);
  }
  if (!scored.length) {
    throw new NoPatchesAlongAxis("None of these signatures share enough genes with this map.");
  }

  const buffer = new Float32Array(along.length);
  const column = (c: number) => {
    for (let i = 0; i < along.length; i++) buffer[i] = scored[c][along[i].index];
    return buffer;
  };
  const len = axisLength(axis);
  return assemble(names, column, along, "signature", len, mppX ? len * mppX : null);
}

export function gradientCsv(result: GradientResult, axis: string): string {
  const head =
    `${result.kind === "signature" ? "cell_type" : "gene"}` +
    ",rho,mean_start,mean_end,p,q,n_patches,axis_um,axis";
  const rows = result.items.map((g) =>
    [
      g.name, g.rho.toFixed(4), g.meanStart.toFixed(5), g.meanEnd.toFixed(5),
      g.p.toExponential(3), g.q.toExponential(3), result.used,
      result.lengthUm ? result.lengthUm.toFixed(0) : "",
      JSON.stringify(axis),
    ].join(","),
  );
  return `${head}\n${rows.join("\n")}\n`;
}
