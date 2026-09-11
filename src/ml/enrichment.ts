import { containsPoint, isAreaGeometry, type Annotation } from "../annotate/types";
import { valueAt, type SpatialResult } from "./spatialResult";

/**
 * Which genes are enriched in a region you drew.
 *
 * Draw round the calcification, and this compares the patches inside it against
 * the patches outside, gene by gene, and ranks the difference. It is the
 * question the expression map exists to answer — the map shows you one gene at
 * a time, and this asks which genes to look at in the first place.
 *
 * Two decisions about the statistics, both of which change how the output
 * should be read.
 *
 * **The test is rank-based.** Predicted expression is not normally distributed
 * and its scale is arbitrary per gene, so a t-test's assumptions do not hold.
 * Mann-Whitney compares orderings, which survives both.
 *
 * **The p-values are optimistic, and deliberately not the headline.** Patches
 * are spatially autocorrelated: neighbouring patches are near-copies, so the
 * effective sample size is far below the patch count and every test is
 * anti-conservative. Ranking is therefore by effect size — the AUC, which says
 * how separable inside is from outside and is unaffected by how many
 * correlated patches went into it. The q-values are there to filter obvious
 * noise, not to be reported as evidence.
 */

export interface GeneStat {
  gene: string;
  meanIn: number;
  meanOut: number;
  /** Difference of means, in the model's own units. */
  diff: number;
  /**
   * Probability a random inside patch exceeds a random outside one. 0.5 is no
   * separation, 1 is perfect. The number to rank and to report.
   */
  auc: number;
  p: number;
  /** Benjamini-Hochberg adjusted p across the genes tested. */
  q: number;
}

export interface EnrichmentResult {
  genes: GeneStat[];
  inside: number;
  outside: number;
}

export class NotEnoughPatches extends Error {}

/** Indices of the patches whose centre falls inside any of these regions. */
export function patchesInside(result: SpatialResult, regions: Annotation[]): Set<number> {
  const areas = regions.filter((a) => isAreaGeometry(a.geometry));
  const inside = new Set<number>();
  if (!areas.length) return inside;

  const half = result.side / 2;
  result.patches.forEach((p, i) => {
    const cx = p.x + half;
    const cy = p.y + half;
    if (areas.some((a) => containsPoint(a.geometry, cx, cy))) inside.add(i);
  });
  return inside;
}

/** Ranks with ties averaged, plus the tie correction the U statistic needs. */
function rank(values: Float32Array): { ranks: Float64Array; tieTerm: number } {
  const n = values.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => values[a] - values[b]);
  const ranks = new Float64Array(n);
  let tieTerm = 0;

  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && values[order[j + 1]] === values[order[i]]) j++;
    // Ranks are 1-based, and a run of equal values shares their average.
    const avg = (i + j + 2) / 2;
    for (let k = i; k <= j; k++) ranks[order[k]] = avg;
    const t = j - i + 1;
    if (t > 1) tieTerm += t * t * t - t;
    i = j + 1;
  }
  return { ranks, tieTerm };
}

/** Normal tail, via a high-accuracy erfc approximation (Numerical Recipes). */
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

/** Benjamini-Hochberg, returning q in the input order. */
export function benjaminiHochberg(p: number[]): number[] {
  const n = p.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => p[a] - p[b]);
  const q = new Array<number>(n).fill(1);
  let previous = 1;
  // Walk from the largest p downward so the enforced monotonicity is a running
  // minimum rather than a second pass.
  for (let k = n - 1; k >= 0; k--) {
    const i = order[k];
    previous = Math.min(previous, (p[i] * n) / (k + 1));
    q[i] = Math.min(1, previous);
  }
  return q;
}

export function differentialExpression(
  result: SpatialResult,
  inside: Set<number>,
  minPatches = 5,
): EnrichmentResult {
  const n = result.patches.length;
  const nIn = inside.size;
  const nOut = n - nIn;

  if (nIn < minPatches || nOut < minPatches) {
    throw new NotEnoughPatches(
      `Needs at least ${minPatches} patches on each side — this region has ${nIn} inside and ${nOut} outside. ` +
        "Draw a larger region, or predict over more of the slide.",
    );
  }

  const stride = result.genes.length;
  const values = new Float32Array(n);
  const stats: GeneStat[] = [];
  const ps: number[] = [];

  for (let g = 0; g < stride; g++) {
    for (let i = 0; i < n; i++) values[i] = valueAt(result, i * stride + g);

    let sumIn = 0;
    let sumOut = 0;
    for (let i = 0; i < n; i++) {
      if (inside.has(i)) sumIn += values[i];
      else sumOut += values[i];
    }

    const { ranks, tieTerm } = rank(values);
    let rankSumIn = 0;
    for (const i of inside) rankSumIn += ranks[i];

    const u = rankSumIn - (nIn * (nIn + 1)) / 2;
    const auc = u / (nIn * nOut);

    const mean = (nIn * nOut) / 2;
    const varU =
      ((nIn * nOut) / 12) * (n + 1 - tieTerm / (n * (n - 1)));
    // Every value identical: no ordering to compare, so no evidence either way.
    const z = varU > 0 ? (u - mean) / Math.sqrt(varU) : 0;
    const p = varU > 0 ? Math.min(1, 2 * normalSf(Math.abs(z))) : 1;

    const meanIn = sumIn / nIn;
    const meanOut = sumOut / nOut;
    stats.push({ gene: result.genes[g], meanIn, meanOut, diff: meanIn - meanOut, auc, p, q: 1 });
    ps.push(p);
  }

  const qs = benjaminiHochberg(ps);
  stats.forEach((s, i) => { s.q = qs[i]; });
  // Most separable first, which is what "over-represented here" means.
  stats.sort((a, b) => b.auc - a.auc);

  return { genes: stats, inside: nIn, outside: nOut };
}

export function enrichmentCsv(result: EnrichmentResult, region: string): string {
  const head = "gene,mean_in,mean_out,diff,auc,p,q,n_in,n_out,region";
  const rows = result.genes.map((g) =>
    [
      g.gene,
      g.meanIn.toFixed(5),
      g.meanOut.toFixed(5),
      g.diff.toFixed(5),
      g.auc.toFixed(4),
      g.p.toExponential(3),
      g.q.toExponential(3),
      result.inside,
      result.outside,
      JSON.stringify(region),
    ].join(","),
  );
  return `${head}\n${rows.join("\n")}\n`;
}
