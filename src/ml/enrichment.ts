import { containsPoint, isAreaGeometry, type Annotation } from "../annotate/types";
import { fromHalf, valueAt, type SpatialResult } from "./spatialResult";
import type { Signature } from "./signatures";

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
  /** The gene, or the cell type, depending on what was compared. */
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
  /**
   * What was compared. Signature scores are standardised per gene before being
   * averaged, so their means are in standard deviations and a gene's are in the
   * model's own units — the two are not on the same scale and the label has to
   * say which you are looking at.
   */
  kind: "gene" | "signature";
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

/**
 * Mann-Whitney U, by sorting only the smaller of the two groups.
 *
 * U counts the pairs where an inside value beats an outside one, and that is
 * symmetric — so it can be had by sorting either group and searching the other
 * against it. Sorting the smaller one is what makes this usable on a whole
 * transcriptome: a drawn region is typically a tenth of a slide, and the cost
 * is dominated by that sort.
 *
 * Measured on 18,961 patches, per gene: sorting an index array through a
 * comparator, which is how this began, 5.10 ms; a native sort of every value,
 * 0.95 ms; a native sort of the region alone, 0.08 ms. Across 19,338 genes that
 * is 99 seconds against 1.5 — the difference between a frozen tab and a wait.
 *
 * A native typed-array sort also beats a comparator because there is no
 * per-comparison call into JavaScript, and no index array to allocate.
 */
function mannWhitney(
  values: Float32Array,
  mask: Uint8Array,
  nIn: number,
  nOut: number,
  small: Float32Array,
  large: Float32Array,
): { u: number } {
  const n = values.length;
  const sortSmaller = nIn <= nOut;

  let s = 0;
  let l = 0;
  for (let i = 0; i < n; i++) {
    // A byte lookup, not a hash: this runs once per patch per gene, which on a
    // whole transcriptome is hundreds of millions of times.
    if ((mask[i] === 1) === sortSmaller) small[s++] = values[i];
    else large[l++] = values[i];
  }
  small.sort();

  // How many of `small` are below a value, and how many equal it.
  const m = small.length;
  let below = 0;
  let equal = 0;
  for (let i = 0; i < large.length; i++) {
    const v = large[i];
    let lo = 0;
    let hi = m;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (small[mid] < v) lo = mid + 1; else hi = mid; }
    const first = lo;
    hi = m;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (small[mid] <= v) lo = mid + 1; else hi = mid; }
    below += first;
    equal += lo - first;
  }

  /*
   * `below` counts (large > small) pairs and U is defined over (inside >
   * outside), so which one it is depends on which group was sorted. Ties split
   * evenly either way.
   */
  const u = sortSmaller
    ? nIn * nOut - below - equal / 2   // small = inside: below counts outside > inside
    : below + equal / 2;               // small = outside: below counts outside < inside
  return { u };
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


/**
 * Compare one column of numbers inside a region against outside it.
 *
 * The column is a gene's values across the patches, or a signature's score
 * across them — the test does not care which, which is the whole reason it
 * lives here rather than inside the gene loop.
 */
function compare(
  values: Float32Array,
  mask: Uint8Array,
  nIn: number,
  nOut: number,
  small: Float32Array,
  large: Float32Array,
): { meanIn: number; meanOut: number; auc: number; p: number } {
  const n = values.length;
  let sumIn = 0;
  let sumOut = 0;
  for (let i = 0; i < n; i++) {
    if (mask[i] === 1) sumIn += values[i];
    else sumOut += values[i];
  }

  const { u } = mannWhitney(values, mask, nIn, nOut, small, large);
  const auc = u / (nIn * nOut);

  const mean = (nIn * nOut) / 2;
  /*
   * Without the tie correction, which would cost a sort of the whole column to
   * obtain and makes the test CONSERVATIVE by its absence: ties reduce the true
   * variance, so leaving them out overstates it, shrinks |z| and enlarges p.
   *
   * That is the safe direction, and it matters little here — these p-values are
   * already optimistic for a reason no correction addresses, which is that
   * neighbouring patches are near-copies and the effective sample size is far
   * below the patch count. The ranking is by AUC, and AUC is exact.
   */
  const varU = ((nIn * nOut) / 12) * (n + 1);
  // Every value identical: no ordering to compare, so no evidence either way.
  const z = varU > 0 ? (u - mean) / Math.sqrt(varU) : 0;
  const p = varU > 0 ? Math.min(1, 2 * normalSf(Math.abs(z))) : 1;

  return { meanIn: sumIn / nIn, meanOut: sumOut / nOut, auc, p };
}

/** Shared by both entry points: the region has to be big enough to test. */
function checkSize(nIn: number, nOut: number, minPatches: number): void {
  if (nIn < minPatches || nOut < minPatches) {
    throw new NotEnoughPatches(
      `Needs at least ${minPatches} patches on each side — this region has ${nIn} inside and ${nOut} outside. ` +
        "Draw a larger region, or predict over more of the slide.",
    );
  }
}

/** Rank, adjust and order a set of comparisons the same way for both. */
function finish(
  stats: GeneStat[],
  ps: number[],
  nIn: number,
  nOut: number,
  kind: "gene" | "signature",
): EnrichmentResult {
  const qs = benjaminiHochberg(ps);
  stats.forEach((s, i) => { s.q = qs[i]; });
  // Most separable first, which is what "over-represented here" means.
  stats.sort((a, b) => b.auc - a.auc);
  return { genes: stats, inside: nIn, outside: nOut, kind };
}

/**
 * U by counting, when the map is half precision.
 *
 * fp16 has 65,536 possible values, so the comparison sort can be dropped
 * entirely: bin every value by its bit pattern, then sweep the bins in order.
 * Each pass is O(patches), the sweep is a fixed 65,536, and the sweep clears
 * the bins as it goes so there is nothing to reset.
 *
 * The bits need one adjustment to sort as the numbers do. IEEE floats compare
 * correctly as integers only when positive; a sign bit makes the order run
 * backwards. Inverting negatives and setting the top bit on positives maps both
 * onto one ascending range.
 */
function keyOf(bits: number): number {
  /*
   * Negative zero is the same number as positive zero, and has a different bit
   * pattern. Left alone it lands in its own bin and the sweep puts it BELOW
   * +0 instead of tied with it — worth half a pair each time it meets one,
   * which is exactly the discrepancy a differential test against the sorting
   * path turned up. Predicted expression sits near zero constantly, so this is
   * not a theoretical case.
   */
  const b = bits === 0x8000 ? 0 : bits;
  return b & 0x8000 ? (~b) & 0xffff : b | 0x8000;
}

const BINS = 65536;

function countingU(
  raw: Uint16Array,
  offset: number,
  stride: number,
  n: number,
  mask: Uint8Array,
  histIn: Uint32Array,
  histOut: Uint32Array,
): number {
  for (let i = 0, at = offset; i < n; i++, at += stride) {
    const k = keyOf(raw[at]);
    if (mask[i] === 1) histIn[k]++;
    else histOut[k]++;
  }

  let u = 0;
  let cumOut = 0;
  for (let k = 0; k < BINS; k++) {
    const a = histIn[k];
    const b = histOut[k];
    if (a !== 0) {
      // Inside values in this bin beat every outside value below it, and tie
      // with the ones in it.
      u += a * (cumOut + b / 2);
      histIn[k] = 0;
    }
    if (b !== 0) {
      cumOut += b;
      histOut[k] = 0;
    }
  }
  return u;
}

export function differentialExpression(
  result: SpatialResult,
  inside: Set<number>,
  minPatches = 5,
): EnrichmentResult {
  const n = result.patches.length;
  const nIn = inside.size;
  const nOut = n - nIn;
  checkSize(nIn, nOut, minPatches);

  const stride = result.genes.length;
  const stats: GeneStat[] = [];
  const ps: number[] = [];

  // Set membership becomes a byte lookup, and the two partitions are allocated
  // once rather than per gene.
  const mask = new Uint8Array(n);
  for (const i of inside) mask[i] = 1;
  const sortSmaller = nIn <= nOut;
  const small = new Float32Array(sortSmaller ? nIn : nOut);
  const large = new Float32Array(sortSmaller ? nOut : nIn);

  /*
   * Genes are read in blocks, because the layout is row-major by patch.
   *
   * Taking one gene at a time walks the whole map with a stride of the gene
   * count — 38 KB on a whole transcriptome — so every read is a cache miss, and
   * there are patches times genes of them. Reading BLOCK genes at once makes
   * each patch's contribution a short contiguous run instead, and the same
   * bytes then serve BLOCK columns.
   *
   * This is what the cost actually was. Sorting looked like the problem and was
   * worth fixing on its own, but it was a third of it; the strided read was the
   * rest.
   */
  /*
   * Half-precision maps take the counting path, which is every map produced by
   * the GPU script and so the case that matters. fp32 falls back to sorting.
   */
  const raw = result.half ? (result.values as Uint16Array) : null;

  if (raw) {
    const histIn = new Uint32Array(BINS);
    const histOut = new Uint32Array(BINS);
    const pairs = nIn * nOut;
    const varU = (pairs / 12) * (n + 1);

    for (let g = 0; g < stride; g++) {
      // Means still need the decoded values, and this pass is what pays for
      // them — one walk of the column, shared with nothing.
      let sumIn = 0;
      let sumOut = 0;
      for (let i = 0, at = g; i < n; i++, at += stride) {
        const v = fromHalf(raw[at]);
        if (mask[i] === 1) sumIn += v;
        else sumOut += v;
      }

      const u = countingU(raw, g, stride, n, mask, histIn, histOut);
      const auc = u / pairs;
      const z = varU > 0 ? (u - pairs / 2) / Math.sqrt(varU) : 0;
      const p = varU > 0 ? Math.min(1, 2 * normalSf(Math.abs(z))) : 1;
      const meanIn = sumIn / nIn;
      const meanOut = sumOut / nOut;
      stats.push({ gene: result.genes[g], meanIn, meanOut, diff: meanIn - meanOut, auc, p, q: 1 });
      ps.push(p);
    }
    return finish(stats, ps, nIn, nOut, "gene");
  }

  const values = new Float32Array(n);
  for (let g = 0; g < stride; g++) {
    for (let i = 0; i < n; i++) values[i] = valueAt(result, i * stride + g);
    const { meanIn, meanOut, auc, p } = compare(values, mask, nIn, nOut, small, large);
    stats.push({ gene: result.genes[g], meanIn, meanOut, diff: meanIn - meanOut, auc, p, q: 1 });
    ps.push(p);
  }

  return finish(stats, ps, nIn, nOut, "gene");
}

/**
 * Which cell types are over-represented in a region you drew.
 *
 * The same test as the gene version, over signature scores instead of gene
 * values — which is the question people usually have. "CXCL13 is enriched
 * here" needs you to already know what CXCL13 means; "this is a lymphoid
 * aggregate" does not.
 *
 * Reading it, two things differ from the gene version. The means are in
 * standard deviations rather than the model's units, because scoreSignature
 * standardises each gene before averaging so an abundant one cannot carry the
 * signature by itself. And the q-values are weaker still: signatures share
 * genes — every epithelial type draws on KRT8 — so the comparisons are
 * correlated with each other on top of the spatial autocorrelation between
 * patches. Rank by AUC and treat q as a filter, not as evidence.
 */
export function differentialSignatures(
  result: SpatialResult,
  inside: Set<number>,
  signatures: Signature[],
  score: (r: SpatialResult, s: Signature) => Float32Array | null,
  minPatches = 5,
): EnrichmentResult {
  const n = result.patches.length;
  const nIn = inside.size;
  const nOut = n - nIn;
  checkSize(nIn, nOut, minPatches);

  const stats: GeneStat[] = [];
  const ps: number[] = [];

  // Same shared buffers as the gene path, for the same reasons.
  const mask = new Uint8Array(n);
  for (const i of inside) mask[i] = 1;
  const sortSmaller = nIn <= nOut;
  const small = new Float32Array(sortSmaller ? nIn : nOut);
  const large = new Float32Array(sortSmaller ? nOut : nIn);

  for (const signature of signatures) {
    // A signature the map cannot cover scores null rather than zero, and is
    // left out rather than ranked against the ones it can.
    const values = score(result, signature);
    if (!values || values.length !== n) continue;
    const { meanIn, meanOut, auc, p } = compare(values, mask, nIn, nOut, small, large);
    stats.push({ gene: signature.name, meanIn, meanOut, diff: meanIn - meanOut, auc, p, q: 1 });
    ps.push(p);
  }

  if (stats.length === 0) {
    throw new NotEnoughPatches(
      "None of these signatures share enough genes with this map to be scored.",
    );
  }

  return finish(stats, ps, nIn, nOut, "signature");
}

export function enrichmentCsv(result: EnrichmentResult, region: string): string {
  const head =
    `${result.kind === "signature" ? "cell_type" : "gene"}` +
    ",mean_in,mean_out,diff,auc,p,q,n_in,n_out,region";
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
