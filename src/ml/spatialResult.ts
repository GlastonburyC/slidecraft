import type { Patch } from "./patchGrid";

/**
 * Predicted expression over a patch grid.
 *
 * Stored as one flat array rather than an object per patch: a grid runs to
 * thousands of patches and a colour pass touches every value on every change
 * of gene, so the layout that matters is the one that iterates without chasing
 * pointers.
 *
 * A whole-transcriptome map is kept in the half precision it arrived in, and
 * decoded one gene at a time. 32,054 patches by 19,338 genes is 620 million
 * values: 1.2 GB as fp16 and 2.5 GB widened to fp32, on top of the file buffer
 * itself — which no browser tab survives, to expand 19,338 genes when one is
 * on screen. Decoding a gene's column costs a pass over the patches, which is
 * what drawing it costs anyway.
 */
export interface SpatialResult {
  /** Gene symbols, in the order the model returns them. */
  genes: string[];
  /**
   * `patches.length * genes.length` values, row-major by patch. Half-precision
   * when `half` is set — read it through `valueAt` rather than indexing.
   */
  values: Float32Array | Uint16Array;
  /** Whether `values` holds IEEE half bit patterns rather than numbers. */
  half?: boolean;
  patches: Patch[];
  /** Side of a patch in level-0 slide pixels, for drawing. */
  side: number;
  modelId: string;
  modelName: string;
  slide: string;
  /** The ROI this covered, when it was scoped to one. */
  roiId: string | null;
  ms: number;
  createdAt: string;
}


/** IEEE half to double. */
export function fromHalf(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits & 0x7c00) >> 10;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

/** One value out of the flat array, whichever precision it is stored in. */
export function valueAt(result: SpatialResult, k: number): number {
  const v = result.values[k];
  return result.half ? fromHalf(v) : v;
}

export function geneIndex(result: SpatialResult, gene: string): number {
  return result.genes.indexOf(gene);
}

/** Every patch's value for one gene. */
export function geneValues(result: SpatialResult, gene: string): Float32Array | null {
  const g = geneIndex(result, gene);
  if (g < 0) return null;
  const n = result.patches.length;
  const stride = result.genes.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = valueAt(result, i * stride + g);
  return out;
}

export interface Range {
  min: number;
  max: number;
}

/**
 * The range to colour across, from percentiles rather than the extremes.
 *
 * One saturated patch — a fold, a pen mark, an edge artefact — sets the maximum
 * and flattens everything else to the bottom of the scale, which reads as "no
 * expression anywhere" when the truth is "one outlier". Clipping at the 2nd and
 * 98th percentiles keeps the visible range on the data.
 */
export function robustRange(values: Float32Array, lowPct = 2, highPct = 98): Range {
  if (values.length === 0) return { min: 0, max: 1 };
  const sorted = Float32Array.from(values).sort();
  const at = (p: number) =>
    sorted[Math.min(sorted.length - 1, Math.max(0, Math.round((p / 100) * (sorted.length - 1))))];
  const min = at(lowPct);
  const max = at(highPct);
  // A flat field would divide by zero; widen it so everything lands mid-scale.
  return max - min > 1e-9 ? { min, max } : { min: min - 0.5, max: min + 0.5 };
}

/** Viridis, sampled. Perceptually uniform, and readable in greyscale print. */
const VIRIDIS: [number, number, number][] = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
  [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
  [180, 222, 44], [253, 231, 37],
];

export function colourFor(value: number, range: Range): [number, number, number] {
  const t = Math.min(1, Math.max(0, (value - range.min) / (range.max - range.min)));
  const scaled = t * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(scaled));
  const f = scaled - i;
  const a = VIRIDIS[i];
  const b = VIRIDIS[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

/** Legend stops, for drawing the colour bar. */
export function legendStops(steps = 8): [number, number, number][] {
  return Array.from({ length: steps }, (_, i) =>
    colourFor(i / (steps - 1), { min: 0, max: 1 }),
  );
}

/** One row per patch, so the prediction can leave for R or scanpy. */
export function toCsv(result: SpatialResult): string {
  const head = ["patch", "col", "row", "x", "y", "size", ...result.genes].join(",");
  const stride = result.genes.length;
  const rows = result.patches.map((p, i) => {
    const vals: string[] = [];
    for (let g = 0; g < stride; g++) vals.push(valueAt(result, i * stride + g).toFixed(5));
    return [p.index, p.col, p.row, p.x, p.y, result.side, ...vals].join(",");
  });
  return `${head}\n${rows.join("\n")}\n`;
}


/**
 * The values currently being drawn, and what to call them.
 *
 * Both a gene and a signature reduce to one number per patch, so the overlay,
 * the legend and the colour scale never need to know which is on screen — only
 * this function does.
 */
export function currentField(
  result: SpatialResult,
  state: {
    mode: "gene" | "signature";
    gene: string | null;
    signatureName: string | null;
    signatures: { signatures: { name: string; genes: { gene: string; weight: number }[] }[] } | null;
  },
  score: (r: SpatialResult, s: { name: string; genes: { gene: string; weight: number }[] }) => Float32Array | null,
): { values: Float32Array; label: string } | null {
  if (state.mode === "signature" && state.signatureName && state.signatures) {
    const sig = state.signatures.signatures.find((s) => s.name === state.signatureName);
    if (!sig) return null;
    const values = score(result, sig);
    return values ? { values, label: sig.name } : null;
  }
  if (state.gene) {
    const values = geneValues(result, state.gene);
    if (values) return { values, label: state.gene };
  }
  return null;
}
