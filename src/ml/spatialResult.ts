import type { Patch } from "./patchGrid";

/**
 * Predicted expression over a patch grid.
 *
 * Stored as one flat Float32Array rather than an object per patch: a grid runs
 * to thousands of patches and a colour pass touches every value on every
 * change of gene, so the layout that matters is the one that iterates without
 * chasing pointers.
 */
export interface SpatialResult {
  /** Gene symbols, in the order the model returns them. */
  genes: string[];
  /** `patches.length * genes.length` values, row-major by patch. */
  values: Float32Array;
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
  for (let i = 0; i < n; i++) out[i] = result.values[i * stride + g];
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
    const vals = Array.from(
      result.values.subarray(i * stride, (i + 1) * stride),
      (v) => v.toFixed(5),
    );
    return [p.index, p.col, p.row, p.x, p.y, result.side, ...vals].join(",");
  });
  return `${head}\n${rows.join("\n")}\n`;
}
