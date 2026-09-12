import type { Patch } from "./patchGrid";

/**
 * Blending overlapping patch predictions into a finer map.
 *
 * The encoder's input size is fixed, so a patch is the smallest thing it can
 * judge — but nothing stops the patch sliding by less than its own width. Step
 * it by half or a quarter and each point of tissue is seen by four or sixteen
 * patches, each from a different offset, and averaging them gives a map at the
 * stride's resolution rather than the patch's.
 *
 * The averaging is weighted by a raised cosine centred on each patch. A patch
 * describes its middle better than its corners — the corners are as much about
 * the neighbouring tissue that shares the field — so weighting by distance from
 * the centre both sharpens boundaries and removes the blocky seams that
 * uniform averaging leaves at patch edges.
 */

export interface Blend {
  /** `cols * rows * classes`, row-major, already normalised to sum to one. */
  probs: Float32Array;
  cols: number;
  rows: number;
  /** Top-left of the raster in level-0 slide pixels. */
  originX: number;
  originY: number;
  /** Side of one output cell in level-0 pixels — the stride. */
  cell: number;
  classes: number;
}

/**
 * Raised cosine over a patch, evaluated at a cell's centre.
 *
 * Never quite zero: a cell covered only by the rim of a single patch still has
 * to get an answer, and a zero weight everywhere would leave it blank.
 */
function taper(dx: number, dy: number, half: number): number {
  const rx = Math.min(1, Math.abs(dx) / half);
  const ry = Math.min(1, Math.abs(dy) / half);
  const w = 0.5 * (1 + Math.cos(Math.PI * rx)) * 0.5 * (1 + Math.cos(Math.PI * ry));
  return 0.02 + 0.98 * w;
}

export function blendPredictions(
  patches: Patch[],
  probs: Float32Array,
  classes: number,
  side: number,
  cell: number,
): Blend | null {
  if (patches.length === 0 || classes === 0) return null;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of patches) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x + side > maxX) maxX = p.x + side;
    if (p.y + side > maxY) maxY = p.y + side;
  }

  const cols = Math.max(1, Math.ceil((maxX - minX) / cell));
  const rows = Math.max(1, Math.ceil((maxY - minY) / cell));
  const acc = new Float64Array(cols * rows * classes);
  const weight = new Float64Array(cols * rows);
  const half = side / 2;

  for (const [n, p] of patches.entries()) {
    const c0 = Math.max(0, Math.floor((p.x - minX) / cell));
    const c1 = Math.min(cols - 1, Math.ceil((p.x + side - minX) / cell) - 1);
    const r0 = Math.max(0, Math.floor((p.y - minY) / cell));
    const r1 = Math.min(rows - 1, Math.ceil((p.y + side - minY) / cell) - 1);
    const cx = p.x + half;
    const cy = p.y + half;

    for (let r = r0; r <= r1; r++) {
      const y = minY + (r + 0.5) * cell;
      for (let c = c0; c <= c1; c++) {
        const x = minX + (c + 0.5) * cell;
        const w = taper(x - cx, y - cy, half);
        const cellIndex = r * cols + c;
        weight[cellIndex] += w;
        for (let k = 0; k < classes; k++) {
          acc[cellIndex * classes + k] += w * probs[n * classes + k];
        }
      }
    }
  }

  const out = new Float32Array(cols * rows * classes);
  for (let i = 0; i < cols * rows; i++) {
    const w = weight[i];
    // A cell no patch reached keeps zeros, and the renderer leaves it blank
    // rather than inventing a class for tissue nothing looked at.
    if (w <= 0) continue;
    for (let k = 0; k < classes; k++) out[i * classes + k] = acc[i * classes + k] / w;
  }

  return { probs: out, cols, rows, originX: minX, originY: minY, cell, classes };
}

/**
 * Blend one scalar field, which is the same operation with a single class.
 *
 * Used when an expression map was computed with a stride finer than its patch
 * size. Without it the overlay draws every patch at full width and they
 * overdraw each other, so a strided run costs many times the compute and looks
 * no finer — worse, in fact, since whichever patch happens to be drawn last
 * wins outright rather than being averaged with its neighbours.
 *
 * A second channel rides along for the tissue mask, so a cell can be left blank
 * when the patches reaching it were off tissue.
 */
export function blendField(
  patches: Patch[],
  values: Float32Array,
  mask: Uint8Array | null,
  side: number,
  cell: number,
): Blend | null {
  const n = patches.length;
  const pairs = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    pairs[i * 2] = values[i];
    pairs[i * 2 + 1] = mask ? mask[i] : 1;
  }
  return blendPredictions(patches, pairs, 2, side, cell);
}
