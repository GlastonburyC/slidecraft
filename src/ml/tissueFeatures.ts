/**
 * Per-cell features for tissue classification.
 *
 * The heuristic detector scores a cell on colour alone — how saturated it is,
 * how far below white. That cannot separate faint tissue from the things a
 * slide collects: dust, bubbles, mounting-medium smears, pen ghosts, the shadow
 * of a coverslip edge. Those are pale like faint tissue and, on colour alone,
 * indistinguishable from it.
 *
 * What separates them is texture. Tissue has structure at the scale of glands
 * and cell clusters, so its neighbourhood varies; an artefact is smooth,
 * because it is a smudge on glass. So every cell carries local variation at two
 * scales alongside its colour, and the classifier is given the chance to learn
 * that "pale but flat" is not tissue while "pale but busy" is.
 *
 * Bumping FEATURE_VERSION invalidates saved models: a weight vector is only
 * meaningful against the features it was fitted to, and silently applying one
 * to a different set produces confident nonsense.
 */

export const FEATURE_VERSION = 1;

export const FEATURE_NAMES = [
  "saturation",
  "darkness",
  "redGreen",
  "blueGreen",
  "detailFine",
  "detailCoarse",
  "saturationDetail",
  "localRange",
  "neighbourDarkness",
  "neighbourSaturation",
  "gradient",
  "chroma",
] as const;

export const FEATURE_COUNT = FEATURE_NAMES.length;

/** Radii, in overview cells, of the two texture scales. */
const FINE = 1;
const COARSE = 4;

/**
 * Summed-area table over `src`, one row and column larger so any window sum is
 * four lookups regardless of its size. Without it the coarse-scale statistics
 * would cost O(radius²) per cell, which at a 2048-cell overview is minutes.
 */
function integral(src: Float32Array, w: number, h: number): Float64Array {
  const out = new Float64Array((w + 1) * (h + 1));
  for (let y = 0; y < h; y++) {
    let rowSum = 0;
    for (let x = 0; x < w; x++) {
      rowSum += src[y * w + x];
      out[(y + 1) * (w + 1) + (x + 1)] = out[y * (w + 1) + (x + 1)] + rowSum;
    }
  }
  return out;
}

function windowSum(
  sat: Float64Array, w: number, h: number,
  x: number, y: number, r: number,
): { sum: number; count: number } {
  const x0 = Math.max(0, x - r);
  const y0 = Math.max(0, y - r);
  const x1 = Math.min(w - 1, x + r);
  const y1 = Math.min(h - 1, y + r);
  const stride = w + 1;
  const sum =
    sat[(y1 + 1) * stride + (x1 + 1)] -
    sat[y0 * stride + (x1 + 1)] -
    sat[(y1 + 1) * stride + x0] +
    sat[y0 * stride + x0];
  return { sum, count: (x1 - x0 + 1) * (y1 - y0 + 1) };
}

/** Standard deviation over a window, from the sum and sum-of-squares tables. */
function windowStd(
  sum: Float64Array, sumSq: Float64Array, w: number, h: number,
  x: number, y: number, r: number,
): number {
  const a = windowSum(sum, w, h, x, y, r);
  const b = windowSum(sumSq, w, h, x, y, r);
  const mean = a.sum / a.count;
  // Clamped because catastrophic cancellation can make this very slightly
  // negative on a perfectly uniform window.
  return Math.sqrt(Math.max(0, b.sum / b.count - mean * mean));
}

export interface Overview {
  rgba: Uint8ClampedArray;
  w: number;
  h: number;
}

/**
 * Feature matrix, row-major: cell `i` occupies `[i * FEATURE_COUNT, …)`.
 *
 * Returned as one flat Float32Array rather than an array of vectors because
 * training walks it several hundred times and a 4-megapixel overview would
 * otherwise mean four million short-lived arrays.
 */
export function computeFeatures({ rgba, w, h }: Overview): Float32Array {
  const n = w * h;
  const lum = new Float32Array(n);
  const lumSq = new Float32Array(n);
  const sat = new Float32Array(n);
  const satSq = new Float32Array(n);
  const dark = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const r = rgba[i * 4];
    const g = rgba[i * 4 + 1];
    const b = rgba[i * 4 + 2];
    const max = r > g ? (r > b ? r : b) : g > b ? g : b;
    const min = r < g ? (r < b ? r : b) : g < b ? g : b;
    const s = max === 0 ? 0 : ((max - min) / max) * 255;
    const l = 0.299 * r + 0.587 * g + 0.114 * b;
    lum[i] = l;
    lumSq[i] = l * l;
    sat[i] = s;
    satSq[i] = s * s;
    dark[i] = 255 - l;
  }

  const iLum = integral(lum, w, h);
  const iLumSq = integral(lumSq, w, h);
  const iSat = integral(sat, w, h);
  const iSatSq = integral(satSq, w, h);
  const iDark = integral(dark, w, h);

  const out = new Float32Array(n * FEATURE_COUNT);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const r = rgba[i * 4];
      const g = rgba[i * 4 + 1];
      const b = rgba[i * 4 + 2];

      // Central differences, clamped at the border.
      const xm = x > 0 ? i - 1 : i;
      const xp = x < w - 1 ? i + 1 : i;
      const ym = y > 0 ? i - w : i;
      const yp = y < h - 1 ? i + w : i;
      const gx = lum[xp] - lum[xm];
      const gy = lum[yp] - lum[ym];

      const nDark = windowSum(iDark, w, h, x, y, COARSE);
      const nSat = windowSum(iSat, w, h, x, y, COARSE);

      const o = i * FEATURE_COUNT;
      out[o + 0] = sat[i];
      out[o + 1] = dark[i];
      out[o + 2] = r - g;
      out[o + 3] = b - g;
      out[o + 4] = windowStd(iLum, iLumSq, w, h, x, y, FINE);
      out[o + 5] = windowStd(iLum, iLumSq, w, h, x, y, COARSE);
      out[o + 6] = windowStd(iSat, iSatSq, w, h, x, y, COARSE);
      out[o + 7] = windowStd(iLum, iLumSq, w, h, x, y, 2) * 2;
      out[o + 8] = nDark.sum / nDark.count;
      out[o + 9] = nSat.sum / nSat.count;
      out[o + 10] = Math.hypot(gx, gy);
      out[o + 11] = Math.max(r, g, b) - Math.min(r, g, b);
    }
  }
  return out;
}
