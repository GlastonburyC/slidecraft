/**
 * Principal components of patch embeddings, by randomised SVD.
 *
 * Clustering runs on components rather than on the raw 2560 dimensions for the
 * usual reason: distances in very high dimensions concentrate, so nearest
 * neighbours stop meaning much, and the leading components carry the structure
 * while the tail is mostly noise. It is also what makes a k-nearest-neighbour
 * graph affordable — fifty numbers per patch instead of two and a half
 * thousand.
 *
 * Randomised rather than exact: a full SVD of an n x 2560 matrix is far more
 * work than the leading fifty components need, and the randomised algorithm
 * (Halko, Martinsson & Tropp) recovers them to well within the precision that
 * a clustering built on them can distinguish.
 */

export interface Pca {
  /** `n * k`, row-major: each patch's coordinates in component space. */
  scores: Float32Array;
  n: number;
  k: number;
  /** Share of total variance each component explains, largest first. */
  explained: number[];
}

/** Deterministic normal deviates, so a rerun gives the same components. */
function gaussian(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    // xorshift32 for the uniforms, Box-Muller for the shape.
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    const u1 = ((s >>> 8) + 1) / 16777217;
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    const u2 = ((s >>> 8) + 1) / 16777217;
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
}

/** Thin QR by modified Gram-Schmidt; `a` is n x m, column-major operations. */
function orthonormalise(a: Float64Array, n: number, m: number): void {
  for (let j = 0; j < m; j++) {
    for (let p = 0; p < j; p++) {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += a[i * m + j] * a[i * m + p];
      for (let i = 0; i < n; i++) a[i * m + j] -= dot * a[i * m + p];
    }
    let norm = 0;
    for (let i = 0; i < n; i++) norm += a[i * m + j] * a[i * m + j];
    norm = Math.sqrt(norm);
    // A dependent column contributes nothing; leaving it zero keeps the rest
    // orthonormal rather than propagating a division by ~0.
    if (norm < 1e-9) continue;
    for (let i = 0; i < n; i++) a[i * m + j] /= norm;
  }
}

/** Eigen-decomposition of a small symmetric matrix, by cyclic Jacobi. */
function jacobi(a: Float64Array, m: number): { values: Float64Array; vectors: Float64Array } {
  const v = new Float64Array(m * m);
  for (let i = 0; i < m; i++) v[i * m + i] = 1;

  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < m; p++) {
      for (let q = p + 1; q < m; q++) off += a[p * m + q] * a[p * m + q];
    }
    if (off < 1e-18) break;

    for (let p = 0; p < m; p++) {
      for (let q = p + 1; q < m; q++) {
        const apq = a[p * m + q];
        if (Math.abs(apq) < 1e-18) continue;
        const theta = (a[q * m + q] - a[p * m + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let i = 0; i < m; i++) {
          const aip = a[i * m + p];
          const aiq = a[i * m + q];
          a[i * m + p] = c * aip - s * aiq;
          a[i * m + q] = s * aip + c * aiq;
        }
        for (let i = 0; i < m; i++) {
          const api = a[p * m + i];
          const aqi = a[q * m + i];
          a[p * m + i] = c * api - s * aqi;
          a[q * m + i] = s * api + c * aqi;
        }
        for (let i = 0; i < m; i++) {
          const vip = v[i * m + p];
          const viq = v[i * m + q];
          v[i * m + p] = c * vip - s * viq;
          v[i * m + q] = s * vip + c * viq;
        }
      }
    }
  }

  const values = new Float64Array(m);
  for (let i = 0; i < m; i++) values[i] = a[i * m + i];
  return { values, vectors: v };
}

export function pca(x: Float32Array, n: number, d: number, want = 50, seed = 1): Pca {
  const k = Math.max(1, Math.min(want, Math.min(n, d) - 1 || 1));
  // Oversampled, as the randomised algorithm requires to separate the leading
  // subspace from the tail it is approximating away.
  const m = Math.min(d, Math.min(n, k + 10));

  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += x[i * d + j];
  for (let j = 0; j < d; j++) mean[j] /= n;

  const centred = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) centred[i * d + j] = x[i * d + j] - mean[j];
  }

  const rand = gaussian(seed);
  const omega = new Float64Array(d * m);
  for (let i = 0; i < d * m; i++) omega[i] = rand();

  // Y = X * Omega, then orthonormalise; two power iterations sharpen the
  // subspace when the spectrum decays slowly, which embeddings tend to do.
  const y = new Float64Array(n * m);
  const mul = (left: Float64Array, right: Float64Array, out: Float64Array) => {
    out.fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) {
        const value = left[i * d + j];
        if (value === 0) continue;
        for (let c = 0; c < m; c++) out[i * m + c] += value * right[j * m + c];
      }
    }
  };
  mul(centred, omega, y);
  orthonormalise(y, n, m);

  const z = new Float64Array(d * m);
  for (let iter = 0; iter < 2; iter++) {
    z.fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) {
        const value = centred[i * d + j];
        if (value === 0) continue;
        for (let c = 0; c < m; c++) z[j * m + c] += value * y[i * m + c];
      }
    }
    mul(centred, z, y);
    orthonormalise(y, n, m);
  }

  // B = Q^T X, then eigen-decompose the small B B^T rather than taking an SVD
  // of a matrix that is still d columns wide.
  const b = new Float64Array(m * d);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < m; c++) {
      const q = y[i * m + c];
      if (q === 0) continue;
      for (let j = 0; j < d; j++) b[c * d + j] += q * centred[i * d + j];
    }
  }

  const bbt = new Float64Array(m * m);
  for (let a = 0; a < m; a++) {
    for (let c = a; c < m; c++) {
      let dot = 0;
      for (let j = 0; j < d; j++) dot += b[a * d + j] * b[c * d + j];
      bbt[a * m + c] = dot;
      bbt[c * m + a] = dot;
    }
  }

  const { values, vectors } = jacobi(bbt, m);
  const order = Array.from({ length: m }, (_, i) => i).sort((p, q) => values[q] - values[p]);

  const scores = new Float32Array(n * k);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < k; c++) {
      // scores = Q * Ub, scaled by the singular value implicitly through Ub.
      let acc = 0;
      const col = order[c];
      for (let a = 0; a < m; a++) acc += y[i * m + a] * vectors[a * m + col];
      scores[i * k + c] = acc * Math.sqrt(Math.max(0, values[col]));
    }
  }

  const total = order.reduce((sum, i) => sum + Math.max(0, values[i]), 0) || 1;
  const explained = order.slice(0, k).map((i) => Math.max(0, values[i]) / total);
  return { scores, n, k, explained };
}
