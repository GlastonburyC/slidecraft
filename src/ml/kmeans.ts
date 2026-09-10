/**
 * k-means over principal components.
 *
 * Discovering classes rather than being told them: cluster the patches, look
 * at what each cluster picked out, rename the ones that mean something. It is
 * deliberately the simple choice — k-means++ initialisation and Lloyd
 * iterations, a few dozen lines, no graph and no tuning beyond k.
 *
 * Run on components, not raw embeddings, because distances concentrate in
 * 2560 dimensions and nearest-neighbour structure stops carrying much; the
 * leading components hold the variation and the tail is mostly noise.
 */

export interface Clusters {
  /** Cluster index per point, renumbered so 0 is the largest. */
  labels: Int32Array;
  k: number;
  sizes: number[];
  /** Mean squared distance to the assigned centre; lower is tighter. */
  inertia: number;
}

/** Deterministic uniforms, so the same patches always cluster the same way. */
function random(seed: number): () => number {
  let s = (seed || 1) >>> 0;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

const distance = (x: Float32Array, a: number, centres: Float64Array, c: number, dim: number) => {
  let sum = 0;
  for (let f = 0; f < dim; f++) {
    const d = x[a * dim + f] - centres[c * dim + f];
    sum += d * d;
  }
  return sum;
};

export function kmeans(x: Float32Array, n: number, dim: number, k: number, seed = 1): Clusters {
  const wanted = Math.max(1, Math.min(k, n));
  const rand = random(seed);
  const centres = new Float64Array(wanted * dim);

  /**
   * k-means++ seeding: each centre after the first is drawn with probability
   * proportional to its squared distance from the nearest existing one.
   * Uniform seeding regularly puts two centres in the same blob and leaves
   * another unfound, which looks like a clustering that simply missed
   * something rather than a bad start.
   */
  const first = Math.floor(rand() * n);
  for (let f = 0; f < dim; f++) centres[f] = x[first * dim + f];

  const nearest = new Float64Array(n);
  for (let i = 0; i < n; i++) nearest[i] = distance(x, i, centres, 0, dim);

  for (let c = 1; c < wanted; c++) {
    let total = 0;
    for (let i = 0; i < n; i++) total += nearest[i];
    let target = rand() * total;
    let pick = n - 1;
    for (let i = 0; i < n; i++) {
      target -= nearest[i];
      if (target <= 0) { pick = i; break; }
    }
    for (let f = 0; f < dim; f++) centres[c * dim + f] = x[pick * dim + f];
    for (let i = 0; i < n; i++) {
      nearest[i] = Math.min(nearest[i], distance(x, i, centres, c, dim));
    }
  }

  const labels = new Int32Array(n);
  const counts = new Int32Array(wanted);
  const sums = new Float64Array(wanted * dim);

  for (let iter = 0; iter < 100; iter++) {
    let moved = false;
    for (let i = 0; i < n; i++) {
      let best = 0;
      let bestD = Infinity;
      for (let c = 0; c < wanted; c++) {
        const d = distance(x, i, centres, c, dim);
        if (d < bestD) { bestD = d; best = c; }
      }
      if (labels[i] !== best) { labels[i] = best; moved = true; }
    }
    if (!moved && iter > 0) break;

    counts.fill(0);
    sums.fill(0);
    for (let i = 0; i < n; i++) {
      const c = labels[i];
      counts[c]++;
      for (let f = 0; f < dim; f++) sums[c * dim + f] += x[i * dim + f];
    }
    for (let c = 0; c < wanted; c++) {
      // An emptied cluster keeps its centre rather than collapsing to the
      // origin, which would drag every point toward it on the next pass.
      if (counts[c] === 0) continue;
      for (let f = 0; f < dim; f++) centres[c * dim + f] = sums[c * dim + f] / counts[c];
    }
  }

  let inertia = 0;
  for (let i = 0; i < n; i++) inertia += distance(x, i, centres, labels[i], dim);
  inertia /= Math.max(1, n);

  // Numbered by size, so "cluster 1" is always the biggest thing found.
  const tally = new Map<number, number>();
  for (let i = 0; i < n; i++) tally.set(labels[i], (tally.get(labels[i]) ?? 0) + 1);
  const bySize = [...tally.entries()].sort((a, b) => b[1] - a[1]);
  const rank = new Map(bySize.map(([id], i) => [id, i]));
  for (let i = 0; i < n; i++) labels[i] = rank.get(labels[i])!;

  return { labels, k: bySize.length, sizes: bySize.map(([, size]) => size), inertia };
}
