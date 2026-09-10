import { describe, expect, it } from "vitest";
import { kmeans } from "../ml/kmeans";
import { pca } from "../ml/pca";

const noise = (i: number, j: number) => {
  let n = (i * 374761393 + j * 668265263) >>> 0;
  n = ((n ^ (n >>> 13)) * 1274126177) >>> 0;
  return ((n >>> 16) & 1023) / 1023 - 0.5;
};

function blobs(perGroup: number, groups: number, dim: number, spread: number) {
  const n = perGroup * groups;
  const x = new Float32Array(n * dim);
  const truth = new Int32Array(n);
  let at = 0;
  for (let g = 0; g < groups; g++) {
    for (let i = 0; i < perGroup; i++) {
      for (let c = 0; c < dim; c++) x[at * dim + c] = (c === g ? 8 : 0) + noise(at, c) * spread;
      truth[at] = g;
      at++;
    }
  }
  return { x, truth, n };
}

/** Do two labellings agree, up to renaming? */
function agrees(a: Int32Array, b: Int32Array): boolean {
  const map = new Map<number, number>();
  const used = new Set<number>();
  for (let i = 0; i < a.length; i++) {
    const seen = map.get(a[i]);
    if (seen === undefined) {
      if (used.has(b[i])) return false;
      map.set(a[i], b[i]);
      used.add(b[i]);
    } else if (seen !== b[i]) return false;
  }
  return true;
}

describe("clustering components", () => {
  it("recovers separated groups", () => {
    const { x, truth, n } = blobs(40, 3, 16, 0.5);
    const { scores, k } = pca(x, n, 16, 6);
    const result = kmeans(scores, n, k, 3, 7);

    expect(result.k).toBe(3);
    expect(agrees(result.labels, truth)).toBe(true);
    expect(result.sizes).toEqual([40, 40, 40]);
  });

  it("numbers clusters by size, largest first", () => {
    const dim = 8;
    const bounds = [0, 60, 80, 90];
    const x = new Float32Array(90 * dim);
    for (let g = 0; g < 3; g++) {
      for (let i = bounds[g]; i < bounds[g + 1]; i++) {
        for (let c = 0; c < dim; c++) x[i * dim + c] = (c === g ? 8 : 0) + noise(i, c) * 0.4;
      }
    }
    const { scores, k } = pca(x, 90, dim, 4);
    const result = kmeans(scores, 90, k, 3, 3);
    expect(result.sizes).toEqual([...result.sizes].sort((a, b) => b - a));
    expect(result.sizes[0]).toBe(60);
  });

  it("labels every point, with no gaps in the numbering", () => {
    const { x, n } = blobs(25, 4, 12, 0.6);
    const { scores, k } = pca(x, n, 12, 5);
    const result = kmeans(scores, n, k, 4, 2);
    const seen = new Set(result.labels);
    expect(result.labels.length).toBe(n);
    expect(seen.size).toBe(result.k);
    for (let c = 0; c < result.k; c++) expect(seen.has(c)).toBe(true);
  });

  it("is deterministic for a given seed", () => {
    const { x, n } = blobs(30, 3, 10, 0.5);
    const { scores, k } = pca(x, n, 10, 5);
    expect(Array.from(kmeans(scores, n, k, 3, 11).labels))
      .toEqual(Array.from(kmeans(scores, n, k, 3, 11).labels));
  });

  /**
   * k-means++ seeding earns its place here: uniform seeding regularly puts two
   * centres in one blob and never finds another, which reads as a clustering
   * that simply missed something.
   */
  it("finds every blob rather than splitting one and missing another", () => {
    const { x, truth, n } = blobs(30, 5, 20, 0.5);
    const { scores, k } = pca(x, n, 20, 8);
    for (const seed of [1, 2, 3, 4, 5]) {
      const result = kmeans(scores, n, k, 5, seed);
      expect(result.k).toBe(5);
      expect(agrees(result.labels, truth)).toBe(true);
    }
  });

  it("asks for more clusters than points without falling over", () => {
    const { x, n } = blobs(2, 2, 6, 0.3);
    const result = kmeans(pca(x, n, 6, 2).scores, n, 2, 99, 1);
    expect(result.labels.length).toBe(n);
    expect(result.k).toBeLessThanOrEqual(n);
  });

  it("gets tighter as k rises", () => {
    const { x, n } = blobs(30, 4, 16, 0.8);
    const { scores, k } = pca(x, n, 16, 6);
    expect(kmeans(scores, n, k, 8, 5).inertia)
      .toBeLessThanOrEqual(kmeans(scores, n, k, 2, 5).inertia);
  });
});
