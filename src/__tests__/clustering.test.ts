import { describe, expect, it } from "vitest";
import { pca } from "../ml/pca";
import { knnGraph } from "../ml/leiden";

/** Deterministic jitter, so a failure is reproducible. */
const noise = (i: number, j: number) => {
  let n = (i * 374761393 + j * 668265263) >>> 0;
  n = ((n ^ (n >>> 13)) * 1274126177) >>> 0;
  return ((n >>> 16) & 1023) / 1023 - 0.5;
};

/** `groups` well-separated blobs in `dim` dimensions. */
function blobs(perGroup: number, groups: number, dim: number, spread: number) {
  const n = perGroup * groups;
  const x = new Float32Array(n * dim);
  const truth = new Int32Array(n);
  let at = 0;
  for (let g = 0; g < groups; g++) {
    for (let i = 0; i < perGroup; i++) {
      for (let c = 0; c < dim; c++) {
        x[at * dim + c] = (c === g ? 8 : 0) + noise(at, c) * spread;
      }
      truth[at] = g;
      at++;
    }
  }
  return { x, truth, n };
}

describe("principal components", () => {
  it("puts the variance in the leading components", () => {
    const { x, n } = blobs(40, 3, 12, 0.4);
    const result = pca(x, n, 12, 5);
    expect(result.k).toBe(5);
    expect(result.scores.length).toBe(n * 5);
    // Descending, and the first few carry nearly everything.
    for (let i = 1; i < result.explained.length; i++) {
      expect(result.explained[i]).toBeLessThanOrEqual(result.explained[i - 1] + 1e-9);
    }
    expect(result.explained.slice(0, 2).reduce((a, b) => a + b, 0)).toBeGreaterThan(0.8);
  });

  it("keeps separated groups separated in component space", () => {
    const { x, truth, n } = blobs(30, 3, 16, 0.4);
    const { scores, k } = pca(x, n, 16, 4);
    // Within-group distance should be far below between-group distance.
    const dist = (i: number, j: number) => {
      let s = 0;
      for (let c = 0; c < k; c++) {
        const d = scores[i * k + c] - scores[j * k + c];
        s += d * d;
      }
      return Math.sqrt(s);
    };
    const within = dist(0, 1);
    const between = dist(0, truth.indexOf(1));
    expect(between).toBeGreaterThan(within * 5);
  });

  it("is deterministic, so a rerun gives the same components", () => {
    const { x, n } = blobs(20, 2, 8, 0.3);
    const a = pca(x, n, 8, 3);
    const b = pca(x, n, 8, 3);
    expect(Array.from(a.scores)).toEqual(Array.from(b.scores));
  });

  it("survives a constant column, which has no variance to normalise by", () => {
    const { x, n } = blobs(20, 2, 6, 0.3);
    for (let i = 0; i < n; i++) x[i * 6 + 5] = 3;
    const result = pca(x, n, 6, 3);
    expect(Array.from(result.scores).every(Number.isFinite)).toBe(true);
  });
});

describe("the neighbour graph", () => {
  it("is symmetric: every edge appears from both ends", () => {
    const { x, n } = blobs(15, 2, 6, 0.3);
    const { scores, k } = pca(x, n, 6, 3);
    const g = knnGraph(scores, n, k, 5);

    for (let i = 0; i < g.n; i++) {
      for (let e = g.offsets[i]; e < g.offsets[i + 1]; e++) {
        const j = g.neighbours[e];
        let back = false;
        for (let f = g.offsets[j]; f < g.offsets[j + 1]; f++) if (g.neighbours[f] === i) back = true;
        expect(back).toBe(true);
      }
    }
  });

  it("never makes a node its own neighbour", () => {
    const { x, n } = blobs(10, 2, 6, 0.3);
    const { scores, k } = pca(x, n, 6, 3);
    const g = knnGraph(scores, n, k, 4);
    for (let i = 0; i < g.n; i++) {
      for (let e = g.offsets[i]; e < g.offsets[i + 1]; e++) expect(g.neighbours[e]).not.toBe(i);
    }
  });

  it("connects each node to at least k others", () => {
    const { x, n } = blobs(20, 2, 6, 0.3);
    const { scores, k } = pca(x, n, 6, 3);
    const g = knnGraph(scores, n, k, 6);
    for (let i = 0; i < g.n; i++) expect(g.offsets[i + 1] - g.offsets[i]).toBeGreaterThanOrEqual(6);
  });
});
