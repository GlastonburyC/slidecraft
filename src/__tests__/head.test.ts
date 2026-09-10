import { describe, expect, it } from "vitest";
import { blockOf, NotEnoughLabels, predict, trainHead, type LabelledSet } from "../ml/head";

/**
 * A separable set: each class sits in its own corner of a low-dimensional
 * embedding space, with noise. Anything that cannot fit this is broken.
 */
function makeSet(
  perClass: number,
  dim: number,
  classes: number,
  spread: number,
  layout: (i: number, k: number) => [number, number],
): LabelledSet {
  const count = perClass * classes;
  const x = new Float32Array(count * dim);
  const y = new Uint8Array(count);
  const px = new Float64Array(count);
  const py = new Float64Array(count);

  let n = 0;
  for (let k = 0; k < classes; k++) {
    for (let i = 0; i < perClass; i++) {
      // Deterministic pseudo-noise: reproducible, and no dependency on a seed.
      const noise = (j: number) => (((n * 9301 + j * 49297) % 233280) / 233280 - 0.5) * spread;
      for (let f = 0; f < dim; f++) x[n * dim + f] = (f === k ? 1 : 0) + noise(f);
      y[n] = k;
      const [a, b] = layout(i, k);
      px[n] = a;
      py[n] = b;
      n++;
    }
  }
  return { x, dim, y, px, py, count };
}

const classes = ["Tumour", "Stroma", "Immune"];

describe("training a head on frozen embeddings", () => {
  it("learns a separable problem and predicts the right class", () => {
    // Classes spread across many blocks, so validation has something to hold out.
    const set = makeSet(60, 8, 3, 0.3, (i, k) => [(i % 10) * 2048, (k * 4 + Math.floor(i / 10)) * 2048]);
    const head = trainHead(set, classes, "uni2-h", 2048);

    for (let k = 0; k < 3; k++) {
      const vector = new Float32Array(8);
      vector[k] = 1;
      const p = predict(head, vector);
      const best = p.indexOf(Math.max(...p));
      expect(best).toBe(k);
      // Probabilities, so they sum to one.
      expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    }
  });

  it("reports a held-out score across spatial blocks", () => {
    const set = makeSet(60, 8, 3, 0.3, (i, k) => [(i % 10) * 2048, (k * 4 + Math.floor(i / 10)) * 2048]);
    const head = trainHead(set, classes, "uni2-h", 2048);

    expect(head.metrics).not.toBe(null);
    expect(head.metrics!.blocks).toBeGreaterThan(0);
    expect(head.metrics!.accuracy).toBeGreaterThan(0.8);
    expect(head.metrics!.confusion.length).toBe(3);
    // Every held-out sample lands somewhere in the confusion matrix.
    const total = head.metrics!.confusion.flat().reduce((a, b) => a + b, 0);
    expect(total).toBe(head.metrics!.heldOut);
  });

  /**
   * The point of spatial blocks. With everything in one block there is nothing
   * to hold out, so no score is reported — rather than a random split reporting
   * a number inflated by neighbouring patches being near-copies.
   */
  it("reports no score when the labels occupy too few blocks", () => {
    const set = makeSet(40, 8, 3, 0.3, () => [0, 0]);
    const head = trainHead(set, classes, "uni2-h", 2048);
    expect(head.metrics).toBe(null);
  });

  it("refuses to train when a class has no labelled patches", () => {
    const set = makeSet(40, 8, 2, 0.3, (i, k) => [(i % 8) * 2048, k * 2048]);
    // Three class names, only two present in the data.
    expect(() => trainHead(set, classes, "uni2-h", 2048)).toThrow(NotEnoughLabels);
    expect(() => trainHead(set, classes, "uni2-h", 2048)).toThrow(/Immune/);
  });

  it("refuses a single class", () => {
    const set = makeSet(40, 8, 1, 0.3, (i) => [i * 2048, 0]);
    expect(() => trainHead(set, ["Only"], "uni2-h", 2048)).toThrow(NotEnoughLabels);
  });

  /**
   * A study area is mostly one thing. Without class balancing the head answers
   * with that thing everywhere and still scores well on accuracy, which is why
   * the rare class's recall is what this checks.
   */
  it("still learns a rare class from an imbalanced set", () => {
    const dim = 6;
    const counts = [200, 200, 12];
    const total = counts.reduce((a, b) => a + b, 0);
    const x = new Float32Array(total * dim);
    const y = new Uint8Array(total);
    const px = new Float64Array(total);
    const py = new Float64Array(total);

    let n = 0;
    counts.forEach((many, k) => {
      for (let i = 0; i < many; i++) {
        for (let f = 0; f < dim; f++) x[n * dim + f] = f === k ? 1 : 0;
        y[n] = k;
        px[n] = (n % 12) * 2048;
        py[n] = Math.floor(n / 12) * 2048;
        n++;
      }
    });

    const head = trainHead({ x, dim, y, px, py, count: total }, classes, "uni2-h", 2048);
    const rare = new Float32Array(dim);
    rare[2] = 1;
    const p = predict(head, rare);
    expect(p.indexOf(Math.max(...p))).toBe(2);
  });

  it("records what it was trained on, so a head cannot outlive its encoder", () => {
    const set = makeSet(40, 8, 3, 0.3, (i, k) => [(i % 8) * 2048, k * 2048]);
    const head = trainHead(set, classes, "virchow2", 2048);
    expect(head.encoderId).toBe("virchow2");
    expect(head.dim).toBe(8);
    expect(head.samples).toEqual([40, 40, 40]);
  });
});

describe("spatial blocks", () => {
  it("puts nearby patches in the same block and distant ones apart", () => {
    expect(blockOf(100, 100, 2048)).toBe(blockOf(2000, 2000, 2048));
    expect(blockOf(100, 100, 2048)).not.toBe(blockOf(5000, 100, 2048));
    expect(blockOf(0, 0, 2048)).toBe("0,0");
    expect(blockOf(4096, 2048, 2048)).toBe("2,1");
  });
});
