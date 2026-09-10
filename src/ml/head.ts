/**
 * A small trainable head over frozen patch embeddings.
 *
 * Softmax regression, not a network, and the reason is the loop rather than the
 * accuracy ceiling. The encoder has already done the representation learning;
 * what is left is a linear read-out over a few hundred labelled patches, which
 * fits in well under a second on typed arrays. That is what makes "correct the
 * prediction and retrain" a single action instead of a wait — and a wait is the
 * difference between iterating twenty times and iterating twice.
 *
 * Nothing here is specific to histology. It is a multinomial logistic
 * regression with L2, class balancing, and validation that holds out whole
 * spatial blocks.
 */

export interface LabelledSet {
  /** Row-major features, `dim` per sample. */
  x: Float32Array;
  dim: number;
  /** Class index per sample. */
  y: Uint8Array;
  /** Level-0 pixel position of each sample, for spatial validation. */
  px: Float64Array;
  py: Float64Array;
  /**
   * Validation group per sample, when there is a better one than geography —
   * the ROI a patch came from. Training on one region and testing on another
   * is the only split that answers the question actually being asked: does
   * this transfer to tissue it has not seen?
   */
  block?: Int32Array;
  count: number;
}

export interface HeadMetrics {
  accuracy: number;
  /** Per class, in class order. */
  precision: number[];
  recall: number[];
  f1: number[];
  /** `confusion[t][p]` — true class t predicted as p. */
  confusion: number[][];
  heldOut: number;
  blocks: number;
}

export interface Head {
  classes: string[];
  dim: number;
  /** `classes.length * dim`, row-major by class. */
  weights: Float32Array;
  bias: Float32Array;
  /** Standardisation captured at training time. */
  mean: Float32Array;
  scale: Float32Array;
  metrics: HeadMetrics | null;
  samples: number[];
  trainedAt: string;
  encoderId: string;
}

export class NotEnoughLabels extends Error {}

const EPOCHS = 220;
const LR = 0.6;
const MOMENTUM = 0.9;

/**
 * Regularisation, scaled to how over-determined the fit is.
 *
 * A foundation model's embedding is 1536 or 2560 numbers wide, and a first
 * pass at annotating gives perhaps twenty labelled patches. A fixed, small
 * penalty lets a head with that many free parameters separate twenty points
 * perfectly and generalise at chance — while reporting 99% confidence,
 * because the margin it found is enormous and meaningless. That is worse than
 * being wrong: the confidence slider stops doing anything, since every patch
 * is above any threshold.
 *
 * Tying the penalty to dimensions-per-sample keeps the fit honest as the ratio
 * changes, and it relaxes on its own as more patches are labelled.
 */
const L2_BASE = 0.002;
/**
 * Capped as well as scaled. The penalty is applied through the same step and
 * momentum as the gradient, so a coefficient above (1 - momentum) / lr makes
 * the decay exceed the weight each iteration: the sign flips every step and
 * the fit converges to the negative of what it should be. That failure is
 * silent and total — predictions come out exactly inverted.
 */
const L2_MAX = (1 - MOMENTUM) / LR / 2;
const l2For = (dim: number, count: number) =>
  Math.min(L2_MAX, L2_BASE * (dim / Math.max(1, count)));

/**
 * Blocks for validation, in slide space.
 *
 * Neighbouring patches are near-copies, so a random split leaks the answer
 * across it and reports a score the head has not earned — often by a wide
 * margin. Holding out contiguous squares instead measures what is actually
 * wanted: whether the head generalises to tissue it has not seen.
 */
export function blockOf(x: number, y: number, blockPx: number): string {
  return `${Math.floor(x / blockPx)},${Math.floor(y / blockPx)}`;
}

/**
 * Centre and scale on the whole population, not the labelled part of it.
 *
 * The head is applied to every patch, so the statistics have to describe every
 * patch. Taking them from twenty patches drawn from one corner of the slide
 * gives a narrow spread, and everything outside that corner then standardises
 * to a large value and saturates the softmax — confident predictions about
 * exactly the tissue the labels never covered.
 */
function standardise(set: LabelledSet, population?: { x: Float32Array; n: number }) {
  const source = population ?? { x: set.x, n: set.count };
  const mean = new Float32Array(set.dim);
  const scale = new Float32Array(set.dim);
  for (let i = 0; i < source.n; i++) {
    for (let f = 0; f < set.dim; f++) mean[f] += source.x[i * set.dim + f];
  }
  for (let f = 0; f < set.dim; f++) mean[f] /= Math.max(1, source.n);
  for (let i = 0; i < source.n; i++) {
    for (let f = 0; f < set.dim; f++) {
      const d = source.x[i * set.dim + f] - mean[f];
      scale[f] += d * d;
    }
  }
  for (let f = 0; f < set.dim; f++) {
    // A floor rather than the raw deviation: a feature that never varies would
    // otherwise turn rounding noise into a very large input.
    scale[f] = Math.max(1e-3, Math.sqrt(scale[f] / Math.max(1, source.n)));
  }
  return { mean, scale };
}

function softmaxInto(z: Float32Array) {
  let max = -Infinity;
  for (const v of z) if (v > max) max = v;
  let sum = 0;
  for (let k = 0; k < z.length; k++) {
    z[k] = Math.exp(z[k] - max);
    sum += z[k];
  }
  for (let k = 0; k < z.length; k++) z[k] /= sum;
}

function fit(set: LabelledSet, rows: number[], nClasses: number, mean: Float32Array, scale: Float32Array) {
  const { dim } = set;
  const l2 = l2For(dim, rows.length);
  const weights = new Float32Array(nClasses * dim);
  const bias = new Float32Array(nClasses);
  const vW = new Float32Array(nClasses * dim);
  const vB = new Float32Array(nClasses);

  // Class weights, because a study area is mostly one thing and an unweighted
  // fit answers with that thing everywhere.
  const counts = new Float64Array(nClasses);
  for (const i of rows) counts[set.y[i]]++;
  const weightFor = new Float64Array(nClasses);
  for (let k = 0; k < nClasses; k++) {
    weightFor[k] = counts[k] > 0 ? rows.length / (nClasses * counts[k]) : 0;
  }

  const gW = new Float32Array(nClasses * dim);
  const gB = new Float32Array(nClasses);
  const z = new Float32Array(nClasses);
  const scaled = new Float32Array(dim);

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    gW.fill(0);
    gB.fill(0);
    let total = 0;

    for (const i of rows) {
      const base = i * dim;
      for (let f = 0; f < dim; f++) scaled[f] = (set.x[base + f] - mean[f]) / scale[f];

      for (let k = 0; k < nClasses; k++) {
        let acc = bias[k];
        const wBase = k * dim;
        for (let f = 0; f < dim; f++) acc += weights[wBase + f] * scaled[f];
        z[k] = acc;
      }
      softmaxInto(z);

      const truth = set.y[i];
      const w = weightFor[truth];
      total += w;
      for (let k = 0; k < nClasses; k++) {
        const err = (z[k] - (k === truth ? 1 : 0)) * w;
        const wBase = k * dim;
        for (let f = 0; f < dim; f++) gW[wBase + f] += err * scaled[f];
        gB[k] += err;
      }
    }

    const norm = 1 / Math.max(1e-9, total);
    for (let k = 0; k < nClasses * dim; k++) {
      // Both terms go through the learning rate. Decaying the weights at full
      // strength while the gradient is scaled by it is what inverted the fit.
      const grad = gW[k] * norm + l2 * weights[k];
      vW[k] = MOMENTUM * vW[k] - LR * grad;
      weights[k] += vW[k];
    }
    for (let k = 0; k < nClasses; k++) {
      // No penalty on the bias: shrinking it biases every prediction toward
      // the classes that happen to be numerous.
      vB[k] = MOMENTUM * vB[k] - LR * gB[k] * norm;
      bias[k] += vB[k];
    }
  }

  return { weights, bias };
}

/** Class probabilities for one embedding. */
export function predict(head: Head, vector: Float32Array, out?: Float32Array): Float32Array {
  const n = head.classes.length;
  const z = out && out.length === n ? out : new Float32Array(n);
  for (let k = 0; k < n; k++) {
    let acc = head.bias[k];
    const base = k * head.dim;
    for (let f = 0; f < head.dim; f++) {
      acc += head.weights[base + f] * ((vector[f] - head.mean[f]) / head.scale[f]);
    }
    z[k] = acc;
  }
  softmaxInto(z);
  return z;
}

function score(
  set: LabelledSet, rows: number[], head: Head, nClasses: number, blocks: number,
): HeadMetrics {
  const confusion = Array.from({ length: nClasses }, () => new Array<number>(nClasses).fill(0));
  const probs = new Float32Array(nClasses);

  for (const i of rows) {
    const vector = set.x.subarray(i * set.dim, (i + 1) * set.dim);
    predict(head, vector, probs);
    let best = 0;
    for (let k = 1; k < nClasses; k++) if (probs[k] > probs[best]) best = k;
    confusion[set.y[i]][best]++;
  }

  const precision: number[] = [];
  const recall: number[] = [];
  const f1: number[] = [];
  let correct = 0;
  for (let k = 0; k < nClasses; k++) {
    const tp = confusion[k][k];
    correct += tp;
    let predicted = 0;
    let actual = 0;
    for (let j = 0; j < nClasses; j++) {
      predicted += confusion[j][k];
      actual += confusion[k][j];
    }
    const p = predicted ? tp / predicted : 0;
    const r = actual ? tp / actual : 0;
    precision.push(p);
    recall.push(r);
    f1.push(p + r ? (2 * p * r) / (p + r) : 0);
  }

  return {
    accuracy: rows.length ? correct / rows.length : 0,
    precision, recall, f1, confusion,
    heldOut: rows.length,
    blocks,
  };
}

export function trainHead(
  set: LabelledSet,
  classes: string[],
  encoderId: string,
  blockPx: number,
  /** Every embedded patch, so the standardisation describes what is predicted. */
  population?: { x: Float32Array; n: number },
): Head {
  const nClasses = classes.length;
  if (nClasses < 2) throw new NotEnoughLabels("Label at least two classes.");

  const counts = new Array<number>(nClasses).fill(0);
  for (let i = 0; i < set.count; i++) counts[set.y[i]]++;
  const empty = classes.filter((_, k) => counts[k] === 0);
  if (empty.length) {
    throw new NotEnoughLabels(
      `No patches labelled ${empty.join(" or ")}. Draw over some, or drop the class.`,
    );
  }

  const { mean, scale } = standardise(set, population);
  const all = Array.from({ length: set.count }, (_, i) => i);

  // Whole spatial blocks are held out; every third one, deterministically, so
  // a retrain is reproducible.
  const blocks = new Map<string, number>();
  const keyFor = (i: number) =>
    set.block ? String(set.block[i]) : blockOf(set.px[i], set.py[i], blockPx);
  for (let i = 0; i < set.count; i++) {
    const key = keyFor(i);
    if (!blocks.has(key)) blocks.set(key, blocks.size);
  }
  const holdOut = new Set<number>();
  const minGroups = set.block ? 2 : 4;
  if (blocks.size >= minGroups) {
    // With two ROIs, hold the second out entirely; with more, every third.
    const stride = blocks.size === 2 ? 2 : 3;
    for (const [, index] of blocks) if (index % stride === stride - 1) holdOut.add(index);
  }
  const inHoldOut = (i: number) => holdOut.has(blocks.get(keyFor(i))!);

  const trainRows = all.filter((i) => !inHoldOut(i));
  const testRows = all.filter((i) => inHoldOut(i));
  const rows = trainRows.length >= nClasses ? trainRows : all;

  const { weights, bias } = fit(set, rows, nClasses, mean, scale);
  const head: Head = {
    classes, dim: set.dim, weights, bias, mean, scale,
    metrics: null,
    samples: counts,
    trainedAt: new Date().toISOString(),
    encoderId,
  };

  // A held-out set missing a class cannot produce a meaningful score, and
  // reporting one anyway is how a head that has learnt nothing looks fine.
  const heldOutClasses = new Set(testRows.map((i) => set.y[i]));
  if (testRows.length > 0 && heldOutClasses.size === nClasses) {
    head.metrics = score(set, testRows, head, nClasses, holdOut.size);
  }
  return head;
}
