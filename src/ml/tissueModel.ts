import { FEATURE_COUNT, FEATURE_NAMES, FEATURE_VERSION } from "./tissueFeatures";

/**
 * A tissue / not-tissue classifier, learned from what you labelled.
 *
 * Logistic regression on a dozen per-cell features, not a network. The reason
 * is the loop rather than the accuracy ceiling: it fits in milliseconds on a
 * few thousand cells, so correcting a detection and seeing the corrected result
 * is one action rather than a wait; it is a few hundred bytes, so it saves and
 * reloads instantly; and its weights are readable, so when it does something
 * surprising you can see which feature drove it. A slide's own artefacts are
 * mostly linearly separable in this feature space once texture is included —
 * and when they are not, the honest answer is more labels, not more capacity.
 *
 * The weights are stored standardised: training centres and scales each feature
 * first, and the centring is kept with the weights so prediction applies the
 * identical transform. Getting that wrong is silent — the model still returns
 * confident probabilities, just meaningless ones.
 */

export interface TissueModel {
  id: string;
  name: string;
  /** Feature layout these weights were fitted against. */
  featureVersion: number;
  weights: number[];
  bias: number;
  /** Per-feature standardisation captured at training time. */
  mean: number[];
  std: number[];
  trainedAt: string;
  samples: { tissue: number; artefact: number };
  /** Held-out scores, so the UI never has to claim a quality it did not measure. */
  metrics: Metrics | null;
  /** Slides whose labels went into this model, for provenance. */
  slides: string[];
}

export interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  heldOut: number;
}

export interface TrainingSet {
  /** Row-major features, `FEATURE_COUNT` per sample. */
  x: Float32Array;
  /** 1 = tissue, 0 = artefact. */
  y: Uint8Array;
  /**
   * Which labelled region each sample came from.
   *
   * Validation holds out whole regions, never individual cells: neighbouring
   * cells in one smear are near-copies of each other, so a random split leaks
   * the answer across it and reports a score far better than the model earns.
   */
  group: Int32Array;
  count: number;
}

const EPOCHS = 400;
const L2 = 1e-3;

function standardise(set: TrainingSet): { mean: number[]; std: number[] } {
  const mean = new Array<number>(FEATURE_COUNT).fill(0);
  const std = new Array<number>(FEATURE_COUNT).fill(0);
  for (let i = 0; i < set.count; i++) {
    for (let f = 0; f < FEATURE_COUNT; f++) mean[f] += set.x[i * FEATURE_COUNT + f];
  }
  for (let f = 0; f < FEATURE_COUNT; f++) mean[f] /= Math.max(1, set.count);
  for (let i = 0; i < set.count; i++) {
    for (let f = 0; f < FEATURE_COUNT; f++) {
      const d = set.x[i * FEATURE_COUNT + f] - mean[f];
      std[f] += d * d;
    }
  }
  for (let f = 0; f < FEATURE_COUNT; f++) {
    // A feature that never varies contributes nothing; a floor of 1 keeps it
    // from dividing by ~0 and turning rounding noise into a huge input.
    std[f] = Math.max(1, Math.sqrt(std[f] / Math.max(1, set.count)));
  }
  return { mean, std };
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));

function fit(
  set: TrainingSet,
  rows: number[],
  mean: number[],
  std: number[],
): { weights: number[]; bias: number } {
  const w = new Array<number>(FEATURE_COUNT).fill(0);
  let b = 0;

  // Class weights, because a few painted artefacts against a whole tissue
  // section is a 50:1 imbalance and an unweighted fit answers "tissue" always.
  let pos = 0;
  for (const i of rows) if (set.y[i]) pos++;
  const neg = rows.length - pos;
  const wPos = pos ? rows.length / (2 * pos) : 0;
  const wNeg = neg ? rows.length / (2 * neg) : 0;

  const gw = new Array<number>(FEATURE_COUNT).fill(0);
  const vw = new Array<number>(FEATURE_COUNT).fill(0);
  let vb = 0;
  const lr = 0.5;
  const momentum = 0.9;

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    gw.fill(0);
    let gb = 0;
    let total = 0;
    for (const i of rows) {
      const base = i * FEATURE_COUNT;
      let z = b;
      for (let f = 0; f < FEATURE_COUNT; f++) {
        z += w[f] * ((set.x[base + f] - mean[f]) / std[f]);
      }
      const p = sigmoid(z);
      const weight = set.y[i] ? wPos : wNeg;
      const err = (p - set.y[i]) * weight;
      for (let f = 0; f < FEATURE_COUNT; f++) {
        gw[f] += err * ((set.x[base + f] - mean[f]) / std[f]);
      }
      gb += err;
      total += weight;
    }
    const scale = 1 / Math.max(1e-9, total);
    for (let f = 0; f < FEATURE_COUNT; f++) {
      const g = gw[f] * scale + L2 * w[f];
      vw[f] = momentum * vw[f] - lr * g;
      w[f] += vw[f];
    }
    vb = momentum * vb - lr * (gb * scale);
    b += vb;
  }
  return { weights: w, bias: b };
}

function score(
  set: TrainingSet, rows: number[],
  weights: number[], bias: number, mean: number[], std: number[],
): Metrics {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (const i of rows) {
    const base = i * FEATURE_COUNT;
    let z = bias;
    for (let f = 0; f < FEATURE_COUNT; f++) {
      z += weights[f] * ((set.x[base + f] - mean[f]) / std[f]);
    }
    const pred = sigmoid(z) >= 0.5 ? 1 : 0;
    if (pred && set.y[i]) tp++;
    else if (pred) fp++;
    else if (set.y[i]) fn++;
    else tn++;
  }
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  return {
    accuracy: rows.length ? (tp + tn) / rows.length : 0,
    precision,
    recall,
    f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0,
    heldOut: rows.length,
  };
}

export class NotEnoughLabels extends Error {}

/**
 * Fit a model, and measure it on regions it never saw.
 *
 * Both classes are required. A model trained only on tissue has no idea what
 * an artefact looks like and will happily call the whole slide tissue, which
 * is worse than the heuristic it replaces because it looks like it was taught.
 */
export function trainTissueModel(
  set: TrainingSet,
  meta: { name: string; slides: string[] },
): TissueModel {
  let pos = 0;
  const groups = new Set<number>();
  for (let i = 0; i < set.count; i++) {
    if (set.y[i]) pos++;
    groups.add(set.group[i]);
  }
  if (pos === 0) throw new NotEnoughLabels("Label some tissue before training.");
  if (pos === set.count) {
    throw new NotEnoughLabels("Label some artefacts too — mark a few wrong detections.");
  }

  const { mean, std } = standardise(set);
  const all = Array.from({ length: set.count }, (_, i) => i);

  /**
   * Hold out whole regions, a share of each class.
   *
   * A class with fewer than three regions contributes none: holding one of two
   * out costs half the examples of that class to measure a score on a single
   * region, which is neither a good fit nor a meaningful measurement. The
   * result is no score at all, which is the honest report — better than a
   * number computed from one region and read as if it meant something.
   */
  const ids = [...groups].sort((a, b) => a - b);
  const holdOut = new Set<number>();
  const byClass = new Map<number, number>();
  for (let i = 0; i < set.count; i++) byClass.set(set.group[i], set.y[i]);
  for (const label of [0, 1]) {
    const mine = ids.filter((g) => byClass.get(g) === label);
    if (mine.length < 3) continue;
    // Every third region, deterministically, so a retrain is reproducible.
    for (let k = 2; k < mine.length; k += 3) holdOut.add(mine[k]);
  }

  const trainRows = all.filter((i) => !holdOut.has(set.group[i]));
  const testRows = all.filter((i) => holdOut.has(set.group[i]));
  const { weights, bias } = fit(set, trainRows.length ? trainRows : all, mean, std);

  // Precision and recall are undefined when one class is missing from the
  // held-out set, and reporting them anyway prints a confident 0. With only a
  // handful of labelled regions that is the normal case, so say nothing rather
  // than something false: no score, and the UI asks for more regions.
  const heldOutClasses = new Set(testRows.map((i) => set.y[i]));
  const measurable = testRows.length > 0 && heldOutClasses.size === 2;

  return {
    id: `tissue-${Date.now().toString(36)}`,
    name: meta.name,
    featureVersion: FEATURE_VERSION,
    weights,
    bias,
    mean,
    std,
    trainedAt: new Date().toISOString(),
    samples: { tissue: pos, artefact: set.count - pos },
    metrics: measurable ? score(set, testRows, weights, bias, mean, std) : null,
    slides: meta.slides,
  };
}

/** Probability that cell `i` of a feature matrix is tissue. */
export function predictCell(model: TissueModel, features: Float32Array, i: number): number {
  const base = i * FEATURE_COUNT;
  let z = model.bias;
  for (let f = 0; f < FEATURE_COUNT; f++) {
    z += model.weights[f] * ((features[base + f] - model.mean[f]) / model.std[f]);
  }
  return sigmoid(z);
}

/** Probability for every cell at once. */
export function predictAll(model: TissueModel, features: Float32Array, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = predictCell(model, features, i);
  return out;
}

/** Which features the model actually leans on, strongest first. */
export function explain(model: TissueModel): { name: string; weight: number }[] {
  return FEATURE_NAMES.map((name, f) => ({ name, weight: model.weights[f] }))
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
}

export function isUsable(model: TissueModel): boolean {
  return model.featureVersion === FEATURE_VERSION && model.weights.length === FEATURE_COUNT;
}
