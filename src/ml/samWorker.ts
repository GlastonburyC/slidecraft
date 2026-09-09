/// <reference lib="webworker" />
import * as ort from "onnxruntime-web";
import { fetchWeights, type DownloadProgress } from "./modelCache";
import type { ModelSpec } from "./registry";

/**
 * Prompt-driven segmentation, off the main thread.
 *
 * SAM splits into a heavy image encoder and a tiny prompt decoder. We exploit
 * that split directly: encode an ROI once (~1s), then every click is a decoder
 * pass over the cached embedding (~10ms). That is what makes click-to-segment
 * feel instant, and it mirrors the "embed once, reuse endlessly" principle the
 * whole ML design rests on.
 */

// Self-hosted: the app is cross-origin isolated, which blocks CDN wasm.
ort.env.wasm.wasmPaths = "/ort/";
// Single-threaded on purpose: ORT's threaded wasm spawns nested workers, and
// doing that from inside a module worker hangs indefinitely rather than
// failing. WebGPU carries the parallelism where it is available, and the
// single-threaded wasm path is a correct (if slower) fallback.
ort.env.wasm.numThreads = 1;

type Req =
  | { type: "load"; id: number; spec: ModelSpec }
  | { type: "encode"; id: number; rgba: ArrayBuffer; width: number; height: number }
  | { type: "decode"; id: number; points: [number, number][]; labels: number[] };

type Res =
  | { type: "progress"; id: number; part: string; received: number; total: number }
  | { type: "loaded"; id: number; backend: string; inputs: string[] }
  | { type: "encoded"; id: number; ms: number }
  | { type: "decoded"; id: number; mask: ArrayBuffer; w: number; h: number; score: number; areaFrac: number; ms: number }
  | { type: "error"; id: number; message: string };

const post = (m: Res, transfer: Transferable[] = []) =>
  (self as unknown as Worker).postMessage(m, transfer);

/**
 * Largest share of the encoded field a single prompted object may cover.
 * Generous: a nucleus in a cell-scale view is well under 1%.
 */
const MAX_OBJECT_FRACTION = 0.25;

let encoder: ort.InferenceSession | null = null;
let decoder: ort.InferenceSession | null = null;
let spec: ModelSpec | null = null;

/** Cached per encoded ROI. */
let embeddings: ort.Tensor | null = null;
let posEmbeddings: ort.Tensor | null = null;
/** Scale from ROI pixels to the model's letterboxed input space. */
let promptScale = 1;
let roiW = 0;
let roiH = 0;

let activeBackend = "wasm";

async function createSession(bytes: Uint8Array, prefer: string): Promise<ort.InferenceSession> {
  if (prefer !== "wasm" && "gpu" in navigator) {
    try {
      const s = await ort.InferenceSession.create(bytes, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
      });
      activeBackend = "webgpu";
      return s;
    } catch (err) {
      // Quantized operators are not uniformly covered by the WebGPU EP yet;
      // falling back keeps results correct rather than failing the load.
      console.warn("[slidecraft] WebGPU unavailable for this model, using wasm", err);
    }
  }
  activeBackend = "wasm";
  return ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
}

async function load(id: number, s: ModelSpec) {
  spec = s;
  const onProgress = (p: DownloadProgress) =>
    post({ type: "progress", id, part: p.part, received: p.received, total: p.total });

  const enc = s.files.find((f) => f.part === "encoder")!;
  const dec = s.files.find((f) => f.part === "decoder")!;
  const [encBytes, decBytes] = await Promise.all([
    fetchWeights(enc.url, "encoder", enc.bytes, onProgress),
    fetchWeights(dec.url, "decoder", dec.bytes, onProgress),
  ]);

  encoder = await createSession(encBytes, s.backend);
  decoder = await createSession(decBytes, s.backend);
  post({
    type: "loaded",
    id,
    backend: activeBackend,
    inputs: [...encoder.inputNames, "|", ...decoder.inputNames],
  });
}

/**
 * RGBA ROI -> normalised CHW float tensor, letterboxed into a square.
 * SAM resizes the longest edge to `inputSize` and pads the remainder, so the
 * padding must be zero *after* normalisation, not before.
 */
function preprocess(rgba: Uint8ClampedArray, w: number, h: number, size: number) {
  const scale = size / Math.max(w, h);
  const rw = Math.round(w * scale);
  const rh = Math.round(h * scale);
  const data = new Float32Array(3 * size * size);
  const { mean, std } = spec!;

  for (let y = 0; y < rh; y++) {
    // Nearest-neighbour source row; SAM is not sensitive to the resampler here
    // and this keeps preprocessing off the critical path.
    const sy = Math.min(h - 1, Math.floor(y / scale));
    for (let x = 0; x < rw; x++) {
      const sx = Math.min(w - 1, Math.floor(x / scale));
      const si = (sy * w + sx) * 4;
      const di = y * size + x;
      data[di] = (rgba[si] - mean[0]) / std[0];
      data[size * size + di] = (rgba[si + 1] - mean[1]) / std[1];
      data[2 * size * size + di] = (rgba[si + 2] - mean[2]) / std[2];
    }
  }
  return { tensor: new ort.Tensor("float32", data, [1, 3, size, size]), scale, rw, rh };
}

async function encode(id: number, rgba: Uint8ClampedArray, w: number, h: number) {
  if (!encoder || !spec) throw new Error("model not loaded");
  const t0 = performance.now();
  const { tensor, scale } = preprocess(rgba, w, h, spec.inputSize);
  const out = await encoder.run({ pixel_values: tensor });
  embeddings = out.image_embeddings as ort.Tensor;
  posEmbeddings = out.image_positional_embeddings as ort.Tensor;
  promptScale = scale;
  roiW = w;
  roiH = h;
  post({ type: "encoded", id, ms: performance.now() - t0 });
}

async function decode(id: number, points: [number, number][], labels: number[]) {
  if (!decoder || !embeddings || !posEmbeddings) throw new Error("no ROI encoded");
  const t0 = performance.now();

  // Points arrive in ROI pixels; the decoder wants the letterboxed input space.
  const flat = new Float32Array(points.length * 2);
  points.forEach((p, i) => {
    flat[i * 2] = p[0] * promptScale;
    flat[i * 2 + 1] = p[1] * promptScale;
  });

  const feeds: Record<string, ort.Tensor> = {
    image_embeddings: embeddings,
    image_positional_embeddings: posEmbeddings,
    input_points: new ort.Tensor("float32", flat, [1, 1, points.length, 2]),
    input_labels: new ort.Tensor(
      "int64",
      BigInt64Array.from(labels.map((l) => BigInt(l))),
      [1, 1, labels.length],
    ),
  };

  const out = await decoder.run(feeds);
  const masks = out.pred_masks as ort.Tensor;      // [1,1,C,H,W]
  const iou = out.iou_scores as ort.Tensor;        // [1,1,C]

  const dims = masks.dims as number[];
  const mh = dims[dims.length - 2];
  const mw = dims[dims.length - 1];
  const channels = dims[dims.length - 3];
  const md = masks.data as Float32Array;
  const scores = iou.data as Float32Array;

  // SAM returns whole / part / subpart candidates for one point, and the
  // coarsest of the three frequently scores highest — clicking a nucleus then
  // yields the entire tissue block. Cells occupy a tiny fraction of a
  // cell-scale field, so prefer the best-scoring candidate that is actually
  // small enough to be a cell, and fall back to the smallest if none are.
  const areaFrac: number[] = [];
  for (let c = 0; c < channels; c++) {
    let pos = 0;
    const base0 = c * mh * mw;
    for (let i = 0; i < mh * mw; i++) if (md[base0 + i] > 0) pos++;
    areaFrac.push(pos / (mh * mw));
  }

  const plausible: number[] = [];
  for (let c = 0; c < channels; c++) {
    if (areaFrac[c] > 0.0002 && areaFrac[c] <= MAX_OBJECT_FRACTION) plausible.push(c);
  }
  let best: number;
  if (plausible.length > 0) {
    best = plausible.reduce((a, b) => (scores[b] > scores[a] ? b : a));
  } else {
    best = areaFrac.reduce((a, b, i) => (areaFrac[i] < areaFrac[a] && areaFrac[i] > 0 ? i : a), 0);
  }

  // Crop away the letterbox padding so the mask maps back to the ROI cleanly.
  const validW = Math.max(1, Math.round((roiW * promptScale * mw) / spec!.inputSize));
  const validH = Math.max(1, Math.round((roiH * promptScale * mh) / spec!.inputSize));
  const crop = new Float32Array(validW * validH);
  const base = best * mh * mw;
  for (let y = 0; y < validH; y++) {
    for (let x = 0; x < validW; x++) crop[y * validW + x] = md[base + y * mw + x];
  }

  post(
    {
      type: "decoded", id, mask: crop.buffer, w: validW, h: validH,
      score: scores[best], areaFrac: areaFrac[best], ms: performance.now() - t0,
    },
    [crop.buffer],
  );
}

self.onmessage = async (ev: MessageEvent<Req>) => {
  const msg = ev.data;
  try {
    if (msg.type === "load") await load(msg.id, msg.spec);
    else if (msg.type === "encode")
      await encode(msg.id, new Uint8ClampedArray(msg.rgba), msg.width, msg.height);
    else if (msg.type === "decode") await decode(msg.id, msg.points, msg.labels);
  } catch (err) {
    post({ type: "error", id: msg.id, message: err instanceof Error ? err.message : String(err) });
  }
};

export type { Req, Res };
