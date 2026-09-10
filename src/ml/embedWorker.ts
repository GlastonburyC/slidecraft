/// <reference lib="webworker" />
import * as ort from "onnxruntime-web";
import { fetchWeights, type DownloadProgress } from "./modelCache";
import type { ModelSpec } from "./registry";

/**
 * Patch embedding, off the main thread.
 *
 * This is the expensive half of the human-in-the-loop cycle and the half that
 * only ever runs once: an encoder's output for a given patch never changes, so
 * everything downstream — labelling, training, correcting, retraining — works
 * on cached vectors and costs milliseconds. Getting this worker's output right
 * matters more than its speed, because a wrong embedding is not detectably
 * wrong, it just makes every head trained on it slightly worse.
 */

ort.env.wasm.wasmPaths = "/ort/";
// See samWorker: ORT's threaded wasm spawns nested workers, which hangs when
// started from inside a module worker rather than failing.
ort.env.wasm.numThreads = 1;

type Req =
  | { type: "load"; id: number; spec: ModelSpec }
  | {
      type: "embed";
      id: number;
      /** Packed RGBA tiles, each `size * size * 4` bytes, back to back. */
      tiles: ArrayBuffer;
      size: number;
      count: number;
    };

export type Res =
  | { type: "progress"; id: number; part: string; received: number; total: number }
  | { type: "loaded"; id: number; backend: string; dim: number }
  | { type: "embedded"; id: number; vectors: ArrayBuffer; dim: number; ms: number }
  | { type: "error"; id: number; message: string };

let session: ort.InferenceSession | null = null;
let spec: ModelSpec | null = null;
let inputName = "pixel_values";
let dim = 0;

const post = (m: Res, transfer: Transferable[] = []) =>
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(m, transfer);

async function load(id: number, next: ModelSpec) {
  const file = next.files[0];
  if (!file) throw new Error(`${next.name} has no weights file`);

  const bytes = await fetchWeights(file.url, file.part, file.bytes, (p: DownloadProgress) =>
    post({ type: "progress", id, part: p.part, received: p.received, total: p.total }),
  );

  const wanted = next.backend === "wasm" ? ["wasm"] : ["webgpu", "wasm"];
  let backend = "";
  let lastError: unknown = null;
  for (const ep of wanted) {
    try {
      session = await ort.InferenceSession.create(bytes, { executionProviders: [ep] });
      backend = ep;
      break;
    } catch (err) {
      lastError = err;
    }
  }
  if (!session) {
    throw new Error(
      `Could not start ${next.name} on ${wanted.join(" or ")}: ` +
        (lastError instanceof Error ? lastError.message : String(lastError)),
    );
  }

  spec = next;
  inputName = session.inputNames[0] ?? "pixel_values";

  // Probe the real output width rather than trusting the registry: a shard of
  // cached vectors is keyed by dimension, and writing 1536-d vectors into a
  // 2560-d shard would be a silent, permanent corruption of the cache.
  const probe = new ort.Tensor(
    "float32",
    new Float32Array(3 * next.inputSize * next.inputSize),
    [1, 3, next.inputSize, next.inputSize],
  );
  const out = await session.run({ [inputName]: probe });
  const first = out[session.outputNames[0]];
  dim = first.dims[first.dims.length - 1];
  if (next.dim && next.dim !== dim) {
    throw new Error(
      `${next.name} declares ${next.dim}-d embeddings but produces ${dim}. ` +
        "Re-export it; the sidecar and the graph disagree.",
    );
  }

  post({ type: "loaded", id, backend, dim });
}

function toTensor(tiles: Uint8ClampedArray, size: number, count: number, s: ModelSpec) {
  const plane = size * size;
  const data = new Float32Array(count * 3 * plane);
  for (let n = 0; n < count; n++) {
    const src = n * plane * 4;
    const dst = n * 3 * plane;
    for (let i = 0; i < plane; i++) {
      data[dst + i] = (tiles[src + i * 4] - s.mean[0]) / s.std[0];
      data[dst + plane + i] = (tiles[src + i * 4 + 1] - s.mean[1]) / s.std[1];
      data[dst + 2 * plane + i] = (tiles[src + i * 4 + 2] - s.mean[2]) / s.std[2];
    }
  }
  return new ort.Tensor("float32", data, [count, 3, size, size]);
}

self.onmessage = async (ev: MessageEvent<Req>) => {
  const msg = ev.data;
  try {
    if (msg.type === "load") {
      await load(msg.id, msg.spec);
      return;
    }

    if (msg.type === "embed") {
      if (!session || !spec) throw new Error("No encoder is loaded");
      const started = performance.now();
      const tiles = new Uint8ClampedArray(msg.tiles);
      const tensor = toTensor(tiles, msg.size, msg.count, spec);

      const out = await session.run({ [inputName]: tensor });
      const values = out[session.outputNames[0]].data as Float32Array;

      const copy = new Float32Array(values);
      post(
        { type: "embedded", id: msg.id, vectors: copy.buffer, dim, ms: performance.now() - started },
        [copy.buffer],
      );
    }
  } catch (err) {
    post({ type: "error", id: msg.id, message: err instanceof Error ? err.message : String(err) });
  }
};
