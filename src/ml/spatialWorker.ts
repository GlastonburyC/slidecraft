/// <reference lib="webworker" />
import * as ort from "onnxruntime-web";
import { fetchWeights, type DownloadProgress } from "./modelCache";
import type { ModelSpec } from "./registry";

/**
 * Virtual spatial transcriptomics, off the main thread.
 *
 * Each patch is one forward pass returning a value per gene, and a grid is
 * hundreds of patches — so this runs in a worker and reports as it goes. There
 * is no encode/decode split to exploit here as there is with SAM: the model
 * reads a tile and answers, and the only lever on cost is how many genes the
 * export was built for. That is why the exporter bakes in a gene subset.
 */

ort.env.wasm.wasmPaths = "/ort/";
// See samWorker: ORT's threaded wasm spawns nested workers, which hangs when
// started from inside a module worker rather than failing.
ort.env.wasm.numThreads = 1;

type Req =
  | { type: "load"; id: number; spec: ModelSpec }
  | {
      type: "predict";
      id: number;
      /** Packed RGBA tiles, each `size * size * 4` bytes, back to back. */
      tiles: ArrayBuffer;
      size: number;
      count: number;
    };

export type Res =
  | { type: "progress"; id: number; part: string; received: number; total: number }
  | { type: "loaded"; id: number; backend: string; genes: number }
  | {
      type: "predicted";
      id: number;
      /** count * genes floats, row-major by tile. */
      values: ArrayBuffer;
      genes: number;
      ms: number;
    }
  | { type: "error"; id: number; message: string };

let session: ort.InferenceSession | null = null;
let spec: ModelSpec | null = null;
let inputName = "pixel_values";

const post = (m: Res, transfer: Transferable[] = []) =>
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(m, transfer);

async function load(id: number, next: ModelSpec) {
  const file = next.files[0];
  if (!file) throw new Error(`${next.name} has no weights file`);

  const bytes = await fetchWeights(file.url, file.part, file.bytes, (p: DownloadProgress) =>
    post({ type: "progress", id, part: p.part, received: p.received, total: p.total }),
  );

  const providers = next.backend === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];
  session = await ort.InferenceSession.create(bytes, { executionProviders: providers });
  spec = next;
  inputName = session.inputNames[0] ?? "pixel_values";
  post({ type: "loaded", id, backend: providers[0], genes: next.genes?.length ?? 0 });
}

/**
 * RGBA tiles to the NCHW float tensor the model expects.
 *
 * Normalisation comes from the export's own sidecar rather than assumed
 * ImageNet statistics: a wrong mean does not fail, it just shifts every
 * prediction, which is indistinguishable from the model being poor.
 */
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

    if (msg.type === "predict") {
      if (!session || !spec) throw new Error("No spatial model is loaded");
      const started = performance.now();
      const tiles = new Uint8ClampedArray(msg.tiles);
      const tensor = toTensor(tiles, msg.size, msg.count, spec);

      const out = await session.run({ [inputName]: tensor });
      const first = out[session.outputNames[0]];
      const values = first.data as Float32Array;
      const genes = values.length / msg.count;

      const copy = new Float32Array(values);
      post(
        {
          type: "predicted",
          id: msg.id,
          values: copy.buffer,
          genes,
          ms: performance.now() - started,
        },
        [copy.buffer],
      );
    }
  } catch (err) {
    post({ type: "error", id: msg.id, message: err instanceof Error ? err.message : String(err) });
  }
};
