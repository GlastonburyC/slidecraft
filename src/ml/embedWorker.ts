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
/**
 * The element type the graph actually wants.
 *
 * An fp16 export takes fp16 input, and feeding it float32 fails at the first
 * run with "Unexpected input data type" — after the model has downloaded,
 * loaded and been compiled. Rather than assume, or make the sidecar carry yet
 * another thing that can disagree with the graph, ask the graph.
 */
let inputType: "float32" | "float16" = "float32";

const post = (m: Res, transfer: Transferable[] = []) =>
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(m, transfer);

async function load(id: number, next: ModelSpec) {
  const file = next.files[0];
  if (!file) throw new Error(`${next.name} has no weights file`);

  /**
   * The weight buffer is scoped as tightly as possible.
   *
   * A ViT-H at fp16 is 1.27 GB, and creating a session copies it again — so
   * for a moment the tab holds two and a half gigabytes for one model, against
   * a heap limit around four. Everything else in the page is competing for
   * what is left, including the slide decoder's own allocations, and a read
   * that normally takes four milliseconds can be starved for minutes. Letting
   * the bytes go the instant the session exists is the cheapest thing that
   * shortens that window.
   */
  const wanted = next.backend === "wasm" ? ["wasm"] : ["webgpu", "wasm"];
  let backend = "";
  let lastError: unknown = null;
  {
    let bytes: Uint8Array | null = await fetchWeights(
      file.url,
      file.part,
      file.bytes,
      (p: DownloadProgress) =>
        post({ type: "progress", id, part: p.part, received: p.received, total: p.total }),
    );
    for (const ep of wanted) {
      try {
        session = await ort.InferenceSession.create(bytes, { executionProviders: [ep] });
        backend = ep;
        break;
      } catch (err) {
        lastError = err;
      }
    }
    bytes = null;
  }
  if (!session) {
    throw new Error(
      `Could not start ${next.name} on ${wanted.join(" or ")}: ` +
        (lastError instanceof Error ? lastError.message : String(lastError)),
    );
  }

  spec = next;
  inputName = session.inputNames[0] ?? "pixel_values";
  const meta = session.inputMetadata?.find((m) => m.name === inputName);
  const declared = (meta as { type?: string } | undefined)?.type;
  inputType = declared === "float16" ? "float16" : "float32";

  // Probe the real output width rather than trusting the registry: a shard of
  // cached vectors is keyed by dimension, and writing 1536-d vectors into a
  // 2560-d shard would be a silent, permanent corruption of the cache.
  const probeDims = [1, 3, next.inputSize, next.inputSize];
  const probe =
    inputType === "float16"
      ? new ort.Tensor("float16", new Uint16Array(3 * next.inputSize * next.inputSize), probeDims)
      : new ort.Tensor("float32", new Float32Array(3 * next.inputSize * next.inputSize), probeDims);
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

/** float32 to IEEE half, as the bit pattern ORT expects in a Uint16Array. */
function toHalf(value: number): number {
  f32[0] = value;
  const bits = u32[0];
  const sign = (bits >>> 16) & 0x8000;
  let exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  const fraction = (bits >>> 13) & 0x3ff;
  if (exponent <= 0) return sign; // underflows to signed zero
  if (exponent >= 31) return sign | 0x7c00; // overflows to infinity
  return sign | (exponent << 10) | fraction;
}
const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/** IEEE half bit pattern back to a double. */
function fromHalf(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits & 0x7c00) >> 10;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

/**
 * Bring an output to float32, whatever the runtime handed back.
 *
 * Only a Uint16Array holds raw half bit patterns that need decoding. An fp16
 * output may instead arrive as an array whose values are already floats, and
 * reinterpreting those as bit patterns is catastrophic and quiet: nearly every
 * value decodes to zero, a few to subnormals, some to NaN. The embeddings look
 * like embeddings — right length, right count — and carry no information, so
 * every patch clusters together and every head trained on them is noise.
 *
 * The type is therefore checked rather than inferred from the model's
 * declared precision.
 */
function widen(data: ArrayLike<number>): Float32Array<ArrayBuffer> {
  // A fresh, unshared buffer either way: the result is transferred to the main
  // thread, and a view onto ORT's own (possibly shared) heap cannot be.
  const out = new Float32Array(data.length);
  if (data instanceof Uint16Array) {
    for (let i = 0; i < data.length; i++) out[i] = fromHalf(data[i]);
  } else {
    out.set(data as ArrayLike<number> & Iterable<number>);
  }
  return out as Float32Array<ArrayBuffer>;
}

function toTensor(tiles: Uint8ClampedArray, size: number, count: number, s: ModelSpec) {
  const plane = size * size;
  const values = new Float32Array(count * 3 * plane);
  for (let n = 0; n < count; n++) {
    const src = n * plane * 4;
    const dst = n * 3 * plane;
    for (let i = 0; i < plane; i++) {
      values[dst + i] = (tiles[src + i * 4] - s.mean[0]) / s.std[0];
      values[dst + plane + i] = (tiles[src + i * 4 + 1] - s.mean[1]) / s.std[1];
      values[dst + 2 * plane + i] = (tiles[src + i * 4 + 2] - s.mean[2]) / s.std[2];
    }
  }
  const dims = [count, 3, size, size];
  if (inputType === "float32") return new ort.Tensor("float32", values, dims);

  // Normalise in float32 and narrow at the end: doing the arithmetic in half
  // precision would lose more than the storage does.
  const half = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) half[i] = toHalf(values[i]);
  return new ort.Tensor("float16", half, dims);
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
      // An fp16 graph answers in fp16. Everything downstream — the cache, the
      // head, the standardisation — is float32, so widen here rather than
      // letting half-precision leak into the rest of the app.
      const copy = widen(out[session.outputNames[0]].data as ArrayLike<number>);

      // An encoder that answers with nothing is not detectable downstream: the
      // vectors are the right shape, every patch looks identical, and the
      // clustering and the head both quietly become noise.
      let finite = 0;
      for (let i = 0; i < Math.min(copy.length, 256); i++) if (Number.isFinite(copy[i]) && copy[i] !== 0) finite++;
      if (finite === 0) {
        throw new Error(
          "The encoder returned all zeros or NaN. Its output type was " +
            `${(out[session.outputNames[0]].data as object).constructor.name}, which this build ` +
            "may be decoding wrongly — please report it.",
        );
      }
      post(
        { type: "embedded", id: msg.id, vectors: copy.buffer, dim, ms: performance.now() - started },
        [copy.buffer],
      );
    }
  } catch (err) {
    post({ type: "error", id: msg.id, message: err instanceof Error ? err.message : String(err) });
  }
};
