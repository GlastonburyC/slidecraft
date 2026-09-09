import { openDB, type IDBPDatabase } from "idb";
import type { ModelFile, ModelSpec } from "./registry";

/**
 * Bring-your-own ONNX models.
 *
 * Licence-gated encoders (UNI, Virchow2, CONCH, GigaPath) cannot be shipped or
 * fetched from a public URL, so they are imported from disk instead. The
 * weights go into the same Cache Storage the downloaded models use, addressed
 * by a synthetic `slidecraft-local:` URL — which means the loader needs no
 * special case: it looks the URL up in the cache and finds it there.
 *
 * Only the metadata lives in IndexedDB; the weights are far too large for it.
 */

const DB_NAME = "slidecraft-models";
const STORE = "specs";
const CACHE_NAME = "slidecraft-models-v1";

export const LOCAL_SCHEME = "slidecraft-local:";

export type NormalisationPreset = "imagenet" | "sam" | "none";

const PRESETS: Record<
  NormalisationPreset,
  { mean: [number, number, number]; std: [number, number, number] }
> = {
  // Rescale 1/255 then ImageNet statistics — what SamImageProcessor and most
  // timm histology encoders use.
  imagenet: {
    mean: [0.485 * 255, 0.456 * 255, 0.406 * 255],
    std: [0.229 * 255, 0.224 * 255, 0.225 * 255],
  },
  // Original SAM: the same numbers, kept separate so the intent is explicit.
  sam: {
    mean: [123.675, 116.28, 103.53],
    std: [58.395, 57.12, 57.375],
  },
  none: { mean: [0, 0, 0], std: [255, 255, 255] },
};

export interface ImportRequest {
  name: string;
  /** Vision encoder weights. */
  encoder: File;
  /** Prompt decoder. Omitted means borrow `fallbackDecoder`. */
  decoder?: File;
  fallbackDecoder?: ModelFile;
  inputSize: number;
  preset: NormalisationPreset;
  licence?: string;
}

function db(): Promise<IDBPDatabase> {
  return openDB(DB_NAME, 1, {
    upgrade(d) {
      if (!d.objectStoreNames.contains(STORE)) d.createObjectStore(STORE, { keyPath: "id" });
    },
  });
}

/** Human-readable reason the import cannot proceed, or null if it can. */
export function validateImport(req: Partial<ImportRequest>): string | null {
  if (!req.name?.trim()) return "Give the model a name.";
  if (!req.encoder) return "Choose an encoder .onnx file.";
  for (const f of [req.encoder, req.decoder]) {
    if (!f) continue;
    if (!/\.onnx$/i.test(f.name)) return `${f.name} is not a .onnx file.`;
    if (f.size < 1024) return `${f.name} looks empty.`;
  }
  if (!req.decoder && !req.fallbackDecoder) {
    return "Choose a decoder .onnx file, or borrow one from a built-in model.";
  }
  const size = req.inputSize ?? 0;
  if (!Number.isInteger(size) || size < 64 || size > 4096) {
    return "Input size must be a whole number between 64 and 4096.";
  }
  return null;
}

const slug = (s: string) =>
  s.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "model";

/** Put a File into the weight cache under a stable synthetic URL. */
async function stash(url: string, file: File): Promise<void> {
  const cache = await caches.open(CACHE_NAME);
  await cache.put(
    url,
    new Response(await file.arrayBuffer(), {
      headers: {
        "content-type": "application/octet-stream",
        "content-length": String(file.size),
      },
    }),
  );
}

export async function importLocalModel(req: ImportRequest): Promise<ModelSpec> {
  const problem = validateImport(req);
  if (problem) throw new Error(problem);

  const id = `local-${slug(req.name)}-${Date.now().toString(36)}`;
  const encoderUrl = `${LOCAL_SCHEME}//${id}/encoder.onnx`;
  await stash(encoderUrl, req.encoder);

  const files: ModelFile[] = [{ part: "encoder", url: encoderUrl, bytes: req.encoder.size }];

  if (req.decoder) {
    const decoderUrl = `${LOCAL_SCHEME}//${id}/decoder.onnx`;
    await stash(decoderUrl, req.decoder);
    files.push({ part: "decoder", url: decoderUrl, bytes: req.decoder.size });
  } else {
    // Borrow a built-in decoder: SAM variants share one decoder architecture,
    // so a swapped encoder usually only needs its own weights.
    files.push({ ...req.fallbackDecoder!, part: "decoder" });
  }

  const { mean, std } = PRESETS[req.preset];
  const spec: ModelSpec = {
    id,
    name: req.name.trim(),
    task: "prompt-segment",
    blurb: `Imported from ${req.encoder.name}${
      req.decoder ? ` + ${req.decoder.name}` : " (built-in decoder)"
    }.`,
    files,
    inputSize: req.inputSize,
    mean,
    std,
    targetMpp: null,
    licence: req.licence?.trim() || "Supplied by you — check your own licence terms.",
    // Quantized weights are unreliable on the WebGPU EP and an imported model's
    // precision is unknown, so start on the backend that is always correct.
    backend: "wasm",
  };

  await (await db()).put(STORE, spec);
  return spec;
}

export async function loadLocalModels(): Promise<ModelSpec[]> {
  try {
    const specs = (await (await db()).getAll(STORE)) as ModelSpec[];
    // A spec whose weights were evicted from the cache is dead metadata.
    const cache = await caches.open(CACHE_NAME);
    const alive: ModelSpec[] = [];
    for (const s of specs) {
      const local = s.files.filter((f) => f.url.startsWith(LOCAL_SCHEME));
      const present = await Promise.all(local.map((f) => cache.match(f.url)));
      if (present.every(Boolean)) alive.push(s);
    }
    return alive;
  } catch {
    return [];
  }
}

export async function removeLocalModel(id: string): Promise<void> {
  try {
    const d = await db();
    const spec = (await d.get(STORE, id)) as ModelSpec | undefined;
    await d.delete(STORE, id);
    if (spec) {
      const cache = await caches.open(CACHE_NAME);
      await Promise.all(
        spec.files.filter((f) => f.url.startsWith(LOCAL_SCHEME)).map((f) => cache.delete(f.url)),
      );
    }
  } catch {
    /* nothing to remove */
  }
}

export const isLocalModel = (m: ModelSpec) => m.id.startsWith("local-");
