import { openDB, type IDBPDatabase } from "idb";
import { deleteLocalWeights, putLocalWeights } from "./localWeights";
import type { ModelFile, ModelSpec } from "./registry";

/**
 * Bring-your-own ONNX models.
 *
 * Licence-gated encoders (UNI, Virchow2, CONCH, GigaPath) cannot be shipped or
 * fetched from a public URL, so they are imported from disk instead. The
 * weights go into the same Cache Storage the downloaded models use, addressed
 * by a synthetic URL — which means the loader needs no
 * special case: it looks the URL up in the cache and finds it there.
 *
 * Only the metadata lives in IndexedDB; the weights are far too large for it.
 */

const DB_NAME = "slidecraft-models";
const STORE = "specs";
const CACHE_NAME = "slidecraft-models-v1";

/**
 * Where imported weights live, as far as the loader is concerned.
 *
 * It has to be https: the Cache Storage API rejects any other scheme outright,
 * so a custom one like `slidecraft-local:` fails at `put` with a message about
 * schemes that gives no hint of what to do. The host is under `.invalid`,
 * which RFC 2606 reserves and guarantees will never resolve — so if this URL
 * ever reaches the network by mistake, it cannot hit somebody's server.
 */
export const LOCAL_PREFIX = "https://local.slidecraft.invalid/";

/** @deprecated the old, unusable scheme; kept only to recognise stale records. */
export const LOCAL_SCHEME = "slidecraft-local:";

const isLocalUrl = (url: string) => url.startsWith(LOCAL_PREFIX) || url.startsWith(LOCAL_SCHEME);

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
/**
 * Weights go to OPFS, streamed.
 *
 * They used to go to Cache Storage like everything else, via
 * `new Response(await file.arrayBuffer())`. That needs the whole model
 * resident plus a copy for the write — and Cache.put fails outright on a body
 * over a gigabyte, in Chrome with "Unexpected internal error", leaving a spec
 * that points at weights which were never stored.
 */
async function stash(url: string, file: File): Promise<void> {
  await putLocalWeights(url, file);
}

export async function importLocalModel(req: ImportRequest): Promise<ModelSpec> {
  const problem = validateImport(req);
  if (problem) throw new Error(problem);

  const id = `local-${slug(req.name)}-${Date.now().toString(36)}`;
  const encoderUrl = `${LOCAL_PREFIX}${id}/encoder.onnx`;
  await stash(encoderUrl, req.encoder);

  const files: ModelFile[] = [{ part: "encoder", url: encoderUrl, bytes: req.encoder.size }];

  if (req.decoder) {
    const decoderUrl = `${LOCAL_PREFIX}${id}/decoder.onnx`;
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
      const local = s.files.filter((f) => isLocalUrl(f.url));
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
      // Weights live in OPFS now; older imports may still have a Cache entry.
      await deleteLocalWeights(id);
      const cache = await caches.open(CACHE_NAME);
      await Promise.all(
        spec.files.filter((f) => isLocalUrl(f.url)).map((f) => cache.delete(f.url)),
      );
    }
  } catch {
    /* nothing to remove */
  }
}

export const isLocalModel = (m: ModelSpec) => m.id.startsWith("local-");


/**
 * Import a virtual-spatial model: the ONNX graph plus its sidecar.
 *
 * Everything that decides whether a prediction means anything — the gene order,
 * the normalisation, the magnification the model was trained at — comes from
 * the sidecar the exporter wrote, not from anything typed here. A wrong mean or
 * a shifted gene list does not fail; it returns confident numbers about the
 * wrong thing, so the export is the only authority on them.
 */
export interface SpatialImport {
  onnx: File;
  sidecar: File;
}

interface Sidecar {
  name?: string;
  task?: "encode" | "virtual-spatial";
  dim?: number;
  blurb?: string;
  inputSize?: number;
  mean?: [number, number, number];
  std?: [number, number, number];
  targetMpp?: number | null;
  genes?: string[];
  licence?: string;
  source?: string;
  backend?: "wasm" | "webgpu" | "auto";
  precision?: "fp16" | "fp32";
}

/**
 * Import any model the exporters produce, spatial or encoder alike.
 *
 * The sidecar says which it is. Asking the user to pick would be asking them
 * to re-state something the export already knows — and to get it wrong, since
 * a patch encoder and a gene predictor differ only in what their output means.
 */
export async function importSidecarModel(req: SpatialImport): Promise<ModelSpec> {
  if (!req.onnx.name.toLowerCase().endsWith(".onnx")) {
    throw new Error("The model file must be a .onnx graph.");
  }

  let meta: Sidecar;
  try {
    meta = JSON.parse(await req.sidecar.text()) as Sidecar;
  } catch (err) {
    throw new Error(
      `Could not read ${req.sidecar.name}: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  const task = meta.task ?? (meta.genes?.length ? "virtual-spatial" : "encode");
  if (!meta.inputSize) throw new Error("The sidecar does not say what input size the model takes.");

  if (task === "virtual-spatial" && !meta.genes?.length) {
    throw new Error(
      "The sidecar lists no genes. Without them a prediction is a row of numbers with " +
        "nothing to name them — re-export with scripts/export_deepspot.py.",
    );
  }
  if (task === "encode" && !meta.dim) {
    throw new Error(
      "The sidecar does not say how wide this encoder's embeddings are. Cached vectors are " +
        "keyed by width, so re-export it with scripts/export_onnx.py rather than guessing.",
    );
  }

  const id = `local-${task === "encode" ? "encoder" : "spatial"}-${slug(meta.name ?? req.onnx.name)}-${Date.now().toString(36)}`;
  const url = `${LOCAL_PREFIX}${id}/model.onnx`;
  await stash(url, req.onnx);

  const spec: ModelSpec = {
    id,
    name: meta.name?.trim() || req.onnx.name.replace(/\.onnx$/i, ""),
    task,
    blurb:
      meta.blurb?.trim() ||
      (task === "encode"
        ? `${meta.dim}-dimensional embeddings, imported from ${req.onnx.name}.`
        : `Imported from ${req.onnx.name}.`),
    files: [{ part: "model", url, bytes: req.onnx.size }],
    inputSize: meta.inputSize,
    mean: meta.mean ?? [0.485 * 255, 0.456 * 255, 0.406 * 255],
    std: meta.std ?? [0.229 * 255, 0.224 * 255, 0.225 * 255],
    targetMpp: meta.targetMpp ?? null,
    genes: meta.genes,
    dim: meta.dim,
    licence: meta.licence?.trim() || "Supplied by you — check your own licence terms.",
    backend: meta.backend ?? "wasm",
  };

  await (await db()).put(STORE, spec);
  return spec;
}


/** Kept for the spatial dialog, which only ever imports one kind. */
export const importSpatialModel = importSidecarModel;
