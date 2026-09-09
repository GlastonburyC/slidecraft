/**
 * Model registry.
 *
 * Every model declares where its weights come from, what input it expects and
 * under what licence. Nothing is bundled: weights are fetched on first use and
 * cached locally, which keeps licence-gated encoders (UNI, Virchow, CONCH) out
 * of the repository while still letting you register them yourself.
 */

export type ModelTask = "prompt-segment" | "encode" | "segment";

export interface ModelFile {
  /** Logical part name, e.g. "encoder" / "decoder". */
  part: string;
  url: string;
  /** Approximate download size, for honest progress reporting. */
  bytes: number;
}

export interface ModelSpec {
  id: string;
  name: string;
  task: ModelTask;
  /** Short description shown in the UI. */
  blurb: string;
  files: ModelFile[];
  /** Square input side the encoder expects, in pixels. */
  inputSize: number;
  /** Per-channel mean/std in 0-255 space. */
  mean: [number, number, number];
  std: [number, number, number];
  /** Microns per pixel the model was trained at; null means "use ROI native". */
  targetMpp: number | null;
  licence: string;
  /**
   * Execution provider this model is known to be *correct* on. WebGPU's
   * coverage of quantized operators is incomplete, and a model that runs fast
   * but returns noise is worse than one that runs slower and is right.
   */
  backend: "wasm" | "webgpu" | "auto";
  /** Embedding dimensionality, for models used as feature extractors. */
  dim?: number;
}

const HF = "https://huggingface.co";

/**
 * Every SAM variant below shares SamImageProcessor's preprocessing (verified
 * against each repo's preprocessor_config.json) and the same ONNX signature,
 * so they differ only in weights, size and quality.
 */
const SAM_COMMON = {
  task: "prompt-segment" as const,
  inputSize: 1024,
  // rescale 1/255 then ImageNet mean/std, expressed in 0-255 space.
  mean: [0.485 * 255, 0.456 * 255, 0.406 * 255] as [number, number, number],
  std: [0.229 * 255, 0.224 * 255, 0.225 * 255] as [number, number, number],
  targetMpp: null,
  // Measured on a real H&E field: on the WebGPU EP these int8 weights return
  // salt-and-pepper noise (best mask density 0.006), while on wasm the same
  // prompt gives a clean nucleus (density 0.725). Correctness wins.
  backend: "wasm" as const,
};

const samFiles = (repo: string, encBytes: number, decBytes: number): ModelFile[] => [
  { part: "encoder", url: `${HF}/${repo}/resolve/main/onnx/vision_encoder_quantized.onnx`, bytes: encBytes },
  { part: "decoder", url: `${HF}/${repo}/resolve/main/onnx/prompt_encoder_mask_decoder_quantized.onnx`, bytes: decBytes },
];

export const BUILTIN_MODELS: ModelSpec[] = [
  {
    ...SAM_COMMON,
    id: "slimsam-77",
    name: "SlimSAM-77",
    blurb:
      "Heavily pruned Segment Anything. Smallest and fastest; a good default. " +
      "The encoder runs once per view, then each click costs milliseconds.",
    files: samFiles("Xenova/slimsam-77-uniform", 8_882_165, 4_903_810),
    licence: "Apache-2.0",
  },
  {
    ...SAM_COMMON,
    id: "slimsam-50",
    name: "SlimSAM-50",
    blurb:
      "Less aggressively pruned than SlimSAM-77. Noticeably better on crowded or " +
      "faint nuclei, for a bigger download and a slower encode.",
    files: samFiles("Xenova/slimsam-50-uniform", 30_100_000, 4_903_810),
    licence: "Apache-2.0",
  },
  {
    ...SAM_COMMON,
    id: "sam-vit-base",
    name: "SAM ViT-B",
    blurb:
      "The original Segment Anything encoder, unpruned. The strongest point-prompt " +
      "quality available here; the encode after each pan is the price.",
    files: samFiles("Xenova/sam-vit-base", 101_100_000, 4_903_810),
    licence: "Apache-2.0",
  },
];

/** User-registered models (BYO ONNX) live alongside the built-ins. */
export interface RegistryState {
  models: ModelSpec[];
}

export function findModel(models: ModelSpec[], id: string): ModelSpec | undefined {
  return models.find((m) => m.id === id);
}

export const totalBytes = (m: ModelSpec) => m.files.reduce((n, f) => n + f.bytes, 0);

export const formatBytes = (n: number) =>
  n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.round(n / 1e3)} kB`;

export const allModelUrls = (m: ModelSpec) => m.files.map((f) => f.url);
