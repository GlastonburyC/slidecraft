import { create } from "zustand";
import { BUILTIN_MODELS, type ModelSpec } from "./registry";
import type { PatchGrid } from "./patchGrid";

export type ModelStatus = "idle" | "downloading" | "compiling" | "ready" | "error";

export interface EncodedView {
  /** Whether this embedding came from the viewport or from a fixed ROI. */
  origin: "view" | "roi";
  /** The ROI it was taken from, when origin is "roi". */
  roiId: string | null;
  /** Region encoded, in level-0 slide pixels. */
  x: number;
  y: number;
  width: number;
  height: number;
  /** Pixel dimensions actually read, i.e. the prompt coordinate space. */
  readW: number;
  readH: number;
  ms: number;
}

export interface PromptPoint {
  /** Level-0 slide pixels. */
  x: number;
  y: number;
  /** 1 = include, 0 = exclude. */
  label: 0 | 1;
}

interface MlState {
  models: ModelSpec[];
  activeModelId: string;
  /**
   * Encoder used to embed patches, chosen separately from the segmenter.
   *
   * They answer different questions and a head trained on one encoder's
   * vectors is meaningless applied to another's, so the choice is recorded on
   * its own rather than sharing a single "active model".
   */
  activeEncoderId: string | null;
  status: ModelStatus;
  backend: string | null;
  error: string | null;
  /** 0..1 across all parts of the current download. */
  progress: number;

  encoding: boolean;
  encoded: EncodedView | null;
  /** Last decode latency, for the UI to show honestly. */
  lastDecodeMs: number | null;
  lastScore: number | null;
  /** Share of the encoded field the last chosen mask covered. */
  lastAreaFrac: number | null;
  /** True when a decode returned nothing traceable. */
  lastEmpty: boolean;
  /** True when the last click fell outside the encoded ROI. */
  outsideRoi: boolean;
  /** Patch grid laid over an ROI, and which ROI it belongs to. */
  grid: { roiId: string; grid: PatchGrid } | null;
  /** Patch side in pixels, at the chosen level. */
  patchPx: number;
  /** Pyramid level the patches are sampled at. */
  patchLevel: number;
  /** Restrict patches to tissue rather than the whole ROI rectangle. */
  patchesOnTissueOnly: boolean;

  /** Result of the last tissue detection, for the UI to report. */
  tissue: { count: number; coverage: number; threshold: number; rejected: number; ms: number; modelName: string | null } | null;

  prompt: PromptPoint[];
  /** Auto-commit each mask when the next fresh click starts. */
  autoCommit: boolean;
  /** Clip each new mask against neighbouring cells so instances never overlap. */
  nonOverlapping: boolean;
  /** Model ids whose weights are already in the local cache. */
  cachedIds: string[];
  /** Non-null while a "download everything" pass is running. */
  prefetch: { modelName: string; done: number; total: number } | null;
  /** True once the browser has agreed to keep our storage. */
  persisted: boolean;

  setStatus: (s: ModelStatus, error?: string | null) => void;
  setBackend: (b: string) => void;
  setProgress: (p: number) => void;
  setEncoding: (b: boolean) => void;
  setEncoded: (e: EncodedView | null) => void;
  setPrompt: (p: PromptPoint[]) => void;
  setDecodeStats: (ms: number, score: number, areaFrac: number) => void;
  setEmpty: (b: boolean) => void;
  setOutsideRoi: (b: boolean) => void;
  setGrid: (g: { roiId: string; grid: PatchGrid } | null) => void;
  setPatchPx: (n: number) => void;
  setPatchLevel: (n: number) => void;
  setPatchesOnTissueOnly: (b: boolean) => void;
  setTissue: (t: { count: number; coverage: number; threshold: number; rejected: number; ms: number; modelName: string | null } | null) => void;
  setAutoCommit: (b: boolean) => void;
  setNonOverlapping: (b: boolean) => void;
  setCachedIds: (ids: string[]) => void;
  setPrefetch: (p: { modelName: string; done: number; total: number } | null) => void;
  setPersisted: (b: boolean) => void;
  setActiveModel: (id: string) => void;
  setActiveEncoder: (id: string | null) => void;
  /** Replace the registry, keeping the active model if it survives. */
  setModels: (models: ModelSpec[]) => void;
  addModel: (spec: ModelSpec) => void;
  reset: () => void;
}

export const useMl = create<MlState>((set) => ({
  models: BUILTIN_MODELS,
  activeModelId: BUILTIN_MODELS[0].id,
  activeEncoderId: null,
  status: "idle",
  backend: null,
  error: null,
  progress: 0,
  encoding: false,
  encoded: null,
  lastDecodeMs: null,
  lastScore: null,
  lastAreaFrac: null,
  lastEmpty: false,
  outsideRoi: false,
  tissue: null,
  grid: null,
  patchPx: 256,
  patchLevel: 0,
  patchesOnTissueOnly: true,
  prompt: [],
  autoCommit: true,
  nonOverlapping: true,
  cachedIds: [],
  prefetch: null,
  persisted: false,

  setStatus: (status, error = null) => set({ status, error }),
  setBackend: (backend) => set({ backend }),
  setProgress: (progress) => set({ progress }),
  setEncoding: (encoding) => set({ encoding }),
  setEncoded: (encoded) => set({ encoded }),
  setPrompt: (prompt) => set({ prompt }),
  setDecodeStats: (lastDecodeMs, lastScore, lastAreaFrac) =>
    set({ lastDecodeMs, lastScore, lastAreaFrac }),
  setEmpty: (lastEmpty) => set({ lastEmpty }),
  setOutsideRoi: (outsideRoi) => set({ outsideRoi }),
  setGrid: (grid) => set({ grid }),
  setPatchPx: (patchPx) => set({ patchPx, grid: null }),
  setPatchLevel: (patchLevel) => set({ patchLevel, grid: null }),
  setPatchesOnTissueOnly: (patchesOnTissueOnly) => set({ patchesOnTissueOnly, grid: null }),
  setTissue: (tissue) => set({ tissue }),
  setAutoCommit: (autoCommit) => set({ autoCommit }),
  setNonOverlapping: (nonOverlapping) => set({ nonOverlapping }),
  setCachedIds: (cachedIds) => set({ cachedIds }),
  setPrefetch: (prefetch) => set({ prefetch }),
  setPersisted: (persisted) => set({ persisted }),
  setActiveModel: (activeModelId) =>
    set({ activeModelId, status: "idle", encoded: null, prompt: [] }),
  // Changing encoder invalidates the grid's embeddings, not the grid itself.
  setActiveEncoder: (activeEncoderId) => set({ activeEncoderId }),

  setModels: (models) =>
    set((s) => ({
      models,
      activeModelId: models.some((m) => m.id === s.activeModelId)
        ? s.activeModelId
        : (models[0]?.id ?? s.activeModelId),
    })),

  addModel: (spec) =>
    set((s) => ({
      models: [...s.models.filter((m) => m.id !== spec.id), spec],
      // A freshly imported model is what the user wants to use next.
      activeModelId: spec.id,
      status: "idle",
      encoded: null,
      prompt: [],
    })),
  reset: () =>
    set({
      encoded: null, prompt: [], lastDecodeMs: null, lastScore: null,
      lastAreaFrac: null, lastEmpty: false, outsideRoi: false,
    }),
}));
