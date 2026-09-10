import { create } from "zustand";
import type { ModelSpec } from "./registry";
import type { SignatureSet } from "./signatures";
import type { SpatialResult } from "./spatialResult";

/**
 * Virtual spatial state: which model, which gene is on screen, and the last
 * prediction. Kept apart from the segmentation store because the two share
 * nothing but the word "model", and merging them would mean every panel
 * re-renders on the other's progress.
 */

export interface SpatialProgress {
  done: number;
  total: number;
  ms: number;
}

interface SpatialState {
  models: ModelSpec[];
  activeModelId: string | null;
  status: "idle" | "loading" | "ready" | "running" | "error";
  error: string | null;
  /** Weight download, 0-1, while loading. */
  download: number;
  /** Execution provider actually in use, once loaded. */
  backend: string | null;
  progress: SpatialProgress | null;
  result: SpatialResult | null;
  /** Gene currently coloured on the slide. */
  gene: string | null;
  /** Whether the map shows one gene or a cell-type signature score. */
  mode: "gene" | "signature";
  signatures: SignatureSet | null;
  signatureName: string | null;
  /** Opacity of the expression overlay, separate from annotation opacity. */
  opacity: number;
  visible: boolean;

  setModels: (m: ModelSpec[]) => void;
  setActiveModel: (id: string | null) => void;
  setStatus: (s: SpatialState["status"], error?: string | null) => void;
  setDownload: (v: number) => void;
  setBackend: (b: string | null) => void;
  setProgress: (p: SpatialProgress | null) => void;
  setResult: (r: SpatialResult | null) => void;
  setGene: (g: string | null) => void;
  setMode: (m: "gene" | "signature") => void;
  setSignatures: (s: SignatureSet | null) => void;
  setSignatureName: (n: string | null) => void;
  setOpacity: (v: number) => void;
  setVisible: (v: boolean) => void;
}

export const useSpatial = create<SpatialState>((set, get) => ({
  models: [],
  activeModelId: null,
  status: "idle",
  error: null,
  download: 0,
  backend: null,
  progress: null,
  result: null,
  gene: null,
  mode: "gene",
  signatures: null,
  signatureName: null,
  opacity: 0.75,
  visible: true,

  setModels: (models) =>
    set((s) => ({
      models,
      activeModelId: models.some((m) => m.id === s.activeModelId)
        ? s.activeModelId
        : (models[0]?.id ?? null),
    })),
  setActiveModel: (activeModelId) => set({ activeModelId, status: "idle", error: null }),
  setStatus: (status, error = null) => set({ status, error }),
  setDownload: (download) => set({ download }),
  setBackend: (backend) => set({ backend }),
  setProgress: (progress) => set({ progress }),
  // A new prediction selects its first gene, so something is on screen without
  // the user having to also pick from a list to see that it worked.
  setResult: (result) =>
    set({ result, gene: result ? (get().gene && result.genes.includes(get().gene!) ? get().gene : result.genes[0]) : null }),
  setGene: (gene) => set({ gene, mode: "gene" }),
  setMode: (mode) => set({ mode }),
  // Importing a set selects its first signature, so the map changes on import
  // rather than leaving the user to guess that anything happened.
  setSignatures: (signatures) =>
    set({
      signatures,
      signatureName: signatures?.signatures[0]?.name ?? null,
      mode: signatures ? "signature" : "gene",
    }),
  setSignatureName: (signatureName) => set({ signatureName, mode: "signature" }),
  setOpacity: (opacity) => set({ opacity }),
  setVisible: (visible) => set({ visible }),
}));
