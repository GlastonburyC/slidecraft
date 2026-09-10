import { create } from "zustand";
import type { ModelSpec } from "./registry";
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
  progress: SpatialProgress | null;
  result: SpatialResult | null;
  /** Gene currently coloured on the slide. */
  gene: string | null;
  /** Opacity of the expression overlay, separate from annotation opacity. */
  opacity: number;
  visible: boolean;

  setModels: (m: ModelSpec[]) => void;
  setActiveModel: (id: string | null) => void;
  setStatus: (s: SpatialState["status"], error?: string | null) => void;
  setDownload: (v: number) => void;
  setProgress: (p: SpatialProgress | null) => void;
  setResult: (r: SpatialResult | null) => void;
  setGene: (g: string | null) => void;
  setOpacity: (v: number) => void;
  setVisible: (v: boolean) => void;
}

export const useSpatial = create<SpatialState>((set, get) => ({
  models: [],
  activeModelId: null,
  status: "idle",
  error: null,
  download: 0,
  progress: null,
  result: null,
  gene: null,
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
  setProgress: (progress) => set({ progress }),
  // A new prediction selects its first gene, so something is on screen without
  // the user having to also pick from a list to see that it worked.
  setResult: (result) =>
    set({ result, gene: result ? (get().gene && result.genes.includes(get().gene!) ? get().gene : result.genes[0]) : null }),
  setGene: (gene) => set({ gene }),
  setOpacity: (opacity) => set({ opacity }),
  setVisible: (visible) => set({ visible }),
}));
