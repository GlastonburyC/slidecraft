import { create } from "zustand";
import type { PatchGrid } from "./patchGrid";
import type { ModelSpec } from "./registry";
import type { Head } from "./head";

/**
 * State for the prediction loop: embed once, then label, train, correct,
 * retrain — the last three costing milliseconds because the first is cached.
 */

export interface Prediction {
  /** Class probabilities, `patches.length * classes.length`, row-major. */
  probs: Float32Array;
  grid: PatchGrid;
  classes: string[];
  classIds: string[];
  ms: number;
}

interface PredictState {
  encoders: ModelSpec[];
  encoderId: string | null;
  status: "idle" | "loading" | "embedding" | "training" | "ready" | "error";
  error: string | null;
  download: number;
  backend: string | null;
  /** Embedding progress over the current grid. */
  embedded: { done: number; total: number; cached: number; ms: number } | null;
  /** Vectors for the grid in view, kept in memory for instant retraining. */
  vectors: Float32Array | null;
  dim: number;
  grid: PatchGrid | null;
  head: Head | null;
  prediction: Prediction | null;
  /** Class shown; null means the most likely class per patch. */
  shownClass: string | null;
  opacity: number;
  visible: boolean;
  /** Probability below which a patch is left uncoloured. */
  threshold: number;

  setEncoders: (m: ModelSpec[]) => void;
  setEncoder: (id: string | null) => void;
  setStatus: (s: PredictState["status"], error?: string | null) => void;
  setDownload: (v: number) => void;
  setBackend: (b: string | null) => void;
  setEmbedded: (e: PredictState["embedded"]) => void;
  setVectors: (v: Float32Array | null, dim: number, grid: PatchGrid | null) => void;
  setHead: (h: Head | null) => void;
  setPrediction: (p: Prediction | null) => void;
  setShownClass: (c: string | null) => void;
  setOpacity: (v: number) => void;
  setVisible: (v: boolean) => void;
  setThreshold: (v: number) => void;
  reset: () => void;
}

export const usePredict = create<PredictState>((set) => ({
  encoders: [],
  encoderId: null,
  status: "idle",
  error: null,
  download: 0,
  backend: null,
  embedded: null,
  vectors: null,
  dim: 0,
  grid: null,
  head: null,
  prediction: null,
  shownClass: null,
  opacity: 0.6,
  visible: true,
  threshold: 0.5,

  setEncoders: (encoders) =>
    set((s) => ({
      encoders,
      encoderId: encoders.some((m) => m.id === s.encoderId) ? s.encoderId : (encoders[0]?.id ?? null),
    })),
  // Changing encoder invalidates everything downstream: vectors from one
  // encoder mean nothing to a head fitted on another's.
  setEncoder: (encoderId) =>
    set({ encoderId, status: "idle", vectors: null, dim: 0, head: null, prediction: null, embedded: null }),
  setStatus: (status, error = null) => set({ status, error }),
  setDownload: (download) => set({ download }),
  setBackend: (backend) => set({ backend }),
  setEmbedded: (embedded) => set({ embedded }),
  setVectors: (vectors, dim, grid) => set({ vectors, dim, grid }),
  setHead: (head) => set({ head }),
  setPrediction: (prediction) => set({ prediction }),
  setShownClass: (shownClass) => set({ shownClass }),
  setOpacity: (opacity) => set({ opacity }),
  setVisible: (visible) => set({ visible }),
  setThreshold: (threshold) => set({ threshold }),
  reset: () =>
    set({
      status: "idle", error: null, embedded: null, vectors: null, dim: 0,
      grid: null, head: null, prediction: null, shownClass: null,
    }),
}));
