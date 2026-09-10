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
  embedded: {
    done: number;
    total: number;
    cached: number;
    ms: number;
    /** Which half of the loop the time is going into. */
    stage?: "reading" | "encoding";
  } | null;
  /** Vectors for the grid in view, kept in memory for instant retraining. */
  vectors: Float32Array | null;
  dim: number;
  grid: PatchGrid | null;
  /**
   * Principal components of those vectors, and the head is fitted on these.
   *
   * An encoder gives 1536 or 2560 numbers per patch and a first pass at
   * annotating gives a few dozen labels; a linear head on the raw width
   * separates them perfectly and generalises at chance, while reporting near
   * certainty. Components keep the structure and drop the ratio to something
   * a head can actually be fitted on.
   */
  scores: Float32Array | null;
  pcs: number;
  /** Which ROI each patch came from, so validation can hold whole ROIs out. */
  patchRoi: Int32Array | null;
  roiIds: string[];
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
  setScores: (s: Float32Array | null, pcs: number) => void;
  setPatchRoi: (r: Int32Array | null, ids: string[]) => void;
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
  scores: null,
  pcs: 0,
  patchRoi: null,
  roiIds: [],
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
  setVectors: (vectors, dim, grid) => set({ vectors, dim, grid, scores: null, pcs: 0 }),
  setScores: (scores, pcs) => set({ scores, pcs }),
  setPatchRoi: (patchRoi, roiIds) => set({ patchRoi, roiIds }),
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
      scores: null, pcs: 0, patchRoi: null, roiIds: [],
    }),
}));
