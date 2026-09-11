import { create } from "zustand";
import type { ModelSpec } from "./registry";
import type { SignatureSet } from "./signatures";
import { BUILTIN_SIGNATURES } from "./builtinSignatures";
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
  /** Never null: falls back to the built-in modules. */
  signatures: SignatureSet;
  signatureName: string | null;
  /** Opacity of the expression overlay, separate from annotation opacity. */
  opacity: number;
  visible: boolean;
  /**
   * Hide patches that do not land on detected tissue.
   *
   * A map arrives with a patch wherever whoever computed it thought there was
   * tissue, and that judgement was made elsewhere — a different detector, a
   * different threshold, or on a slide edge that looked dark enough. Masking
   * to the tissue objects in THIS session lets the slide in front of you have
   * the last word, and it is the difference between a gene that looks
   * expressed in the background and one that is only read where there are
   * cells to express it.
   *
   * Display-only: the values are untouched, so unticking it brings them back.
   */
  onTissueOnly: boolean;
  /** Per-patch, 1 where the patch centre falls inside a tissue object. */
  tissueMask: Uint8Array | null;
  /**
   * Bumped whenever the mask is replaced. The renderer needs to know the mask
   * CHANGED, and a recomputed mask has the same length and the same identity
   * as often as not — editing one tissue object leaves both untouched.
   */
  tissueMaskVersion: number;

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
  setOnTissueOnly: (v: boolean) => void;
  setTissueMask: (m: Uint8Array | null) => void;
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
  // Built in, so a map that has just loaded already has something worth
  // looking at. Modules whose genes this map does not carry are filtered out
  // downstream, so the same list serves eight genes and nineteen thousand.
  signatures: BUILTIN_SIGNATURES,
  signatureName: null,
  opacity: 0.75,
  visible: true,
  onTissueOnly: true,
  tissueMask: null,
  tissueMaskVersion: 0,

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
  // Passing null goes back to the built-ins rather than to nothing: losing the
  // modules is never what "use different signatures" was asking for.
  setSignatures: (signatures) => {
    const next = signatures ?? BUILTIN_SIGNATURES;
    set({
      signatures: next,
      signatureName: next.signatures[0]?.name ?? null,
      mode: "signature",
    });
  },
  setSignatureName: (signatureName) => set({ signatureName, mode: "signature" }),
  setOpacity: (opacity) => set({ opacity }),
  setVisible: (visible) => set({ visible }),
  setOnTissueOnly: (onTissueOnly) => set({ onTissueOnly }),
  setTissueMask: (tissueMask) =>
    set((s) => ({ tissueMask, tissueMaskVersion: s.tissueMaskVersion + 1 })),
}));
