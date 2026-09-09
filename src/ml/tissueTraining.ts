import { create } from "zustand";
import type { SlideSamples } from "./tissueLabels";
import type { TissueModel } from "./tissueModel";

/** The class whose regions mean "this is tissue". */
export const TISSUE_CLASS = "Tissue";
/** The class whose regions mean "this is not" — smears, dust, pen, bubbles. */
export const ARTEFACT_CLASS = "Not tissue";

interface TrainingState {
  /**
   * Labels gathered per slide.
   *
   * Held in memory across slide switches on purpose: teaching the model needs
   * examples from several slides, and losing the last slide's labels the moment
   * you open the next one would make that impossible to do.
   */
  samples: SlideSamples[];
  models: TissueModel[];
  staleModels: TissueModel[];
  activeModelId: string | null;
  /** Applied by "Detect tissue" when one is active. */
  useModel: boolean;
  busy: null | "sampling" | "training" | "batch";
  notice: string | null;

  addSamples: (s: SlideSamples) => void;
  removeSamples: (slide: string) => void;
  clearSamples: () => void;
  setModels: (models: TissueModel[], stale?: TissueModel[]) => void;
  addModel: (m: TissueModel) => void;
  setActiveModel: (id: string | null) => void;
  setUseModel: (on: boolean) => void;
  setBusy: (b: TrainingState["busy"]) => void;
  setNotice: (n: string | null) => void;
}

export const useTraining = create<TrainingState>((set) => ({
  samples: [],
  models: [],
  staleModels: [],
  activeModelId: null,
  useModel: true,
  busy: null,
  notice: null,

  // Re-adding a slide replaces its labels rather than stacking them, so
  // relabelling and pressing the button again is the obvious thing to do.
  addSamples: (s) =>
    set((st) => ({ samples: [...st.samples.filter((p) => p.slide !== s.slide), s] })),
  removeSamples: (slide) => set((st) => ({ samples: st.samples.filter((p) => p.slide !== slide) })),
  clearSamples: () => set({ samples: [] }),
  setModels: (models, stale) =>
    set((st) => ({
      models,
      staleModels: stale ?? st.staleModels,
      activeModelId: models.some((m) => m.id === st.activeModelId)
        ? st.activeModelId
        : (models[0]?.id ?? null),
    })),
  addModel: (m) => set((st) => ({ models: [m, ...st.models], activeModelId: m.id })),
  setActiveModel: (activeModelId) => set({ activeModelId }),
  setUseModel: (useModel) => set({ useModel }),
  setBusy: (busy) => set({ busy }),
  setNotice: (notice) => set({ notice }),
}));

export function activeModel(): TissueModel | null {
  const s = useTraining.getState();
  if (!s.useModel || !s.activeModelId) return null;
  return s.models.find((m) => m.id === s.activeModelId) ?? null;
}
