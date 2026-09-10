import { openDB, type IDBPDatabase } from "idb";
import type { SlideSamples } from "./tissueLabels";
import { isUsable, type TissueModel } from "./tissueModel";

/**
 * Saved tissue models.
 *
 * The point of training one is that the next slide from the same scanner, stain
 * and lab starts where the last one finished, so the models outlive the session
 * that made them. They are a few hundred bytes each, so IndexedDB holds them
 * comfortably and there is no reason to prune.
 *
 * A model fitted against an older feature layout is kept but reported as stale
 * rather than loaded: its weights would still produce confident probabilities
 * against the new features, and they would mean nothing.
 *
 * The labelled cells are stored beside the model, not just the slide names.
 * Without them "retrain on everything" is impossible after a reload — you would
 * have to reopen each slide and mark it again — so adding one more slide to a
 * model that already knows four means loading four slides' worth of labels back
 * out of here and appending to them. They are a few megabytes per slide and
 * live in their own store, so listing models stays cheap.
 */

const DB_NAME = "slidecraft-tissue";
const STORE = "models";
const SAMPLES = "samples";

let handle: Promise<IDBPDatabase> | null = null;

function db(): Promise<IDBPDatabase> {
  handle ??= openDB(DB_NAME, 2, {
    upgrade(d) {
      if (!d.objectStoreNames.contains(STORE)) d.createObjectStore(STORE, { keyPath: "id" });
      // Added in v2. Models saved before it simply have no stored labels, and
      // the panel says so rather than pretending it can retrain them.
      if (!d.objectStoreNames.contains(SAMPLES)) d.createObjectStore(SAMPLES);
    },
  });
  return handle;
}

export async function saveTissueModel(
  model: TissueModel,
  samples?: SlideSamples[],
): Promise<void> {
  try {
    const d = await db();
    await d.put(STORE, model);
    // Typed arrays go through structured clone unchanged, so the features are
    // stored as they were computed rather than re-encoded into JSON.
    if (samples) await d.put(SAMPLES, samples, model.id);
  } catch (err) {
    throw new Error(
      `Could not save the tissue model: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

/**
 * The labelled cells a model was fitted on, so it can be extended rather than
 * replaced. Absent for models saved before labels were kept.
 */
export async function loadTissueSamples(id: string): Promise<SlideSamples[] | null> {
  try {
    const stored = (await (await db()).get(SAMPLES, id)) as SlideSamples[] | undefined;
    return stored ?? null;
  } catch {
    return null;
  }
}

export interface StoredModels {
  usable: TissueModel[];
  /** Saved against an older feature layout; shown, but not offered for use. */
  stale: TissueModel[];
}

export async function loadTissueModels(): Promise<StoredModels> {
  try {
    const all = (await (await db()).getAll(STORE)) as TissueModel[];
    all.sort((a, b) => b.trainedAt.localeCompare(a.trainedAt));
    return { usable: all.filter(isUsable), stale: all.filter((m) => !isUsable(m)) };
  } catch {
    return { usable: [], stale: [] };
  }
}

export async function deleteTissueModel(id: string): Promise<void> {
  try {
    const d = await db();
    await d.delete(STORE, id);
    await d.delete(SAMPLES, id);
  } catch {
    /* nothing to remove */
  }
}
