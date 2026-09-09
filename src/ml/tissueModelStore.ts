import { openDB, type IDBPDatabase } from "idb";
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
 */

const DB_NAME = "slidecraft-tissue";
const STORE = "models";

let handle: Promise<IDBPDatabase> | null = null;

function db(): Promise<IDBPDatabase> {
  handle ??= openDB(DB_NAME, 1, {
    upgrade(d) {
      if (!d.objectStoreNames.contains(STORE)) d.createObjectStore(STORE, { keyPath: "id" });
    },
  });
  return handle;
}

export async function saveTissueModel(model: TissueModel): Promise<void> {
  try {
    await (await db()).put(STORE, model);
  } catch (err) {
    throw new Error(
      `Could not save the tissue model: ${err instanceof Error ? err.message : String(err)}`,
    );
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
    await (await db()).delete(STORE, id);
  } catch {
    /* nothing to remove */
  }
}
