import { openDB, type IDBPDatabase } from "idb";
import type { Annotation, AnnotationClass } from "../annotate/types";
import type { SlideMeta } from "../slide/types";

/**
 * Local autosave. Annotations are the user's work and must survive a reload or
 * a crash, so they are written to IndexedDB rather than held in memory only.
 * Slide pixels are never stored — only geometry.
 */

const DB_NAME = "slidecraft";
const STORE = "documents";

export interface StoredDocument {
  slideKey: string;
  slideName: string;
  annotations: Annotation[];
  classes: AnnotationClass[];
  updatedAt: number;
}

let dbPromise: Promise<IDBPDatabase> | null = null;

function db() {
  if (!dbPromise) {
    dbPromise = openDB(DB_NAME, 1, {
      upgrade(database) {
        if (!database.objectStoreNames.contains(STORE)) {
          database.createObjectStore(STORE, { keyPath: "slideKey" });
        }
      },
    });
  }
  return dbPromise;
}

/**
 * Identifies a slide across sessions without hashing gigabytes: vendor
 * dimensions plus byte count plus name is unique in practice, and costs
 * nothing to compute.
 */
export function slideKeyOf(meta: SlideMeta): string {
  const l0 = meta.levels[0];
  return `${meta.name}|${l0.width}x${l0.height}|${meta.bytes}`;
}

export async function saveDocument(doc: StoredDocument): Promise<void> {
  try {
    await (await db()).put(STORE, doc);
  } catch (err) {
    console.warn("[slidecraft] autosave failed", err);
  }
}

export async function loadDocument(slideKey: string): Promise<StoredDocument | undefined> {
  try {
    return await (await db()).get(STORE, slideKey);
  } catch (err) {
    console.warn("[slidecraft] could not read saved annotations", err);
    return undefined;
  }
}

export async function deleteDocument(slideKey: string): Promise<void> {
  try {
    await (await db()).delete(STORE, slideKey);
  } catch { /* nothing useful to do */ }
}

export async function listDocuments(): Promise<StoredDocument[]> {
  try {
    return await (await db()).getAll(STORE);
  } catch {
    return [];
  }
}
