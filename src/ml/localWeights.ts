/**
 * Imported model weights, in OPFS.
 *
 * Not the Cache Storage API, which is where every other fetched asset lives.
 * A modern histology encoder is over a gigabyte, and Cache.put fails on a body
 * that size — in Chrome with "Unexpected internal error", which says nothing
 * about the cause and leaves a spec pointing at weights that were never
 * written. OPFS is built for files this large and takes a stream, so the whole
 * model is never held in memory on the way in.
 *
 * Weights are addressed by the same synthetic https URL the registry records,
 * so nothing above this layer needs to know where the bytes actually sit.
 */

const DIR = "model-weights";

interface Location {
  id: string;
  part: string;
}

/** `https://local.slidecraft.invalid/<id>/<part>` -> its place in OPFS. */
export function locate(url: string): Location | null {
  try {
    const { pathname } = new URL(url);
    const parts = pathname.split("/").filter(Boolean);
    if (parts.length !== 2) return null;
    return { id: parts[0], part: parts[1] };
  } catch {
    return null;
  }
}

async function dir(create: boolean): Promise<FileSystemDirectoryHandle | null> {
  try {
    const root = await navigator.storage?.getDirectory?.();
    if (!root) return null;
    return await root.getDirectoryHandle(DIR, { create });
  } catch {
    return null; // private mode, or no OPFS
  }
}

export class WeightsUnavailable extends Error {}

/**
 * Write a file's bytes, streaming.
 *
 * `file.stream().pipeTo(writable)` moves the model in chunks. Reading it into
 * an ArrayBuffer first — which is what the Cache Storage path did — needs the
 * entire model resident *and* a second copy for the write, which is how a
 * 1.27 GB import came to need well over 2.5 GB to land.
 */
export async function putLocalWeights(url: string, file: File): Promise<void> {
  const where = locate(url);
  const root = await dir(true);
  if (!where || !root) {
    throw new WeightsUnavailable(
      "This browser will not give the app private storage, so imported weights cannot be kept. " +
        "A private window usually causes this.",
    );
  }

  const folder = await root.getDirectoryHandle(where.id, { create: true });
  const handle = await folder.getFileHandle(where.part, { create: true });
  const writable = await handle.createWritable();
  try {
    await file.stream().pipeTo(writable);
  } catch (err) {
    // A partial file is worse than none: it would load as a corrupt model.
    await folder.removeEntry(where.part).catch(() => undefined);
    throw new WeightsUnavailable(
      `Could not store ${file.name} (${(file.size / 1e9).toFixed(2)} GB): ` +
        (err instanceof Error ? err.message : String(err)),
    );
  }
}

export async function getLocalWeights(url: string): Promise<Uint8Array | null> {
  const where = locate(url);
  const root = await dir(false);
  if (!where || !root) return null;
  try {
    const folder = await root.getDirectoryHandle(where.id);
    const handle = await folder.getFileHandle(where.part);
    const file = await handle.getFile();
    return new Uint8Array(await file.arrayBuffer());
  } catch {
    return null;
  }
}

export async function hasLocalWeights(url: string): Promise<boolean> {
  const where = locate(url);
  const root = await dir(false);
  if (!where || !root) return false;
  try {
    const folder = await root.getDirectoryHandle(where.id);
    await folder.getFileHandle(where.part);
    return true;
  } catch {
    return false;
  }
}

export async function deleteLocalWeights(id: string): Promise<void> {
  const root = await dir(false);
  if (!root) return;
  await root.removeEntry(id, { recursive: true }).catch(() => undefined);
}

/** Bytes held, so the UI can say what imported models are costing. */
export async function localWeightsUsage(): Promise<number> {
  const root = await dir(false);
  if (!root) return 0;
  let total = 0;
  try {
    for await (const [, folder] of root.entries()) {
      if (folder.kind !== "directory") continue;
      for await (const [, entry] of folder.entries()) {
        if (entry.kind === "file") total += (await entry.getFile()).size;
      }
    }
  } catch {
    return total;
  }
  return total;
}
