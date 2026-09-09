/**
 * Patch embeddings, kept between sessions.
 *
 * The encoder is the expensive half of the loop and its output never changes
 * for a given (slide, model, level, patch) — so it should be computed once,
 * ever. Caching is what makes the human-in-the-loop cycle work: correcting
 * labels and retraining a head is instant precisely because no pixels are
 * re-encoded.
 *
 * Vectors are packed into one contiguous buffer per shard rather than a file
 * per patch: a slide can have 10^5 patches, and that many OPFS files is slow to
 * enumerate and worse to delete.
 */

export interface ShardId {
  slideKey: string;
  modelId: string;
  level: number;
  patchPx: number;
  dim: number;
}

const shardName = (id: ShardId) =>
  `${sanitise(id.slideKey)}__${sanitise(id.modelId)}__L${id.level}__p${id.patchPx}__d${id.dim}`;

const sanitise = (s: string) => s.replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 120);

async function opfsDir(): Promise<FileSystemDirectoryHandle | null> {
  try {
    const root = await navigator.storage?.getDirectory?.();
    if (!root) return null;
    return await root.getDirectoryHandle("embeddings", { create: true });
  } catch {
    return null; // private mode, or no OPFS
  }
}

export class EmbeddingCache {
  private keys: string[] = [];
  private index = new Map<string, number>();
  private data: Float32Array;
  private dirty = false;
  private flushTimer = 0;

  private constructor(
    readonly id: ShardId,
    capacity: number,
  ) {
    this.data = new Float32Array(capacity * id.dim);
  }

  /** Open a shard, restoring anything previously written. */
  static async open(id: ShardId, expectedPatches = 1024): Promise<EmbeddingCache> {
    const cache = new EmbeddingCache(id, Math.max(64, expectedPatches));
    await cache.load();
    return cache;
  }

  get size(): number {
    return this.keys.length;
  }

  has(key: string): boolean {
    return this.index.has(key);
  }

  get(key: string): Float32Array | undefined {
    const slot = this.index.get(key);
    if (slot === undefined) return undefined;
    return this.data.subarray(slot * this.id.dim, (slot + 1) * this.id.dim);
  }

  put(key: string, vector: Float32Array): void {
    if (vector.length !== this.id.dim) {
      throw new Error(`embedding dim ${vector.length} does not match shard dim ${this.id.dim}`);
    }
    let slot = this.index.get(key);
    if (slot === undefined) {
      slot = this.keys.length;
      this.keys.push(key);
      this.index.set(key, slot);
      this.grow(slot + 1);
    }
    this.data.set(vector, slot * this.id.dim);
    this.dirty = true;
    this.scheduleFlush();
  }

  /** Which of these keys still need encoding. */
  missing(keys: string[]): string[] {
    return keys.filter((k) => !this.index.has(k));
  }

  private grow(needed: number) {
    const capacity = this.data.length / this.id.dim;
    if (needed <= capacity) return;
    const next = new Float32Array(Math.max(needed, capacity * 2) * this.id.dim);
    next.set(this.data);
    this.data = next;
  }

  private scheduleFlush() {
    if (this.flushTimer) return;
    // Encoding arrives in bursts; one write per burst rather than per patch.
    this.flushTimer = self.setTimeout(() => {
      this.flushTimer = 0;
      void this.flush();
    }, 1500);
  }

  async flush(): Promise<void> {
    if (!this.dirty) return;
    const dir = await opfsDir();
    if (!dir) return;
    this.dirty = false;
    try {
      const name = shardName(this.id);
      const meta = await dir.getFileHandle(`${name}.json`, { create: true });
      const bin = await dir.getFileHandle(`${name}.bin`, { create: true });

      const metaWriter = await meta.createWritable();
      await metaWriter.write(JSON.stringify({ ...this.id, keys: this.keys }));
      await metaWriter.close();

      const binWriter = await bin.createWritable();
      await binWriter.write(
        this.data.subarray(0, this.keys.length * this.id.dim).slice().buffer,
      );
      await binWriter.close();
    } catch (err) {
      this.dirty = true;
      console.warn("[slidecraft] could not persist embeddings", err);
    }
  }

  private async load(): Promise<void> {
    const dir = await opfsDir();
    if (!dir) return;
    try {
      const name = shardName(this.id);
      const meta = JSON.parse(await (await (await dir.getFileHandle(`${name}.json`)).getFile()).text()) as
        ShardId & { keys: string[] };
      // A shard whose shape no longer matches is stale, not usable.
      if (meta.dim !== this.id.dim || meta.patchPx !== this.id.patchPx) return;

      const buf = await (await (await dir.getFileHandle(`${name}.bin`)).getFile()).arrayBuffer();
      const stored = new Float32Array(buf);
      if (stored.length < meta.keys.length * this.id.dim) return;

      this.keys = meta.keys;
      this.index = new Map(meta.keys.map((k, i) => [k, i]));
      this.grow(this.keys.length);
      this.data.set(stored.subarray(0, this.keys.length * this.id.dim));
    } catch {
      // Nothing cached yet, which is the normal first-run case.
    }
  }

  /** Forget this shard, on disk and in memory. */
  async clear(): Promise<void> {
    this.keys = [];
    this.index.clear();
    this.dirty = false;
    const dir = await opfsDir();
    if (!dir) return;
    const name = shardName(this.id);
    for (const f of [`${name}.json`, `${name}.bin`]) {
      try { await dir.removeEntry(f); } catch { /* not there */ }
    }
  }
}

/** Total bytes OPFS is holding for embeddings. */
export async function embeddingCacheSize(): Promise<number> {
  const dir = await opfsDir();
  if (!dir) return 0;
  let total = 0;
  try {
    for await (const [, handle] of dir.entries()) {
      if (handle.kind === "file") total += (await handle.getFile()).size;
    }
  } catch { /* enumeration unsupported */ }
  return total;
}
