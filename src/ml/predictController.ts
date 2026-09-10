import type { Annotation } from "../annotate/types";
import type { SlideSource } from "../slide/types";
import { EmbeddingCache } from "./embeddingCache";
import type { Res } from "./embedWorker";
import { predict, trainHead, type Head, type LabelledSet } from "./head";
import { patchKey, type PatchGrid } from "./patchGrid";
import { labelPatches } from "./patchLabels";
import { usePredict, type Prediction } from "./predictStore";
import type { ModelSpec } from "./registry";
import { resampleTo } from "./spatialController";

/**
 * The prediction loop.
 *
 * Embed the ROI once — cached, so it is once ever — then label, train, predict,
 * correct, retrain. Only the first step touches pixels; everything after is
 * arithmetic on vectors already in memory, which is what makes correcting a
 * prediction and seeing the result a single action.
 */

/**
 * Patches per forward pass.
 *
 * Small on purpose. The load-time probe proves a ViT-H runs at batch 1, but
 * activations scale with the batch, and a WebGPU device that manages one patch
 * can stall or fail allocating for eight — with no error, because the failure
 * is a buffer request that never returns. Whatever is gained by batching is
 * not worth a run that appears to hang.
 */
const BATCH = 2;

/** A batch slower than this is reported, rather than left looking like a hang. */
const SLOW_BATCH_MS = 45_000;

/**
 * A single patch read that takes longer than this has not simply been slow.
 *
 * Reading a 224px region off an open slide is milliseconds; a minute means the
 * decoder pool is wedged, and waiting forever turns that into a progress bar
 * that never moves. Better to say so and let the run fail with a cause.
 */
const READ_TIMEOUT_MS = 60_000;

function withTimeout<T>(work: Promise<T>, ms: number, what: string): Promise<T> {
  return Promise.race([
    work,
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(`${what} did not return within ${ms / 1000}s`)), ms),
    ),
  ]);
}

/** Thrown into any in-flight request when the user presses stop. */
class Cancelled extends Error {
  constructor() {
    super("cancelled");
  }
}

export class PredictController {
  private worker: Worker | null = null;
  private seq = 0;
  private pending = new Map<number, { resolve: (v: Res) => void; reject: (e: Error) => void }>();
  private cancelled = false;
  private cache: EmbeddingCache | null = null;
  /** Which spec the worker currently holds, so it is loaded exactly once. */
  private loadedId: string | null = null;

  constructor(
    private readonly source: SlideSource,
    private readonly makeWorker: () => Worker = () =>
      new Worker(new URL("./embedWorker.ts", import.meta.url), { type: "module" }),
  ) {}

  private ensure(): Worker {
    if (this.worker) return this.worker;
    const w = this.makeWorker();
    w.onmessage = (ev: MessageEvent<Res>) => {
      const msg = ev.data;
      if (msg.type === "progress") {
        usePredict.getState().setDownload(msg.total ? msg.received / msg.total : 0);
        return;
      }
      const entry = this.pending.get(msg.id);
      if (!entry) return;
      this.pending.delete(msg.id);
      if (msg.type === "error") entry.reject(new Error(msg.message));
      else entry.resolve(msg);
    };
    w.onerror = (e) => {
      const err = new Error(e.message || "encoder worker failed");
      this.pending.forEach((p) => p.reject(err));
      this.pending.clear();
    };
    this.worker = w;
    return w;
  }

  private send(msg: Record<string, unknown>, transfer: Transferable[] = []): Promise<Res> {
    const id = ++this.seq;
    const w = this.ensure();
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      w.postMessage({ ...msg, id }, transfer);
    });
  }

  /**
   * Stop, and mean it.
   *
   * Setting a flag the loop checks between batches is not enough: the loop
   * spends nearly all its time awaiting a batch, so a stop pressed during one
   * did nothing visible until that batch finished — which, on a ViT-H under
   * wasm, can be minutes. Rejecting the in-flight request unblocks the await
   * immediately. The worker carries on with the batch it already has and its
   * reply is discarded, because interrupting it would mean tearing down a
   * session that took a gigabyte and a minute to build.
   */
  cancel() {
    this.cancelled = true;
    const err = new Cancelled();
    this.pending.forEach((p) => p.reject(err));
    this.pending.clear();
  }

  async load(spec: ModelSpec): Promise<number> {
    const store = usePredict.getState();
    store.setStatus("loading");
    store.setDownload(0);
    try {
      const res = await this.send({ type: "load", spec });
      if (res.type !== "loaded") throw new Error("unexpected reply from the encoder");
      this.loadedId = spec.id;
      store.setBackend(res.backend);
      store.setStatus("ready");
      return res.dim;
    } catch (err) {
      store.setStatus("error", err instanceof Error ? err.message : String(err));
      throw err;
    }
  }

  /**
   * Embed every patch in a grid, reusing anything already cached.
   *
   * The cache is keyed by slide, encoder, level, patch size and position, so a
   * second pass over the same ROI — or an overlapping one — encodes only what
   * is genuinely new. That is the difference between iterating on an ROI and
   * waiting for it each time.
   */
  async embed(grid: PatchGrid, spec: ModelSpec): Promise<Float32Array | null> {
    const store = usePredict.getState();
    this.cancelled = false;

    /**
     * The worker has to hold the session, whatever the sidecar says.
     *
     * This once read `spec.dim ?? await load(spec)`, which skips loading
     * entirely for any export that declares its width — which is all of them.
     * The encoder was never sent to the worker, and every batch failed with
     * "No encoder is loaded" while the panel sat at zero. The declared width
     * is worth having, but it answers a different question from "is the model
     * in memory".
     */
    if (this.loadedId !== spec.id) {
      await this.load(spec);
      // A breath after the session is built. Creating it leaves the tab at its
      // memory ceiling, and starting to read the slide in the same tick makes
      // the decoder compete with a collection that has not happened yet.
      await new Promise((r) => setTimeout(r, 250));
    }
    const dim = spec.dim ?? (await this.load(spec));
    const slideKey = this.source.meta.name;
    const cache = await EmbeddingCache.open(
      { slideKey, modelId: spec.id, level: grid.level, patchPx: grid.patchPx, dim },
      grid.patches.length,
    );
    this.cache = cache;

    const keys = grid.patches.map((p) => patchKey(slideKey, spec.id, grid.level, grid.patchPx, p));
    const todo = grid.patches.filter((_, i) => !cache.has(keys[i]));
    const cached = grid.patches.length - todo.length;

    store.setStatus("embedding");
    store.setEmbedded({ done: cached, total: grid.patches.length, cached, ms: 0, stage: "reading" });

    const started = performance.now();
    const size = spec.inputSize;
    const level = grid.level;
    const ds = this.source.meta.levels[level]?.downsample ?? 1;
    const readSide = Math.max(1, Math.round((grid.patchPx * grid.downsample) / ds));

    try {
      for (let at = 0; at < todo.length; at += BATCH) {
        if (this.cancelled) break;
        const batch = todo.slice(at, at + BATCH);
        // Traced because a stalled run is otherwise indistinguishable at every
        // step: the console says which patch, at which level, and how long.
        console.info(
          `[predict] batch ${at / BATCH + 1}: reading ${batch.length} patches ` +
            `at level ${level}, ${readSide}px -> ${size}px`,
        );

        const tiles = new Uint8ClampedArray(batch.length * size * size * 4);
        for (const [n, p] of batch.entries()) {
          // Checked per patch, not per batch: reading several regions off a
          // slide is itself long enough for a stop to feel ignored.
          if (this.cancelled) break;
          const readAt = performance.now();
          const rgba = await withTimeout(
            this.source.readRegion(p.x, p.y, level, readSide, readSide),
            READ_TIMEOUT_MS,
            `Reading patch ${p.index} at level ${level}, ${readSide}px, at (${p.x}, ${p.y})`,
          );
          const readMs = performance.now() - readAt;
          // A read is milliseconds when nothing is competing for memory. One
          // that takes seconds is the encoder's session squeezing the decoder,
          // not the slide being slow, and the number is the evidence.
          if (readMs > 1000) {
            console.warn(`[predict] patch ${p.index} read took ${Math.round(readMs)} ms`);
          }
          resampleTo(rgba, readSide, readSide, tiles.subarray(n * size * size * 4, (n + 1) * size * size * 4), size);
        }

        // Reported before the forward pass, so a slow encoder and a slow slide
        // reader are distinguishable rather than one undifferentiated wait.
        store.setEmbedded({
          done: cached + at,
          total: grid.patches.length,
          cached,
          ms: performance.now() - started,
          stage: "encoding",
        });

        const batchStarted = performance.now();
        const res = await withTimeout(
          this.send({ type: "embed", tiles: tiles.buffer, size, count: batch.length }, [tiles.buffer]),
          10 * 60_000,
          `Encoding a batch of ${batch.length}`,
        );
        const batchMs = performance.now() - batchStarted;
        console.info(`[predict] batch encoded in ${Math.round(batchMs)} ms`);
        if (batchMs > SLOW_BATCH_MS && at === 0) {
          store.setStatus(
            "embedding",
            `The first batch took ${Math.round(batchMs / 1000)}s. At that rate this ROI needs ` +
              `about ${Math.round((batchMs / BATCH) * grid.patches.length / 60000)} minutes.`,
          );
        }
        if (this.cancelled) break;
        if (res.type !== "embedded") throw new Error("unexpected reply from the encoder");

        const vectors = new Float32Array(res.vectors);
        batch.forEach((p, n) => {
          const key = patchKey(slideKey, spec.id, level, grid.patchPx, p);
          cache.put(key, vectors.subarray(n * dim, (n + 1) * dim));
        });

        store.setEmbedded({
          done: cached + at + batch.length,
          total: grid.patches.length,
          cached,
          ms: performance.now() - started,
          stage: "reading",
        });
      }
      await cache.flush();
    } catch (err) {
      // A stop is not a failure, and must not be reported as one.
      if (!(err instanceof Cancelled) && !this.cancelled) {
        store.setStatus("error", err instanceof Error ? err.message : String(err));
        return null;
      }
    }

    if (this.cancelled) {
      // Whatever was encoded before the stop is kept: the cache is written per
      // batch, so resuming later re-encodes only what is genuinely missing.
      await this.cache?.flush();
      store.setStatus("ready");
      return null;
    }

    // Gathered into one contiguous block in grid order, so training and
    // prediction can walk it without a lookup per patch.
    const all = new Float32Array(grid.patches.length * dim);
    grid.patches.forEach((p, i) => {
      const v = cache.get(patchKey(slideKey, spec.id, level, grid.patchPx, p));
      if (v) all.set(v, i * dim);
    });

    store.setVectors(all, dim, grid);
    store.setStatus("ready");
    return all;
  }

  /**
   * Fit a head on the patches your annotations cover, then predict everywhere.
   *
   * Both halves are here because they are one action: a head that is not
   * immediately applied tells you nothing, and the point of the loop is to see
   * what it thinks so you can disagree.
   */
  train(
    classes: { classId: string; name: string; regions: Annotation[] }[],
    blockPx: number,
  ): { head: Head; prediction: Prediction } | null {
    const store = usePredict.getState();
    const { vectors, dim, grid } = store;
    if (!vectors || !grid) {
      store.setStatus("error", "Embed the ROI first.");
      return null;
    }

    const { labelled, perClass } = labelPatches(
      grid.patches,
      classes.map((c) => ({ classId: c.classId, regions: c.regions })),
    );
    if (labelled.length === 0) {
      store.setStatus("error", "None of the patches fall inside a labelled region.");
      return null;
    }

    const index = new Map(grid.patches.map((p, i) => [p.index, i]));
    const set: LabelledSet = {
      x: new Float32Array(labelled.length * dim),
      dim,
      y: new Uint8Array(labelled.length),
      px: new Float64Array(labelled.length),
      py: new Float64Array(labelled.length),
      count: labelled.length,
    };
    labelled.forEach((l, i) => {
      const row = index.get(l.patch.index)!;
      set.x.set(vectors.subarray(row * dim, (row + 1) * dim), i * dim);
      set.y[i] = l.label;
      set.px[i] = l.patch.x;
      set.py[i] = l.patch.y;
    });

    store.setStatus("training");
    const started = performance.now();
    let head: Head;
    try {
      head = trainHead(set, classes.map((c) => c.name), store.encoderId ?? "unknown", blockPx);
    } catch (err) {
      store.setStatus("error", err instanceof Error ? err.message : String(err));
      return null;
    }

    const nClasses = classes.length;
    const probs = new Float32Array(grid.patches.length * nClasses);
    const scratch = new Float32Array(nClasses);
    for (let i = 0; i < grid.patches.length; i++) {
      predict(head, vectors.subarray(i * dim, (i + 1) * dim), scratch);
      probs.set(scratch, i * nClasses);
    }

    const prediction: Prediction = {
      probs,
      grid,
      classes: classes.map((c) => c.name),
      classIds: classes.map((c) => c.classId),
      ms: performance.now() - started,
    };
    store.setHead(head);
    store.setPrediction(prediction);
    store.setStatus("ready");
    void perClass;
    return { head, prediction };
  }

  async destroy() {
    await this.cache?.flush();
    this.worker?.terminate();
    this.worker = null;
    this.pending.clear();
  }
}
