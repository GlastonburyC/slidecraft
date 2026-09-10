import type { SlideSource } from "../slide/types";
import type { Patch, PatchGrid } from "./patchGrid";
import type { ModelSpec } from "./registry";
import type { Res } from "./spatialWorker";
import { useSpatial } from "./spatialStore";
import type { SpatialResult } from "./spatialResult";

/**
 * Runs a virtual-spatial model over a patch grid.
 *
 * Two things decide whether the answer means anything, and both are handled
 * here rather than left to the caller:
 *
 * The model was trained at a magnification, and reading tiles at whatever the
 * slide happens to be scanned at silently changes the scale it reasons about —
 * a 40x tile shows it half the tissue it expects. So each patch is read at the
 * pyramid level closest to the model's target µm/pixel and resampled to the
 * input side.
 *
 * And a batch is a batch: tiles go over in groups so the worker is not woken
 * once per patch, while staying small enough that progress moves and the run
 * can be stopped.
 */

const BATCH = 8;

/**
 * How the controller reaches its worker.
 *
 * A seam, so the batching, the level choice and the assembly can be checked
 * without a WebGPU context and a hundred megabytes of weights — the parts most
 * likely to be wrong are the ones around the model, not the forward pass.
 */
export type WorkerFactory = () => Worker;

export class SpatialController {
  private worker: Worker | null = null;
  private seq = 0;
  private pending = new Map<number, { resolve: (v: Res) => void; reject: (e: Error) => void }>();
  private cancelled = false;

  constructor(
    private readonly source: SlideSource,
    private readonly makeWorker: WorkerFactory = () =>
      new Worker(new URL("./spatialWorker.ts", import.meta.url), { type: "module" }),
  ) {}

  private ensure(): Worker {
    if (this.worker) return this.worker;
    const w = this.makeWorker();
    w.onmessage = (ev: MessageEvent<Res>) => {
      const msg = ev.data;
      if (msg.type === "progress") {
        useSpatial.getState().setDownload(msg.total ? msg.received / msg.total : 0);
        return;
      }
      const entry = this.pending.get(msg.id);
      if (!entry) return;
      this.pending.delete(msg.id);
      if (msg.type === "error") entry.reject(new Error(msg.message));
      else entry.resolve(msg);
    };
    w.onerror = (e) => {
      const err = new Error(e.message || "spatial worker failed");
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

  async load(spec: ModelSpec): Promise<void> {
    const store = useSpatial.getState();
    store.setStatus("loading");
    store.setDownload(0);
    try {
      await this.send({ type: "load", spec });
      store.setStatus("ready");
    } catch (err) {
      store.setStatus("error", err instanceof Error ? err.message : String(err));
      throw err;
    }
  }

  cancel() {
    this.cancelled = true;
  }

  /**
   * Predict over every patch in a grid.
   *
   * `grid` comes from the patch panel, so what is predicted is exactly what was
   * drawn and checked — the same squares, at the same places.
   */
  async run(grid: PatchGrid, spec: ModelSpec, roiId: string | null): Promise<SpatialResult | null> {
    const store = useSpatial.getState();
    if (!spec.genes?.length) {
      store.setStatus("error", `${spec.name} does not declare which genes it predicts.`);
      return null;
    }
    if (grid.patches.length === 0) {
      store.setStatus("error", "That grid has no patches.");
      return null;
    }

    this.cancelled = false;
    store.setStatus("running");
    store.setProgress({ done: 0, total: grid.patches.length, ms: 0 });

    const side = grid.patchPx * grid.downsample;
    const size = spec.inputSize;
    const level = this.levelFor(spec.targetMpp);

    const values = new Float32Array(grid.patches.length * spec.genes.length);
    const started = performance.now();
    let done = 0;

    try {
      for (let at = 0; at < grid.patches.length; at += BATCH) {
        if (this.cancelled) break;
        const batch = grid.patches.slice(at, at + BATCH);
        const tiles = await this.readTiles(batch, side, size, level);

        const res = await this.send(
          { type: "predict", tiles: tiles.buffer, size, count: batch.length },
          [tiles.buffer],
        );
        if (res.type !== "predicted") throw new Error("unexpected worker reply");

        const got = new Float32Array(res.values);
        values.set(got, at * spec.genes.length);
        done += batch.length;
        store.setProgress({ done, total: grid.patches.length, ms: performance.now() - started });
      }
    } catch (err) {
      store.setStatus("error", err instanceof Error ? err.message : String(err));
      store.setProgress(null);
      return null;
    }

    store.setProgress(null);
    if (this.cancelled) {
      store.setStatus("ready");
      return null;
    }

    const result: SpatialResult = {
      genes: spec.genes,
      values,
      patches: grid.patches,
      side,
      modelId: spec.id,
      modelName: spec.name,
      slide: this.source.meta.name,
      roiId,
      ms: performance.now() - started,
      createdAt: new Date().toISOString(),
    };
    store.setResult(result);
    store.setStatus("ready");
    return result;
  }

  /** The pyramid level whose scale is nearest what the model was trained at. */
  private levelFor(targetMpp: number | null): number {
    const { levels, mppX } = this.source.meta;
    if (!targetMpp || !mppX) return 0;
    let best = 0;
    let bestErr = Infinity;
    for (const l of levels) {
      const err = Math.abs(l.downsample * mppX - targetMpp);
      // Ties and near-ties go to the finer level: downsampling a sharper read
      // is safe, upsampling a coarser one invents detail.
      if (err < bestErr - 1e-9) { bestErr = err; best = l.level; }
    }
    return best;
  }

  /** Read a batch of patches and pack them as RGBA at the model's input size. */
  private async readTiles(
    patches: Patch[],
    side: number,
    size: number,
    level: number,
  ): Promise<Uint8ClampedArray> {
    const ds = this.source.meta.levels[level]?.downsample ?? 1;
    const readSide = Math.max(1, Math.round(side / ds));
    const packed = new Uint8ClampedArray(patches.length * size * size * 4);

    for (const [n, p] of patches.entries()) {
      const rgba = await this.source.readRegion(p.x, p.y, level, readSide, readSide);
      const out = packed.subarray(n * size * size * 4, (n + 1) * size * size * 4);
      resampleTo(rgba, readSide, readSide, out, size);
    }
    return packed;
  }

  destroy() {
    this.worker?.terminate();
    this.worker = null;
    this.pending.clear();
  }
}

/**
 * Nearest-neighbour resample into the model's input square.
 *
 * Deliberately not bilinear: at these ratios the read is usually already close
 * to the target and the difference is invisible, while a filter would cost a
 * pass over every tile in a run of thousands. If a model ever needs a large
 * downscale this is the place to reconsider.
 */
export function resampleTo(
  src: Uint8ClampedArray,
  sw: number,
  sh: number,
  dst: Uint8ClampedArray,
  size: number,
): void {
  for (let y = 0; y < size; y++) {
    const sy = Math.min(sh - 1, Math.floor((y * sh) / size));
    for (let x = 0; x < size; x++) {
      const sx = Math.min(sw - 1, Math.floor((x * sw) / size));
      const s = (sy * sw + sx) * 4;
      const d = (y * size + x) * 4;
      dst[d] = src[s];
      dst[d + 1] = src[s + 1];
      dst[d + 2] = src[s + 2];
      dst[d + 3] = 255;
    }
  }
}
