import type { ModelSpec } from "./registry";
import type { Res } from "./samWorker";

export interface DecodedMask {
  mask: Float32Array;
  w: number;
  h: number;
  score: number;
  /** Share of the encoded field the chosen mask covers. */
  areaFrac: number;
  ms: number;
}

/** Promise-shaped wrapper over the segmentation worker. */
export class SamClient {
  private worker: Worker;
  private seq = 0;
  private pending = new Map<
    number,
    { resolve: (v: unknown) => void; reject: (e: Error) => void }
  >();
  private onProgress?: (part: string, received: number, total: number) => void;

  constructor() {
    this.worker = new Worker(new URL("./samWorker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (ev: MessageEvent<Res>) => {
      const msg = ev.data;
      if (msg.type === "progress") {
        this.onProgress?.(msg.part, msg.received, msg.total);
        return;
      }
      const entry = this.pending.get(msg.id);
      if (!entry) return;
      this.pending.delete(msg.id);
      if (msg.type === "error") entry.reject(new Error(msg.message));
      else entry.resolve(msg);
    };
    this.worker.onerror = (e) => {
      const err = new Error(e.message || "segmentation worker failed");
      this.pending.forEach((p) => p.reject(err));
      this.pending.clear();
    };
  }

  private send<T>(msg: object, transfer: Transferable[] = []): Promise<T> {
    const id = ++this.seq;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
      this.worker.postMessage({ ...msg, id }, transfer);
    });
  }

  load(spec: ModelSpec, onProgress?: (part: string, received: number, total: number) => void) {
    this.onProgress = onProgress;
    return this.send<{ backend: string; inputs: string[] }>({ type: "load", spec });
  }

  encode(rgba: Uint8ClampedArray, width: number, height: number) {
    // Copy so the transfer cannot detach a buffer the caller still holds.
    const buf = rgba.slice().buffer;
    return this.send<{ ms: number }>({ type: "encode", rgba: buf, width, height }, [buf]);
  }

  async decode(points: [number, number][], labels: number[]): Promise<DecodedMask> {
    const r = await this.send<{
      mask: ArrayBuffer; w: number; h: number; score: number; areaFrac: number; ms: number;
    }>({ type: "decode", points, labels });
    return {
      mask: new Float32Array(r.mask), w: r.w, h: r.h,
      score: r.score, areaFrac: r.areaFrac, ms: r.ms,
    };
  }

  destroy() {
    this.worker.terminate();
    this.pending.clear();
  }
}
