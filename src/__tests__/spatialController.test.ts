import { beforeEach, describe, expect, it } from "vitest";
import { SpatialController } from "../ml/spatialController";
import { buildPatchGrid } from "../ml/patchGrid";
import { useSpatial } from "../ml/spatialStore";
import type { ModelSpec } from "../ml/registry";
import type { SlideMeta, SlideSource } from "../slide/types";

/**
 * A worker that answers the way the real one does, recording what it was asked.
 *
 * The forward pass is three lines of ONNX Runtime; what is worth checking is
 * everything around it — that tiles are read at the magnification the model was
 * trained at, that batches cover every patch exactly once, and that a run which
 * is stopped does not leave a half-filled map on screen looking complete.
 */
class FakeWorker implements Partial<Worker> {
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onerror: ((ev: ErrorEvent) => void) | null = null;
  readonly batches: number[] = [];
  readonly tileSizes: number[] = [];

  postMessage(msg: Record<string, unknown>) {
    queueMicrotask(() => {
      if (msg.type === "load") {
        this.onmessage?.({ data: { type: "loaded", id: msg.id, backend: "wasm", genes: 2 } } as MessageEvent);
        return;
      }
      const count = msg.count as number;
      this.batches.push(count);
      this.tileSizes.push(msg.size as number);
      // One value per gene per tile, increasing, so mis-ordering is visible.
      const values = new Float32Array(count * 2);
      for (let i = 0; i < values.length; i++) values[i] = i;
      this.onmessage?.({
        data: { type: "predicted", id: msg.id, values: values.buffer, genes: 2, ms: 1 },
      } as MessageEvent);
    });
  }
  terminate() { /* nothing to release */ }
}

const meta: SlideMeta = {
  name: "case.svs", bytes: 0, vendor: "test", mppX: 0.25, mppY: 0.25, objectivePower: 40,
  bounds: null, backgroundColor: null,
  levels: [
    { level: 0, width: 40000, height: 20000, downsample: 1 },   // 0.25 µm/px
    { level: 1, width: 20000, height: 10000, downsample: 2 },   // 0.50 µm/px
    { level: 2, width: 10000, height: 5000, downsample: 4 },    // 1.00 µm/px
  ],
  properties: {},
};

const reads: { level: number; w: number; h: number }[] = [];
const source: SlideSource = {
  meta,
  async readRegion(_x, _y, level, w, h) {
    reads.push({ level, w, h });
    return new Uint8ClampedArray(w * h * 4).fill(200);
  },
  async bestLevelForDownsample() { return 0; },
  async close() { /* nothing */ },
};

const spec: ModelSpec = {
  id: "deepspot-m-2", name: "DeepSpot-M (2 genes)", task: "virtual-spatial",
  blurb: "", files: [{ part: "model", url: "local://m", bytes: 1 }],
  inputSize: 224, mean: [0, 0, 0], std: [255, 255, 255],
  targetMpp: 0.5, genes: ["EPCAM", "CD3D"], licence: "test", backend: "wasm",
};

const region = { x: 0, y: 0, width: 224 * 2 * 5, height: 224 * 2 * 2 };

beforeEach(() => {
  reads.length = 0;
  useSpatial.setState({ result: null, gene: null, status: "idle", progress: null });
});

describe("running a spatial model over a grid", () => {
  it("reads at the level nearest the magnification it was trained at", async () => {
    const fake = new FakeWorker();
    const c = new SpatialController(source, () => fake as unknown as Worker);
    const grid = buildPatchGrid(region, 2, meta.mppX, { patchPx: 224, level: 1 });

    await c.run(grid, spec, null);

    // 0.5 µm/px is level 1 on this slide; reading level 0 would show the model
    // half the tissue it expects, which changes the answer without failing.
    expect(reads.every((r) => r.level === 1)).toBe(true);
    expect(reads.every((r) => r.w === 224 && r.h === 224)).toBe(true);
    expect(fake.tileSizes.every((s) => s === 224)).toBe(true);
  });

  it("covers every patch exactly once", async () => {
    const fake = new FakeWorker();
    const c = new SpatialController(source, () => fake as unknown as Worker);
    const grid = buildPatchGrid(region, 2, meta.mppX, { patchPx: 224, level: 1 });

    const result = await c.run(grid, spec, null);

    expect(reads.length).toBe(grid.patches.length);
    expect(fake.batches.reduce((a, b) => a + b, 0)).toBe(grid.patches.length);
    expect(result!.values.length).toBe(grid.patches.length * spec.genes!.length);
    expect(result!.patches.length).toBe(grid.patches.length);
  });

  it("records the ROI it was scoped to", async () => {
    const fake = new FakeWorker();
    const c = new SpatialController(source, () => fake as unknown as Worker);
    const grid = buildPatchGrid(region, 2, meta.mppX, { patchPx: 224, level: 1 });

    const result = await c.run(grid, spec, "roi-7");
    expect(result!.roiId).toBe("roi-7");
    expect(result!.modelId).toBe(spec.id);
    expect(result!.slide).toBe("case.svs");
  });

  it("selects a gene so the map is visible without a second action", async () => {
    const fake = new FakeWorker();
    const c = new SpatialController(source, () => fake as unknown as Worker);
    const grid = buildPatchGrid(region, 2, meta.mppX, { patchPx: 224, level: 1 });

    await c.run(grid, spec, null);
    expect(useSpatial.getState().gene).toBe("EPCAM");
  });

  /**
   * A stopped run must leave nothing behind. Half a map drawn over the tissue
   * is indistinguishable from a finished one, and would be read as a result.
   */
  it("publishes nothing when the run is stopped part way", async () => {
    const fake = new FakeWorker();
    const c = new SpatialController(source, () => fake as unknown as Worker);
    const grid = buildPatchGrid(region, 2, meta.mppX, { patchPx: 224, level: 1 });
    expect(grid.patches.length).toBeGreaterThan(8); // more than one batch

    // Stop it the way the button does: after the first batch has come back.
    const original = fake.postMessage.bind(fake);
    let seen = 0;
    fake.postMessage = (msg: Record<string, unknown>) => {
      if (msg.type === "predict" && ++seen === 1) c.cancel();
      original(msg);
    };

    const result = await c.run(grid, spec, null);
    expect(result).toBe(null);
    expect(useSpatial.getState().result).toBe(null);
    // And it stopped early rather than quietly finishing the whole grid.
    expect(reads.length).toBeLessThan(grid.patches.length);
  });

  it("treats a stop with nothing running as a no-op", () => {
    const fake = new FakeWorker();
    const c = new SpatialController(source, () => fake as unknown as Worker);
    // Pressing stop before starting must not poison the next run.
    expect(() => c.cancel()).not.toThrow();
  });

  it("refuses a model that does not say which genes it predicts", async () => {
    const fake = new FakeWorker();
    const c = new SpatialController(source, () => fake as unknown as Worker);
    const grid = buildPatchGrid(region, 2, meta.mppX, { patchPx: 224, level: 1 });

    const result = await c.run(grid, { ...spec, genes: undefined }, null);
    expect(result).toBe(null);
    expect(useSpatial.getState().status).toBe("error");
    expect(useSpatial.getState().error).toMatch(/genes/i);
  });

  it("says so rather than running on an empty grid", async () => {
    const fake = new FakeWorker();
    const c = new SpatialController(source, () => fake as unknown as Worker);
    const empty = buildPatchGrid({ x: 0, y: 0, width: 10, height: 10 }, 2, meta.mppX, { patchPx: 224, level: 1 });

    expect(empty.patches.length).toBe(0);
    expect(await c.run(empty, spec, null)).toBe(null);
    expect(useSpatial.getState().status).toBe("error");
  });
});

describe("the encoder actually reaches the worker", () => {
  /**
   * This regressed in the prediction controller: the load was written as
   * `spec.dim ?? await load(spec)`, so any export declaring its embedding
   * width — which is all of them — skipped loading entirely. The worker never
   * received the model, every batch failed with "No encoder is loaded", and
   * the panel sat at zero patches looking merely slow.
   *
   * A declared width and a loaded session are different facts. Nothing that
   * answers the first should be allowed to stand in for the second.
   */
  it("loads the model even when the spec already declares its width", async () => {
    const { PredictController } = await import("../ml/predictController");
    const seen: string[] = [];

    class Recorder implements Partial<Worker> {
      onmessage: ((ev: MessageEvent) => void) | null = null;
      onerror: ((ev: ErrorEvent) => void) | null = null;
      postMessage(msg: Record<string, unknown>) {
        seen.push(msg.type as string);
        queueMicrotask(() => {
          if (msg.type === "load") {
            this.onmessage?.({
              data: { type: "loaded", id: msg.id, backend: "wasm", dim: 2560 },
            } as MessageEvent);
          } else {
            const count = msg.count as number;
            const vectors = new Float32Array(count * 2560);
            this.onmessage?.({
              data: { type: "embedded", id: msg.id, vectors: vectors.buffer, dim: 2560, ms: 1 },
            } as MessageEvent);
          }
        });
      }
      terminate() { /* nothing to release */ }
    }

    const recorder = new Recorder();
    const c = new PredictController(source, () => recorder as unknown as Worker);
    const grid = buildPatchGrid(region, 2, meta.mppX, { patchPx: 224, level: 1 });

    // A spec that declares its width, exactly as every export does.
    await c.embed(grid, { ...spec, id: "virchow2", dim: 2560, genes: undefined });

    expect(seen[0]).toBe("load");
    expect(seen.filter((t) => t === "embed").length).toBeGreaterThan(0);
  }, 30000);
});

describe("stopping an embedding run", () => {
  /**
   * A flag checked between batches is not a stop button. The loop spends
   * nearly all its time inside a batch, so pressing stop did nothing until
   * that batch returned — minutes, on a ViT-H under wasm.
   */
  it("unblocks immediately, mid-batch, and does not report a failure", async () => {
    const { PredictController } = await import("../ml/predictController");
    const { usePredict } = await import("../ml/predictStore");

    // A worker that accepts the load and then never answers an embed.
    class Silent implements Partial<Worker> {
      onmessage: ((ev: MessageEvent) => void) | null = null;
      onerror: ((ev: ErrorEvent) => void) | null = null;
      embeds = 0;
      postMessage(msg: Record<string, unknown>) {
        if (msg.type === "load") {
          queueMicrotask(() =>
            this.onmessage?.({
              data: { type: "loaded", id: msg.id, backend: "wasm", dim: 2560 },
            } as MessageEvent),
          );
        } else {
          this.embeds++; // and never replies
        }
      }
      terminate() { /* nothing to release */ }
    }

    const silent = new Silent();
    const c = new PredictController(source, () => silent as unknown as Worker);
    const grid = buildPatchGrid(region, 2, meta.mppX, { patchPx: 224, level: 1 });

    const run = c.embed(grid, { ...spec, id: "virchow2", dim: 2560, genes: undefined });
    // Let it reach the first batch and hang there.
    await new Promise((r) => setTimeout(r, 30));
    expect(silent.embeds).toBeGreaterThan(0);

    c.cancel();
    const result = await run;

    expect(result).toBe(null);
    // A stop is not a failure and must not be shown as one.
    expect(usePredict.getState().status).not.toBe("error");
  }, 30000);
});
