import OpenSlide, { applyAlpha } from "@conflux-xyz/openslide-wasm";
import type { OpenSlideImage, FileEntry } from "@conflux-xyz/openslide-wasm";
import type { LevelInfo, ResolvedSlide, SlideMeta, SlideSource } from "./types";

/**
 * Worker pools, keyed by size.
 *
 * Each worker holds its own OpenSlide handle *and its own decode cache*, so a
 * pool does not share work — it multiplies it. Measured on a 370 MB NDPI whose
 * coarsest level is only 32x: decoding that overview costs ~29 s in wasm, and
 * a four-worker pool paid it up to four times before anything appeared. The
 * same read takes ~2 s in native OpenSlide.
 *
 * So the pool size is decided per slide, from a measurement.
 */
const runtimes = new Map<number, Promise<OpenSlide>>();

const POOL_SIZE = Math.max(2, Math.min(4, (navigator.hardwareConcurrency ?? 4) - 1));

/**
 * Overview decode time above which a pool is a net loss: the duplicated decode
 * costs far more than the parallelism can return, so the slide stays on one
 * already-warm worker.
 */
const FAST_OVERVIEW_MS = 2000;

let activeWorkers = 1;

export function getWorkerCount(): number {
  return activeWorkers;
}

function getRuntime(workers: number): Promise<OpenSlide> {
  let rt = runtimes.get(workers);
  if (!rt) {
    rt = (async () => {
      const os = new OpenSlide({ workers });
      await os.initialize();
      return os;
    })();
    runtimes.set(workers, rt);
  }
  return rt;
}

/**
 * Recently opened slides, most-recently-used last.
 *
 * Opening is expensive and the cost is *per handle*, so closing a slide the
 * moment you click another one means paying it again on the way back — 29 s
 * each way on a shallow-pyramid NDPI. Keeping a few open makes switching
 * between slides in a study instant, which is the whole point of a slide list.
 */
const openCache = new Map<string, SlideSource & { workers: number }>();
const MAX_OPEN_SLIDES = 3;

const cacheKey = (slide: ResolvedSlide) => `${slide.entryPath}|${slide.bytes}`;

/**
 * Drop one slide from the cache and release its decoder handles.
 *
 * Removing a slide from the list should give the memory back rather than
 * leaving several hundred megabytes of mounted file and decode cache pinned in
 * the worker pool for a slide nobody can reach any more.
 */
export async function closeSlide(slide: ResolvedSlide): Promise<void> {
  const key = cacheKey(slide);
  const source = openCache.get(key);
  if (!source) return;
  openCache.delete(key);
  await source.close().catch(() => undefined);
}

/** Close every cached slide; for tests and teardown. */
export async function closeAllSlides(): Promise<void> {
  const sources = [...openCache.values()];
  openCache.clear();
  await Promise.all(sources.map((s) => s.close().catch(() => undefined)));
}

export interface OpenProgress {
  phase: "opening" | "reading-metadata" | "warming";
}

const num = (v: string | null): number | null => {
  if (v === null) return null;
  const n = Number.parseFloat(v);
  return Number.isFinite(n) ? n : null;
};

class OpenSlideSource implements SlideSource {
  constructor(
    private readonly image: OpenSlideImage,
    readonly meta: SlideMeta,
    /** Size of the pool this slide was opened on. */
    readonly workers: number,
  ) {}

  async readRegion(
    x: number,
    y: number,
    level: number,
    width: number,
    height: number,
    signal?: AbortSignal,
  ): Promise<Uint8ClampedArray> {
    const raw = await this.image.readRegion(x, y, level, width, height, { signal });
    // OpenSlide returns premultiplied ARGB; composite onto white so that the
    // padding outside a MIRAX scan region reads as slide background, not black.
    return applyAlpha(raw, { backgroundColor: [255, 255, 255], inPlace: true });
  }

  bestLevelForDownsample(downsample: number): Promise<number> {
    return this.image.getBestLevelForDownsample(downsample);
  }

  close(): Promise<void> {
    return this.image.close();
  }
}

const KNOWN = {
  vendor: "openslide.vendor",
  mppX: "openslide.mpp-x",
  mppY: "openslide.mpp-y",
  objective: "openslide.objective-power",
  background: "openslide.background-color",
  boundsX: "openslide.bounds-x",
  boundsY: "openslide.bounds-y",
  boundsW: "openslide.bounds-width",
  boundsH: "openslide.bounds-height",
} as const;

async function readMeta(image: OpenSlideImage, slide: ResolvedSlide): Promise<SlideMeta> {
  const names = await image.getPropertyNames();
  const properties: Record<string, string> = {};
  await Promise.all(
    names.map(async (n) => {
      const v = await image.getPropertyValue(n);
      if (v !== null) properties[n] = v;
    }),
  );

  const levelCount = await image.getLevelCount();
  const levels: LevelInfo[] = await Promise.all(
    Array.from({ length: levelCount }, async (_, level): Promise<LevelInfo> => {
      const [width, height] = await image.getLevelDimensions(level);
      return { level, width, height, downsample: await image.getLevelDownsample(level) };
    }),
  );

  const bx = num(properties[KNOWN.boundsX] ?? null);
  const by = num(properties[KNOWN.boundsY] ?? null);
  const bw = num(properties[KNOWN.boundsW] ?? null);
  const bh = num(properties[KNOWN.boundsH] ?? null);

  return {
    name: slide.name,
    bytes: slide.bytes,
    vendor: properties[KNOWN.vendor] ?? null,
    mppX: num(properties[KNOWN.mppX] ?? null),
    mppY: num(properties[KNOWN.mppY] ?? null),
    objectivePower: num(properties[KNOWN.objective] ?? null),
    bounds:
      bx !== null && by !== null && bw !== null && bh !== null
        ? { x: bx, y: by, width: bw, height: bh }
        : null,
    backgroundColor: properties[KNOWN.background] ?? null,
    levels,
    properties,
  };
}

export async function openSlide(
  slide: ResolvedSlide,
  onProgress?: (p: OpenProgress) => void,
): Promise<SlideSource> {
  const key = cacheKey(slide);
  const cached = openCache.get(key);
  if (cached) {
    // Refresh recency without reopening.
    openCache.delete(key);
    openCache.set(key, cached);
    activeWorkers = cached.workers;
    return cached;
  }

  onProgress?.({ phase: "opening" });

  // The entry file must be first — openslide-wasm opens files[0].
  const ordered = [...slide.files].sort((a, b) =>
    a.path === slide.entryPath ? -1 : b.path === slide.entryPath ? 1 : 0,
  );
  const entries: FileEntry[] = ordered.map((f) => ({ path: f.path, file: f.file }));

  // Probe on ONE worker. A cheap slide loses nothing by being measured; an
  // expensive one is spared having that cost multiplied across a pool.
  const probeRuntime = await getRuntime(1);
  const probe = await probeRuntime.open(entries);

  onProgress?.({ phase: "reading-metadata" });
  const meta = await readMeta(probe, slide);

  const coarsest = meta.levels[meta.levels.length - 1];
  const w = Math.min(coarsest.width, 2048);
  const h = Math.min(coarsest.height, 2048);

  // Decoding the overview is both the measurement and the warm-up: whatever
  // happens next, this worker's cache now holds the opening view.
  onProgress?.({ phase: "warming" });
  const started = performance.now();
  await probe.readRegion(0, 0, coarsest.level, w, h).catch(() => undefined);
  const overviewMs = performance.now() - started;

  if (overviewMs > FAST_OVERVIEW_MS) {
    activeWorkers = 1;
    console.info(
      `[slidecraft] ${slide.name}: overview decoded in ${Math.round(overviewMs)} ms — ` +
        `staying on a single worker to avoid repeating that cost.`,
    );
    return remember(key, new OpenSlideSource(probe, meta, 1));
  }

  activeWorkers = POOL_SIZE;
  const os = await getRuntime(POOL_SIZE);
  const image = await os.open(entries);
  void probe.close().catch(() => undefined);
  // Prime the pool in the background so the first pans do not stutter.
  void (async () => {
    for (let i = 0; i < POOL_SIZE; i++) {
      await image.readRegion(0, 0, coarsest.level, w, h).catch(() => undefined);
    }
  })();

  return remember(key, new OpenSlideSource(image, meta, POOL_SIZE));
}

function remember(key: string, source: SlideSource & { workers: number }): SlideSource {
  openCache.set(key, source);
  // Evict least-recently-used beyond the cap and release its handles.
  while (openCache.size > MAX_OPEN_SLIDES) {
    const oldest = openCache.keys().next().value as string;
    const evicted = openCache.get(oldest);
    openCache.delete(oldest);
    void evicted?.close().catch(() => undefined);
  }
  return source;
}
