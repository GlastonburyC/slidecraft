import type OpenSeadragon from "openseadragon";
import { makeAnnotation, queryBox, useAnnotations } from "../annotate/store";
import { booleanOp } from "../annotate/geometry";
import type { Annotation } from "../annotate/types";
import {
  areaOf,
  isAreaGeometry,
  ROI_CLASS_ID,
  simplifyRing,
  type AreaGeometry,
  type Position,
  type Ring,
} from "../annotate/types";
import type { SlideSource } from "../slide/types";
import type { AnnotationOverlay } from "../viewer/annotationOverlay";
import { largestRing, traceMask } from "./contour";
import { detectTissue, type TissueOptions } from "./tissue";
import { allModelUrls, BUILTIN_MODELS, findModel, totalBytes } from "./registry";
import { loadLocalModels } from "./localModels";
import { fetchWeights, isCached, requestPersistence } from "./modelCache";
import { SamClient } from "./samClient";
import { useMl } from "./mlStore";

/** Largest region we will read and encode in one pass, in slide pixels. */
const MAX_READ = 2048;

/**
 * A neighbour is treated as "another cell" — and therefore clipped against —
 * when its area is within this factor of the new mask. Keeps clicking a
 * nucleus from carving a hole out of a large tissue-class region that happens
 * to sit underneath.
 */
const NEIGHBOUR_AREA_FACTOR = 8;

/**
 * Marks an annotation as tissue detection, whichever thing decided it.
 *
 * Two jobs pull in opposite directions here. Each detection should record the
 * model that actually produced it, so an exported GeoJSON says whether the
 * colour rule or your own classifier drew that boundary. But re-running has to
 * replace the previous run, and if that match is on the exact model id then
 * detecting, training, and detecting again leaves both runs stacked on top of
 * each other — the duplicate-tissue bug, back by another route.
 *
 * So the id carries the specific model and the shared prefix identifies the
 * detector. Both id shapes are minted here and in `trainTissueModel`, which is
 * what makes the prefix a guarantee rather than a coincidence.
 */
const TISSUE_PREFIX = "tissue-";

/** The heuristic detector, used when no model is active. */
const TISSUE_MODEL_ID = `${TISSUE_PREFIX}otsu`;

export const isTissueDetection = (a: { modelId?: string; source: string }) =>
  a.source === "model" && !!a.modelId?.startsWith(TISSUE_PREFIX);

/**
 * Click-to-segment.
 *
 * The encoder runs once over the visible region; each click after that is a
 * decoder pass on the cached embedding, so segmenting the next cell costs
 * milliseconds rather than seconds. Clicking a fresh cell commits the previous
 * mask, which makes "click every nucleus" a continuous motion rather than a
 * click-confirm-click-confirm loop.
 */
export class SegmentController {
  private client: SamClient | null = null;
  private pendingRing: Ring | null = null;
  private decodeSeq = 0;

  constructor(
    private readonly viewer: OpenSeadragon.Viewer,
    private readonly source: SlideSource,
    private readonly overlay: AnnotationOverlay,
    private readonly offset: { x: number; y: number },
  ) {}

  private ensureClient(): SamClient {
    if (!this.client) this.client = new SamClient();
    return this.client;
  }

  /** Refresh which models are already on disk, so the UI can say so. */
  async refreshCacheStatus(): Promise<void> {
    // Imported models live in IndexedDB, so the registry has to be rebuilt from
    // it each session rather than assumed.
    const local = await loadLocalModels();
    if (local.length > 0) {
      useMl.getState().setModels([...BUILTIN_MODELS, ...local]);
    }
    const { models } = useMl.getState();
    const flags = await Promise.all(models.map((m) => isCached(allModelUrls(m))));
    useMl.getState().setCachedIds(models.filter((_, i) => flags[i]).map((m) => m.id));
    useMl.getState().setPersisted(await navigator.storage?.persisted?.().catch(() => false) ?? false);
  }

  /**
   * Fetch every model's weights up front so switching later is instant and
   * offline. Downloads run in the main thread's cache; the worker picks them up
   * from there without refetching.
   */
  async prefetchAll(): Promise<void> {
    const ml = useMl.getState();
    if (ml.prefetch) return;
    await requestPersistence();

    const models = ml.models;
    const grand = models.reduce((n, m) => n + totalBytes(m), 0);
    let done = 0;
    try {
      for (const m of models) {
        for (const f of m.files) {
          const base = done;
          await fetchWeights(f.url, f.part, f.bytes, (p) => {
            useMl.getState().setPrefetch({
              modelName: m.name,
              done: base + p.received,
              total: grand,
            });
          });
          done = base + f.bytes;
        }
        await this.refreshCacheStatus();
      }
    } catch (err) {
      useMl.getState().setStatus("error", err instanceof Error ? err.message : String(err));
    } finally {
      useMl.getState().setPrefetch(null);
      await this.refreshCacheStatus();
    }
  }

  /** Download + compile the active model. Safe to call repeatedly. */
  async loadModel(): Promise<void> {
    const ml = useMl.getState();
    if (ml.status === "ready" || ml.status === "downloading" || ml.status === "compiling") return;
    const spec = findModel(ml.models, ml.activeModelId);
    if (!spec) return;

    ml.setStatus("downloading");
    ml.setProgress(0);
    void requestPersistence();
    const totals = new Map<string, { received: number; total: number }>();

    try {
      const res = await this.ensureClient().load(spec, (part, received, total) => {
        totals.set(part, { received, total });
        let got = 0;
        let all = 0;
        totals.forEach((v) => { got += v.received; all += v.total; });
        useMl.getState().setProgress(all > 0 ? got / all : 0);
        if (got >= all) useMl.getState().setStatus("compiling");
      });
      useMl.getState().setBackend(res.backend);
      useMl.getState().setStatus("ready");
      void this.refreshCacheStatus();
    } catch (err) {
      useMl.getState().setStatus("error", err instanceof Error ? err.message : String(err));
    }
  }

  /** Region currently on screen, in level-0 slide pixels. */
  private viewRegion(): Region {
    const vp = this.viewer.viewport;
    const r = vp.viewportToImageRectangle(vp.getBounds(true));
    return {
      x: Math.max(0, Math.floor(r.x + this.offset.x)),
      y: Math.max(0, Math.floor(r.y + this.offset.y)),
      width: Math.ceil(r.width),
      height: Math.ceil(r.height),
    };
  }

  /**
   * Encode an arbitrary slide region.
   *
   * Resolution is chosen so the read stays within MAX_READ per axis: a small
   * ROI is encoded at or near native magnification, a large one is encoded
   * coarser. That is the honest trade — SAM sees a fixed 1024px canvas either
   * way, so a smaller region simply means more detail per cell.
   */
  private async encodeRegion(region: Region, origin: EncodedOrigin): Promise<void> {
    const ml = useMl.getState();
    if (ml.status !== "ready" || ml.encoding) return;

    const levels = this.source.meta.levels;
    const wanted = Math.max(region.width, region.height) / MAX_READ;
    // Highest-resolution level whose downsample still satisfies the budget.
    let osLevel = 0;
    for (let i = 0; i < levels.length; i++) {
      if (levels[i].downsample <= Math.max(1, wanted) + 1e-6) osLevel = i;
      else break;
    }
    const ds = levels[osLevel].downsample;
    const readW = Math.max(1, Math.min(MAX_READ, Math.round(region.width / ds)));
    const readH = Math.max(1, Math.min(MAX_READ, Math.round(region.height / ds)));

    useMl.getState().setEncoding(true);
    try {
      const rgba = await this.source.readRegion(region.x, region.y, osLevel, readW, readH);
      const { ms } = await this.ensureClient().encode(rgba, readW, readH);
      useMl.getState().setEncoded({ ...region, readW, readH, ms, ...origin });
      useMl.getState().setPrompt([]);
      this.pendingRing = null;
      this.overlay.setDraft(null);
    } catch (err) {
      useMl.getState().setStatus("error", err instanceof Error ? err.message : String(err));
    } finally {
      useMl.getState().setEncoding(false);
    }
  }

  /**
   * Outline the tissue and commit it as annotations.
   *
   * Needs no model and no download: it thresholds the overview the slide
   * already decoded when it opened.
   */
  async detectTissue(classId: string | null, opts: TissueOptions = {}): Promise<number> {
    const res = await detectTissue(this.source, opts);
    const store = useAnnotations.getState();
    const added = res.polygons
      .filter((rings) => rings[0] && rings[0].length >= 4)
      .map((rings) =>
        makeAnnotation(
          { type: "Polygon", coordinates: rings },
          { classId, source: "model", modelId: opts.model?.id ?? TISSUE_MODEL_ID },
        ),
      );

    // Re-running replaces the previous result rather than stacking a second
    // copy on top of it. Only this detector's own output is cleared — hand-drawn
    // regions and locked ones are left alone.
    const previous = [...store.items.values()].filter((a) => isTissueDetection(a) && !a.locked);

    if (added.length > 0 || previous.length > 0) {
      store.apply({
        label: previous.length
          ? `Re-detect tissue (${added.length})`
          : `Detect tissue (${added.length})`,
        removed: previous,
        added,
      });
    }
    useMl.getState().setTissue({
      count: added.length,
      coverage: res.coverage,
      threshold: res.saturationThreshold,
      rejected: res.rejected,
      modelName: opts.model?.name ?? null,
      ms: res.ms,
    });
    return added.length;
  }

  /** Encode whatever is on screen so subsequent clicks are instant. */
  encodeView(): Promise<void> {
    return this.encodeRegion(this.viewRegion(), { origin: "view", roiId: null });
  }

  /**
   * Encode a region of interest instead of the viewport.
   *
   * Scoping to an ROI is what makes the work reproducible: the encoded pixels
   * stay fixed while you pan and zoom within it, so every mask in that ROI came
   * from the same embedding rather than from whatever happened to be on screen.
   */
  encodeRoi(roi: Annotation): Promise<void> {
    const [minX, minY, maxX, maxY] = roi.bbox;
    return this.encodeRegion(
      {
        x: Math.max(0, Math.floor(minX)),
        y: Math.max(0, Math.floor(minY)),
        width: Math.max(1, Math.ceil(maxX - minX)),
        height: Math.max(1, Math.ceil(maxY - minY)),
      },
      { origin: "roi", roiId: roi.id },
    );
  }

  private inEncodedRegion(p: Position): boolean {
    const e = useMl.getState().encoded;
    if (!e) return false;
    return p[0] >= e.x && p[1] >= e.y && p[0] < e.x + e.width && p[1] < e.y + e.height;
  }

  /**
   * A click while the segment tool is active.
   * plain  = commit any pending mask, then start a new one here
   * shift  = add an include point to the current prompt
   * alt    = add an exclude point to the current prompt
   */
  async handleClick(p: Position, mods: { shift: boolean; alt: boolean }): Promise<void> {
    const ml = useMl.getState();
    if (ml.status !== "ready") { void this.loadModel(); return; }

    if (!this.inEncodedRegion(p)) {
      // An encoded ROI defines the working area: silently re-encoding the
      // viewport would throw away the embedding the ROI's masks came from.
      if (ml.encoded?.origin === "roi") {
        useMl.getState().setOutsideRoi(true);
        return;
      }
      await this.encodeView();
      if (!this.inEncodedRegion(p)) return;
    }
    useMl.getState().setOutsideRoi(false);

    let prompt = useMl.getState().prompt;
    if (!mods.shift && !mods.alt) {
      if (useMl.getState().autoCommit) this.commit();
      prompt = [{ x: p[0], y: p[1], label: 1 }];
    } else {
      prompt = [...prompt, { x: p[0], y: p[1], label: mods.alt ? 0 : 1 }];
    }
    useMl.getState().setPrompt(prompt);
    await this.decode();
  }

  private async decode(): Promise<void> {
    const ml = useMl.getState();
    const e = ml.encoded;
    if (!e || ml.prompt.length === 0) return;

    const seq = ++this.decodeSeq;
    // Slide pixels -> the read's own pixel space, which is what SAM was given.
    const sx = e.readW / e.width;
    const sy = e.readH / e.height;
    const points = ml.prompt.map((q): [number, number] => [
      (q.x - e.x) * sx,
      (q.y - e.y) * sy,
    ]);
    const labels = ml.prompt.map((q) => q.label);

    try {
      const res = await this.ensureClient().decode(points, labels);
      if (seq !== this.decodeSeq) return; // a newer click superseded this one

      const rings = traceMask(res.mask, res.w, res.h, { threshold: 0, simplify: 0.75, minArea: 8 });
      const ring = largestRing(rings);
      useMl.getState().setDecodeStats(res.ms, res.score, res.areaFrac);

      if (!ring) {
        this.pendingRing = null;
        useMl.getState().setEmpty(true);
        this.overlay.setDraft(null);
        return;
      }
      useMl.getState().setEmpty(false);

      // Mask space -> read space -> level-0 slide pixels.
      const toSlide: Ring = ring.map(([mx, my]) => [
        e.x + (mx / res.w) * e.width,
        e.y + (my / res.h) * e.height,
      ]);
      const simplified = simplifyRing(toSlide, Math.max(0.5, e.width / e.readW));

      const clipped = useMl.getState().nonOverlapping
        ? this.clipAgainstNeighbours({ type: "Polygon", coordinates: [simplified] })
        : ({ type: "Polygon", coordinates: [simplified] } as AreaGeometry);

      if (!clipped) {
        // The click landed inside a cell that is already segmented.
        this.pendingRing = null;
        useMl.getState().setEmpty(true);
        this.overlay.setDraft(null);
        return;
      }
      this.pendingRing = outerRingOf(clipped);

      const cls = useAnnotations.getState().classes.find(
        (c) => c.id === useAnnotations.getState().activeClassId,
      );
      this.overlay.setDraft({
        kind: "polygon",
        rings: this.pendingRing ? [this.pendingRing] : [],
        color: cls?.color ?? [80, 220, 200],
      });
    } catch (err) {
      useMl.getState().setStatus("error", err instanceof Error ? err.message : String(err));
    }
  }

  /**
   * Subtract neighbouring cells so instances never overlap.
   *
   * Only comparable-sized, non-ROI neighbours are subtracted: a nucleus should
   * be clipped by the nucleus beside it, not by the tissue region or ROI it
   * happens to sit inside.
   */
  private clipAgainstNeighbours(mask: AreaGeometry): AreaGeometry | null {
    const own = areaOf(mask);
    if (own <= 0) return null;

    const [minX, minY, maxX, maxY] = bboxOf(mask);
    const neighbours = queryBox(minX, minY, maxX, maxY).filter(
      (a) =>
        a.classId !== ROI_CLASS_ID &&
        !a.locked &&
        isAreaGeometry(a.geometry) &&
        areaOf(a.geometry) <= own * NEIGHBOUR_AREA_FACTOR,
    );

    let result: AreaGeometry | null = mask;
    for (const n of neighbours) {
      if (!result) return null;
      result = booleanOp("subtract", result, n.geometry as AreaGeometry);
    }
    if (!result) return null;

    // Subtraction can shatter the mask; a cell is one piece, so keep the
    // largest and drop slivers left behind between neighbours.
    if (result.type === "MultiPolygon") {
      const best = result.coordinates.reduce((a, b) =>
        Math.abs(areaOf({ type: "Polygon", coordinates: b })) >
        Math.abs(areaOf({ type: "Polygon", coordinates: a }))
          ? b
          : a,
      );
      result = { type: "Polygon", coordinates: best };
    }
    // Anything under a tenth of the original is a leftover edge, not a cell.
    return areaOf(result) > own * 0.1 ? result : null;
  }

  /** Turn the pending mask into a real annotation. */
  commit(): boolean {
    const ring = this.pendingRing;
    if (!ring || ring.length < 4) return false;
    this.pendingRing = null;
    this.overlay.setDraft(null);

    const store = useAnnotations.getState();
    const a = makeAnnotation(
      { type: "Polygon", coordinates: [ring] },
      {
        classId: store.activeClassId,
        source: "model",
        modelId: useMl.getState().activeModelId,
        confidence: useMl.getState().lastScore ?? undefined,
      },
    );
    store.apply({ label: "Segment", added: [a] });
    useMl.getState().setPrompt([]);
    return true;
  }

  discard() {
    this.pendingRing = null;
    useMl.getState().setPrompt([]);
    this.overlay.setDraft(null);
  }

  hasPending(): boolean {
    return this.pendingRing !== null;
  }

  destroy() {
    this.client?.destroy();
    this.client = null;
  }
}

function outerRingOf(g: AreaGeometry): Ring {
  return g.type === "Polygon" ? g.coordinates[0] : g.coordinates[0][0];
}

function bboxOf(g: AreaGeometry): [number, number, number, number] {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const rings = g.type === "Polygon" ? g.coordinates : g.coordinates.flat();
  for (const ring of rings) {
    for (const [x, y] of ring) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  return [minX, minY, maxX, maxY];
}

interface Region {
  x: number;
  y: number;
  width: number;
  height: number;
}

type EncodedOrigin = { origin: "view" | "roi"; roiId: string | null };
