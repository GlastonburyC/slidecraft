import { largestRing, ringArea, traceMask } from "./contour";
import { cleanGeometry } from "../annotate/geometry";
import { computeFeatures } from "./tissueFeatures";
import { predictAll, type TissueModel } from "./tissueModel";
import type { Ring } from "../annotate/types";
import type { LevelInfo, SlideSource } from "../slide/types";

/**
 * Tissue detection, without a model.
 *
 * H&E background is bright and colourless; tissue is not. Saturation separates
 * them far more reliably than brightness alone, which confuses pale tissue with
 * background and dark artefacts (pen marks, coverslip edges, dust) with tissue.
 * Otsu picks the threshold from the slide's own histogram, so it adapts to
 * staining strength rather than relying on a tuned constant.
 *
 * It runs on the overview level, which the slide already decoded on open — so
 * this costs almost nothing and needs no download.
 */

export interface TissueOptions {
  /** Discard regions smaller than this, in µm² (null = keep everything). */
  minAreaUm2?: number;
  /** Override the automatic threshold, 0-1. */
  saturation?: number;
  /**
   * Where the tissue boundary sits between the blank glass and the automatic
   * threshold, 0-1. Lower moves the boundary outward to capture faded edges;
   * too low and the glass itself creeps in.
   */
  edgeBias?: number;
  /**
   * Reject long thin strips above this length:width ratio — scanner seams and
   * coverslip edges. Only applied to regions that are ALSO narrower than
   * `minWidthUm`, because a needle core is legitimately 20:1 and must survive.
   */
  maxAspect?: number;
  /** Width below which a very elongated region is treated as an artefact. */
  minWidthUm?: number;
  /**
   * Structures this many overview cells thick or less are removed before
   * anything else. 0 disables it.
   */
  seamRadius?: number;
  /**
   * How far apart two pieces of confident tissue may be and still count as one
   * fragment, in overview cells. Above the speckle inside a section, below the
   * gap between sections.
   */
  bindRadius?: number;
  /**
   * Largest enclosed gap to treat as part of the tissue, in µm². A gland lumen
   * is far below this; the blank slide between two fragments is far above it.
   */
  maxHoleUm2?: number;
  /**
   * A trained tissue / not-tissue classifier. When present it decides which
   * cells are tissue and the automatic threshold is not consulted, so a model
   * taught that a pale smooth smear is not tissue overrides the colour rule
   * that cannot see the difference.
   */
  model?: TissueModel | null;
  /** Probability above which the classifier calls a cell tissue. */
  modelThreshold?: number;
  /** An overview already read, so training and detection share one grid. */
  grid?: OverviewGrid;
  /**
   * Where the growth stops, on the same glass-to-automatic scale as
   * `edgeBias` and below it. Faded tissue between the two is kept when it
   * joins confident tissue, and dropped when it stands alone.
   */
  weakBias?: number;
}

/** Longest side of the working overview, in pixels. */
const MAX_OVERVIEW = 2048;

export interface TissueResult {
  /** Outer boundaries with their holes, in level-0 slide pixels. */
  polygons: Ring[][];
  /** Threshold actually used, so the UI can show and let the user adjust it. */
  saturationThreshold: number;
  /** The classifier that decided this, or null when the colour rule did. */
  modelId: string | null;
  /** Share of the slide covered by tissue. */
  coverage: number;
  /** Regions discarded as scanner artefacts, so the count is never silent. */
  rejected: number;
  levelUsed: number;
  ms: number;
}

/**
 * Otsu's method over a 256-bin histogram; returns the bin index.
 *
 * Where several thresholds tie, this returns the middle of the tied run rather
 * than the first. On a slide the tie is the empty valley between glass and
 * tissue — every cut through it separates the two classes equally well — and
 * taking the first puts the threshold hard against the glass, where sensor
 * noise then reads as tissue. The middle of the valley is the stable choice,
 * and it is what Otsu is usually taken to mean.
 */
function otsu(histogram: Uint32Array, total: number): number {
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * histogram[i];

  let sumB = 0;
  let wB = 0;
  let first = 0;
  let last = 0;
  let bestVariance = -1;
  for (let t = 0; t < 256; t++) {
    wB += histogram[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * histogram[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    // A relative tolerance, because the variance is a product of large numbers
    // and exact equality across a flat valley is a floating-point accident.
    if (between > bestVariance * (1 + 1e-9)) {
      bestVariance = between;
      first = t;
      last = t;
    } else if (between >= bestVariance * (1 - 1e-9)) {
      last = t;
    }
  }
  return Math.round((first + last) / 2);
}

/**
 * The overview grid a detection works on.
 *
 * Shared with label collection so that a cell the classifier is trained on is
 * exactly the cell it later predicts. Rebuilding the grid separately for
 * training would put the labels a fraction of a cell out from the features,
 * which is invisible and quietly corrupts every model fitted through it.
 */
export interface OverviewGrid {
  rgba: Uint8ClampedArray;
  w: number;
  h: number;
  step: number;
  level: LevelInfo;
  originX: number;
  originY: number;
  spanX: number;
  spanY: number;
}

export async function readOverviewGrid(source: SlideSource): Promise<OverviewGrid> {
  const { levels, bounds } = source.meta;
  const level = levels[levels.length - 1];
  const originX = bounds?.x ?? 0;
  const originY = bounds?.y ?? 0;
  const spanX = bounds?.width ?? levels[0].width;
  const spanY = bounds?.height ?? levels[0].height;

  // Full extent of the overview level, in its own pixels.
  const fullW = Math.max(1, Math.round(spanX / level.downsample));
  const fullH = Math.max(1, Math.round(spanY / level.downsample));

  /**
   * Read the WHOLE overview, downsampling if it is large — never a crop.
   *
   * Clamping the read size instead of the sampling rate silently reads only the
   * top-left corner and then maps it across the entire slide, so detections
   * land on blank glass while real tissue is never examined. The bug is
   * invisible on slides whose coarsest level is already small, which is exactly
   * why it survived: it only bites when `full > MAX_OVERVIEW`.
   */
  const step = Math.max(1, Math.ceil(Math.max(fullW, fullH) / MAX_OVERVIEW));
  const w = Math.max(1, Math.floor(fullW / step));
  const h = Math.max(1, Math.floor(fullH / step));
  const rgba = await readDownsampled(source, level.level, originX, originY, fullW, fullH, w, h, step);
  return { rgba, w, h, step, level, originX, originY, spanX, spanY };
}

/** Grid cell -> level-0 slide pixel, at the centre of the cell. */
export function cellToSlide(g: OverviewGrid, x: number, y: number): [number, number] {
  return [g.originX + ((x + 0.5) / g.w) * g.spanX, g.originY + ((y + 0.5) / g.h) * g.spanY];
}

/**
 * The colour-only reading of an overview: a per-cell score and the levels
 * derived from it.
 *
 * Exported because training needs the same numbers detection uses. Confident
 * glass is the one part of a slide nobody should have to label by hand — it is
 * unambiguous and it is most of the slide — so the trainer takes its negatives
 * from here, and gets them from exactly the cells the detector would have
 * called background.
 */
export interface OverviewScore {
  score: Uint8Array;
  /** Otsu's split, at the centre of the valley. */
  auto: number;
  /** Modal score below the split: the blank slide itself. */
  glass: number;
  span: number;
  /** Where tissue is taken to start. */
  relaxed: number;
  /** Where growth into faint tissue stops. */
  weakLevel: number;
}

export function scoreOverview(
  grid: OverviewGrid,
  opts: Pick<TissueOptions, "edgeBias" | "weakBias"> = {},
): OverviewScore {
  const { rgba, w, h } = grid;
  /**
   * Tissue score: how far a pixel is from "blank slide", taking the stronger of
   * two independent cues.
   *
   * Saturation alone finds well-stained tissue but misses faded H&E, which is
   * pale yet still darker than the glass. Darkness alone catches faded tissue
   * but also catches shadows and coverslip edges. Taking the maximum means
   * either cue can claim a pixel, so a section with both strong and faded areas
   * is captured whole rather than in patches.
   */
  const score = new Uint8Array(w * h);
  const histogram = new Uint32Array(256);
  for (let i = 0; i < w * h; i++) {
    const r = rgba[i * 4];
    const g = rgba[i * 4 + 1];
    const b = rgba[i * 4 + 2];
    const max = r > g ? (r > b ? r : b) : g > b ? g : b;
    const min = r < g ? (r < b ? r : b) : g < b ? g : b;
    const saturation = max === 0 ? 0 : ((max - min) / max) * 255;
    // Luminance, then its distance below white.
    const darkness = 255 - (0.299 * r + 0.587 * g + 0.114 * b);
    const v = Math.round(Math.max(saturation, darkness));
    score[i] = v;
    histogram[v]++;
  }

  const auto = otsu(histogram, w * h);

  /**
   * Blank slide is the tallest peak below the automatic threshold, so the
   * histogram says where "certainly not tissue" ends without being told.
   *
   * Both working thresholds are then placed on the span from that glass level
   * up to the automatic one. Anchoring at both ends is what makes them stable:
   * a plain fraction of the automatic threshold moves with wherever that
   * threshold happens to land, and can slip underneath the glass on a slide
   * with little tissue — at which point the scan's own sensor noise is tissue.
   */
  let glass = 0;
  let glassCount = -1;
  for (let v = 0; v <= auto; v++) {
    if (histogram[v] > glassCount) { glassCount = histogram[v]; glass = v; }
  }
  const span = Math.max(1, auto - glass);

  // Otsu splits at the point of maximum separation, which sits inside the
  // tissue's own falloff — so its boundary lands short of the true edge.
  // Relaxing it toward the glass recovers the pale rim.
  const relaxed = Math.max(6, Math.round(glass + span * (opts.edgeBias ?? 0.22)));
  const weakLevel = Math.max(glass + 2, Math.round(glass + span * (opts.weakBias ?? 0.1)));
  return { score, auto, glass, span, relaxed, weakLevel };
}

export async function detectTissue(
  source: SlideSource,
  opts: TissueOptions = {},
): Promise<TissueResult> {
  const started = performance.now();
  const { mppX } = source.meta;
  const grid = opts.grid ?? (await readOverviewGrid(source));
  const { rgba, w, h, step, level, originX, originY, spanX, spanY } = grid;

  const { score, relaxed, weakLevel } = scoreOverview(grid, opts);
  const threshold =
    opts.saturation !== undefined ? Math.round(opts.saturation * 255) : relaxed;

  /**
   * A trained classifier, when there is one, replaces the colour threshold.
   *
   * Its probability is used the same two ways the score was: a confident level
   * that decides where tissue starts, and a looser one the detection may grow
   * into. Keeping the same two-level structure means everything downstream —
   * de-speckling, binding, growth, tracing — is unchanged, and the only thing
   * the model alters is the judgement about each individual cell.
   */
  const modelUsed = opts.model ?? null;
  const raw = new Uint8Array(w * h);
  let probability: Float32Array | null = null;

  if (modelUsed) {
    probability = predictAll(modelUsed, computeFeatures({ rgba, w, h }), w * h);
    const cut = opts.modelThreshold ?? 0.5;
    for (let i = 0; i < w * h; i++) raw[i] = probability[i] >= cut ? 1 : 0;
  } else {
    for (let i = 0; i < w * h; i++) raw[i] = score[i] > threshold ? 1 : 0;
  }

  // Open first, then close. Opening deletes structures thinner than the
  // structuring element — scanner seams and coverslip edges are one or two
  // cells wide at overview scale, while the thinnest real section here is an
  // order of magnitude wider. Doing it BEFORE the closing matters: otherwise
  // the closing bridges a seam to the fragment beside it and they survive as
  // one region with a spur, which no per-region shape test can then separate.
  const seamRadius = opts.seamRadius ?? 1;
  const opened = opening(raw, w, h, seamRadius);

  /**
   * Grow the detection into faded tissue by hysteresis.
   *
   * One threshold cannot serve a slide whose staining varies across it: set it
   * where blank glass stays out and a faded fragment is missed entirely or
   * caught only in patches. So the confident threshold only decides where
   * tissue *starts*, and a second, lower one decides how far it extends —
   * accepted only where it joins something already confident. A faint fragment
   * therefore survives whole, while equally faint noise sitting alone on glass
   * still does not, because it connects to nothing.
   */
  const weakRaw = new Uint8Array(w * h);
  if (probability) {
    // Half the confident probability: the same "plausible but not certain"
    // band the score's weak level describes.
    const weakCut = (opts.modelThreshold ?? 0.5) * 0.5;
    for (let i = 0; i < w * h; i++) weakRaw[i] = probability[i] >= weakCut ? 1 : 0;
  } else {
    for (let i = 0; i < w * h; i++) weakRaw[i] = score[i] > weakLevel ? 1 : 0;
  }
  const weak = opening(weakRaw, w, h, seamRadius);

  /**
   * Fragments are identified at the confident threshold and grown from there,
   * each into its own territory. Growing first and labelling afterwards would
   * let the faint halo around two neighbouring fragments meet in the gap
   * between them, at which point they are one region and no later step can
   * tell them apart again.
   */
  /**
   * Bind each fragment together before it is used to identify anything.
   *
   * Thresholded real tissue is speckled — stroma and fat fall below the
   * confident threshold while the nuclei around them clear it — so the raw
   * mask of ONE section is dozens of disconnected islands. Seeding fragment
   * identity from that shatters a single section into dozens of objects.
   * Closing over a few cells rejoins what is obviously one piece; the gap
   * between two sections is an order of magnitude wider than the gaps inside
   * one, so it survives the same closing untouched.
   */
  const seeds = morphology(opened, w, h, opts.bindRadius ?? 3);
  const components = labelAndGrow(seeds, weak, w, h);

  let tissuePixels = 0;
  for (let i = 0; i < w * h; i++) if (components.labelAt(i) >= 0) tissuePixels++;

  // One overview pixel covers downsample² slide pixels.
  const pxPerCell = level.downsample * level.downsample;
  const um2PerCell = mppX ? pxPerCell * mppX * mppX : null;
  const minCells =
    opts.minAreaUm2 && um2PerCell ? Math.max(4, opts.minAreaUm2 / um2PerCell) : 16;

  // Fill lumina, not the slide. An unbounded fill swallows whatever blank space
  // happens to be enclosed by a ring of fragments, which both invents tissue
  // where there is none and welds separate fragments into a single object.
  const cellUm2 = mppX ? (level.downsample * step * mppX) ** 2 : null;
  const maxHoleCells = cellUm2
    ? Math.max(1, Math.round((opts.maxHoleUm2 ?? 200_000) / cellUm2))
    : Math.max(1, Math.round(w * h * 0.002));

  const toSlide = (ring: Ring): Ring =>
    ring.map(([x, y]) => [originX + (x / w) * spanX, originY + (y / h) * spanY]);

  /**
   * Drop long thin strips. Scanner seams run the height of a scan row at a
   * near-constant width; tissue does not. The absolute-width guard is what
   * keeps a needle core — thin, but far wider than a seam — out of this.
   */
  const maxAspect = opts.maxAspect ?? 10;
  const minWidthUm = opts.minWidthUm ?? 400;
  const umPerCell = mppX ? level.downsample * step * mppX : null;

  const isArtefact = (ring: Ring): boolean => {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const [x, y] of ring) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const wCells = Math.max(1, maxX - minX);
    const hCells = Math.max(1, maxY - minY);
    const aspect = Math.max(wCells, hCells) / Math.min(wCells, hCells);
    if (aspect < maxAspect) return false;
    if (!umPerCell) return true;
    return Math.min(wCells, hCells) * umPerCell < minWidthUm;
  };

  /**
   * Trace each connected component on its own, in its own window.
   *
   * Two fragments that are separate on the slide are separate components here,
   * so handling them one at a time is what makes them separate objects — and
   * it holds even when the closing that de-speckles one of them would have
   * reached across the gap to the other. It also confines hole assignment to
   * the fragment the hole is actually in: a lumen can no longer be matched to a
   * ring on the far side of the slide, which is what drew a bridge across the
   * image when the tessellator cut into it.
   */
  const assembled: Ring[][] = [];
  let rejected = 0;

  for (const comp of components) {
    if (comp.cells < minCells) continue;

    // A one-cell margin so the closing has room to work at the boundary and
    // the tracer sees background on every side of the fragment.
    const x0 = Math.max(0, comp.minX - 2);
    const y0 = Math.max(0, comp.minY - 2);
    const x1 = Math.min(w - 1, comp.maxX + 2);
    const y1 = Math.min(h - 1, comp.maxY + 2);
    const sw = x1 - x0 + 1;
    const sh = y1 - y0 + 1;

    const cell = new Uint8Array(sw * sh);
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        if (components.labelAt(x + y * w) === comp.id) cell[(y - y0) * sw + (x - x0)] = 1;
      }
    }

    const closed = morphology(cell, sw, sh, 1);
    const filled = fillHoles(closed, sw, sh, maxHoleCells);

    const sub = new Float32Array(sw * sh);
    for (let i = 0; i < sw * sh; i++) sub[i] = filled[i] ? 1 : -1;

    const rings = traceMask(sub, sw, sh, { threshold: 0, simplify: 0.8, minArea: minCells })
      .map((ring) => ring.map(([x, y]) => [x + x0, y + y0] as [number, number]));
    if (!rings.length) continue;

    // Classify outers and holes by containment, not by winding direction: the
    // sign convention is an implementation detail of the tracer, and getting it
    // backwards silently turns every region into an orphaned hole.
    const containers = rings.map((r, i) =>
      rings.reduce(
        (n, other, j) => (i !== j && pointInRing(other, r[0][0], r[0][1]) ? n + 1 : n),
        0,
      ),
    );
    const outerIdx = rings.map((_, i) => i).filter((i) => containers[i] % 2 === 0);
    const holeIdx = rings.map((_, i) => i).filter((i) => containers[i] % 2 === 1);

    const kept = outerIdx.filter((i) => !isArtefact(rings[i]));
    rejected += outerIdx.length - kept.length;
    if (!kept.length) continue;

    const slots: Ring[][] = kept.map((i) => [toSlide(rings[i])]);
    for (const hi of holeIdx) {
      const [hx, hy] = rings[hi][0];
      // Innermost containing outer wins, so nested structures nest correctly.
      let bestSlot = -1;
      let bestArea = Infinity;
      kept.forEach((oi, slot) => {
        const a = Math.abs(ringArea(rings[oi]));
        if (a < bestArea && pointInRing(rings[oi], hx, hy)) {
          bestArea = a;
          bestSlot = slot;
        }
      });
      if (bestSlot >= 0) slots[bestSlot].push(toSlide(rings[hi]));
    }
    assembled.push(...slots);
  }

  /**
   * Normalise every polygon through the boolean library before emitting it.
   *
   * Douglas-Peucker can make a convoluted tissue boundary self-intersect, and a
   * hole is matched to its parent by testing a single vertex, which is wrong
   * when that vertex sits exactly on a boundary. Either produces a polygon
   * whose triangulation bridges across the region as long thin slivers. A union
   * with nothing returns the same region as a valid, non-self-intersecting
   * polygon with holes correctly nested and wound, which is precisely the
   * guarantee the renderer needs.
   */
  const polygons: Ring[][] = [];
  for (const rings2 of assembled) {
    const cleaned = cleanGeometry({ type: "Polygon", coordinates: rings2 });
    if (!cleaned) continue;
    if (cleaned.type === "Polygon") polygons.push(cleaned.coordinates);
    else for (const part of cleaned.coordinates) polygons.push(part);
  }

  return {
    polygons,
    saturationThreshold: threshold / 255,
    modelId: modelUsed?.id ?? null,
    coverage: tissuePixels / (w * h),
    rejected,
    levelUsed: level.level,
    ms: performance.now() - started,
  };
}

interface Component {
  id: number;
  cells: number;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

interface Labelling extends Array<Component> {
  /** Component id at a cell index, or -1 for background. */
  labelAt(i: number): number;
}

/**
 * Label the connected fragments of `strong`, then grow each one into `weak`.
 *
 * Two ideas in one pass, because they are the same pass.
 *
 * The growth is hysteresis thresholding: the strict threshold decides where
 * tissue certainly is, a looser one decides how far it plausibly extends, and
 * loose cells are accepted only where they join something already certain. On a
 * slide whose staining fades across it, this is the difference between a pale
 * fragment coming out whole and coming out as islands — while equally pale
 * noise alone on the glass is still rejected, because it anchors to nothing.
 *
 * The labelling is what keeps fragments apart. Every cell is claimed by the
 * fragment that reaches it first, so two fragments growing toward each other
 * meet at a boundary instead of merging: the number of objects is fixed by the
 * confident mask, and nothing downstream can fuse them. That is also why the
 * frontier is a queue rather than a stack — breadth-first, all fragments
 * advancing at the same rate, so the boundary lands between them rather than
 * wherever the iteration order happened to start.
 */
function labelAndGrow(
  strong: Uint8Array,
  weak: Uint8Array,
  w: number,
  h: number,
): Labelling {
  const labels = new Int32Array(w * h).fill(-1);
  const comps: Component[] = [];

  const note = (comp: Component, i: number) => {
    const x = i % w;
    const y = (i - x) / w;
    comp.cells++;
    if (x < comp.minX) comp.minX = x;
    if (x > comp.maxX) comp.maxX = x;
    if (y < comp.minY) comp.minY = y;
    if (y > comp.maxY) comp.maxY = y;
  };

  // Seeds: the connected components of the confident mask.
  const frontier: number[] = [];
  for (let start = 0; start < w * h; start++) {
    if (!strong[start] || labels[start] >= 0) continue;
    const comp: Component = { id: comps.length, cells: 0, minX: w, minY: h, maxX: 0, maxY: 0 };
    const stack = [start];
    labels[start] = comp.id;
    while (stack.length) {
      const i = stack.pop()!;
      note(comp, i);
      frontier.push(i);
      const x = i % w;
      const push = (j: number) => {
        if (strong[j] && labels[j] < 0) { labels[j] = comp.id; stack.push(j); }
      };
      if (x > 0) push(i - 1);
      if (x < w - 1) push(i + 1);
      if (i >= w) push(i - w);
      if (i < w * (h - 1)) push(i + w);
    }
    comps.push(comp);
  }

  // Grow every fragment outward one ring at a time, into weak cells only.
  for (let head = 0; head < frontier.length; head++) {
    const i = frontier[head];
    const id = labels[i];
    const comp = comps[id];
    const x = i % w;
    const grow = (j: number) => {
      if (!weak[j] || labels[j] >= 0) return;
      labels[j] = id;
      note(comp, j);
      frontier.push(j);
    };
    if (x > 0) grow(i - 1);
    if (x < w - 1) grow(i + 1);
    if (i >= w) grow(i - w);
    if (i < w * (h - 1)) grow(i + w);
  }

  const out = comps as Labelling;
  out.labelAt = (i: number) => labels[i];
  return out;
}

function pointInRing(ring: Ring, x: number, y: number): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export { largestRing };

/**
 * Binary closing: dilate then erode by `radius`, using a square structuring
 * element. Bridges the speckled gaps that thresholding leaves in pale tissue
 * without growing the outline overall.
 */
function morphology(mask: Uint8Array, w: number, h: number, radius: number): Uint8Array {
  return erode(dilate(mask, w, h, radius), w, h, radius);
}

/** Erode then dilate: deletes anything thinner than the element, keeps the rest. */
function opening(mask: Uint8Array, w: number, h: number, radius: number): Uint8Array {
  if (radius <= 0) return mask;
  return dilate(erode(mask, w, h, radius), w, h, radius);
}

function dilate(mask: Uint8Array, w: number, h: number, r: number): Uint8Array {
  return sweep(mask, w, h, r, true);
}

function erode(mask: Uint8Array, w: number, h: number, r: number): Uint8Array {
  return sweep(mask, w, h, r, false);
}

/** Separable min/max filter: two 1-D passes instead of a full window. */
function sweep(mask: Uint8Array, w: number, h: number, r: number, max: boolean): Uint8Array {
  const tmp = new Uint8Array(w * h);
  const out = new Uint8Array(w * h);
  const pick = max ? 1 : 0;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let hit = max ? 0 : 1;
      for (let d = -r; d <= r; d++) {
        const xx = x + d;
        if (xx < 0 || xx >= w) continue;
        if (mask[y * w + xx] === pick) { hit = pick; break; }
      }
      tmp[y * w + x] = hit;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let hit = max ? 0 : 1;
      for (let d = -r; d <= r; d++) {
        const yy = y + d;
        if (yy < 0 || yy >= h) continue;
        if (tmp[yy * w + x] === pick) { hit = pick; break; }
      }
      out[y * w + x] = hit;
    }
  }
  return out;
}

/**
 * Fill background regions that do not touch the border.
 *
 * A lumen or a pale patch inside a section is still that section; leaving them
 * as holes fragments the outline and, worse, excludes them from any patch grid
 * laid over the tissue.
 */
function fillHoles(mask: Uint8Array, w: number, h: number, maxCells: number): Uint8Array {
  const seen = new Uint8Array(w * h);
  const out = Uint8Array.from(mask);

  // Everything reachable from the border is outside, never a hole.
  const stack: number[] = [];
  const visit = (i: number) => {
    if (!mask[i] && !seen[i]) { seen[i] = 1; stack.push(i); }
  };
  for (let x = 0; x < w; x++) { visit(x); visit((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { visit(y * w); visit(y * w + w - 1); }
  while (stack.length) {
    const i = stack.pop()!;
    const x = i % w;
    if (x > 0) visit(i - 1);
    if (x < w - 1) visit(i + 1);
    if (i >= w) visit(i - w);
    if (i < w * (h - 1)) visit(i + w);
  }

  // Each remaining background component is an enclosed gap. Fill it only if it
  // is small enough to be part of the tissue rather than a space between.
  for (let start = 0; start < w * h; start++) {
    if (mask[start] || seen[start]) continue;
    const component: number[] = [];
    const queue = [start];
    seen[start] = 1;
    while (queue.length) {
      const i = queue.pop()!;
      component.push(i);
      const x = i % w;
      const push2 = (j: number) => {
        if (!mask[j] && !seen[j]) { seen[j] = 1; queue.push(j); }
      };
      if (x > 0) push2(i - 1);
      if (x < w - 1) push2(i + 1);
      if (i >= w) push2(i - w);
      if (i < w * (h - 1)) push2(i + w);
    }
    if (component.length <= maxCells) for (const i of component) out[i] = 1;
  }
  return out;
}

/**
 * Read a whole pyramid level, box-averaging it down to `outW` x `outH`.
 *
 * Done in horizontal strips so peak memory is bounded by the strip, not by the
 * level: a shallow-pyramid slide can have a coarsest level of several thousand
 * pixels a side, and reading that whole thing as RGBA at once is tens of
 * megabytes per call.
 */
async function readDownsampled(
  source: SlideSource,
  level: number,
  originX: number,
  originY: number,
  fullW: number,
  fullH: number,
  outW: number,
  outH: number,
  step: number,
): Promise<Uint8ClampedArray> {
  const out = new Uint8ClampedArray(outW * outH * 4);
  if (step === 1 && outW === fullW && outH === fullH) {
    return source.readRegion(originX, originY, level, fullW, fullH);
  }

  const downsample = source.meta.levels[level]?.downsample ?? 1;
  const STRIP_ROWS = 256; // output rows per pass

  for (let oy0 = 0; oy0 < outH; oy0 += STRIP_ROWS) {
    const rows = Math.min(STRIP_ROWS, outH - oy0);
    const srcY = oy0 * step;
    const srcRows = Math.min(fullH - srcY, rows * step);
    if (srcRows <= 0) break;

    // readRegion takes x/y in level-0 pixels but width/height in level pixels.
    const strip = await source.readRegion(
      originX,
      originY + Math.round(srcY * downsample),
      level,
      fullW,
      srcRows,
    );

    for (let oy = 0; oy < rows; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        let r = 0, g = 0, b = 0, n = 0;
        const sy0 = oy * step;
        const sx0 = ox * step;
        for (let dy = 0; dy < step; dy++) {
          const sy = sy0 + dy;
          if (sy >= srcRows) break;
          for (let dx = 0; dx < step; dx++) {
            const sx = sx0 + dx;
            if (sx >= fullW) break;
            const i = (sy * fullW + sx) * 4;
            r += strip[i]; g += strip[i + 1]; b += strip[i + 2]; n++;
          }
        }
        const o = ((oy0 + oy) * outW + ox) * 4;
        if (n === 0) { out[o] = 255; out[o + 1] = 255; out[o + 2] = 255; out[o + 3] = 255; continue; }
        out[o] = r / n; out[o + 1] = g / n; out[o + 2] = b / n; out[o + 3] = 255;
      }
    }
  }
  return out;
}
