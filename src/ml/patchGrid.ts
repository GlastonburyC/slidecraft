import { containsPoint, isAreaGeometry, type Annotation, type Geometry } from "../annotate/types";

/**
 * Patch grids over a region.
 *
 * Patches are specified in **pixels at a chosen pyramid level** — 128x128,
 * 256x256 — not in microns. That is how every histology encoder is actually
 * defined (a ViT sees a fixed pixel tensor), and it means the grid is
 * reproducible across slides regardless of their scanner resolution. The µm
 * size each patch covers is reported so the physical scale stays visible.
 */

export interface PatchGridOptions {
  /** Square patch side, in pixels at the chosen level. */
  patchPx: number;
  /** Step between patch origins, in pixels. Defaults to `patchPx` (no overlap). */
  stridePx?: number;
  /** Pyramid level to sample at; 0 is full resolution. */
  level?: number;
  /** Keep only patches whose centre falls inside one of these geometries. */
  within?: Annotation[];
  /**
   * A second, independent test the centre must ALSO pass.
   *
   * Two filters rather than one longer list, because `within` is an OR: adding
   * the tissue to the ROI's own list would mean "inside the ROI *or* on tissue",
   * which is every patch in the ROI — the filter silently does nothing, and the
   * grid covers the glass it was supposed to avoid.
   */
  restrictTo?: Annotation[];
  /** Drop patches that fall outside the slide's scanned bounds. */
  bounds?: { x: number; y: number; width: number; height: number } | null;
}

export interface Patch {
  /** Index in the grid, row-major. */
  index: number;
  /** Column and row, so a result can be laid back onto the grid. */
  col: number;
  row: number;
  /** Top-left in level-0 slide pixels — what readRegion expects. */
  x: number;
  y: number;
  /** Side in level-0 slide pixels (patchPx scaled by the level downsample). */
  size: number;
}

export interface PatchGrid {
  patches: Patch[];
  cols: number;
  rows: number;
  /** Side of one patch in µm, when the slide reports a scale. */
  patchUm: number | null;
  patchPx: number;
  level: number;
  downsample: number;
}

/**
 * Lay a grid over a region.
 *
 * `region` and the returned coordinates are in level-0 slide pixels, because
 * that is the frame annotations use; `patchPx` is in level pixels, because that
 * is the frame models use. The downsample converts between them once, here,
 * rather than at every call site.
 */
export function buildPatchGrid(
  region: { x: number; y: number; width: number; height: number },
  downsample: number,
  mppX: number | null,
  opts: PatchGridOptions,
): PatchGrid {
  const patchPx = Math.max(8, Math.round(opts.patchPx));
  const stridePx = Math.max(1, Math.round(opts.stridePx ?? patchPx));
  const level = opts.level ?? 0;

  // One patch spans this many level-0 pixels.
  const size = patchPx * downsample;
  const step = stridePx * downsample;

  const cols = Math.max(0, Math.floor((region.width - size) / step) + 1);
  const rows = Math.max(0, Math.floor((region.height - size) / step) + 1);

  const within = (opts.within ?? []).filter((a) => isAreaGeometry(a.geometry));
  const restrictTo = (opts.restrictTo ?? []).filter((a) => isAreaGeometry(a.geometry));
  const patches: Patch[] = [];

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const x = region.x + col * step;
      const y = region.y + row * step;

      if (opts.bounds) {
        const b = opts.bounds;
        if (x < b.x || y < b.y || x + size > b.x + b.width || y + size > b.y + b.height) continue;
      }

      if (within.length > 0 || restrictTo.length > 0) {
        // Centre-in-region: a patch belongs to the tissue it is mostly on, and
        // testing one point keeps this linear in patches rather than in vertices.
        const cx = x + size / 2;
        const cy = y + size / 2;
        if (within.length > 0 && !within.some((a) => containsPoint(a.geometry as Geometry, cx, cy))) {
          continue;
        }
        if (
          restrictTo.length > 0 &&
          !restrictTo.some((a) => containsPoint(a.geometry as Geometry, cx, cy))
        ) {
          continue;
        }
      }

      patches.push({ index: patches.length, col, row, x, y, size });
    }
  }

  return {
    patches,
    cols,
    rows,
    patchUm: mppX ? patchPx * downsample * mppX : null,
    patchPx,
    level,
    downsample,
  };
}

/** Stable key for caching one patch's embedding. */
export function patchKey(
  slideKey: string,
  modelId: string,
  level: number,
  patchPx: number,
  p: Patch,
): string {
  return `${slideKey}|${modelId}|L${level}|${patchPx}|${p.x},${p.y}`;
}
