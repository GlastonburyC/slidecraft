import OpenSeadragon from "openseadragon";
import type { SlideSource } from "../slide/types";

export const TILE_SIZE = 512;

/**
 * Bridges OpenSlide's pyramid to OpenSeadragon's.
 *
 * OpenSeadragon assumes a clean power-of-two pyramid. OpenSlide exposes
 * whatever the vendor stored — NDPI is often 1, 4, 16, 64; MIRAX 1, 2, 4, 8.
 * So we present a synthetic power-of-two pyramid and, per tile, read from the
 * nearest OpenSlide level at or above the required resolution and downscale.
 * This is the same strategy as openslide-python's DeepZoomGenerator.
 */

interface TileStats {
  tiles: number;
  totalMs: number;
  maxMs: number;
  bytes: number;
  errors: number;
}

export interface OpenSlideTileSource extends OpenSeadragon.TileSource {
  stats: TileStats;
  resetStats(): void;
}

/**
 * OpenSlide's own rule: pick the highest-resolution level whose downsample
 * does not exceed what we need, so we only ever downscale, never upscale.
 */
function bestLevelFor(downsamples: number[], wanted: number): number {
  let best = 0;
  for (let i = 0; i < downsamples.length; i++) {
    if (downsamples[i] <= wanted + 1e-6) best = i;
    else break;
  }
  return best;
}

/**
 * `ImageData` refuses a view backed by a SharedArrayBuffer, and openslide-wasm
 * runs its heap in shared memory. Copy into unshared storage only when needed,
 * so the common case stays zero-copy.
 */
function toUnshared(a: Uint8ClampedArray): Uint8ClampedArray<ArrayBuffer> {
  return (
    typeof SharedArrayBuffer !== "undefined" && a.buffer instanceof SharedArrayBuffer
      ? new Uint8ClampedArray(a)
      : a
  ) as Uint8ClampedArray<ArrayBuffer>;
}

/** What one OSD tile resolves to: a region to read, and the size to draw it at. */
export interface TilePlan {
  /** Top-left of the read, in level-0 slide pixels (what readRegion expects). */
  x0: number;
  y0: number;
  /** OpenSlide level to read from. */
  osLevel: number;
  /** Read size, in that level's own pixels. */
  rw: number;
  rh: number;
  /** Size the tile occupies on the OSD grid. */
  tw: number;
  th: number;
}

/**
 * Map an OSD tile to an OpenSlide read.
 *
 * Pulled out of the download path so it can be checked without a viewer: the
 * arithmetic crosses three coordinate frames — the synthetic power-of-two grid,
 * level-0 slide pixels (offset by the MIRAX scan region), and the chosen
 * OpenSlide level's own pixels — and a mistake in any of them reads the wrong
 * part of the slide rather than failing.
 */
export function planTile(
  meta: { levels: { width: number; height: number; downsample: number }[] },
  frame: { width: number; height: number; offsetX: number; offsetY: number; maxLevel: number },
  level: number,
  col: number,
  row: number,
): TilePlan | null {
  const { levels } = meta;
  const downsamples = levels.map((l) => l.downsample);

  // Downsample from level 0 that this OSD level represents.
  const scale = 2 ** (frame.maxLevel - level);
  const levelW = Math.ceil(frame.width / scale);
  const levelH = Math.ceil(frame.height / scale);

  const tx = col * TILE_SIZE;
  const ty = row * TILE_SIZE;
  const tw = Math.min(TILE_SIZE, levelW - tx);
  const th = Math.min(TILE_SIZE, levelH - ty);
  if (tw <= 0 || th <= 0) return null;

  // Tile origin in the level-0 frame (what readRegion expects for x/y).
  const x0 = Math.round(frame.offsetX + tx * scale);
  const y0 = Math.round(frame.offsetY + ty * scale);

  const osLevel = bestLevelFor(downsamples, scale);
  const osDs = downsamples[osLevel];

  // Read size expressed in the chosen OpenSlide level's own frame, clamped to
  // what that level actually contains. Without the clamp, coarse tiles ask for
  // regions far past the level's edge, which OpenSlide services by decoding and
  // zero-filling a huge area for a tiny output.
  const maxW = Math.max(1, levels[osLevel].width - Math.floor(x0 / osDs));
  const maxH = Math.max(1, levels[osLevel].height - Math.floor(y0 / osDs));
  const rw = Math.min(maxW, Math.max(1, Math.round((tw * scale) / osDs)));
  const rh = Math.min(maxH, Math.max(1, Math.round((th * scale) / osDs)));

  return { x0, y0, osLevel, rw, rh, tw, th };
}

export function createOpenSlideTileSource(source: SlideSource): OpenSlideTileSource {
  const { levels, bounds } = source.meta;
  const downsamples = levels.map((l) => l.downsample);

  // MIRAX stores a large mostly-empty level-0 canvas with the real scan region
  // marked by bounds-*. Presenting the bounds as the image avoids dropping the
  // user into an ocean of blank background.
  const offsetX = bounds?.x ?? 0;
  const offsetY = bounds?.y ?? 0;
  const width = bounds?.width ?? levels[0].width;
  const height = bounds?.height ?? levels[0].height;

  const maxLevel = Math.max(0, Math.ceil(Math.log2(Math.max(width, height))));

  /**
   * How coarse we are willing to go is set by the pyramid, not by the image.
   *
   * Zooming out past the vendor's coarsest level means reading from that level
   * and downscaling, and the cost of that read grows without bound. A slide
   * whose coarsest level is only 32x (some NDPI) would need a 3.6-megapixel
   * read to fill ONE 512px tile at the opening view, which stalls the viewer
   * long enough to look like a black canvas.
   *
   * So stop at the level where each tile reads at most `SOURCE_TILE_BUDGET`
   * tiles' worth of source pixels per axis. The whole slide is then covered by
   * a handful of bounded reads that run in parallel across the worker pool,
   * instead of one enormous serial one.
   */
  const SOURCE_TILE_BUDGET = 2;
  const coarsestDownsample = downsamples[downsamples.length - 1] || 1;
  const minLevel = Math.max(
    0,
    maxLevel - Math.floor(Math.log2(Math.max(1, coarsestDownsample * SOURCE_TILE_BUDGET))),
  );

  const stats: TileStats = { tiles: 0, totalMs: 0, maxMs: 0, bytes: 0, errors: 0 };

  const ts = new OpenSeadragon.TileSource({
    width,
    height,
    tileSize: TILE_SIZE,
    tileOverlap: 0,
    minLevel,
    maxLevel,
  }) as OpenSlideTileSource;

  ts.stats = stats;
  ts.resetStats = () => {
    stats.tiles = 0;
    stats.totalMs = 0;
    stats.maxMs = 0;
    stats.bytes = 0;
    stats.errors = 0;
  };

  // OSD identifies tiles by URL; we encode coordinates and parse them back.
  ts.getTileUrl = (level: number, x: number, y: number) => `osl:${level}/${x}/${y}`;
  ts.hasTransparency = () => false;

  ts.downloadTileStart = function (context: OpenSeadragon.ImageJob) {
    const parsed = /^osl:(\d+)\/(\d+)\/(\d+)$/.exec(String(context.src));
    if (!parsed) {
      context.finish(null, null as never, "Unparseable tile key");
      return;
    }
    const level = Number(parsed[1]);
    const col = Number(parsed[2]);
    const row = Number(parsed[3]);

    const controller = new AbortController();
    context.userData = { controller };

    void (async () => {
      const started = performance.now();
      try {
        const plan = planTile(
          source.meta,
          { width, height, offsetX, offsetY, maxLevel },
          level,
          col,
          row,
        );
        if (!plan) {
          context.finish(null, null as never, "Tile out of bounds");
          return;
        }
        const { x0, y0, osLevel, rw, rh, tw, th } = plan;

        const rgba = await source.readRegion(x0, y0, osLevel, rw, rh, controller.signal);
        if (controller.signal.aborted) return;

        const imageData = new ImageData(toUnshared(rgba), rw, rh);
        // Resize during decode — cheaper and better filtered than drawImage.
        const bitmap =
          rw === tw && rh === th
            ? await createImageBitmap(imageData)
            : await createImageBitmap(imageData, {
                resizeWidth: tw,
                resizeHeight: th,
                resizeQuality: "high",
              });
        if (controller.signal.aborted) {
          bitmap.close();
          return;
        }

        const elapsed = performance.now() - started;
        stats.tiles += 1;
        stats.totalMs += elapsed;
        stats.maxMs = Math.max(stats.maxMs, elapsed);
        stats.bytes += rgba.length;

        context.finish(bitmap, null as never, "imageBitmap");
      } catch (err) {
        if (controller.signal.aborted) return;
        stats.errors += 1;
        context.finish(null, null as never, String(err));
      }
    })();
  };

  ts.downloadTileAbort = function (context: OpenSeadragon.ImageJob) {
    (context.userData as { controller?: AbortController } | undefined)?.controller?.abort();
  };

  return ts;
}
