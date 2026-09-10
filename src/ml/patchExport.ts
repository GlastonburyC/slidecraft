import type { PatchGrid } from "./patchGrid";
import type { SlideMeta } from "../slide/types";

/**
 * Hand a patch grid to whatever will actually consume it.
 *
 * The grid on screen is for judging — is this scale right, is it sitting on
 * tissue. Acting on it happens elsewhere: a training script, a feature
 * extractor, another lab's pipeline. So the grid leaves as coordinates rather
 * than as pixels, and the coordinates are the ones OpenSlide takes, so
 * `read_region(x, y, level, size, size)` on the same slide reproduces exactly
 * the patch that was drawn.
 */

/**
 * Marks an annotation as one cell of a patch grid.
 *
 * Recorded on `modelId` rather than inferred from a class name, so renaming the
 * class does not change what the app believes the object is — rebuilding the
 * grid still replaces the right objects, and the resize handles still appear.
 */
export const PATCH_MODEL_ID = "patch-grid";

export interface PatchManifest {
  slide: string;
  /** Microns per pixel at level 0, when the slide reports it. */
  mpp: number | null;
  level: number;
  downsample: number;
  /** Patch side in level pixels — what a model is defined in. */
  patchPx: number;
  /** Patch side in level-0 pixels — what read_region takes. */
  patchPx0: number;
  /** Patch side in microns, which is what decides the scale being looked at. */
  patchUm: number | null;
  cols: number;
  rows: number;
  count: number;
  generated: string;
  /** Top-left of each patch in level-0 pixels, row-major within the grid. */
  patches: { index: number; col: number; row: number; x: number; y: number }[];
}

export function toPatchManifest(grid: PatchGrid, meta: SlideMeta): PatchManifest {
  return {
    slide: meta.name,
    mpp: meta.mppX,
    level: grid.level,
    downsample: grid.downsample,
    patchPx: grid.patchPx,
    patchPx0: grid.patchPx * grid.downsample,
    patchUm: grid.patchUm,
    cols: grid.cols,
    rows: grid.rows,
    count: grid.patches.length,
    generated: new Date().toISOString(),
    patches: grid.patches.map((p) => ({
      index: p.index,
      col: p.col,
      row: p.row,
      x: p.x,
      y: p.y,
    })),
  };
}

/**
 * The same grid as GeoJSON, so it can be opened next to the slide.
 *
 * A manifest is what a script wants; a square you can see is what a person
 * wants when the question is "did it patch the right thing". Both describe the
 * identical geometry, which is the point — the check and the input cannot
 * drift apart.
 */
export function toPatchGeoJSON(grid: PatchGrid, meta: SlideMeta) {
  const side = grid.patchPx * grid.downsample;
  return {
    type: "FeatureCollection" as const,
    slidecraft: { slide: meta.name, patchPx: grid.patchPx, level: grid.level },
    features: grid.patches.map((p) => ({
      type: "Feature" as const,
      geometry: {
        type: "Polygon" as const,
        coordinates: [[
          [p.x, p.y],
          [p.x + side, p.y],
          [p.x + side, p.y + side],
          [p.x, p.y + side],
          [p.x, p.y],
        ]],
      },
      properties: {
        objectType: "annotation",
        classification: { name: "Patch", colorRGB: -8355712 },
        slidecraft: { index: p.index, col: p.col, row: p.row, patchPx: grid.patchPx, level: grid.level },
      },
    })),
  };
}

/** Python that reads exactly these patches, so the manifest is self-describing. */
export function patchReaderSnippet(m: PatchManifest): string {
  return `# ${m.count} patches from ${m.slide}
# ${m.patchPx}x${m.patchPx} px at level ${m.level}${m.patchUm ? ` (${m.patchUm.toFixed(1)} um)` : ""}
import json, openslide
manifest = json.load(open("${stem(m.slide)}.patches.json"))
slide = openslide.OpenSlide("${m.slide}")
size = (manifest["patchPx"], manifest["patchPx"])
for p in manifest["patches"]:
    tile = slide.read_region((p["x"], p["y"]), manifest["level"], size).convert("RGB")
    # tile.save(f'patch_{p["index"]:05d}.png')
`;
}

const stem = (name: string) => {
  const i = name.lastIndexOf(".");
  return i === -1 ? name : name.slice(0, i);
};

export { stem as patchStem };
