/**
 * Slide I/O abstraction.
 *
 * Everything above this interface is backend-agnostic, so a format that
 * openslide-wasm cannot handle in-browser can fall back to sidecar-served
 * tiles without touching the viewer, annotation or ML layers.
 */

export interface LevelInfo {
  level: number;
  width: number;
  height: number;
  /** Linear downsample from level 0. Rarely an exact power of two. */
  downsample: number;
}

export interface SlideMeta {
  name: string;
  /** Total bytes across all constituent files (MRXS is a directory). */
  bytes: number;
  vendor: string | null;
  /** Microns per pixel at level 0. Null when the vendor omits it. */
  mppX: number | null;
  mppY: number | null;
  objectivePower: number | null;
  /** Non-empty scan region within the level-0 frame (MIRAX uses this heavily). */
  bounds: { x: number; y: number; width: number; height: number } | null;
  backgroundColor: string | null;
  levels: LevelInfo[];
  properties: Record<string, string>;
}

export interface SlideSource {
  readonly meta: SlideMeta;
  /**
   * Read a region and return premultiplication-corrected RGBA.
   *
   * NOTE (OpenSlide convention): `x`/`y` are in the **level-0** reference
   * frame, while `width`/`height` are in the frame of `level`. Getting this
   * wrong is the classic OpenSlide bug.
   */
  readRegion(
    x: number,
    y: number,
    level: number,
    width: number,
    height: number,
    signal?: AbortSignal,
  ): Promise<Uint8ClampedArray>;
  bestLevelForDownsample(downsample: number): Promise<number>;
  close(): Promise<void>;
}

/** A file plus its virtual path, as multi-file formats (MRXS, VMS, DICOM) require. */
export interface ResolvedFile {
  path: string;
  file: File;
}

/** One openable slide discovered from a drop: entry file + any siblings it needs. */
export interface ResolvedSlide {
  name: string;
  /** The primary file (.svs/.ndpi/.mrxs). Must be first when opening. */
  entryPath: string;
  files: ResolvedFile[];
  bytes: number;
  /** Set when we detect a multi-file format whose companion data is missing. */
  warning: string | null;
  /**
   * A GeoJSON file sitting beside the slide and named after it.
   *
   * This is how a batch run hands its work back: it writes `<slide>.geojson`
   * next to each slide, and dropping that folder in later brings the
   * annotations with the slide instead of leaving them to be imported by hand.
   */
  annotations: File | null;
}
