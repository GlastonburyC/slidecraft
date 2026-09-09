import { makeAnnotation } from "../annotate/store";
import type { AnnotationClass } from "../annotate/types";
import { toFeatureCollection } from "../io/geojson";
import { resolveSlides } from "../slide/dropResolver";
import { closeSlide, openSlide } from "../slide/openslideSource";
import type { ResolvedFile, ResolvedSlide } from "../slide/types";
import { detectTissue } from "./tissue";
import type { TissueModel } from "./tissueModel";

/**
 * Tissue-segment a whole folder of slides, unattended.
 *
 * Once a model is good enough to trust, the work stops being interactive and
 * becomes a queue, so this takes a directory and writes `<slide>.geojson` beside
 * each slide. Writing the result next to its input, named after it, is what
 * makes the pairing automatic later: dropping the folder back into Slidecraft
 * brings each slide in with its annotations already attached, and QuPath reads
 * the same files.
 *
 * Slides are opened and closed one at a time. A folder can be hundreds of
 * gigabytes and the decoder holds a slide's pyramid in a wasm heap with a hard
 * ceiling, so keeping two open to overlap the work is how a batch dies at slide
 * ninety with everything still to do.
 */

export interface BatchProgress {
  done: number;
  total: number;
  /** Slide currently being worked on. */
  current: string;
  /** Finished slides, newest last. */
  results: BatchResult[];
}

export interface BatchResult {
  slide: string;
  regions: number;
  ms: number;
  /** Written file, or null when it failed. */
  written: string | null;
  error: string | null;
}

export interface BatchOptions {
  model: TissueModel | null;
  minAreaUm2: number;
  className: string;
  classes: AnnotationClass[];
  /** Leave existing sidecars alone rather than redoing them. */
  skipExisting: boolean;
  signal?: AbortSignal;
  onProgress?: (p: BatchProgress) => void;
}

/** True when this browser can pick a directory and write back into it. */
export function canRunBatch(): boolean {
  return typeof (globalThis as { showDirectoryPicker?: unknown }).showDirectoryPicker === "function";
}

interface DirectoryHandle {
  values(): AsyncIterableIterator<FileSystemHandleLike>;
  getFileHandle(name: string, opts?: { create?: boolean }): Promise<FileHandleLike>;
  name: string;
}
interface FileSystemHandleLike {
  kind: "file" | "directory";
  name: string;
  getFile?: () => Promise<File>;
  values?: () => AsyncIterableIterator<FileSystemHandleLike>;
}
interface FileHandleLike {
  createWritable(): Promise<{ write(data: string): Promise<void>; close(): Promise<void> }>;
  getFile(): Promise<File>;
}

/**
 * Ask for a folder, with permission to write the results back into it.
 *
 * Read-only access would mean handing back a hundred downloads instead, which
 * is unusable at this scale and loses the naming that makes the pairing work.
 */
export async function pickBatchDirectory(): Promise<DirectoryHandle | null> {
  const picker = (globalThis as {
    showDirectoryPicker?: (o: { mode: string }) => Promise<DirectoryHandle>;
  }).showDirectoryPicker;
  if (!picker) return null;
  try {
    return await picker({ mode: "readwrite" });
  } catch {
    return null; // the user cancelled
  }
}

/** Flatten a directory into the {path, file} records the resolver understands. */
async function readTree(
  dir: DirectoryHandle | FileSystemHandleLike,
  prefix = "",
  depth = 0,
): Promise<ResolvedFile[]> {
  if (depth > 3) return [];
  const out: ResolvedFile[] = [];
  const values = (dir as DirectoryHandle).values?.bind(dir);
  if (!values) return out;
  for await (const entry of values()) {
    if (entry.kind === "file" && entry.getFile) {
      out.push({ path: prefix + entry.name, file: await entry.getFile() });
    } else if (entry.kind === "directory") {
      out.push(...(await readTree(entry, `${prefix}${entry.name}/`, depth + 1)));
    }
  }
  return out;
}

const stemOf = (name: string) => {
  const i = name.lastIndexOf(".");
  return i === -1 ? name : name.slice(0, i);
};

/** Slides a directory holds, in the order they will be processed. */
export async function listBatchSlides(dir: DirectoryHandle): Promise<ResolvedSlide[]> {
  return resolveSlides(await readTree(dir));
}

export async function runBatch(
  dir: DirectoryHandle,
  slides: ResolvedSlide[],
  opts: BatchOptions,
): Promise<BatchResult[]> {
  const results: BatchResult[] = [];
  const classes = opts.classes;
  const cls = classes.find((c) => c.name === opts.className) ?? null;

  for (const slide of slides) {
    if (opts.signal?.aborted) break;
    opts.onProgress?.({ done: results.length, total: slides.length, current: slide.name, results });

    const outName = `${stemOf(slide.name)}.geojson`;
    if (opts.skipExisting && slide.annotations) {
      results.push({ slide: slide.name, regions: 0, ms: 0, written: null, error: null });
      continue;
    }

    const started = performance.now();
    try {
      const source = await openSlide(slide);
      try {
        const res = await detectTissue(source, {
          model: opts.model,
          minAreaUm2: opts.minAreaUm2,
        });
        const annotations = res.polygons
          .filter((rings) => rings[0] && rings[0].length >= 4)
          .map((rings) =>
            makeAnnotation(
              { type: "Polygon", coordinates: rings },
              { classId: cls?.id ?? null, source: "model", modelId: opts.model?.id ?? "tissue-otsu" },
            ),
          );

        const fc = toFeatureCollection(annotations, classes, source.meta);
        const handle = await dir.getFileHandle(outName, { create: true });
        const writable = await handle.createWritable();
        await writable.write(JSON.stringify(fc));
        await writable.close();

        results.push({
          slide: slide.name,
          regions: annotations.length,
          ms: performance.now() - started,
          written: outName,
          error: null,
        });
      } finally {
        // Released before the next slide opens, not after the batch ends.
        await closeSlide(slide);
      }
    } catch (err) {
      // One unreadable slide must not end the run; it is recorded and the
      // queue moves on, because the whole point is leaving this unattended.
      results.push({
        slide: slide.name,
        regions: 0,
        ms: performance.now() - started,
        written: null,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }

  opts.onProgress?.({
    done: results.length,
    total: slides.length,
    current: "",
    results,
  });
  return results;
}
