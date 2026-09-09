import type { ResolvedFile, ResolvedSlide } from "./types";

/**
 * Turns a drag-and-drop (or folder picker) into openable slides.
 *
 * The hard case is multi-file formats. A MIRAX slide is `Foo.mrxs` *plus* a
 * sibling directory `Foo/` holding Slidedat.ini, Index.dat and Data####.dat.
 * Dropping only the .mrxs file cannot work — the pixels are all in the
 * directory — so we detect that and say so rather than failing opaquely.
 */

/** Primary files that identify a slide. */
const ENTRY_EXTENSIONS = new Set([
  "svs", "ndpi", "mrxs", "scn", "vms", "vmu",
  "tif", "tiff", "btf", "bif", "svslide", "dcm",
]);

/** Formats whose pixel data lives in a sibling directory named after the file. */
const SIDECAR_DIR_FORMATS = new Set(["mrxs"]);

/** Formats whose parts are sibling files sharing the basename prefix. */
const SIDECAR_PREFIX_FORMATS = new Set(["vms", "vmu"]);

/** Annotation sidecars, in the order they are preferred. */
const ANNOTATION_SUFFIXES = [".slidecraft.geojson", ".geojson", ".json"];

const extOf = (p: string) => p.slice(p.lastIndexOf(".") + 1).toLowerCase();
const baseOf = (p: string) => p.slice(p.lastIndexOf("/") + 1);
const dirOf = (p: string) => (p.includes("/") ? p.slice(0, p.lastIndexOf("/")) : "");
const stemOf = (p: string) => {
  const b = baseOf(p);
  const i = b.lastIndexOf(".");
  return i === -1 ? b : b.slice(0, i);
};

/** Recursively read a webkit FileSystemEntry tree into flat {path,file} records. */
async function walkEntry(entry: FileSystemEntry, prefix: string): Promise<ResolvedFile[]> {
  if (entry.isFile) {
    const file = await new Promise<File>((resolve, reject) =>
      (entry as FileSystemFileEntry).file(resolve, reject),
    );
    return [{ path: prefix + entry.name, file }];
  }

  const reader = (entry as FileSystemDirectoryEntry).createReader();
  const children: FileSystemEntry[] = [];
  // readEntries yields at most 100 per call and must be drained.
  for (;;) {
    const batch = await new Promise<FileSystemEntry[]>((resolve, reject) =>
      reader.readEntries(resolve, reject),
    );
    if (batch.length === 0) break;
    children.push(...batch);
  }

  const nested = await Promise.all(
    children.map((c) => walkEntry(c, `${prefix}${entry.name}/`)),
  );
  return nested.flat();
}

/** Flatten a DataTransfer into every file it contains, paths preserved. */
export async function filesFromDataTransfer(dt: DataTransfer): Promise<ResolvedFile[]> {
  // Entries must be captured synchronously — the item list is neutered after await.
  const entries: FileSystemEntry[] = [];
  const loose: File[] = [];
  for (const item of Array.from(dt.items)) {
    if (item.kind !== "file") continue;
    const entry = item.webkitGetAsEntry?.();
    if (entry) entries.push(entry);
    else {
      const f = item.getAsFile();
      if (f) loose.push(f);
    }
  }

  const walked = await Promise.all(entries.map((e) => walkEntry(e, "")));
  return [...walked.flat(), ...loose.map((file) => ({ path: file.name, file }))];
}

/** Files from an <input type="file" webkitdirectory>, preserving relative paths. */
export function filesFromInput(fileList: FileList): ResolvedFile[] {
  return Array.from(fileList).map((file) => ({
    // webkitRelativePath includes the picked folder as its first segment.
    path: (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name,
    file,
  }));
}

/**
 * Group a flat file list into openable slides, attaching each slide's
 * companion files and flagging any whose data is missing.
 */
export function resolveSlides(files: ResolvedFile[]): ResolvedSlide[] {
  const byPath = new Map(files.map((f) => [f.path, f]));
  const entries = files.filter((f) => ENTRY_EXTENSIONS.has(extOf(f.path)));

  // A DICOM directory is one slide made of many .dcm files, not many slides.
  const dicom = entries.filter((f) => extOf(f.path) === "dcm");
  const nonDicom = entries.filter((f) => extOf(f.path) !== "dcm");

  const slides: ResolvedSlide[] = [];

  for (const entry of nonDicom) {
    const ext = extOf(entry.path);
    const dir = dirOf(entry.path);
    const stem = stemOf(entry.path);
    const companions: ResolvedFile[] = [];
    let warning: string | null = null;

    if (SIDECAR_DIR_FORMATS.has(ext)) {
      const sidecarDir = dir ? `${dir}/${stem}/` : `${stem}/`;
      for (const f of files) if (f.path.startsWith(sidecarDir)) companions.push(f);
      if (companions.length === 0) {
        warning =
          `Missing the "${stem}/" data folder. A .mrxs file holds no pixels — ` +
          `drop the folder that contains both the .mrxs and its data directory.`;
      } else if (!byPath.has(`${sidecarDir}Slidedat.ini`)) {
        warning = `"${stem}/" is present but has no Slidedat.ini; the slide may be incomplete.`;
      }
    } else if (SIDECAR_PREFIX_FORMATS.has(ext)) {
      const prefix = dir ? `${dir}/${stem}` : stem;
      for (const f of files) {
        if (f.path !== entry.path && f.path.startsWith(prefix)) companions.push(f);
      }
      if (companions.length === 0) {
        warning = `No companion files found for "${baseOf(entry.path)}"; drop the whole folder.`;
      }
    }

    const all = [entry, ...companions];

    // A sidecar named after the slide travels with it. Matching on the stem
    // rather than on content means a folder of slides and a folder of results
    // can be dropped together and pair themselves up.
    const stemPath = dir ? `${dir}/${stem}` : stem;
    let annotations: File | null = null;
    for (const suffix of ANNOTATION_SUFFIXES) {
      const hit = byPath.get(stemPath + suffix);
      if (hit) { annotations = hit.file; break; }
    }

    slides.push({
      name: baseOf(entry.path),
      entryPath: entry.path,
      files: all,
      bytes: all.reduce((n, f) => n + f.file.size, 0),
      warning,
      annotations,
    });
  }

  if (dicom.length > 0) {
    // Group DICOM parts by containing directory.
    const groups = new Map<string, ResolvedFile[]>();
    for (const f of dicom) {
      const d = dirOf(f.path);
      const g = groups.get(d);
      if (g) g.push(f);
      else groups.set(d, [f]);
    }
    for (const [dir, group] of groups) {
      const sorted = [...group].sort((a, b) => a.path.localeCompare(b.path));
      slides.push({
        name: dir ? baseOf(dir) : sorted[0].path,
        entryPath: sorted[0].path,
        files: sorted,
        bytes: sorted.reduce((n, f) => n + f.file.size, 0),
        warning: null,
        annotations: null,
      });
    }
  }

  return slides.sort((a, b) => a.name.localeCompare(b.name));
}
