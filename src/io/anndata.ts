import { makeZip, zipSize, type ZipEntry } from "./zip";
import { valueAt, type SpatialResult } from "../ml/spatialResult";

/**
 * An expression map as AnnData, so the analysis can continue somewhere else.
 *
 * Slidecraft answers spatial questions on the slide. It does not do trajectory
 * inference, or differential testing across patients, or anything else the
 * Python stack already does well — and a map that cannot leave is a dead end.
 *
 * Written as **zarr inside a zip** rather than `.h5ad`, because HDF5 from
 * scratch in JavaScript is an unreasonable amount of code to get subtly wrong,
 * while zarr v2 is a handful of JSON documents beside raw little-endian blocks.
 * Uncompressed, so every chunk is exactly the bytes of the array it holds.
 *
 *     import anndata
 *     adata = anndata.read_zarr("slide.anndata.zarr")   # after unzipping
 *
 * The layout follows AnnData's on-disk spec: each group carries an
 * `encoding-type` and `encoding-version` in its attributes, and the reader
 * dispatches on those rather than on shape. Get one wrong and the file opens as
 * a bare zarr hierarchy instead of an AnnData, which is why they are spelled
 * out here rather than inferred.
 */

const ZARR_FORMAT = 2;
/** Roughly this many bytes per chunk, so no single block is unwieldy. */
const CHUNK_BYTES = 4 << 20;

const utf8 = new TextEncoder();
const json = (o: unknown): Uint8Array => utf8.encode(JSON.stringify(o));

function zarray(shape: number[], chunks: number[], dtype: string, extra: object = {}) {
  return json({
    zarr_format: ZARR_FORMAT,
    shape,
    chunks,
    dtype,
    compressor: null,
    fill_value: 0,
    order: "C",
    filters: null,
    ...extra,
  });
}

/**
 * Variable-length strings, in the layout numcodecs' `vlen-utf8` expects.
 *
 * A chunk is a little-endian count, then for each element its byte length
 * followed by its UTF-8 bytes. There is no padding and no terminator — the
 * lengths are the only structure — so an off-by-one here produces a file that
 * opens and then hands back nonsense, rather than one that fails.
 */
function vlenUtf8(values: string[]): Uint8Array {
  const encoded = values.map((v) => utf8.encode(v));
  const total = 4 + encoded.reduce((n, b) => n + 4 + b.length, 0);
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  view.setUint32(0, values.length, true);
  let at = 4;
  for (const b of encoded) {
    view.setUint32(at, b.length, true);
    at += 4;
    out.set(b, at);
    at += b.length;
  }
  return out;
}

function stringArray(path: string, values: string[], entries: ZipEntry[]): void {
  entries.push({
    name: `${path}/.zarray`,
    data: json({
      zarr_format: ZARR_FORMAT,
      shape: [values.length],
      chunks: [values.length],
      dtype: "|O",
      compressor: null,
      fill_value: null,
      order: "C",
      filters: [{ id: "vlen-utf8" }],
    }),
  });
  entries.push({
    name: `${path}/.zattrs`,
    data: json({ "encoding-type": "string-array", "encoding-version": "0.2.0" }),
  });
  entries.push({ name: `${path}/0`, data: vlenUtf8(values) });
}

function numericArray(
  path: string, values: Float32Array | Int32Array, entries: ZipEntry[],
): void {
  const dtype = values instanceof Float32Array ? "<f4" : "<i4";
  entries.push({ name: `${path}/.zarray`, data: zarray([values.length], [values.length], dtype) });
  entries.push({
    name: `${path}/.zattrs`,
    data: json({ "encoding-type": "array", "encoding-version": "0.2.0" }),
  });
  entries.push({
    name: `${path}/0`,
    data: new Uint8Array(values.buffer, values.byteOffset, values.byteLength).slice(),
  });
}

function group(path: string, attrs: object, entries: ZipEntry[]): void {
  entries.push({ name: `${path}/.zgroup`, data: json({ zarr_format: ZARR_FORMAT }) });
  entries.push({ name: `${path}/.zattrs`, data: json(attrs) });
}

function dataframe(
  path: string, indexName: string, index: string[], columns: Record<string, Float32Array | Int32Array>,
  entries: ZipEntry[],
): void {
  group(path, {
    "encoding-type": "dataframe",
    "encoding-version": "0.2.0",
    _index: indexName,
    "column-order": Object.keys(columns),
  }, entries);
  stringArray(`${path}/${indexName}`, index, entries);
  for (const [name, values] of Object.entries(columns)) {
    numericArray(`${path}/${name}`, values, entries);
  }
}

/** Bytes the dense expression matrix will occupy, before anything is built. */
export function matrixBytes(result: SpatialResult): number {
  return result.patches.length * result.genes.length * 4;
}

export interface AnnDataOptions {
  /** 1 where the patch centre falls on detected tissue; becomes `obs.in_tissue`. */
  tissueMask?: Uint8Array | null;
}

/**
 * Build the archive. Returns the bytes of a `.zarr.zip`.
 *
 * `X` is dense float32. Predicted expression has no zeros to speak of — every
 * gene gets a value for every patch — so a sparse layout would store the same
 * numbers plus two index arrays.
 */
export function toAnnDataZip(result: SpatialResult, opts: AnnDataOptions = {}): Uint8Array {
  const n = result.patches.length;
  const g = result.genes.length;
  const entries: ZipEntry[] = [];

  entries.push({ name: ".zgroup", data: json({ zarr_format: ZARR_FORMAT }) });
  entries.push({
    name: ".zattrs",
    data: json({ "encoding-type": "anndata", "encoding-version": "0.1.0" }),
  });

  // ---- X, chunked by row so no single block is unwieldy --------------------
  const rowsPerChunk = Math.max(1, Math.min(n, Math.floor(CHUNK_BYTES / (g * 4)) || 1));
  entries.push({
    name: "X/.zarray",
    data: zarray([n, g], [rowsPerChunk, g], "<f4"),
  });
  entries.push({
    name: "X/.zattrs",
    data: json({ "encoding-type": "array", "encoding-version": "0.2.0" }),
  });
  for (let start = 0, chunk = 0; start < n; start += rowsPerChunk, chunk++) {
    const rows = Math.min(rowsPerChunk, n - start);
    // A chunk is always the declared size; the tail is padded with the fill
    // value, which is what a reader expects to find there.
    const block = new Float32Array(rowsPerChunk * g);
    for (let r = 0; r < rows; r++) {
      const base = (start + r) * g;
      for (let c = 0; c < g; c++) block[r * g + c] = valueAt(result, base + c);
    }
    entries.push({
      name: `X/${chunk}.0`,
      data: new Uint8Array(block.buffer, block.byteOffset, block.byteLength).slice(),
    });
  }

  // ---- obs: one row per patch ---------------------------------------------
  const x = new Int32Array(n);
  const y = new Int32Array(n);
  const names: string[] = [];
  for (let i = 0; i < n; i++) {
    x[i] = result.patches[i].x;
    y[i] = result.patches[i].y;
    // Coordinates, not an index, because a patch's identity IS its position —
    // and this survives being concatenated with another slide's table.
    names.push(`${result.patches[i].x}_${result.patches[i].y}`);
  }
  const obsColumns: Record<string, Float32Array | Int32Array> = { x, y };
  if (opts.tissueMask && opts.tissueMask.length === n) {
    const inTissue = new Int32Array(n);
    for (let i = 0; i < n; i++) inTissue[i] = opts.tissueMask[i] ? 1 : 0;
    obsColumns.in_tissue = inTissue;
  }
  dataframe("obs", "patch", names, obsColumns, entries);

  // ---- var: one row per gene ----------------------------------------------
  dataframe("var", "gene", [...result.genes], {}, entries);

  // ---- obsm/spatial: patch CENTRES, which is what plotting expects ---------
  group("obsm", { "encoding-type": "dict", "encoding-version": "0.1.0" }, entries);
  const spatial = new Float32Array(n * 2);
  const half = result.side / 2;
  for (let i = 0; i < n; i++) {
    spatial[i * 2] = result.patches[i].x + half;
    spatial[i * 2 + 1] = result.patches[i].y + half;
  }
  entries.push({ name: "obsm/spatial/.zarray", data: zarray([n, 2], [n, 2], "<f4") });
  entries.push({
    name: "obsm/spatial/.zattrs",
    data: json({ "encoding-type": "array", "encoding-version": "0.2.0" }),
  });
  entries.push({
    name: "obsm/spatial/0.0",
    data: new Uint8Array(spatial.buffer, spatial.byteOffset, spatial.byteLength).slice(),
  });

  // ---- uns: where this came from ------------------------------------------
  /*
   * Provenance travels with the numbers or it is lost. These values are
   * predicted from morphology rather than measured, and the model and the
   * pathway that produced them decide whether two maps can be compared at all
   * — so they belong in the file, not in the filename.
   */
  group("uns", { "encoding-type": "dict", "encoding-version": "0.1.0" }, entries);
  const meta: Record<string, string> = {
    slide: result.slide,
    model: result.modelName,
    model_id: result.modelId,
    created_at: result.createdAt,
    patch_side_px: String(result.side),
    units: "level-0 slide pixels",
    provenance: "Predicted from H&E by Slidecraft. Not a measurement.",
  };
  for (const [key, value] of Object.entries(meta)) {
    entries.push({
      name: `uns/${key}/.zarray`,
      data: json({
        zarr_format: ZARR_FORMAT,
        shape: [1], chunks: [1], dtype: "|O",
        compressor: null, fill_value: null, order: "C",
        filters: [{ id: "vlen-utf8" }],
      }),
    });
    entries.push({
      name: `uns/${key}/.zattrs`,
      data: json({ "encoding-type": "string-array", "encoding-version": "0.2.0" }),
    });
    entries.push({ name: `uns/${key}/0`, data: vlenUtf8([value]) });
  }

  return makeZip(entries);
}

/** What the archive will weigh, for deciding whether to offer it at all. */
export function annDataZipBytes(result: SpatialResult): number {
  // The matrix dominates; everything else is a few hundred KB of names.
  return matrixBytes(result) + result.genes.length * 24 + result.patches.length * 40 + 65536;
}

export { zipSize };
