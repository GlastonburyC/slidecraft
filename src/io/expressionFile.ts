import type { SpatialResult } from "../ml/spatialResult";

/**
 * A precomputed expression map, as a file.
 *
 * Running all 19,338 genes over a slide is a GPU job, not a browser one — a
 * 1B-parameter encoder plus a decoder that scales with gene count is minutes
 * per patch in WASM. So the heavy pass happens wherever the GPU is, writes one
 * of these next to the slide, and Slidecraft loads it in a second and treats it
 * exactly like a prediction it computed itself.
 *
 * The container is a JSON header followed by a raw value block, rather than
 * JSON throughout: 10,000 patches by 19,338 genes is 193 million numbers, which
 * is about 40 MB as fp16 and well over a gigabyte as JSON text.
 *
 *   magic      8 bytes   "SCEXPR1\0"
 *   headerLen  uint32    little-endian
 *   header     JSON, utf8
 *   values     patches x genes, row-major, fp16/fp32 little-endian or uint8
 *
 * The uint8 form is what `scripts/subset_expression.py` writes, and it is the
 * only way a whole-transcriptome map reaches a browser at all: 306,587 patches
 * by 19,338 genes is 11.9 GB as fp16, and a tab manages about 1.2 GB. Each gene
 * carries its own `scale` and `zero`, so a byte spans that gene's own range
 * rather than a range set by whichever gene in the map happened to be loudest.
 *
 * Coordinates in the header are level-0 pixels **of the slide it was computed
 * on**. A map is therefore tied to its slide, which is why the loader checks
 * before drawing it over a different one.
 */

const MAGIC = "SCEXPR1\0";

export interface ExpressionHeader {
  slide: string;
  genes: string[];
  /** Top-left of each patch, level-0 pixels, matching the value rows. */
  patches: { x: number; y: number }[];
  /** Patch side in level-0 pixels. */
  side: number;
  dtype: "float16" | "float32" | "uint8";
  /**
   * Per gene, for `uint8` maps: `value = code * scale[g] + zero[g]`.
   *
   * `zero` is at most 0, so a gene that never goes negative decodes code 0 to
   * exactly 0 — a patch with no expression has to stay indistinguishable from
   * one, or every sparse gene picks up a floor it does not have.
   */
  scale?: number[];
  zero?: number[];
  model: string;
  modelId?: string;
  /**
   * Which of DeepSpot-M's five gene-embedding pathways produced this, if any.
   * The same tile through two pathways gives two different numbers, so two maps
   * are only comparable when this agrees.
   */
  source?: string;
  createdAt?: string;
  /** Microns per pixel of the slide it was computed on, for a sanity check. */
  mpp?: number | null;
}

export class InvalidExpressionFile extends Error {}

export function parseExpressionFile(buffer: ArrayBuffer): SpatialResult {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 12) throw new InvalidExpressionFile("That file is too short to be an expression map.");

  const magic = new TextDecoder().decode(bytes.subarray(0, 8));
  if (magic !== MAGIC) {
    throw new InvalidExpressionFile(
      "That is not a Slidecraft expression map. Write one with scripts/predict_expression.py.",
    );
  }

  const view = new DataView(buffer);
  const headerLen = view.getUint32(8, true);
  const headerEnd = 12 + headerLen;
  if (headerEnd > bytes.length) throw new InvalidExpressionFile("The header is truncated.");

  let header: ExpressionHeader;
  try {
    header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, headerEnd)));
  } catch (err) {
    throw new InvalidExpressionFile(
      `The header is not readable: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  const nPatches = header.patches?.length ?? 0;
  const nGenes = header.genes?.length ?? 0;
  if (!nPatches || !nGenes) throw new InvalidExpressionFile("The header lists no patches or no genes.");

  const expected = nPatches * nGenes;
  const wide = header.dtype === "float32";
  const quantised = header.dtype === "uint8";
  const unit = wide ? 4 : quantised ? 1 : 2;
  const payloadBytes = bytes.length - headerEnd;
  const got = Math.floor(payloadBytes / unit);

  if (got !== expected) {
    throw new InvalidExpressionFile(
      `The header describes ${nPatches} patches by ${nGenes} genes (${expected} values), ` +
        `but the file holds ${got}. It is truncated or was written by a different tool.`,
    );
  }

  /*
   * Kept in the precision it arrived in, and viewed rather than copied.
   *
   * A whole-transcriptome map is 32,054 patches by 19,338 genes. Widening that
   * to fp32 up front costs 2.5 GB, and the `buffer.slice` that used to feed it
   * another 1.2 GB on top of the 1.2 GB buffer — about 5 GB peak to display one
   * gene at a time, which no tab survives. A view over the original bytes costs
   * nothing, and `valueAt` decodes a half where one is actually read.
   *
   * A typed-array view has to start on a multiple of its element size, and the
   * header length is whatever the JSON came to, so an odd-length header still
   * needs the copy. That is a rare case and a correct one, not a silent
   * fallback to something wrong.
   */
  let values: Float32Array | Uint16Array | Uint8Array;
  const aligned = headerEnd % unit === 0;
  if (wide) {
    values = aligned
      ? new Float32Array(buffer, headerEnd, expected)
      : new Float32Array(buffer.slice(headerEnd));
  } else if (quantised) {
    // A byte view needs no alignment at all, so this branch never copies.
    values = new Uint8Array(buffer, headerEnd, expected);
  } else {
    values = aligned
      ? new Uint16Array(buffer, headerEnd, expected)
      : new Uint16Array(buffer.slice(headerEnd));
  }

  /*
   * A quantised map without its scales decodes to whatever the bytes happen to
   * be — a plausible-looking field of numbers between 0 and 255, drawn with a
   * colour ramp and tested for enrichment like any other. There is no way to
   * notice that downstream, so it is refused here.
   */
  let scale: Float32Array | undefined;
  let zero: Float32Array | undefined;
  if (quantised) {
    if (header.scale?.length !== nGenes || header.zero?.length !== nGenes) {
      throw new InvalidExpressionFile(
        `A uint8 map needs one scale and one zero per gene; this one lists ` +
          `${header.scale?.length ?? 0} and ${header.zero?.length ?? 0} for ${nGenes} genes.`,
      );
    }
    scale = Float32Array.from(header.scale);
    zero = Float32Array.from(header.zero);
  }

  return {
    genes: header.genes,
    values,
    half: !wide && !quantised,
    scale,
    zero,
    patches: header.patches.map((p, index) => ({
      index,
      col: 0,
      row: 0,
      x: p.x,
      y: p.y,
      size: header.side,
    })),
    side: header.side,
    modelId: header.modelId ?? "precomputed",
    modelName: header.model,
    slide: header.slide,
    roiId: null,
    ms: 0,
    createdAt: header.createdAt ?? new Date().toISOString(),
  };
}

/**
 * Whether a map belongs to the slide now open.
 *
 * Compared on the name rather than on pixel content, because that is what the
 * file records — and drawing another slide's expression over this one would be
 * the most convincing wrong result the app could produce, so it is refused
 * rather than warned about.
 */
export function matchesSlide(result: SpatialResult, slideName: string): boolean {
  const stem = (s: string) => s.replace(/\.[^.]+$/, "").toLowerCase();
  return stem(result.slide) === stem(slideName);
}
