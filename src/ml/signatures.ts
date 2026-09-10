import { geneValues, type SpatialResult } from "./spatialResult";

/**
 * Cell-type signatures, scored over predicted expression.
 *
 * Reading one predicted gene inherits all of that gene's error, and per-gene
 * accuracy from H&E is modest across most of the transcriptome. A signature is
 * a weighted average over tens of genes, so the independent part of the noise
 * averages down while the shared signal does not — and "where are the T cells"
 * is usually the question anyway.
 *
 * Derived from a single-cell atlas by `scripts/signatures_from_cellxgene.py`.
 */

export interface SignatureGene {
  gene: string;
  /** How specific this gene is to the type; log fold change from the atlas. */
  weight: number;
}

export interface Signature {
  name: string;
  genes: SignatureGene[];
  /** Cells the atlas averaged to derive it, for judging how much to trust it. */
  cells?: number;
}

export interface SignatureSet {
  source: string;
  organism?: string;
  signatures: Signature[];
}

export class InvalidSignatures extends Error {}

export function parseSignatures(raw: unknown): SignatureSet {
  if (!raw || typeof raw !== "object") throw new InvalidSignatures("That is not a signature file.");
  const obj = raw as Record<string, unknown>;
  const list = obj.signatures;
  if (!Array.isArray(list) || list.length === 0) {
    throw new InvalidSignatures("No signatures in that file.");
  }

  const signatures: Signature[] = [];
  for (const entry of list) {
    const s = entry as Record<string, unknown>;
    const name = typeof s.name === "string" ? s.name.trim() : "";
    const genes = Array.isArray(s.genes) ? s.genes : [];
    if (!name || genes.length === 0) continue;

    const parsed: SignatureGene[] = [];
    for (const g of genes) {
      // Accept either {gene, weight} or a bare symbol, so a hand-written list
      // of markers works without anyone having to invent weights for it.
      if (typeof g === "string") parsed.push({ gene: g, weight: 1 });
      else if (g && typeof g === "object") {
        const rec = g as Record<string, unknown>;
        const gene = typeof rec.gene === "string" ? rec.gene : null;
        const weight = typeof rec.weight === "number" && Number.isFinite(rec.weight) ? rec.weight : 1;
        if (gene) parsed.push({ gene, weight });
      }
    }
    if (parsed.length) {
      signatures.push({ name, genes: parsed, cells: typeof s.cells === "number" ? s.cells : undefined });
    }
  }

  if (!signatures.length) throw new InvalidSignatures("No usable signatures in that file.");
  return {
    source: typeof obj.source === "string" ? obj.source : "imported",
    organism: typeof obj.organism === "string" ? obj.organism : undefined,
    signatures,
  };
}

export interface Coverage {
  /** Signature genes the loaded model actually predicts. */
  found: number;
  total: number;
}

export function coverageOf(result: SpatialResult, signature: Signature): Coverage {
  const have = new Set(result.genes);
  return {
    found: signature.genes.filter((g) => have.has(g.gene)).length,
    total: signature.genes.length,
  };
}

/**
 * Score a signature over every patch.
 *
 * Each gene is standardised across the patches before being averaged. That step
 * is not optional: predicted expression carries an arbitrary scale and offset
 * per gene, so an unstandardised mean is dominated by whichever gene happens to
 * have the largest numbers rather than by the biology. Standardising first puts
 * every gene on the same footing, which is what makes the weights mean what
 * they say.
 *
 * The consequence is that a score is **relative within this slide**. It says
 * where a cell type is concentrated here; it does not compare across slides,
 * and a slide with none of a cell type still has high-scoring patches. That is
 * inherent to scoring without a calibrated reference, and worth saying out loud
 * rather than hiding behind a colour bar.
 */
export function scoreSignature(result: SpatialResult, signature: Signature): Float32Array | null {
  const n = result.patches.length;
  if (n === 0) return null;

  const score = new Float32Array(n);
  let weightSum = 0;

  for (const { gene, weight } of signature.genes) {
    const values = geneValues(result, gene);
    if (!values) continue;

    let mean = 0;
    for (let i = 0; i < n; i++) mean += values[i];
    mean /= n;

    let variance = 0;
    for (let i = 0; i < n; i++) {
      const d = values[i] - mean;
      variance += d * d;
    }
    const sd = Math.sqrt(variance / n);
    // A gene that is flat across every patch says nothing about where anything
    // is; including it would only add its weight to the denominator.
    if (sd < 1e-6) continue;

    const w = Math.abs(weight) > 1e-9 ? weight : 1;
    for (let i = 0; i < n; i++) score[i] += (w * (values[i] - mean)) / sd;
    weightSum += Math.abs(w);
  }

  if (weightSum === 0) return null;
  for (let i = 0; i < n; i++) score[i] /= weightSum;
  return score;
}

/** Signatures the loaded model can say anything about, best covered first. */
export function usableSignatures(
  result: SpatialResult,
  set: SignatureSet,
  minGenes = 3,
): { signature: Signature; coverage: Coverage }[] {
  return set.signatures
    .map((signature) => ({ signature, coverage: coverageOf(result, signature) }))
    .filter((s) => s.coverage.found >= minGenes)
    .sort((a, b) => b.coverage.found - a.coverage.found);
}
