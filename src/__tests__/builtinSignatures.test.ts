import { describe, expect, it } from "vitest";
import { BUILTIN_SIGNATURES } from "../ml/builtinSignatures";
import { scoreSignature, usableSignatures } from "../ml/signatures";
import type { SpatialResult } from "../ml/spatialResult";

/** A map over `genes`, with `values[patch][gene]`. */
function mapOf(genes: string[], values: number[][]): SpatialResult {
  const flat = new Float32Array(values.length * genes.length);
  values.forEach((row, i) => row.forEach((v, j) => { flat[i * genes.length + j] = v; }));
  return {
    slide: "test.svs",
    genes,
    patches: values.map((_, i) => ({ x: i * 100, y: 0 })),
    side: 100,
    values: flat,
    modelId: "test",
    modelName: "test",
    roiId: null,
    ms: 0,
    createdAt: new Date().toISOString(),
  } as unknown as SpatialResult;
}

describe("the built-in modules", () => {
  it("name every gene as a bare symbol with unit weight", () => {
    for (const sig of BUILTIN_SIGNATURES.signatures) {
      expect(sig.genes.length).toBeGreaterThan(2);
      for (const g of sig.genes) {
        expect(g.weight).toBe(1);
        expect(g.gene).toMatch(/^[A-Z0-9][A-Z0-9-]*$/);
      }
    }
  });

  it("has no duplicate names, and no gene twice inside one module", () => {
    const names = BUILTIN_SIGNATURES.signatures.map((s) => s.name);
    expect(new Set(names).size).toBe(names.length);
    for (const sig of BUILTIN_SIGNATURES.signatures) {
      const genes = sig.genes.map((g) => g.gene);
      expect(new Set(genes).size).toBe(genes.length);
    }
  });

  it("hides modules a narrow map cannot cover, and keeps the ones it can", () => {
    // Eight genes, as the first DeepSpot-M run used.
    const narrow = mapOf(
      ["EPCAM", "PTPRC", "CD3D", "COL1A1", "MKI67", "VIM", "KRT19", "ACTA2"],
      Array.from({ length: 6 }, (_, i) => [i, 8 - i, i, 8 - i, i, 8 - i, i, 8 - i]),
    );
    const usable = usableSignatures(narrow, BUILTIN_SIGNATURES);
    const names = usable.map((u) => u.signature.name);

    // Stroma has COL1A1, VIM, ACTA2; T cells have PTPRC and CD3D — only two,
    // which is below the floor, so it stays out rather than being scored on a
    // pair of genes and presented as if it meant something.
    expect(names).toContain("Stroma / fibrosis");
    expect(names).not.toContain("Neuroendocrine");
    expect(names).not.toContain("T cells");
    for (const u of usable) expect(u.coverage.found).toBeGreaterThanOrEqual(3);
  });

  it("standardises each gene, so an abundant one cannot drown the module", () => {
    /*
     * COL1A1 is two orders of magnitude larger than the rest and varies the
     * opposite way. Without the per-gene standardisation in scoreSignature it
     * would set the module's shape by itself.
     */
    const genes = ["COL1A1", "ACTA2", "VIM"];
    const result = mapOf(genes, [
      [900, 1, 1],
      [600, 2, 2],
      [300, 3, 3],
      [100, 4, 4],
    ]);
    const stroma = BUILTIN_SIGNATURES.signatures.find((s) => s.name === "Stroma / fibrosis")!;
    const score = scoreSignature(result, stroma)!;
    expect(score).not.toBeNull();

    // Two of the three genes rise across the patches and one falls, so the
    // module rises. A module dominated by COL1A1 alone would fall.
    expect(score[3]).toBeGreaterThan(score[0]);
  });

  it("scores nothing when the map shares no genes with a module", () => {
    const unrelated = mapOf(["AAAA", "BBBB"], [[1, 2], [3, 4]]);
    const stroma = BUILTIN_SIGNATURES.signatures.find((s) => s.name === "Stroma / fibrosis")!;
    expect(scoreSignature(unrelated, stroma)).toBeNull();
  });
});
