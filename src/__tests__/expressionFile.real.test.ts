import { describe, expect, it } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { parseExpressionFile } from "../io/expressionFile";
import { geneValues } from "../ml/spatialResult";

// A map produced by the real thing: scripts/predict_expression.py over
// 1480_E1-1_L1-3.svs on a cluster V100, DeepSpot-M with the scgpt pathway.
// Skipped unless the file is present, because it is not in the repo -- the
// point is to catch the writer and the reader drifting apart, which a
// fixture written by the reader's own author cannot do.
const FILE = process.env.SLIDECRAFT_EXPRESSION_FIXTURE;

describe.runIf(FILE && existsSync(FILE))("a map written by the GPU script", () => {
  it("parses, and arrives with its provenance intact", () => {
    const buf = readFileSync(FILE!);
    const result = parseExpressionFile(
      buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer,
    );

    expect(result.genes).toEqual([
      "EPCAM", "PTPRC", "CD3D", "COL1A1", "MKI67", "VIM", "KRT19", "ACTA2",
    ]);
    expect(result.patches.length).toBe(4026);
    expect(result.values.length).toBe(4026 * 8);
    expect(result.values.every(Number.isFinite)).toBe(true);

    // The pathway has to survive the round trip, or two maps that cannot be
    // compared will look like they can.
    expect(result.modelId).toContain("scgpt");

    // fp16 stays fp16 in memory; geneValues is what decodes a column.
    expect(result.half).toBe(true);
    expect(result.values).toBeInstanceOf(Uint16Array);

    const epcam = geneValues(result, "EPCAM")!;
    expect(epcam.length).toBe(4026);
    const mean = epcam.reduce((a, b) => a + b, 0) / epcam.length;
    expect(mean).toBeGreaterThan(0.10);
    expect(mean).toBeLessThan(0.20);
  });
});
