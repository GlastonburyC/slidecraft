import { describe, expect, it } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { parseExpressionFile } from "../io/expressionFile";
import { geneValues } from "../ml/spatialResult";
import { differentialExpression } from "../ml/enrichment";

const PATH = "/Users/craig.glastonbury/EXETER/deepspot/TB_411 3A.browse.bin";
const run = existsSync(PATH) ? describe : describe.skip;

run("a real quantised whole-slide map", () => {
  const bytes = readFileSync(PATH);
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);

  it("parses, and holds what the run produced", { timeout: 60000 }, () => {
    const t0 = performance.now();
    const r = parseExpressionFile(buffer);
    const ms = performance.now() - t0;
    console.log(`  parsed in ${ms.toFixed(0)} ms`);
    expect(r.patches.length).toBe(306587);
    expect(r.genes.length).toBe(3000);
    expect(r.values).toBeInstanceOf(Uint8Array);
    expect(r.scale!.length).toBe(3000);
    // A byte view over the file's own bytes, not a copy of a gigabyte.
    expect(r.values.buffer).toBe(buffer);
  });

  it("decodes a gene to plausible expression, quickly", { timeout: 60000 }, () => {
    const r = parseExpressionFile(buffer);
    const gene = r.genes.includes("COL1A1") ? "COL1A1" : r.genes[0];
    const t0 = performance.now();
    const v = geneValues(r, gene)!;
    console.log(`  ${gene}: ${(performance.now() - t0).toFixed(0)} ms for ${v.length} patches`);
    expect(v.length).toBe(306587);
    expect([...v].every(Number.isFinite)).toBe(true);
    const max = v.reduce((a, b) => Math.max(a, b), -Infinity);
    const min = v.reduce((a, b) => Math.min(a, b), Infinity);
    console.log(`  range ${min.toFixed(3)} .. ${max.toFixed(3)}`);
    expect(max).toBeGreaterThan(0.05);
    expect(max).toBeLessThan(20);
  });

  it("runs a differential test over the whole transcriptome subset", { timeout: 120000 }, () => {
    const r = parseExpressionFile(buffer);
    // A contiguous block of patches, which is roughly what drawing a region gives.
    const inside = new Set<number>();
    for (let i = 20000; i < 40000; i++) inside.add(i);
    const t0 = performance.now();
    const res = differentialExpression(r, inside, 5, 0.05);
    const ms = performance.now() - t0;
    console.log(`  ${res.genes.length} genes tested in ${(ms / 1000).toFixed(1)} s`);
    console.log(`  top: ${res.genes.slice(0, 5).map((g) => `${g.gene} ${g.auc.toFixed(3)}`).join(", ")}`);
    expect(res.genes.length).toBeGreaterThan(100);
    expect(res.genes[0].auc).toBeGreaterThan(0.5);
    expect(res.genes.every((g) => Number.isFinite(g.auc) && Number.isFinite(g.p))).toBe(true);
    // Ordered most separable first.
    for (let i = 1; i < res.genes.length; i++) {
      expect(res.genes[i].auc).toBeLessThanOrEqual(res.genes[i - 1].auc);
    }
  });
});
