import { describe, expect, it } from "vitest";
import { EmbeddingCache } from "../ml/embeddingCache";

const id = { slideKey: "slide-a", modelId: "uni", level: 0, patchPx: 128, dim: 4 };

describe("embedding cache", () => {
  it("stores and returns vectors by key", async () => {
    const c = await EmbeddingCache.open(id, 8);
    c.put("p0", Float32Array.from([1, 2, 3, 4]));
    expect(Array.from(c.get("p0")!)).toEqual([1, 2, 3, 4]);
    expect(c.get("nope")).toBeUndefined();
    expect(c.size).toBe(1);
  });

  it("reports which patches still need encoding", async () => {
    const c = await EmbeddingCache.open(id, 8);
    c.put("p0", Float32Array.from([1, 1, 1, 1]));
    c.put("p2", Float32Array.from([2, 2, 2, 2]));
    expect(c.missing(["p0", "p1", "p2", "p3"])).toEqual(["p1", "p3"]);
  });

  it("overwrites in place rather than growing", async () => {
    const c = await EmbeddingCache.open(id, 8);
    c.put("p0", Float32Array.from([1, 1, 1, 1]));
    c.put("p0", Float32Array.from([9, 9, 9, 9]));
    expect(c.size).toBe(1);
    expect(Array.from(c.get("p0")!)).toEqual([9, 9, 9, 9]);
  });

  it("grows past its initial capacity without corrupting earlier vectors", async () => {
    const c = await EmbeddingCache.open(id, 2);
    for (let i = 0; i < 50; i++) c.put(`p${i}`, Float32Array.from([i, i, i, i]));
    expect(c.size).toBe(50);
    expect(Array.from(c.get("p0")!)).toEqual([0, 0, 0, 0]);
    expect(Array.from(c.get("p49")!)).toEqual([49, 49, 49, 49]);
  });

  it("refuses a vector of the wrong dimensionality", async () => {
    const c = await EmbeddingCache.open(id, 8);
    expect(() => c.put("p0", Float32Array.from([1, 2, 3]))).toThrow(/dim/);
  });

  it("works without OPFS, which is the private-window case", async () => {
    const c = await EmbeddingCache.open(id, 8);
    c.put("p0", Float32Array.from([1, 2, 3, 4]));
    await expect(c.flush()).resolves.toBeUndefined();
    expect(Array.from(c.get("p0")!)).toEqual([1, 2, 3, 4]);
  });
});
