import { describe, expect, it } from "vitest";
import { locate } from "../ml/localWeights";

/**
 * Where imported weights live.
 *
 * A modern encoder is over a gigabyte, and Cache.put fails on a body that
 * size — in Chrome with "Unexpected internal error", which names nothing and
 * leaves a model spec pointing at weights that were never written. OPFS takes
 * files this large and accepts a stream, so the model is never held whole in
 * memory on the way in.
 */

describe("addressing imported weights", () => {
  it("maps a weights URL to a folder and a file", () => {
    expect(locate("https://local.slidecraft.invalid/local-encoder-virchow2-abc/model.onnx")).toEqual({
      id: "local-encoder-virchow2-abc",
      part: "model.onnx",
    });
  });

  it("keeps each model's parts together under its own id", () => {
    const enc = locate("https://local.slidecraft.invalid/local-sam-x/encoder.onnx")!;
    const dec = locate("https://local.slidecraft.invalid/local-sam-x/decoder.onnx")!;
    expect(enc.id).toBe(dec.id);
    expect(enc.part).not.toBe(dec.part);
  });

  /** Anything that is not <id>/<part> is not ours to resolve. */
  it("refuses a path it cannot place", () => {
    expect(locate("https://local.slidecraft.invalid/model.onnx")).toBe(null);
    expect(locate("https://local.slidecraft.invalid/a/b/c.onnx")).toBe(null);
    expect(locate("not a url")).toBe(null);
  });

  it("is not confused by a query or a fragment", () => {
    expect(locate("https://local.slidecraft.invalid/id-1/model.onnx?v=2#x")).toEqual({
      id: "id-1",
      part: "model.onnx",
    });
  });
});
