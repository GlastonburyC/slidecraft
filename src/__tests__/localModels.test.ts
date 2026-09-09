import { describe, expect, it } from "vitest";
import { validateImport, isLocalModel, LOCAL_SCHEME } from "../ml/localModels";
import { BUILTIN_MODELS } from "../ml/registry";

const onnx = (name: string, size = 5_000_000) =>
  new File([new Uint8Array(Math.min(size, 4096))], name, { type: "application/octet-stream" });
const big = (name: string) => Object.defineProperty(onnx(name), "size", { value: 5_000_000 });

const decoderOf = (id: string) =>
  BUILTIN_MODELS.find((m) => m.id === id)!.files.find((f) => f.part === "decoder")!;

describe("import validation", () => {
  const base = {
    name: "UNI v1",
    encoder: big("uni_encoder.onnx"),
    fallbackDecoder: decoderOf("slimsam-77"),
    inputSize: 1024,
    preset: "imagenet" as const,
  };

  it("accepts an encoder that borrows a built-in decoder", () => {
    expect(validateImport(base)).toBeNull();
  });

  it("requires a name", () => {
    expect(validateImport({ ...base, name: "  " })).toMatch(/name/i);
  });

  it("requires an encoder", () => {
    expect(validateImport({ ...base, encoder: undefined })).toMatch(/encoder/i);
  });

  it("rejects a file that is not .onnx", () => {
    expect(validateImport({ ...base, encoder: big("weights.pt") })).toMatch(/not a \.onnx/);
  });

  it("rejects an empty file", () => {
    const empty = Object.defineProperty(onnx("enc.onnx"), "size", { value: 10 });
    expect(validateImport({ ...base, encoder: empty })).toMatch(/empty/);
  });

  it("requires a decoder from somewhere", () => {
    expect(validateImport({ ...base, fallbackDecoder: undefined })).toMatch(/decoder/i);
  });

  it("accepts an explicit decoder without a fallback", () => {
    expect(
      validateImport({ ...base, fallbackDecoder: undefined, decoder: big("dec.onnx") }),
    ).toBeNull();
  });

  it("rejects an implausible input size", () => {
    for (const size of [0, 32, 8192, 1024.5]) {
      expect(validateImport({ ...base, inputSize: size })).toMatch(/Input size/);
    }
  });
});

describe("local model identity", () => {
  it("recognises imported models and leaves built-ins alone", () => {
    expect(isLocalModel({ ...BUILTIN_MODELS[0], id: "local-uni-abc" })).toBe(true);
    for (const m of BUILTIN_MODELS) expect(isLocalModel(m)).toBe(false);
  });

  it("addresses imported weights under a scheme that never hits the network", () => {
    expect(LOCAL_SCHEME).toBe("slidecraft-local:");
    expect(`${LOCAL_SCHEME}//local-x/encoder.onnx`.startsWith("http")).toBe(false);
  });
});
