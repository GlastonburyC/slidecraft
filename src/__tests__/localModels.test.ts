import { describe, expect, it } from "vitest";
import { validateImport, isLocalModel, LOCAL_PREFIX } from "../ml/localModels";
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

  /**
   * The Cache Storage API rejects any scheme but http and https, at `put`,
   * with a message about schemes that says nothing about what to do. A custom
   * `slidecraft-local:` URL therefore made every import fail the moment it
   * tried to store the weights — so what this has to check is not that the URL
   * looks synthetic, but that a Request can actually be built from it.
   */
  it("addresses imported weights with a URL Cache Storage will accept", () => {
    const url = `${LOCAL_PREFIX}local-x/encoder.onnx`;
    expect(() => new Request(url)).not.toThrow();
    expect(new URL(url).protocol).toBe("https:");
  });

  /**
   * ...and that it can never reach anyone's server. `.invalid` is reserved by
   * RFC 2606 and guaranteed not to resolve.
   */
  it("points at a host that cannot resolve", () => {
    expect(new URL(LOCAL_PREFIX).hostname.endsWith(".invalid")).toBe(true);
  });

  it("does not send a Hugging Face token to it", () => {
    const host = new URL(`${LOCAL_PREFIX}x/encoder.onnx`).hostname;
    expect(host === "huggingface.co" || host.endsWith(".huggingface.co")).toBe(false);
  });
});
