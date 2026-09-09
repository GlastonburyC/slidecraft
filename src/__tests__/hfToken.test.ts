import { beforeEach, describe, expect, it, vi } from "vitest";
import { getHfToken, setHfToken, fetchWeights } from "../ml/modelCache";

describe("Hugging Face token", () => {
  beforeEach(() => {
    setHfToken(null);
    vi.restoreAllMocks();
  });

  it("round-trips through local storage", () => {
    expect(getHfToken()).toBeNull();
    setHfToken("hf_example");
    expect(getHfToken()).toBe("hf_example");
    setHfToken(null);
    expect(getHfToken()).toBeNull();
  });

  it("ignores blank input rather than storing an empty token", () => {
    setHfToken("   ");
    expect(getHfToken()).toBeNull();
  });

  it("trims surrounding whitespace from a pasted token", () => {
    setHfToken("  hf_pasted \n");
    expect(getHfToken()).toBe("hf_pasted");
  });

  it("sends the token to huggingface.co", async () => {
    setHfToken("hf_secret");
    const fetchMock = vi.fn(
      async (_url: string, _init?: RequestInit) =>
        new Response(new Uint8Array([1, 2, 3, 4]), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    await fetchWeights("https://huggingface.co/org/repo/resolve/main/m.onnx", "encoder", 4);
    const headers = fetchMock.mock.calls[0][1]?.headers as Record<string, string> | undefined;
    expect(headers?.Authorization).toBe("Bearer hf_secret");
  });

  it("never sends the token to any other host", async () => {
    setHfToken("hf_secret");
    const fetchMock = vi.fn(
      async (_url: string, _init?: RequestInit) =>
        new Response(new Uint8Array([1, 2, 3, 4]), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    for (const url of [
      "https://example.com/weights.onnx",
      "https://huggingface.co.evil.test/weights.onnx",
      "http://localhost:5199/weights.onnx",
    ]) {
      fetchMock.mockClear();
      await fetchWeights(url, "encoder", 4);
      const headers = fetchMock.mock.calls[0][1]?.headers as Record<string, string> | undefined;
      expect(headers).toBeUndefined();
    }
  });

  it("explains a gated refusal instead of surfacing a bare 403", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("", { status: 403 })));
    await expect(
      fetchWeights("https://huggingface.co/org/gated/resolve/main/m.onnx", "encoder", 4),
    ).rejects.toThrow(/gated/i);
  });

  it("does not try to fetch imported weights over the network", async () => {
    const fetchMock = vi.fn((_url: string, _init?: RequestInit) => undefined);
    vi.stubGlobal("fetch", fetchMock);
    await expect(
      fetchWeights("slidecraft-local://local-uni/encoder.onnx", "encoder", 4),
    ).rejects.toThrow(/local storage/i);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
