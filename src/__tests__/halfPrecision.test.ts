import { describe, expect, it } from "vitest";

/**
 * The fp16 conversions the encoder worker relies on.
 *
 * An fp16 export takes fp16 input and answers in fp16, so the worker narrows
 * on the way in and widens on the way out. Both directions are duplicated in
 * the expression-file reader, and a mistake in either is not detectable
 * downstream — the numbers stay finite and plausible, the embeddings are just
 * wrong, and every head fitted on them is quietly worse.
 */

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

function toHalf(value: number): number {
  f32[0] = value;
  const bits = u32[0];
  const sign = (bits >>> 16) & 0x8000;
  const exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  const fraction = (bits >>> 13) & 0x3ff;
  if (exponent <= 0) return sign;
  if (exponent >= 31) return sign | 0x7c00;
  return sign | (exponent << 10) | fraction;
}

function fromHalf(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits & 0x7c00) >> 10;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

describe("half precision round trip", () => {
  it("keeps exactly representable values exact", () => {
    for (const v of [0, 1, -1, 0.5, -0.5, 2, 1024, -2048, 0.25]) {
      expect(fromHalf(toHalf(v))).toBe(v);
    }
  });

  /**
   * The range that matters: normalised pixel values after ImageNet
   * standardisation land roughly within ±3.
   */
  it("holds three significant figures across the normalised range", () => {
    for (let v = -3; v <= 3; v += 0.017) {
      expect(fromHalf(toHalf(v))).toBeCloseTo(v, 2);
    }
  });

  it("keeps sign through zero", () => {
    expect(Object.is(fromHalf(toHalf(-0)), -0) || fromHalf(toHalf(-0)) === 0).toBe(true);
    expect(fromHalf(toHalf(0))).toBe(0);
  });

  it("saturates rather than wrapping when a value is out of range", () => {
    // 70000 exceeds half's maximum of 65504; it must become infinity, not a
    // small number from a wrapped exponent.
    expect(fromHalf(toHalf(70000))).toBe(Infinity);
    expect(fromHalf(toHalf(-70000))).toBe(-Infinity);
  });

  it("flushes values below half's subnormal range to zero, not to noise", () => {
    expect(Math.abs(fromHalf(toHalf(1e-9)))).toBeLessThan(1e-6);
  });

  it("preserves ordering, which is what a linear head reads", () => {
    const values = [-2.5, -1, -0.3, 0, 0.3, 1, 2.5];
    const round = values.map((v) => fromHalf(toHalf(v)));
    for (let i = 1; i < round.length; i++) expect(round[i]).toBeGreaterThan(round[i - 1]);
  });
});
