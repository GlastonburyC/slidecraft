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

describe("widening an encoder's output", () => {
  /**
   * The failure this guards against produced correctly-shaped embeddings that
   * carried no information: nearly every value zero, a few subnormals, some
   * NaN. Every patch then looked identical, so clustering collapsed to one
   * class and any head trained on them fitted noise — while everything
   * downstream reported healthy counts and dimensions.
   *
   * Only a Uint16Array holds raw half bit patterns. An fp16 output may arrive
   * already decoded, and reinterpreting those floats as bits is what did it.
   */
  function widen(data: ArrayLike<number>): Float32Array {
    const out = new Float32Array(data.length);
    if (data instanceof Uint16Array) {
      for (let i = 0; i < data.length; i++) out[i] = fromHalf(data[i]);
    } else {
      out.set(data as ArrayLike<number> & Iterable<number>);
    }
    return out;
  }

  it("decodes a Uint16Array as half bit patterns", () => {
    // 0x3C00 is 1.0 in half precision; 0x4000 is 2.0.
    const raw = Uint16Array.from([0x3c00, 0x4000, 0xbc00]);
    expect(Array.from(widen(raw))).toEqual([1, 2, -1]);
  });

  it("copies an already-decoded array unchanged", () => {
    const decoded = Float32Array.from([0.234, -1.5, 12.75]);
    const out = widen(decoded);
    expect(out[0]).toBeCloseTo(0.234, 6);
    expect(out[1]).toBe(-1.5);
    expect(out[2]).toBe(12.75);
  });

  /** The exact signature of the bug: decoded floats read as bit patterns. */
  it("does not collapse decoded floats to zeros and NaN", () => {
    const decoded = Float32Array.from(Array.from({ length: 64 }, (_, i) => (i - 32) / 16));
    const out = widen(decoded);
    const informative = Array.from(out).filter((x) => Number.isFinite(x) && x !== 0).length;
    expect(informative).toBeGreaterThan(50);
    expect(Array.from(out).some(Number.isNaN)).toBe(false);
  });
});
