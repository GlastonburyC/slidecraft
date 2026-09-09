import { describe, expect, it } from "vitest";
import { cleanGeometry, renderableRings } from "../annotate/geometry";
import { areaOf, type Ring } from "../annotate/types";

/**
 * Polygons handed to the renderer must be tessellation-safe.
 *
 * deck.gl bridges each hole to its outer ring with a cut. Give it a
 * self-intersecting ring, or a hole that is not inside its parent, and the
 * triangulation runs that bridge across the scene as long thin slivers — the
 * "lines between the tissue masks". These test the normalisation that prevents
 * it, at the level where the fault actually occurs.
 */

const bowtie: Ring = [
  [0, 0],
  [100, 100],
  [100, 0],
  [0, 100],
  [0, 0],
];

const square = (x: number, y: number, s: number): Ring => [
  [x, y],
  [x + s, y],
  [x + s, y + s],
  [x, y + s],
  [x, y],
];

describe("polygon hygiene", () => {
  it("resolves a self-intersecting ring into valid pieces", () => {
    const out = cleanGeometry({ type: "Polygon", coordinates: [bowtie] });
    expect(out).not.toBeNull();
    // A bowtie is two triangles; either form is valid, neither may self-cross.
    const parts = out!.type === "Polygon" ? [out!.coordinates] : out!.coordinates;
    expect(parts.length).toBeGreaterThanOrEqual(1);
    // Total area is the two triangles, not the 100x100 hull.
    expect(areaOf(out!)).toBeLessThan(0.6 * 100 * 100);
    expect(areaOf(out!)).toBeGreaterThan(0.3 * 100 * 100);
  });

  it("discards a hole that is not inside its outer ring", () => {
    // The classic mis-assignment: a hole belonging to a distant polygon.
    const out = cleanGeometry({
      type: "Polygon",
      coordinates: [square(0, 0, 100), square(5000, 5000, 40)],
    });
    expect(out).not.toBeNull();
    const parts = out!.type === "Polygon" ? [out!.coordinates] : out!.coordinates;
    for (const rings of parts) {
      for (const ring of rings) {
        for (const [x, y] of ring) {
          // Nothing may reach out to the stray hole's position.
          expect(x).toBeLessThan(1000);
          expect(y).toBeLessThan(1000);
        }
      }
    }
  });

  it("keeps a hole that really is inside", () => {
    const out = cleanGeometry({
      type: "Polygon",
      coordinates: [square(0, 0, 100), square(30, 30, 40)],
    });
    expect(out).not.toBeNull();
    // 100^2 minus 40^2.
    expect(Math.round(areaOf(out!))).toBe(10000 - 1600);
  });

  it("does not merge two separate regions into one", () => {
    const out = cleanGeometry({
      type: "MultiPolygon",
      coordinates: [[square(0, 0, 100)], [square(5000, 0, 100)]],
    });
    expect(out!.type).toBe("MultiPolygon");
    expect((out as { coordinates: Ring[][] }).coordinates.length).toBe(2);
  });

  it("leaves an already-clean polygon alone", () => {
    const out = cleanGeometry({ type: "Polygon", coordinates: [square(10, 10, 80)] });
    expect(Math.round(areaOf(out!))).toBe(6400);
  });
});

/**
 * The renderer's own last line of defence.
 *
 * These polygons are the shapes that draw a chord across the slide. They can
 * reach the overlay from a document saved by an older build or from GeoJSON
 * written by another tool, so it is not enough that the detector no longer
 * produces them.
 */
describe("rings handed to the tessellator", () => {
  const square = (x: number, y: number, s: number): Ring => [
    [x, y], [x + s, y], [x + s, y + s], [x, y + s], [x, y],
  ];

  it("keeps a hole that really is inside its outer ring", () => {
    const rings = renderableRings([square(0, 0, 100), square(30, 30, 20)]);
    expect(rings).not.toBe(null);
    expect(rings!.length).toBe(2);
  });

  it("drops a hole that lies outside its outer ring", () => {
    // This is the chord: the bridge from the outer ring to a hole 40,000 slide
    // pixels away is drawn straight across everything in between.
    const rings = renderableRings([square(0, 0, 100), square(40000, 40000, 20)]);
    expect(rings!.length).toBe(1);
  });

  it("drops a degenerate outer ring rather than drawing it", () => {
    expect(renderableRings([[[0, 0], [10, 0], [0, 0]]])).toBe(null);
    expect(renderableRings([])).toBe(null);
  });

  it("refuses a ring carrying a non-finite coordinate", () => {
    expect(renderableRings([[[0, 0], [NaN, 0], [10, 10], [0, 0]]])).toBe(null);
  });

  it("drops only the bad hole, keeping the good one", () => {
    const rings = renderableRings([
      square(0, 0, 100), square(20, 20, 10), square(9000, 9000, 10), square(60, 60, 10),
    ]);
    expect(rings!.length).toBe(3);
  });
});
