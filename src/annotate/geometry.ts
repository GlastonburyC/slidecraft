import { difference, intersection, union, type Geom } from "polyclip-ts";
import {
  isAreaGeometry,
  simplifyRing,
  type AreaGeometry,
  type Geometry,
  type Position,
  type Ring,
} from "./types";

export type BooleanOp = "union" | "subtract" | "intersect";

/** GeoJSON area geometry -> polyclip MultiPoly. */
function toMulti(g: AreaGeometry): Ring[][] {
  return g.type === "Polygon" ? [g.coordinates] : g.coordinates;
}

/** polyclip MultiPoly -> GeoJSON, collapsing the single-polygon case. */
export function fromMulti(mp: Ring[][]): AreaGeometry | null {
  const clean = mp.filter((poly) => poly.length > 0 && poly[0].length >= 4);
  if (clean.length === 0) return null;
  return clean.length === 1
    ? { type: "Polygon", coordinates: clean[0] }
    : { type: "MultiPolygon", coordinates: clean };
}

export function booleanOp(
  op: BooleanOp,
  a: AreaGeometry,
  b: AreaGeometry,
): AreaGeometry | null {
  const ga = toMulti(a) as Geom;
  const gb = toMulti(b) as Geom;
  try {
    const out =
      op === "union"
        ? union(ga, gb)
        : op === "subtract"
          ? difference(ga, gb)
          : intersection(ga, gb);
    return fromMulti(out as Ring[][]);
  } catch {
    // polyclip throws on some degenerate self-touching input; treat as a no-op
    // rather than losing the user's existing object.
    return null;
  }
}

/** Repair self-intersections by unioning a geometry with itself. */
export function cleanGeometry(g: AreaGeometry): AreaGeometry | null {
  try {
    return fromMulti(union(toMulti(g) as Geom) as Ring[][]);
  } catch {
    return null;
  }
}

export function closeRing(ring: Ring): Ring {
  if (ring.length < 3) return ring;
  const [fx, fy] = ring[0];
  const [lx, ly] = ring[ring.length - 1];
  return fx === lx && fy === ly ? ring : [...ring, [fx, fy]];
}

export function circleRing(cx: number, cy: number, r: number, segments = 32): Ring {
  const ring: Ring = [];
  for (let i = 0; i < segments; i++) {
    const t = (i / segments) * Math.PI * 2;
    ring.push([cx + Math.cos(t) * r, cy + Math.sin(t) * r]);
  }
  ring.push(ring[0]);
  return ring;
}

export function rectRing(x0: number, y0: number, x1: number, y1: number): Ring {
  const [ax, bx] = x0 <= x1 ? [x0, x1] : [x1, x0];
  const [ay, by] = y0 <= y1 ? [y0, y1] : [y1, y0];
  return [
    [ax, ay],
    [bx, ay],
    [bx, by],
    [ax, by],
    [ax, ay],
  ];
}

/**
 * Area swept by a round brush along a polyline.
 *
 * Built as the union of a disc at every vertex (round joins and caps) and a
 * quad per segment. That is deliberately naive rather than clever: an
 * offset-outline construction self-intersects at bends and reversals, and the
 * boolean cleanup then shreds the stroke into fragments instead of repairing
 * it. Unioning convex pieces is correct by construction at any shape.
 *
 * The path is simplified first, so the piece count follows the stroke's real
 * shape rather than the pointer's sampling rate.
 */
export function strokePolygon(path: Position[], radius: number): AreaGeometry | null {
  if (path.length === 0 || radius <= 0) return null;

  const pts = dedupe(simplifyRing(path as Ring, radius / 3));
  const parts: Ring[][] = pts.map((p) => [circleRing(p[0], p[1], radius, 24)]);

  for (let i = 1; i < pts.length; i++) {
    const [x0, y0] = pts[i - 1];
    const [x1, y1] = pts[i];
    const dx = x1 - x0;
    const dy = y1 - y0;
    const len = Math.hypot(dx, dy);
    if (len < 1e-9) continue;
    // Segment normal, scaled to the brush radius.
    const nx = (-dy / len) * radius;
    const ny = (dx / len) * radius;
    parts.push([
      closeRing([
        [x0 + nx, y0 + ny],
        [x1 + nx, y1 + ny],
        [x1 - nx, y1 - ny],
        [x0 - nx, y0 - ny],
      ]),
    ]);
  }

  try {
    const merged = union(parts[0] as Geom, ...(parts.slice(1) as Geom[]));
    return fromMulti(merged as Ring[][]);
  } catch {
    return null;
  }
}

function dedupe(pts: Ring): Ring {
  const out: Ring = [];
  for (const p of pts) {
    const last = out[out.length - 1];
    if (!last || last[0] !== p[0] || last[1] !== p[1]) out.push(p);
  }
  return out;
}

/** Shortest distance from a point to any vertex-to-vertex segment. */
export function distanceToGeometry(g: Geometry, x: number, y: number): number {
  let best = Infinity;
  const scan = (ring: Position[]) => {
    for (let i = 1; i < ring.length; i++) {
      best = Math.min(best, segmentDistance(x, y, ring[i - 1], ring[i]));
    }
  };
  if (g.type === "Point") {
    return Math.hypot(g.coordinates[0] - x, g.coordinates[1] - y);
  }
  if (g.type === "LineString") scan(g.coordinates);
  else if (g.type === "Polygon") g.coordinates.forEach(scan);
  else g.coordinates.forEach((p) => p.forEach(scan));
  return best;
}

function segmentDistance(px: number, py: number, a: Position, b: Position): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lenSq = dx * dx + dy * dy;
  let t = lenSq > 0 ? ((px - a[0]) * dx + (py - a[1]) * dy) / lenSq : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return Math.hypot(a[0] + t * dx - px, a[1] + t * dy - py);
}

/**
 * Guard so tools never commit a shape the user cannot see or select.
 *
 * Non-finite coordinates are rejected too: a viewport that is momentarily
 * unmeasurable (a hidden tab, a zero-sized container) yields NaN through the
 * pixel-to-slide transform, and a NaN geometry silently corrupts the spatial
 * index and every bounding-box test that follows.
 */
export function isDegenerate(g: Geometry): boolean {
  if (!allFinite(g)) return true;
  if (!isAreaGeometry(g)) return false;
  const rings = g.type === "Polygon" ? [g.coordinates[0]] : g.coordinates.map((p) => p[0]);
  return rings.every((r) => !r || r.length < 4);
}

function allFinite(g: Geometry): boolean {
  const ok = (p: Position) => Number.isFinite(p[0]) && Number.isFinite(p[1]);
  if (g.type === "Point") return ok(g.coordinates);
  if (g.type === "LineString") return g.coordinates.every(ok);
  if (g.type === "Polygon") return g.coordinates.every((r) => r.every(ok));
  return g.coordinates.every((poly) => poly.every((r) => r.every(ok)));
}

/** Translate any geometry by a slide-pixel offset. */
export function translateGeometry(g: Geometry, dx: number, dy: number): Geometry {
  const pt = (p: Position): Position => [p[0] + dx, p[1] + dy];
  switch (g.type) {
    case "Point":
      return { type: "Point", coordinates: pt(g.coordinates) };
    case "LineString":
      return { type: "LineString", coordinates: g.coordinates.map(pt) };
    case "Polygon":
      return { type: "Polygon", coordinates: g.coordinates.map((r) => r.map(pt)) };
    case "MultiPolygon":
      return {
        type: "MultiPolygon",
        coordinates: g.coordinates.map((poly) => poly.map((r) => r.map(pt))),
      };
  }
}

/** True when a geometry's bounding box overlaps the given rectangle. */
export function intersectsRect(
  g: Geometry,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
): boolean {
  const b = bboxOfGeometry(g);
  return !(b[2] < minX || b[0] > maxX || b[3] < minY || b[1] > maxY);
}

function bboxOfGeometry(g: Geometry): [number, number, number, number] {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const visit = (p: Position) => {
    if (p[0] < minX) minX = p[0];
    if (p[0] > maxX) maxX = p[0];
    if (p[1] < minY) minY = p[1];
    if (p[1] > maxY) maxY = p[1];
  };
  if (g.type === "Point") visit(g.coordinates);
  else if (g.type === "LineString") g.coordinates.forEach(visit);
  else if (g.type === "Polygon") g.coordinates.forEach((r) => r.forEach(visit));
  else g.coordinates.forEach((poly) => poly.forEach((r) => r.forEach(visit)));
  return [minX, minY, maxX, maxY];
}

/**
 * Rings that are safe to hand to a tessellator.
 *
 * The renderer cuts a bridge from each hole to its outer ring so the two can be
 * triangulated as one loop. That bridge is invisible when the hole really is
 * inside the outer, and is drawn as a long thin sliver right across the image
 * when it is not — the "chords" between tissue fragments. A polygon can arrive
 * malformed from more places than the detector: a document saved by an older
 * build, GeoJSON exported from another tool, a boolean op on degenerate input.
 *
 * So the guarantee is enforced here, at the last point before drawing, rather
 * than trusted from upstream: an outer ring with too few points draws nothing,
 * and a hole that is not inside its outer is dropped instead of bridged. A
 * dropped hole shows as filled tissue that should have had a gap — wrong in a
 * small, local, visible way, rather than a red band across the slide.
 */
export function renderableRings(rings: Ring[]): Ring[] | null {
  if (!rings.length) return null;
  const outer = rings[0];
  // Three distinct vertices plus the repeated closing point.
  if (outer.length < 4) return null;
  if (!outer.every((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]))) return null;

  const kept: Ring[] = [outer];
  for (let i = 1; i < rings.length; i++) {
    const hole = rings[i];
    if (hole.length < 4) continue;
    if (!hole.every((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]))) continue;
    // Test the hole's mean vertex rather than one of its corners: a corner can
    // sit exactly on the outer boundary, where inside-ness is a coin toss.
    let sx = 0;
    let sy = 0;
    for (const [x, y] of hole) { sx += x; sy += y; }
    if (pointInPolygonRing(outer, sx / hole.length, sy / hole.length)) kept.push(hole);
  }
  return kept;
}

function pointInPolygonRing(ring: Ring, x: number, y: number): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
