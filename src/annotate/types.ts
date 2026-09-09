/**
 * Annotation model.
 *
 * Geometry is stored as GeoJSON in **level-0 slide pixels of the full vendor
 * frame** — including the MIRAX bounds offset. That makes coordinates mean the
 * same thing here, in OpenSlide, in QuPath and in every downstream Python tool,
 * so export is a relabelling rather than a transform.
 */

export type Position = [number, number];
export type Ring = Position[];

export interface PolygonGeometry { type: "Polygon"; coordinates: Ring[] }
export interface MultiPolygonGeometry { type: "MultiPolygon"; coordinates: Ring[][] }
export interface PointGeometry { type: "Point"; coordinates: Position }
export interface LineStringGeometry { type: "LineString"; coordinates: Ring }

export type Geometry =
  | PolygonGeometry
  | MultiPolygonGeometry
  | PointGeometry
  | LineStringGeometry;

export type AreaGeometry = PolygonGeometry | MultiPolygonGeometry;

export const isAreaGeometry = (g: Geometry): g is AreaGeometry =>
  g.type === "Polygon" || g.type === "MultiPolygon";

/** QuPath's object taxonomy, which we adopt wholesale for interop. */
export type ObjectType = "annotation" | "detection";

export type Provenance = "human" | "model";

export interface Annotation {
  id: string;
  geometry: Geometry;
  /** Null means unclassified, which QuPath renders in grey. */
  classId: string | null;
  objectType: ObjectType;
  source: Provenance;
  /** Encoder + head that produced this, so predictions stay traceable. */
  modelId?: string;
  confidence?: number;
  /** The ROI this was predicted inside, for the Phase 3 loop. */
  roiId?: string | null;
  locked: boolean;
  name?: string;
  measurements?: Record<string, number>;
  createdAt: number;
  updatedAt: number;
  /** Cached [minX, minY, maxX, maxY]; kept in sync by the store. */
  bbox: [number, number, number, number];
}

export interface AnnotationClass {
  id: string;
  name: string;
  /** 0-255 RGB. Kept as a triple so deck.gl and QuPath conversion are both trivial. */
  color: [number, number, number];
}

/** Regions of interest are ordinary annotations flagged for the ML loop. */
export const ROI_CLASS_ID = "__roi__";

/**
 * Colours handed out to new classes, in order, skipping any already in use.
 * There are deliberately **no default classes** — a class scheme belongs to the
 * study, not to us, and pre-seeding one only invites people to annotate against
 * labels that do not match their protocol.
 */
export const CLASS_PALETTE: [number, number, number][] = [
  [220, 60, 80],   [70, 160, 220],  [250, 180, 60],  [150, 110, 200],
  [90, 200, 130],  [235, 120, 190], [120, 200, 235], [200, 160, 90],
  [140, 140, 150], [110, 190, 90],  [230, 140, 100], [160, 130, 230],
];

/** First palette colour not already taken, falling back to a hashed hue. */
export function nextClassColor(used: AnnotationClass[]): [number, number, number] {
  const taken = new Set(used.map((c) => c.color.join(",")));
  const free = CLASS_PALETTE.find((c) => !taken.has(c.join(",")));
  if (free) return free;
  const h = (used.length * 137.508) % 360; // golden-angle spacing
  return hslToRgb(h, 0.62, 0.58);
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const k = (n: number) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)];
}

export const ROI_COLOR: [number, number, number] = [76, 194, 196];

// ---------------------------------------------------------------- geometry ---

export function bboxOf(g: Geometry): [number, number, number, number] {
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

/** Signed area of a ring; positive is counter-clockwise in a y-down frame. */
function ringArea(ring: Ring): number {
  let a = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    a += (ring[j][0] - ring[i][0]) * (ring[j][1] + ring[i][1]);
  }
  return a / 2;
}

/** Area in slide pixels², holes subtracted. Zero for points and lines. */
export function areaOf(g: Geometry): number {
  if (g.type === "Polygon") {
    return g.coordinates.reduce(
      (sum, ring, i) => sum + (i === 0 ? Math.abs(ringArea(ring)) : -Math.abs(ringArea(ring))),
      0,
    );
  }
  if (g.type === "MultiPolygon") {
    return g.coordinates.reduce(
      (sum, poly) =>
        sum +
        poly.reduce(
          (s, ring, i) => s + (i === 0 ? Math.abs(ringArea(ring)) : -Math.abs(ringArea(ring))),
          0,
        ),
      0,
    );
  }
  return 0;
}

/** Even-odd/nonzero point-in-polygon across all rings, holes respected. */
function pointInRings(rings: Ring[], x: number, y: number): boolean {
  let inside = false;
  for (const ring of rings) {
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i];
      const [xj, yj] = ring[j];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
  }
  return inside;
}

export function containsPoint(g: Geometry, x: number, y: number): boolean {
  if (g.type === "Polygon") return pointInRings(g.coordinates, x, y);
  if (g.type === "MultiPolygon") return g.coordinates.some((p) => pointInRings(p, x, y));
  return false;
}

/** Ramer-Douglas-Peucker. Vertex counts explode with freehand and brush tools. */
export function simplifyRing(ring: Ring, tolerance: number): Ring {
  if (ring.length <= 3 || tolerance <= 0) return ring;

  const sqTol = tolerance * tolerance;
  const keep = new Uint8Array(ring.length);
  keep[0] = 1;
  keep[ring.length - 1] = 1;

  const stack: [number, number][] = [[0, ring.length - 1]];
  while (stack.length) {
    const [first, last] = stack.pop()!;
    let maxSq = 0;
    let index = -1;
    const [ax, ay] = ring[first];
    const [bx, by] = ring[last];
    const dx = bx - ax;
    const dy = by - ay;
    const lenSq = dx * dx + dy * dy;

    for (let i = first + 1; i < last; i++) {
      const [px, py] = ring[i];
      let t = lenSq > 0 ? ((px - ax) * dx + (py - ay) * dy) / lenSq : 0;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const cx = ax + t * dx - px;
      const cy = ay + t * dy - py;
      const sq = cx * cx + cy * cy;
      if (sq > maxSq) { maxSq = sq; index = i; }
    }

    if (maxSq > sqTol && index !== -1) {
      keep[index] = 1;
      stack.push([first, index], [index, last]);
    }
  }

  const out: Ring = [];
  for (let i = 0; i < ring.length; i++) if (keep[i]) out.push(ring[i]);
  return out.length >= 3 ? out : ring;
}

export function simplifyGeometry(g: Geometry, tolerance: number): Geometry {
  if (g.type === "Polygon") {
    return { type: "Polygon", coordinates: g.coordinates.map((r) => simplifyRing(r, tolerance)) };
  }
  if (g.type === "MultiPolygon") {
    return {
      type: "MultiPolygon",
      coordinates: g.coordinates.map((p) => p.map((r) => simplifyRing(r, tolerance))),
    };
  }
  return g;
}
