import { simplifyRing, type Ring } from "../annotate/types";

/**
 * Marching squares: binary mask -> closed contour rings.
 *
 * Model output is a raster; annotations are vectors. Marching squares is used
 * rather than boundary tracing because it handles holes and multiple
 * components without extra bookkeeping, and its half-pixel vertices sit on the
 * true edge between inside and outside rather than on pixel centres.
 */

/** Segment endpoints are on half-pixel boundaries, so work in doubled ints. */
const key = (x: number, y: number, stride: number) => y * stride + x;

export interface TraceOptions {
  /** Values strictly above this count as inside. */
  threshold?: number;
  /** Douglas-Peucker tolerance in mask pixels. */
  simplify?: number;
  /** Discard rings enclosing fewer than this many pixels. */
  minArea?: number;
}

export function traceMask(
  mask: Float32Array | Uint8Array,
  width: number,
  height: number,
  opts: TraceOptions = {},
): Ring[] {
  const { threshold = 0, simplify = 1, minArea = 12 } = opts;
  const inside = (x: number, y: number) =>
    x >= 0 && y >= 0 && x < width && y < height && mask[y * width + x] > threshold;

  // Doubled-integer lattice: a vertex at (x+0.5, y) becomes (2x+1, 2y).
  const stride = 2 * width + 3;
  /**
   * Outgoing edges per lattice point. This must be a LIST, not a single value:
   * two contours can pass through the same point (saddle cells, and shapes that
   * touch corner-to-corner), and storing one edge silently discards the other —
   * leaving one contour to dead-end and get closed with a straight chord across
   * the image.
   */
  const next = new Map<number, number[]>();
  let edgeCount = 0;

  const add = (ax: number, ay: number, bx: number, by: number) => {
    const from = key(ax, ay, stride);
    const to = key(bx, by, stride);
    const list = next.get(from);
    if (list) list.push(to);
    else next.set(from, [to]);
    edgeCount++;
  };

  for (let y = -1; y < height; y++) {
    for (let x = -1; x < width; x++) {
      const tl = inside(x, y) ? 8 : 0;
      const tr = inside(x + 1, y) ? 4 : 0;
      const br = inside(x + 1, y + 1) ? 2 : 0;
      const bl = inside(x, y + 1) ? 1 : 0;
      const code = tl | tr | br | bl;
      if (code === 0 || code === 15) continue;

      // Edge midpoints in doubled coordinates, offset by +1 so x = -1 is valid.
      const X = 2 * (x + 1);
      const Y = 2 * (y + 1);
      const T: [number, number] = [X + 1, Y];
      const R: [number, number] = [X + 2, Y + 1];
      const B: [number, number] = [X + 1, Y + 2];
      const L: [number, number] = [X, Y + 1];

      // Every segment is oriented with the inside on its LEFT. Consistency is
      // the whole game: with mixed winding, the walk below links segments from
      // unrelated contours and returns one long spurious ring.
      switch (code) {
        case 1:  add(...B, ...L); break;
        case 2:  add(...R, ...B); break;
        case 3:  add(...R, ...L); break;
        case 4:  add(...T, ...R); break;
        case 5:  add(...T, ...R); add(...B, ...L); break; // saddle
        case 6:  add(...T, ...B); break;
        case 7:  add(...T, ...L); break;
        case 8:  add(...L, ...T); break;
        case 9:  add(...B, ...T); break;
        case 10: add(...L, ...T); add(...R, ...B); break; // saddle
        case 11: add(...R, ...T); break;
        case 12: add(...L, ...R); break;
        case 13: add(...B, ...R); break;
        case 14: add(...L, ...B); break;
      }
    }
  }

  const rings: Ring[] = [];
  const pointAt = (k: number): [number, number] => {
    const kx = k % stride;
    const ky = (k - kx) / stride;
    // Undo the doubling and the +1 offset used above.
    return [kx / 2 - 1, ky / 2 - 1];
  };

  for (const start of [...next.keys()]) {
    // A point may begin several contours; drain all of them.
    while ((next.get(start)?.length ?? 0) > 0) {
      const ring: Ring = [];
      let cur = start;
      let closed = false;

      for (let guard = 0; guard <= edgeCount + 1; guard++) {
        const outgoing = next.get(cur);
        if (!outgoing || outgoing.length === 0) break;
        const nxt = outgoing.pop()!;
        ring.push(pointAt(cur));
        if (nxt === start) {
          closed = true;
          break;
        }
        cur = nxt;
      }

      // An unclosed walk means the field was malformed. Drop it rather than
      // joining the loose ends, which would draw a chord through the image.
      if (!closed || ring.length < 3) continue;
      ring.push(ring[0]);
      if (Math.abs(ringArea(ring)) < minArea) continue;
      rings.push(simplify > 0 ? simplifyRing(ring, simplify) : ring);
    }
  }

  return rings;
}

function ringArea(ring: Ring): number {
  let a = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    a += (ring[j][0] - ring[i][0]) * (ring[j][1] + ring[i][1]);
  }
  return a / 2;
}

/** Keep only the ring enclosing the most area — SAM returns one object. */
export function largestRing(rings: Ring[]): Ring | null {
  if (rings.length === 0) return null;
  return rings.reduce((best, r) =>
    Math.abs(ringArea(r)) > Math.abs(ringArea(best)) ? r : best,
  );
}

export { ringArea };
