import { PATCH_MODEL_ID } from "../ml/patchExport";
import { bboxOf, ROI_CLASS_ID, type Annotation, type Geometry, type Position } from "./types";

/**
 * Resizing an ROI by its corners and edges.
 *
 * An ROI is a working frame — the area you embed, patch and predict over — and
 * getting it right is iterative: draw it roughly, look at the tissue, pull an
 * edge out. Redrawing it from scratch each time loses whatever is already
 * pinned to it, so the box has to be adjustable in place.
 *
 * The handles are limited to *frames* — ROIs, and the cells of a patch grid.
 * A frame is a window you chose, so moving its edge is editing your choice. A
 * traced nucleus or a tissue boundary is a measurement, and dragging its corner
 * would rescale it into a shape that never existed on the slide, quietly
 * turning the area recorded against it into fiction.
 *
 * A patch is a frame by the same logic: the grid puts it somewhere reasonable
 * and you decide whether that is where you wanted it.
 */

/** True for the objects that are a chosen window rather than a measurement. */
export function isFrame(a: Annotation): boolean {
  return a.classId === ROI_CLASS_ID || a.modelId === PATCH_MODEL_ID;
}

export type HandleId = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";

export interface Box {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export interface Handle {
  id: HandleId;
  x: number;
  y: number;
  cursor: string;
}

/** Screen-space radius of a handle's grab area, in CSS pixels. */
export const HANDLE_HIT_PX = 9;

/** The one annotation showing resize handles, if any. */
export function resizeTarget(
  items: Map<string, Annotation>,
  selection: Set<string>,
): Annotation | null {
  if (selection.size !== 1) return null;
  const [id] = selection;
  const a = items.get(id);
  if (!a || a.locked || !isFrame(a)) return null;
  if (a.geometry.type === "Point" || a.geometry.type === "LineString") return null;
  return a;
}

export function boxOf(a: Annotation): Box {
  const [minX, minY, maxX, maxY] = a.bbox ?? bboxOf(a.geometry);
  return { minX, minY, maxX, maxY };
}

export function handlesOf(b: Box): Handle[] {
  const midX = (b.minX + b.maxX) / 2;
  const midY = (b.minY + b.maxY) / 2;
  return [
    { id: "nw", x: b.minX, y: b.minY, cursor: "nwse-resize" },
    { id: "n", x: midX, y: b.minY, cursor: "ns-resize" },
    { id: "ne", x: b.maxX, y: b.minY, cursor: "nesw-resize" },
    { id: "e", x: b.maxX, y: midY, cursor: "ew-resize" },
    { id: "se", x: b.maxX, y: b.maxY, cursor: "nwse-resize" },
    { id: "s", x: midX, y: b.maxY, cursor: "ns-resize" },
    { id: "sw", x: b.minX, y: b.maxY, cursor: "nesw-resize" },
    { id: "w", x: b.minX, y: midY, cursor: "ew-resize" },
  ];
}

/** The handle under a point, given how many slide pixels one screen pixel is. */
export function handleAt(b: Box, p: Position, slidePxPerScreenPx: number): Handle | null {
  const reach = HANDLE_HIT_PX * slidePxPerScreenPx;
  let best: Handle | null = null;
  let bestD = Infinity;
  for (const h of handlesOf(b)) {
    const d = Math.hypot(h.x - p[0], h.y - p[1]);
    if (d <= reach && d < bestD) { best = h; bestD = d; }
  }
  return best;
}

/**
 * The box a drag produces. The opposite corner is the anchor and never moves,
 * so the gesture reads the way a selection rectangle does everywhere else.
 *
 * A minimum extent keeps a box from being collapsed to nothing — a zero-width
 * ROI cannot be grabbed again, so an over-eager drag would otherwise destroy it
 * with no way back short of undo.
 */
export function resizeBox(b: Box, handle: HandleId, p: Position, minExtent: number): Box {
  const next = { ...b };
  if (handle.includes("w")) next.minX = Math.min(p[0], b.maxX - minExtent);
  if (handle.includes("e")) next.maxX = Math.max(p[0], b.minX + minExtent);
  if (handle.includes("n")) next.minY = Math.min(p[1], b.maxY - minExtent);
  if (handle.includes("s")) next.maxY = Math.max(p[1], b.minY + minExtent);
  return next;
}

/**
 * Map a geometry from one box onto another.
 *
 * An affine fit rather than a redrawn rectangle, so a non-rectangular ROI — a
 * lasso around a fragment, say — keeps its shape while its extent changes.
 */
export function fitGeometry(g: Geometry, from: Box, to: Box): Geometry {
  const fw = from.maxX - from.minX || 1;
  const fh = from.maxY - from.minY || 1;
  const sx = (to.maxX - to.minX) / fw;
  const sy = (to.maxY - to.minY) / fh;
  const pt = (p: Position): Position => [
    to.minX + (p[0] - from.minX) * sx,
    to.minY + (p[1] - from.minY) * sy,
  ];
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
