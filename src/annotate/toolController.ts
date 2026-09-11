import OpenSeadragon from "openseadragon";
import type { AnnotationOverlay } from "../viewer/annotationOverlay";
import {
  booleanOp,
  circleRing,
  closeRing,
  intersectsRect,
  isDegenerate,
  rectRing,
  strokePolygon,
  translateGeometry,
} from "./geometry";
import {
  boxOf,
  fitGeometry,
  handleAt,
  resizeBox,
  resizeTarget,
  type Box,
  type HandleId,
} from "./resize";
import { makeAnnotation, queryBox, useAnnotations, withGeometry, type Patch } from "./store";
import {
  areaOf,
  containsPoint,
  isAreaGeometry,
  ROI_CLASS_ID,
  ROI_COLOR,
  simplifyGeometry,
  type Annotation,
  type AreaGeometry,
  type Geometry,
  type Position,
} from "./types";

/**
 * Owns all pointer interaction for drawing and editing.
 *
 * It listens on its own transparent element stacked above the viewer rather
 * than fighting OpenSeadragon's mouse tracker. When a drawing tool is active
 * that element takes pointer events and OSD sees none of them; wheel zoom is
 * forwarded explicitly so zooming keeps working mid-draw.
 */
export class ToolController {
  private drawing = false;
  private path: Position[] = [];
  private polygonPoints: Position[] = [];
  private spacePanning = false;
  private unsubscribeTool: (() => void) | null = null;
  private pointerId: number | null = null;
  private menuHost!: HTMLElement;
  /** In-progress move: originals to diff against when the gesture ends. */
  private moving: { origin: Position; before: Annotation[] } | null = null;
  /** In-progress corner/edge resize of an ROI. */
  private resizing: { before: Annotation; from: Box; handle: HandleId } | null = null;
  /** In-progress rubber-band selection. */
  private banding: {
    origin: Position;
    additive: boolean;
    /** Selected instead if the gesture never becomes a drag. */
    clickTarget: Annotation | null;
  } | null = null;

  constructor(
    private readonly viewer: OpenSeadragon.Viewer,
    private readonly element: HTMLElement,
    private readonly overlay: AnnotationOverlay,
    private readonly offset: { x: number; y: number },
    /** Owns the prompt-segmentation model; absent means the tool is inert. */
    private readonly segmenter?: {
      handleClick: (p: Position, mods: { shift: boolean; alt: boolean }) => Promise<void>;
      commit: () => boolean;
      discard: () => void;
      hasPending: () => boolean;
    },
    /** Called with the ROI the patch tool just drew, to lay its grid. */
    private readonly onPatch?: (roiId: string) => void,
  ) {
    element.addEventListener("pointerdown", this.onPointerDown);
    element.addEventListener("pointermove", this.onPointerMove);
    element.addEventListener("pointerup", this.onPointerUp);
    element.addEventListener("pointercancel", this.onPointerUp);
    element.addEventListener("pointerleave", this.onPointerLeave);
    element.addEventListener("wheel", this.onWheel, { passive: false });
    element.addEventListener("dblclick", this.onDoubleClick);
    // The input layer is pointer-transparent while panning, so bind the
    // context menu to the stage: right-click must work with any tool active.
    this.menuHost = element.parentElement ?? element;
    this.menuHost.addEventListener("contextmenu", this.onContextMenu);
    window.addEventListener("keydown", this.onKeyDown);
    window.addEventListener("keyup", this.onKeyUp);

    /**
     * Switching tools abandons whatever was half-drawn.
     *
     * A polygon two vertices in, or a brush cursor, belongs to the tool that
     * was active. Leaving it on screen after the tool changes is not just
     * untidy — the outline invites you to keep clicking it, and the clicks now
     * go somewhere else entirely.
     */
    this.unsubscribeTool = useAnnotations.subscribe((state, prev) => {
      if (state.tool === prev.tool) return;
      this.cancelDraft();
      this.syncPointerEvents();
    });
  }

  destroy() {
    this.unsubscribeTool?.();
    const e = this.element;
    e.removeEventListener("pointerdown", this.onPointerDown);
    e.removeEventListener("pointermove", this.onPointerMove);
    e.removeEventListener("pointerup", this.onPointerUp);
    e.removeEventListener("pointercancel", this.onPointerUp);
    e.removeEventListener("pointerleave", this.onPointerLeave);
    e.removeEventListener("wheel", this.onWheel);
    e.removeEventListener("dblclick", this.onDoubleClick);
    this.menuHost.removeEventListener("contextmenu", this.onContextMenu);
    window.removeEventListener("keydown", this.onKeyDown);
    window.removeEventListener("keyup", this.onKeyUp);
  }

  // ------------------------------------------------------------ geometry ---

  /** Client pixel -> level-0 slide pixel. */
  private toSlide(clientX: number, clientY: number): Position {
    const rect = this.element.getBoundingClientRect();
    const pixel = new OpenSeadragon.Point(clientX - rect.left, clientY - rect.top);
    const vp = this.viewer.viewport.pointFromPixel(pixel);
    const img = this.viewer.viewport.viewportToImageCoordinates(vp);
    return [img.x + this.offset.x, img.y + this.offset.y];
  }

  /** Screen pixels per slide pixel, for zoom-aware tolerances. */
  private scale(): number {
    const vp = this.viewer.viewport;
    const rect = vp.viewportToImageRectangle(vp.getBounds(true));
    return vp.getContainerSize().x / Math.max(rect.width, 1e-9);
  }

  private activeColour(): [number, number, number] {
    const { tool, classes, activeClassId } = useAnnotations.getState();
    // The patch tool draws an ROI, so its draft is the ROI's colour.
    if (tool === "roi" || tool === "patch") return ROI_COLOR;
    return classes.find((c) => c.id === activeClassId)?.color ?? [200, 200, 200];
  }

  // -------------------------------------------------------------- events ---

  private onWheel = (ev: WheelEvent) => {
    ev.preventDefault();
    const rect = this.element.getBoundingClientRect();
    const pixel = new OpenSeadragon.Point(ev.clientX - rect.left, ev.clientY - rect.top);
    const refPoint = this.viewer.viewport.pointFromPixel(pixel);
    // Match OSD's default feel rather than inventing a new one.
    const factor = Math.pow(1.0015, -ev.deltaY);
    this.viewer.viewport.zoomBy(factor, refPoint);
    this.viewer.viewport.applyConstraints();
  };

  private onPointerDown = (ev: PointerEvent) => {
    // Middle-drag pans, whatever tool is active: the overlay steps out of the
    // way and lets OpenSeadragon have the gesture, then takes it back on
    // release. Listening on the window for that release matters — once pointer
    // events are off, this element never sees the pointerup itself.
    if (ev.button === 1) {
      ev.preventDefault();
      this.spacePanning = true;
      this.element.style.pointerEvents = "none";
      const done = () => {
        this.spacePanning = false;
        this.syncPointerEvents();
        window.removeEventListener("pointerup", done);
        window.removeEventListener("pointercancel", done);
      };
      window.addEventListener("pointerup", done);
      window.addEventListener("pointercancel", done);
      return;
    }
    if (ev.button !== 0) return;
    const { tool } = useAnnotations.getState();
    if (tool === "pan" || this.spacePanning) return;

    // Capture keeps a drag alive outside the element; synthetic pointers in
    // tests have no real capture target, so a failure here must not stop drawing.
    try { this.element.setPointerCapture(ev.pointerId); this.pointerId = ev.pointerId; }
    catch { this.pointerId = null; }
    const p = this.toSlide(ev.clientX, ev.clientY);

    if (tool === "polygon") {
      this.polygonPoints.push(p);
      this.updatePolygonDraft(p);
      return;
    }

    if (tool === "point") {
      this.commitPoint(p);
      return;
    }

    if (tool === "segment") {
      void this.segmenter?.handleClick(p, { shift: ev.shiftKey, alt: ev.altKey });
      return;
    }

    if (tool === "select") {
      const state = useAnnotations.getState();

      // Handles take precedence over the object under the cursor: they sit on
      // the ROI's own boundary, so a hit test would always claim the press
      // first and the corner would never be grabbable.
      const target = resizeTarget(state.items, state.selection);
      if (target) {
        const from = boxOf(target);
        const handle = handleAt(from, p, 1 / this.scale());
        if (handle) {
          this.resizing = { before: target, from, handle: handle.id };
          return;
        }
      }

      const hit = this.hitTest(p);

      if (hit && !hit.locked) {
        // Clicking an unselected object selects it; clicking within the
        // existing selection keeps it, so a multi-object drag is possible.
        if (!state.selection.has(hit.id)) state.select([hit.id], ev.shiftKey);
        const ids = [...useAnnotations.getState().selection];
        this.moving = {
          origin: p,
          before: ids
            .map((id) => state.items.get(id))
            .filter((a): a is Annotation => !!a && !a.locked),
        };
        return;
      }

      // Pressing on a locked object — or on nothing — starts a rubber band.
      // A locked region is usually a large container (tissue, a study area),
      // and being unable to box-select the objects inside it would make it a
      // dead zone. If the gesture turns out to be a click rather than a drag,
      // `bandClick` selects the locked object instead.
      this.banding = { origin: p, additive: ev.shiftKey, clickTarget: hit ?? null };
      this.drawing = true;
      this.path = [p];
      return;
    }

    this.drawing = true;
    this.path = [p];
    this.renderDraft(p);
  };

  private onPointerMove = (ev: PointerEvent) => {
    const { tool, brushRadius } = useAnnotations.getState();
    const p = this.toSlide(ev.clientX, ev.clientY);

    // A handle under the cursor announces itself before the press, which is
    // the only cue that the ROI can be resized at all.
    if (!this.drawing && !this.moving && !this.resizing && tool === "select") {
      const state = useAnnotations.getState();
      const target = resizeTarget(state.items, state.selection);
      const handle = target ? handleAt(boxOf(target), p, 1 / this.scale()) : null;
      this.element.style.cursor = handle ? handle.cursor : "";
    }

    // Brush and eraser show a live cursor even when not pressed.
    if (!this.drawing && (tool === "brush" || tool === "eraser")) {
      this.overlay.setDraft({
        kind: "path",
        color: this.activeColour(),
        cursorAt: p,
        cursorRadius: brushRadius,
        erasing: tool === "eraser",
      });
      return;
    }

    if (this.resizing) {
      const { before, from, handle } = this.resizing;
      // One screen pixel is the floor, so the box always stays grabbable.
      const to = resizeBox(from, handle, p, 1 / this.scale());
      useAnnotations.getState().previewGeometries([
        { id: before.id, geometry: fitGeometry(before.geometry, from, to) },
      ]);
      return;
    }

    if (this.moving) {
      const dx = p[0] - this.moving.origin[0];
      const dy = p[1] - this.moving.origin[1];
      useAnnotations.getState().previewGeometries(
        this.moving.before.map((a) => ({
          id: a.id,
          geometry: translateGeometry(a.geometry, dx, dy),
        })),
      );
      return;
    }

    if (this.banding) {
      const [x0, y0] = this.banding.origin;
      this.overlay.setDraft({
        kind: "polygon",
        rings: [rectRing(x0, y0, p[0], p[1])],
        color: [140, 200, 255],
      });
      return;
    }

    if (tool === "polygon" && this.polygonPoints.length > 0) {
      this.updatePolygonDraft(p);
      return;
    }

    if (!this.drawing) return;
    this.path.push(p);
    this.renderDraft(p);
  };

  private onPointerUp = (ev: PointerEvent) => {
    if (this.pointerId !== null) {
      try { this.element.releasePointerCapture(this.pointerId); } catch { /* already released */ }
      this.pointerId = null;
    }
    const p = this.toSlide(ev.clientX, ev.clientY);

    if (this.resizing) {
      const { before, from, handle } = this.resizing;
      this.resizing = null;
      const to = resizeBox(from, handle, p, 1 / this.scale());
      const state = useAnnotations.getState();
      const moved =
        Math.abs(to.minX - from.minX) + Math.abs(to.maxX - from.maxX) +
        Math.abs(to.minY - from.minY) + Math.abs(to.maxY - from.maxY);
      // A press that never became a drag leaves the ROI exactly as it was.
      if (moved < 0.5) {
        state.previewGeometries([{ id: before.id, geometry: before.geometry }]);
        return;
      }
      const after = withGeometry(before, fitGeometry(before.geometry, from, to));
      state.previewGeometries([{ id: before.id, geometry: before.geometry }]);
      state.apply({ label: "Resize ROI", updated: [{ before, after }] });
      return;
    }

    if (this.moving) {
      const move = this.moving;
      this.moving = null;
      const dx = p[0] - move.origin[0];
      const dy = p[1] - move.origin[1];
      // Sub-pixel drags are clicks, not moves; do not litter the undo stack.
      if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5) {
        useAnnotations.getState().previewGeometries(
          move.before.map((a) => ({ id: a.id, geometry: a.geometry })),
        );
        return;
      }
      const state = useAnnotations.getState();
      state.apply({
        label: `Move ${move.before.length}`,
        updated: move.before.map((a) => {
          const current = state.items.get(a.id);
          return { before: a, after: current ?? a };
        }),
      });
      return;
    }

    if (this.banding) {
      const band = this.banding;
      this.banding = null;
      this.drawing = false;
      this.path = [];
      this.overlay.setDraft(null);
      this.resolveBand(band.origin, p, band.additive, band.clickTarget);
      return;
    }

    if (!this.drawing) return;
    this.drawing = false;
    this.path.push(p);
    this.commitDrag();
  };

  private onPointerLeave = () => {
    const { tool } = useAnnotations.getState();
    if (!this.drawing && (tool === "brush" || tool === "eraser")) this.overlay.setDraft(null);
  };

  private onDoubleClick = (ev: MouseEvent) => {
    if (useAnnotations.getState().tool !== "polygon") return;
    ev.preventDefault();
    this.closePolygon();
  };

  private onKeyDown = (ev: KeyboardEvent) => {
    const target = ev.target as HTMLElement | null;
    if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
    const store = useAnnotations.getState();

    /**
     * Space shows and hides the annotations.
     *
     * Comparing a boundary against the tissue under it is the single most
     * repeated action in this app, and it wants a key you can hold and release
     * without looking. Temporary panning, which used to live here, moved to the
     * middle mouse button — it is the same gesture every map viewer uses, and
     * it leaves the keyboard free for the thing done far more often.
     */
    if (ev.code === "Space") {
      ev.preventDefault();
      if (!ev.repeat) store.toggleAnnotations();
      return;
    }
    if (ev.key === "Escape" && this.segmenter?.hasPending()) {
      this.segmenter.discard();
      return;
    }
    if (ev.key === "Enter" && store.tool === "segment" && this.segmenter?.hasPending()) {
      ev.preventDefault();
      this.segmenter.commit();
      return;
    }
    // Backspace is the "that mask is wrong" key while segmenting: throw away
    // the one on screen, or — since a fresh click auto-keeps the previous cell
    // — take back the one that was just kept.
    if ((ev.key === "Backspace" || ev.key === "Delete") && store.tool === "segment") {
      if (this.segmenter?.hasPending()) {
        ev.preventDefault();
        this.segmenter.discard();
        return;
      }
      if (store.undoStack.at(-1)?.label === "Segment") {
        ev.preventDefault();
        store.undo();
        return;
      }
    }
    if (ev.key === "Escape") {
      this.cancelDraft();
      store.openContextMenu(null);
      store.openClassPicker(null);
      store.clearSelection();
      return;
    }
    // Name / classify whatever is selected.
    if ((ev.key === "c" || ev.key === "C" || ev.key === "F2") && store.selection.size > 0) {
      ev.preventDefault();
      const ids = [...store.selection];
      store.openClassPicker({ ...this.anchorFor(ids), targetIds: ids });
      return;
    }
    if (ev.key === "Enter" && store.tool === "polygon") {
      this.closePolygon();
      return;
    }
    if ((ev.key === "Backspace" || ev.key === "Delete") && store.tool === "polygon" && this.polygonPoints.length) {
      ev.preventDefault();
      this.polygonPoints.pop();
      this.updatePolygonDraft(null);
      return;
    }
    if (ev.key === "Delete" || ev.key === "Backspace") {
      if (store.selection.size === 0) return;
      ev.preventDefault();
      this.deleteSelection();
      return;
    }
    if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "z") {
      ev.preventDefault();
      if (ev.shiftKey) store.redo();
      else store.undo();
    }
  };

  private onKeyUp = (ev: KeyboardEvent) => {
    if (ev.code === "Space") ev.preventDefault();
  };

  /** Called by React when the active tool changes. */
  syncPointerEvents() {
    const { tool } = useAnnotations.getState();
    this.element.style.pointerEvents = tool === "pan" ? "none" : "auto";
    this.element.style.cursor =
      tool === "select" ? "default" : tool === "pan" ? "grab" : "crosshair";
  }

  // --------------------------------------------------------------- draft ---

  private renderDraft(cursor: Position) {
    const { tool, brushRadius } = useAnnotations.getState();
    const colour = this.activeColour();

    if (tool === "rectangle" || tool === "roi" || tool === "patch") {
      const [x0, y0] = this.path[0];
      const [x1, y1] = cursor;
      this.overlay.setDraft({ kind: "polygon", rings: [rectRing(x0, y0, x1, y1)], color: colour });
      return;
    }
    if (tool === "freehand") {
      this.overlay.setDraft({ kind: "path", path: this.path, color: colour });
      return;
    }
    if (tool === "brush" || tool === "eraser") {
      this.overlay.setDraft({
        kind: "stroke",
        path: this.path,
        color: colour,
        cursorAt: cursor,
        cursorRadius: brushRadius,
        erasing: tool === "eraser",
      });
    }
  }

  private updatePolygonDraft(cursor: Position | null) {
    const pts = cursor ? [...this.polygonPoints, cursor] : this.polygonPoints;
    if (pts.length < 2) {
      this.overlay.setDraft({ kind: "path", path: pts, color: this.activeColour() });
      return;
    }
    this.overlay.setDraft({
      kind: "polygon",
      rings: [closeRing(pts as Position[])],
      color: this.activeColour(),
    });
  }

  private cancelDraft() {
    if (this.resizing) {
      useAnnotations.getState().previewGeometries([
        { id: this.resizing.before.id, geometry: this.resizing.before.geometry },
      ]);
      this.resizing = null;
    }
    if (this.moving) {
      useAnnotations.getState().previewGeometries(
        this.moving.before.map((a) => ({ id: a.id, geometry: a.geometry })),
      );
      this.moving = null;
    }
    this.banding = null;
    this.drawing = false;
    this.path = [];
    this.polygonPoints = [];
    this.overlay.setDraft(null);
  }

  // -------------------------------------------------------------- commit ---

  private commitPoint(p: Position) {
    if (!Number.isFinite(p[0]) || !Number.isFinite(p[1])) return;
    const state = useAnnotations.getState();
    const a = makeAnnotation({ type: "Point", coordinates: p }, { classId: state.activeClassId });
    state.apply({ label: "Add point", added: [a] });
    state.select([a.id]);
  }

  private closePolygon() {
    if (this.polygonPoints.length < 3) {
      this.cancelDraft();
      return;
    }
    const ring = closeRing(this.polygonPoints as Position[]);
    this.polygonPoints = [];
    this.overlay.setDraft(null);
    this.commitArea({ type: "Polygon", coordinates: [ring] });
  }

  private commitDrag() {
    const { tool, brushRadius, simplifyTolerance } = useAnnotations.getState();
    const path = this.path;
    this.path = [];
    this.overlay.setDraft(null);
    if (path.length === 0) return;

    if (tool === "rectangle" || tool === "roi" || tool === "patch") {
      const [x0, y0] = path[0];
      const [x1, y1] = path[path.length - 1];
      if (Math.abs(x1 - x0) < 2 || Math.abs(y1 - y0) < 2) return;
      const isRoi = tool === "roi" || tool === "patch";
      const made = this.commitArea(
        { type: "Polygon", coordinates: [rectRing(x0, y0, x1, y1)] }, isRoi,
      );
      /**
       * Patching is drawn, not configured.
       *
       * The region is still an ROI — it is the frame the grid is laid in, and
       * everything downstream asks "which ROI am I working in" — but the grid
       * arrives with the drag instead of waiting for a button in a panel. The
       * grid itself needs the slide's scale and levels, which this controller
       * has no business knowing, so it is handed back out.
       */
      if (tool === "patch" && made) this.onPatch?.(made.id);
      return;
    }

    if (tool === "freehand") {
      if (path.length < 3) return;
      // Tolerance in slide pixels: keep ~1.5 screen px of fidelity at this zoom.
      const tol = simplifyTolerance / this.scale();
      const g = simplifyGeometry(
        { type: "Polygon", coordinates: [closeRing(path as Position[])] },
        tol,
      );
      this.commitArea(g as AreaGeometry);
      return;
    }

    if (tool === "brush" || tool === "eraser") {
      const stroke = strokePolygon(path, brushRadius);
      if (!stroke) return;
      if (tool === "brush") this.commitBrush(stroke);
      else this.commitErase(stroke);
    }
  }

  private commitArea(geometry: Geometry, isRoi = false) {
    if (isDegenerate(geometry)) return null;
    const state = useAnnotations.getState();
    const a = makeAnnotation(geometry, {
      classId: isRoi ? ROI_CLASS_ID : state.activeClassId,
    });
    state.apply({ label: isRoi ? "Add ROI" : "Add region", added: [a] });
    // Select it so "C" names the thing just drawn, with no extra click.
    state.select([a.id]);
    return a;
  }

  /**
   * Brushing merges into whatever it touches of the same class, so repeated
   * strokes grow one object instead of leaving a pile of overlapping shards.
   */
  private commitBrush(stroke: AreaGeometry) {
    const state = useAnnotations.getState();
    const [minX, minY, maxX, maxY] = bboxOfArea(stroke);
    const touching = queryBox(minX, minY, maxX, maxY).filter(
      (a) =>
        !a.locked &&
        a.classId === state.activeClassId &&
        isAreaGeometry(a.geometry) &&
        intersects(a.geometry, stroke),
    );

    if (touching.length === 0) {
      state.apply({
        label: "Brush",
        added: [makeAnnotation(stroke, { classId: state.activeClassId })],
      });
      return;
    }

    let merged: AreaGeometry = stroke;
    for (const a of touching) {
      const next = booleanOp("union", merged, a.geometry as AreaGeometry);
      if (next) merged = next;
    }

    const [keep, ...rest] = touching;
    state.apply({
      label: "Brush",
      updated: [{ before: keep, after: withGeometry(keep, merged) }],
      removed: rest,
    });
  }

  private commitErase(stroke: AreaGeometry) {
    const state = useAnnotations.getState();
    const [minX, minY, maxX, maxY] = bboxOfArea(stroke);
    const touching = queryBox(minX, minY, maxX, maxY).filter(
      (a) => !a.locked && isAreaGeometry(a.geometry) && intersects(a.geometry, stroke),
    );
    if (touching.length === 0) return;

    const updated: { before: Annotation; after: Annotation }[] = [];
    const removed: Annotation[] = [];
    for (const a of touching) {
      const next = booleanOp("subtract", a.geometry as AreaGeometry, stroke);
      if (!next || isDegenerate(next)) removed.push(a);
      else if (JSON.stringify(next) !== JSON.stringify(a.geometry)) {
        updated.push({ before: a, after: withGeometry(a, next) });
      }
    }
    if (updated.length === 0 && removed.length === 0) return;
    state.apply({ label: "Erase", updated, removed });
  }

  private deleteSelection() {
    const state = useAnnotations.getState();
    const selected = [...state.selection]
      .map((id) => state.items.get(id))
      .filter((a): a is Annotation => !!a);
    const removed = selected.filter((a) => !a.locked);

    if (removed.length === 0) {
      // Silence here reads as "delete is broken"; say what actually happened.
      if (selected.length > 0) {
        state.setNotice(
          selected.length === 1
            ? "That object is locked — unlock it to delete."
            : `All ${selected.length} selected objects are locked.`,
        );
      }
      return;
    }
    state.setNotice(null);
    state.apply({ label: `Delete ${removed.length}`, removed });
    state.clearSelection();
  }

  // -------------------------------------------------------------- select ---

  /**
   * Topmost object under a slide point, or null.
   * Points win over regions, then the smallest containing region, so nested
   * objects and small details stay reachable.
   */
  private hitTest(p: Position): Annotation | null {
    const tolerance = 6 / this.scale();
    const hits = queryBox(p[0] - tolerance, p[1] - tolerance, p[0] + tolerance, p[1] + tolerance);

    const pointHit = hits.find(
      (a) =>
        a.geometry.type === "Point" &&
        Math.hypot(a.geometry.coordinates[0] - p[0], a.geometry.coordinates[1] - p[1]) <= tolerance,
    );
    if (pointHit) return pointHit;

    return (
      hits
        .filter((a) => isAreaGeometry(a.geometry) && containsPoint(a.geometry, p[0], p[1]))
        .sort((a, b) => areaOf(a.geometry) - areaOf(b.geometry))[0] ?? null
    );
  }

  /** Select everything the rubber band touched. */
  private resolveBand(
    a: Position,
    b: Position,
    additive: boolean,
    clickTarget: Annotation | null,
  ) {
    const minX = Math.min(a[0], b[0]);
    const maxX = Math.max(a[0], b[0]);
    const minY = Math.min(a[1], b[1]);
    const maxY = Math.max(a[1], b[1]);
    const state = useAnnotations.getState();

    // Under a couple of screen pixels this was a click, not a drag.
    if (maxX - minX < 2 / this.scale() && maxY - minY < 2 / this.scale()) {
      if (clickTarget) state.select([clickTarget.id], additive);
      else if (!additive) state.clearSelection();
      return;
    }

    // Locked objects are containers here, not targets: a band drawn inside a
    // locked region should pick up what is in it, not the region itself.
    const hits = queryBox(minX, minY, maxX, maxY)
      .filter((x) => !x.locked && intersectsRect(x.geometry, minX, minY, maxX, maxY))
      .map((x) => x.id);
    state.setSelection(additive ? [...new Set([...state.selection, ...hits])] : hits);
  }

  private onContextMenu = (ev: MouseEvent) => {
    // Only the image surface opens the annotation menu. The tool rail and the
    // popovers live inside the same stage element, and React's delegated
    // stopPropagation cannot prevent this native listener from firing first —
    // so the check has to be positive, not a list of exclusions.
    const target = ev.target as HTMLElement | null;
    if (!target?.closest(".osd-host, .anno-input, .anno-canvas")) return;
    ev.preventDefault();

    const state = useAnnotations.getState();
    const hit = this.hitTest(this.toSlide(ev.clientX, ev.clientY));

    let targetIds: string[] = [];
    if (hit) {
      // Right-clicking outside the selection retargets it, the way file
      // managers and editors behave.
      targetIds = state.selection.has(hit.id) ? [...state.selection] : [hit.id];
      if (!state.selection.has(hit.id)) state.select([hit.id]);
    } else {
      state.clearSelection();
    }
    state.openContextMenu({ x: ev.clientX, y: ev.clientY, targetIds });
  };

  /** Screen position to anchor a menu on, given the objects it acts upon. */
  private anchorFor(ids: string[]): { x: number; y: number } {
    const state = useAnnotations.getState();
    const rect = this.element.getBoundingClientRect();
    const first = ids.map((id) => state.items.get(id)).find((a) => !!a);
    if (!first) return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
    const [minX, minY, maxX, maxY] = first.bbox;
    const vp = this.viewer.viewport;
    const img = new OpenSeadragon.Point(
      (minX + maxX) / 2 - this.offset.x,
      (minY + maxY) / 2 - this.offset.y,
    );
    const px = vp.pixelFromPoint(vp.imageToViewportCoordinates(img), true);
    return { x: rect.left + px.x, y: rect.top + px.y };
  }
}

// ------------------------------------------------------------------ utils ---

function bboxOfArea(g: AreaGeometry): [number, number, number, number] {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const rings = g.type === "Polygon" ? g.coordinates : g.coordinates.flat();
  for (const ring of rings) {
    for (const [x, y] of ring) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  return [minX, minY, maxX, maxY];
}

/** Cheap bbox rejection, used only to prefilter before a real boolean test. */
function bboxOverlaps(a: AreaGeometry, b: AreaGeometry): boolean {
  const ba = bboxOfArea(a);
  const bb = bboxOfArea(b);
  return !(ba[2] < bb[0] || ba[0] > bb[2] || ba[3] < bb[1] || ba[1] > bb[3]);
}

/**
 * True geometric intersection. Overlapping bounding boxes are not enough: two
 * same-class regions can sit inside each other's bbox without touching, and
 * merging those would silently glue unrelated annotations into one object.
 */
function intersects(a: Geometry, b: AreaGeometry): boolean {
  if (!isAreaGeometry(a) || !bboxOverlaps(a, b)) return false;
  return booleanOp("intersect", a, b) !== null;
}

export function circleGeometry(cx: number, cy: number, r: number): AreaGeometry {
  return { type: "Polygon", coordinates: [circleRing(cx, cy, r)] };
}

export type { Patch };
