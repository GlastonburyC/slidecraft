import { Deck, OrthographicView } from "@deck.gl/core";
import { PathLayer, PolygonLayer, ScatterplotLayer } from "@deck.gl/layers";
import type OpenSeadragon from "openseadragon";
import { renderableRings } from "../annotate/geometry";
import { handlesOf, boxOf, resizeTarget, type Handle } from "../annotate/resize";
import { useAnnotations } from "../annotate/store";
import { useMl } from "../ml/mlStore";
import { colourFor, currentField, robustRange } from "../ml/spatialResult";
import { scoreSignature } from "../ml/signatures";
import { useSpatial } from "../ml/spatialStore";
import type { Annotation, Position, Ring } from "../annotate/types";
import { ROI_CLASS_ID, ROI_COLOR } from "../annotate/types";

/** A shape being drawn but not yet committed. */
export interface Draft {
  kind: "polygon" | "path" | "point" | "stroke";
  rings?: Ring[];
  path?: Position[];
  point?: Position;
  color: [number, number, number];
  /** Brush/eraser radius in slide pixels; also the half-width of a stroke draft. */
  cursorRadius?: number;
  cursorAt?: Position;
  erasing?: boolean;
}

interface PolyEntry {
  id: string;
  rings: Ring[];
  fill: [number, number, number, number];
  line: [number, number, number, number];
  width: number;
}

/**
 * Renders annotations on a GPU canvas locked to the OpenSeadragon viewport.
 *
 * deck.gl draws; it never handles input (`controller: false`, and the canvas is
 * pointer-events:none). OSD keeps ownership of pan/zoom and of converting
 * pointer positions to slide coordinates, so there is exactly one source of
 * truth for the camera.
 */
export class AnnotationOverlay {
  private deck: Deck<OrthographicView>;
  private draft: Draft | null = null;
  private unsubscribe: () => void;
  private frame = 0;

  constructor(
    private readonly viewer: OpenSeadragon.Viewer,
    private readonly canvas: HTMLCanvasElement,
    /** Level-0 offset of the tile source frame (MIRAX bounds). */
    private readonly offset: { x: number; y: number },
  ) {
    // A single view (not an array) keeps deck.gl's viewState in the plain
    // form rather than the {viewId: state} map.
    this.deck = new Deck<OrthographicView>({
      canvas,
      views: new OrthographicView({ flipY: true }),
      controller: false,
      useDevicePixels: true,
      initialViewState: { target: [0, 0, 0], zoom: 0 },
      layers: [],
    });

    // `update-viewport` fires synchronously inside OpenSeadragon's own draw
    // cycle, so rendering here puts the annotations and the image in the SAME
    // frame. Going through requestAnimationFrame instead leaves the overlay one
    // frame behind the tiles, which reads as annotations sliding around while
    // you pan or zoom.
    const syncRedraw = () => this.render();
    this.viewer.addHandler("update-viewport", syncRedraw);
    this.viewer.addHandler("resize", syncRedraw);

    // Document edits are not tied to a viewport frame; coalescing is fine.
    const unsubStore = useAnnotations.subscribe(() => this.scheduleRender());
    // The patch grid lives in the ML store, and it is drawn here.
    const unsubMl = useMl.subscribe((s, prev) => {
      if (s.grid !== prev.grid) this.scheduleRender();
    });
    const unsubSpatial = useSpatial.subscribe((s, prev) => {
      if (
        s.result !== prev.result ||
        s.gene !== prev.gene ||
        s.mode !== prev.mode ||
        s.signatureName !== prev.signatureName ||
        s.signatures !== prev.signatures ||
        s.opacity !== prev.opacity ||
        s.visible !== prev.visible
      ) {
        this.scheduleRender();
      }
    });
    this.unsubscribe = () => {
      this.viewer.removeHandler("update-viewport", syncRedraw);
      this.viewer.removeHandler("resize", syncRedraw);
      unsubStore();
      unsubMl();
      unsubSpatial();
    };

    this.scheduleRender();
  }

  setDraft(draft: Draft | null) {
    this.draft = draft;
    this.scheduleRender();
  }

  /** Coalesce the several events a single OSD frame can emit. */
  private scheduleRender() {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.render();
    });
  }

  /** Public so a test driver can force a frame when rAF is suspended. */
  render() {
    const vp = this.viewer.viewport;
    const container = vp.getContainerSize();
    if (container.x < 1 || container.y < 1) return;

    // OSD viewport -> tile-source image pixels -> level-0 slide pixels.
    const rect = vp.viewportToImageRectangle(vp.getBounds(true));
    const centre: [number, number, number] = [
      rect.x + rect.width / 2 + this.offset.x,
      rect.y + rect.height / 2 + this.offset.y,
      0,
    ];
    // Orthographic zoom: one world unit spans 2^zoom screen pixels.
    const zoom = Math.log2(container.x / Math.max(rect.width, 1e-9));

    if (this.canvas.width !== container.x || this.canvas.height !== container.y) {
      this.canvas.style.width = `${container.x}px`;
      this.canvas.style.height = `${container.y}px`;
    }

    this.deck.setProps({
      width: container.x,
      height: container.y,
      viewState: { target: centre, zoom },
      layers: this.buildLayers(rect, zoom),
    });
    // deck.gl would otherwise defer to its own animation frame, reintroducing
    // exactly the lag the synchronous handler above exists to remove.
    this.deck.redraw("osd-sync");
  }

  private buildLayers(rect: OpenSeadragon.Rect, zoom: number) {
    const state = useAnnotations.getState();
    const layers: unknown[] = [];
    const pxPerSlidePx = 2 ** zoom;

    if (state.showAnnotations) {
      /**
       * Everything is handed to the GPU, and the viewport does the clipping.
       *
       * Culling on the CPU first seems obviously cheaper, and is the opposite:
       * it makes `data` a different set on every pan, so deck.gl re-tessellates
       * each frame — and when its cached index buffer outlives the vertex
       * buffer by a frame, indices address a different polygon's vertices and
       * the triangles between them are drawn as long slivers right across the
       * slide. Feeding it a set that changes only when the document does means
       * one upload per edit instead of one per frame, and nothing to go stale.
       * The spatial index still earns its keep for hit-testing, where the
       * question really is "which object is under this point".
       */
      const visible = state.items.values();

      const polys: PolyEntry[] = [];
      /**
       * Identity of the tessellated geometry — the document, and nothing else.
       *
       * The layer is rebuilt from scratch when this changes, which is the only
       * reliable way to be sure no tessellation state outlives the vertices it
       * described. Selection and opacity deliberately do not appear here: they
       * only change colours, which are ordinary attribute updates.
       */
      /**
       * Hiding a class changes what is tessellated, so it belongs in the key
       * alongside the document revision — otherwise the layer keeps the
       * geometry of a class that is no longer being drawn.
       */
      let dataKey = state.version;
      for (const id of state.hiddenClasses) {
        for (let i = 0; i < id.length; i++) {
          dataKey = Math.imul(dataKey ^ id.charCodeAt(i), 16777619) >>> 0;
        }
      }

      // Selection and opacity drive colours only, so they update attributes in
      // place rather than rebuilding the layer.
      let selectionKey = state.selection.size;
      for (const id of state.selection) {
        for (let i = 0; i < id.length; i++) {
          selectionKey = Math.imul(selectionKey ^ id.charCodeAt(i), 16777619) >>> 0;
        }
      }
      const points: Annotation[] = [];
      const paths: { id: string; path: Position[]; color: [number, number, number, number] }[] = [];

      const colourOf = (a: Annotation): [number, number, number] => {
        if (a.classId === ROI_CLASS_ID) return ROI_COLOR;
        const cls = state.classes.find((c) => c.id === a.classId);
        return cls ? cls.color : [150, 150, 160];
      };

      for (const a of visible) {
        if (a.classId && state.hiddenClasses.has(a.classId)) continue;
        const selected = state.selection.has(a.id);
        const [r, g, b] = colourOf(a);
        const isRoi = a.classId === ROI_CLASS_ID;
        // Predictions read as provisional until a human accepts them.
        const alpha = Math.round(
          255 * state.fillOpacity * (a.source === "model" ? 0.65 : 1) * (isRoi ? 0.25 : 1),
        );
        const fill: [number, number, number, number] = [r, g, b, alpha];
        const line: [number, number, number, number] = selected
          ? [255, 255, 255, 255]
          : [r, g, b, 235];
        const width = selected ? 3 : isRoi ? 2 : 1.5;

        if (a.geometry.type === "Polygon") {
          const rings = renderableRings(a.geometry.coordinates);
          if (rings) polys.push({ id: a.id, rings, fill, line, width });
        } else if (a.geometry.type === "MultiPolygon") {
          a.geometry.coordinates.forEach((part, i) => {
            const rings = renderableRings(part);
            if (rings) polys.push({ id: `${a.id}:${i}`, rings, fill, line, width });
          });
        } else if (a.geometry.type === "Point") {
          points.push(a);
        } else {
          paths.push({ id: a.id, path: a.geometry.coordinates, color: line });
        }
      }

      if (polys.length) {
        layers.push(
          new PolygonLayer<PolyEntry>({
            // A new layer per document revision; see `dataKey`.
            id: `annotations-${dataKey}`,
            data: polys,
            getPolygon: (d) => d.rings,
            getFillColor: (d) => d.fill,
            getLineColor: (d) => d.line,
            getLineWidth: (d) => d.width,
            lineWidthUnits: "pixels",
            lineWidthMinPixels: 1,
            filled: true,
            stroked: true,
            pickable: false,
            updateTriggers: {
              getFillColor: `${selectionKey}|${state.fillOpacity}`,
              getLineColor: selectionKey,
              getLineWidth: selectionKey,
            },
          }),
        );
      }

      if (paths.length) {
        layers.push(
          new PathLayer<(typeof paths)[number]>({
            id: "annotation-lines",
            data: paths,
            getPath: (d) => d.path,
            getColor: (d) => d.color,
            getWidth: 2,
            widthUnits: "pixels",
            widthMinPixels: 2,
          }),
        );
      }

      if (points.length) {
        layers.push(
          new ScatterplotLayer<Annotation>({
            id: "annotation-points",
            data: points,
            getPosition: (d) => (d.geometry as { coordinates: Position }).coordinates,
            getFillColor: (d) => {
              const [r, g, b] = colourOf(d);
              return [r, g, b, 230];
            },
            getLineColor: (d) => (state.selection.has(d.id) ? [255, 255, 255, 255] : [20, 20, 24, 200]),
            getRadius: 5,
            radiusUnits: "pixels",
            radiusMinPixels: 3,
            stroked: true,
            lineWidthUnits: "pixels",
            getLineWidth: 1.5,
          }),
        );
      }
    }

    // ------------------------------------------------------------- draft ---
    /**
     * Predicted expression, one square per patch.
     *
     * Drawn beneath the annotations rather than over them: the expression is
     * the field being read, and the boundaries drawn on it have to stay legible
     * against it. It has its own opacity for the same reason — this is the one
     * layer you want to fade in and out against the tissue underneath.
     */
    const spatial = useSpatial.getState();
    if (spatial.visible && spatial.result) {
      const field = currentField(spatial.result, spatial, scoreSignature);
      if (field) {
        const values = field.values;
        const range = robustRange(values);
        const side = spatial.result.side;
        const alpha = Math.round(255 * spatial.opacity);
        const patches = spatial.result.patches;
        layers.push(
          new PolygonLayer<{ i: number }>({
            id: `spatial-${spatial.result.createdAt}-${field.label}`,
            data: patches.map((_, i) => ({ i })),
            getPolygon: (d) => {
              const p = patches[d.i];
              return [[
                [p.x, p.y],
                [p.x + side, p.y],
                [p.x + side, p.y + side],
                [p.x, p.y + side],
              ]];
            },
            getFillColor: (d) => {
              const [r, g, b] = colourFor(values[d.i], range);
              return [r, g, b, alpha];
            },
            filled: true,
            stroked: false,
            updateTriggers: { getFillColor: `${field.label}|${spatial.opacity}` },
          }),
        );
      }
    }

    /**
     * The patch grid an ROI will be sampled into.
     *
     * Drawn as outlines rather than filled cells: the point of showing it is to
     * check the patches against the tissue underneath, which a wash of colour
     * would hide. Cells are cheap to draw but not free, so past a few thousand
     * only the outline of the covered area is worth the frame — and at that
     * count the individual cells are sub-pixel anyway.
     */
    const gridState = useMl.getState().grid;
    if (gridState && state.showAnnotations) {
      const { patches, downsample, patchPx } = gridState.grid;
      const side = patchPx * downsample;
      /**
       * Drawn wherever a cell is at least a couple of pixels across.
       *
       * The old floor of three pixels meant that laying a grid and then looking
       * at the whole ROI showed nothing at all — the one moment you most want
       * to see it, to judge whether the scale is right and whether it is
       * sitting on tissue.
       */
      if (patches.length <= 20000 && side * pxPerSlidePx > 1.5) {
        layers.push(
          new PolygonLayer<{ x: number; y: number }>({
            id: "patch-grid",
            data: patches,
            getPolygon: (d2) => [[
              [d2.x, d2.y],
              [d2.x + side, d2.y],
              [d2.x + side, d2.y + side],
              [d2.x, d2.y + side],
            ]],
            filled: false,
            stroked: true,
            getLineColor: [120, 220, 255, 150],
            getLineWidth: 1,
            lineWidthUnits: "pixels",
            updateTriggers: { getPolygon: side },
          }),
        );
      }
    }

    // Resize handles for the selected ROI, drawn last so nothing covers them.
    const target = resizeTarget(state.items, state.selection);
    if (target && state.showAnnotations) {
      layers.push(
        new ScatterplotLayer<Handle>({
          id: "resize-handles",
          data: handlesOf(boxOf(target)),
          getPosition: (d2) => [d2.x, d2.y],
          getRadius: 4.5,
          radiusUnits: "pixels",
          filled: true,
          stroked: true,
          getFillColor: [255, 255, 255, 255],
          getLineColor: [20, 20, 24, 220],
          getLineWidth: 1.5,
          lineWidthUnits: "pixels",
        }),
      );
    }

    const d = this.draft;
    if (d) {
      const [r, g, b] = d.color;
      if (d.kind === "polygon" && d.rings?.length) {
        layers.push(
          new PolygonLayer<{ rings: Ring[] }>({
            id: "draft-polygon",
            data: [{ rings: d.rings }],
            getPolygon: (x) => x.rings,
            getFillColor: [r, g, b, d.erasing ? 40 : 90],
            getLineColor: d.erasing ? [255, 120, 120, 255] : [255, 255, 255, 255],
            getLineWidth: 2,
            lineWidthUnits: "pixels",
            filled: true,
            stroked: true,
          }),
        );
      }
      // A brush stroke previews as a wide round-capped path: the GPU draws the
      // exact swept area, so there is no need to union geometry on every move.
      if (d.kind === "stroke" && d.path && d.path.length > 0 && d.cursorRadius) {
        layers.push(
          new PathLayer<{ path: Position[] }>({
            id: "draft-stroke",
            data: [{ path: d.path.length === 1 ? [d.path[0], d.path[0]] : d.path }],
            getPath: (x) => x.path,
            getColor: d.erasing ? [255, 120, 120, 110] : [r, g, b, 130],
            getWidth: d.cursorRadius * 2,
            widthUnits: "common",
            capRounded: true,
            jointRounded: true,
            updateTriggers: { getWidth: d.cursorRadius },
          }),
        );
      }
      if (d.kind === "path" && d.path && d.path.length > 1) {
        layers.push(
          new PathLayer<{ path: Position[] }>({
            id: "draft-path",
            data: [{ path: d.path }],
            getPath: (x) => x.path,
            getColor: [255, 255, 255, 255],
            getWidth: 2,
            widthUnits: "pixels",
            widthMinPixels: 2,
          }),
        );
      }
      if (d.cursorAt && d.cursorRadius) {
        layers.push(
          new ScatterplotLayer<{ p: Position }>({
            id: "brush-cursor",
            data: [{ p: d.cursorAt }],
            getPosition: (x) => x.p,
            getRadius: d.cursorRadius,
            radiusUnits: "common",
            filled: false,
            stroked: true,
            getLineColor: d.erasing ? [255, 120, 120, 230] : [255, 255, 255, 230],
            getLineWidth: 1.5,
            lineWidthUnits: "pixels",
            updateTriggers: { getRadius: d.cursorRadius * pxPerSlidePx },
          }),
        );
      }
    }

    return layers as never[];
  }

  destroy() {
    if (this.frame) cancelAnimationFrame(this.frame);
    this.unsubscribe();
    this.deck.finalize();
  }
}
