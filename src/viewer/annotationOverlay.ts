import { Deck, OrthographicView } from "@deck.gl/core";
import { PathLayer, PolygonLayer, ScatterplotLayer } from "@deck.gl/layers";
import type OpenSeadragon from "openseadragon";
import { renderableRings } from "../annotate/geometry";
import { handlesOf, boxOf, resizeTarget, type Handle } from "../annotate/resize";
import { useAnnotations } from "../annotate/store";
import { useMl } from "../ml/mlStore";
import { colourFor, currentField, robustRange } from "../ml/spatialResult";
import { blendField, type Blend } from "../ml/blend";
import { scoreSignature } from "../ml/signatures";
import { useSpatial } from "../ml/spatialStore";
import { usePredict } from "../ml/predictStore";
import type { Annotation, Position, Ring } from "../annotate/types";
import { AXIS_CLASS_ID, AXIS_COLOR, ROI_CLASS_ID, ROI_COLOR } from "../annotate/types";

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

/**
 * The spacing a map's patches were laid on, which is the stride rather than the
 * patch width. The smallest gap between distinct coordinates, because an
 * overlapping run has many patches sharing a row.
 */
function strideOf(patches: { x: number; y: number }[], fallback: number): number {
  const seen = (key: "x" | "y") => {
    const v = [...new Set(patches.map((p) => p[key]))].sort((a, b) => a - b);
    let g = Infinity;
    for (let i = 1; i < v.length; i++) g = Math.min(g, v[i] - v[i - 1]);
    return g;
  };
  const g = Math.min(seen("x"), seen("y"));
  return Number.isFinite(g) && g > 0 ? g : fallback;
}

/**
 * One blend kept between renders.
 *
 * Blending a whole-slide map is a pass over every patch times the cells it
 * covers, and the overlay rebuilds its layers whenever anything in the document
 * changes — a pan, a selection, a new annotation. Recomputing it each time
 * would make the map the slowest thing on screen.
 */
let cached: { key: string; value: Blend | null } | null = null;
function blendCache(a: string, b: string, c: string, make: () => Blend | null): Blend | null {
  const key = `${a}|${b}|${c}`;
  if (!cached || cached.key !== key) cached = { key, value: make() };
  return cached.value;
}

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
    const unsubPredict = usePredict.subscribe((s, prev) => {
      if (
        s.prediction !== prev.prediction ||
        s.grid !== prev.grid ||
        s.shownClass !== prev.shownClass ||
        s.opacity !== prev.opacity ||
        s.threshold !== prev.threshold ||
        s.visible !== prev.visible
      ) {
        this.scheduleRender();
      }
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
      unsubPredict();
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
      const paths: {
        id: string; path: Position[]; color: [number, number, number, number]; width: number;
      }[] = [];

      const colourOf = (a: Annotation): [number, number, number] => {
        if (a.classId === ROI_CLASS_ID) return ROI_COLOR;
        if (a.classId === AXIS_CLASS_ID) return AXIS_COLOR;
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
          // An axis is drawn to be followed across a whole section, so it is
          // heavier than a plain imported line — a hairline over a coloured
          // expression map is not findable.
          const isAxis = a.classId === AXIS_CLASS_ID;
          paths.push({
            id: a.id, path: a.geometry.coordinates, color: line,
            width: isAxis ? (selected ? 5 : 4) : width,
          });
          /**
           * An axis gets a head, because its direction is the whole content.
           *
           * Every gradient reported along it is signed by which end is the
           * start, and a plain line says nothing about that — so the reader
           * would have no way to tell "rises toward the surface" from the
           * opposite claim.
           *
           * Sized as a fraction of the line so it stays in proportion as the
           * view zooms, with a floor so a short axis still shows one.
           */
          if (isAxis && a.geometry.coordinates.length >= 2) {
            const pts = a.geometry.coordinates;
            const [x0, y0] = pts[pts.length - 2];
            const [x1, y1] = pts[pts.length - 1];
            const dx = x1 - x0;
            const dy = y1 - y0;
            const len = Math.hypot(dx, dy) || 1;
            const head = Math.max(len * 0.12, 24);
            const ux = dx / len;
            const uy = dy / len;
            // Rotate the reversed direction by +/- 28 degrees for the barbs.
            const rot = (ang: number): [number, number] => [
              x1 + head * (-ux * Math.cos(ang) + uy * Math.sin(ang)),
              y1 + head * (-uy * Math.cos(ang) - ux * Math.sin(ang)),
            ];
            const a1 = rot(0.49);
            const a2 = rot(-0.49);
            paths.push({
              id: `${a.id}:head`, path: [a1, [x1, y1], a2], color: line,
              width: selected ? 5 : 4,
            });
          }
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
            // Per entry, not fixed: an axis needs to be heavier than an
            // imported outline, and this layer carries both.
            getWidth: (d) => d.width,
            widthUnits: "pixels",
            widthMinPixels: 2,
            updateTriggers: { getColor: selectionKey, getWidth: selectionKey },
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
        const mask = spatial.onTissueOnly ? spatial.tissueMask : null;
        const maskKey = mask ? `t${spatial.tissueMaskVersion}` : "off";

        /*
         * A strided map is blended; an unstrided one is drawn as it is.
         *
         * When the stride is finer than the patch, every patch still covers a
         * full patch-width and they overlap. Drawn as squares they overdraw,
         * and whichever happens to be last wins outright — so the finer run
         * costs many times the compute and looks no better. Blending them with
         * a raised cosine is what converts that overlap into resolution: each
         * output cell is the weighted average of every patch covering it,
         * weighted by how near its centre the cell sits.
         */
        const step = strideOf(patches, side);
        const blended = step < side * 0.99
          ? blendCache(spatial.result.createdAt, field.label, maskKey,
                       () => blendField(patches, values, mask, side, step))
          : null;

        if (blended) {
          const { cols, rows, originX, originY, cell } = blended;
          const cells: { i: number }[] = [];
          for (let i = 0; i < cols * rows; i++) cells.push({ i });
          const at = (i: number) => blended.probs[i * 2];
          const covered = (i: number) => blended.probs[i * 2 + 1];
          const cellRange = robustRange(
            Float32Array.from({ length: cols * rows }, (_, i) => at(i)),
          );
          layers.push(
            new PolygonLayer<{ i: number }>({
              id: `spatial-${spatial.result.createdAt}-${field.label}-blend`,
              data: cells,
              getPolygon: (d) => {
                const x = originX + (d.i % cols) * cell;
                const y = originY + Math.floor(d.i / cols) * cell;
                return [[[x, y], [x + cell, y], [x + cell, y + cell], [x, y + cell]]];
              },
              getFillColor: (d) => {
                // Half is the midpoint of a blended mask: a cell most of whose
                // weight came from off-tissue patches is left blank.
                if (covered(d.i) < 0.5) return [0, 0, 0, 0];
                const [r, g, b] = colourFor(at(d.i), cellRange);
                return [r, g, b, alpha];
              },
              filled: true,
              stroked: false,
              updateTriggers: {
                getFillColor: `${field.label}|${spatial.opacity}|${maskKey}`,
              },
            }),
          );
        } else {
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
                // A patch off the tissue is drawn fully transparent rather than
                // dropped from the data, so toggling the mask is a colour change
                // and not a rebuild of every polygon.
                if (mask && !mask[d.i]) return [0, 0, 0, 0];
                const [r, g, b] = colourFor(values[d.i], range);
                return [r, g, b, alpha];
              },
              filled: true,
              stroked: false,
              updateTriggers: {
                getFillColor: `${field.label}|${spatial.opacity}|${maskKey}`,
              },
            }),
          );
        }
      }
    }

    /**
     * What the head thinks, one square per patch.
     *
     * Drawn under the annotations so the regions you are correcting stay
     * legible on top of it, and thresholded so patches the head is unsure
     * about are left alone rather than coloured with false confidence — an
     * uncertain patch shown at full strength is how a heatmap becomes more
     * persuasive than the model that made it.
     */
    const pred = usePredict.getState();

    /**
     * What has been embedded, before anything has been predicted.
     *
     * Embedding produces no picture of its own, so without this the slide is
     * unchanged after a run that took minutes — leaving no way to tell a
     * finished embed from one that silently did nothing, or to see that the
     * grid covers the tissue you meant. Drawn as outlines, and replaced by the
     * prediction as soon as there is one.
     */
    if (pred.visible && pred.grid && !pred.prediction && state.showAnnotations) {
      const { patches, patchPx, downsample } = pred.grid;
      const side = Math.round(patchPx * downsample);
      if (patches.length <= 20000 && side * pxPerSlidePx > 1.5) {
        layers.push(
          new PolygonLayer<{ i: number }>({
            id: "embedded-grid",
            data: patches.map((_, i) => ({ i })),
            getPolygon: (d) => {
              const q = patches[d.i];
              return [[
                [q.x, q.y],
                [q.x + side, q.y],
                [q.x + side, q.y + side],
                [q.x, q.y + side],
              ]];
            },
            filled: true,
            stroked: true,
            getFillColor: [120, 220, 255, 26],
            getLineColor: [120, 220, 255, 150],
            getLineWidth: 1,
            lineWidthUnits: "pixels",
            updateTriggers: { getPolygon: side },
          }),
        );
      }
    }

    /**
     * A blended prediction is drawn at the stride's resolution, not the
     * patch's. Overlapping patches each judge a point of tissue from a
     * different offset, so the average is finer than any one of them — and
     * drawing the patches instead would throw that away, since the last square
     * painted would simply cover the rest.
     */
    const blend = pred.prediction?.blend ?? null;
    if (pred.visible && blend && state.showAnnotations) {
      const { classes, classIds } = pred.prediction!;
      const n = blend.classes;
      const alpha = Math.round(255 * pred.opacity);
      const shown = pred.shownClass ? classes.indexOf(pred.shownClass) : -1;
      const palette = classes.map((_, k) => {
        const cls = state.classes.find((c) => c.id === classIds[k]);
        return cls ? cls.color : ([150, 150, 160] as [number, number, number]);
      });

      if (blend.cell * pxPerSlidePx > 1) {
        layers.push(
          new PolygonLayer<{ i: number }>({
            id: `blend-${pred.prediction!.ms}-${pred.shownClass ?? "argmax"}`,
            data: Array.from({ length: blend.cols * blend.rows }, (_, i) => ({ i })),
            getPolygon: (d) => {
              const c = d.i % blend.cols;
              const r = (d.i - c) / blend.cols;
              const x = blend.originX + c * blend.cell;
              const y = blend.originY + r * blend.cell;
              return [[
                [x, y],
                [x + blend.cell, y],
                [x + blend.cell, y + blend.cell],
                [x, y + blend.cell],
              ]];
            },
            getFillColor: (d) => {
              const base = d.i * n;
              let total = 0;
              for (let k = 0; k < n; k++) total += blend.probs[base + k];
              // Nothing covered this cell; leave it rather than guess.
              if (total <= 0) return [0, 0, 0, 0];
              if (shown >= 0) {
                const p = blend.probs[base + shown];
                if (p < pred.threshold) return [0, 0, 0, 0];
                const [r2, g2, b2] = palette[shown];
                return [r2, g2, b2, Math.round(alpha * p)];
              }
              let best = 0;
              for (let k = 1; k < n; k++) if (blend.probs[base + k] > blend.probs[base + best]) best = k;
              if (blend.probs[base + best] < pred.threshold) return [0, 0, 0, 0];
              const [r2, g2, b2] = palette[best];
              return [r2, g2, b2, alpha];
            },
            filled: true,
            stroked: false,
            updateTriggers: {
              getFillColor: `${pred.shownClass}|${pred.opacity}|${pred.threshold}|${state.version}`,
            },
          }),
        );
      }
    } else if (pred.visible && pred.prediction && state.showAnnotations) {
      const { probs, grid, classes, classIds } = pred.prediction;
      const n = classes.length;
      const side = grid.patchPx * grid.downsample;
      const alpha = Math.round(255 * pred.opacity);
      const shown = pred.shownClass ? classes.indexOf(pred.shownClass) : -1;

      const colourOfClass = (k: number): [number, number, number] => {
        const cls = state.classes.find((c) => c.id === classIds[k]);
        return cls ? cls.color : [150, 150, 160];
      };
      const palette = classes.map((_, k) => colourOfClass(k));

      if (side * pxPerSlidePx > 1.5) {
        layers.push(
          new PolygonLayer<{ i: number }>({
            id: `prediction-${pred.head?.trainedAt ?? "none"}-${pred.shownClass ?? "argmax"}`,
            data: grid.patches.map((_, i) => ({ i })),
            getPolygon: (d) => {
              const p = grid.patches[d.i];
              return [[
                [p.x, p.y],
                [p.x + side, p.y],
                [p.x + side, p.y + side],
                [p.x, p.y + side],
              ]];
            },
            getFillColor: (d) => {
              const base = d.i * n;
              if (shown >= 0) {
                // One class: opacity carries the probability, so a confident
                // patch reads as solid and an unsure one nearly vanishes.
                const p = probs[base + shown];
                if (p < pred.threshold) return [0, 0, 0, 0];
                const [r, g, b] = palette[shown];
                return [r, g, b, Math.round(alpha * p)];
              }
              let best = 0;
              for (let k = 1; k < n; k++) if (probs[base + k] > probs[base + best]) best = k;
              if (probs[base + best] < pred.threshold) return [0, 0, 0, 0];
              const [r, g, b] = palette[best];
              return [r, g, b, alpha];
            },
            filled: true,
            stroked: false,
            updateTriggers: {
              getFillColor: `${pred.shownClass}|${pred.opacity}|${pred.threshold}|${state.version}`,
            },
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
