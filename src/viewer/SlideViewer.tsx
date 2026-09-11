import { useEffect, useRef, useState } from "react";
import OpenSeadragon from "openseadragon";
import { useAnnotations } from "../annotate/store";
import { buildPatchGrid } from "../ml/patchGrid";
import { ToolController } from "../annotate/toolController";
import type { SlideSource } from "../slide/types";
import { AnnotationOverlay } from "./annotationOverlay";
import { SegmentController } from "../ml/segmentController";
import { useMl } from "../ml/mlStore";
import { createOpenSlideTileSource, type OpenSlideTileSource } from "./openslideTileSource";

interface Props {
  source: SlideSource;
  /** Receives the segmentation controller once the viewer exists. */
  onSegmenter?: (s: SegmentController | null) => void;
  /** Focuses a slide-pixel rectangle; handed out once the viewer exists. */
  onFocuser?: (f: ((bbox: [number, number, number, number]) => void) | null) => void;
  onStats?: (s: { tiles: number; avgMs: number; maxMs: number; errors: number }) => void;
}

export function SlideViewer({ source, onStats, onSegmenter, onFocuser }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const inputRef = useRef<HTMLDivElement>(null);
  const controllerRef = useRef<ToolController | null>(null);
  const [firstTileMs, setFirstTileMs] = useState<number | null>(null);

  const tool = useAnnotations((s) => s.tool);

  useEffect(() => {
    const host = hostRef.current;
    const canvas = canvasRef.current;
    const input = inputRef.current;
    if (!host || !canvas || !input) return;

    const openedAt = performance.now();
    let tileSource: OpenSlideTileSource;
    try {
      tileSource = createOpenSlideTileSource(source);
    } catch (err) {
      console.error("Failed to build tile source", err);
      return;
    }

    const viewer = OpenSeadragon({
      element: host,
      tileSources: [tileSource],
      drawer: "webgl",
      prefixUrl: "",
      showNavigationControl: false,
      showNavigator: true,
      navigatorPosition: "TOP_RIGHT",
      navigatorSizeRatio: 0.11,
      navigatorBorderColor: "#2e343d",
      navigatorDisplayRegionColor: "#4cc2c4",
      navigatorAutoFade: false,
      maxZoomPixelRatio: 4,
      minZoomImageRatio: 0.6,
      visibilityRatio: 0.6,
      constrainDuringPan: true,
      smoothTileEdgesMinZoom: Infinity,
      imageLoaderLimit: 12,
      timeout: 120000,
      animationTime: 0.5,
      springStiffness: 8,
      gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: false },
    });

    // Annotations live in the full vendor frame; the tile source is cropped to
    // the MIRAX scan region, so the overlay shifts by that offset.
    const bounds = source.meta.bounds;
    const offset = { x: bounds?.x ?? 0, y: bounds?.y ?? 0 };

    const overlay = new AnnotationOverlay(viewer, canvas, offset);
    const segmenter = new SegmentController(viewer, source, overlay, offset);
    /**
     * Lay the grid for a region the patch tool just drew.
     *
     * State is read at call time rather than captured: the controller is built
     * once per slide, and the patch size is chosen after that — a closure over
     * the size at construction would tile every region at whatever it was when
     * the slide opened.
     */
    const layPatchGrid = (roiId: string) => {
      const { items, classes } = useAnnotations.getState();
      const roi = items.get(roiId);
      if (!roi) return;

      const { patchPx, patchLevel, patchesOnTissueOnly, setGrid } = useMl.getState();
      const meta = source.meta;
      const level = meta.levels[Math.min(patchLevel, meta.levels.length - 1)];
      const tissueClass = classes.find((c) => c.name.toLowerCase() === "tissue");
      const tissue = tissueClass
        ? [...items.values()].filter((a) => a.classId === tissueClass.id)
        : [];

      const [minX, minY, maxX, maxY] = roi.bbox;
      const grid = buildPatchGrid(
        { x: minX, y: minY, width: maxX - minX, height: maxY - minY },
        level?.downsample ?? 1,
        meta.mppX,
        {
          patchPx,
          level: level?.level ?? 0,
          within: [roi],
          restrictTo: patchesOnTissueOnly && tissue.length > 0 ? tissue : undefined,
          bounds: meta.bounds,
        },
      );
      setGrid({ roiId: roi.id, grid });
    };

    const controller = new ToolController(
      viewer, input, overlay, offset, segmenter, layPatchGrid,
    );
    onSegmenter?.(segmenter);
    void segmenter.refreshCacheStatus();

    // Frame a slide-pixel box with margin, so a clicked annotation lands in
    // context rather than filling the viewport edge to edge.
    onFocuser?.(([minX, minY, maxX, maxY]) => {
      const pad = Math.max(24, (maxX - minX) * 0.35, (maxY - minY) * 0.35);
      const item = viewer.world.getItemAt(0);
      if (!item) return;
      const tl = item.imageToViewportCoordinates(
        minX - pad - offset.x,
        minY - pad - offset.y,
      );
      const br = item.imageToViewportCoordinates(
        maxX + pad - offset.x,
        maxY + pad - offset.y,
      );
      viewer.viewport.fitBounds(
        new OpenSeadragon.Rect(tl.x, tl.y, Math.max(br.x - tl.x, 1e-6), Math.max(br.y - tl.y, 1e-6)),
        false,
      );
    });
    controllerRef.current = controller;
    controller.syncPointerEvents();

    if (import.meta.env.DEV) {
      const w = window as unknown as Record<string, unknown>;
      w.__osd = viewer;
      w.__slide = source;
      w.__overlay = overlay;
      w.__segmenter = segmenter;
      // Hand tests the app's own store instances: a dynamic import of the same
      // path can resolve to a second copy under Vite's HMR-versioned URLs.
      w.__ml = useMl;
    }

    let seenFirst = false;
    const onTileLoaded = () => {
      if (!seenFirst) {
        seenFirst = true;
        setFirstTileMs(performance.now() - openedAt);
      }
    };
    viewer.addHandler("tile-loaded", onTileLoaded);

    const interval = window.setInterval(() => {
      const s = tileSource.stats;
      onStats?.({
        tiles: s.tiles,
        avgMs: s.tiles ? s.totalMs / s.tiles : 0,
        maxMs: s.maxMs,
        errors: s.errors,
      });
    }, 400);

    // OSD measures its container once at construction and clamps a zero-sized
    // one to 1x1, which silently wedges level selection forever. Keep the
    // viewport in step with the element instead.
    let hasValidSize = false;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width < 1 || height < 1) return;
      const current = viewer.viewport.getContainerSize();
      if (Math.abs(current.x - width) < 1 && Math.abs(current.y - height) < 1) return;

      // Only the first real measurement should frame the slide. Later resizes
      // keep the user where they are — snapping home on every window resize
      // would throw away their position mid-annotation.
      viewer.viewport.resize(new OpenSeadragon.Point(width, height), hasValidSize);
      if (!hasValidSize) {
        hasValidSize = true;
        viewer.viewport.goHome(true);
      }
      viewer.forceRedraw();
      overlay.render();
    });
    ro.observe(host);

    return () => {
      ro.disconnect();
      window.clearInterval(interval);
      viewer.removeHandler("tile-loaded", onTileLoaded);
      controller.destroy();
      controllerRef.current = null;
      onSegmenter?.(null);
      onFocuser?.(null);
      segmenter.destroy();
      overlay.destroy();
      viewer.destroy();
    };
  }, [source, onStats, onSegmenter, onFocuser]);

  // Only the active tool decides whether the input layer swallows pointer events.
  useEffect(() => {
    controllerRef.current?.syncPointerEvents();
  }, [tool]);

  return (
    <>
      <div className="osd-host" ref={hostRef} />
      <canvas className="anno-canvas" ref={canvasRef} />
      <div className="anno-input" ref={inputRef} />
      {firstTileMs !== null && (
        <div className="status" style={{ left: "auto", right: 12 }}>
          <span>
            first tile <b>{firstTileMs.toFixed(0)} ms</b>
          </span>
        </div>
      )}
    </>
  );
}
