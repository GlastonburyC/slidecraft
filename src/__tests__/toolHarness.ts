import OpenSeadragon from "openseadragon";
import { ToolController } from "../annotate/toolController";
import { useAnnotations } from "../annotate/store";
import type { AnnotationOverlay } from "../viewer/annotationOverlay";

/**
 * Drives ToolController without a browser.
 *
 * The controller only needs a viewport that can map pixels to slide
 * coordinates and an element that emits pointer events, so both are stubbed.
 * That keeps these tests fast and, more importantly, deterministic — the real
 * viewer's behaviour depends on compositing and animation frames, which is
 * exactly the sort of thing that hides an intermittent bug.
 */

/** Slide pixels per screen pixel; 1 keeps test coordinates readable. */
const SCALE = 1;

export interface Harness {
  controller: ToolController;
  element: HTMLElement;
  down: (x: number, y: number, mods?: PointerEventInit) => void;
  move: (x: number, y: number, mods?: PointerEventInit) => void;
  up: (x: number, y: number, mods?: PointerEventInit) => void;
  drag: (a: [number, number], b: [number, number], mods?: PointerEventInit) => void;
  click: (x: number, y: number, mods?: PointerEventInit) => void;
  contextMenu: (x: number, y: number) => void;
  key: (key: string, mods?: KeyboardEventInit) => void;
  destroy: () => void;
}

function stubViewer() {
  const point = (x: number, y: number) => ({ x, y });
  const viewport = {
    getContainerSize: () => point(960, 720),
    getBounds: () => ({ x: 0, y: 0, width: 960 * SCALE, height: 720 * SCALE }),
    // Pixels map straight through, so a click at (x, y) is slide (x, y).
    pointFromPixel: (p: { x: number; y: number }) => point(p.x, p.y),
    viewportToImageCoordinates: (p: { x: number; y: number }) => point(p.x, p.y),
    imageToViewportCoordinates: (x: number, y: number) => point(x, y),
    pixelFromPoint: (p: { x: number; y: number }) => point(p.x, p.y),
    viewportToImageRectangle: (r: { x: number; y: number; width: number; height: number }) => r,
    zoomBy: () => undefined,
    applyConstraints: () => undefined,
  };
  return { viewport, world: { getItemAt: () => null } } as unknown as OpenSeadragon.Viewer;
}

function stubOverlay(): AnnotationOverlay {
  return { setDraft: () => undefined, render: () => undefined } as unknown as AnnotationOverlay;
}

export function makeHarness(): Harness {
  const parent = document.createElement("div");
  const element = document.createElement("div");
  // The controller only opens its context menu for the image surface, so the
  // stub must present itself as that surface.
  element.className = "anno-input";
  parent.appendChild(element);
  document.body.appendChild(parent);
  element.getBoundingClientRect = () =>
    ({ left: 0, top: 0, width: 960, height: 720, right: 960, bottom: 720, x: 0, y: 0 }) as DOMRect;
  element.setPointerCapture = () => undefined;
  element.releasePointerCapture = () => undefined;

  const controller = new ToolController(stubViewer(), element, stubOverlay(), { x: 0, y: 0 });

  const fire = (type: string, x: number, y: number, mods: PointerEventInit = {}) => {
    element.dispatchEvent(
      new PointerEvent(type, {
        bubbles: true,
        clientX: x,
        clientY: y,
        pointerId: 1,
        button: 0,
        buttons: type === "pointerup" ? 0 : 1,
        ...mods,
      }),
    );
  };

  const h: Harness = {
    controller,
    element,
    down: (x, y, m) => fire("pointerdown", x, y, m),
    move: (x, y, m) => fire("pointermove", x, y, m),
    up: (x, y, m) => fire("pointerup", x, y, m),
    drag: ([ax, ay], [bx, by], m) => {
      fire("pointerdown", ax, ay, m);
      for (let i = 1; i <= 6; i++) {
        fire("pointermove", ax + ((bx - ax) * i) / 6, ay + ((by - ay) * i) / 6, m);
      }
      fire("pointerup", bx, by, m);
    },
    click: (x, y, m) => {
      fire("pointerdown", x, y, m);
      fire("pointerup", x, y, m);
    },
    contextMenu: (x, y) =>
      element.dispatchEvent(
        new MouseEvent("contextmenu", { bubbles: true, clientX: x, clientY: y }),
      ),
    key: (key, mods = {}) =>
      window.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true, ...mods })),
    destroy: () => {
      controller.destroy();
      parent.remove();
    },
  };
  return h;
}

export function resetStore() {
  useAnnotations.getState().resetFor("test-slide");
  useAnnotations.setState({ classes: [], activeClassId: null, tool: "pan" });
}
