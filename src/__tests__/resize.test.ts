import { beforeEach, afterEach, describe, expect, it } from "vitest";
import { makeHarness, resetStore, type Harness } from "./toolHarness";
import { useAnnotations } from "../annotate/store";
import { handlesOf, resizeTarget } from "../annotate/resize";
import { ROI_CLASS_ID, type Annotation } from "../annotate/types";

const S = () => useAnnotations.getState();
const all = () => [...S().items.values()];
const box = (a: Annotation) => S().items.get(a.id)!.bbox.map(Math.round);

let h: Harness;
beforeEach(() => {
  resetStore();
  h = makeHarness();
});
afterEach(() => h.destroy());

function drawRoi(a: [number, number], b: [number, number]): Annotation {
  S().setTool("roi");
  h.drag(a, b);
  const roi = all().at(-1)!;
  S().setTool("select");
  return roi;
}

describe("resizing an ROI", () => {
  it("offers handles only for a single unlocked ROI", () => {
    const roi = drawRoi([200, 200], [400, 300]);
    expect(resizeTarget(S().items, S().selection)?.id).toBe(roi.id);

    // A locked ROI is a fixed frame; resizing it would defeat the lock.
    S().apply({ label: "lock", updated: [{ before: roi, after: { ...roi, locked: true } }] });
    expect(resizeTarget(S().items, S().selection)).toBe(null);
  });

  it("does not offer handles for ordinary annotations", () => {
    // Rescaling a traced object would turn its recorded area into fiction.
    const cls = S().ensureClass("Tumour");
    S().setActiveClass(cls.id);
    S().setTool("rectangle");
    h.drag([100, 100], [200, 200]);
    const ann = all().at(-1)!;
    expect(ann.classId).not.toBe(ROI_CLASS_ID);
    S().setTool("select");
    S().select([ann.id]);
    expect(resizeTarget(S().items, S().selection)).toBe(null);
  });

  it("drags the south-east corner out and leaves the anchor put", () => {
    const roi = drawRoi([200, 200], [400, 300]);
    const se = handlesOf({ minX: 200, minY: 200, maxX: 400, maxY: 300 })
      .find((x) => x.id === "se")!;

    h.drag([se.x, se.y], [520, 380]);

    expect(box(roi)).toEqual([200, 200, 520, 380]);
    expect(S().items.size).toBe(1);
  });

  it("drags a west edge without moving the other three", () => {
    const roi = drawRoi([200, 200], [400, 300]);
    const w = handlesOf({ minX: 200, minY: 200, maxX: 400, maxY: 300 }).find((x) => x.id === "w")!;

    h.drag([w.x, w.y], [140, 260]);

    // Only minX changes: an edge handle must not drag the corner with it.
    expect(box(roi)).toEqual([140, 200, 400, 300]);
  });

  it("is one undoable step, and undo restores the original box", () => {
    const roi = drawRoi([200, 200], [400, 300]);
    const se = handlesOf({ minX: 200, minY: 200, maxX: 400, maxY: 300 })
      .find((x) => x.id === "se")!;
    h.drag([se.x, se.y], [520, 380]);
    expect(box(roi)).toEqual([200, 200, 520, 380]);

    S().undo();
    expect(box(roi)).toEqual([200, 200, 400, 300]);
    S().redo();
    expect(box(roi)).toEqual([200, 200, 520, 380]);
  });

  /**
   * Dragging a corner past its opposite must not invert or collapse the box.
   * A zero-extent ROI cannot be grabbed again, so it would be unrecoverable
   * except by undo — and it would silently take its patch grid with it.
   */
  it("refuses to collapse the box when a corner is dragged past its anchor", () => {
    const roi = drawRoi([200, 200], [400, 300]);
    const se = handlesOf({ minX: 200, minY: 200, maxX: 400, maxY: 300 })
      .find((x) => x.id === "se")!;

    h.drag([se.x, se.y], [100, 100]);

    const [minX, minY, maxX, maxY] = box(roi);
    expect(maxX).toBeGreaterThan(minX);
    expect(maxY).toBeGreaterThan(minY);
    expect(minX).toBe(200);
    expect(minY).toBe(200);
  });

  /**
   * A handle sits exactly on the ROI's own boundary, so a plain hit test always
   * claims the press first. If it wins, the corner is ungrabbable and the ROI
   * just moves instead — which is precisely how "resizable" quietly is not.
   */
  it("resizes rather than moves when the press lands on a handle", () => {
    const roi = drawRoi([200, 200], [400, 300]);
    const ne = handlesOf({ minX: 200, minY: 200, maxX: 400, maxY: 300 })
      .find((x) => x.id === "ne")!;

    h.drag([ne.x, ne.y], [460, 150]);

    // A move would have shifted all four edges by the same delta.
    expect(box(roi)).toEqual([200, 150, 460, 300]);
  });

  it("still moves the ROI when the press lands in its middle", () => {
    const roi = drawRoi([200, 200], [400, 300]);
    h.drag([300, 250], [330, 280]);
    expect(box(roi)).toEqual([230, 230, 430, 330]);
  });

  it("leaves the ROI untouched when a handle is clicked but not dragged", () => {
    const roi = drawRoi([200, 200], [400, 300]);
    const se = handlesOf({ minX: 200, minY: 200, maxX: 400, maxY: 300 })
      .find((x) => x.id === "se")!;
    h.click(se.x, se.y);

    expect(box(roi)).toEqual([200, 200, 400, 300]);
    // And undo must not now unwind the *drawing* of the ROI, which is what
    // happens when a no-op gesture still pushes a patch.
    S().undo();
    expect(S().items.size).toBe(0);
  });

  it("can still be deleted after being resized", () => {
    const roi = drawRoi([200, 200], [400, 300]);
    const se = handlesOf({ minX: 200, minY: 200, maxX: 400, maxY: 300 })
      .find((x) => x.id === "se")!;
    h.drag([se.x, se.y], [520, 380]);

    S().select([roi.id]);
    h.key("Backspace");
    expect(S().items.size).toBe(0);
  });
});
