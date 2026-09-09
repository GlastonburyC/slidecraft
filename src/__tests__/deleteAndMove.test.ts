import { beforeEach, afterEach, describe, expect, it } from "vitest";
import { makeHarness, resetStore, type Harness } from "./toolHarness";
import { useAnnotations, makeAnnotation } from "../annotate/store";
import { ROI_CLASS_ID, type Annotation } from "../annotate/types";

const S = () => useAnnotations.getState();
const count = () => S().items.size;
const all = () => [...S().items.values()];

let h: Harness;
beforeEach(() => {
  resetStore();
  h = makeHarness();
});
afterEach(() => h.destroy());

/** Draw an ROI with the roi tool and return the committed annotation. */
function drawRoi(a: [number, number], b: [number, number]): Annotation {
  S().setTool("roi");
  h.drag(a, b);
  return all().at(-1)!;
}

function lock(a: Annotation) {
  S().apply({ label: "lock", updated: [{ before: a, after: { ...a, locked: true } }] });
}

describe("drawing produces usable objects", () => {
  it("commits an ROI with finite coordinates and selects it", () => {
    const roi = drawRoi([260, 180], [360, 260]);
    expect(count()).toBe(1);
    expect(roi.bbox.every(Number.isFinite)).toBe(true);
    expect(roi.classId).toBe(ROI_CLASS_ID);
    expect(S().selection.has(roi.id)).toBe(true);
  });
});

describe("deleting", () => {
  it("Backspace deletes the freshly drawn ROI", () => {
    drawRoi([260, 180], [360, 260]);
    S().setTool("select");
    h.key("Backspace");
    expect(count()).toBe(0);
  });

  it("Delete deletes the freshly drawn ROI", () => {
    drawRoi([260, 180], [360, 260]);
    S().setTool("select");
    h.key("Delete");
    expect(count()).toBe(0);
  });

  it("deletes after the ROI has been moved", () => {
    const roi = drawRoi([260, 180], [360, 260]);
    S().setTool("select");
    h.drag([300, 220], [420, 300]);
    expect(S().undoStack.at(-1)?.label).toMatch(/^Move/);
    // The moved object must still be the same id, and still selected.
    expect(S().selection.has(roi.id)).toBe(true);
    h.key("Backspace");
    expect(count()).toBe(0);
  });

  it("deletes after a move that was undone", () => {
    drawRoi([260, 180], [360, 260]);
    S().setTool("select");
    h.drag([300, 220], [420, 300]);
    S().undo();
    h.key("Backspace");
    expect(count()).toBe(0);
  });

  it("deletes several moved ROIs in one step", () => {
    drawRoi([260, 180], [320, 240]);
    drawRoi([360, 180], [420, 240]);
    drawRoi([460, 180], [520, 240]);
    S().setTool("select");
    h.drag([240, 150], [560, 270]);
    expect(S().selection.size).toBe(3);
    h.drag([290, 210], [330, 250]);
    expect(count()).toBe(3);
    h.key("Backspace");
    expect(count()).toBe(0);
  });

  it("keeps locked objects when Backspace is pressed", () => {
    const roi = drawRoi([260, 180], [360, 260]);
    lock(all()[0]);
    S().select([roi.id]);
    h.key("Backspace");
    expect(count()).toBe(1);
  });

  it("deletes unlocked objects even when a locked one is also selected", () => {
    const a = drawRoi([260, 180], [320, 240]);
    const b = drawRoi([360, 180], [420, 240]);
    lock(S().items.get(a.id)!);
    S().setSelection([a.id, b.id]);
    h.key("Backspace");
    expect(count()).toBe(1);
    expect(S().items.has(a.id)).toBe(true);
  });

  it("does nothing when the selection is empty", () => {
    drawRoi([260, 180], [360, 260]);
    S().setTool("select");
    S().clearSelection();
    h.key("Backspace");
    expect(count()).toBe(1);
  });

  it("ignores Backspace while typing in a field", () => {
    drawRoi([260, 180], [360, 260]);
    S().setTool("select");
    const input = document.createElement("input");
    document.body.appendChild(input);
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Backspace", bubbles: true }));
    expect(count()).toBe(1);
    input.remove();
  });

  it("undo restores a deleted ROI with its geometry intact", () => {
    const roi = drawRoi([260, 180], [360, 260]);
    const before = JSON.stringify(roi.geometry);
    S().setTool("select");
    h.key("Backspace");
    expect(count()).toBe(0);
    S().undo();
    expect(count()).toBe(1);
    expect(JSON.stringify(all()[0].geometry)).toBe(before);
  });
});

describe("moving", () => {
  it("moves the selection by the drag delta", () => {
    const roi = drawRoi([260, 180], [360, 260]);
    const before = [...roi.bbox];
    S().setTool("select");
    h.drag([300, 220], [340, 250]);
    const after = S().items.get(roi.id)!.bbox;
    expect(Math.round(after[0] - before[0])).toBe(40);
    expect(Math.round(after[1] - before[1])).toBe(30);
  });

  it("records exactly one undo step for a multi-object move", () => {
    drawRoi([260, 180], [320, 240]);
    drawRoi([360, 180], [420, 240]);
    S().setTool("select");
    h.drag([240, 150], [460, 270]);
    const depth = S().undoStack.length;
    h.drag([290, 210], [330, 250]);
    expect(S().undoStack.length).toBe(depth + 1);
    expect(S().undoStack.at(-1)?.label).toBe("Move 2");
  });

  it("undo returns objects to their original position", () => {
    const roi = drawRoi([260, 180], [360, 260]);
    const before = [...roi.bbox];
    S().setTool("select");
    h.drag([300, 220], [420, 300]);
    S().undo();
    expect(S().items.get(roi.id)!.bbox).toEqual(before);
  });

  it("a click without movement does not create a move step", () => {
    drawRoi([260, 180], [360, 260]);
    S().setTool("select");
    const depth = S().undoStack.length;
    h.click(300, 220);
    expect(S().undoStack.length).toBe(depth);
  });

  it("never moves a locked object", () => {
    const roi = drawRoi([260, 180], [360, 260]);
    lock(all()[0]);
    const before = [...S().items.get(roi.id)!.bbox];
    S().setTool("select");
    h.drag([300, 220], [420, 300]);
    expect(S().items.get(roi.id)!.bbox).toEqual(before);
  });
});

describe("selecting", () => {
  it("shift-click accumulates a selection", () => {
    const a = drawRoi([260, 180], [320, 240]);
    const b = drawRoi([360, 180], [420, 240]);
    S().setTool("select");
    S().clearSelection();
    h.click(290, 210);
    h.click(390, 210, { shiftKey: true });
    expect(S().selection).toEqual(new Set([a.id, b.id]));
  });

  it("a band inside a locked container selects the children, not the container", () => {
    const big = drawRoi([250, 150], [560, 300]);
    lock(all()[0]);
    const c1 = drawRoi([290, 190], [330, 230]);
    const c2 = drawRoi([400, 190], [440, 230]);
    S().setTool("select");
    S().clearSelection();
    h.drag([270, 170], [540, 280]);
    expect(S().selection).toEqual(new Set([c1.id, c2.id]));
    expect(S().selection.has(big.id)).toBe(false);
  });

  it("children inside a locked container can then be deleted", () => {
    const big = drawRoi([250, 150], [560, 300]);
    lock(all()[0]);
    drawRoi([290, 190], [330, 230]);
    drawRoi([400, 190], [440, 230]);
    S().setTool("select");
    S().clearSelection();
    h.drag([270, 170], [540, 280]);
    h.key("Backspace");
    expect(count()).toBe(1);
    expect(S().items.has(big.id)).toBe(true);
  });

  it("clicking a locked object still selects it", () => {
    const roi = drawRoi([260, 180], [420, 300]);
    lock(all()[0]);
    S().setTool("select");
    S().clearSelection();
    h.click(340, 240);
    expect(S().selection).toEqual(new Set([roi.id]));
  });

  it("clicking empty space clears the selection", () => {
    drawRoi([260, 180], [320, 240]);
    S().setTool("select");
    h.click(700, 500);
    expect(S().selection.size).toBe(0);
  });
});

describe("right-click", () => {
  it("opens a menu targeting the object under the pointer", () => {
    const roi = drawRoi([260, 180], [360, 260]);
    S().setTool("select");
    S().clearSelection();
    h.contextMenu(300, 220);
    expect(S().contextMenu?.targetIds).toEqual([roi.id]);
  });

  it("targets the whole selection when right-clicking inside it", () => {
    const a = drawRoi([260, 180], [320, 240]);
    const b = drawRoi([360, 180], [420, 240]);
    S().setTool("select");
    S().setSelection([a.id, b.id]);
    h.contextMenu(290, 210);
    expect(new Set(S().contextMenu?.targetIds)).toEqual(new Set([a.id, b.id]));
  });

  it("targets the canvas when right-clicking empty space", () => {
    drawRoi([260, 180], [320, 240]);
    S().setTool("select");
    h.contextMenu(700, 500);
    expect(S().contextMenu?.targetIds).toEqual([]);
  });

  it("still targets a moved object at its new position", () => {
    const roi = drawRoi([260, 180], [360, 260]);
    S().setTool("select");
    h.drag([300, 220], [420, 300]);
    S().clearSelection();
    h.contextMenu(420, 300);
    expect(S().contextMenu?.targetIds).toEqual([roi.id]);
  });

  it("deleting through the menu's patch removes the object", () => {
    drawRoi([260, 180], [360, 260]);
    S().setTool("select");
    h.drag([300, 220], [420, 300]);
    h.contextMenu(420, 300);
    // Mirrors what the menu's Delete item does.
    const targets = S().contextMenu!.targetIds.map((id) => S().items.get(id)!).filter((a) => !a.locked);
    S().apply({ label: `Delete ${targets.length}`, removed: targets });
    S().clearSelection();
    expect(count()).toBe(0);
  });
});

describe("refusals are explained", () => {
  it("says so when the only selected object is locked", () => {
    const roi = drawRoi([260, 180], [360, 260]);
    lock(all()[0]);
    S().select([roi.id]);
    h.key("Backspace");
    expect(S().notice).toMatch(/locked/i);
    expect(count()).toBe(1);
  });

  it("reports the count when several locked objects are selected", () => {
    const a = drawRoi([260, 180], [320, 240]);
    const b = drawRoi([360, 180], [420, 240]);
    lock(S().items.get(a.id)!);
    lock(S().items.get(b.id)!);
    S().setSelection([a.id, b.id]);
    h.key("Backspace");
    expect(S().notice).toMatch(/All 2/);
  });

  it("deletes once unlocked, and clears the notice", () => {
    const roi = drawRoi([260, 180], [360, 260]);
    lock(all()[0]);
    S().select([roi.id]);
    h.key("Backspace");
    expect(count()).toBe(1);
    const locked = S().items.get(roi.id)!;
    S().apply({ label: "unlock", updated: [{ before: locked, after: { ...locked, locked: false } }] });
    S().select([roi.id]);
    h.key("Backspace");
    expect(count()).toBe(0);
    expect(S().notice).toBeNull();
  });

  it("stays silent when nothing is selected", () => {
    drawRoi([260, 180], [360, 260]);
    S().clearSelection();
    S().setNotice(null);
    h.key("Backspace");
    expect(S().notice).toBeNull();
  });
});

describe("tissue re-detection", () => {
  it("replaces its own previous output instead of stacking", () => {
    const mk = (x: number) =>
      makeAnnotation(
        { type: "Polygon", coordinates: [[[x, 0], [x + 10, 0], [x + 10, 10], [x, 10], [x, 0]]] },
        { classId: "tissue", source: "model", modelId: "tissue-otsu" },
      );
    S().apply({ label: "Detect tissue (2)", added: [mk(0), mk(50)] });
    expect(count()).toBe(2);
    const previous = all().filter((a) => a.modelId === "tissue-otsu" && !a.locked);
    S().apply({ label: "Re-detect tissue (2)", removed: previous, added: [mk(0), mk(50)] });
    expect(count()).toBe(2);
  });
});
