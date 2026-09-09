import { describe, expect, it } from "vitest";
import { pickRoi, roiHint } from "../annotate/pickRoi";
import { makeAnnotation } from "../annotate/store";
import { ROI_CLASS_ID, type Annotation } from "../annotate/types";

const roi = (x: number): Annotation =>
  makeAnnotation(
    { type: "Polygon", coordinates: [[[x, 0], [x + 10, 0], [x + 10, 10], [x, 10], [x, 0]]] },
    { classId: ROI_CLASS_ID },
  );
const other = (): Annotation =>
  makeAnnotation({ type: "Polygon", coordinates: [[[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]]] }, { classId: "tumour" });

const map = (...a: Annotation[]) => new Map(a.map((x) => [x.id, x]));

describe("choosing the ROI an action applies to", () => {
  it("finds the only ROI with nothing selected", () => {
    const a = roi(0);
    const p = pickRoi(map(a, other()), new Set());
    expect(p.roi?.id).toBe(a.id);
    expect(p.implicit).toBe(false);
  });

  it("uses the selected ROI when one is selected", () => {
    const a = roi(0), b = roi(100);
    const p = pickRoi(map(a, b), new Set([a.id]));
    expect(p.roi?.id).toBe(a.id);
  });

  /**
   * The bug this replaces: needing *exactly one* ROI on the slide meant that
   * drawing a second one silently disabled every ROI action, with the buttons
   * simply going grey and nothing saying why.
   */
  it("falls back to the most recent ROI rather than giving up", () => {
    const a = roi(0), b = roi(100);
    const p = pickRoi(map(a, b), new Set());
    expect(p.roi?.id).toBe(b.id);
    expect(p.implicit).toBe(true);
    expect(p.total).toBe(2);
  });

  it("refuses when several ROIs are selected, and says so", () => {
    const a = roi(0), b = roi(100);
    const p = pickRoi(map(a, b), new Set([a.id, b.id]));
    expect(p.roi).toBe(null);
    expect(roiHint(p.total, p.roi)).toMatch(/Select just the one/);
  });

  it("says to draw one when there are none", () => {
    const p = pickRoi(map(other()), new Set());
    expect(p.roi).toBe(null);
    expect(roiHint(p.total, p.roi)).toMatch(/Draw an ROI/);
  });

  it("says nothing when an ROI is available", () => {
    const p = pickRoi(map(roi(0)), new Set());
    expect(roiHint(p.total, p.roi)).toBe(null);
  });
});
