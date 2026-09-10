import { isFrame } from "./resize";
import { ROI_CLASS_ID, type Annotation } from "./types";

/**
 * Which ROI an action applies to.
 *
 * A selected ROI wins, because selecting one is the user saying which. With
 * nothing selected the most recently drawn one is used, rather than refusing:
 * the previous rule needed *exactly* one ROI on the slide, so drawing a second
 * silently greyed out every ROI action with no visible reason — the buttons
 * simply stopped working and nothing said why.
 *
 * Callers show the returned name, so "the most recent one" is a statement on
 * screen rather than a guess the user has to reverse-engineer.
 */
export function pickRoi(
  items: Map<string, Annotation>,
  selection: Set<string>,
): { roi: Annotation | null; total: number; implicit: boolean } {
  /**
   * A single selected frame wins even when it is a patch.
   *
   * Selecting one patch and asking to encode it is a coherent request — it is a
   * window you picked — so it is honoured. What patches never do is join the
   * pool the fallback draws from: with a few thousand of them, "the most recent
   * ROI" would always be a patch and the ROI you actually drew would be
   * unreachable.
   */
  const selectedFrames = [...selection]
    .map((id) => items.get(id))
    .filter((a): a is Annotation => !!a && isFrame(a));
  const rois = [...items.values()].filter((a) => a.classId === ROI_CLASS_ID);
  if (selectedFrames.length === 1) {
    return { roi: selectedFrames[0], total: rois.length, implicit: false };
  }
  if (rois.length === 0) return { roi: null, total: 0, implicit: false };

  const selected = rois.filter((a) => selection.has(a.id));
  if (selected.length === 1) return { roi: selected[0], total: rois.length, implicit: false };
  // More than one selected is genuinely ambiguous; ask rather than pick.
  if (selected.length > 1) return { roi: null, total: rois.length, implicit: false };

  return { roi: rois[rois.length - 1], total: rois.length, implicit: rois.length > 1 };
}

/** Why an ROI action is unavailable, in words the panel can show. */
export function roiHint(total: number, roi: Annotation | null): string | null {
  if (roi) return null;
  if (total === 0) return "Draw an ROI first — press O and drag.";
  return "Several ROIs are selected. Select just the one you want.";
}
