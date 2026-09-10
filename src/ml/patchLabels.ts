import { containsPoint, isAreaGeometry, type Annotation } from "../annotate/types";
import type { Patch } from "./patchGrid";

/**
 * Turning annotations into patch labels.
 *
 * A patch takes the class of the region its centre falls in. Testing one point
 * keeps this linear in patches rather than in vertices, and a patch that is
 * mostly one thing is that thing — the alternative, area-weighted assignment,
 * costs a polygon intersection per patch per class and changes the answer only
 * for patches straddling a boundary, which are ambiguous anyway.
 *
 * Patches inside no labelled region are unlabelled, not background. Treating
 * unlabelled tissue as a negative is the classic way to teach a head that
 * everything you have not looked at is "normal".
 */

export interface LabelledPatch {
  patch: Patch;
  /** Index into the class list. */
  label: number;
}

export function labelPatches(
  patches: Patch[],
  regionsByClass: { classId: string; regions: Annotation[] }[],
): { labelled: LabelledPatch[]; classIds: string[]; perClass: number[] } {
  const classIds = regionsByClass.map((r) => r.classId);
  const areas = regionsByClass.map((r) => r.regions.filter((a) => isAreaGeometry(a.geometry)));
  const perClass = new Array<number>(classIds.length).fill(0);
  const labelled: LabelledPatch[] = [];

  for (const patch of patches) {
    const cx = patch.x + patch.size / 2;
    const cy = patch.y + patch.size / 2;

    // First match wins, so class order is the tie-break for overlapping
    // regions. Deterministic, and the panel lists classes in that same order.
    for (let k = 0; k < areas.length; k++) {
      if (areas[k].some((a) => containsPoint(a.geometry, cx, cy))) {
        labelled.push({ patch, label: k });
        perClass[k]++;
        break;
      }
    }
  }

  return { labelled, classIds, perClass };
}
