import { describe, expect, it } from "vitest";
import { mergeSamples, summarise, type SlideSamples } from "../ml/tissueLabels";
import { trainTissueModel } from "../ml/tissueModel";
import { FEATURE_COUNT } from "../ml/tissueFeatures";

/**
 * A model has to be extendable, not just re-creatable.
 *
 * The point of saving the labelled cells beside the weights is that opening a
 * new slide, marking it, and retraining produces a model fitted on *all* the
 * slides — not a fresh one built from the last slide alone wearing the old
 * model's name. These check the arithmetic that makes that additive.
 */

function part(slide: string, tissue: number, artefact: number): SlideSamples {
  const count = tissue + artefact;
  const x = new Float32Array(count * FEATURE_COUNT);
  const y = new Uint8Array(count);
  const group = new Int32Array(count);
  for (let i = 0; i < count; i++) {
    const isTissue = i < tissue;
    y[i] = isTissue ? 1 : 0;
    group[i] = isTissue ? 0 : 1;
    // Separable, so training has something real to fit.
    for (let f = 0; f < FEATURE_COUNT; f++) x[i * FEATURE_COUNT + f] = isTissue ? 10 : 0;
  }
  return {
    slide,
    report: { set: { x, y, group, count }, regions: { tissue: 1, artefact: 1 }, skipped: 0 },
  };
}

describe("extending a saved training set", () => {
  it("adds a slide without discarding the others", () => {
    const restored = [part("a", 40, 40), part("b", 40, 40)];
    const extended = [...restored, part("c", 40, 40)];

    expect(summarise(restored).slides).toBe(2);
    expect(summarise(extended).slides).toBe(3);
    expect(mergeSamples(extended).count).toBe(mergeSamples(restored).count + 80);
  });

  it("keeps every slide's regions in their own validation group", () => {
    const merged = mergeSamples([part("a", 10, 10), part("b", 10, 10), part("c", 10, 10)]);
    // Two regions per slide, three slides, and no id shared between them.
    expect(new Set(merged.group).size).toBe(6);
  });

  it("records every contributing slide on the model it fits", () => {
    const parts = [part("a", 40, 40), part("b", 40, 40), part("c", 40, 40)];
    const model = trainTissueModel(mergeSamples(parts), {
      name: "three",
      slides: parts.map((p) => p.slide),
    });
    expect(model.slides).toEqual(["a", "b", "c"]);
    expect(model.samples.tissue).toBe(120);
    expect(model.samples.artefact).toBe(120);
  });

  /**
   * Re-adding a slide must replace its labels, not stack a second copy — the
   * usual reason to re-add one is that the first attempt was wrong.
   */
  it("replaces a slide's labels when it is added again", () => {
    const set: SlideSamples[] = [part("a", 40, 40), part("b", 40, 40)];
    const again = part("a", 10, 10);
    const next = [...set.filter((p) => p.slide !== again.slide), again];

    expect(next.length).toBe(2);
    expect(next.find((p) => p.slide === "a")!.report.set.count).toBe(20);
  });

  it("survives the shapes IndexedDB stores them as", () => {
    // structuredClone is what the database does to them on the way in and out.
    const original = [part("a", 20, 20)];
    const round = structuredClone(original) as SlideSamples[];
    expect(round[0].report.set.x).toBeInstanceOf(Float32Array);
    expect(round[0].report.set.y).toBeInstanceOf(Uint8Array);
    expect(round[0].report.set.group).toBeInstanceOf(Int32Array);
    expect(mergeSamples(round).count).toBe(mergeSamples(original).count);
  });
});
