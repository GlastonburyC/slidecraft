import { describe, expect, it, beforeEach } from "vitest";
import { useAnnotations } from "../annotate/store";
import { resetStore } from "./toolHarness";

const S = () => useAnnotations.getState();

describe("class creation and the active brush", () => {
  beforeEach(() => resetStore());

  it("creating a class by hand selects it to draw in", () => {
    const cls = S().createClass("Tumour");
    expect(S().activeClassId).toBe(cls.id);
  });

  /**
   * Running a detector must not take over the brush.
   *
   * "Detect tissue" needs a Tissue class to exist, and creating it used to make
   * it active. Every cell segmented afterwards was then committed as Tissue —
   * drawn in the tissue colour, on top of the tissue region it sits inside, and
   * therefore invisible. The masks were there; nothing showed them.
   */
  it("a class a detector needed does not take over the brush", () => {
    const mine = S().createClass("Tumour");
    const tissue = S().ensureClass("Tissue");
    expect(tissue.id).not.toBe(mine.id);
    expect(S().activeClassId).toBe(mine.id);
  });

  it("ensureClass reuses an existing class rather than making a second", () => {
    const a = S().ensureClass("Tissue");
    const b = S().ensureClass("tissue");
    expect(b.id).toBe(a.id);
    expect(S().classes.filter((c) => c.name.toLowerCase() === "tissue").length).toBe(1);
  });

  it("still activates when a class is created with no active one", () => {
    expect(S().activeClassId).toBe(null);
    const cls = S().createClass("First");
    expect(S().activeClassId).toBe(cls.id);
  });
});

describe("hiding a class", () => {
  beforeEach(() => resetStore());

  /**
   * Hiding is a view setting, not an edit. The objects must survive it — they
   * are still selectable by other means, still counted, and still exported.
   */
  it("hides and shows without touching the objects", () => {
    const cls = S().createClass("Tissue");
    expect(S().hiddenClasses.has(cls.id)).toBe(false);

    S().toggleClassVisibility(cls.id);
    expect(S().hiddenClasses.has(cls.id)).toBe(true);

    S().toggleClassVisibility(cls.id);
    expect(S().hiddenClasses.has(cls.id)).toBe(false);
  });

  it("hides one class without hiding the others", () => {
    const tissue = S().createClass("Tissue");
    const nuclei = S().createClass("Nuclei");
    S().setClassHidden(tissue.id, true);
    expect(S().hiddenClasses.has(tissue.id)).toBe(true);
    expect(S().hiddenClasses.has(nuclei.id)).toBe(false);
  });

  it("replaces the set rather than mutating it, so subscribers re-render", () => {
    const cls = S().createClass("Tissue");
    const before = S().hiddenClasses;
    S().toggleClassVisibility(cls.id);
    expect(S().hiddenClasses).not.toBe(before);
  });
});
