import { describe, expect, it } from "vitest";
import { resolveSlides } from "../slide/dropResolver";
import { mergeSamples } from "../ml/tissueLabels";
import type { ResolvedFile } from "../slide/types";
import { FEATURE_COUNT } from "../ml/tissueFeatures";

const file = (path: string, body = "x"): ResolvedFile =>
  ({ path, file: new File([body], path.slice(path.lastIndexOf("/") + 1)) });

describe("annotation sidecars", () => {
  it("pairs a GeoJSON named after the slide", () => {
    const [slide] = resolveSlides([
      file("run/A1.svs"),
      file("run/A1.geojson", "{}"),
      file("run/A2.svs"),
    ]).filter((s) => s.name === "A1.svs");
    expect(slide.annotations).not.toBe(null);
    expect(slide.annotations!.name).toBe("A1.geojson");
  });

  it("leaves a slide with no sidecar alone", () => {
    const slides = resolveSlides([file("run/A2.svs"), file("run/A1.geojson", "{}")]);
    expect(slides.find((s) => s.name === "A2.svs")!.annotations).toBe(null);
  });

  /** A batch writes one file per slide; they must not cross-pair. */
  it("gives each slide its own sidecar", () => {
    const slides = resolveSlides([
      file("run/A1.svs"), file("run/A1.geojson", "{}"),
      file("run/A2.ndpi"), file("run/A2.geojson", "{}"),
      file("run/A3.svs"),
    ]);
    const named = Object.fromEntries(slides.map((s) => [s.name, s.annotations?.name ?? null]));
    expect(named).toEqual({
      "A1.svs": "A1.geojson",
      "A2.ndpi": "A2.geojson",
      "A3.svs": null,
    });
  });

  it("does not mistake another slide's name for a sidecar", () => {
    // "A1_extra.geojson" starts with the stem but is not named after it.
    const slides = resolveSlides([file("run/A1.svs"), file("run/A1_extra.geojson", "{}")]);
    expect(slides[0].annotations).toBe(null);
  });

  it("prefers the explicit slidecraft sidecar over a bare one", () => {
    const slides = resolveSlides([
      file("run/A1.svs"),
      file("run/A1.geojson", "{}"),
      file("run/A1.slidecraft.geojson", "{}"),
    ]);
    expect(slides[0].annotations!.name).toBe("A1.slidecraft.geojson");
  });

  it("still pairs a sidecar for a MIRAX slide with its data folder", () => {
    const slides = resolveSlides([
      file("run/M1.mrxs"),
      file("run/M1/Slidedat.ini"),
      file("run/M1/Data0000.dat"),
      file("run/M1.geojson", "{}"),
    ]);
    expect(slides[0].warning).toBe(null);
    expect(slides[0].annotations!.name).toBe("M1.geojson");
  });
});

describe("labels from several slides", () => {
  const part = (slide: string, groups: number[], labels: number[]) => ({
    slide,
    report: {
      set: {
        x: new Float32Array(groups.length * FEATURE_COUNT),
        y: Uint8Array.from(labels),
        group: Int32Array.from(groups),
        count: groups.length,
      },
      regions: { tissue: 1, artefact: 1 },
      skipped: 0,
    },
  });

  it("renumbers regions so two slides cannot share one", () => {
    const merged = mergeSamples([
      part("a", [0, 0, 1, 1], [1, 1, 0, 0]),
      part("b", [0, 1, 2], [1, 0, 0]),
    ]);
    expect(merged.count).toBe(7);
    expect([...merged.group]).toEqual([0, 0, 1, 1, 2, 3, 4]);
    expect([...merged.y]).toEqual([1, 1, 0, 0, 1, 0, 0]);
  });

  it("keeps every slide's samples", () => {
    const merged = mergeSamples([part("a", [0], [1]), part("b", [0], [0]), part("c", [0], [1])]);
    expect(merged.count).toBe(3);
    expect(new Set(merged.group).size).toBe(3);
  });
});
