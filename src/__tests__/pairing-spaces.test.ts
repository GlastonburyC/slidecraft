import { describe, expect, it } from "vitest";
import { resolveSlides } from "../slide/dropResolver";
import type { ResolvedFile } from "../slide/types";

const f = (path: string): ResolvedFile =>
  ({ path, file: new File([new Uint8Array(4)], path.split("/").pop()!) });

describe("pairing a slide with its sidecars", () => {
  it("works when the name contains a space", () => {
    // Exactly what a CUH export looks like on disk.
    const [slide] = resolveSlides([
      f("CUH/TB_411 3A.svs"),
      f("CUH/TB_411 3A.expression.bin"),
    ]);
    expect(slide.name).toBe("TB_411 3A.svs");
    expect(slide.expression?.name).toBe("TB_411 3A.expression.bin");
  });

  it("works at the top level, with no folder", () => {
    const [slide] = resolveSlides([
      f("TB_411 3A.svs"),
      f("TB_411 3A.expression.bin"),
    ]);
    expect(slide.expression).not.toBeNull();
  });

  it("pairs a GeoJSON with a spaced name too", () => {
    const [slide] = resolveSlides([
      f("TB_411 3A.svs"),
      f("TB_411 3A.geojson"),
    ]);
    expect(slide.annotations?.name).toBe("TB_411 3A.geojson");
  });

  it("does not pair a map belonging to a different slide", () => {
    const [slide] = resolveSlides([
      f("TB_411 3A.svs"),
      f("TB_411 3B.expression.bin"),
    ]);
    expect(slide.expression).toBeNull();
  });
});
