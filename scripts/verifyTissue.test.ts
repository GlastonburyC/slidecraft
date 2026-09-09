import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { detectTissue } from "../src/ml/tissue";
import type { SlideMeta, SlideSource } from "../src/slide/types";

/** Minimal binary PPM (P6) reader. */
function readPpm(path: string) {
  const buf = readFileSync(path);
  let at = 0;
  const token = () => {
    while (at < buf.length) {
      if (buf[at] === 0x23) { while (buf[at] !== 0x0a) at++; }
      else if (buf[at] === 0x20 || buf[at] === 0x0a || buf[at] === 0x0d || buf[at] === 0x09) at++;
      else break;
    }
    const start = at;
    while (at < buf.length && ![0x20,0x0a,0x0d,0x09].includes(buf[at])) at++;
    return buf.subarray(start, at).toString();
  };
  const magic = token(); if (magic !== "P6") throw new Error("not P6: " + magic);
  const w = +token(), h = +token(); token(); at++;
  return { w, h, rgb: buf.subarray(at, at + w * h * 3) };
}

/**
 * A slide whose only level IS the overview we extracted. That isolates the
 * detector's image reasoning — thresholds, morphology, labelling — from the
 * pyramid arithmetic the synthetic tests already cover, and runs it on what
 * real H&E actually looks like.
 */
function slideFrom(path: string, mpp = 0.5): { source: SlideSource; w: number; h: number } {
  const { w, h, rgb } = readPpm(path);
  const DS = 32;
  const meta: SlideMeta = {
    name: path, bytes: 0, vendor: "test", mppX: mpp, mppY: mpp, objectivePower: 20,
    bounds: null, backgroundColor: null,
    levels: [
      { level: 0, width: w * DS, height: h * DS, downsample: 1 },
      { level: 1, width: w, height: h, downsample: DS },
    ],
    properties: {},
  };
  return {
    w, h,
    source: {
      meta,
      async readRegion(x, y, level, rw, rh) {
        const ds = meta.levels[level].downsample;
        const out = new Uint8ClampedArray(rw * rh * 4);
        for (let j = 0; j < rh; j++) {
          for (let i = 0; i < rw; i++) {
            const sx = Math.min(w - 1, Math.max(0, Math.round((x + i * ds) / DS)));
            const sy = Math.min(h - 1, Math.max(0, Math.round((y + j * ds) / DS)));
            const s = (sy * w + sx) * 3;
            const o = (j * rw + i) * 4;
            out[o] = rgb[s]; out[o + 1] = rgb[s + 1]; out[o + 2] = rgb[s + 2]; out[o + 3] = 255;
          }
        }
        return out;
      },
      async bestLevelForDownsample() { return 1; },
      async close() {},
    },
  };
}

/**
 * Tissue detection over every overview on disk.
 *
 * Not part of `npm test`: it needs real slides, which are not in the repo. Run
 * `scripts/extract-overviews.sh <dir>` first, then `npm run verify:tissue`. It
 * writes a table to stdout and an SVG per slide with the detected polygons
 * drawn over the overview, because the question this answers — does the
 * boundary sit on the tissue — is one you have to look at.
 */
describe("real slides", () => {
  it("detects tissue on every overview on disk", async () => {
    const dir = process.env.SLIDECRAFT_OVERVIEWS ?? "/tmp/slidecraft-overviews";
    const rows: string[] = [];
    for (const f of readdirSync(dir).filter((n) => n.endsWith(".ppm")).sort()) {
      const { source, w, h } = slideFrom(`${dir}/${f}`);
      const r = await detectTissue(source, { minAreaUm2: 20000 });
      const areas = r.polygons.map((rings) => {
        const xs = rings[0].map((p) => p[0]); const ys = rings[0].map((p) => p[1]);
        return ((Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys))) / (w * 32 * h * 32);
      }).sort((a, b) => b - a);
      // Emit an SVG overlay per slide so the result can actually be looked at.
      const palette = ["#ff3b30","#34c759","#0a84ff","#ff9f0a","#bf5af2","#ffd60a","#64d2ff","#ff6482"];
      const paths = r.polygons.map((rings, i) => {
        const d = rings.map((ring) =>
          "M" + ring.map(([x, y]) => `${(x / 32).toFixed(1)},${(y / 32).toFixed(1)}`).join("L") + "Z",
        ).join(" ");
        const c = palette[i % palette.length];
        return `<path d="${d}" fill="${c}" fill-opacity="0.28" stroke="${c}" stroke-width="1.2" fill-rule="evenodd"/>`;
      }).join("\n");
      const png = readFileSync(`${dir}/${f.replace(".ppm", ".png")}`).toString("base64");
      writeFileSync(`/tmp/sc_ov/${f.replace(".ppm", ".svg")}`,
        `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">` +
        `<image href="data:image/png;base64,${png}" width="${w}" height="${h}"/>${paths}</svg>`);

      rows.push([f.replace(".ppm", "").padEnd(34),
        String(r.polygons.length).padStart(3),
        `${(r.coverage * 100).toFixed(1)}%`.padStart(6),
        `thr${String(Math.round(r.saturationThreshold * 255)).padStart(4)}`,
        `rej${String(r.rejected).padStart(3)}`,
        `top ${areas.slice(0, 3).map((a) => (a * 100).toFixed(1) + "%").join(" ")}`,
      ].join(" "));
    }
    const report = rows.join("\n");
    writeFileSync(`${dir}/report.txt`, report);
    console.log(`\n${report}\n\nOverlays and report written to ${dir}`);
    expect(rows.length).toBeGreaterThan(0);
  }, 120000);
});
