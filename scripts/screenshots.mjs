/**
 * Capture the screenshots the README and the site use, from the running app.
 *
 * Real captures, not mock-ups: this drives the dev server with an actual slide
 * and an actual expression map, so a picture that shows something the app
 * cannot do would fail here rather than mislead a reader.
 *
 *   npm run dev                 # in another shell, it must be serving
 *   node scripts/screenshots.mjs
 *
 * The fixtures live in public/_shot and are gitignored -- they are a crop of a
 * real resection and its predicted expression, which are not ours to publish.
 * Point SHOT_SLIDE/SHOT_EXPR at your own to regenerate.
 */
import { chromium } from "playwright";
import { mkdir, rm } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const run = promisify(execFile);

const BASE = process.env.SHOT_URL ?? "http://localhost:5199/";
const SLIDE = process.env.SHOT_SLIDE ?? "0143_E_X_mucosa.tif";
const EXPR = process.env.SHOT_EXPR ?? "0143_E_X_mucosa.expression.bin";
const OUT = "docs/shots";

// 2x, so the images stay sharp on the displays people actually read them on.
const VIEWPORT = { width: 1440, height: 860 };
/** Width the site serves them at. */
const WIDTH = 1800;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  await mkdir(OUT, { recursive: true });
  const browser = await chromium.launch({
    args: [
      // openslide-wasm needs SharedArrayBuffer, which needs cross-origin
      // isolation; the dev server sends the headers, this lets them apply.
      "--enable-features=SharedArrayBuffer",
    ],
  });
  const page = await browser.newPage({
    viewport: VIEWPORT,
    deviceScaleFactor: 2,
    colorScheme: "dark",
  });
  page.on("console", (m) => { if (m.type() === "error") console.log("  console:", m.text()); });

  console.log(`Opening ${BASE} …`);
  await page.goto(BASE, { waitUntil: "networkidle" });

  const isolated = await page.evaluate(() => globalThis.crossOriginIsolated);
  console.log(`cross-origin isolated: ${isolated}`);
  if (!isolated) throw new Error("not cross-origin isolated — the slide will not open");

  console.log(`Loading ${SLIDE} …`);
  await page.evaluate(async ([slide, expr]) => {
    const input = document.querySelector('[data-testid="file-input"]');
    const dt = new DataTransfer();
    for (const name of [slide, expr]) {
      const r = await fetch("/_shot/" + name);
      if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
      dt.items.add(new File([await r.blob()], name));
    }
    input.files = dt.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }, [SLIDE, EXPR]);

  await page.waitForFunction(
    () => /Level 0/.test(document.body.innerText) || /µm\/px/.test(document.body.innerText),
    { timeout: 60_000 },
  );
  await sleep(6000); // let the pyramid settle so no tile is mid-fade

  // By text content, not by accessible name: each tab button carries a title
  // with its hint appended, and the icon sits inside the same element.
  const tab = async (name) => {
    await page.evaluate((want) => {
      const b = [...document.querySelectorAll("button")]
        .find((x) => (x.textContent || "").trim().replace(/\d+$/, "") === want);
      if (!b) throw new Error("no tab named " + want);
      b.click();
    }, name);
  };
  /**
   * Capture, then downscale and encode to JPEG.
   *
   * The raw 2x PNG is around 5 MB and the site needs four of them; at 1800px
   * wide and Q92 the same picture is about 600 KB and the UI text is still
   * sharp. Done here rather than by hand afterwards, so `npm run screenshots`
   * produces exactly what the site serves.
   *
   * vips is the same dependency the tissue verification already needs. Without
   * it the PNG is kept, which is correct but heavy — the warning says so
   * rather than failing the run.
   */
  const shot = async (name) => {
    const png = `${OUT}/${name}.png`;
    await page.screenshot({ path: png });
    try {
      await run("vips", ["thumbnail", png, `${OUT}/.tmp.v`, String(WIDTH)]);
      await run("vips", ["jpegsave", `${OUT}/.tmp.v`, `${OUT}/${name}.jpg`,
                         "--Q", "92", "--strip"]);
      await rm(`${OUT}/.tmp.v`, { force: true });
      await rm(png, { force: true });
      console.log(`  wrote ${OUT}/${name}.jpg`);
    } catch {
      console.log(`  wrote ${OUT}/${name}.png (vips not found; not resized)`);
    }
  };

  /** Draw a rectangle over a fraction of the viewer, and select it. */
  const drawBox = async (x0f, y0f, x1f, y1f, key = "r") => {
    await page.keyboard.press(key);
    await sleep(400);
    const b = await page.locator(".anno-input").boundingBox();
    const x0 = b.x + b.width * x0f, y0 = b.y + b.height * y0f;
    const x1 = b.x + b.width * x1f, y1 = b.y + b.height * y1f;
    await page.mouse.move(x0, y0);
    await page.mouse.down();
    await page.mouse.move(x1, y1, { steps: 26 });
    await page.mouse.up();
    await sleep(1200);
    return { x0, y0, x1, y1 };
  };

  /**
   * Scroll the sidebar until the text is in view.
   *
   * These panels are long, and an analysis puts its answer below the controls
   * that started it — so a capture taken where the panel happens to be sitting
   * shows the buttons and none of the result.
   */
  const reveal = (re) =>
    page.evaluate((src) => {
      const rx = new RegExp(src);
      const pane = document.querySelector(".sidebar-scroll");
      const hit = [...pane.querySelectorAll("div, h2, button")]
        .reverse()
        .find((el) => rx.test(el.textContent || ""));
      if (hit) hit.scrollIntoView({ block: "center" });
    }, re.source);

  /** Click a button by its visible text, which is stabler than its role name. */
  const clickText = (re) =>
    page.evaluate((src) => {
      const rx = new RegExp(src);
      const b = [...document.querySelectorAll("button")]
        .find((x) => rx.test((x.textContent || "").trim()));
      if (!b) throw new Error("no button matching " + src);
      b.click();
    }, re.source);

  // ---- 1. the expression map, which is the thing people come for -----------
  await tab("Virtual ST");
  await sleep(1200);
  await page.evaluate(() => {
    const row = [...document.querySelectorAll(".gene-row")].find((b) =>
      /Goblet/.test(b.textContent || ""));
    row?.click();
    const f = [...document.querySelectorAll(".field")].find((x) => /Opacity/.test(x.innerText || ""));
    const r = f?.querySelector('input[type=range]');
    if (r) {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(r, "0.5");
      r.dispatchEvent(new Event("input", { bubbles: true }));
    }
  });
  await sleep(2500);
  await shot("virtual-st");

  // ---- 2. the same tissue, read as a different module ----------------------
  await page.evaluate(() => {
    const row = [...document.querySelectorAll(".gene-row")].find((b) =>
      /Stroma/.test(b.textContent || ""));
    row?.click();
  });
  await sleep(2000);
  await shot("virtual-st-stroma");

  /*
   * The remaining two are analyses rather than tools, so each needs its result
   * on screen before the capture — a picture of an empty panel would show the
   * buttons and none of the point.
   *
   * The regions are drawn where this particular crop has its mucosa: lower
   * left, running up to the wall at the top right. Point SHOT_SLIDE somewhere
   * else and these fractions stop meaning anything, which is the cost of
   * capturing a real result rather than a mock.
   */

  // ---- 3. which cell types are in a region --------------------------------
  await page.evaluate(() => {
    const store = window.__store;
    if (store) store.getState().apply({ label: "clear", removed: [...store.getState().items.values()] });
  });
  await sleep(600);
  await drawBox(0.10, 0.55, 0.45, 0.95);
  await page.keyboard.press("v");
  await sleep(300);
  const b = await page.locator(".anno-input").boundingBox();
  await page.mouse.click(b.x + b.width * 0.27, b.y + b.height * 0.75);
  await sleep(800);
  await tab("Virtual ST");
  await sleep(900);
  await clickText(/Which cell types\?/);
  await page.waitForFunction(() => /patches inside/.test(document.body.innerText), { timeout: 60_000 });
  await sleep(1200);
  await reveal(/patches inside/);
  await sleep(600);
  await shot("enrichment");

  // ---- 4. what changes along an axis --------------------------------------
  await page.evaluate(() => {
    const store = window.__store;
    if (store) store.getState().apply({ label: "clear", removed: [...store.getState().items.values()] });
  });
  await sleep(600);
  await drawBox(0.22, 0.86, 0.72, 0.18, "a");
  // Deselect, so the axis shows its own colour rather than the selection white.
  await page.keyboard.press("v");
  await sleep(200);
  await page.mouse.click(b.x + b.width * 0.93, b.y + b.height * 0.06);
  await sleep(700);
  await tab("Virtual ST");
  await sleep(900);
  await clickText(/Which cell types change\?/);
  await page.waitForFunction(() => /patches along the axis/.test(document.body.innerText), { timeout: 60_000 });
  await sleep(1200);
  await reveal(/patches along the axis/);
  await sleep(600);
  await shot("axis");

  // ---- 5. tissue detection ------------------------------------------------
  await tab("Tissue");
  await sleep(800);
  const detect = page.getByRole("button", { name: /Detect tissue/i }).first();
  if (await detect.count()) {
    await detect.click();
    await sleep(9000);
  }
  await shot("tissue");

  // ---- 6. the patch tool, mid-grid ----------------------------------------
  await page.keyboard.press("t");
  await sleep(600);
  const box = await page.locator(".anno-input").boundingBox();
  if (box) {
    const x = box.x + box.width * 0.34, y = box.y + box.height * 0.3;
    await page.mouse.move(x, y);
    await page.mouse.down();
    await page.mouse.move(x + box.width * 0.36, y + box.height * 0.42, { steps: 24 });
    await page.mouse.up();
    await sleep(3500);
  }
  await shot("patch-tool");

  await browser.close();
  console.log("done");
}

main().catch((err) => { console.error(err); process.exit(1); });
