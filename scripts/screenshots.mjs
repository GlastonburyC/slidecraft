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
import { mkdir } from "node:fs/promises";

const BASE = process.env.SHOT_URL ?? "http://localhost:5199/";
const SLIDE = process.env.SHOT_SLIDE ?? "0143_E_X_mucosa.tif";
const EXPR = process.env.SHOT_EXPR ?? "0143_E_X_mucosa.expression.bin";
const OUT = "docs/shots";

// 2x, so the images stay sharp on the displays people actually read them on.
const VIEWPORT = { width: 1440, height: 860 };

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
  const shot = async (file) => {
    await page.screenshot({ path: `${OUT}/${file}` });
    console.log(`  wrote ${OUT}/${file}`);
  };

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
  await shot("virtual-st.png");

  // ---- 2. the same tissue, read as a different module ----------------------
  await page.evaluate(() => {
    const row = [...document.querySelectorAll(".gene-row")].find((b) =>
      /Stroma/.test(b.textContent || ""));
    row?.click();
  });
  await sleep(2000);
  await shot("virtual-st-stroma.png");

  // ---- 3. tissue detection ------------------------------------------------
  await tab("Tissue");
  await sleep(800);
  const detect = page.getByRole("button", { name: /Detect tissue/i }).first();
  if (await detect.count()) {
    await detect.click();
    await sleep(9000);
  }
  await shot("tissue.png");

  // ---- 4. the patch tool, mid-grid ----------------------------------------
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
  await shot("patch-tool.png");

  await browser.close();
  console.log("done");
}

main().catch((err) => { console.error(err); process.exit(1); });
