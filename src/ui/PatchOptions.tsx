import { useMemo } from "react";
import { makeAnnotation, useAnnotations } from "../annotate/store";
import { pickRoi, roiHint } from "../annotate/pickRoi";
import { useMl } from "../ml/mlStore";
import { buildPatchGrid } from "../ml/patchGrid";
import {
  PATCH_MODEL_ID, patchReaderSnippet, patchStem, toPatchGeoJSON, toPatchManifest,
} from "../ml/patchExport";
import type { SlideMeta } from "../slide/types";

/**
 * Settings for the patch tool, and what to do with the grid it laid.
 *
 * These were a sidebar tab, which put the act of patching two steps away from
 * the slide: pick a tab, pick an ROI, press a button. Patching is a thing you
 * do TO a piece of tissue you are looking at, so it is a tool now — drag a
 * region and it tiles — and this is the tool's own settings panel, on screen
 * only while the tool is held.
 *
 * Patches are specified in pixels at a pyramid level — 128x128, 256x256 —
 * because that is how an encoder is defined: a ViT sees a fixed pixel tensor.
 * The micron size each patch covers is shown alongside, since that is what
 * decides whether the grid is looking at cells or at architecture, and it
 * changes from scanner to scanner for the same pixel count.
 *
 * The grid is drawn before anything is embedded on purpose. A grid that is too
 * coarse for the question, or sitting half on glass, is obvious in one look and
 * expensive to discover after an encoder has run over all of it.
 */

const SIZES = [64, 128, 224, 256, 384, 512];

/**
 * Patches are committed into their own class, not as ROIs.
 *
 * They are regions that will *carry* a classification — that is the whole point
 * of feeding them to an encoder — and a class is what holds one. An ROI is the
 * working frame you patch *from*, and every ROI action means "the one I am
 * working in", so turning a few thousand patches into ROIs would make that
 * question unanswerable.
 */
const PATCH_CLASS = "Patch";

/** Beyond this, committing patches individually costs more than it returns. */
const MAX_COMMIT = 20000;

function download(text: string, filename: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Revoked on the next tick, so the download has certainly started.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function PatchOptions({ meta }: { meta: SlideMeta }) {
  const items = useAnnotations((s) => s.items);
  const version = useAnnotations((s) => s.version);
  const selection = useAnnotations((s) => s.selection);
  const classes = useAnnotations((s) => s.classes);

  const { grid, patchPx, patchLevel, patchesOnTissueOnly, setGrid, setPatchPx, setPatchLevel,
    setPatchesOnTissueOnly } = useMl();

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const picked = useMemo(() => pickRoi(items, selection), [version, selection, items]);
  const roi = picked.roi;

  const tissue = useMemo(() => {
    const cls = classes.find((c) => c.name.toLowerCase() === "tissue");
    if (!cls) return [];
    return [...items.values()].filter((a) => a.classId === cls.id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, items, classes]);

  const level = meta.levels[Math.min(patchLevel, meta.levels.length - 1)];
  const downsample = level?.downsample ?? 1;
  const sideUm = meta.mppX ? patchPx * downsample * meta.mppX : null;

  const lay = () => {
    if (!roi) return;
    const [minX, minY, maxX, maxY] = roi.bbox;
    const g = buildPatchGrid(
      { x: minX, y: minY, width: maxX - minX, height: maxY - minY },
      downsample,
      meta.mppX,
      {
        patchPx,
        level: level?.level ?? 0,
        // Always clip to the ROI itself, so a lassoed ROI does not get a grid
        // over its bounding box; optionally require tissue underneath as well.
        within: [roi],
        restrictTo: patchesOnTissueOnly && tissue.length > 0 ? tissue : undefined,
        bounds: meta.bounds,
      },
    );
    setGrid({ roiId: roi.id, grid: g });
  };

  const shown = grid && roi && grid.roiId === roi.id ? grid.grid : null;

  return (
    <section className="section patch-options">
      <h2>Patch</h2>
      <div className="picker-hint">
        Drag on the slide to tile a region. These settings apply to the next one
        you draw, and <b>Re-tile</b> applies them to the last.
      </div>

      <label className="field">
        <span>Patch size</span>
        <select
          value={patchPx}
          onChange={(e) => setPatchPx(Number(e.target.value))}
        >
          {SIZES.map((n) => (
            <option key={n} value={n}>{n}×{n} px</option>
          ))}
        </select>
      </label>

      <label className="field">
        <span>Level</span>
        <select value={patchLevel} onChange={(e) => setPatchLevel(Number(e.target.value))}>
          {meta.levels.map((l) => (
            <option key={l.level} value={l.level}>
              {l.level === 0 ? "0 — full resolution" : `${l.level} — ${l.downsample.toFixed(0)}× down`}
            </option>
          ))}
        </select>
      </label>

      <div className="hint">
        {sideUm
          ? `Each patch covers ${sideUm.toFixed(1)} µm across.`
          : "This slide reports no scale, so the physical patch size is unknown."}
      </div>

      <label className="check">
        <input
          type="checkbox"
          checked={patchesOnTissueOnly}
          onChange={(e) => setPatchesOnTissueOnly(e.target.checked)}
          disabled={tissue.length === 0}
        />
        Only where there is tissue
      </label>
      {tissue.length === 0 && (
        <div className="picker-hint">Run tissue detection first to use this.</div>
      )}

      <div className="row-actions">
        <button
          className="btn"
          onClick={lay}
          disabled={!roi}
          title={roiHint(picked.total, roi) ?? "Re-tile this region with the settings above"}
        >
          Re-tile
        </button>
        <button className="btn" onClick={() => setGrid(null)} disabled={!grid}>
          Clear
        </button>
      </div>

      {roiHint(picked.total, roi) && (
        <div className="hint">{roiHint(picked.total, roi)}</div>
      )}
      {picked.implicit && (
        <div className="hint">
          {picked.total} ROIs on this slide — the most recent will be used. Select one to choose.
        </div>
      )}

      {shown && (
        <>
          <dl className="kv">
            <dt>Patches</dt>
            <dd>{shown.patches.length.toLocaleString()} of {shown.cols * shown.rows}</dd>
            <dt>Grid</dt>
            <dd>{shown.cols} × {shown.rows}</dd>
          </dl>

          {/* The grid leaves as coordinates, not pixels: whatever consumes it
              re-reads the slide, so nothing has to be copied out of here. */}
          <button
            className="btn"
            style={{ width: "100%" }}
            disabled={shown.patches.length === 0 || shown.patches.length > MAX_COMMIT}
            title={
              shown.patches.length > MAX_COMMIT
                ? `Too many to make objects (${shown.patches.length.toLocaleString()}). Use a larger patch or a smaller ROI.`
                : "Make each patch a selectable object that can carry a class"
            }
            onClick={() => {
              const store = useAnnotations.getState();
              const cls = store.ensureClass(PATCH_CLASS);
              const side = shown.patchPx * shown.downsample;
              const added = shown.patches.map((p) =>
                makeAnnotation(
                  {
                    type: "Polygon",
                    coordinates: [[
                      [p.x, p.y],
                      [p.x + side, p.y],
                      [p.x + side, p.y + side],
                      [p.x, p.y + side],
                      [p.x, p.y],
                    ]],
                  },
                  {
                    classId: cls.id,
                    source: "model",
                    modelId: PATCH_MODEL_ID,
                    name: `${p.col},${p.row}`,
                  },
                ),
              );
              // Re-running replaces the previous grid's objects rather than
              // laying a second set on top of the first.
              const previous = [...store.items.values()].filter(
                (a) => a.modelId === PATCH_MODEL_ID && !a.locked,
              );
              store.apply({
                label: `Patch objects (${added.length})`,
                removed: previous,
                added,
              });
            }}
          >
            Make {shown.patches.length.toLocaleString()} patch object
            {shown.patches.length === 1 ? "" : "s"}
          </button>
          <div className="picker-hint">
            Each becomes a region in the <b>{PATCH_CLASS}</b> class that you can select, drag by
            its corners, and classify — so a model's answer lands on a patch you were able to
            correct first, and it exports with everything else.
          </div>

          <div className="row-actions">
            <button
              className="btn"
              title="Coordinates plus a snippet that reads them with OpenSlide"
              onClick={() => {
                const manifest = toPatchManifest(shown, meta);
                download(
                  JSON.stringify(manifest, null, 2),
                  `${patchStem(meta.name)}.patches.json`,
                  "application/json",
                );
              }}
            >
              Export coordinates
            </button>
            <button
              className="btn"
              title="The same squares as GeoJSON, to open beside the slide"
              onClick={() =>
                download(
                  JSON.stringify(toPatchGeoJSON(shown, meta)),
                  `${patchStem(meta.name)}.patches.geojson`,
                  "application/geo+json",
                )
              }
            >
              Export GeoJSON
            </button>
          </div>
          <details className="picker-hint">
            <summary>Read these in Python</summary>
            <pre className="snippet">{patchReaderSnippet(toPatchManifest(shown, meta))}</pre>
          </details>
        </>
      )}
      {grid && roi && grid.roiId !== roi.id && (
        <div className="hint">The grid belongs to another ROI. Rebuild it for this one.</div>
      )}
    </section>
  );
}
