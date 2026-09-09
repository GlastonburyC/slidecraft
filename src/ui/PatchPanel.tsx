import { useMemo } from "react";
import { useAnnotations } from "../annotate/store";
import { pickRoi, roiHint } from "../annotate/pickRoi";
import { useMl } from "../ml/mlStore";
import { buildPatchGrid } from "../ml/patchGrid";
import type { SlideMeta } from "../slide/types";

/**
 * Laying a patch grid over an ROI.
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

export function PatchPanel({ meta }: { meta: SlideMeta }) {
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
        // over its bounding box; optionally clip to the tissue as well.
        within: patchesOnTissueOnly && tissue.length > 0 ? [roi, ...tissue] : [roi],
        bounds: meta.bounds,
      },
    );
    setGrid({ roiId: roi.id, grid: g });
  };

  const shown = grid && roi && grid.roiId === roi.id ? grid.grid : null;

  return (
    <section className="section">
      <h2>Patches</h2>

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
          title={roiHint(picked.total, roi) ?? "Lay the grid over this ROI"}
        >
          {shown ? "Rebuild grid" : "Lay grid over ROI"}
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
        <dl className="kv">
          <dt>Patches</dt>
          <dd>{shown.patches.length.toLocaleString()} of {shown.cols * shown.rows}</dd>
          <dt>Grid</dt>
          <dd>{shown.cols} × {shown.rows}</dd>
        </dl>
      )}
      {grid && roi && grid.roiId !== roi.id && (
        <div className="hint">The grid belongs to another ROI. Rebuild it for this one.</div>
      )}
    </section>
  );
}
