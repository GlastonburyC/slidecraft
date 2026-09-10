import { useMemo, useState } from "react";
import { useAnnotations } from "../annotate/store";
import { pickRoi, roiHint } from "../annotate/pickRoi";
import { useMl } from "../ml/mlStore";
import { buildPatchGrid } from "../ml/patchGrid";
import { findModel, formatBytes, totalBytes } from "../ml/registry";
import type { SpatialController } from "../ml/spatialController";
import { legendStops, robustRange, geneValues, toCsv } from "../ml/spatialResult";
import { useSpatial } from "../ml/spatialStore";
import type { SlideMeta } from "../slide/types";
import { SpatialMark } from "./SpatialMark";

/**
 * Virtual spatial transcriptomics: expression predicted from the H&E itself.
 *
 * A model like DeepSpot-M reads a tile and answers with a value per gene, so a
 * patch grid becomes a coarse expression map without any assay having been run
 * on this section. That is a genuinely useful thing and also an easy thing to
 * over-read, which is why the panel says what produced a map and keeps the
 * word *predicted* in front of it.
 *
 * Scope is either the whole tissue or one ROI. The ROI case is the one that
 * gets used: it is quick enough to iterate on, and it is how you check whether
 * the model says anything sensible about your material before spending an hour
 * on a slide.
 */
export function SpatialPanel({
  meta, controller, onImport,
}: {
  meta: SlideMeta;
  controller: SpatialController | null;
  onImport: () => void;
}) {
  const items = useAnnotations((s) => s.items);
  const version = useAnnotations((s) => s.version);
  const selection = useAnnotations((s) => s.selection);
  const classes = useAnnotations((s) => s.classes);

  const {
    models, activeModelId, status, error, download, progress, result,
    gene, setGene, opacity, setOpacity, visible, setVisible, setActiveModel, setResult,
  } = useSpatial();

  const patchPx = useMl((s) => s.patchPx);
  const [scope, setScope] = useState<"roi" | "tissue">("roi");
  const [query, setQuery] = useState("");

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const picked = useMemo(() => pickRoi(items, selection), [version, selection, items]);
  const roi = picked.roi;

  const tissue = useMemo(() => {
    const cls = classes.find((c) => c.name.toLowerCase() === "tissue");
    if (!cls) return [];
    return [...items.values()].filter((a) => a.classId === cls.id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, items, classes]);

  const spec = findModel(models, activeModelId ?? "");
  const busy = status === "running" || status === "loading";

  const run = async () => {
    if (!controller || !spec) return;
    if (status !== "ready") await controller.load(spec);

    // The model decides the scale, so the grid is laid at its own µm/pixel
    // rather than at whatever the patch panel is set to.
    const level = levelForMpp(meta, spec.targetMpp);
    const downsample = meta.levels[level]?.downsample ?? 1;

    const region =
      scope === "roi" && roi
        ? { x: roi.bbox[0], y: roi.bbox[1], width: roi.bbox[2] - roi.bbox[0], height: roi.bbox[3] - roi.bbox[1] }
        : boundsOf(meta);
    if (!region) return;

    const grid = buildPatchGrid(region, downsample, meta.mppX, {
      patchPx: spec.inputSize,
      level,
      within: scope === "roi" && roi ? [roi] : undefined,
      restrictTo: tissue.length ? tissue : undefined,
      bounds: meta.bounds,
    });
    await controller.run(grid, spec, scope === "roi" && roi ? roi.id : null);
  };

  const shownGenes = useMemo(() => {
    const list = result?.genes ?? spec?.genes ?? [];
    const q = query.trim().toUpperCase();
    return q ? list.filter((g) => g.toUpperCase().includes(q)) : list;
  }, [result, spec, query]);

  const range = useMemo(() => {
    if (!result || !gene) return null;
    const v = geneValues(result, gene);
    return v ? robustRange(v) : null;
  }, [result, gene]);

  return (
    <section className="section">
      <h2 className="with-mark">
        <SpatialMark size={16} />
        Virtual spatial
      </h2>

      {models.length === 0 ? (
        <>
          <div className="hint">
            No spatial model imported. These predict gene expression from the H&E itself —
            DeepSpot-M is the one this was built against.
          </div>
          <ol className="steps-hint">
            <li>
              Accept the terms at <code>huggingface.co/ratschlab/DeepSpotM</code> — the weights
              are gated to academic and non-profit use.
            </li>
            <li><code>huggingface-cli login</code></li>
            <li><code>python scripts/export_deepspot.py --genes EPCAM CD3D PTPRC</code></li>
            <li>Import the <code>.onnx</code> and its <code>.onnx.json</code> below.</li>
          </ol>
          <div className="picker-hint">
            The conversion is a one-off: the released weights are PyTorch, and the browser runs
            ONNX. Slidecraft cannot accept the licence for you, so it cannot fetch them for you.
          </div>
          <button className="btn" style={{ width: "100%" }} onClick={onImport}>
            Import model…
          </button>
        </>
      ) : (
        <>
          {models.map((m) => (
            <button
              key={m.id}
              className="model-option"
              aria-current={m.id === activeModelId}
              onClick={() => setActiveModel(m.id)}
            >
              <div className="model-option-head">
                <span className="radio" data-on={m.id === activeModelId} />
                <span className="model-option-name">{m.name}</span>
                <span className="model-option-size">{formatBytes(totalBytes(m))}</span>
              </div>
              <div className="model-option-blurb">
                {m.blurb}
                {m.genes?.length ? ` · ${m.genes.length} genes` : ""}
                {m.targetMpp ? ` · ${m.inputSize}px at ${m.targetMpp} µm/px` : ""}
              </div>
            </button>
          ))}

          <div className="seg-choice">
            <button
              className="seg"
              aria-pressed={scope === "roi"}
              onClick={() => setScope("roi")}
              title="Predict inside one ROI — quick enough to iterate on"
            >
              This ROI
            </button>
            <button
              className="seg"
              aria-pressed={scope === "tissue"}
              onClick={() => setScope("tissue")}
              title="Predict over all detected tissue on the slide"
            >
              Whole slide
            </button>
          </div>

          {scope === "roi" && roiHint(picked.total, roi) && (
            <div className="hint">{roiHint(picked.total, roi)}</div>
          )}
          {scope === "tissue" && tissue.length === 0 && (
            <div className="hint">Run tissue detection first, or this covers the whole slide.</div>
          )}

          <button
            className="btn"
            style={{ width: "100%" }}
            disabled={busy || !spec || (scope === "roi" && !roi)}
            onClick={() => void run()}
          >
            {status === "loading"
              ? `Loading model… ${Math.round(download * 100)}%`
              : status === "running"
                ? "Predicting…"
                : "Predict expression"}
          </button>

          {progress && (
            <div className="progress-wrap">
              <div className="progress">
                <div style={{ width: `${Math.round((progress.done / Math.max(1, progress.total)) * 100)}%` }} />
              </div>
              <div className="hint">
                {progress.done} of {progress.total} patches ·{" "}
                {(progress.ms / Math.max(1, progress.done)).toFixed(0)} ms each
                <button className="mini" style={{ marginLeft: 8 }} onClick={() => controller?.cancel()}>
                  stop
                </button>
              </div>
            </div>
          )}

          {status === "error" && error && <div className="note err">{error}</div>}
        </>
      )}

      {result && (
        <>
          <div className="ctx-sep" />
          <dl className="kv">
            <dt>Predicted</dt>
            <dd>
              {result.patches.length.toLocaleString()} patches · {Math.round(result.ms / 1000)} s
            </dd>
            <dt>Model</dt>
            <dd>{result.modelName}</dd>
          </dl>

          {result.genes.length > 8 && (
            <input
              className="class-edit search"
              type="search"
              placeholder="Find a gene…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          )}
          <div className="scroll-list scroll-list--short">
            {shownGenes.map((g) => (
              <button
                key={g}
                className="gene-row"
                aria-current={g === gene}
                onClick={() => setGene(g)}
              >
                <span className="radio" data-on={g === gene} />
                <span className="gene-name">{g}</span>
              </button>
            ))}
            {shownGenes.length === 0 && (
              <div className="picker-hint">Nothing matches “{query}”.</div>
            )}
          </div>

          {range && (
            <>
              <div className="legend">
                {legendStops().map((c, i) => (
                  <span key={i} style={{ background: `rgb(${c[0]},${c[1]},${c[2]})` }} />
                ))}
              </div>
              <div className="legend-ends">
                <span>{range.min.toFixed(2)}</span>
                <span>predicted {gene}</span>
                <span>{range.max.toFixed(2)}</span>
              </div>
            </>
          )}

          <label className="check">
            <input type="checkbox" checked={visible} onChange={(e) => setVisible(e.target.checked)} />
            Show the expression map
          </label>
          <label className="field">
            <span>Opacity</span>
            <input
              type="range"
              min={0.1}
              max={1}
              step={0.05}
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
            />
          </label>

          <div className="row-actions">
            <button
              className="btn"
              onClick={() => {
                const url = URL.createObjectURL(
                  new Blob([toCsv(result)], { type: "text/csv" }),
                );
                const a = document.createElement("a");
                a.href = url;
                a.download = `${result.slide.replace(/\.[^.]+$/, "")}.expression.csv`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                setTimeout(() => URL.revokeObjectURL(url), 1000);
              }}
            >
              Export CSV
            </button>
            <button className="btn" onClick={() => setResult(null)}>Clear</button>
          </div>

          <div className="picker-hint">
            These values are <b>predicted from morphology</b>, not measured. Treat them as a
            hypothesis to check against an assay, not as one.
          </div>
        </>
      )}

      {patchPx !== 224 && models.length > 0 && (
        <div className="picker-hint">
          The grid here uses the model's own patch size and magnification, not the Patches tab's.
        </div>
      )}
    </section>
  );
}

/** Pyramid level nearest a target µm/pixel; finer wins a tie. */
function levelForMpp(meta: SlideMeta, targetMpp: number | null): number {
  if (!targetMpp || !meta.mppX) return 0;
  let best = 0;
  let bestErr = Infinity;
  for (const l of meta.levels) {
    const err = Math.abs(l.downsample * meta.mppX - targetMpp);
    if (err < bestErr - 1e-9) { bestErr = err; best = l.level; }
  }
  return best;
}

function boundsOf(meta: SlideMeta) {
  const b = meta.bounds;
  if (b) return { x: b.x, y: b.y, width: b.width, height: b.height };
  const l0 = meta.levels[0];
  return l0 ? { x: 0, y: 0, width: l0.width, height: l0.height } : null;
}
