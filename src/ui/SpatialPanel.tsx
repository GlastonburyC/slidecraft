import { useEffect, useMemo, useState } from "react";
import { useAnnotations } from "../annotate/store";
import { pickRoi, roiHint } from "../annotate/pickRoi";
import { useMl } from "../ml/mlStore";
import { buildPatchGrid } from "../ml/patchGrid";
import { findModel, formatBytes, totalBytes } from "../ml/registry";
import type { SpatialController } from "../ml/spatialController";
import { currentField, legendStops, robustRange, toCsv } from "../ml/spatialResult";
import { parseSignatures, scoreSignature, usableSignatures } from "../ml/signatures";
import {
  differentialExpression, enrichmentCsv, NotEnoughPatches, patchesInside,
  type EnrichmentResult,
} from "../ml/enrichment";
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

  const spatialState = useSpatial();
  const {
    models, activeModelId, status, error, download, progress, result, backend,
    gene, setGene, opacity, setOpacity, visible, setVisible, setActiveModel, setResult,
    onTissueOnly, setOnTissueOnly, setTissueMask,
    mode, setMode, signatures, setSignatures, signatureName, setSignatureName,
  } = spatialState;

  const patchPx = useMl((s) => s.patchPx);
  const [scope, setScope] = useState<"roi" | "tissue">("roi");
  const [query, setQuery] = useState("");
  const [enrichment, setEnrichment] = useState<EnrichmentResult | null>(null);
  const [enrichmentError, setEnrichmentError] = useState<string | null>(null);
  const [regionName, setRegionName] = useState("");

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

  /**
   * Which patches land on tissue, recomputed whenever either side moves.
   *
   * A map carries a patch wherever whoever computed it thought there was
   * tissue. That was a different detector on a different machine, and on a
   * slide with a dark edge or a coverslip line it can be generous. The tissue
   * objects on this slide are the ones you can see and correct, so they get the
   * last word about where a gene is allowed to be read.
   */
  const onTissue = useMemo(() => {
    if (!result) return null;
    if (!tissue.length) return null;
    const inside = patchesInside(result, tissue);
    const mask = new Uint8Array(result.patches.length);
    for (const i of inside) mask[i] = 1;
    return mask;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result, tissue, version]);

  useEffect(() => { setTissueMask(onTissue); }, [onTissue, setTissueMask]);

  const shown = onTissueOnly && onTissue
    ? onTissue.reduce((n, v) => n + v, 0)
    : (result?.patches.length ?? 0);

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

  const field = useMemo(
    () => (result ? currentField(result, spatialState, scoreSignature) : null),
    // The whole state object is the input; React's exhaustive rule cannot see that.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [result, mode, gene, signatureName, signatures],
  );
  const range = useMemo(() => (field ? robustRange(field.values) : null), [field]);

  const selected = useMemo(() => {
    const chosen = [...selection].map((id) => items.get(id)).filter((a): a is NonNullable<typeof a> => !!a);
    return chosen.filter((a) => a.geometry.type === "Polygon" || a.geometry.type === "MultiPolygon");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, selection, items]);

  const selectedName = useMemo(() => {
    const names = new Set(
      selected.map((a) => classes.find((c) => c.id === a.classId)?.name ?? "unclassified"),
    );
    return names.size === 1 ? [...names][0] : "";
  }, [selected, classes]);

  const covered = useMemo(
    () => (result && signatures ? usableSignatures(result, signatures) : []),
    [result, signatures],
  );

  const loadSignatures = async (file: File) => {
    try {
      setSignatures(parseSignatures(JSON.parse(await file.text())));
    } catch (err) {
      window.alert(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <section className="section">
      <h2 className="with-mark">
        <SpatialMark size={16} />
        Virtual spatial
      </h2>

      {models.length === 0 ? (
        <>
          <div className="hint">
            Gene expression predicted from the H&amp;E itself, no assay run on this section.
            DeepSpot-M is the model this was built against. There are two ways to get a map,
            and for a whole slide only the first is practical.
          </div>

          <div className="hint" style={{ marginTop: 10, fontWeight: 600 }}>
            1. Compute it on a GPU and drop the folder in
          </div>
          <div className="picker-hint">
            DeepSpot-M is a 1B-parameter encoder, which is minutes per patch in a browser and
            seconds on a GPU. So run it where the GPU is:
          </div>
          <ol className="steps-hint">
            <li>
              <code>python scripts/predict_expression.py slide.svs --panel ibd-colon</code>
              {" "}— add <code>--submit HOST</code> to send it to Slurm.
            </li>
            <li>
              It writes <code>slide.expression.bin</code> beside the slide. Drop the whole
              folder here and the map loads with it.
            </li>
          </ol>

          <div className="hint" style={{ marginTop: 10, fontWeight: 600 }}>
            2. Import an ONNX export and run it here
          </div>
          <div className="picker-hint">
            For one ROI and a handful of genes, when there is no GPU to reach. Export with{" "}
            <code>scripts/export_deepspot.py</code> after accepting the terms at{" "}
            <code>huggingface.co/ratschlab/DeepSpotM</code> — the weights are gated to academic
            and non-profit use, and Slidecraft cannot accept that licence for you.
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
            {backend && (
              <>
                <dt>Runtime</dt>
                <dd
                  className={backend === "wasm" ? "muted" : undefined}
                  title={
                    backend === "wasm"
                      ? "WebGPU was unavailable, so this ran on WASM — tens of seconds per patch rather than seconds."
                      : undefined
                  }
                >
                  {backend === "webgpu" ? "WebGPU" : "WASM"}
                </dd>
              </>
            )}
          </dl>

          <div className="seg-choice">
            <button className="seg" aria-pressed={mode === "gene"} onClick={() => setMode("gene")}>
              Genes
            </button>
            <button
              className="seg"
              aria-pressed={mode === "signature"}
              onClick={() => setMode("signature")}
              title="Score a cell-type signature instead of reading one gene"
            >
              Signatures
            </button>
          </div>

          {mode === "gene" ? (
            <>
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
                  <button key={g} className="gene-row" aria-current={g === gene} onClick={() => setGene(g)}>
                    <span className="radio" data-on={g === gene} />
                    <span className="gene-name">{g}</span>
                  </button>
                ))}
                {shownGenes.length === 0 && (
                  <div className="picker-hint">Nothing matches “{query}”.</div>
                )}
              </div>
            </>
          ) : !signatures ? (
            <>
              <div className="hint">
                Average a marker set instead of reading one gene — the independent part of the
                per-gene error averages down, and it answers where a cell type is rather than what
                one transcript is doing.
              </div>
              <label className="field">
                <span>Signatures</span>
                <input
                  type="file"
                  accept=".json"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) void loadSignatures(f);
                    e.target.value = "";
                  }}
                />
              </label>
              <div className="picker-hint">
                Derive them from a single-cell atlas with{" "}
                <code>scripts/signatures_from_cellxgene.py --tissue lung</code>.
              </div>
            </>
          ) : (
            <>
              <div className="scroll-list scroll-list--short">
                {covered.map(({ signature, coverage }) => (
                  <button
                    key={signature.name}
                    className="gene-row"
                    aria-current={signature.name === signatureName}
                    onClick={() => setSignatureName(signature.name)}
                    title={`${coverage.found} of ${coverage.total} genes are predicted by this model`}
                  >
                    <span className="radio" data-on={signature.name === signatureName} />
                    <span className="gene-name">{signature.name}</span>
                    <span
                      className="sig-coverage"
                      data-thin={coverage.found < 8}
                    >
                      {coverage.found}/{coverage.total}
                    </span>
                  </button>
                ))}
                {covered.length === 0 && (
                  <div className="picker-hint">
                    None of these signatures share enough genes with this model. Re-export it over
                    the genes in <code>signatures.genes.txt</code>.
                  </div>
                )}
              </div>
              <div className="picker-hint">
                {signatures.source}
                {covered.some((c) => c.coverage.found < 8) &&
                  " · a signature scored on only a handful of its genes is a weak one"}
              </div>
              <button className="mini" onClick={() => setSignatures(null)}>use different signatures</button>
            </>
          )}

          {range && (
            <>
              <div className="legend">
                {legendStops().map((c, i) => (
                  <span key={i} style={{ background: `rgb(${c[0]},${c[1]},${c[2]})` }} />
                ))}
              </div>
              <div className="legend-ends">
                <span>{range.min.toFixed(2)}</span>
                <span>
                  {mode === "signature" ? "score" : "predicted"} {field?.label}
                </span>
                <span>{range.max.toFixed(2)}</span>
              </div>
            </>
          )}

          <label className="check">
            <input type="checkbox" checked={visible} onChange={(e) => setVisible(e.target.checked)} />
            Show the expression map
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={onTissueOnly}
              disabled={!onTissue}
              onChange={(e) => setOnTissueOnly(e.target.checked)}
            />
            Only where tissue was detected
          </label>
          <div className="picker-hint">
            {!onTissue
              ? "No tissue objects on this slide yet — run Detect tissue and this can mask the map to them."
              : `${shown.toLocaleString()} of ${result.patches.length.toLocaleString()} patches shown. `
                + "Hidden patches keep their values; this only changes what is drawn."}
          </div>
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
              onClick={() =>
                saveText(
                  toCsv(result),
                  `${result.slide.replace(/\.[^.]+$/, "")}.expression.csv`,
                  "text/csv",
                )
              }
            >
              Export CSV
            </button>
            <button className="btn" onClick={() => setResult(null)}>Clear</button>
          </div>

          <div className="ctx-sep" />

          {/*
            Which genes stand out in a region you drew. This is the question the
            map exists to serve — the map shows one gene at a time, and this says
            which gene to look at.
          */}
          <div className="model-group-head">Enrichment in a region</div>
          {selected.length === 0 ? (
            <div className="hint">
              Draw round an area — brush, polygon, whatever suits — then select it to compare the
              patches inside it against the rest.
            </div>
          ) : (
            <>
              <div className="hint">
                {selected.length} region{selected.length === 1 ? "" : "s"} selected
                {selectedName ? ` · ${selectedName}` : ""}
              </div>
              <button
                className="btn"
                style={{ width: "100%" }}
                onClick={() => {
                  setEnrichmentError(null);
                  try {
                    const inside = patchesInside(result, selected);
                    setEnrichment(differentialExpression(result, inside));
                    setRegionName(selectedName || "selection");
                  } catch (err) {
                    setEnrichment(null);
                    setEnrichmentError(
                      err instanceof NotEnoughPatches
                        ? err.message
                        : err instanceof Error
                          ? err.message
                          : String(err),
                    );
                  }
                }}
              >
                Which genes are enriched here?
              </button>
            </>
          )}
          {enrichmentError && <div className="note warn">{enrichmentError}</div>}

          {enrichment && (
            <>
              <div className="hint">
                {enrichment.inside} patches inside · {enrichment.outside} outside
              </div>
              <div className="scroll-list scroll-list--short">
                <div className="de-row de-head">
                  <span>gene</span><span>AUC</span><span>diff</span><span>q</span>
                </div>
                {enrichment.genes.slice(0, 60).map((g) => (
                  <button
                    key={g.gene}
                    className="de-row"
                    aria-current={g.gene === gene}
                    onClick={() => setGene(g.gene)}
                    title="Show this gene on the slide"
                  >
                    <span className="gene-name">{g.gene}</span>
                    <span className="de-auc" data-strong={g.auc > 0.7}>{g.auc.toFixed(2)}</span>
                    <span className="de-diff">{g.diff > 0 ? "+" : ""}{g.diff.toFixed(2)}</span>
                    <span className="de-q">{g.q < 0.001 ? "<1e-3" : g.q.toFixed(3)}</span>
                  </button>
                ))}
              </div>
              <div className="picker-hint">
                Ranked by AUC — how separable inside is from outside. The q-values are
                Benjamini-Hochberg but <b>optimistic</b>: neighbouring patches are near-copies, so
                the effective sample size is well below the patch count. Read the AUC, use q only
                to filter noise.
              </div>
              <div className="row-actions">
                <button
                  className="btn"
                  onClick={() =>
                    saveText(
                      enrichmentCsv(enrichment, regionName),
                      `${result.slide.replace(/\.[^.]+$/, "")}.enrichment.csv`,
                      "text/csv",
                    )
                  }
                >
                  Export CSV
                </button>
                <button className="btn" onClick={() => setEnrichment(null)}>Clear</button>
              </div>
            </>
          )}

          <div className="ctx-sep" />
          <div className="picker-hint">
            These values are <b>predicted from morphology</b>, not measured. Treat them as a
            hypothesis to check against an assay, not as one.
            {mode === "signature" &&
              " A signature score is also relative to this slide: it says where a cell type is" +
                " concentrated here, not how much of it there is compared with another slide."}
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
function saveText(text: string, filename: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

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
