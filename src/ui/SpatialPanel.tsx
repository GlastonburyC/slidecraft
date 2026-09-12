import { useEffect, useMemo, useState } from "react";
import { makeAnnotation, useAnnotations } from "../annotate/store";
import { closeRing } from "../annotate/geometry";
import { pickRoi, roiHint } from "../annotate/pickRoi";
import { useMl } from "../ml/mlStore";
import { buildPatchGrid } from "../ml/patchGrid";
import { findModel, formatBytes, totalBytes } from "../ml/registry";
import type { SpatialController } from "../ml/spatialController";
import { currentField, legendStops, robustRange, toCsv } from "../ml/spatialResult";
import { parseSignatures, scoreSignature, usableSignatures } from "../ml/signatures";
import {
  differentialExpression, differentialSignatures, enrichmentCsv, patchesInside,
  type EnrichmentResult,
} from "../ml/enrichment";
import {
  axesIn, geneGradient, gradientCsv, signatureGradient, type GradientResult,
} from "../ml/gradient";
import { matrixBytes, toAnnDataZip } from "../io/anndata";
import {
  latticeOf, moduleFromEnrichment, moduleFromGradient, percentileThreshold, similarRegions,
} from "../ml/findSimilar";
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
    onTissueOnly, setOnTissueOnly, setTissueMask, addSignature,
    mode, setMode, signatures, setSignatures, signatureName, setSignatureName, attaching,
  } = spatialState;

  const patchPx = useMl((s) => s.patchPx);
  const [scope, setScope] = useState<"roi" | "tissue">("roi");
  const [query, setQuery] = useState("");
  const [enrichment, setEnrichment] = useState<EnrichmentResult | null>(null);
  const [enrichmentError, setEnrichmentError] = useState<string | null>(null);
  const [enrichmentBusy, setEnrichmentBusy] = useState<string | null>(null);
  const [regionName, setRegionName] = useState("");
  const [gradient, setGradient] = useState<GradientResult | null>(null);
  const [gradientError, setGradientError] = useState<string | null>(null);
  const [gradientBusy, setGradientBusy] = useState<string | null>(null);
  /**
   * Corridor half-width, in patches rather than pixels.
   *
   * The arrow says where to look; this says how wide a band around it counts.
   * Expressed in patches because that is the unit the answer is computed in —
   * "three patches either side" states how much averaging is happening, where
   * a figure in microns hides it.
   */
  const [corridor, setCorridor] = useState(3);
  /**
   * Floor on a gene's mean, below which it is not reported.
   *
   * Both analyses rank by a scale-free statistic, so a gene the model predicts
   * at 0.001 everywhere can top either one on the ordering of noise. On a whole
   * transcriptome roughly half the genes sit below 0.05, and the markers that
   * matter — COL1A1 at 2.3, EPCAM at 0.69, CD3D at 0.096 — are well clear of
   * it, while the ones known to be dead here (AQP8 0.007, PYY 0.004) are not.
   */
  const [minExpression, setMinExpression] = useState(0.05);

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


  /**
   * Compare the selected region against the rest of the slide.
   *
   * Both questions share everything except the column being tested, so they
   * share the error handling too — a region too small to test is too small for
   * either of them.
   */
  const compareRegion = (kind: "gene" | "signature") => {
    if (!result || enrichmentBusy) return;
    setEnrichmentError(null);
    const n = kind === "signature" ? signatures.signatures.length : result.genes.length;
    setEnrichmentBusy(
      `Comparing ${n.toLocaleString()} ${kind === "signature" ? "cell types" : "genes"}…`,
    );

    // Painted first, then the work: a whole transcriptome takes seconds, and a
    // button that does not come back reads as broken rather than busy.
    requestAnimationFrame(() => {
      setTimeout(() => {
        try {
          const inside = patchesInside(result, selected);
          setEnrichment(
            kind === "signature"
              ? differentialSignatures(result, inside, signatures.signatures, scoreSignature)
              : differentialExpression(result, inside, 5, minExpression),
          );
          setRegionName(selectedName || "selection");
        } catch (err) {
          setEnrichment(null);
          setEnrichmentError(err instanceof Error ? err.message : String(err));
        } finally {
          setEnrichmentBusy(null);
        }
      }, 0);
    });
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const axes = useMemo(() => axesIn(items.values()), [version, items]);
  // The selected axis, or the last one drawn — the same rule ROIs follow.
  const axis = useMemo(() => {
    const picked = axes.filter((a) => selection.has(a.id));
    return picked.length ? picked[picked.length - 1] : (axes[axes.length - 1] ?? null);
  }, [axes, selection]);

  /**
   * Rank what changes along the axis.
   *
   * Ranking a whole transcriptome is seconds of work on the main thread —
   * 19,338 columns, each sorted to get its ranks — and a button that does not
   * come back for six seconds reads as broken rather than busy. So the busy
   * state is painted first and the work starts on the next frame, which is the
   * difference between a frozen window and a visibly working one.
   */
  const runGradient = (kind: "gene" | "signature") => {
    if (!result || !axis || gradientBusy) return;
    setGradientError(null);
    const n = kind === "signature" ? signatures.signatures.length : result.genes.length;
    setGradientBusy(
      `Ranking ${n.toLocaleString()} ${kind === "signature" ? "cell types" : "genes"}…`,
    );

    requestAnimationFrame(() => {
      setTimeout(() => {
        try {
          const width = corridor * result.side;
          setGradient(
            kind === "signature"
              ? signatureGradient(
                  result, axis, width, meta.mppX, signatures.signatures, scoreSignature)
              : geneGradient(result, axis, width, meta.mppX, minExpression),
          );
        } catch (err) {
          setGradient(null);
          setGradientError(err instanceof Error ? err.message : String(err));
        } finally {
          setGradientBusy(null);
        }
      }, 0);
    });
  };

  /*
   * A whole transcriptome is 32,000 patches by 19,338 genes, which is 2.5 GB
   * once it is widened to the float32 AnnData stores — several times what a tab
   * can put in one array, let alone zip. The button says so and points at the
   * script rather than offering an export that would take the page down.
   */
  const [findPercentile, setFindPercentile] = useState(95);
  const [found, setFound] = useState<string | null>(null);

  /**
   * Save what is in this region as a module, then colour the slide by it.
   *
   * The ranked list the enrichment produced is already a description of the
   * thing you drew. Scoring it everywhere says where else that description
   * fits — no model trained, nothing labelled, and the answer is a field you
   * can look at before deciding whether it is worth pursuing.
   */
  /**
   * Draw the gradient instead of reading it.
   *
   * Each gene weighted by its own rho, so the module scores high where the
   * genes that rise toward the arrowhead are high and the ones that fall are
   * low — which is the gradient itself, as a field. Over the whole slide, not
   * just the corridor, so you can see whether the same trend holds away from
   * where it was measured.
   */
  const saveGradientAsModule = () => {
    if (!gradient || gradient.kind !== "gene") return;
    addSignature(moduleFromGradient("Along the axis", gradient.items));
    setFound(null);
  };

  const saveAsModule = () => {
    if (!enrichment) return;
    /*
     * Named after the region's class, when it has one.
     *
     * An unclassified rectangle would otherwise produce a module called
     * "unclassified", which is the name of nothing and collides with the next
     * one. Classify the region first and the module inherits that name; until
     * then it gets a neutral one that at least says where it came from.
     */
    const base = regionName?.trim();
    const meaningful = base && base !== "unclassified" && base !== "selection";
    addSignature(moduleFromEnrichment(meaningful ? base : "Like this region", enrichment.genes));
    setFound(null);
  };

  /**
   * Turn the field on screen into candidate objects.
   *
   * Thresholded at a percentile rather than a value: a module's score is
   * standardised per slide, so 0.8 means something different on each one,
   * while "the top 5% of this slide" asks the same question everywhere.
   */
  const findSimilar = () => {
    if (!result || !field) return;
    const lattice = latticeOf(result);
    const cut = percentileThreshold(field.values, findPercentile);
    const regions = similarRegions(field.values, lattice, { threshold: cut, minPatches: 4 });
    if (!regions.length) {
      setFound("Nothing cleared that threshold. Lower it, or pick a different module.");
      return;
    }
    const store = useAnnotations.getState();
    const cls = store.ensureClass(field.label);
    // Marked as model output, so they read as provisional until judged — and so
    // a second pass replaces them rather than laying a new set on top.
    const previous = [...store.items.values()].filter(
      (a) => a.classId === cls.id && a.source === "model" && !a.locked,
    );
    const added = regions.map((r) =>
      makeAnnotation({ type: "Polygon", coordinates: [closeRing(r.ring)] }, {
        classId: cls.id,
        source: "model",
        modelId: `similar:${field.label}`,
      }),
    );
    store.apply({ label: `Similar to ${field.label} (${added.length})`, removed: previous, added });
    setFound(`${added.length} region${added.length === 1 ? "" : "s"} above the top ${100 - findPercentile}%.`);
  };

  const annDataMB = result ? matrixBytes(result) / 1e6 : 0;
  const tooBigForAnnData = annDataMB > 600;

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
  /**
   * Show the best-covered module as soon as a map arrives.
   *
   * The alternative is an expression map that loads and draws nothing until
   * someone picks a gene out of a list of nineteen thousand — and the first
   * gene anyone picks reads as static, because one predicted gene carries all
   * of its own error. Opening on a module means the first thing seen is the
   * thing worth seeing. Only ever fills a blank: any choice already made is
   * left alone.
   */
  useEffect(() => {
    if (!result || signatureName) return;
    const best = covered[0];
    // Only when the module is actually carried by this map. Loading a result
    // always selects a gene, so the choice here is between one arbitrary gene
    // and a module -- and a module scored on three of its twenty genes is the
    // worse of the two. Eight is the same bar the panel calls thin below.
    if (best && best.coverage.found >= 8) setSignatureName(best.signature.name);
  }, [result, signatureName, covered, setSignatureName]);

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

      {/*
        * A map that is already loaded outranks the absence of a model.
        *
        * These two are independent: a precomputed .expression.bin arrives with
        * the slide and needs no model imported at all. Gating the panel on the
        * model list alone drew the map on the slide and then offered
        * instructions for obtaining one instead of the controls for the map
        * that was already there -- visible, and with no way to change the gene.
        */}
      {attaching ? (
        /*
         * A map on its way is not the absence of one. Reading a whole
         * transcriptome off disk takes seconds, and offering instructions for
         * obtaining a model during that window tells the user the opposite of
         * what is happening.
         */
        <div className="hint notice">
          <span className="spinner" /> Reading {attaching}…
        </div>
      ) : models.length === 0 && !result ? (
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
                A module averages its genes after standardising each one, so an abundant gene
                cannot drown the rest — which is why a module reads as tissue architecture where
                a single gene reads as static.
              </div>
              <div className="picker-hint">
                {signatures.source}
                {covered.some((c) => c.coverage.found < 8) &&
                  " · a module scored on only a handful of its genes is a weak one"}
              </div>
              <label className="field">
                <span>Load more</span>
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
                Derive weighted ones from a single-cell atlas with{" "}
                <code>scripts/signatures_from_cellxgene.py --tissue colon</code>.
              </div>
              {signatures.source !== "Slidecraft built-in modules" && (
                <button className="mini" onClick={() => setSignatures(null)}>
                  back to the built-in modules
                </button>
              )}
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

          {/*
            * One floor, two analyses.
            *
            * Both rank by a scale-free statistic, so both are vulnerable to the
            * same thing: a gene that is not expressed anywhere separating
            * perfectly on noise. The control sits with the map rather than
            * inside either analysis because it describes the map.
            */}
          <label className="field">
            <span>Min. expression</span>
            <select
              value={minExpression}
              onChange={(e) => setMinExpression(Number(e.target.value))}
            >
              <option value={0}>no floor — every gene</option>
              <option value={0.01}>0.01 — barely detected</option>
              <option value={0.02}>0.02</option>
              <option value={0.05}>0.05 — default</option>
              <option value={0.1}>0.1</option>
              <option value={0.25}>0.25 — well expressed only</option>
            </select>
          </label>
          <div className="picker-hint">
            {minExpression > 0 ? (
              <>
                Genes averaging below {minExpression} across this slide are left out of
                both gene analyses. They rank as well as anything when the test is
                rank-based, and mean nothing.
              </>
            ) : (
              <>
                Every gene is reported, including ones the model predicts at
                effectively zero — which will rank as highly as real markers,
                because the test compares orderings rather than amounts.
              </>
            )}
          </div>

          {/*
            * Turn whatever is on screen into objects.
            *
            * Placed with the field rather than with the enrichment, because it
            * applies to any module — a built-in one, a cell type from the
            * Census, or one just derived from a region you drew.
            */}
          {field && (
            <>
              <label className="field">
                <span>Top</span>
                <input
                  type="range"
                  min={80}
                  max={99}
                  step={1}
                  value={findPercentile}
                  onChange={(e) => { setFindPercentile(Number(e.target.value)); setFound(null); }}
                />
              </label>
              <div className="row-actions">
                <button className="btn" onClick={findSimilar}>
                  Find the top {100 - findPercentile}% as objects
                </button>
              </div>
              <div className="picker-hint">
                {found ?? (
                  <>
                    Traces where <b>{field.label}</b> is in the top{" "}
                    {100 - findPercentile}% of this slide, as regions you can select,
                    edit and classify. A percentile rather than a value, because a
                    module&rsquo;s score is standardised per slide.
                  </>
                )}
              </div>
              <div className="ctx-sep" />
            </>
          )}

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
            <button
              className="btn"
              disabled={tooBigForAnnData}
              title={
                tooBigForAnnData
                  ? `${(annDataMB).toFixed(0)} MB dense — too large to assemble in a tab. `
                    + "Convert the .expression.bin with scripts/expression_to_anndata.py instead."
                  : "AnnData as zipped zarr, for scanpy, squidpy or SpatialData"
              }
              onClick={() => {
                const zip = toAnnDataZip(result, { tissueMask: onTissueOnly ? onTissue : null });
                saveBytes(
                  zip,
                  `${result.slide.replace(/\.[^.]+$/, "")}.anndata.zarr.zip`,
                  "application/zip",
                );
              }}
            >
              Export AnnData
            </button>
            <button className="btn" onClick={() => setResult(null)}>Clear</button>
          </div>
          <div className="picker-hint">
            {tooBigForAnnData ? (
              <>
                AnnData would be {annDataMB.toFixed(0)} MB dense — more than a tab should
                assemble. Convert the <code>.expression.bin</code> with{" "}
                <code>scripts/expression_to_anndata.py</code>, which streams it.
              </>
            ) : (
              <>
                AnnData is written as zipped zarr — unzip it and{" "}
                <code>anndata.read_zarr</code> opens it, with patch centres in{" "}
                <code>obsm[&quot;spatial&quot;]</code> and the model it came from in{" "}
                <code>uns</code>. CSV is a row per patch, for anything that reads a table.
              </>
            )}
          </div>

          <div className="ctx-sep" />

          {/*
            Which genes stand out in a region you drew. This is the question the
            map exists to serve — the map shows one gene at a time, and this says
            which gene to look at.
          */}
          {/*
            * A gradient answers what enrichment cannot.
            *
            * Enrichment asks whether a region differs from the rest, which
            * suits a thing with a boundary. Much of mucosa has none: expression
            * varies ALONG an axis — crypt base to surface, mucosa to
            * muscularis — and splitting that into inside and outside discards
            * the ordering, which was the signal.
            */}
          <div className="model-group-head">Change along an axis</div>
          {axes.length === 0 ? (
            <div className="hint">
              Press <kbd>A</kbd> and drag an arrow across the tissue — crypt base to surface, say
              — and this ranks what rises and falls along it.
            </div>
          ) : (
            <>
              <div className="hint">
                {axes.length === 1
                  ? "1 axis on this slide"
                  : `${axes.length} axes · ${axes.some((a) => selection.has(a.id))
                      ? "using the selected one"
                      : "using the most recent; select one to choose"}`}
                {axis && meta.mppX
                  ? ` · ${(Math.hypot(
                      axis.geometry.type === "LineString"
                        ? axis.geometry.coordinates[axis.geometry.coordinates.length - 1][0]
                          - axis.geometry.coordinates[0][0] : 0,
                      axis.geometry.type === "LineString"
                        ? axis.geometry.coordinates[axis.geometry.coordinates.length - 1][1]
                          - axis.geometry.coordinates[0][1] : 0,
                    ) * meta.mppX / 1000).toFixed(2)} mm long`
                  : ""}
              </div>
              <label className="field">
                <span>Corridor</span>
                <input
                  type="range"
                  min={1}
                  max={12}
                  step={1}
                  value={corridor}
                  onChange={(e) => setCorridor(Number(e.target.value))}
                />
              </label>
              <div className="picker-hint">
                {corridor} patch{corridor === 1 ? "" : "es"} either side of the arrow
                {meta.mppX && result
                  ? ` · ${Math.round(corridor * result.side * meta.mppX)} µm`
                  : ""}
                . Patches outside the band, or beyond either end, are not counted.
              </div>
              <div className="row-actions">
                <button
                  className="btn"
                  disabled={!!gradientBusy}
                  onClick={() => runGradient("signature")}
                >
                  Which cell types change?
                </button>
                <button
                  className="btn"
                  disabled={!!gradientBusy}
                  onClick={() => runGradient("gene")}
                >
                  Which genes?
                </button>
              </div>
            </>
          )}
          {gradientBusy && (
            <div className="hint notice">
              <span className="spinner" /> {gradientBusy}
            </div>
          )}
          {gradientError && <div className="note warn">{gradientError}</div>}

          {gradient && (
            <>
              <div className="hint">
                {gradient.used} patches along the axis
                {gradient.lengthUm ? ` · ${(gradient.lengthUm / 1000).toFixed(2)} mm` : ""}
              </div>
              <div className="scroll-list scroll-list--short">
                <div className="de-row de-head">
                  <span>{gradient.kind === "signature" ? "cell type" : "gene"}</span>
                  <span>rho</span><span>start→end</span><span>q</span>
                </div>
                {gradient.items.slice(0, 60).map((g) => (
                  <button
                    key={g.name}
                    className="de-row"
                    onClick={() =>
                      gradient.kind === "signature" ? setSignatureName(g.name) : setGene(g.name)
                    }
                    title="Show this on the slide"
                  >
                    <span className="gene-name">{g.name}</span>
                    <span className="de-auc" data-strong={Math.abs(g.rho) > 0.5}>
                      {g.rho > 0 ? "+" : ""}{g.rho.toFixed(2)}
                    </span>
                    <span className="de-diff">
                      {g.meanStart.toFixed(1)}→{g.meanEnd.toFixed(1)}
                    </span>
                    <span className="de-q">{g.q < 0.001 ? "<1e-3" : g.q.toFixed(3)}</span>
                  </button>
                ))}
              </div>
              <div className="picker-hint">
                Spearman against position along the arrow: <b>positive rises toward the
                head</b>. Reverse the arrow and every sign flips. The q-values carry the same
                caveat as the region test and then some — patches along one axis are neighbours,
                so they are about as far from independent as patches get.
              </div>
              <div className="row-actions">
                <button
                  className="btn"
                  disabled={gradient.kind !== "gene"}
                  title={
                    gradient.kind === "gene"
                      ? "Weight each gene by its rho and colour the slide by the result"
                      : "A module is a set of genes. Ask “Which genes?” along this axis instead."
                  }
                  onClick={saveGradientAsModule}
                >
                  Plot it as a module
                </button>
              </div>
              <div className="picker-hint">
                {gradient.kind === "gene" ? (
                  <>
                    Each gene weighted by its own rho, so the map shows the gradient itself —
                    high where the risers are high and the fallers are low. Drawn over the whole
                    slide, not just the corridor, so you can see whether it holds away from the
                    arrow.
                  </>
                ) : (
                  <>
                    A module is a set of <i>genes</i>, and this ranks cell types. Ask
                    <b> Which genes?</b> along the same axis and plot that.
                  </>
                )}
              </div>

              <div className="row-actions">
                <button
                  className="btn"
                  onClick={() =>
                    saveText(
                      gradientCsv(gradient, "axis"),
                      `${result.slide.replace(/\.[^.]+$/, "")}.gradient.csv`,
                      "text/csv",
                    )
                  }
                >
                  Export CSV
                </button>
                <button className="btn" onClick={() => setGradient(null)}>Clear</button>
              </div>
            </>
          )}

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
              {/*
                * Two questions of the same region, and the cell-type one is
                * usually the one being asked. "CXCL13 is enriched here" needs
                * you to know already what CXCL13 means; "this is a lymphoid
                * aggregate" does not.
                */}
              <div className="row-actions">
                <button
                  className="btn"
                  disabled={!!enrichmentBusy}
                  onClick={() => compareRegion("signature")}
                  title="Rank the cell-type modules by how well they separate this region from the rest"
                >
                  Which cell types?
                </button>
                <button
                  className="btn"
                  disabled={!!enrichmentBusy}
                  onClick={() => compareRegion("gene")}
                >
                  Which genes?
                </button>
              </div>
              {enrichmentBusy && (
                <div className="hint notice">
                  <span className="spinner" /> {enrichmentBusy}
                </div>
              )}
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
                  <span>{enrichment.kind === "signature" ? "cell type" : "gene"}</span>
                  <span>AUC</span><span>diff</span><span>q</span>
                </div>
                {enrichment.genes.slice(0, 60).map((g) => (
                  <button
                    key={g.gene}
                    className="de-row"
                    aria-current={
                      enrichment.kind === "signature" ? g.gene === signatureName : g.gene === gene
                    }
                    // A row draws whatever it ranked. Calling setGene for a cell
                    // type would look for a gene named "plasma cell" and quietly
                    // change nothing.
                    onClick={() =>
                      enrichment.kind === "signature" ? setSignatureName(g.gene) : setGene(g.gene)
                    }
                    title={
                      enrichment.kind === "signature"
                        ? "Show this cell type on the slide"
                        : "Show this gene on the slide"
                    }
                  >
                    <span className="gene-name">{g.gene}</span>
                    <span className="de-auc" data-strong={g.auc > 0.7}>{g.auc.toFixed(2)}</span>
                    <span className="de-diff">{g.diff > 0 ? "+" : ""}{g.diff.toFixed(2)}</span>
                    <span className="de-q">{g.q < 0.001 ? "<1e-3" : g.q.toFixed(3)}</span>
                  </button>
                ))}
              </div>
              {enrichment.kind === "signature" && (
                <div className="picker-hint">
                  Differences are in standard deviations, not the model&rsquo;s units — a module
                  standardises each gene before averaging. And these modules share genes, so the
                  comparisons are correlated with each other as well as across patches.
                </div>
              )}
              <div className="picker-hint">
                Ranked by AUC — how separable inside is from outside. The q-values are
                Benjamini-Hochberg but <b>optimistic</b>: neighbouring patches are near-copies, so
                the effective sample size is well below the patch count. Read the AUC, use q only
                to filter noise.
              </div>
              {/*
                * From one example to the others.
                *
                * The ranked list is already a description of what you drew, so
                * saving it as a module and scoring it everywhere says where
                * else that description fits — the candidates a training loop
                * wants, with nothing trained yet.
                */}
              <div className="row-actions">
                <button
                  className="btn"
                  disabled={enrichment.kind !== "gene"}
                  title={
                    enrichment.kind === "gene"
                      ? "Keep these genes as a module, and score it over the whole slide"
                      : "A module is a set of genes. Ask “Which genes?” of this region instead."
                  }
                  onClick={saveAsModule}
                >
                  Save as a module
                </button>
              </div>
              <div className="picker-hint">
                {enrichment.kind === "gene" ? (
                  <>
                    Keeps the genes that separate this region either way — depleted
                    describes it as well as enriched — and colours the slide by how
                    well the rest of it fits.
                  </>
                ) : (
                  <>
                    A module is a set of <i>genes</i>, so this ranking of cell types
                    cannot become one — the names in it are modules, not genes. Ask
                    <b> Which genes?</b> of the same region and save that.
                  </>
                )}
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
/** Same as saveText, but for bytes that are already assembled. */
function saveBytes(bytes: Uint8Array, filename: string, type: string) {
  // Copied into a fresh buffer: the view may be over a larger allocation, and
  // Blob would otherwise take the whole thing.
  const url = URL.createObjectURL(new Blob([bytes.slice()], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

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
