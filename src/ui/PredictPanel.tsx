import { useMemo, useState } from "react";
import { pickRoi, roiHint } from "../annotate/pickRoi";
import { useAnnotations } from "../annotate/store";
import { ROI_CLASS_ID } from "../annotate/types";
import { buildPatchGrid } from "../ml/patchGrid";
import type { PredictController } from "../ml/predictController";
import { removeLocalModel } from "../ml/localModels";
import { usePredict } from "../ml/predictStore";
import { findModel, formatBytes, totalBytes } from "../ml/registry";
import type { SlideMeta } from "../slide/types";
import { ImportSpatialDialog } from "./ImportSpatialDialog";

/**
 * The human-in-the-loop prediction panel.
 *
 * Embed once, then label, train, look, disagree, retrain. Only the first step
 * touches pixels — and it is cached, so it is once ever for a given ROI. Every
 * step after is arithmetic on vectors already in memory, which is why the
 * retrain button does not have a progress bar: there is nothing to show.
 *
 * The classes are your ordinary annotation classes. There is no separate
 * labelling mode, because the labelling you would do anyway *is* the training
 * set — the same principle the tissue classifier runs on.
 */

/** Side of a spatial validation block, in level-0 pixels. */
const BLOCK_PX = 2048;

export function PredictPanel({
  meta, controller,
}: {
  meta: SlideMeta;
  controller: PredictController | null;
}) {
  const items = useAnnotations((s) => s.items);
  const version = useAnnotations((s) => s.version);
  const selection = useAnnotations((s) => s.selection);
  const classes = useAnnotations((s) => s.classes);

  const {
    encoders, encoderId, setEncoder, setEncoders, status, error, download, backend,
    embedded, vectors, grid, head, prediction,
    shownClass, setShownClass, opacity, setOpacity, visible, setVisible,
    threshold, setThreshold,
  } = usePredict();

  const [patchPx, setPatchPx] = useState(224);
  const [importing, setImporting] = useState(false);
  const [readTest, setReadTest] = useState<string | null>(null);

  /**
   * Read one patch straight off the slide, bypassing the embedding loop.
   *
   * When a run stalls at "reading the slide", the question is whether the
   * slide layer is wedged or whether something about the way the loop calls it
   * is. That is one line in a console, which is one line too many to ask for
   * mid-investigation — so it is a button.
   */
  const testRead = async () => {
    setReadTest("reading…");
    const started = performance.now();
    try {
      const [x, y] = roi ? [roi.bbox[0], roi.bbox[1]] : [0, 0];
      const result = await Promise.race([
        meta && controller
          ? (async () => {
              await (controller as unknown as { source: { readRegion: (
                x: number, y: number, l: number, w: number, h: number,
              ) => Promise<unknown> } }).source.readRegion(x, y, 0, 224, 224);
              return "ok";
            })()
          : Promise.resolve("no slide"),
        new Promise<string>((r) => setTimeout(() => r("HUNG"), 15000)),
      ]);
      setReadTest(
        result === "ok"
          ? `read 224×224 at (${Math.round(x)}, ${Math.round(y)}) in ${Math.round(performance.now() - started)} ms — the slide layer is fine`
          : result === "HUNG"
            ? `the slide layer is wedged: a plain read at (${Math.round(x)}, ${Math.round(y)}) did not return in 15s`
            : result,
      );
    } catch (err) {
      setReadTest(`read failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const picked = useMemo(() => pickRoi(items, selection), [version, selection, items]);
  const roi = picked.roi;
  const spec = findModel(encoders, encoderId ?? "");

  /**
   * Classes with regions inside the ROI, in the order the class list defines.
   * Only classes you have actually drawn count — an empty class in the list
   * would otherwise make training fail with a confusing message.
   */
  const trainable = useMemo(() => {
    if (!roi) return [];
    const all = [...items.values()];
    return classes
      .filter((c) => c.id !== ROI_CLASS_ID)
      .map((c) => ({
        classId: c.id,
        name: c.name,
        regions: all.filter(
          (a) => a.classId === c.id && (a.geometry.type === "Polygon" || a.geometry.type === "MultiPolygon"),
        ),
      }))
      .filter((c) => c.regions.length > 0);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, items, classes, roi]);

  const busy = status === "loading" || status === "embedding" || status === "training";

  const embed = async () => {
    if (!controller || !spec || !roi) return;
    const level = levelForMpp(meta, spec.targetMpp);
    const downsample = meta.levels[level]?.downsample ?? 1;
    const [minX, minY, maxX, maxY] = roi.bbox;
    const built = buildPatchGrid(
      { x: minX, y: minY, width: maxX - minX, height: maxY - minY },
      downsample,
      meta.mppX,
      { patchPx, level, within: [roi], bounds: meta.bounds },
    );
    await controller.embed(built, spec);
  };

  return (
    <section className="section">
      <h2>Predict</h2>

      {encoders.length === 0 ? (
        <>
          <div className="hint">
            No encoder imported. These turn a patch into a feature vector, which is what the
            head learns from.
          </div>
          <ol className="steps-hint">
            <li>
              <code>python scripts/export_onnx.py MahmoodLab/UNI2-h --preset uni2 --fp16</code>
            </li>
            <li>Import the <code>.onnx</code> and its <code>.onnx.json</code> below.</li>
          </ol>
          <button className="btn" style={{ width: "100%" }} onClick={() => setImporting(true)}>
            Import encoder…
          </button>
        </>
      ) : (
        <>
          {encoders.map((m) => (
            <div key={m.id} className="model-row">
              <button
                className="model-option"
                aria-current={m.id === encoderId}
                onClick={() => setEncoder(m.id)}
                disabled={busy}
              >
                <div className="model-option-head">
                  <span className="radio" data-on={m.id === encoderId} />
                  <span className="model-option-name">{m.name}</span>
                  <span className="model-option-size">{formatBytes(totalBytes(m))}</span>
                </div>
                <div className="model-option-blurb">
                  {m.dim ? `${m.dim}-d` : "unknown width"}
                  {m.targetMpp ? ` · ${m.inputSize}px at ${m.targetMpp} µm/px` : ` · ${m.inputSize}px`}
                </div>
              </button>
              <button
                className="model-remove"
                aria-label={`Remove ${m.name}`}
                title="Remove this encoder and its weights"
                disabled={busy}
                onClick={() => {
                  if (!window.confirm(`Remove "${m.name}" and its weights?`)) return;
                  void removeLocalModel(m.id).then(() =>
                    setEncoders(encoders.filter((x) => x.id !== m.id)),
                  );
                }}
              >
                ×
              </button>
            </div>
          ))}

          <label className="field">
            <span>Patch</span>
            <select value={patchPx} onChange={(e) => setPatchPx(Number(e.target.value))} disabled={busy}>
              {[112, 224, 256, 384].map((n) => (
                <option key={n} value={n}>{n}×{n} px</option>
              ))}
            </select>
          </label>

          {roiHint(picked.total, roi) && <div className="hint">{roiHint(picked.total, roi)}</div>}

          <button
            className="btn"
            style={{ width: "100%" }}
            disabled={busy || !roi || !spec}
            onClick={() => void embed()}
          >
            {status === "loading"
              ? `Loading encoder… ${Math.round(download * 100)}%`
              : status === "embedding"
                ? "Embedding…"
                : vectors
                  ? "Re-embed this ROI"
                  : "Embed this ROI"}
          </button>

          {backend && (
            <div className="hint">
              Running on <b>{backend === "webgpu" ? "WebGPU" : "WASM"}</b>
              {backend === "wasm" && (
                <> — a ViT-H is minutes per patch here. WebGPU was unavailable.</>
              )}
            </div>
          )}

          {embedded && (
            <div className="progress-wrap">
              <div className="progress">
                <div style={{ width: `${Math.round((embedded.done / Math.max(1, embedded.total)) * 100)}%` }} />
              </div>
              <div className="hint">
                {embedded.done} of {embedded.total} patches
                {embedded.stage === "encoding" ? " · encoding…" : " · reading the slide…"}
                {embedded.cached > 0 && ` · ${embedded.cached} cached`}
                {embedded.done > embedded.cached && embedded.ms > 0 && (
                  <> · {(embedded.ms / (embedded.done - embedded.cached)).toFixed(0)} ms each</>
                )}
                {embedded.ms > 5000 && embedded.done === embedded.cached && (
                  <> · {Math.round(embedded.ms / 1000)}s elapsed on the first batch</>
                )}
                {status === "embedding" && (
                  <button className="mini" style={{ marginLeft: 8 }} onClick={() => controller?.cancel()}>
                    stop
                  </button>
                )}
              </div>
            </div>
          )}

          {status === "error" && error && <div className="note err">{error}</div>}
          {status === "embedding" && error && <div className="note warn">{error}</div>}

          {(status === "error" || readTest) && (
            <>
              <button className="btn" style={{ width: "100%" }} onClick={() => void testRead()}>
                Test a single slide read
              </button>
              {readTest && <div className="picker-hint">{readTest}</div>}
            </>
          )}

          <button className="mini" onClick={() => setImporting(true)}>import another encoder</button>
        </>
      )}

      {importing && <ImportSpatialDialog kind="encode" close={() => setImporting(false)} />}

      {vectors && grid && (
        <>
          <div className="ctx-sep" />
          <dl className="kv">
            <dt>Embedded</dt>
            <dd>{grid.patches.length.toLocaleString()} patches</dd>
            {backend && (
              <>
                <dt>Runtime</dt>
                <dd className={backend === "wasm" ? "muted" : undefined}>
                  {backend === "webgpu" ? "WebGPU" : "WASM"}
                </dd>
              </>
            )}
          </dl>

          <div className="model-group-head">Classes to learn</div>
          {trainable.length < 2 ? (
            <div className="hint">
              Draw regions in at least two classes inside this ROI. Whatever you annotate is the
              training set — there is no separate labelling step.
            </div>
          ) : (
            <div className="scroll-list scroll-list--short">
              {trainable.map((c) => (
                <div key={c.classId} className="gene-row" style={{ cursor: "default" }}>
                  <span
                    className="swatch sm"
                    style={{ background: swatch(classes, c.classId) }}
                  />
                  <span className="gene-name">{c.name}</span>
                  <span className="sig-coverage">{c.regions.length}</span>
                </div>
              ))}
            </div>
          )}

          <button
            className="btn"
            style={{ width: "100%" }}
            disabled={busy || trainable.length < 2}
            onClick={() => controller?.train(trainable, BLOCK_PX)}
          >
            {head ? "Retrain" : "Train and predict"}
          </button>
        </>
      )}

      {head && prediction && (
        <>
          <div className="hint">
            Fitted in {Math.round(prediction.ms)} ms on{" "}
            {head.samples.reduce((a, b) => a + b, 0).toLocaleString()} labelled patches.
          </div>

          {head.metrics ? (
            <>
              <dl className="kv">
                <dt>Held-out</dt>
                <dd>
                  {(head.metrics.accuracy * 100).toFixed(0)}% over{" "}
                  {head.metrics.heldOut.toLocaleString()} patches in {head.metrics.blocks} blocks
                </dd>
              </dl>
              <div className="scroll-list scroll-list--short">
                <div className="de-row de-head">
                  <span>class</span><span>P</span><span>R</span><span>F1</span>
                </div>
                {head.classes.map((c, k) => (
                  <div key={c} className="de-row" style={{ cursor: "default" }}>
                    <span className="gene-name">{c}</span>
                    <span className="de-auc">{head.metrics!.precision[k].toFixed(2)}</span>
                    <span className="de-diff">{head.metrics!.recall[k].toFixed(2)}</span>
                    <span className="de-q" data-strong={head.metrics!.f1[k] > 0.7}>
                      {head.metrics!.f1[k].toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
              <div className="picker-hint">
                Measured on whole spatial blocks the head never saw. Neighbouring patches are
                near-copies, so a random split would report a much better number than this and
                mean much less.
              </div>
            </>
          ) : (
            <div className="picker-hint">
              No held-out score: the labels sit in too few places on the slide to hold any back.
              Annotate in a few separate spots to get one.
            </div>
          )}

          <div className="ctx-sep" />
          <div className="model-group-head">Showing</div>
          <div className="seg-choice">
            <button className="seg" aria-pressed={!shownClass} onClick={() => setShownClass(null)}>
              Most likely
            </button>
            {head.classes.map((c) => (
              <button
                key={c}
                className="seg"
                aria-pressed={shownClass === c}
                onClick={() => setShownClass(c)}
              >
                {c}
              </button>
            ))}
          </div>

          <label className="field">
            <span>Confidence</span>
            <input
              type="range" min={0} max={0.95} step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
          </label>
          <div className="picker-hint">
            Below {threshold.toFixed(2)} a patch is left uncoloured rather than shown with
            confidence the head does not have.
          </div>

          <label className="field">
            <span>Opacity</span>
            <input
              type="range" min={0.1} max={1} step={0.05}
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
            />
          </label>
          <label className="check">
            <input type="checkbox" checked={visible} onChange={(e) => setVisible(e.target.checked)} />
            Show the prediction
          </label>

          <div className="picker-hint">
            Disagree with it? Draw over what it got wrong, then <b>Retrain</b> — the embeddings are
            cached, so it refits in milliseconds.
          </div>
        </>
      )}
    </section>
  );
}

function swatch(classes: { id: string; color: [number, number, number] }[], id: string): string {
  const c = classes.find((x) => x.id === id)?.color ?? [150, 150, 160];
  return `rgb(${c[0]},${c[1]},${c[2]})`;
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
