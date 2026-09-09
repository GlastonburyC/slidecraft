import { useMemo } from "react";
import { useAnnotations } from "../annotate/store";
import { pickRoi, roiHint } from "../annotate/pickRoi";
import { useMl } from "../ml/mlStore";
import { findModel } from "../ml/registry";

/**
 * Status and actions for click-to-segment.
 *
 * Choosing a model lives in the Models panel above. This panel only reports
 * what is running and how it is performing — a model whose cost is invisible is
 * a model you cannot trust.
 */
export function SegmentPanel({
  onEncode, onEncodeRoi,
}: {
  onEncode: () => void;
  onEncodeRoi: (roiId: string) => void;
}) {
  const {
    models, activeModelId, status, backend, error, progress,
    encoding, encoded, lastDecodeMs, lastScore, prompt, autoCommit, setAutoCommit,
    nonOverlapping, setNonOverlapping, lastEmpty, outsideRoi,
  } = useMl();
  const tool = useAnnotations((s) => s.tool);
  const items = useAnnotations((s) => s.items);
  const version = useAnnotations((s) => s.version);
  const selection = useAnnotations((s) => s.selection);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const picked = useMemo(() => pickRoi(items, selection), [version, selection, items]);
  const roi = picked.roi;
  const spec = findModel(models, activeModelId);
  if (!spec) return null;

  const ready = status === "ready";

  return (
    <section className="section">
      <h2>Click-to-segment</h2>

      {status === "idle" && (
        <div className="hint">
          Pick a model under Models — choosing one loads it.
        </div>
      )}

      {(status === "downloading" || status === "compiling") && (
        <div className="progress-wrap">
          <div className="progress"><div style={{ width: `${Math.round(progress * 100)}%` }} /></div>
          <div className="hint">
            {status === "downloading"
              ? `Downloading ${Math.round(progress * 100)}% — cached after this`
              : "Preparing model…"}
          </div>
        </div>
      )}

      {status === "error" && <div className="note err">{error}</div>}

      {ready && (
        <>
          <dl className="kv">
            <dt>Model</dt>
            <dd>{spec.name}</dd>
            <dt>Runtime</dt>
            <dd title={
              backend === "wasm"
                ? "This model is pinned to WASM: on WebGPU its quantized weights return noise."
                : undefined
            }>{backend === "webgpu" ? "WebGPU" : "WASM"}</dd>
            <dt>Encoded</dt>
            <dd className={encoded ? undefined : "muted"}>
              {encoding
                ? "encoding…"
                : encoded
                  ? `${encoded.origin === "roi" ? "ROI" : "view"} · ${encoded.readW}×${encoded.readH}`
                  : "not yet"}
            </dd>
            {lastDecodeMs !== null && (
              <>
                <dt>Last click</dt>
                <dd>{lastDecodeMs.toFixed(0)} ms · IoU {lastScore?.toFixed(2)}</dd>
              </>
            )}
          </dl>

          <div className="row-actions" style={{ marginTop: 8 }}>
            <button className="btn" onClick={onEncode} disabled={encoding}>
              {encoded?.origin === "view" ? "Re-encode view" : "Encode view"}
            </button>
            <button
              className="btn"
              onClick={() => roi && onEncodeRoi(roi.id)}
              disabled={encoding || !roi}
              title={roiHint(picked.total, roi) ?? "Encode this ROI and work inside it"}
            >
              {encoded?.roiId && roi && encoded.roiId === roi.id ? "Re-encode ROI" : "Encode ROI"}
            </button>
          </div>
          {roiHint(picked.total, roi) && (
            <div className="hint">{roiHint(picked.total, roi)}</div>
          )}
          {picked.implicit && (
            <div className="hint">
              {picked.total} ROIs on this slide — the most recent will be used. Select one to
              choose.
            </div>
          )}
          {encoded?.origin === "roi" && (
            <div className="hint">
              Working inside an ROI — the embedding stays fixed while you pan and zoom.
            </div>
          )}
          {outsideRoi && (
            <div className="hint" style={{ color: "var(--warn)" }}>
              That click was outside the encoded ROI. Encode the view, or click inside it.
            </div>
          )}

          <label className="check">
            <input type="checkbox" checked={autoCommit} onChange={(e) => setAutoCommit(e.target.checked)} />
            Keep each cell when I click the next
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={nonOverlapping}
              onChange={(e) => setNonOverlapping(e.target.checked)}
            />
            Never overlap a cell already segmented
          </label>
          {lastEmpty && (
            <div className="hint" style={{ color: "var(--warn)" }}>
              Nothing left after clipping — that cell is already segmented.
            </div>
          )}

          {tool !== "segment" ? (
            <div className="hint">Press <kbd>G</kbd> to start clicking cells.</div>
          ) : (
            <div className="hint">
              Click a cell centre. <kbd>Shift</kbd>-click adds to it, <kbd>Alt</kbd>-click cuts away.
              <kbd>⌫</kbd> throws away a bad mask — or takes back the one just kept.
              {prompt.length > 0 && ` · ${prompt.length} prompt point${prompt.length === 1 ? "" : "s"}`}
            </div>
          )}
        </>
      )}
    </section>
  );
}
