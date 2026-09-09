import { useState } from "react";
import { useMl } from "../ml/mlStore";
import { formatBytes, totalBytes, type ModelSpec, type ModelTask } from "../ml/registry";
import { isLocalModel, removeLocalModel } from "../ml/localModels";
import { ImportModelDialog } from "./ImportModelDialog";
import { TokenField } from "./TokenField";

/**
 * Model chooser, grouped by what each model is for.
 *
 * Two different jobs live here and they are not interchangeable: a prompt
 * segmenter turns a click into one cell, an encoder turns a patch into a
 * vector for a classifier to learn from. Listing them together without saying
 * which is which invites picking a segmenter to embed patches with, so the
 * grouping is the point rather than decoration.
 *
 * Each option states its size and what the trade-off buys, because "which
 * model" is really a question about download size and encode time versus how
 * well it finds a faint nucleus.
 */

const GROUPS: { task: ModelTask | "encode"; title: string; blurb: string }[] = [
  {
    task: "prompt-segment",
    title: "Click to segment",
    blurb: "Point at a cell and get its outline. The encoder runs once per view or ROI.",
  },
  {
    task: "encode",
    title: "Patch encoders",
    blurb:
      "Foundation models that turn each patch into a feature vector. Import one as ONNX — " +
      "scripts/export_onnx.py converts a gated model with your own token.",
  },
];

export function ModelPanel({
  onPrefetchAll, onLoad,
}: {
  onPrefetchAll: () => void;
  /** Selecting a model should also make it the running one. */
  onLoad: () => void;
}) {
  const models = useMl((s) => s.models);
  const activeModelId = useMl((s) => s.activeModelId);
  const activeEncoderId = useMl((s) => s.activeEncoderId);
  const status = useMl((s) => s.status);
  const cachedIds = useMl((s) => s.cachedIds);
  const prefetch = useMl((s) => s.prefetch);
  const persisted = useMl((s) => s.persisted);
  const setActiveModel = useMl((s) => s.setActiveModel);
  const setActiveEncoder = useMl((s) => s.setActiveEncoder);
  const [importing, setImporting] = useState(false);

  const busy = status === "downloading" || status === "compiling" || prefetch !== null;
  const downloadable = models.filter((m) => !isLocalModel(m));
  const allCached = downloadable.every((m) => cachedIds.includes(m.id));
  const remaining = downloadable
    .filter((m) => !cachedIds.includes(m.id))
    .reduce((n, m) => n + totalBytes(m), 0);

  const choose = (m: ModelSpec) => {
    if (m.task === "encode") {
      // An encoder is a choice about how patches get embedded; it is not loaded
      // until there are patches to embed, so nothing is fetched on selection.
      setActiveEncoder(activeEncoderId === m.id ? null : m.id);
      return;
    }
    if (m.id !== activeModelId) setActiveModel(m.id);
    // Picking a model is the request to use it; downloading it in a second
    // place was the redundancy.
    queueMicrotask(onLoad);
  };

  return (
    <section className="section">
      <h2>Models</h2>

      {GROUPS.map((group) => {
        const inGroup = models.filter((m) =>
          group.task === "encode" ? m.task === "encode" : m.task !== "encode",
        );
        return (
          <div key={group.task} className="model-group">
            <div className="model-group-head">{group.title}</div>
            {inGroup.length === 0 ? (
              <div className="picker-hint">{group.blurb}</div>
            ) : (
              inGroup.map((m) => {
                const active = m.task === "encode" ? m.id === activeEncoderId : m.id === activeModelId;
                return (
                  <button
                    key={m.id}
                    className="model-option"
                    aria-current={active}
                    disabled={busy && !active}
                    onClick={() => choose(m)}
                  >
                    <div className="model-option-head">
                      <span className="radio" data-on={active} />
                      <span className="model-option-name">{m.name}</span>
                      {isLocalModel(m) && (
                        <span className="cached-badge" title="Imported from disk">local</span>
                      )}
                      {active && m.task !== "encode" && status === "ready" ? (
                        <span className="cached-badge" title="Loaded and running">running</span>
                      ) : cachedIds.includes(m.id) ? (
                        <span className="cached-badge" title="Downloaded; click to load">cached</span>
                      ) : (
                        <span className="model-option-size">{formatBytes(totalBytes(m))}</span>
                      )}
                    </div>
                    <div className="model-option-blurb">{m.blurb}</div>
                  </button>
                );
              })
            )}
          </div>
        );
      })}

      <TokenField />

      <div className="row-actions">
        <button className="btn" onClick={() => setImporting(true)}>Import ONNX…</button>
        {models.some(isLocalModel) && (
          <button
            className="btn danger"
            onClick={() => {
              const local = models.filter(isLocalModel);
              const victim = local.find((m) => m.id === activeModelId) ?? local[0];
              if (!victim) return;
              if (!window.confirm(`Remove "${victim.name}" and its weights?`)) return;
              void removeLocalModel(victim.id).then(() => {
                const ml = useMl.getState();
                ml.setModels(ml.models.filter((m) => m.id !== victim.id));
              });
            }}
          >
            Remove imported
          </button>
        )}
      </div>

      {prefetch ? (
        <div className="progress-wrap">
          <div className="progress">
            <div style={{ width: `${Math.round((prefetch.done / prefetch.total) * 100)}%` }} />
          </div>
          <div className="hint">
            Fetching {prefetch.modelName} — {formatBytes(prefetch.done)} of {formatBytes(prefetch.total)}
          </div>
        </div>
      ) : allCached ? (
        <div className="picker-hint">
          All models cached{persisted ? " and protected from eviction" : ""}. Switching is instant.
        </div>
      ) : (
        <>
          <button className="btn" style={{ width: "100%" }} onClick={onPrefetchAll} disabled={busy}>
            Download all ({formatBytes(remaining)})
          </button>
          <div className="picker-hint">
            Fetched once, then kept on disk — no re-download on reload.
          </div>
        </>
      )}

      {importing && <ImportModelDialog close={() => setImporting(false)} />}
    </section>
  );
}
