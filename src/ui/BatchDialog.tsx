import { useState } from "react";
import type { AnnotationClass } from "../annotate/types";
import {
  canRunBatch, listBatchSlides, pickBatchDirectory, runBatch,
  type BatchProgress, type BatchResult,
} from "../ml/batchTissue";
import { activeModel, TISSUE_CLASS, useTraining } from "../ml/tissueTraining";

/**
 * Run the detector over a folder of slides and write the results beside them.
 *
 * The directory is chosen through the browser's own picker, which is also what
 * grants permission to write into it — the results have to land next to their
 * slides, named after them, or the pairing that makes them load automatically
 * later does not happen.
 */
export function BatchDialog({
  classes, close,
}: {
  classes: AnnotationClass[];
  close: () => void;
}) {
  const [dirName, setDirName] = useState<string | null>(null);
  const [slides, setSlides] = useState<Awaited<ReturnType<typeof listBatchSlides>>>([]);
  const [progress, setProgress] = useState<BatchProgress | null>(null);
  const [results, setResults] = useState<BatchResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [skipExisting, setSkipExisting] = useState(true);
  const [controller, setController] = useState<AbortController | null>(null);
  const setBusy = useTraining((s) => s.setBusy);
  const model = activeModel();

  const supported = canRunBatch();

  const choose = async () => {
    setError(null);
    const dir = await pickBatchDirectory();
    if (!dir) return;
    try {
      const found = await listBatchSlides(dir);
      setDirName(dir.name);
      setSlides(found);
      setResults(null);
      (window as unknown as { __batchDir?: unknown }).__batchDir = dir;
      if (!found.length) setError("No slides in that folder.");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const start = async () => {
    const dir = (window as unknown as { __batchDir?: Parameters<typeof runBatch>[0] }).__batchDir;
    if (!dir || !slides.length) return;
    const ac = new AbortController();
    setController(ac);
    setBusy("batch");
    setError(null);
    try {
      const done = await runBatch(dir, slides, {
        model,
        minAreaUm2: 20000,
        className: TISSUE_CLASS,
        classes,
        skipExisting,
        signal: ac.signal,
        onProgress: setProgress,
      });
      setResults(done);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
      setController(null);
    }
  };

  const failed = results?.filter((r) => r.error) ?? [];
  const written = results?.filter((r) => r.written) ?? [];

  return (
    <div className="modal-backdrop" onClick={close}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">Segment a folder of slides</div>

        {!supported ? (
          <div className="note warn">
            This browser cannot write files back into a folder. Chrome or Edge can; in Safari and
            Firefox, open slides one at a time and export the GeoJSON.
          </div>
        ) : (
          <>
            <div className="picker-hint">
              Each slide gets a <code>.geojson</code> written next to it, named after it. Drop the
              folder back in later and every slide arrives with its annotations already attached.
            </div>

            <div className="row-actions">
              <button className="btn" onClick={() => void choose()} disabled={!!controller}>
                {dirName ? "Choose another folder…" : "Choose folder…"}
              </button>
            </div>

            {dirName && (
              <dl className="kv">
                <dt>Folder</dt>
                <dd>{dirName}</dd>
                <dt>Slides</dt>
                <dd>{slides.length}</dd>
                <dt>Model</dt>
                <dd className={model ? undefined : "muted"}>
                  {model ? model.name : "none — the colour rule will be used"}
                </dd>
              </dl>
            )}

            {dirName && !results && (
              <label className="check">
                <input
                  type="checkbox"
                  checked={skipExisting}
                  onChange={(e) => setSkipExisting(e.target.checked)}
                />
                Skip slides that already have a GeoJSON
              </label>
            )}

            {progress && !results && (
              <div className="progress-wrap">
                <div className="progress">
                  <div style={{ width: `${Math.round((progress.done / Math.max(1, progress.total)) * 100)}%` }} />
                </div>
                <div className="hint">
                  {progress.done} of {progress.total}
                  {progress.current ? ` — ${progress.current}` : ""}
                </div>
              </div>
            )}

            {results && (
              <div className="hint">
                Wrote {written.length} file{written.length === 1 ? "" : "s"}
                {failed.length > 0 && (
                  <>
                    {" "}· <span style={{ color: "var(--warn)" }}>{failed.length} failed</span>
                  </>
                )}
                .
                {failed.slice(0, 4).map((r) => (
                  <div key={r.slide} className="picker-hint">{r.slide}: {r.error}</div>
                ))}
              </div>
            )}

            {error && <div className="note err">{error}</div>}

            <div className="row-actions">
              <button className="btn" onClick={close}>{results ? "Done" : "Cancel"}</button>
              {controller ? (
                <button className="btn danger" onClick={() => controller.abort()}>
                  Stop after this slide
                </button>
              ) : (
                <button
                  className="btn"
                  onClick={() => void start()}
                  disabled={!slides.length || !!results}
                >
                  Run
                </button>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
