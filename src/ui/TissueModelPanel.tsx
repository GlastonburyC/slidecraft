import { useEffect, useState } from "react";
import { useAnnotations } from "../annotate/store";
import { useMl } from "../ml/mlStore";
import { ROI_CLASS_ID } from "../annotate/types";
import { readOverviewGrid } from "../ml/tissue";
import { collectSamples, mergeSamples, summarise } from "../ml/tissueLabels";
import { explain, NotEnoughLabels, trainTissueModel } from "../ml/tissueModel";
import {
  deleteTissueModel, loadTissueModels, loadTissueSamples, saveTissueModel,
} from "../ml/tissueModelStore";
import { ARTEFACT_CLASS, TISSUE_CLASS, useTraining } from "../ml/tissueTraining";
import type { SlideSource } from "../slide/types";
import { BatchDialog } from "./BatchDialog";

/**
 * Teaching the detector what is tissue on this slide set.
 *
 * The loop is deliberately the work you would do anyway: run detection, and
 * where it is wrong, reclassify the region instead of deleting it. A wrong
 * detection reclassified as "Not tissue" both fixes the picture and becomes the
 * example that stops it happening again, so correcting and teaching are the
 * same action rather than two.
 */
export function TissueModelPanel({
  source, onDetectTissue,
}: {
  source: SlideSource;
  onDetectTissue: () => void;
}) {
  const classes = useAnnotations((s) => s.classes);
  const tissue = useMl((s) => s.tissue);
  const hiddenClasses = useAnnotations((s) => s.hiddenClasses);
  const toggleClassVisibility = useAnnotations((s) => s.toggleClassVisibility);
  const items = useAnnotations((s) => s.items);
  const version = useAnnotations((s) => s.version);
  const selection = useAnnotations((s) => s.selection);

  const {
    samples, models, staleModels, activeModelId, useModel, busy, notice,
    addSamples, removeSamples, replaceSamples, setModels, addModel, setActiveModel, setUseModel,
    setBusy, setNotice,
  } = useTraining();

  const [name, setName] = useState("");
  const [batching, setBatching] = useState(false);

  useEffect(() => {
    void loadTissueModels().then((s) => setModels(s.usable, s.stale));
  }, [setModels]);

  const tissueClass = classes.find((c) => c.name === TISSUE_CLASS);
  const artefactClass = classes.find((c) => c.name === ARTEFACT_CLASS);
  const all = [...items.values()];
  const tissueRegions = tissueClass ? all.filter((a) => a.classId === tissueClass.id) : [];
  const artefactRegions = artefactClass ? all.filter((a) => a.classId === artefactClass.id) : [];

  const selectable = [...selection].filter((id) => {
    const a = items.get(id);
    return a && a.classId !== ROI_CLASS_ID;
  });

  /** Move the selected regions into a class — the correction *is* the label. */
  const reclassify = (className: string) => {
    const store = useAnnotations.getState();
    const cls = store.ensureClass(className);
    const chosen = selectable.map((id) => store.items.get(id)!).filter((a) => !a.locked);
    if (!chosen.length) {
      setNotice("Select the regions to mark first.");
      return;
    }
    store.apply({
      label: `Mark ${chosen.length} as ${className}`,
      // Recorded as human: the point of the correction is that a person
      // decided it, and it must not be cleared as stale model output.
      updated: chosen.map((a) => ({
        before: a,
        after: { ...a, classId: cls.id, source: "human" as const },
      })),
    });
    setNotice(null);
  };

  const addThisSlide = async () => {
    if (!tissueRegions.length && !artefactRegions.length) {
      setNotice(`Mark some regions as ${TISSUE_CLASS} or ${ARTEFACT_CLASS} first.`);
      return;
    }
    setBusy("sampling");
    setNotice(null);
    try {
      const grid = await readOverviewGrid(source);
      const report = collectSamples(grid, { tissue: tissueRegions, artefact: artefactRegions });
      addSamples({ slide: source.meta.name, report });
    } catch (err) {
      setNotice(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const train = async () => {
    if (!samples.length) {
      setNotice("Add this slide's labels first.");
      return;
    }
    setBusy("training");
    setNotice(null);
    try {
      const merged = mergeSamples(samples);
      const model = trainTissueModel(merged, {
        name: name.trim() || `Tissue ${new Date().toLocaleDateString()}`,
        slides: samples.map((s) => s.slide),
      });
      // The labels go with it, so a later session can add a slide and retrain.
      await saveTissueModel(model, samples);
      addModel(model);
      setUseModel(true);
      setName("");
    } catch (err) {
      setNotice(
        err instanceof NotEnoughLabels
          ? err.message
          : `Training failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      setBusy(null);
    }
  };

  /**
   * Picking a model brings back the labels it was fitted on.
   *
   * Otherwise "add one more slide and retrain" is impossible after a reload:
   * the model remembers which slides it used, but without their labelled cells
   * a retrain would silently be a fresh model built from one slide, wearing the
   * old one's name. Restoring them makes adding a slide additive, which is what
   * training across a set is for.
   */
  const choose = async (id: string) => {
    setActiveModel(id);
    setNotice(null);
    const stored = await loadTissueSamples(id);
    if (!stored) {
      const m = models.find((x) => x.id === id);
      setNotice(
        m && m.slides.length
          ? `This model was saved before labels were kept. Re-mark its slides to extend it.`
          : null,
      );
      return;
    }
    replaceSamples(stored);
  };

  const totals = summarise(samples);
  const active = models.find((m) => m.id === activeModelId) ?? null;

  return (
    <section className="section">
      <h2>Tissue</h2>

      <button className="btn" style={{ width: "100%" }} onClick={onDetectTissue}>
        Detect tissue
      </button>
      {tissue && (
        <div className="hint" style={{ marginTop: 6 }}>
          {tissue.count} region{tissue.count === 1 ? "" : "s"} ·{" "}
          {Math.round(tissue.coverage * 100)}% of the slide · {Math.round(tissue.ms)} ms
          {tissue.modelName ? <> · {tissue.modelName}</> : <> · colour rule</>}
          {tissue.rejected > 0 && (
            <> · {tissue.rejected} thin strip{tissue.rejected === 1 ? "" : "s"} rejected</>
          )}
        </div>
      )}

      {(tissueClass || artefactClass) && (
        <div className="row-actions" style={{ marginTop: 6 }}>
          {/* Tissue regions are large and sit under everything else, so
              dropping them out of the view is how you see the work on top. */}
          {tissueClass && (
            <button
              className="btn"
              aria-pressed={hiddenClasses.has(tissueClass.id)}
              onClick={() => toggleClassVisibility(tissueClass.id)}
            >
              {hiddenClasses.has(tissueClass.id) ? "Show tissue" : "Hide tissue"}
            </button>
          )}
          {artefactClass && (
            <button
              className="btn"
              aria-pressed={hiddenClasses.has(artefactClass.id)}
              onClick={() => toggleClassVisibility(artefactClass.id)}
            >
              {hiddenClasses.has(artefactClass.id) ? "Show not-tissue" : "Hide not-tissue"}
            </button>
          )}
        </div>
      )}

      <div className="ctx-sep" />

      <div className="hint">
        Where detection is wrong, select the regions and mark them. Corrections
        are the training set.
      </div>
      <div className="row-actions">
        <button className="btn" onClick={() => reclassify(TISSUE_CLASS)} disabled={!selectable.length}>
          Mark tissue
        </button>
        <button className="btn" onClick={() => reclassify(ARTEFACT_CLASS)} disabled={!selectable.length}>
          Mark not tissue
        </button>
      </div>

      <dl className="kv">
        <dt>On this slide</dt>
        <dd>
          {tissueRegions.length} tissue · {artefactRegions.length} not
        </dd>
      </dl>

      <button
        className="btn"
        style={{ width: "100%" }}
        onClick={() => void addThisSlide()}
        disabled={busy !== null}
      >
        {busy === "sampling"
          ? "Sampling…"
          : samples.some((s) => s.slide === source.meta.name)
            ? "Update this slide's labels"
            : "Add this slide's labels"}
      </button>

      {samples.length > 0 && (
        <>
          <div className="model-group-head" style={{ marginTop: 10 }}>
            Training set — {totals.slides} slide{totals.slides === 1 ? "" : "s"}
          </div>
          <div className="scroll-list scroll-list--short">
          {samples.map((s) => (
            <div key={s.slide} className="token-row">
              <span className="token-label" title={s.slide}>{s.slide}</span>
              <span className="model-option-size">
                {s.report.regions.tissue}/{s.report.regions.artefact}
              </span>
              <button className="mini danger" title="Remove" onClick={() => removeSamples(s.slide)}>
                ×
              </button>
            </div>
          ))}
          </div>
          <div className="picker-hint">
            {totals.cells.toLocaleString()} cells. Labels from several slides make a model that
            survives a change of stain or scanner — open another slide, mark it, and add it too.
          </div>

          <div className="field">
            <span>Name</span>
            <input
              className="class-edit"
              placeholder="e.g. Colon H&E, Aperio"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
          <button
            className="btn"
            style={{ width: "100%" }}
            onClick={() => void train()}
            disabled={busy !== null}
          >
            {busy === "training" ? "Training…" : "Train and save"}
          </button>
        </>
      )}

      {models.length > 0 && (
        <>
          <div className="model-group-head" style={{ marginTop: 10 }}>Saved models</div>
          {models.map((m) => (
            <div key={m.id} className="model-row">
              <button
                className="model-option"
                aria-current={m.id === activeModelId}
                onClick={() => void choose(m.id)}
                title="Use this model, and load the labels it was trained on"
              >
                <div className="model-option-head">
                  <span className="radio" data-on={m.id === activeModelId} />
                  <span className="model-option-name">{m.name}</span>
                  <span className="model-option-size">
                    {m.slides.length} slide{m.slides.length === 1 ? "" : "s"}
                  </span>
                </div>
                <div className="model-option-blurb">
                  {m.samples.tissue.toLocaleString()} tissue ·{" "}
                  {m.samples.artefact.toLocaleString()} not ·{" "}
                  {m.metrics
                    ? `F1 ${m.metrics.f1.toFixed(2)} on ${m.metrics.heldOut.toLocaleString()} held-out cells`
                    : "no held-out score — label a third region of each kind to get one"}
                </div>
                {m.slides.length > 0 && (
                  <div className="model-slides" title={m.slides.join("\n")}>
                    {m.slides.join(" · ")}
                  </div>
                )}
              </button>
              <button
                className="model-remove"
                aria-label={`Delete ${m.name}`}
                title="Delete this model and the labels saved with it"
                onClick={() => {
                  if (!window.confirm(`Delete "${m.name}" and its saved labels?`)) return;
                  void deleteTissueModel(m.id).then(() =>
                    loadTissueModels().then((st) => setModels(st.usable, st.stale)),
                  );
                }}
              >
                ×
              </button>
            </div>
          ))}

          <label className="check">
            <input type="checkbox" checked={useModel} onChange={(e) => setUseModel(e.target.checked)} />
            Use the model when detecting tissue
          </label>

          {active && (
            <div className="picker-hint">
              Leans on: {explain(active).slice(0, 3).map((f) => f.name).join(", ")}.
            </div>
          )}

          <button
            className="btn"
            style={{ width: "100%" }}
            onClick={() => setBatching(true)}
            disabled={busy !== null}
          >
            Run tissue classifier on every slide in a folder…
          </button>
        </>
      )}

      {staleModels.length > 0 && (
        <div className="picker-hint">
          {staleModels.length} saved model{staleModels.length === 1 ? " was" : "s were"} fitted
          against an older feature set and cannot be used. Retrain to replace{" "}
          {staleModels.length === 1 ? "it" : "them"}.
        </div>
      )}

      {notice && <div className="note warn">{notice}</div>}
      {batching && <BatchDialog classes={classes} close={() => setBatching(false)} />}
      {/* Re-render when the document changes so the counts stay honest. */}
      <span hidden>{version}</span>
    </section>
  );
}
