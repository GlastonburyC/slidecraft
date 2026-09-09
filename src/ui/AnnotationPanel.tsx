import { useEffect, useMemo, useRef, useState } from "react";
import { useAnnotations } from "../annotate/store";
import { areaOf, ROI_CLASS_ID, type Annotation } from "../annotate/types";
import { ClassManager } from "./ClassManager";
import { downloadGeoJSON, fromGeoJSON, toFeatureCollection } from "../io/geojson";
import type { SlideMeta } from "../slide/types";

/** Area in µm² when the vendor gave us a scale, otherwise pixels². */
function formatArea(px2: number, mpp: number | null): string {
  if (mpp) {
    const um2 = px2 * mpp * mpp;
    if (um2 > 1e6) return `${(um2 / 1e6).toFixed(2)} mm²`;
    return `${Math.round(um2).toLocaleString("en-US")} µm²`;
  }
  return `${Math.round(px2).toLocaleString("en-US")} px²`;
}

export function AnnotationPanel({ meta }: { meta: SlideMeta }) {
  const {
    items, version, classes, selection, showAnnotations, fillOpacity,
    brushRadius, tool, setBrushRadius, setFillOpacity, toggleAnnotations,
    undo, redo, undoStack, redoStack, apply, clearSelection, openClassPicker,
    notice: storeNotice, setNotice: setStoreNotice,
  } = useAnnotations();

  const importRef = useRef<HTMLInputElement>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const stats = useMemo(() => {
    const counts = new Map<string, number>();
    let rois = 0;
    let unclassified = 0;
    for (const a of items.values()) {
      if (a.classId === ROI_CLASS_ID) { rois += 1; continue; }
      if (!a.classId) unclassified += 1;
      else counts.set(a.classId, (counts.get(a.classId) ?? 0) + 1);
    }
    return { counts, rois, unclassified, total: items.size };
    // `version` is the change signal for the mutable Map.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, items]);

  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(null), 4000);
    return () => clearTimeout(t);
  }, [notice]);

  // Refusals from the tool controller (locked objects, mostly) clear themselves.
  useEffect(() => {
    if (!storeNotice) return;
    const t = setTimeout(() => setStoreNotice(null), 4000);
    return () => clearTimeout(t);
  }, [storeNotice, setStoreNotice]);

  // `items` is mutated in place, so its identity never changes — depending on
  // it would hand back stale Annotation objects whose geometry or class had
  // since been edited, and those would then be written into the undo stack.
  // `version` is the real change signal.
  const selectedList = useMemo(
    () => [...selection].map((id) => items.get(id)).filter((a): a is Annotation => !!a),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [selection, version],
  );

  const handleExport = () => {
    const fc = toFeatureCollection([...items.values()], classes, meta);
    downloadGeoJSON(fc, meta.name.replace(/\.[^.]+$/, ""));
    setNotice(`Exported ${fc.features.length} object${fc.features.length === 1 ? "" : "s"}`);
  };

  const handleImport = async (file: File) => {
    try {
      const parsed: unknown = JSON.parse(await file.text());
      const { annotations, classes: nextClasses, skipped } = fromGeoJSON(parsed, classes);
      if (annotations.length === 0) {
        setNotice("No usable features found in that file");
        return;
      }
      // Import is one undoable step, merged into whatever is already here.
      useAnnotations.setState({ classes: nextClasses });
      apply({ label: `Import ${annotations.length}`, added: annotations });
      setNotice(
        `Imported ${annotations.length} object${annotations.length === 1 ? "" : "s"}` +
          (skipped ? ` · skipped ${skipped} unsupported` : ""),
      );
    } catch (err) {
      setNotice(`Could not read that file: ${String(err)}`);
    }
  };

  const deleteSelected = () => {
    const removable = selectedList.filter((a) => !a.locked);
    if (removable.length === 0) return;
    apply({ label: `Delete ${removable.length}`, removed: removable });
    clearSelection();
  };

  return (
    <>
      <section className="section">
        <h2>Annotations — {stats.total}</h2>

        <div className="row-actions">
          <button className="btn" onClick={undo} disabled={undoStack.length === 0} title="Cmd/Ctrl+Z">
            Undo
          </button>
          <button className="btn" onClick={redo} disabled={redoStack.length === 0} title="Shift+Cmd/Ctrl+Z">
            Redo
          </button>
          <button className="btn" onClick={toggleAnnotations}>
            {showAnnotations ? "Hide" : "Show"}
          </button>
        </div>

        {undoStack.length > 0 && (
          <div className="hint">Last: {undoStack[undoStack.length - 1].label}</div>
        )}

        <div className="row-actions" style={{ marginTop: 8 }}>
          <button className="btn" onClick={handleExport} disabled={stats.total === 0}>
            Export GeoJSON
          </button>
          <button className="btn" onClick={() => importRef.current?.click()}>
            Import
          </button>
        </div>
        <input
          ref={importRef}
          type="file"
          accept=".geojson,.json,application/geo+json,application/json"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void handleImport(f);
            e.target.value = "";
          }}
        />
        {notice && <div className="hint notice">{notice}</div>}
        {storeNotice && <div className="note warn">{storeNotice}</div>}
      </section>

      <ClassManager counts={stats.counts} roiCount={stats.rois} />

      {stats.unclassified > 0 && (
        <div className="section subtle">
          <div className="hint">
            {stats.unclassified} unclassified — select and press <kbd>C</kbd> to name.
          </div>
        </div>
      )}

      {(tool === "brush" || tool === "eraser") && (
        <section className="section">
          <h2>{tool === "brush" ? "Brush" : "Eraser"}</h2>
          <label className="slider">
            <span>Radius</span>
            <input
              type="range"
              min={4}
              max={600}
              step={2}
              value={brushRadius}
              onChange={(e) => setBrushRadius(Number(e.target.value))}
            />
            <b>
              {meta.mppX ? `${Math.round(brushRadius * meta.mppX)} µm` : `${brushRadius} px`}
            </b>
          </label>
        </section>
      )}

      <section className="section">
        <h2>Display</h2>
        <label className="slider">
          <span>Fill</span>
          <input
            type="range"
            min={0}
            max={0.8}
            step={0.02}
            value={fillOpacity}
            onChange={(e) => setFillOpacity(Number(e.target.value))}
          />
          <b>{Math.round(fillOpacity * 100)}%</b>
        </label>
      </section>

      {selectedList.length > 0 && (
        <section className="section">
          <h2>Selected — {selectedList.length}</h2>
          {selectedList.length === 1 && (
            <dl className="kv">
              <dt>Type</dt>
              <dd>{selectedList[0].geometry.type}</dd>
              <dt>Source</dt>
              <dd>{selectedList[0].source}</dd>
              <dt>Area</dt>
              <dd>{formatArea(areaOf(selectedList[0].geometry), meta.mppX)}</dd>
            </dl>
          )}
          <div className="row-actions">
            <button className="btn" onClick={() => openClassPicker({
              x: window.innerWidth / 2, y: window.innerHeight / 3, targetIds: [...selection],
            })}>
              Set class… (C)
            </button>
          </div>
          <div className="row-actions" style={{ marginTop: 8 }}>
            <button className="btn danger" onClick={deleteSelected}>
              Delete
            </button>
            <button className="btn" onClick={clearSelection}>
              Deselect
            </button>
          </div>
        </section>
      )}
    </>
  );
}

export { formatArea };
