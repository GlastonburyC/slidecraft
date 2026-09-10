import { useState } from "react";
import { importSpatialModel } from "../ml/localModels";
import { useSpatial } from "../ml/spatialStore";

/**
 * Import a virtual-spatial model.
 *
 * Two files, both required: the graph, and the sidecar the exporter wrote
 * beside it. The sidecar carries the gene order, the normalisation and the
 * magnification, none of which can be guessed — which is why this asks for it
 * rather than offering a set of presets to pick wrongly from.
 */
export function ImportSpatialDialog({ close }: { close: () => void }) {
  const [onnx, setOnnx] = useState<File | null>(null);
  const [sidecar, setSidecar] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const setModels = useSpatial((s) => s.setModels);
  const models = useSpatial((s) => s.models);

  const submit = async () => {
    if (!onnx || !sidecar) return;
    setBusy(true);
    setError(null);
    try {
      const spec = await importSpatialModel({ onnx, sidecar });
      setModels([spec, ...models]);
      close();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={close}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">Import a spatial model</div>

        <div className="picker-hint">
          Export one first with <code>scripts/export_deepspot.py</code>, which writes both files.
          Nothing is uploaded — the weights are stored in this browser.
        </div>

        <label className="field">
          <span>Model</span>
          <input type="file" accept=".onnx" onChange={(e) => setOnnx(e.target.files?.[0] ?? null)} />
        </label>
        <label className="field">
          <span>Sidecar</span>
          <input type="file" accept=".json" onChange={(e) => setSidecar(e.target.files?.[0] ?? null)} />
        </label>
        <div className="field-note">the .onnx.json written next to the graph</div>

        {error && <div className="note err">{error}</div>}

        <div className="row-actions">
          <button className="btn" onClick={close}>Cancel</button>
          <button className="btn" disabled={!onnx || !sidecar || busy} onClick={() => void submit()}>
            {busy ? "Importing…" : "Import"}
          </button>
        </div>
      </div>
    </div>
  );
}
