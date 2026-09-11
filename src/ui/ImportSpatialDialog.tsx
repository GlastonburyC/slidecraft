import { useState } from "react";
import { importSidecarModel } from "../ml/localModels";
import { useSpatial } from "../ml/spatialStore";
import { usePredict } from "../ml/predictStore";

/**
 * Import a virtual-spatial model.
 *
 * Two files, both required: the graph, and the sidecar the exporter wrote
 * beside it. The sidecar carries the gene order, the normalisation and the
 * magnification, none of which can be guessed — which is why this asks for it
 * rather than offering a set of presets to pick wrongly from.
 */
export function ImportSpatialDialog({
  close, kind = "virtual-spatial",
}: {
  close: () => void;
  /** Which panel opened it — only the wording and the destination differ. */
  kind?: "virtual-spatial" | "encode";
}) {
  const [onnx, setOnnx] = useState<File | null>(null);
  const [sidecar, setSidecar] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const setSpatialModels = useSpatial((s) => s.setModels);
  const spatialModels = useSpatial((s) => s.models);
  const setEncoders = usePredict((s) => s.setEncoders);
  const encoders = usePredict((s) => s.encoders);

  const submit = async () => {
    if (!onnx || !sidecar) return;
    setBusy(true);
    setError(null);
    try {
      const spec = await importSidecarModel({ onnx, sidecar });
      // The sidecar decides where it belongs, not the dialog that opened it —
      // so importing an encoder from the Spatial panel still lands correctly.
      if (spec.task === "encode") setEncoders([spec, ...encoders]);
      else setSpatialModels([spec, ...spatialModels]);
      close();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    // modal-scrim, not a bare div: it is what carries the fixed position, the
    // centring and the z-index that lifts this above the toolbar.
    <div
      className="modal-scrim"
      onPointerDown={(e) => { if (e.target === e.currentTarget) close(); }}
    >
      <div className="modal">
        <div className="modal-head">
          {kind === "encode" ? "Import a patch encoder" : "Import a spatial model"}
        </div>

        <div className="picker-hint">
          Export one first with{" "}
          <code>
            {kind === "encode"
              ? "scripts/export_onnx.py --preset uni2 --fp16"
              : "scripts/export_deepspot.py"}
          </code>
          , which writes both files. Nothing is uploaded — the weights are stored in this browser.
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
