import { useState } from "react";
import { useMl } from "../ml/mlStore";
import { findModel, formatBytes } from "../ml/registry";
import {
  importLocalModel,
  validateImport,
  type NormalisationPreset,
} from "../ml/localModels";

const PRESETS: { id: NormalisationPreset; label: string; hint: string }[] = [
  { id: "imagenet", label: "ImageNet", hint: "1/255 then ImageNet mean/std — most timm and HF encoders" },
  { id: "sam", label: "SAM", hint: "Original Segment Anything statistics" },
  { id: "none", label: "None", hint: "Raw 0-1, no mean subtraction" },
];

/**
 * Import an ONNX encoder from disk.
 *
 * The gated histology encoders — UNI, Virchow2, CONCH, GigaPath — cannot be
 * bundled or fetched from a public URL, so this is how they get in. Only the
 * encoder is usually needed: SAM variants share a decoder architecture, so an
 * imported encoder can borrow the decoder of whichever built-in it was derived
 * from. Getting `inputSize` or normalisation wrong produces plausible-looking
 * nonsense rather than an error, so both are asked for explicitly.
 */
export function ImportModelDialog({ close }: { close: () => void }) {
  const models = useMl((s) => s.models);
  const addModel = useMl((s) => s.addModel);

  const [name, setName] = useState("");
  const [encoder, setEncoder] = useState<File | null>(null);
  const [decoder, setDecoder] = useState<File | null>(null);
  const [borrowFrom, setBorrowFrom] = useState(models[0]?.id ?? "");
  const [inputSize, setInputSize] = useState(1024);
  const [preset, setPreset] = useState<NormalisationPreset>("imagenet");
  const [licence, setLicence] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fallback = findModel(models, borrowFrom)?.files.find((f) => f.part === "decoder");
  const request = {
    name,
    encoder: encoder ?? undefined,
    decoder: decoder ?? undefined,
    fallbackDecoder: decoder ? undefined : fallback,
    inputSize,
    preset,
    licence,
  };
  const problem = validateImport(request);

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      const spec = await importLocalModel({
        name,
        encoder: encoder!,
        decoder: decoder ?? undefined,
        fallbackDecoder: decoder ? undefined : fallback,
        inputSize,
        preset,
        licence,
      });
      addModel(spec);
      close();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal-scrim" onPointerDown={(e) => { if (e.target === e.currentTarget) close(); }}>
      <div className="modal" role="dialog" aria-label="Import ONNX model">
        <div className="modal-head">Import ONNX model</div>

        <label className="field">
          <span>Name</span>
          <input
            className="class-edit"
            value={name}
            placeholder="UNI v1"
            onChange={(e) => setName(e.target.value)}
          />
        </label>

        <label className="field">
          <span>Encoder</span>
          <input
            type="file"
            accept=".onnx"
            onChange={(e) => setEncoder(e.target.files?.[0] ?? null)}
          />
        </label>
        {encoder && <div className="field-note">{encoder.name} · {formatBytes(encoder.size)}</div>}

        <label className="field">
          <span>Decoder</span>
          <input
            type="file"
            accept=".onnx"
            onChange={(e) => setDecoder(e.target.files?.[0] ?? null)}
          />
        </label>
        {decoder ? (
          <div className="field-note">{decoder.name} · {formatBytes(decoder.size)}</div>
        ) : (
          <label className="field">
            <span>Borrow</span>
            <select
              className="model-select"
              value={borrowFrom}
              onChange={(e) => setBorrowFrom(e.target.value)}
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>{m.name}</option>
              ))}
            </select>
          </label>
        )}

        <label className="field">
          <span>Input px</span>
          <input
            className="class-edit"
            type="number"
            min={64}
            max={4096}
            value={inputSize}
            onChange={(e) => setInputSize(Number(e.target.value))}
          />
        </label>

        <label className="field">
          <span>Normalise</span>
          <select
            className="model-select"
            value={preset}
            onChange={(e) => setPreset(e.target.value as NormalisationPreset)}
          >
            {PRESETS.map((p) => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
          </select>
        </label>
        <div className="field-note">{PRESETS.find((p) => p.id === preset)?.hint}</div>

        <label className="field">
          <span>Licence</span>
          <input
            className="class-edit"
            value={licence}
            placeholder="e.g. CC-BY-NC-ND 4.0 (optional)"
            onChange={(e) => setLicence(e.target.value)}
          />
        </label>

        {error && <div className="note err">{error}</div>}
        {!error && problem && <div className="hint">{problem}</div>}

        <div className="row-actions" style={{ marginTop: 10 }}>
          <button className="btn" onClick={close} disabled={busy}>Cancel</button>
          <button className="btn" onClick={() => void submit()} disabled={busy || !!problem}>
            {busy ? "Importing…" : "Import"}
          </button>
        </div>
        <div className="picker-hint">
          Weights stay on this machine — nothing is uploaded.
        </div>
      </div>
    </div>
  );
}
