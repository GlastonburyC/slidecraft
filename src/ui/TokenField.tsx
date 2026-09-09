import { useState } from "react";
import { getHfToken, setHfToken } from "../ml/modelCache";

/**
 * Hugging Face access token.
 *
 * Gated encoders need one to fetch at all. It is stored in this browser's
 * localStorage and sent only to huggingface.co — never bundled, never written
 * into the project, and never attached to any other host. The field is a
 * password input and only ever shows the last four characters back, so a token
 * cannot be read off a shared screen or a screenshot.
 */
export function TokenField() {
  const [stored, setStored] = useState<string | null>(() => getHfToken());
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");

  const save = () => {
    setHfToken(draft);
    setStored(draft.trim() ? draft.trim() : null);
    setDraft("");
    setEditing(false);
  };

  const clear = () => {
    setHfToken(null);
    setStored(null);
    setDraft("");
    setEditing(false);
  };

  if (!editing) {
    return (
      <div className="token-row">
        <span className="token-label">Hugging Face</span>
        {stored ? (
          <>
            <span className="token-mask" title="Stored in this browser only">
              ••••{stored.slice(-4)}
            </span>
            <button className="mini" title="Replace" onClick={() => setEditing(true)}>edit</button>
            <button className="mini danger" title="Forget this token" onClick={clear}>×</button>
          </>
        ) : (
          <button className="mini" onClick={() => setEditing(true)}>add token</button>
        )}
      </div>
    );
  }

  return (
    <div className="token-edit">
      <input
        className="class-edit"
        type="password"
        autoFocus
        autoComplete="off"
        spellCheck={false}
        placeholder="hf_…"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") save();
          if (e.key === "Escape") { setDraft(""); setEditing(false); }
        }}
      />
      <div className="row-actions">
        <button className="btn" onClick={() => { setDraft(""); setEditing(false); }}>Cancel</button>
        <button className="btn" onClick={save} disabled={!draft.trim()}>Save</button>
      </div>
      <div className="picker-hint">
        Kept in this browser and sent only to huggingface.co. Needed for gated
        repos; accept the model's terms on its page first.
      </div>
    </div>
  );
}
