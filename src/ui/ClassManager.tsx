import { useEffect, useRef, useState } from "react";
import { useAnnotations } from "../annotate/store";
import { ROI_COLOR } from "../annotate/types";

const rgb = (c: [number, number, number]) => `rgb(${c[0]},${c[1]},${c[2]})`;
const hex = (c: [number, number, number]) =>
  `#${c.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
const fromHex = (h: string): [number, number, number] => [
  parseInt(h.slice(1, 3), 16),
  parseInt(h.slice(3, 5), 16),
  parseInt(h.slice(5, 7), 16),
];

/**
 * The class scheme is the user's, not ours: it starts empty and is built here
 * or inline while annotating. Counts are live so it doubles as a tally.
 */
export function ClassManager({ counts, roiCount }: { counts: Map<string, number>; roiCount: number }) {
  const classes = useAnnotations((s) => s.classes);
  const activeClassId = useAnnotations((s) => s.activeClassId);
  const selection = useAnnotations((s) => s.selection);
  // Subscribed, not read once: the toggle has to re-render the row it is on.
  const hiddenClasses = useAnnotations((s) => s.hiddenClasses);
  const toggleClassVisibility = useAnnotations((s) => s.toggleClassVisibility);
  const {
    setActiveClass, createClass, updateClass, deleteClass, reorderClass, assignClass,
  } = useAnnotations.getState();

  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const addRef = useRef<HTMLInputElement>(null);

  useEffect(() => { if (adding) addRef.current?.focus(); }, [adding]);

  const submitNew = () => {
    if (draft.trim()) createClass(draft);
    setDraft("");
    setAdding(false);
  };

  return (
    <section className="section">
      <h2>Classes {classes.length > 0 && `— ${classes.length}`}</h2>

      {classes.length === 0 && !adding && (
        <p className="empty-note">
          No classes yet. Draw a region and press <kbd>C</kbd> to name it, or add one below.
        </p>
      )}

      {classes.map((c, i) => (
        <div key={c.id} className="class-row-wrap">
          {editingId === c.id ? (
            <input
              className="class-edit"
              autoFocus
              defaultValue={c.name}
              onBlur={(e) => { updateClass(c.id, { name: e.target.value.trim() || c.name }); setEditingId(null); }}
              onKeyDown={(e) => {
                if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                if (e.key === "Escape") setEditingId(null);
              }}
            />
          ) : (
            <button
              className="class-row"
              aria-current={activeClassId === c.id}
              onClick={() => setActiveClass(c.id)}
              onDoubleClick={() => setEditingId(c.id)}
              title={`Click to make active (${i < 9 ? i + 1 : "—"}) · double-click to rename`}
            >
              <label className="swatch-btn" onClick={(e) => e.stopPropagation()}>
                <span className="swatch" style={{ background: rgb(c.color) }} />
                <input
                  type="color"
                  value={hex(c.color)}
                  onChange={(e) => updateClass(c.id, { color: fromHex(e.target.value) })}
                />
              </label>
              <span className="class-name">{c.name}</span>
              <span className="class-count">{counts.get(c.id) || ""}</span>
              <span className="class-key">{i < 9 ? i + 1 : ""}</span>
            </button>
          )}

          <div className="class-tools">
            {/* Hiding is not deleting: the objects stay, and stay exported. */}
            <button
              className="mini"
              aria-pressed={hiddenClasses.has(c.id)}
              title={hiddenClasses.has(c.id) ? `Show ${c.name}` : `Hide ${c.name}`}
              onClick={() => toggleClassVisibility(c.id)}
            >
              {hiddenClasses.has(c.id) ? "◌" : "◉"}
            </button>
            {selection.size > 0 && (
              <button
                className="mini"
                title={`Assign ${selection.size} selected to ${c.name}`}
                onClick={() => assignClass([...selection], c.id)}
              >
                →
              </button>
            )}
            <button className="mini" title="Move up" disabled={i === 0}
              onClick={() => reorderClass(c.id, -1)}>↑</button>
            <button className="mini" title="Move down" disabled={i === classes.length - 1}
              onClick={() => reorderClass(c.id, 1)}>↓</button>
            <button
              className="mini danger"
              title="Delete class (objects become unclassified)"
              onClick={() => {
                const n = counts.get(c.id) || 0;
                if (n > 0 && !window.confirm(
                  `Delete "${c.name}"? ${n} object${n === 1 ? "" : "s"} will become unclassified.`,
                )) return;
                deleteClass(c.id);
              }}
            >
              ×
            </button>
          </div>
        </div>
      ))}

      {adding ? (
        <input
          ref={addRef}
          className="class-edit"
          value={draft}
          placeholder="Class name"
          onChange={(e) => setDraft(e.target.value)}
          onBlur={submitNew}
          onKeyDown={(e) => {
            if (e.key === "Enter") { e.preventDefault(); submitNew(); }
            if (e.key === "Escape") { setDraft(""); setAdding(false); }
          }}
        />
      ) : (
        <button className="btn add-class" onClick={() => setAdding(true)}>+ Add class</button>
      )}

      {roiCount > 0 && (
        <div className="class-row static" style={{ marginTop: 6 }}>
          <span className="swatch" style={{ background: rgb(ROI_COLOR) }} />
          <span className="class-name">Regions of interest</span>
          <span className="class-count">{roiCount}</span>
        </div>
      )}
    </section>
  );
}
