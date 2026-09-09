import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useAnnotations } from "../annotate/store";
import { ROI_CLASS_ID, type AnnotationClass } from "../annotate/types";
import { BrushSizeControl } from "./BrushPopover";

const rgb = (c: [number, number, number]) => `rgb(${c[0]},${c[1]},${c[2]})`;

/** Keep a floating panel inside the window, flipping rather than clipping. */
function useClamped(x: number, y: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ left: x, top: y, ready: false });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const { width, height } = el.getBoundingClientRect();
    const pad = 8;
    setPos({
      left: Math.max(pad, Math.min(x, window.innerWidth - width - pad)),
      top: Math.max(pad, Math.min(y, window.innerHeight - height - pad)),
      ready: true,
    });
  }, [x, y]);
  return { ref, style: { left: pos.left, top: pos.top, visibility: pos.ready ? "visible" : "hidden" } as const };
}

/**
 * Return keyboard focus to the document body.
 *
 * A picker leaves focus inside its text input, and the tool controller
 * deliberately ignores Delete and Backspace while a field is focused — so
 * without this, closing a picker silently disables the delete keys until the
 * user happens to click the canvas.
 */
function releaseFocus() {
  const active = document.activeElement as HTMLElement | null;
  if (active && /^(INPUT|TEXTAREA|SELECT)$/.test(active.tagName)) active.blur();
}

/** Dismiss on outside pointerdown or Escape. */
function useDismiss(active: boolean, close: () => void, selector: string) {
  useEffect(() => {
    if (!active) return;
    const onDown = (e: PointerEvent) => {
      if (!(e.target as HTMLElement | null)?.closest(selector)) close();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.stopPropagation(); close(); }
    };
    // Capture phase, so the tool controller does not also act on the click.
    window.addEventListener("pointerdown", onDown, true);
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("pointerdown", onDown, true);
      window.removeEventListener("keydown", onKey, true);
    };
  }, [active, close, selector]);
}

export function AnnotationMenus({ mppX }: { mppX: number | null }) {
  const contextMenu = useAnnotations((s) => s.contextMenu);
  const classPicker = useAnnotations((s) => s.classPicker);
  const [renaming, setRenaming] = useState<{ x: number; y: number; id: string } | null>(null);

  return (
    <>
      {contextMenu && <ContextMenu onRename={setRenaming} mppX={mppX} />}
      {classPicker && <ClassPicker />}
      {renaming && <RenamePrompt {...renaming} close={() => setRenaming(null)} />}
    </>
  );
}

// ------------------------------------------------------------ context menu ---

function ContextMenu({
  onRename,
  mppX,
}: {
  onRename: (r: { x: number; y: number; id: string } | null) => void;
  mppX: number | null;
}) {
  const anchor = useAnnotations((s) => s.contextMenu)!;
  const items = useAnnotations((s) => s.items);
  const classes = useAnnotations((s) => s.classes);
  const tool = useAnnotations((s) => s.tool);
  const {
    openContextMenu, openClassPicker, assignClass, apply, clearSelection, createClass,
  } = useAnnotations.getState();

  const close = () => { releaseFocus(); openContextMenu(null); };
  const { ref, style } = useClamped(anchor.x, anchor.y);
  useDismiss(true, close, ".ctx-menu");

  const targets = anchor.targetIds
    .map((id) => items.get(id))
    .filter((a): a is NonNullable<typeof a> => !!a);
  const n = targets.length;
  const single = n === 1 ? targets[0] : null;
  const allLocked = n > 0 && targets.every((a) => a.locked);

  const act = (fn: () => void) => () => { fn(); close(); };

  return (
    <div className="ctx-menu" ref={ref} style={style} role="menu">
      {tool === "brush" || tool === "eraser" ? (
        <>
          <BrushSizeControl mppX={mppX} />
          <div className="ctx-sep" />
          {n > 0 && (
            <button className="ctx-item" onClick={act(() =>
              openClassPicker({ x: anchor.x, y: anchor.y, targetIds: anchor.targetIds }))}>
              Set class…<kbd>C</kbd>
            </button>
          )}
          <button className="ctx-item" onClick={act(() => {
            const name = window.prompt("New class name");
            if (name && name.trim()) createClass(name);
          })}>
            New class…
          </button>
        </>
      ) : n === 0 ? (
        <>
          <div className="ctx-head">Canvas</div>
          <button className="ctx-item" onClick={act(() => {
            const name = window.prompt("New class name");
            if (name && name.trim()) createClass(name);
          })}>
            New class…
          </button>
        </>
      ) : (
        <>
          <div className="ctx-head">
            {single ? (single.name ?? single.geometry.type) : `${n} objects`}
          </div>

          <button className="ctx-item" onClick={act(() =>
            openClassPicker({ x: anchor.x, y: anchor.y, targetIds: anchor.targetIds }))}>
            Set class…<kbd>C</kbd>
          </button>

          {single && (
            <button className="ctx-item" onClick={act(() =>
              onRename({ x: anchor.x, y: anchor.y, id: single.id }))}>
              Rename object…
            </button>
          )}

          {classes.length > 0 && (
            <>
              <div className="ctx-sep" />
              <div className="ctx-classes">
                {classes.map((c, i) => (
                  <button
                    key={c.id}
                    className="ctx-item compact"
                    onClick={act(() => assignClass(anchor.targetIds, c.id))}
                  >
                    <span className="swatch sm" style={{ background: rgb(c.color) }} />
                    {c.name}
                    {i < 9 && <kbd>{i + 1}</kbd>}
                  </button>
                ))}
                <button className="ctx-item compact" onClick={act(() =>
                  assignClass(anchor.targetIds, null))}>
                  <span className="swatch sm" style={{ background: "#6b7280" }} />
                  Unclassified
                </button>
              </div>
            </>
          )}

          <div className="ctx-sep" />
          <button className="ctx-item" onClick={act(() => {
            apply({
              label: allLocked ? "Unlock" : "Lock",
              updated: targets.map((a) => ({ before: a, after: { ...a, locked: !allLocked } })),
            });
          })}>
            {allLocked ? "Unlock" : "Lock"}
          </button>
          <button className="ctx-item danger" onClick={act(() => {
            const removable = targets.filter((a) => !a.locked);
            if (removable.length) {
              apply({ label: `Delete ${removable.length}`, removed: removable });
              clearSelection();
            }
          })}>
            Delete<kbd>⌫</kbd>
          </button>
        </>
      )}
    </div>
  );
}

// ------------------------------------------------------------- class picker ---

/**
 * Type-to-name. Filters existing classes and offers to create the typed name,
 * so labelling a new region is one keystroke and one word — no trip to a
 * settings panel to define the class first.
 */
function ClassPicker() {
  const anchor = useAnnotations((s) => s.classPicker)!;
  const classes = useAnnotations((s) => s.classes);
  const { openClassPicker, assignClass, ensureClass } = useAnnotations.getState();

  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const close = () => { releaseFocus(); openClassPicker(null); };
  const { ref, style } = useClamped(anchor.x, anchor.y);
  useDismiss(true, close, ".class-picker");

  useEffect(() => { inputRef.current?.focus(); }, []);

  const q = query.trim().toLowerCase();
  const matches = useMemo(
    () => classes.filter((c) => c.id !== ROI_CLASS_ID && c.name.toLowerCase().includes(q)),
    [classes, q],
  );
  const exact = matches.some((c) => c.name.toLowerCase() === q);
  const canCreate = q.length > 0 && !exact;
  const rows: ({ kind: "class"; cls: AnnotationClass } | { kind: "create" } | { kind: "clear" })[] = [
    ...matches.map((cls) => ({ kind: "class" as const, cls })),
    ...(canCreate ? [{ kind: "create" as const }] : []),
    { kind: "clear" as const },
  ];

  const commit = (row: (typeof rows)[number]) => {
    if (row.kind === "class") assignClass(anchor.targetIds, row.cls.id);
    else if (row.kind === "create") assignClass(anchor.targetIds, ensureClass(query).id);
    else assignClass(anchor.targetIds, null);
    close();
  };

  return (
    <div className="class-picker" ref={ref} style={style}>
      <input
        ref={inputRef}
        value={query}
        placeholder={classes.length ? "Name or pick a class…" : "Name this annotation…"}
        onChange={(e) => { setQuery(e.target.value); setActive(0); }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") { e.preventDefault(); setActive((i) => Math.min(i + 1, rows.length - 1)); }
          else if (e.key === "ArrowUp") { e.preventDefault(); setActive((i) => Math.max(i - 1, 0)); }
          else if (e.key === "Enter") { e.preventDefault(); if (rows[active]) commit(rows[active]); }
          else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); }
        }}
      />
      <div className="picker-rows">
        {rows.map((row, i) => (
          <button
            key={row.kind === "class" ? row.cls.id : row.kind}
            className="picker-row"
            aria-selected={i === active}
            onPointerEnter={() => setActive(i)}
            onClick={() => commit(row)}
          >
            {row.kind === "class" && (
              <>
                <span className="swatch sm" style={{ background: rgb(row.cls.color) }} />
                {row.cls.name}
              </>
            )}
            {row.kind === "create" && (
              <>
                <span className="swatch sm plus">+</span>
                Create “{query.trim()}”
              </>
            )}
            {row.kind === "clear" && (
              <>
                <span className="swatch sm" style={{ background: "#6b7280" }} />
                Unclassified
              </>
            )}
          </button>
        ))}
      </div>
      <div className="picker-hint">
        {anchor.targetIds.length} object{anchor.targetIds.length === 1 ? "" : "s"} · ↑↓ then Enter
      </div>
    </div>
  );
}

// ----------------------------------------------------------- rename prompt ---

function RenamePrompt({
  x, y, id, close: rawClose,
}: { x: number; y: number; id: string; close: () => void }) {
  const close = () => { releaseFocus(); rawClose(); };
  const item = useAnnotations((s) => s.items.get(id));
  const renameAnnotation = useAnnotations((s) => s.renameAnnotation);
  const [value, setValue] = useState(item?.name ?? "");
  const inputRef = useRef<HTMLInputElement>(null);
  const { ref, style } = useClamped(x, y);
  useDismiss(true, close, ".class-picker");
  useEffect(() => { inputRef.current?.select(); }, []);

  return (
    <div className="class-picker" ref={ref} style={style}>
      <input
        ref={inputRef}
        value={value}
        placeholder="Object name"
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") { e.preventDefault(); renameAnnotation(id, value); close(); }
          else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); }
        }}
      />
      <div className="picker-hint">Names one object; use Set class to label many</div>
    </div>
  );
}
