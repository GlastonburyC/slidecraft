import { useMemo, useState } from "react";
import { useAnnotations } from "../annotate/store";
import {
  areaOf, AXIS_CLASS_ID, AXIS_COLOR, ROI_CLASS_ID, ROI_COLOR, type Annotation,
} from "../annotate/types";

const rgb = (c: [number, number, number]) => `rgb(${c[0]},${c[1]},${c[2]})`;

/** Rendered at once; beyond this the list is capped rather than windowed. */
const MAX_ROWS = 400;

function formatArea(px2: number, mpp: number | null): string {
  if (px2 <= 0) return "—";
  if (!mpp) return `${Math.round(px2).toLocaleString("en-US")} px²`;
  const um2 = px2 * mpp * mpp;
  if (um2 >= 1e6) return `${(um2 / 1e6).toFixed(2)} mm²`;
  if (um2 >= 1000) return `${Math.round(um2).toLocaleString("en-US")} µm²`;
  return `${um2.toFixed(1)} µm²`;
}

/**
 * The objects on this slide, largest first.
 *
 * This replaced the pyramid-level readout: level dimensions are a property of
 * the file that never changes and is checked once, whereas the annotation list
 * is the thing being worked on. Clicking a row selects the object and frames it
 * on the slide, which is the only way to find a small annotation again on a
 * gigapixel image.
 */
export function AnnotationList({
  mpp, onFocus,
}: {
  mpp: number | null;
  onFocus: ((bbox: [number, number, number, number]) => void) | null;
}) {
  const items = useAnnotations((s) => s.items);
  const version = useAnnotations((s) => s.version);
  const classes = useAnnotations((s) => s.classes);
  const selection = useAnnotations((s) => s.selection);
  const [query, setQuery] = useState("");

  const rows = useMemo(() => {
    const byId = new Map(classes.map((c) => [c.id, c]));
    const q = query.trim().toLowerCase();
    return [...items.values()]
      .map((a) => {
        const cls = a.classId === ROI_CLASS_ID || a.classId === AXIS_CLASS_ID
          ? null
          : a.classId ? byId.get(a.classId) : undefined;
        return {
          a,
          label: a.name ?? (a.classId === ROI_CLASS_ID
            ? "ROI"
            : a.classId === AXIS_CLASS_ID
              ? "Axis"
              : (cls?.name ?? "Unclassified")),
          color: a.classId === ROI_CLASS_ID
            ? ROI_COLOR
            : a.classId === AXIS_CLASS_ID
              ? AXIS_COLOR
              : (cls?.color ?? [150, 150, 160]),
          area: areaOf(a.geometry),
        };
      })
      .filter((r) => !q || r.label.toLowerCase().includes(q))
      .sort((x, y) => y.area - x.area);
    // `version` is the change signal for the mutable item map.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, items, classes, query]);

  const select = (a: Annotation) => {
    // Clicking a row is a canvas action, so keyboard focus must leave the list
    // for Delete and Backspace to reach the tool controller.
    (document.activeElement as HTMLElement | null)?.blur();
    useAnnotations.getState().select([a.id]);
    onFocus?.(a.bbox);
  };

  if (items.size === 0) return null;

  return (
    <section className="section">
      <h2>Objects — {items.size}</h2>
      {items.size > 8 && (
        <input
          className="class-edit"
          placeholder="Filter by class…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          // Delete and Backspace are ignored while a field has focus, so give
          // it up as soon as the user is done filtering.
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === "Escape") e.currentTarget.blur();
          }}
        />
      )}
      <div className="obj-list">
        {rows.slice(0, MAX_ROWS).map(({ a, label, color, area }) => (
          <button
            key={a.id}
            className="obj-row"
            aria-current={selection.has(a.id)}
            onClick={() => select(a)}
            title="Select and zoom to this object"
          >
            <span className="swatch sm" style={{ background: rgb(color as [number, number, number]) }} />
            <span className="obj-label">{label}</span>
            {a.locked && <span className="obj-lock" title="Locked">🔒</span>}
            <span className="obj-area">
              {a.geometry.type === "Point" ? "point" : formatArea(area, mpp)}
            </span>
          </button>
        ))}
      </div>
      {rows.length > MAX_ROWS && (
        <div className="hint">
          Showing the {MAX_ROWS} largest of {rows.length}.
        </div>
      )}
    </section>
  );
}
