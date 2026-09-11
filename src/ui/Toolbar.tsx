import { useEffect, useState } from "react";
import { useAnnotations, type ToolId } from "../annotate/store";
import { BrushPopover } from "./BrushPopover";

interface ToolSpec {
  id: ToolId;
  key: string;
  label: string;
  hint: string;
  path: string;
}

/** 20x20 viewBox glyphs, stroked. Drawn inline to avoid an icon dependency. */
const TOOLS: ToolSpec[] = [
  { id: "pan", key: "h", label: "Pan", hint: "Pan and zoom (hold Space anywhere)", path: "M10 3v9M10 12V7a1.5 1.5 0 0 1 3 0v5M13 12V8.5a1.5 1.5 0 0 1 3 0V13a4 4 0 0 1-4 4H10a4 4 0 0 1-4-4v-1.5L4.5 10a1.5 1.5 0 0 1 2.2-2L7 8.5" },
  { id: "select", key: "v", label: "Select", hint: "Select an object; Shift adds; Delete removes", path: "M5 3l11 7-4.7 1.3L9.5 17z" },
  { id: "rectangle", key: "r", label: "Rectangle", hint: "Drag a rectangular region", path: "M3.5 4.5h13v11h-13z" },
  { id: "polygon", key: "p", label: "Polygon", hint: "Click vertices; Enter or double-click closes; Esc cancels", path: "M10 3l7 5-2.5 8h-9L3 8z" },
  { id: "freehand", key: "f", label: "Freehand", hint: "Drag to trace an outline", path: "M3 13c3-7 6 3 8-2s4 1 6-3" },
  { id: "brush", key: "b", label: "Brush", hint: "Paint; merges with the same class underneath", path: "M4 16s0-3 2-4 5-1 6-3 2-4 2-4 2 1.5 2 4-2 4-4 5-5 1-6 2-2 0-2 0z" },
  { id: "eraser", key: "e", label: "Eraser", hint: "Subtract from any region it touches", path: "M7 16h9M4.5 13.5l5-5 5 5-2.5 2.5h-5z" },
  { id: "point", key: "n", label: "Point", hint: "Drop a counting point", path: "M10 4a4 4 0 1 1 0 8 4 4 0 0 1 0-8zM10 12v5" },
  { id: "segment", key: "g", label: "Click-to-segment", hint: "Click a cell centre and the model outlines it; Shift adds, Alt excludes", path: "M10 4.5a5.5 5.5 0 1 1 0 11 5.5 5.5 0 0 1 0-11zM10 8.6a1.4 1.4 0 1 1 0 2.8 1.4 1.4 0 0 1 0-2.8zM10 2v2M10 16v2M2 10h2M16 10h2" },
  { id: "roi", key: "o", label: "ROI", hint: "Drag a region of interest for the ML loop", path: "M3.5 7V4.5H6M14 4.5h2.5V7M16.5 13v2.5H14M6 15.5H3.5V13" },
  { id: "patch", key: "t", label: "Patch", hint: "Drag a region and it is tiled into patches at the chosen size", path: "M3.5 3.5h13v13h-13zM8 3.5v13M12 3.5v13M3.5 8h13M3.5 12h13" },
];

export function Toolbar({
  mppX,
}: { mppX: number | null }) {
  const tool = useAnnotations((s) => s.tool);
  const setTool = useAnnotations((s) => s.setTool);
  const [brushAt, setBrushAt] = useState<{ x: number; y: number } | null>(null);

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      const target = ev.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
      // [ and ] resize the brush, the way raster editors do.
      if (ev.key === "[" || ev.key === "]") {
        const { tool: cur, brushRadius, setBrushRadius } = useAnnotations.getState();
        if (cur === "brush" || cur === "eraser") {
          ev.preventDefault();
          setBrushRadius(ev.key === "[" ? brushRadius / 1.25 : brushRadius * 1.25);
          return;
        }
      }
      const match = TOOLS.find((t) => t.key === ev.key.toLowerCase());
      if (match) {
        ev.preventDefault();
        setTool(match.id);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setTool]);

  return (
    <>
    <div className="toolrail">
      {TOOLS.map((t) => (
        <button
          key={t.id}
          className="tool"
          aria-pressed={tool === t.id}
          onClick={() => setTool(t.id)}
          onContextMenu={(e) => {
            // Right-clicking a tool opens that tool's own settings.
            if (t.id === "brush" || t.id === "eraser") {
              e.preventDefault();
              e.stopPropagation();
              setTool(t.id);
              setBrushAt({ x: e.clientX + 6, y: e.clientY - 40 });
            }
          }}
          title={
            t.id === "brush" || t.id === "eraser"
              ? `${t.label} · ${t.key.toUpperCase()} — ${t.hint} · right-click for size`
              : `${t.label} · ${t.key.toUpperCase()} — ${t.hint}`
          }
        >
          <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
            <path
              d={t.path}
              fill="none"
              stroke="currentColor"
              strokeWidth="1.4"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          </svg>
          <span className="tool-key">{t.key.toUpperCase()}</span>
        </button>
      ))}
    </div>
    {brushAt && <BrushPopover {...brushAt} mppX={mppX} close={() => setBrushAt(null)} />}
    </>
  );
}
