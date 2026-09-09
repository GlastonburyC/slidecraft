import { useAnnotations } from "../annotate/store";
import { Popover } from "./Popover";

/**
 * Brush/eraser radius control.
 *
 * Radius is a slide-space quantity, so it is shown in µm whenever the vendor
 * gave us a scale: a "50 µm brush" means the same thing at every zoom and on
 * every slide, which a screen-pixel brush cannot promise.
 */
export function BrushSizeControl({ mppX }: { mppX: number | null }) {
  const brushRadius = useAnnotations((s) => s.brushRadius);
  const setBrushRadius = useAnnotations((s) => s.setBrushRadius);
  const tool = useAnnotations((s) => s.tool);

  const label = mppX
    ? `${Math.round(brushRadius * mppX * 2).toLocaleString("en-US")} µm`
    : `${Math.round(brushRadius * 2).toLocaleString("en-US")} px`;

  return (
    <div className="brush-control">
      <div className="brush-head">
        <span>{tool === "eraser" ? "Eraser" : "Brush"} diameter</span>
        <b>{label}</b>
      </div>
      <input
        type="range"
        min={4}
        max={2000}
        step={2}
        value={Math.min(2000, brushRadius)}
        onChange={(e) => setBrushRadius(Number(e.target.value))}
      />
      <div className="brush-presets">
        {[10, 25, 50, 100, 250, 500].map((r) => (
          <button key={r} className="mini" onClick={() => setBrushRadius(r)}>
            {mppX ? `${Math.round(r * mppX * 2)}µm` : `${r * 2}px`}
          </button>
        ))}
      </div>
      <div className="picker-hint">Radius {Math.round(brushRadius)} slide px · [ and ] to resize</div>
    </div>
  );
}

/** Floating version, anchored where the user right-clicked. */
export function BrushPopover({
  x, y, mppX, close,
}: { x: number; y: number; mppX: number | null; close: () => void }) {
  return (
    <Popover x={x} y={y} className="brush-popover" onClose={close}>
      <BrushSizeControl mppX={mppX} />
    </Popover>
  );
}
