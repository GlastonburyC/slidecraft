import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";

/**
 * Floating panel anchored to a right-click, kept inside the window and
 * dismissed on outside click or Escape. Shared by the tool popovers so they
 * behave identically wherever they are summoned from.
 */
export function Popover({
  x, y, className, onClose, children,
}: {
  x: number;
  y: number;
  className: string;
  onClose: () => void;
  children: ReactNode;
}) {
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

  useEffect(() => {
    const sel = `.${className}`;
    const onDown = (e: PointerEvent) => {
      if (!(e.target as HTMLElement | null)?.closest(sel)) onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.stopPropagation(); onClose(); }
    };
    window.addEventListener("pointerdown", onDown, true);
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("pointerdown", onDown, true);
      window.removeEventListener("keydown", onKey, true);
    };
  }, [className, onClose]);

  return (
    <div
      className={className}
      ref={ref}
      style={{ left: pos.left, top: pos.top, visibility: pos.ready ? "visible" : "hidden" }}
    >
      {children}
    </div>
  );
}
