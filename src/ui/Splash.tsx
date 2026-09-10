import { useEffect, useState } from "react";

/**
 * The mark, drawn once while the app starts.
 *
 * It covers the moment before anything is on screen — worker pool spinning up,
 * fonts settling — which otherwise reads as a blank window. It draws the slide,
 * settles the specimen, and picks out one cell: the app's loop in about two
 * seconds.
 *
 * It is never in the way. It dismisses on any click or key, it is
 * pointer-transparent while fading, and anyone who asked for reduced motion
 * gets the finished mark and a much shorter hold rather than the animation.
 */

const HOLD_MS = 1900;
const FADE_MS = 420;

export function Splash() {
  const [leaving, setLeaving] = useState(false);
  const [gone, setGone] = useState(false);

  useEffect(() => {
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    const hold = window.setTimeout(() => setLeaving(true), reduced ? 450 : HOLD_MS);

    // Any deliberate input means the user is ready before the animation is.
    const skip = () => setLeaving(true);
    window.addEventListener("pointerdown", skip);
    window.addEventListener("keydown", skip);
    return () => {
      window.clearTimeout(hold);
      window.removeEventListener("pointerdown", skip);
      window.removeEventListener("keydown", skip);
    };
  }, []);

  useEffect(() => {
    if (!leaving) return;
    const done = window.setTimeout(() => setGone(true), FADE_MS);
    return () => window.clearTimeout(done);
  }, [leaving]);

  if (gone) return null;

  return (
    <div className="splash" data-leaving={leaving} aria-hidden="true">
      <div className="splash-mark">
        <svg viewBox="0 0 32 32" fill="none">
          <defs>
            <linearGradient id="splashSweep" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--accent)" stopOpacity="0" />
              <stop offset="45%" stopColor="var(--accent)" stopOpacity="0.22" />
              <stop offset="55%" stopColor="var(--accent)" stopOpacity="0.22" />
              <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
            </linearGradient>
          </defs>

          <rect className="glass" x="6.5" y="2.5" width="19" height="27" rx="3"
            stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
          <path className="frost" d="M6.5 8.5h19"
            stroke="currentColor" strokeWidth="1.2" opacity="0.35" strokeLinecap="round" />

          <path className="tissue a"
            d="M12.4 14.6c1.9-2.6 5.6-2.4 7 .2 1.2 2.2.4 4.6-1.6 5.7-2.2 1.2-5 .4-6-1.6-.8-1.6-.6-3.1.6-4.3Z" />
          <path className="tissue b"
            d="M17.6 20.4c1.5-1.1 3.7-.5 4.3 1.2.5 1.5-.4 3-2 3.4-1.7.4-3.2-.6-3.4-2.2-.1-1 .3-1.9 1.1-2.4Z" />

          <circle className="ring" cx="14.6" cy="17.2" r="2.5" />
          <circle className="nucleus" cx="14.6" cy="17.2" r="1.1" />

          <rect className="sweep" x="6.5" y="2.5" width="19" height="27" rx="3" />
        </svg>
      </div>
      <div className="splash-word">Slidecraft</div>
    </div>
  );
}
