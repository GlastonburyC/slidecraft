/**
 * Slidecraft mark.
 *
 * A microscope slide seen head-on, with a specimen on it and one cell picked
 * out — the app's whole loop in one glyph: a slide, tissue, and the single
 * object you clicked. Drawn as inline SVG so it inherits the theme's colours
 * and stays sharp at any size; `currentColor` on the glass means it recolours
 * with the header text rather than needing a second asset for light mode.
 */
export function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      aria-hidden="true"
      style={{ flex: "none", display: "block" }}
    >
      {/* Glass: a slide standing slightly proud, with the frosted label end. */}
      <rect
        x="6.5"
        y="2.5"
        width="19"
        height="27"
        rx="3"
        stroke="currentColor"
        strokeWidth="1.6"
        opacity="0.55"
      />
      <path
        d="M6.5 8.5h19"
        stroke="currentColor"
        strokeWidth="1.6"
        opacity="0.35"
        strokeLinecap="round"
      />

      {/* Specimen: two overlapping blobs, the shape tissue actually makes. */}
      <path
        d="M12.4 14.6c1.9-2.6 5.6-2.4 7 .2 1.2 2.2.4 4.6-1.6 5.7-2.2 1.2-5 .4-6-1.6-.8-1.6-.6-3.1.6-4.3Z"
        fill="var(--accent)"
        opacity="0.85"
      />
      <path
        d="M17.6 20.4c1.5-1.1 3.7-.5 4.3 1.2.5 1.5-.4 3-2 3.4-1.7.4-3.2-.6-3.4-2.2-.1-1 .3-1.9 1.1-2.4Z"
        fill="var(--accent)"
        opacity="0.45"
      />

      {/* The one cell you clicked: a ring, the way a selected object is drawn. */}
      <circle cx="14.6" cy="17.2" r="2.5" fill="none" stroke="#0b0d10" strokeWidth="1.5" opacity="0.55" />
      <circle cx="14.6" cy="17.2" r="1.1" fill="#0b0d10" opacity="0.7" />
    </svg>
  );
}
