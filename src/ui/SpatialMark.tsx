/**
 * The spatial mark: a DNA helix.
 *
 * Drawn at the same stroke weight and on the same 32-unit grid as the slide
 * logo, so the two read as one family — the slide stands for the morphology,
 * the helix for what is being expressed on it. Two strands crossing, with the
 * base pairs between them picked out in the accent colour the way a selected
 * object is.
 *
 * The rungs are individual lines rather than a hatch pattern because their
 * spacing is the thing that makes it read as a helix rather than as a ribbon;
 * at 16 px the outer two on each side are what the eye actually resolves.
 */
export function SpatialMark({ size = 22 }: { size?: number }) {
  // Sampled from two counter-phase sines, so the rungs sit where the strands
  // actually are rather than at guessed positions.
  const rungs = [6, 9.5, 13, 16, 19, 22.5, 26].map((y) => {
    const t = ((y - 4) / 24) * Math.PI * 2;
    const dx = Math.sin(t) * 6.5;
    return { y, x1: 16 - dx, x2: 16 + dx };
  });

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      aria-hidden="true"
      style={{ flex: "none", display: "block" }}
    >
      {rungs.map((r, i) => (
        <path
          key={i}
          d={`M${r.x1.toFixed(2)} ${r.y}H${r.x2.toFixed(2)}`}
          stroke="var(--accent)"
          strokeWidth="1.3"
          strokeLinecap="round"
          // The rungs nearest a crossing are foreshortened to almost nothing,
          // so fading them keeps the twist legible instead of cluttered.
          opacity={Math.min(0.9, 0.15 + Math.abs(r.x2 - r.x1) / 13)}
        />
      ))}

      {/* Two strands, a half-period apart. */}
      <path
        d="M9.5 3c0 6.5 13 6.5 13 13S9.5 22.5 9.5 29"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        opacity="0.85"
      />
      <path
        d="M22.5 3c0 6.5-13 6.5-13 13s13 6.5 13 13"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        opacity="0.5"
      />
    </svg>
  );
}
