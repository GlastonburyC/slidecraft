/**
 * The sidebar's mode switch.
 *
 * Every capability added its own panel until the sidebar was one endless
 * column and finding the tissue controls meant scrolling past the model list.
 * Grouping them behind tabs trades one scroll for one click, and the click is
 * predictable where the scroll was not.
 *
 * Annotations are deliberately not a tab. They are the thing being worked on
 * whatever mode you are in — you want to see what you just drew while drawing
 * the next one — so the object list stays pinned below the tabs and only the
 * tools above it change.
 */

export type SidebarTab = "slides" | "annotate" | "tissue" | "cells" | "patches";

interface TabDef {
  id: SidebarTab;
  label: string;
  hint: string;
  /** 20x20 stroke icon. */
  path: string;
}

export const SIDEBAR_TABS: TabDef[] = [
  {
    id: "slides",
    label: "Slides",
    hint: "Open slides and switch between them",
    path: "M3.5 5.5h13v9h-13zM3.5 8.5h13M7 5.5v9",
  },
  {
    id: "annotate",
    label: "Annotate",
    hint: "Classes and how annotations are drawn",
    path: "M4 16l1-3.5L13.5 4a1.8 1.8 0 0 1 2.5 2.5L7.5 15z",
  },
  {
    id: "tissue",
    label: "Tissue",
    hint: "Detect tissue, correct it, and train on the corrections",
    path: "M5 12c1.5-4 5-1 6.5-3.5S15 4 15.5 6.5 14 12 11 13.5 4.5 15 5 12Z",
  },
  {
    id: "cells",
    label: "Cells",
    hint: "Click-to-segment and its models",
    path: "M10 4.5a5.5 5.5 0 1 1 0 11 5.5 5.5 0 0 1 0-11zM10 8.6a1.4 1.4 0 1 1 0 2.8 1.4 1.4 0 0 1 0-2.8z",
  },
  {
    id: "patches",
    label: "Patches",
    hint: "Lay a patch grid over an ROI",
    path: "M3.5 3.5h5v5h-5zM11.5 3.5h5v5h-5zM3.5 11.5h5v5h-5zM11.5 11.5h5v5h-5z",
  },
];

/**
 * Counts short enough to sit on an icon.
 *
 * A patch grid runs to tens of thousands, and "23716" on a 60-pixel tab is
 * unreadable and pushes the label out of the way. One significant decimal is
 * enough to answer the only question a badge is asked — roughly how many — and
 * the exact figure stays in the tooltip.
 */
export function compact(n: number): string {
  if (n < 1000) return String(n);
  if (n < 1_000_000) {
    const k = n / 1000;
    return `${k < 10 ? k.toFixed(1).replace(/\.0$/, "") : Math.round(k)}k`;
  }
  const m = n / 1_000_000;
  return `${m < 10 ? m.toFixed(1).replace(/\.0$/, "") : Math.round(m)}M`;
}

export function SidebarTabs({
  active, onChange, badge,
}: {
  active: SidebarTab;
  onChange: (t: SidebarTab) => void;
  /** Small count shown on a tab, e.g. how many slides are open. */
  badge?: Partial<Record<SidebarTab, number>>;
}) {
  return (
    <div className="tabs" role="tablist">
      {SIDEBAR_TABS.map((t) => (
        <button
          key={t.id}
          role="tab"
          className="tab"
          aria-selected={active === t.id}
          title={`${t.label} — ${t.hint}`}
          onClick={() => onChange(t.id)}
        >
          <svg viewBox="0 0 20 20" width="17" height="17" aria-hidden="true">
            <path d={t.path} fill="none" stroke="currentColor" strokeWidth="1.4"
              strokeLinejoin="round" strokeLinecap="round" />
          </svg>
          <span>{t.label}</span>
          {badge?.[t.id] ? (
            <span className="tab-badge" title={badge[t.id]!.toLocaleString()}>
              {compact(badge[t.id]!)}
            </span>
          ) : null}
        </button>
      ))}
    </div>
  );
}
