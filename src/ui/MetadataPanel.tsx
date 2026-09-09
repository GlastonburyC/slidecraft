import type { SlideMeta } from "../slide/types";

const fmtBytes = (n: number) => {
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
};

const fmtPx = (n: number) => n.toLocaleString("en-US");

/** Gigapixels at level 0 — the number that conveys why this is hard. */
const gigapixels = (w: number, h: number) => ((w * h) / 1e9).toFixed(2);

export function MetadataPanel({ meta }: { meta: SlideMeta }) {
  const l0 = meta.levels[0];
  const mpp = meta.mppX;

  return (
    <>
      <section className="section">
        <h2>Slide</h2>
        <dl className="kv">
          <dt>Vendor</dt>
          <dd>{meta.vendor ?? <span className="muted">unknown</span>}</dd>
          <dt>Size on disk</dt>
          <dd>{fmtBytes(meta.bytes)}</dd>
          <dt>Level 0</dt>
          <dd>{fmtPx(l0.width)} × {fmtPx(l0.height)}</dd>
          <dt>Pixels</dt>
          <dd>{gigapixels(l0.width, l0.height)} GP · {meta.levels.length} levels</dd>
          <dt>Objective</dt>
          <dd>{meta.objectivePower ? `${meta.objectivePower}×` : <span className="muted">—</span>}</dd>
          <dt>Resolution</dt>
          <dd>
            {mpp ? `${mpp.toFixed(4)} µm/px` : <span className="muted">—</span>}
          </dd>
          {meta.bounds && (
            <>
              <dt>Scan region</dt>
              <dd>{fmtPx(meta.bounds.width)} × {fmtPx(meta.bounds.height)}</dd>
            </>
          )}
        </dl>
      </section>

    </>
  );
}
