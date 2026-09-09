import { useCallback, useEffect, useRef, useState, useMemo } from "react";
import { filesFromDataTransfer, filesFromInput, resolveSlides } from "../slide/dropResolver";
import { closeSlide, getWorkerCount, openSlide, type OpenProgress } from "../slide/openslideSource";
import type { ResolvedSlide, SlideSource } from "../slide/types";
import { SlideViewer } from "../viewer/SlideViewer";
import { MetadataPanel } from "./MetadataPanel";
import { AnnotationPanel } from "./AnnotationPanel";
import { AnnotationList } from "./AnnotationList";
import { Logo } from "./Logo";
import { Toolbar } from "./Toolbar";
import { AnnotationMenus } from "./AnnotationMenus";
import { activeModel, TISSUE_CLASS } from "../ml/tissueTraining";
import { ModelPanel } from "./ModelPanel";
import { PatchPanel } from "./PatchPanel";
import { SidebarTabs, type SidebarTab } from "./SidebarTabs";
import { TissueModelPanel } from "./TissueModelPanel";
import { SegmentPanel } from "./SegmentPanel";
import type { SegmentController } from "../ml/segmentController";
import { useAnnotations } from "../annotate/store";
import { loadDocument, saveDocument, slideKeyOf } from "../io/persistence";
import { fromGeoJSON, isGeoJSONFile } from "../io/geojson";

interface Stats { tiles: number; avgMs: number; maxMs: number; errors: number }

const fmtBytes = (n: number) => {
  const u = ["B", "KB", "MB", "GB"];
  let i = 0; let v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
};

export function App() {
  const [slides, setSlides] = useState<ResolvedSlide[]>([]);
  const [activeIdx, setActiveIdx] = useState<number | null>(null);
  const [source, setSource] = useState<SlideSource | null>(null);
  const [slideFilter, setSlideFilter] = useState("");
  const [tab, setTab] = useState<SidebarTab>("slides");
  const [loading, setLoading] = useState(false);
  const [openPhase, setOpenPhase] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [segmenter, setSegmenter] = useState<SegmentController | null>(null);
  // Held in a ref, not state: this is passed into the viewer's effect deps, and
  // a value that changes identity every render would tear the viewer down and
  // rebuild it on each keystroke.
  const focuserRef = useRef<((b: [number, number, number, number]) => void) | null>(null);
  const handleFocuser = useCallback(
    (f: ((b: [number, number, number, number]) => void) | null) => {
      focuserRef.current = f;
    },
    [],
  );
  const focusAnnotation = useCallback(
    (bbox: [number, number, number, number]) => focuserRef.current?.(bbox),
    [],
  );
  const [dragging, setDragging] = useState(false);
  const dragDepth = useRef(0);
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);

  const isolated = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;

  const ingest = useCallback((found: ResolvedSlide[]) => {
    if (found.length === 0) {
      setError("No recognisable slides found. Supported: .svs .ndpi .mrxs .tif .scn .vms .dcm");
      return;
    }
    setError(null);
    setSlides((prev) => {
      const seen = new Set(prev.map((s) => s.entryPath));
      const merged = [...prev, ...found.filter((s) => !seen.has(s.entryPath))];
      setActiveIdx((cur) => (cur === null ? prev.length : cur));
      return merged;
    });
  }, []);

  /**
   * Take a slide out of the list and release its decoder handles.
   *
   * Annotations are keyed by slide, not by list position, so they survive and
   * come back if the slide is opened again — which is why this needs no
   * confirmation step.
   */
  const removeSlide = useCallback((index: number) => {
    setSlides((prev) => {
      const victim = prev[index];
      if (!victim) return prev;
      void closeSlide(victim);
      const next = prev.filter((_, i) => i !== index);

      setActiveIdx((cur) => {
        if (cur === null) return null;
        if (next.length === 0) return null;
        // Keep looking at the same slide where possible; otherwise step back.
        if (index < cur) return cur - 1;
        if (index === cur) return Math.min(cur, next.length - 1);
        return cur;
      });
      if (next.length === 0) setSource(null);
      return next;
    });
  }, []);

  const importGeoJSONFile = useCallback(async (file: File) => {
    try {
      const parsed: unknown = JSON.parse(await file.text());
      const store = useAnnotations.getState();
      const { annotations, classes, skipped } = fromGeoJSON(parsed, store.classes);
      if (annotations.length === 0) {
        setError(`No usable features in ${file.name}`);
        return;
      }
      useAnnotations.setState({ classes });
      store.apply({ label: `Import ${annotations.length}`, added: annotations });
      setError(skipped ? `Imported ${annotations.length}, skipped ${skipped} unsupported` : null);
    } catch (err) {
      setError(`Could not read ${file.name}: ${String(err)}`);
    }
  }, []);

  // Open whichever slide is selected.
  useEffect(() => {
    if (activeIdx === null) return;
    const slide = slides[activeIdx];
    if (!slide) return;

    let cancelled = false;

    setLoading(true);
    setError(null);
    setStats(null);
    setSource(null);

    // A wedged decoder worker would otherwise leave the spinner up forever,
    // which is indistinguishable from "the app is broken".
    const OPEN_TIMEOUT_MS = 45_000;
    const timeout = new Promise<never>((_, reject) =>
      window.setTimeout(
        () => reject(new Error(`timed out after ${OPEN_TIMEOUT_MS / 1000}s`)),
        OPEN_TIMEOUT_MS,
      ),
    );

    const describe = (p: OpenProgress) =>
      p.phase === "opening"
        ? "Opening slide…"
        : p.phase === "reading-metadata"
          ? "Reading metadata…"
          : "Preparing decoder…";

    Promise.race([openSlide(slide, (p) => { if (!cancelled) setOpenPhase(describe(p)); }), timeout])
      .then((s) => {
        // Lifetime belongs to the slide cache, not to this effect: closing on
        // every switch would make going back to a slide pay its full open cost
        // again, which is ~29 s on a shallow-pyramid NDPI.
        if (!cancelled) setSource(s);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(
            `Could not open ${slide.name}: ${String(err)}. ` +
              `If the dev server was restarted, reload the page.`,
          );
        }
      })
      .finally(() => { if (!cancelled) { setLoading(false); setOpenPhase(null); } });

    return () => { cancelled = true; };
  }, [activeIdx, slides]);

  // Window-wide drag and drop.
  useEffect(() => {
    const onOver = (e: DragEvent) => { e.preventDefault(); };
    const onEnter = (e: DragEvent) => {
      e.preventDefault();
      if (Array.from(e.dataTransfer?.types ?? []).includes("Files")) {
        dragDepth.current += 1;
        setDragging(true);
      }
    };
    const onLeave = () => {
      dragDepth.current = Math.max(0, dragDepth.current - 1);
      if (dragDepth.current === 0) setDragging(false);
    };
    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      dragDepth.current = 0;
      setDragging(false);
      if (!e.dataTransfer) return;
      void filesFromDataTransfer(e.dataTransfer)
        .then(async (files) => {
          // A dropped .geojson is annotations for the open slide, not a slide.
          const geo = files.filter((f) => isGeoJSONFile(f.path));
          const rest = files.filter((f) => !isGeoJSONFile(f.path));
          if (rest.length > 0) ingest(resolveSlides(rest));
          for (const g of geo) await importGeoJSONFile(g.file);
        })
        .catch((err: unknown) => setError(`Could not read the drop: ${String(err)}`));
    };

    window.addEventListener("dragover", onOver);
    window.addEventListener("dragenter", onEnter);
    window.addEventListener("dragleave", onLeave);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragover", onOver);
      window.removeEventListener("dragenter", onEnter);
      window.removeEventListener("dragleave", onLeave);
      window.removeEventListener("drop", onDrop);
    };
  }, [ingest]);

  // DEV ONLY: lets the test harness push slides through the real code path.
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    window.__slidecraftIngest = (files) => ingest(resolveSlides(files));
    // Hand the test driver *this* module instance. A dynamic import of the
    // same path can resolve to a second copy (Vite serves HMR-versioned URLs),
    // and a test driving a different store than the app proves nothing.
    (window as unknown as Record<string, unknown>).__store = useAnnotations;
  }, [ingest]);

  // ---- annotation document lifecycle -------------------------------------

  // Restore whatever was autosaved for this slide, or start a clean document.
  useEffect(() => {
    if (!source) {
      useAnnotations.getState().resetFor(null);
      return;
    }
    const key = slideKeyOf(source.meta);
    const sidecar = activeIdx !== null ? (slides[activeIdx]?.annotations ?? null) : null;
    let cancelled = false;

    void loadDocument(key).then(async (doc) => {
      if (cancelled) return;
      const store = useAnnotations.getState();
      if (doc && doc.annotations.length > 0) {
        store.loadDocument(key, doc.annotations, doc.classes);
        return;
      }
      store.resetFor(key);

      /**
       * A GeoJSON dropped alongside the slide loads with it.
       *
       * Only when there is nothing saved for this slide already: work done in
       * the app outranks a file on disk, and silently replacing a session's
       * annotations with an older batch result would be the worst kind of
       * data loss — the kind you do not notice.
       */
      if (!sidecar) return;
      try {
        const parsed: unknown = JSON.parse(await sidecar.text());
        if (cancelled) return;
        const { annotations, classes } = fromGeoJSON(parsed, useAnnotations.getState().classes);
        if (!annotations.length) return;
        useAnnotations.setState({ classes });
        useAnnotations.getState().apply({
          label: `Load ${annotations.length} from ${sidecar.name}`,
          added: annotations,
        });
      } catch (err) {
        if (!cancelled) setError(`Could not read ${sidecar.name}: ${String(err)}`);
      }
    });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source]);

  // Autosave, debounced so a brush stroke does not thrash IndexedDB.
  useEffect(() => {
    if (!source) return;
    let timer = 0;
    const unsub = useAnnotations.subscribe((state, prev) => {
      if (state.version === prev.version) return;
      if (!state.slideKey) return;
      window.clearTimeout(timer);
      const key = state.slideKey;
      timer = window.setTimeout(() => {
        void saveDocument({
          slideKey: key,
          slideName: source.meta.name,
          annotations: [...state.items.values()],
          classes: state.classes,
          updatedAt: Date.now(),
        });
      }, 600);
    });
    return () => { window.clearTimeout(timer); unsub(); };
  }, [source]);

  // Number keys pick the active class.
  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      const target = ev.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
      const n = Number(ev.key);
      if (!Number.isInteger(n) || n < 1 || n > 9) return;
      const { classes, setActiveClass } = useAnnotations.getState();
      if (classes[n - 1]) { ev.preventDefault(); setActiveClass(classes[n - 1].id); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const active = activeIdx !== null ? slides[activeIdx] : null;

  const visibleSlides = useMemo(() => {
    const q = slideFilter.trim().toLowerCase();
    return slides
      .map((slide, index) => ({ slide, index }))
      .filter(({ slide }) => !q || slide.name.toLowerCase().includes(q));
  }, [slides, slideFilter]);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <Logo />
          <h1>Slidecraft</h1>
          <span className="phase">PHASE 2</span>
        </div>

        <SidebarTabs
          active={tab}
          onChange={setTab}
          badge={{ slides: slides.length || undefined }}
        />

        <div className="sidebar-scroll">
          {!isolated && (
            <div className="section">
              <div className="note err">
                Not cross-origin isolated — SharedArrayBuffer is unavailable and slides cannot
                open. The server must send COOP/COEP headers.
              </div>
            </div>
          )}

          {tab === "slides" && (
          <div className="dropzone">
            <strong>Drop slides here</strong>
            <span>.svs · .ndpi · .mrxs · .tiff · .scn · .vms · DICOM</span>
            <div className="dropzone-actions">
              <button className="btn" onClick={() => fileInput.current?.click()}>
                Choose files
              </button>
              <button className="btn" onClick={() => folderInput.current?.click()}>
                Choose folder
              </button>
            </div>
          </div>
          )}
          <input
            ref={fileInput}
            data-testid="file-input"
            type="file"
            multiple
            hidden
            onChange={(e) => {
              if (e.target.files) ingest(resolveSlides(filesFromInput(e.target.files)));
              e.target.value = "";
            }}
          />
          <input
            ref={folderInput}
            type="file"
            multiple
            hidden
            // @ts-expect-error non-standard, but the only way to pick a MIRAX data folder
            webkitdirectory=""
            onChange={(e) => {
              if (e.target.files) ingest(resolveSlides(filesFromInput(e.target.files)));
              e.target.value = "";
            }}
          />

          {error && (
            <div className="section"><div className="note err">{error}</div></div>
          )}

          {tab === "slides" && slides.length > 0 && (
            <section className="section">
              <h2>Slides — {slides.length}</h2>
              {/* A batch is hundreds of slides; without a filter the list is a
                  scroll hunt, and without a fixed height it pushes every other
                  panel off the screen. */}
              {slides.length > 6 && (
                <input
                  className="class-edit search"
                  type="search"
                  placeholder="Filter slides…"
                  value={slideFilter}
                  onChange={(e) => setSlideFilter(e.target.value)}
                />
              )}
              <div className="scroll-list">
              {visibleSlides.length === 0 && (
                <div className="picker-hint">Nothing matches “{slideFilter}”.</div>
              )}
              {visibleSlides.map(({ slide: s, index: i }: { slide: ResolvedSlide; index: number }) => (
                <div className="slide-row" key={s.entryPath}>
                  <button
                    className="slide-item"
                    aria-current={i === activeIdx}
                    onClick={() => setActiveIdx(i)}
                    onContextMenu={(e) => { e.preventDefault(); removeSlide(i); }}
                    onKeyDown={(e) => {
                      if (e.key === "Backspace" || e.key === "Delete") {
                        e.preventDefault();
                        removeSlide(i);
                      }
                    }}
                    title={`${s.entryPath}\nRight-click or press Backspace to remove`}
                  >
                    <div className="name">{s.name}</div>
                    <div className="sub">
                      {fmtBytes(s.bytes)}
                      {s.files.length > 1 && ` · ${s.files.length} files`}
                    </div>
                  </button>
                  <button
                    className="slide-remove"
                    aria-label={`Remove ${s.name}`}
                    title="Remove from the list (annotations are kept)"
                    onClick={(e) => { e.stopPropagation(); removeSlide(i); }}
                  >
                    ×
                  </button>
                </div>
              ))}
              </div>
            </section>
          )}

          {active?.warning && (
            <div className="section"><div className="note warn">{active.warning}</div></div>
          )}

          {/* Detecting tissue, correcting it and training on those
              corrections are one activity, so they sit together — the
              correction is the training example. */}
          {tab === "tissue" && source && segmenter && (
            <TissueModelPanel
              source={source}
              onDetectTissue={() => {
                // ensureClass is name-idempotent, so repeated runs reuse the
                // same class instead of stacking duplicates.
                const cls = useAnnotations.getState().ensureClass(TISSUE_CLASS);
                void segmenter.detectTissue(cls.id, {
                  minAreaUm2: 20000,
                  // Whatever the user trained, if they have one turned on.
                  model: activeModel(),
                });
              }}
            />
          )}
          {tab === "cells" && source && segmenter && (
            <SegmentPanel
              onEncode={() => void segmenter.encodeView()}
              onEncodeRoi={(id) => {
                const roi = useAnnotations.getState().items.get(id);
                if (roi) void segmenter.encodeRoi(roi);
              }}
            />
          )}
          {tab === "cells" && source && segmenter && (
            <ModelPanel
              onPrefetchAll={() => void segmenter.prefetchAll()}
              onLoad={() => void segmenter.loadModel()}
            />
          )}
          {tab === "patches" && source && <PatchPanel meta={source.meta} />}
          {tab === "annotate" && source && <AnnotationPanel meta={source.meta} />}
          {/* Slide metadata belongs with the slide it describes. */}
          {tab === "slides" && source && <MetadataPanel meta={source.meta} />}
        </div>

        {/* Pinned: what you are working on stays visible in every mode. */}
        {source && (
          <div className="sidebar-pinned">
            <AnnotationList mpp={source.meta.mppX} onFocus={focusAnnotation} />
          </div>
        )}
      </aside>

      <main className="stage">
        {source ? (
          <>
            <SlideViewer source={source} onStats={setStats} onSegmenter={setSegmenter}
              onFocuser={handleFocuser} />
            <Toolbar mppX={source.meta.mppX} />
            <AnnotationMenus mppX={source.meta.mppX} />
          </>
        ) : (
          !loading && (
            <div className="empty">
              <div>
                <h2>No slide open</h2>
                <p>
                  Drop a whole-slide image anywhere in this window. For MIRAX, drop the folder
                  containing both the <code>.mrxs</code> file and its data directory.
                </p>
              </div>
            </div>
          )
        )}

        {loading && (
          <div className="loading-veil">
            <div>
              <span className="spinner" />
              {openPhase ?? "Opening slide…"}
            </div>
            <div className="veil-note">
              Some formats index themselves on first open; this happens once per slide.
            </div>
          </div>
        )}

        {source && stats && stats.tiles > 0 && (
          <div className="status">
            <span>tiles <b>{stats.tiles}</b></span>
            <span>avg <b>{stats.avgMs.toFixed(0)} ms</b></span>
            <span>max <b>{stats.maxMs.toFixed(0)} ms</b></span>
            <span>workers <b>{getWorkerCount()}</b></span>
            {stats.errors > 0 && <span style={{ color: "var(--err)" }}>errors <b>{stats.errors}</b></span>}
          </div>
        )}
      </main>

      {dragging && (
        <div className="drop-overlay">
          <div className="inner">
            <h2>Drop to open</h2>
            <p>Folders are scanned recursively for slides</p>
          </div>
        </div>
      )}
    </div>
  );
}
