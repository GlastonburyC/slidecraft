# Slidecraft — browser-native WSI annotation + human-in-the-loop ML

*(working name; placeholder until you pick one)*

## Context

You work with WSIs daily across SVS, NDPI, TIFF/BigTIFF and MRXS. Today there is no browser tool that
does the full loop: open a slide instantly, annotate it well, run a model, **edit the model's
predictions**, and retrain from those edits. Existing options each cover one slice — QuPath is
desktop-only and heavyweight, OpenSeadragon-based viewers are read-mostly, and the ML tooling
(LazySlide, CLAM, Trident) is Python-batch with no interactive correction loop.

The goal is a lightweight, sleek, zero-install web app where slides and annotations are drag-and-drop,
annotations are GeoJSON, and modern histology foundation models are usable — including on your Slurm
GPU nodes over SSH.

**Two decisions from you shape this plan:**
1. **ROI-scoped, not whole-slide.** Prediction, correction and retraining all happen inside a
   user-chosen ROI. Correcting a full-slide prediction with thousands of errors is not usable, so
   whole-slide inference is an explicit, deliberate "expand" action *after* the head is trusted — never
   the default.
2. **HPC over SSH.** The GPU backend is a Python sidecar that can run locally *or* as a Slurm job on a
   GPU node, reached through an SSH port-forward.

**Assumption I am making** (flag if wrong): the day-one HITL task is **region/tissue classification** —
patch embeddings → light trainable head → class heatmap over the ROI. This is the only formulation that
retrains in seconds and therefore the only one that feels interactive. Nucleus/cell instance
segmentation lands in Phase 5 and reuses the same annotation and correction machinery.

---

## Architecture

```
┌── Browser (cross-origin isolated, COOP/COEP) ───────────────────────────┐
│                                                                         │
│  UI  React + TS + Tailwind/Radix + Zustand   ← command-pattern undo      │
│   │                                                                     │
│   ├─ Viewer     OpenSeadragon 5 (WebGL drawer)  ← imagery                │
│   │             + deck.gl overlay               ← annotations/heatmaps   │
│   │                                                                     │
│   ├─ Slide worker pool ── @conflux-xyz/openslide-wasm ── File/FileEntry  │
│   │                       (SVS NDPI MRXS BigTIFF DICOM VMS/VMU)         │
│   ├─ Geometry worker ──── polygon boolean ops, simplify, mask→poly       │
│   ├─ Inference worker ─── ONNX Runtime Web (WebGPU EP)  ← ROI-scale      │
│   └─ Trainer ──────────── head training on cached embeddings (seconds)   │
│                                                                         │
│  Storage   OPFS: tile + embedding cache │ IndexedDB: annotations, heads  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ HTTP + bearer token, over SSH tunnel
                               ▼
┌── slidecraft-sidecar (optional, Python) ────────────────────────────────┐
│  FastAPI · torch · timm · LazySlide/wsidata · openslide                  │
│  runs: laptop GPU  |  lab workstation  |  Slurm GPU node via sbatch      │
└─────────────────────────────────────────────────────────────────────────┘
```

The browser is **fully self-sufficient** for viewing, annotation and ROI-scale inference. The sidecar
is strictly an accelerator; every UI affordance works without it, just slower and with smaller encoders.

---

## Key design decisions

### Slide I/O — `@conflux-xyz/openslide-wasm`
Real OpenSlide compiled with Emscripten, so format coverage is OpenSlide's, not ours: SVS, NDPI,
MRXS/MIRAX, BigTIFF, generic tiled TIFF, DICOM, Hamamatsu VMS/VMU. Critically it accepts `File[]` /
`FileEntry[]`, which is what makes **MRXS** work — drop the `.mrxs` *and* its sibling data directory and
both go in together. Also supports HTTP byte-range reads, so remote/S3 slides come free later.

- Drag-and-drop uses `DataTransferItem.webkitGetAsEntry()` and recurses directories, so a dropped MRXS
  folder, a bare `.svs`, or a mixed batch all resolve to the right file sets.
- It needs `SharedArrayBuffer` → the app **must** be served with `Cross-Origin-Opener-Policy:
  same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. This is a hard constraint that shapes
  dev server config, hosting, and every cross-origin fetch (including the sidecar — see below).
- Licensing: wrapper MIT, but OpenSlide + glib are **LGPL-2.1**. Fine for a web app shipping the
  unmodified WASM, but must be documented in `THIRD_PARTY.md` with the object files available.

**Go/no-go gate in Phase 0:** benchmark on *your* real slides (one per format, including a large MRXS)
before anything else is built. wasm32 has a 4 GB address space ceiling and MRXS index handling is
memory-hungry. If a format fails, that format falls back to sidecar-served DZI tiles — the
`SlideSource` interface below makes that a swap, not a rewrite.

```ts
interface SlideSource {
  open(files: File[] | FileEntry[] | URL): Promise<SlideHandle>
  levels(): LevelInfo[]          // dims, downsample, tile size
  properties(): Record<string,string>  // mpp-x/y, objective power, vendor, ICC
  readRegion(level: number, x: number, y: number, w: number, h: number): Promise<ImageBitmap>
  associated(name: 'thumbnail'|'label'|'macro'): Promise<ImageBitmap|null>
}
```
Two implementations from day one: `OpenSlideWasmSource` and `RemoteTileSource` (sidecar/DZI/IIIF).

### Viewer — OpenSeadragon 5 + deck.gl overlay
OSD gives mature pyramid scheduling, inertial navigation, rotation, scalebar and filters; v5's WebGL
drawer removes the old canvas2d bottleneck. A custom `TileSource` bridges to the slide worker pool.
Annotations do **not** go through OSD — a deck.gl `OrthographicView` overlay is locked to the OSD
viewport matrix and renders `GeoJsonLayer` (polygons), `ScatterplotLayer` (points) and a `BitmapLayer`
(prediction heatmap) on the GPU. This keeps 10⁵–10⁶ objects interactive, which a DOM/SVG overlay
cannot, and it means the in-memory annotation format *is* GeoJSON — no conversion at export.

### Annotation model — GeoJSON, QuPath-compatible
Canonical coordinates are **level-0 slide pixels** (integer, origin top-left), with `mpp` and an affine
carried in metadata so µm conversion is lossless and reversible.

```jsonc
{ "type": "Feature",
  "geometry": { "type": "Polygon", "coordinates": [[...outer], [...hole]] },
  "properties": {
    "objectType": "annotation",                              // QuPath interop
    "classification": { "name": "Tumour", "colorRGB": -3670016 },
    "measurements": { "Area µm^2": 10432.1 },
    "slidecraft": {                                          // namespaced extras
      "id": "01J…", "source": "model", "modelId": "uni-v1+head-3",
      "confidence": 0.94, "roiId": "01J…", "author": "cg",
      "createdAt": "2026-09-08T…", "reviewState": "accepted", "version": 3
    } } }
```
Round-trips with QuPath both directions (tolerant importer for QuPath's own exports). Spatial index is
Flatbush, rebuilt incrementally; only features intersecting the viewport are uploaded to deck.gl.

### Tools
Polygon · freehand lasso · brush + eraser (adjustable radius, live boolean against existing objects) ·
rectangle/ellipse ROI · point counter · magic-wand (flood fill in stain-deconvolved space) · scissors
(split) · merge · boolean ops (union/subtract/intersect) · simplify · class palette with `1`–`9`
hotkeys. Brush strokes are rasterised in a scratch texture and vectorised via marching squares on
commit, so brushing stays 60 fps but the stored object is still a polygon.

Undo/redo is a **command stack** (not state snapshots) — required because a brush stroke over a 50k-vertex
polygon must not clone the document.

### The HITL loop (the core of the product)

```
1. Draw ROI ──► 2. Tile ROI into patches at chosen mpp/stride
                     │
3. Embed patches ────┤  browser: ONNX/WebGPU  ·  sidecar: torch on GPU
   → cache in OPFS   │  keyed (slideHash, level, x, y, size, modelId) — never recomputed
                     ▼
4. Label: annotations inside ROI → patch labels (majority/centroid/area-weighted)
                     ▼
5. Train head in-browser: logistic regression or 2-layer MLP over frozen fp16 embeddings
   ~10³–10⁵ × 768–1536 dims → plain Adam loop on typed arrays → **< 1 s**
                     ▼
6. Predict over the whole ROI → class heatmap + thresholded polygons on the overlay
                     ▼
7. Correct: brush over wrong regions, accept/reject objects, drag the confidence threshold
                     ▼
8. Retrain (goto 5) — instant, because embeddings are cached. This is why it feels alive.
                     ▼
9. Active learning: rank unlabelled patches by margin/entropy + k-center-greedy diversity on
   embeddings → "review these 12 tiles" queue → user labels only what matters
                     ▼
10. Trust it? → "Expand to slide": batch the trained head over the full slide (sidecar if present)
```

The insight that makes this work: **embed once, retrain endlessly.** The expensive frozen encoder runs
once per patch; the trainable part is a few hundred KB of head weights. Retraining is therefore a
sub-second operation, and the user can iterate dozens of times per ROI.

### Compute backends — including Slurm over SSH

`slidecraft-sidecar`, a pip-installable FastAPI service, in three deployment shapes:

| Mode | Command | Use |
|---|---|---|
| Local | `slidecraft-sidecar serve --port 8787` | laptop / workstation GPU |
| Slurm | `slidecraft-sidecar slurm --partition gpu --gres gpu:1 --time 4:00:00` | GPU node on your cluster |
| None | — | browser-only, ONNX/WebGPU |

The Slurm path is a local launcher CLI that: submits an `sbatch` job running the server on a free port,
polls `squeue` until it is `RUNNING`, reads back the allocated node hostname and a random bearer token,
opens `ssh -N -L 8787:<node>:<port> <login-host>` using **your existing SSH config, keys and agent**, and
prints `http://localhost:8787` + token to paste into the app's Compute panel. It never handles or stores
your password — authentication is delegated entirely to your SSH agent/config.

Two ways the GPU gets pixels, and both are supported:
- **Patch push (default, matches ROI-first).** Browser cuts the ROI's patches with openslide-wasm and
  POSTs them as a packed JPEG batch. A 4096² ROI at 20× ≈ 256 patches ≈ 8 MB — trivial over a tunnel,
  and the slide never leaves your laptop.
- **Cluster-side slide.** If the same WSI already lives on cluster storage, pass its path and the
  sidecar reads it directly with LazySlide/openslide — the right choice for "expand to slide".

```
GET  /health                → {gpu, vram_free, models, queue_depth, version}
GET  /models                → registry: encoders + segmenters, dims, licences
POST /embed                 → packed patch batch + modelId → fp16 embeddings
POST /segment               → patches → instance polygons as GeoJSON
POST /slide/embed           → {path, roi, level, patch_size, stride, model} → streamed embeddings
WS   /progress              → job progress, so long runs are cancellable from the UI
```

**COEP gotcha:** because the app is cross-origin isolated, the sidecar must return
`Cross-Origin-Resource-Policy: cross-origin` alongside CORS headers or every fetch fails silently.
Building this into the sidecar from the start avoids a very confusing afternoon.

### Encoder weights — bring your own
Ship an openly-licensed default (Phikon / a distilled ViT-B) as ONNX so the app is useful out of the
box. For gated models (UNI, CONCH, Virchow2, GigaPath): a `scripts/export_onnx.py` converter plus an
import UI that registers a local `.onnx` + config, and a sidecar path that loads them from HF with
**your** token. No gated weights ever enter the repo. The model registry records dim, patch size,
target mpp, normalisation and licence, so heads can never be silently applied to the wrong encoder.

---

## Repository layout

```
slidecraft/
  apps/web/                 Vite + React + TS
    src/slide/              SlideSource iface, openslide-wasm worker pool, tile cache (OPFS)
    src/viewer/             OSD setup, custom TileSource, deck.gl overlay sync
    src/annotate/           geometry ops, tools, command stack, Flatbush index, class palette
    src/ml/                 ORT session mgmt, patch grid, embedding cache, head trainer, AL sampler
    src/compute/            backend abstraction: LocalBackend | SidecarBackend, health/failover
    src/io/                 drag-drop resolver, GeoJSON import/export, QuPath compat, project files
    src/ui/                 Radix + Tailwind, panels, command palette, keyboard map
  packages/geojson-schema/  shared TS types + zod validators + QuPath mapping (used by both sides)
  services/sidecar/         FastAPI app, model registry, LazySlide integration, slurm launcher CLI
  scripts/export_onnx.py    HF/timm → ONNX converter for BYO foundation models
  fixtures/                 tiny golden slides per format + expected tile hashes
```

---

## Phases

**Phase 0 — Format spike (go/no-go).** Vite skeleton with COOP/COEP dev server, openslide-wasm in a
worker pool, drag-drop resolver handling directories, OSD custom TileSource, metadata + associated-image
panel. Benchmark every one of your formats. *Deliverable: drop any slide, pan/zoom smoothly, see MPP and
objective power.* Nothing else is built until this passes on real MRXS and NDPI.

**Phase 1 — Annotation engine.** GeoJSON model, deck.gl overlay, all drawing tools, boolean ops, brush
rasterise→vectorise, command-stack undo, class palette, Flatbush hit-testing, IndexedDB autosave,
GeoJSON import/export with QuPath round-trip. *Deliverable: a genuinely good annotator, ML aside.*

**Phase 2 — ROI inference in-browser.** ROI concept, patch grid at target mpp, ONNX/WebGPU encoder in a
worker, OPFS embedding cache, tissue/background detection, model registry + BYO ONNX import.
*Deliverable: embed an ROI and see a tissue mask.*

**Phase 3 — The HITL loop.** Annotations→patch labels, in-browser head trainer, live prediction heatmap,
threshold + accept/reject + brush correction, retrain button, metrics panel (per-class P/R/F1, confusion
matrix, train/val split by spatial block to avoid leakage), active-learning review queue.
*Deliverable: the loop in your request, end to end, on one ROI.*

**Phase 4 — Sidecar + HPC.** FastAPI service, model registry, `/embed` + `/segment` + `/slide/embed`,
Slurm launcher with SSH tunnel, Compute panel with backend status and failover, LazySlide integration,
"expand to slide". *Deliverable: same UI, your cluster's GPU, real UNI/Virchow embeddings.*

**Phase 5 — Depth.** Nucleus/cell instance segmentation with per-object editing, multi-slide projects,
measurements and stats export, stain normalisation, session/report export, ICC colour handling.

---

## Verification

- **Format fidelity:** golden fixture per format; tile-hash comparison against native
  `openslide-python` for the same regions. Any mismatch is a bug, not a rounding difference.
- **Perf budgets, measured in CI:** time-to-first-tile < 1.5 s on a 2 GB SVS; ≥ 60 fps pan/zoom with
  10⁵ annotation objects on screen; ROI embed ≥ 40 patches/s in-browser (WebGPU, ViT-B), ≥ 500/s via
  sidecar; head retrain < 1 s at 10⁴ × 1024.
- **Interop:** automated round-trip — export from Slidecraft → open in QuPath → re-export → diff
  geometry within 1e-6 and confirm classifications and holes survive.
- **HITL correctness:** synthetic ROI with a known ground-truth pattern; assert that adding corrections
  monotonically improves held-out F1, and that spatial-block validation is actually blocking leakage.
- **E2E (Playwright):** drop slide → draw ROI → label two classes → embed → train → predict → correct →
  retrain → export GeoJSON, asserted on file contents.
- **Manual, with you:** a real slide from your own set at the end of each phase — the only test that
  tells us whether it actually feels fast.

## Principal risks

| Risk | Mitigation |
|---|---|
| openslide-wasm chokes on large MRXS/NDPI (wasm32 4 GB limit) | Phase 0 gate on real files; `SlideSource` lets any format fall back to sidecar tiles |
| COOP/COEP breaks third-party embeds and sidecar fetches | Own the header story from day 0; sidecar sends CORP `cross-origin` |
| JPEG2000-compressed SVS needs openjpeg in the WASM build | Verify in Phase 0; if absent, rebuild the WASM with openjpeg or fall back to sidecar |
| WebGPU absent (Safari/older) | ORT WASM EP fallback with SIMD+threads; sidecar covers the rest |
| Head trained on encoder A silently applied to encoder B | Model registry binds head→encoder id; refuse mismatches |
| Cluster policy blocks long-lived services / port-forwards | Sidecar also runs as a batch job writing results to disk; document both |
