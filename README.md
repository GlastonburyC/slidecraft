# Slidecraft

Browser-native whole-slide image annotation with a human-in-the-loop tissue
classifier. Drop a slide in, annotate it, train a tissue model from your own
corrections, and run it over a folder of slides — without installing anything
or uploading a single pixel.

Slides are read in the browser by real OpenSlide, compiled to WebAssembly. They
never leave your machine.

## Why

Existing tools each cover one slice of the problem. QuPath is desktop-only and
heavyweight. Browser viewers are read-mostly. The Python ML stack (LazySlide,
CLAM, Trident) is batch, with no interactive correction loop. Nothing does the
whole thing: open a slide instantly, annotate it well, run a model, **edit the
model's output**, and retrain from those edits.

## What it does

- **Opens what you actually have.** SVS, NDPI, MRXS/MIRAX, BigTIFF, generic
  tiled TIFF, DICOM, Hamamatsu VMS/VMU — OpenSlide's format coverage, because it
  *is* OpenSlide. Drag and drop, including a MIRAX folder with its data
  directory.
- **Annotates properly.** Polygon, freehand, brush and eraser with live boolean
  ops, rectangle and ROI, point counter, your own classes, patch-based undo that
  does not clone the document on every brush stroke.
- **Detects tissue**, then lets you correct it — and the corrections are the
  training set.
- **Learns tissue vs not-tissue** from those corrections, across several slides,
  and saves the weights to reuse on the next batch.
- **Segments cells** from a single click (SAM / SlimSAM), with the encoder run
  once per view and each click costing milliseconds.
- **Runs over a folder** unattended, writing one GeoJSON per slide.
- **Speaks GeoJSON**, QuPath-compatible in both directions.

## Getting started

Requires Node 20+ and a Chromium-based browser for the folder-batch feature.

```bash
git clone https://github.com/GlastonburyC/slidecraft.git
cd slidecraft
npm install
npm run dev
```

Then open the URL it prints and drag a slide onto the window.

`npm install` copies ONNX Runtime's wasm assets into `public/ort` — they must be
served from our own origin, see below.

## The tissue model, in one minute

1. **Detect tissue** in the Tissue panel.
2. Press <kbd>V</kbd>, click a wrong detection, <kbd>Shift</kbd>-click to add
   more, then **Mark not tissue**. Mark good ones as tissue.
3. **Add this slide's labels.**
4. Open another slide and repeat — labels accumulate across slides, which is
   what makes the model survive a change of stain or scanner.
5. Name it, **Train and save**.

Detection now uses your model. It is saved, so it is there after a reload and on
every future slide. **Run on a folder…** applies it to a whole directory.

Full walkthrough: [TUTORIAL.md](TUTORIAL.md).

## How it works

```
Browser (cross-origin isolated)
  UI            React + TypeScript + Zustand, patch-based undo
  Viewer        OpenSeadragon (WebGL) for imagery
                + deck.gl overlay for annotations, locked to the OSD viewport
  Slide I/O     @conflux-xyz/openslide-wasm in a worker pool, lazy File.slice reads
  Geometry      polyclip-ts booleans, Flatbush index, marching squares
  ML            ONNX Runtime Web (SAM/SlimSAM) + an in-browser logistic
                regression over per-cell texture features
  Storage       IndexedDB for documents and models, OPFS for embeddings,
                Cache Storage for weights
```

Three decisions worth knowing about:

**Annotations are rendered, never handled, by deck.gl.** All pointer input goes
through a transparent overlay that owns interaction; OpenSeadragon owns the
camera. One source of truth for the viewport.

**The overlay draws the whole document, not the visible subset.** CPU culling
looks cheaper and is not: it makes deck.gl's `data` a different set on every
pan, and a cached index buffer outliving its vertex buffer draws triangles
between unrelated polygons — long slivers across the slide.

**Tissue detection identifies fragments before it grows them.** Fragments are
found at a confident threshold, bound together so one speckled section is not
forty objects, then each grown into its own territory. Two fragments whose faint
halos meet stay two objects.

## Cross-origin isolation

`SharedArrayBuffer` is required by openslide-wasm, so the app must be served
with:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

The dev server sets these. Any host you deploy to must too, and every
cross-origin asset needs `Cross-Origin-Resource-Policy` — which is why ONNX
Runtime's wasm is self-hosted rather than loaded from a CDN.

## Development

```bash
npm run dev        # dev server with the required headers
npm test           # 127 tests, headless
npm run build      # typecheck + production build
npm run typecheck
```

Tissue detection is also checked against real slides, which synthetic images
cannot stand in for:

```bash
scripts/extract-overviews.sh ~/slides   # needs libvips
npm run verify:tissue                   # writes a table + an SVG per slide
```

## Licensing

Slidecraft is MIT. It ships no model weights and no slides.

OpenSlide and glib are **LGPL-2.1**; the app loads the unmodified WebAssembly
build of them. Foundation-model weights (UNI, Virchow2, CONCH, GigaPath) are
licence-gated and are **not** included — `scripts/export_onnx.py` converts your
own copy with your own Hugging Face token, and the token is stored in your
browser and sent only to huggingface.co.
