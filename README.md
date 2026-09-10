<div align="center">

<img src="docs/logo-animated.svg" width="170" alt="Slidecraft">

# Slidecraft

**Whole-slide pathology in the browser — annotate it, train on your corrections, and predict spatial expression from the H&E itself.**

[![License](https://img.shields.io/badge/license-MIT-4fd1c5?style=flat-square)](LICENSE)
[![Node](https://img.shields.io/badge/node-20%2B-4fd1c5?style=flat-square)](package.json)
[![Tests](https://img.shields.io/badge/tests-212%20passing-4fd1c5?style=flat-square)](src/__tests__)
[![Slides](https://img.shields.io/badge/slides-never%20uploaded-8b949e?style=flat-square)](#privacy)

[Tutorial](TUTORIAL.md) · [Roadmap](ROADMAP.md) · [Website](https://glastonburyc.github.io/slidecraft/)

</div>

---

Drop an SVS, NDPI or MIRAX slide onto the page and it opens — read by real
OpenSlide, compiled to WebAssembly. Nothing is uploaded. A 2 GB slide never
leaves your machine.

## Why

Existing tools each cover one slice. QuPath is desktop-only and heavyweight.
Browser viewers are read-mostly. The Python stack (LazySlide, CLAM, Trident) is
batch, with no interactive correction loop. Nothing does the whole thing: open a
slide instantly, annotate it well, run a model, **edit what the model got
wrong**, and retrain from those edits.

## What it does

| | |
|---|---|
| **Opens what you have** | SVS, NDPI, MRXS/MIRAX, BigTIFF, tiled TIFF, DICOM, VMS/VMU — OpenSlide's coverage, because it *is* OpenSlide. Drag a MIRAX folder in, data directory and all. |
| **Annotates properly** | Polygon, freehand, brush and eraser with live boolean ops, resizable ROIs, your own classes, and undo that does not clone the document per stroke. |
| **Learns tissue from you** | Correct a detection and the correction *is* the training example. A classifier fits in about a second, saves with its labels, and extends across slides. |
| **Segments cells on a click** | SAM and SlimSAM in-browser. The encoder runs once per view; each click after that costs milliseconds. |
| **Predicts expression** | DeepSpot-M reads the H&E and answers with a value per gene. Score cell-type signatures, or ask which genes are enriched in a region you drew. |
| **Runs over a folder** | Unattended, one GeoJSON per slide — or a whole transcriptome on your GPU cluster. |
| **Speaks GeoJSON** | QuPath-compatible in both directions, so nothing dead-ends here. |

## Quick start

```bash
git clone https://github.com/GlastonburyC/slidecraft.git
cd slidecraft
npm install
npm run dev
```

Open the URL it prints and drag a slide onto the window. Node 20+; the
folder-batch feature needs Chrome or Edge.

## The loop

```
   Detect tissue  ─────►  Correct what it got wrong
        ▲                          │
        │                          ▼
   Train (~1s)  ◄─────  Corrections are the labels
        │
        ▼
   Run the folder  ─────►  one GeoJSON per slide
```

Correcting the model and teaching it are the same action. That is the whole
idea, and everything else is arranged around it.

<details>
<summary><b>Keys worth knowing</b></summary>

| Key | Does |
|---|---|
| <kbd>Space</kbd> | Show / hide annotations |
| <kbd>V</kbd> | Select — <kbd>Shift</kbd> adds, <kbd>Backspace</kbd> deletes |
| Middle drag | Pan, with any tool active |
| <kbd>B</kbd> | Brush — right-click the tool for size |
| <kbd>O</kbd> | Draw an ROI; drag its corners to resize |
| <kbd>G</kbd> | Click-to-segment |
| <kbd>1</kbd>–<kbd>9</kbd> | Switch class |

</details>

## How it works

```
Browser (cross-origin isolated)
  UI          React + TypeScript + Zustand, patch-based undo
  Viewer      OpenSeadragon (WebGL) for imagery
              + deck.gl overlay for annotations, locked to the OSD viewport
  Slide I/O   @conflux-xyz/openslide-wasm in a worker pool, lazy File.slice reads
  Geometry    polyclip-ts booleans, Flatbush index, marching squares
  ML          ONNX Runtime Web (SAM, DeepSpot-M) + in-browser logistic regression
  Storage     IndexedDB for documents and models, OPFS, Cache Storage for weights
```

Three decisions worth knowing about:

**deck.gl renders annotations; it never handles them.** All pointer input goes
through a transparent overlay, and OpenSeadragon owns the camera. One source of
truth for the viewport.

**The overlay draws the whole document, not the visible subset.** CPU culling
looks cheaper and is not: it makes deck.gl's `data` a different set every frame,
and a cached index buffer outliving its vertex buffer draws triangles between
unrelated polygons — long slivers across the slide.

**Tissue detection identifies fragments before it grows them.** Found at a
confident threshold, bound together so one speckled section is not forty
objects, then each grown into its own territory. Two fragments whose faint halos
meet stay two objects.

## Privacy

Slides are read in your browser. There is no server, no upload, and no
telemetry. A Hugging Face token, if you add one, is stored in that browser and
sent only to `huggingface.co`.

## Cross-origin isolation

`SharedArrayBuffer` is required by openslide-wasm, so the app must be served
with:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

The dev server sets these. Any host you deploy to must too — which is why ONNX
Runtime's wasm is self-hosted rather than loaded from a CDN.

## Development

```bash
npm run dev            # dev server with the required headers
npm test               # 212 tests, headless
npm run test:scripts   # the Python launcher's tests
npm run build          # typecheck + production build
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
build. Gated models (DeepSpot-M, UNI, Virchow2, CONCH) are **not** included —
`scripts/export_deepspot.py` and `scripts/export_onnx.py` convert your own copy
under your own licence. DeepSpot-M's weights are CC-BY-NC-SA-4.0, non-commercial.

> Slidecraft is a research tool. It is not a medical device and is not for
> diagnostic use.
