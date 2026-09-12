# Slidecraft roadmap

Where the project actually stands, and where it goes next. This replaces the
original plan, which was written before any of it existed and guessed wrong
about several things — those are called out rather than quietly deleted, since
the reasons are the useful part.

---

## Done

### Phase 0 — Format spike ✅

Slides open in the browser through OpenSlide compiled to WebAssembly: SVS, NDPI,
MRXS/MIRAX, BigTIFF, tiled TIFF, DICOM, VMS/VMU. Drag-and-drop resolves whole
directories, so a MIRAX file arrives with its data folder. Cross-origin
isolation (COOP/COEP) is set up end to end, including self-hosted ONNX Runtime
assets, because under COEP a cross-origin script fails silently.

**What the plan got wrong.** It assumed the risk was format coverage. The real
risk was *pyramid shape*: a slide whose coarsest level is 32× needs a 3.6 MP
read to fill one 512 px tile, which reads on screen as a black slide rather than
a slow one. The fix was to derive the coarsest usable level from the pyramid
rather than from the image, plus an adaptive worker pool and an LRU of open
slides. Time-to-first-tile went from 63 s to under 2 s on the worst slide on
hand.

### Phase 1 — Annotation engine ✅

GeoJSON in level-0 slide pixels, QuPath-compatible both directions. deck.gl
overlay locked to the OpenSeadragon viewport, rendering only — all pointer input
goes through a transparent layer so there is one source of truth for the camera.
Polygon, freehand, brush and eraser with live boolean ops, rectangle, ROI with
corner and edge resize, point counter, user-defined classes with per-class
visibility, patch-based undo, Flatbush hit-testing, IndexedDB autosave.

**What the plan got wrong.** It said cull annotations to the viewport. Doing so
makes deck.gl's `data` a different set every frame, and a cached index buffer
outliving its vertex buffer draws triangles between unrelated polygons — long
red slivers across the slide. The whole document now goes to the GPU and the
viewport clips it; the spatial index earns its keep on hit-testing instead.

### Phase 2 — ROI inference in-browser ✅

Click-to-segment with SAM and SlimSAM: the encoder runs once per view or ROI,
each click costs milliseconds, new masks are clipped against neighbours. Model
registry with BYO ONNX import, weights cached in Cache Storage, Hugging Face
token held in the browser and sent only to huggingface.co. Patch grids specified
in pixels at a pyramid level, laid over an ROI, optionally restricted to
detected tissue, committed as objects or exported as coordinates and GeoJSON.

**Tissue detection turned into its own thing.** It was one line in the original
plan and is now the part that gets used most, so it grew a proper pipeline:
fragments identified at a confident threshold, bound together so one speckled
section is not forty objects, then each grown into its own territory so two
fragments whose faint halos meet stay two objects.

### Phase 2.5 — The human-in-the-loop tissue classifier ✅

Not in the original plan at all, and it should have been: the first thing anyone
does with a detector is disagree with it.

Correcting a detection *is* the training example. Mark wrong regions as "Not
tissue", right ones as "Tissue", add the slide's labels, train. A logistic
regression over twelve per-cell features — colour, and local variation at two
scales — fits in about a second, saves to IndexedDB with the labelled cells
beside it, and can be extended later by opening another slide and retraining on
everything.

Texture is the point. An artefact is pale *and smooth*, because it is a smudge
on glass; faint tissue is pale *and textured*. No brightness threshold separates
those, which is why the colour rule keeps taking dust and bubbles for tissue.

Once it is good enough, **Run tissue classifier on every slide in a folder**
writes one GeoJSON per slide beside it, and dropping that folder back in
reattaches each result to its slide by name.

---

### Phase 3 — The prediction loop ✅

**Built:** patch embedding with UNI2-h and Virchow2, cached in OPFS keyed by
slide, encoder, level and patch position, so a second pass over the same ROI
encodes only what is new. Annotations become patch labels — whatever you draw
is the training set, with unlabelled tissue left unlabelled rather than treated
as background. A softmax head fits over the frozen vectors in well under a
second, validated on whole spatial blocks, and predicts across the ROI as a
class heatmap with a confidence threshold. Disagree, redraw, retrain.

Two encoder details are not recoverable from the hub and are named explicitly
in the exporter: UNI2-h needs its own timm configuration, and Virchow2's
embedding is the class token concatenated with the mean of the patch tokens —
2560-d, not the 1280-d its bare forward returns.

Patching became a tool rather than a panel along the way: press <kbd>T</kbd> and
drag, and the region tiles at the chosen size, clipped to detected tissue. The
grid arriving with the gesture is the difference between checking a grid and
discovering after an hour of encoder time that it sat half on glass.

**Left:** an active-learning queue that ranks unlabelled patches by margin and
diversity, and persisting a trained head so it can be reused on the next slide.

### Phase 3 — original scope

The classifier proved the loop on one binary question. The same machinery
generalises: patch embeddings → a light trainable head → a class heatmap over an
ROI, corrected and retrained in seconds because embeddings are cached.

- Patch embeddings via ONNX/WebGPU, cached in OPFS keyed by
  `(slide, level, x, y, size, model)` so they are computed once.
- Annotations → patch labels; head trained in-browser on frozen fp16 features.
- Live heatmap, threshold slider, accept/reject, brush correction, retrain.
- Metrics with spatial-block validation, and an active-learning queue that ranks
  unlabelled patches by margin and diversity.

The pieces already exist — patch grids, the embedding cache, a trainer that
fits in a second, and the correction UI. What is missing is an encoder wired to
the grid.

### Phase 3½ — Virtual spatial transcriptomics ✅

Expression predicted from the H&E itself, via DeepSpot-M. This was not in the
original plan at all, and is now the half of the app people ask about first.

**Built:** a whole-slide GPU path — `scripts/predict_expression.py`, which
writes an `.expression.bin` beside the slide that the browser loads with it.
One flag submits the same command to Slurm over SSH using your own keys and
agent, watches the queue and brings the result back. The cluster uses the
browser's own tissue detector, ported line for line in `scripts/tissue.py`, so a
patch chosen on a GPU node is one Slidecraft would have chosen.

Reading a map is where most of the design went, because a single predicted gene
is mostly its own error. Fourteen cell-type modules ship built in and 111 more
are derived from the CELLxGENE Census, each averaging its genes after
standardising them so an abundant one cannot carry the module. Draw round a
region and rank the cell types in it; drag an arrow and rank what rises and
falls along it. A module the map cannot cover is hidden rather than scored on
two genes and presented as if it meant something.

A whole transcriptome is 32,000 patches by 19,338 genes — 1.2 GB — and loads in
under a second because the values stay in the half precision they arrived in and
are decoded one gene at a time.

**Left:** blending overlapping patches, so `--stride` produces a finer map
instead of overdrawing; the same raised-cosine weighting the trained-head path
already uses. And moving whole-transcriptome ranking off the main thread — six
seconds is honest about itself now, but a worker would be better.

## Next

### Phase 4 — Sidecar and HPC (half done)

**Built:** the batch half. `--submit HOST` copies the script and the slide,
submits to Slurm, polls, and brings the map back; `--remote-slide` uses a copy
already on cluster storage; `--dry-run` prints everything and sends nothing. No
password is ever handled — authentication is delegated to your SSH agent and
config, which is also why a browser cannot do this part itself: page JavaScript
cannot open a TCP connection, so SSH from the tab is not a thing that exists.

**Left:** the interactive half. A pip-installable FastAPI service that runs locally or as a Slurm job on a GPU
node, reached through an SSH port-forward using your existing keys and agent.
Patch-push by default so the slide never leaves your machine; cluster-side slide
reading when the WSI already lives there. `Cross-Origin-Resource-Policy:
cross-origin` from day one, or every fetch fails silently under COEP.

### Phase 5 — Cells at scale

Nucleus and cell instance segmentation over a whole ROI rather than one click at
a time, with per-object editing, measurements, and stain normalisation.

---

## Phase 6 — Measured spatial transcriptomics 🧬

Not to be confused with Phase 3½, which is *predicted* expression from the H&E.
This is the real assay: bringing Visium and Xenium runs into the same viewer,
with everything already built applying to them — annotate, detect tissue, patch,
correct, train. The two meet at the obvious question, which is whether a
prediction agrees with a measurement on the same tissue.

The premise is that spatial data is a *slide problem* before it is an omics
problem. The expression matrix is well served by existing tools; what is not
served is looking at expression on the tissue, at full resolution, next to the
H&E, while drawing on it — which is what this app already does.

### No image registration. Read the transforms that already exist.

This is the design decision the phase turns on, so it is worth being explicit:
**Slidecraft does not compute alignments.** Both platforms already emit the
mapping, and re-deriving it would be inventing a second answer that can silently
disagree with the one every other tool in the lab is using.

**Visium is already in H&E pixel space.** Space Ranger writes
`spatial/tissue_positions.csv` with `pxl_row_in_fullres` and
`pxl_col_in_fullres` — the spot centres *in the full-resolution image you gave
it*. `spatial/scalefactors_json.json` carries `spot_diameter_fullres` (spot
diameter in those same pixels; for Visium HD, the bin side length) plus
`tissue_hires_scalef` / `tissue_lowres_scalef` for the downsampled previews. So
a Visium spot is a circle at a known pixel centre with a known pixel diameter,
in the frame Slidecraft already works in. There is nothing to align.

**Xenium is in microns, with a known scale.** `transcripts.parquet` gives
`x_location` / `y_location` / `z_location`, `cell_boundaries.parquet` gives
`vertex_x` / `vertex_y` — all in microns in the morphology image's frame.
Dividing by the run's pixel size (0.2125 µm at the time of writing; read it from
the run rather than hardcoding it) converts to morphology pixels.

**An added H&E carries its own matrix.** A post-Xenium H&E is imaged on a
different microscope and is *not* registered to the morphology image. 10x's
answer is an image alignment file: a 3×3 affine whose last row is `[0,0,1]`,
produced by Xenium Explorer's alignment or by a community tool, and consumed by
Xenium Ranger. Slidecraft imports that CSV and applies it at the viewport. If a
user has no matrix, the honest response is to say so and show the morphology
image alone — not to guess one.

### What has to arrive

| Platform | Comes as | Coordinate frame |
|---|---|---|
| Visium | H&E TIFF, `scalefactors_json.json`, `tissue_positions.csv`, filtered matrix (h5/mtx) | Already full-res H&E pixels |
| Visium HD | `binned_outputs/square_002um/spatial/…` | The same, at bin resolution |
| Xenium | `morphology_mip.ome.tif`, `transcripts.parquet`, `cells.parquet`, boundaries, `experiment.xenium` | Microns; ÷ pixel size → morphology pixels |
| Xenium + H&E | the above, plus a 3×3 alignment CSV | Matrix maps the added image into the Xenium frame |

### Sub-phases

**6a — Read the images.** Xenium morphology is a 16-bit grayscale OME-TIFF,
JPEG-2000 compressed, and OpenSlide does not read it. This needs a second
`SlideSource` over a TIFF/Zarr reader with channel selection, per-channel
windowing, and false colour — DAPI is a single 16-bit plane and must be
windowed, not treated as RGB. `SlideSource` was built as an interface for
exactly this swap.

**6b — Load the coordinate frames.** Parse `scalefactors_json.json` and
`tissue_positions.csv`; parse `experiment.xenium` for pixel size; parse an
alignment CSV when one is present. One `SpatialFrame` per dataset that converts
platform coordinates to slide pixels, and is the only place that conversion
happens. Show the user which transform is in force and where it came from, so a
misalignment is attributable rather than mysterious.

**6c — Overlay the measurements.** Visium spots as circles at
`spot_diameter_fullres`, Xenium transcripts as points, cell boundaries as
polygons — all on the existing deck.gl overlay, in slide pixels, which is what
everything else already speaks. Colour by a chosen gene or a precomputed
cluster, with a legend. Millions of transcripts means binning and
level-of-detail: a scatter layer at whole-slide zoom is not readable even when
it renders.

**6c½ — Virtual spatial (built).** Expression predicted from the H&E itself,
without an assay on this section. DeepSpot-M reads a 224 px tile at ~20× and
answers with a value per gene, using a frozen pathology encoder and a
cross-attention decoder in which each gene is a query — which is what makes it
tractable here, because asking for eight genes costs a fraction of asking for
19,338, and the exporter bakes a chosen subset into the graph.

Scope is one ROI or the whole detected tissue; the ROI case is the one that gets
used, being quick enough to iterate on. Results colour the patch grid, export as
CSV, and are labelled *predicted* everywhere they appear — a map that looks like
an assay and is not is the main thing that can go wrong with this feature.

The weights are gated to non-commercial academic use and are released as PyTorch
safetensors, so the conversion to ONNX is a one-off script the user runs with
their own token. Nothing is bundled.

**6d — The loop, on expression.** Everything already built then applies. Draw an
ROI, patch it, and the patches carry expression as well as pixels: label regions
by morphology and ask what is differentially expressed in them; or label by
expression and train the tissue classifier to recognise it from H&E alone. The
second direction is the interesting one — a morphology model supervised by
transcriptomics, trained in the browser on your own sections.

**6e — Export.** Annotations out as GeoJSON with per-region expression
summaries, and as an AnnData-friendly table keyed by spot or cell id, so the
work returns to scanpy or Seurat rather than dead-ending here.

### Risks

| Risk | Mitigation |
|---|---|
| A Xenium run is tens of GB; the browser cannot hold it | Read lazily from disk exactly as slides already are; bin transcripts on first load and cache in OPFS |
| The user has no alignment matrix for their H&E | Say so, and show the morphology image alone. Never fabricate a transform |
| A stale or wrong matrix makes misalignment look like biology | Name the transform's source in the UI, make the overlay toggleable, keep the raw image one key away |
| 16-bit multi-channel rendering is not what OpenSlide does | Separate `SlideSource`; the interface was built for this |
| Gene selection across 20k genes in a browser | Index once on load; search rather than a dropdown |

### The mark

Spatial gets its own glyph: a DNA helix, drawn at the same line weight and on the
same grid as the slide mark so the two read as one family — the slide for the
morphology, the helix for what is being expressed on it.

## Verification, throughout

- **Formats:** golden fixture per format, tile hashes against `openslide-python`.
- **Tissue detection:** synthetic cases for the mechanisms, plus
  `npm run verify:tissue` over a real slide set, which writes an SVG overlay per
  slide — the question "is the boundary on the tissue" has to be looked at.
- **Interop:** round-trip through QuPath, geometry within 1e-6.
- **Coordinate frames (Phase 6):** a spot at a known `pxl_*_in_fullres` lands on
  that pixel; a transcript at a known micron position lands where the pixel size
  says. Round-trip a published dataset and check against its own figures.
- **Manual:** a real slide from your own set at the end of every phase. It is
  the only test that says whether it feels fast.
