# Slidecraft tutorial

A full pass through the tool: open a slide, annotate it, teach it what tissue
looks like on your material, and run that over a folder.

Nothing is uploaded. Slides are read in your browser by OpenSlide compiled to
WebAssembly, so a 2 GB SVS never leaves the machine.

---

## 1. Open a slide

Drag a slide anywhere onto the window, or use **Choose files**.

For **MIRAX**, drag the *folder* containing both `Foo.mrxs` and its `Foo/` data
directory. The `.mrxs` file holds no pixels; dropping it alone cannot work, and
Slidecraft will say so rather than failing quietly.

Formats: `.svs` `.ndpi` `.mrxs` `.tif`/`.tiff`/`.btf` `.scn` `.vms`/`.vmu`
DICOM.

Drop many at once — the **Slides** list is filterable once there are more than a
handful. Right-click a slide, or press <kbd>Backspace</kbd> on it, to remove it
from the list; its annotations are kept.

## 2. Get around

| | |
|---|---|
| Zoom | Scroll wheel |
| Pan | <kbd>H</kbd> tool, or hold the **middle mouse button** with any tool |
| Switch panel | The tabs at the top of the left panel: Slides, Annotate, Tissue, Cells, Patches |
| Show/hide annotations | <kbd>Space</kbd> |
| Select | <kbd>V</kbd> — <kbd>Shift</kbd>-click adds, <kbd>Backspace</kbd> deletes |

<kbd>Space</kbd> is worth the muscle memory: comparing a boundary against the
tissue underneath it is the most repeated action in the app.

## 3. Annotate

Pick a tool from the rail. Each has a one-key shortcut, shown on the button.

| Key | Tool | Notes |
|---|---|---|
| <kbd>R</kbd> | Rectangle | |
| <kbd>P</kbd> | Polygon | Click vertices; <kbd>Enter</kbd> or double-click closes, <kbd>Esc</kbd> cancels |
| <kbd>F</kbd> | Freehand | Drag to trace |
| <kbd>B</kbd> | Brush | Merges with the same class underneath. **Right-click the tool for size.** |
| <kbd>E</kbd> | Eraser | Subtracts from anything it touches |
| <kbd>N</kbd> | Point | Counting points |
| <kbd>O</kbd> | ROI | The working frame for patching and prediction |
| <kbd>G</kbd> | Click-to-segment | One click per cell |

**Classes are yours.** There are no defaults. Right-click an object to name a
class, or use the **Annotate** tab; <kbd>1</kbd>–<kbd>9</kbd> switch between the
first nine. Each class has a visibility toggle — hiding the tissue class is the
quickest way to see the nuclei drawn on top of it, and hiding is not deleting:
the objects stay, and stay exported.

**ROIs resize.** Select one and drag any corner or edge handle.

## 4. Detect tissue

In the **Tissue** panel, click **Detect tissue**.

With no trained model this uses a colour rule: it scores every cell of the slide
overview on saturation and darkness, splits at the valley of that histogram,
then grows the detection into faded tissue that connects to something confident.
Separate fragments come out as separate objects.

It is genuinely good on well-stained material and it will make mistakes on
yours — dust, bubbles, mounting medium, pen marks, a faded section. That is what
the next step is for.

## 5. Teach it

The correction *is* the training example. There is no separate labelling chore.

1. Press <kbd>V</kbd>.
2. Click a wrong detection. <kbd>Shift</kbd>-click to add more.
3. **Mark not tissue.**
4. Do the same for correct ones with **Mark tissue**.
5. **Add this slide's labels.**

You need at least one region of each kind. Three or more of each gets you a
held-out score instead of "no score" — Slidecraft will not report a number it
cannot honestly measure.

You do **not** need to label blank glass. Confident background is sampled
automatically, because it is unambiguous and it is most of every slide.

### Use more than one slide

Open another slide, correct its detection, and **Add this slide's labels**
again. Labels accumulate — the **Training set** box lists every slide
contributing, and `×` drops one whose labels turned out to be wrong.

This matters more than it sounds. A model fitted on a single slide learns that
slide's stain and scanner as much as it learns tissue, and the first slide from
another batch undoes it. Two or three slides across your range of staining is
usually enough.

### Train

Name it something you will recognise — "Colon H&E, Aperio" — and **Train and
save**.

Training is a logistic regression over a dozen per-cell features: colour, and
local variation at two scales. Texture is the point. An artefact is pale *and
smooth*, because it is a smudge on glass; faint tissue is pale *and textured*.
No threshold on brightness can separate those, and that is exactly what the
colour rule keeps getting wrong.

It fits in about a second, so retraining after another handful of corrections is
not a chore.

## 6. Use it

**Use the model when detecting tissue** is ticked automatically. **Detect
tissue** now runs your classifier. The panel tells you which model produced the
current result, and which features it leans on.

The model is saved in the browser, so it survives a reload and applies to every
future slide. Switch between saved models by clicking one.

Still wrong somewhere? Mark those regions, **Update this slide's labels**, train
again.

## 7. Run it over a folder

Click **Run on a folder…**, choose a directory, **Run**.

Each slide gets a `<slide>.geojson` written next to it. Slides are opened and
closed one at a time; a slide that fails is recorded and the queue continues, so
it can be left alone.

Drop that folder back into Slidecraft later and every slide arrives with its
annotations already attached — the sidecar is matched to the slide by name. It
never overwrites annotations you already saved in the app.

> Needs Chrome or Edge. Safari and Firefox cannot write files back into a chosen
> folder; the dialog says so rather than failing silently.

## 8. Segment cells

Press <kbd>G</kbd>, then **Encode view** in the Click-to-segment panel. The
encoder runs once over what you are looking at; each click after that costs
milliseconds.

- Click a cell centre for its outline.
- <kbd>Shift</kbd>-click adds to the current mask, <kbd>Alt</kbd>-click cuts away.
- <kbd>Backspace</kbd> throws away a bad mask, or takes back the one just kept.
- **Never overlap a cell already segmented** clips each new mask against its
  neighbours.

Pick a model under **Models**: SlimSAM-77 is smallest and fastest, SAM ViT-B is
the best quality for a bigger download. Weights are cached after the first
fetch.

> New cells take the **active class**. If that is the same class as the tissue
> region under them, they are drawn in the same colour and you will not see
> them. Pick or create a separate class — e.g. "Nuclei" — before you start.

## 9. Patch an ROI

Draw an ROI with <kbd>O</kbd>, then in **Patches** choose a size in pixels
(128×128, 256×256) and a pyramid level, and **Lay grid over ROI**.

Patches are specified in pixels at a level, not in microns, because that is how
an encoder is defined — a ViT sees a fixed pixel tensor. The panel reports the
micron size each patch covers, since that is what decides whether the grid is
looking at cells or at architecture, and it differs between scanners for the
same pixel count.

**Only where there is tissue** clips the grid to detected tissue, so an ROI
drawn loosely round a fragment does not spend the encoder on glass.

**Make patch objects** turns the grid into real regions in a `Patch` class. Each
one behaves like an ROI: select it, drag its corners or edges to adjust it, give
it a class. That matters because a patch is a window you chose rather than a
measurement — if the grid put one half off the tissue, you can move it before
anything is computed on it.

The grid also leaves as coordinates rather than as image files:

- **Export coordinates** writes `<slide>.patches.json` — every patch's top-left
  in level-0 pixels, plus the level and size, which is exactly what
  `read_region(x, y, level, size, size)` takes. The panel includes the Python
  to read them back.
- **Export GeoJSON** writes the same squares as polygons, to open beside the
  slide and check what was patched.

Nothing is copied out of the browser, so the patches cannot drift from the
slide they came from.

## 10. Virtual spatial transcriptomics

The **Spatial** tab predicts gene expression from the H&E itself — no assay on
this section. It is built around DeepSpot-M, which reads a 224 px tile at about
20× and answers with a value per gene.

### Getting the model

The weights are gated and released as PyTorch, so there is a one-off setup:

1. Accept the terms at `huggingface.co/ratschlab/DeepSpotM`. They are limited to
   academic and public non-profit research, with no concurrent commercial role.
2. `.venv-export/bin/hf auth login` — note the path; `hf` is not on your PATH
3. `.venv-export/bin/python scripts/export_deepspot.py --genes EPCAM CD3D PTPRC COL1A1`
4. **Import model…** in the Spatial tab, and give it both the `.onnx` and the
   `.onnx.json` written beside it.

Slidecraft cannot accept that licence on your behalf, so it cannot fetch the
weights for you, and it bundles nothing.

Pick your genes at export time. Each gene is a query into the decoder, so eight
genes cost a fraction of all 19,338 — that is what makes this run in a browser
at all.

### Running it

Choose **This ROI** or **Whole slide**, then **Predict expression**.

Start with an ROI. It is quick enough to iterate on, and it is how you find out
whether the model says anything sensible about your material before spending an
hour on a slide. Progress shows the per-patch cost as it goes, and **stop** ends
the run without leaving a half-finished map on screen.

The grid is laid at the model's own patch size and magnification, not the
Patches tab's — feeding a 40× tile to a model trained at 20× shows it half the
tissue it expects, which changes the answer without failing.

### Reading it

Pick a gene from the list to colour the map. The scale is viridis, clipped to
the 2nd and 98th percentiles so one saturated patch — a fold, a pen mark —
cannot flatten everything else to the bottom of the range.

**Export CSV** writes a row per patch with its coordinates and every gene, so
the result goes back to R or scanpy.

> These values are **predicted from morphology, not measured**. They are a
> hypothesis to check against an assay, not a substitute for one. The panel says
> so, and it is worth repeating to anyone you show a map to.

## 11. Export

The Annotations panel exports GeoJSON in level-0 slide pixels, with QuPath's
`objectType` and `classification` fields, so it round-trips with QuPath in both
directions.

---

## Troubleshooting

**"Not cross-origin isolated."** The server is not sending COOP/COEP headers.
`npm run dev` does; a custom host must too.

**A slide opens black or very slowly.** Some slides have a shallow pyramid, so
one screen tile means decoding a huge region. Slidecraft adapts its worker pool
and pyramid mapping for this, but the first tile on such a slide is genuinely
slow. The status bar shows tile timings.

**Cells segment but nothing appears.** The active class is probably the tissue
class — see the note in step 8.

**Tissue detection misses a faded section.** Mark it as tissue and retrain; that
is the case hysteresis and the classifier exist for.
