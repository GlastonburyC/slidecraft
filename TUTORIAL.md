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

## 10. Train a classifier on your own annotations

The **Predict** tab is the loop the whole app is arranged around: embed once,
then label, train, look, disagree, retrain.

### Get an encoder

```bash
python scripts/export_onnx.py MahmoodLab/UNI2-h --preset uni2
python scripts/export_onnx.py paige-ai/Virchow2 --preset virchow2
```

Both are gated: accept their terms on Hugging Face first, and `hf auth login`.
The preset matters — UNI2-h needs a specific timm configuration, and Virchow2's
published embedding is the class token concatenated with the mean of its patch
tokens, which its plain forward pass does not return. Import the `.onnx` under
**Models**.

### The loop

1. Draw an ROI with <kbd>O</kbd>.
2. **Embed this ROI.** This is the only step that touches pixels, and it is
   cached — re-embedding the same ROI is instant, and an overlapping one only
   encodes what is new.
3. Annotate inside it in at least two classes. There is no separate labelling
   mode: whatever you draw is the training set.
4. **Train and predict.** The head fits in well under a second and colours every
   patch by class.
5. Disagree with it — draw over what it got wrong — and **Retrain**. Because the
   embeddings are cached, this is immediate. That is the point.

**Confidence** leaves unsure patches uncoloured rather than showing them with
certainty the head does not have. **Showing** switches between the most likely
class and one class at a time, where opacity carries the probability.

### Reading the score

The held-out accuracy is measured on whole **spatial blocks** the head never
saw. Neighbouring patches are near-copies of each other, so a random split
reports a much better number and means much less. If there is no score, your
labels sit in too few places on the slide to hold any back — annotate in a few
separate spots.

## 11. Virtual spatial transcriptomics

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

### All 19,338 genes: run it offline

You cannot do the whole transcriptome in a browser — a 1B-parameter encoder plus
a decoder that scales with gene count is minutes per patch on WASM. So run it
where the GPU is:

```bash
python scripts/predict_expression.py slide.svs --all --device cuda
```

That writes `slide.expression.bin` beside the slide. Drop the folder into
Slidecraft and each slide opens with its own map already attached, exactly like
the tissue GeoJSON sidecars.

### Running it on a cluster

Add `--submit` and it goes to Slurm instead: copies the script and the slide,
submits to the queue, waits, and brings the map back.

```bash
python scripts/predict_expression.py slide.svs --all \
    --submit cluster --partition gpuq \
    --remote-python ~/venvs/deepspot/bin/python \
    --module cuda/12.4
```

Authentication is your own SSH config, keys and agent — no password is asked
for or stored. If `ssh cluster` works in your terminal, this works.

**If your cluster wants a password**, authenticate once in your own terminal and
share that connection, so the password stays with you:

```bash
ssh -M -S ~/.ssh/cm-cluster -o ControlPersist=8h -N -f cluster
```

then add `--ssh-option='-S ~/.ssh/cm-cluster'` to the command above. Close it
when you are done with `ssh -S ~/.ssh/cm-cluster -O exit cluster`.

Useful flags: `--remote-slide /data/slide.svs` when the slide already lives on
cluster storage, so it is not copied; `--no-watch` to submit and walk away;
`--account`, `--time-limit`, `--mem`, `--gres`. `HF_TOKEN` is forwarded if set,
exported inside the batch script rather than placed on the command line where
`ps` on a shared login node would show it.

**Always dry-run an unfamiliar cluster first:**

```bash
python scripts/predict_expression.py slide.svs --all --submit cluster --dry-run
```

That prints the batch script and every command, and sends nothing.

A map is tied to the slide it was computed on — the patch coordinates are that
slide's level-0 pixels — so Slidecraft checks the name and refuses to draw one
over a different slide rather than silently misplacing it.

### Cell-type signatures instead of single genes

Per-gene prediction from H&E is noisy. Averaging a marker set buys that back:
the independent part of each gene's error averages down while the shared signal
does not, and "where are the T cells" is usually the real question.

```bash
.venv-export/bin/pip install cellxgene-census
python scripts/signatures_from_cellxgene.py --tissue lung --top 40
```

That writes `signatures.json` plus `signatures.genes.txt` — export the model to
cover exactly those genes, then import the signatures under **Signatures** in
the Spatial tab. Each row shows how many of its genes the loaded model actually
predicts; a signature scored on three of forty is a weak one.

Scores are standardised per gene across the patches before averaging, so they
are **relative to this slide**: they say where a cell type is concentrated here,
not how much of it there is compared with another slide.

### Which genes are enriched in an area?

Draw round something — a calcified focus, a tumour nest — with any tool, select
it, and hit **Which genes are enriched here?** in the Spatial tab. It compares
the patches inside against the rest and ranks every gene.

The ranking is by **AUC**: the probability a random inside patch exceeds a
random outside one. 0.5 is nothing, 1.0 is perfect separation. Export gives you
mean in, mean out, difference, AUC, p and q.

> The q-values are Benjamini-Hochberg, and they are **optimistic**. Neighbouring
> patches are near-copies of each other, so the effective sample size is well
> below the patch count and every test is anti-conservative. Rank by AUC; use q
> to filter obvious noise, not as evidence.

### Reading it

Pick a gene from the list to colour the map. The scale is viridis, clipped to
the 2nd and 98th percentiles so one saturated patch — a fold, a pen mark —
cannot flatten everything else to the bottom of the range.

**Export CSV** writes a row per patch with its coordinates and every gene, so
the result goes back to R or scanpy.

> These values are **predicted from morphology, not measured**. They are a
> hypothesis to check against an assay, not a substitute for one. The panel says
> so, and it is worth repeating to anyone you show a map to.

## 12. Export

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
