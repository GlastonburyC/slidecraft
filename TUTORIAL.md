# Slidecraft tutorial

A full pass through the tool: open a slide, annotate it, teach it what tissue
looks like on your material, run that over a folder — and read gene expression
off the H&E itself.

Nothing is uploaded. Slides are read in your browser by OpenSlide compiled to
WebAssembly, so a 2 GB SVS never leaves the machine.

Each section stands on its own, so jump to what you need:
[annotating](#annotate) · [tissue detection](#tissue) ·
[training a tissue classifier](#train-tissue) · [batch over a folder](#batch) ·
[cell segmentation](#cells) · [patching](#patches) ·
[the prediction loop](#predict) · [virtual spatial transcriptomics](#spatial) ·
[modules](#modules) · [region enrichment](#enrichment) ·
[find the others like it](#similar) · [gradients along an axis](#gradients) ·
[the expression floor](#expression-floor) ·
[cutting a big map down](#subset) · [sharding across GPUs](#shards) ·
[running on a cluster](#cluster) ·
[export to scanpy](#anndata)

---

## 1. Open a slide {#open}

Drag a slide anywhere onto the window, or use **Choose files**.

For **MIRAX**, drag the *folder* containing both `Foo.mrxs` and its `Foo/` data
directory. The `.mrxs` file holds no pixels; dropping it alone cannot work, and
Slidecraft will say so rather than failing quietly.

Formats: `.svs` `.ndpi` `.mrxs` `.tif`/`.tiff`/`.btf` `.scn` `.vms`/`.vmu`
DICOM.

Drop many at once — the **Slides** list is filterable once there are more than a
handful. Right-click a slide, or press <kbd>Backspace</kbd> on it, to remove it
from the list; its annotations are kept.

## 2. Get around {#navigate}

| | |
|---|---|
| Zoom | Scroll wheel |
| Pan | <kbd>H</kbd> tool, or hold the **middle mouse button** with any tool |
| Switch panel | The tabs at the top of the left panel: Slides, Annotate, Tissue, Cells, Patches |
| Show/hide annotations | <kbd>Space</kbd> |
| Select | <kbd>V</kbd> — <kbd>Shift</kbd>-click adds, <kbd>Backspace</kbd> deletes |

<kbd>Space</kbd> is worth the muscle memory: comparing a boundary against the
tissue underneath it is the most repeated action in the app.

## 3. What comes with the slide {#associated}

A slide rarely arrives alone. A batch run writes `<slide>.geojson` beside each
slide and a GPU run writes `<slide>.expression.bin`, so dropping the folder back
in brings the work with it rather than leaving it to be imported by hand, one
slide at a time.

The **Slides** panel lists what arrived — how many annotations, how many patches
by how many genes, and from which file — because annotations appearing that you
did not draw are unsettling when nothing says why.

A map arriving switches you to **Virtual ST**, since seeing it is why the file
was dropped.

**Dropping a slide on its own cannot bring its map with it.** A browser only
sees the files it was handed, however adjacent they are on disk — so drop the
*folder*, or drop the slide and then drop the `.expression.bin` onto it
afterwards, which attaches it to whatever is open. A map computed on a different
slide is refused rather than drawn in the wrong place.

The lamp beside it turns this off, for a slide whose neighbours on disk are
stale: an old segmentation you do not want coloured over the new one, or a map
from a model you have since replaced. It never overwrites either way — anything
already saved for the slide wins, and a map computed on a different slide is
refused rather than drawn in the wrong place.

## 4. Annotate {#annotate}

Pick a tool from the rail. Each has a one-key shortcut, shown on the button.

| Key | Tool | Notes |
|---|---|---|
| <kbd>R</kbd> | Rectangle | |
| <kbd>P</kbd> | Polygon | Click vertices; <kbd>Enter</kbd> or double-click closes, <kbd>Esc</kbd> cancels |
| <kbd>F</kbd> | Freehand | Drag to trace |
| <kbd>B</kbd> | Brush | Merges with the same class underneath. **Right-click the tool for size.** |
| <kbd>E</kbd> | Eraser | Subtracts from anything it touches |
| <kbd>N</kbd> | Point | Counting points |
| <kbd>O</kbd> | ROI | The working frame for prediction |
| <kbd>T</kbd> | Patch | Drag a region and it tiles — see [patching](#patches) |
| <kbd>A</kbd> | Axis | Drag an arrow to read a gradient — see [gradients](#gradients) |
| <kbd>G</kbd> | Click-to-segment | One click per cell |

**Classes are yours.** There are no defaults. Right-click an object to name a
class, or use the **Annotate** tab; <kbd>1</kbd>–<kbd>9</kbd> switch between the
first nine. Each class has a visibility toggle — hiding the tissue class is the
quickest way to see the nuclei drawn on top of it, and hiding is not deleting:
the objects stay, and stay exported.

**ROIs resize.** Select one and drag any corner or edge handle.

## 5. Detect tissue {#tissue}

In the **Tissue** panel, click **Detect tissue**.

With no trained model this uses a colour rule: it scores every cell of the slide
overview on saturation and darkness, splits at the valley of that histogram,
then grows the detection into faded tissue that connects to something confident.
Separate fragments come out as separate objects.

It is genuinely good on well-stained material and it will make mistakes on
yours — dust, bubbles, mounting medium, pen marks, a faded section. That is what
the next step is for.

## 6. Teach it {#train-tissue}

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

## 7. Use it {#use-tissue}

**Use the model when detecting tissue** is ticked automatically. **Detect
tissue** now runs your classifier. The panel tells you which model produced the
current result, and which features it leans on.

The model is saved in the browser, so it survives a reload and applies to every
future slide. Switch between saved models by clicking one.

Still wrong somewhere? Mark those regions, **Update this slide's labels**, train
again.

## 8. Run it over a folder {#batch}

Click **Run on a folder…**, choose a directory, **Run**.

Each slide gets a `<slide>.geojson` written next to it. Slides are opened and
closed one at a time; a slide that fails is recorded and the queue continues, so
it can be left alone.

Drop that folder back into Slidecraft later and every slide arrives with its
annotations already attached — the sidecar is matched to the slide by name. It
never overwrites annotations you already saved in the app.

> Needs Chrome or Edge. Safari and Firefox cannot write files back into a chosen
> folder; the dialog says so rather than failing silently.

## 9. Segment cells {#cells}

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

## 10. Patch a region {#patches}

Press <kbd>T</kbd> and drag across the tissue. The grid arrives with the drag,
at whatever size the tool's panel is set to — there is no separate step.

The settings sit beside the tool rail and are on screen only while the tool is
held. **Re-tile** applies a changed size to the region you last drew; the drag
itself is what creates a new one.

Patches are specified in pixels at a pyramid level, not in microns, because
that is how an encoder is defined — a ViT sees a fixed pixel tensor. The panel
reports the micron size each patch covers, since that is what decides whether
the grid is looking at cells or at architecture, and it differs between
scanners for the same pixel count.

**Only where there is tissue** clips the grid to detected tissue, so a region
drawn loosely round a fragment does not spend the encoder on glass. Run
[Detect tissue](#tissue) first for this to have anything to clip to.

The region itself is still an ROI. It is the frame the grid sits in, and
everything downstream asks which ROI you are working in, so making it anything
else would put that question out of reach.

**Make N patch objects** turns the grid into real regions in a `Patch` class. Each
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

## 11. Train a classifier on your own annotations {#predict}

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

1. Choose what to cover. Draw an ROI with <kbd>O</kbd>, or — if you have run
   **Detect tissue** — switch to **All tissue** and patch every detected
   fragment without drawing anything. Each fragment stays its own region, so
   the head is validated on fragments it was not trained on.
2. **Embed this ROI.** This is the only step that touches pixels, and it is
   cached — re-embedding the same ROI is instant, and an overlapping one only
   encodes what is new.
3. Annotate inside it in at least two classes. There is no separate labelling
   mode: whatever you draw is the training set.
4. **Train and predict.** The head fits in well under a second and colours every
   patch by class.
5. Disagree with it — draw over what it got wrong — and **Retrain**. Because the
   embeddings are cached, this is immediate. That is the point.

**Clear the prediction** takes the overlay off the slide. It is a result rather
than an annotation, so it is not removed by deleting objects — though deleting
every ROI and tissue region does clear it, since it no longer covers anything.

**Confidence** leaves unsure patches uncoloured rather than showing them with
certainty the head does not have. **Showing** switches between the most likely
class and one class at a time, where opacity carries the probability.

### Finer than a patch

The encoder's input is fixed at 224 px, but the patch can slide by less than
its own width. Set **Stride** to a half or a quarter and each point of tissue
is seen by four or sixteen patches, each from a different offset; averaging
them gives a map at the stride's resolution rather than the patch's.

The average is weighted by a raised cosine centred on each patch — a patch
describes its middle better than its corners, where the field is as much about
the neighbouring tissue. That sharpens boundaries and removes the blocky seams
uniform averaging leaves at patch edges.

Eighth and sixteenth strides are there too, at 64× and 256×. The panel
estimates the patch count and, once you have run once, the time on your own
machine — a multiplier is abstract, twenty minutes is not.

Past a quarter the gain is mostly smoothing rather than detail. The encoder
still judges 224 px at a time whatever the stride, so patches offset by 14 px
see almost the same field and increasingly agree; what improves is the
smoothness of the boundary, not the information behind it. Worth it on a small
ROI where a boundary matters, rarely worth it across a whole slide.

### Reading the score

The held-out accuracy is measured on whole **spatial blocks** the head never
saw. Neighbouring patches are near-copies of each other, so a random split
reports a much better number and means much less. If there is no score, your
labels sit in too few places on the slide to hold any back — annotate in a few
separate spots.

## 12. Virtual spatial transcriptomics {#spatial}

The **Virtual ST** tab predicts gene expression from the H&E itself — no assay
on this section. It is built around DeepSpot-M, which reads a 224 px tile at
about 20× and answers with a value per gene.

There are two ways to get a map, and for a whole slide only the first is
practical.

### 1. Compute it on a GPU, drop the folder in {#gpu-map}

DeepSpot-M is a 1B-parameter encoder: minutes per patch in a browser, seconds
on a GPU. So run it where the GPU is.

```bash
python scripts/predict_expression.py slide.svs --panel ibd-colon
```

It writes `slide.expression.bin` beside the slide. Drop that folder into
Slidecraft and the map opens with the slide — see
[associated data](#associated).

Three arguments are worth knowing:

- `--panel ibd-colon` is 140 curated markers of colonic inflammatory bowel
  disease, grouped by readout. `--genes EPCAM CD3D …` takes your own list, and
  `--all` does the whole 19,338-gene transcriptome.
- `--source` picks which of DeepSpot-M's five frozen gene-embedding pathways
  conditions the gene router — `evo2`, `orthrus`, `prott5`, `scgpt` or
  `apertus`. The default is `scgpt`, which is what the model card's own example
  uses. They are not interchangeable, so the choice is written into the file's
  header and into its `modelId`.
- `--stride` overlaps the patches, and the overlay blends them. Each output
  cell is the raised-cosine weighted average of every patch covering it — a
  patch describes its middle better than its corners, so weighting by distance
  from the centre sharpens boundaries instead of smearing them. Without that
  the overlapping squares would simply overdraw and the finer run would cost
  many times the compute for nothing.

The tissue detection is the same algorithm the browser uses, ported line for
line in `scripts/tissue.py`, so a patch chosen on a GPU node is a patch
Slidecraft would have chosen.

### 2. Import an ONNX export and run it here {#onnx-map}

For one ROI and a handful of genes, when there is no GPU to reach. The weights
are gated and released as PyTorch, so there is a one-off setup:

1. Accept the terms at `huggingface.co/ratschlab/DeepSpotM`. They are limited to
   academic and public non-profit research, with no concurrent commercial role.
2. `.venv-export/bin/hf auth login` — note the path; `hf` is not on your PATH
3. `.venv-export/bin/python scripts/export_deepspot.py --genes EPCAM CD3D PTPRC COL1A1`
4. **Import model…** in the Virtual ST tab, and give it both the `.onnx` and the
   `.onnx.json` written beside it.

Slidecraft cannot accept that licence on your behalf, so it cannot fetch the
weights for you, and it bundles nothing.

Pick your genes at export time. Each gene is a query into the decoder, so eight
genes cost a fraction of all 19,338 — that is what makes this run in a browser
at all.

### Running an imported model here {#run-onnx}

With a model imported, choose **This ROI** or **Whole slide**, then **Predict
expression**.

Start with an ROI. It is quick enough to iterate on, and it is how you find out
whether the model says anything sensible about your material before spending an
hour on a slide. Progress shows the per-patch cost as it goes, and **stop** ends
the run without leaving a half-finished map on screen.

The grid is laid at the model's own patch size and magnification, not the
patch tool's — feeding a 40× tile to a model trained at 20× shows it half the
tissue it expects, which changes the answer without failing.

### What a whole transcriptome costs {#cost}

All 19,338 genes over a 400 mm² resection — about 32,000 patches — is roughly
40 minutes on one V100 and 1.2 GB on disk. Gene count barely changes that: the
vision backbone dominates and the gene decoder rides along, so `--all` costs
little more than a 140-gene panel. Patch count is the axis that matters, which
is why `--stride` gets expensive fast: halving it quadruples the patches, and an
eighth of the patch width is sixty-four times as many.

Slidecraft loads a map that size in well under a second and switches gene in
about 70 ms. The values stay in the half precision they arrived in and are
decoded one gene at a time, because widening 620 million of them to fp32 up
front is several gigabytes to show one gene.

About 1.2 GB is where a tab gives out, and a quarter-stride whole-transcriptome
run goes well past it — a 400 mm² resection at `--stride 56` is 306,587 patches
by 19,338 genes, or 11.9 GB. That map is worth having; it just has to be cut
down before it can be opened.

### Cutting a big map down to something that opens {#subset}

```bash
python scripts/subset_expression.py "slide.all.bin" \
    --out "slide.expression.bin" --min-expression 0.05 --top 3000
```

The output is named `<slide>.expression.bin` because that is the name Slidecraft
looks for beside a slide — the map you browse is the one that carries the
slide's name. Give the full-transcriptome original something else (`--all` runs
here write `slide.all.bin`) and keep it wherever analysis happens; it is not a
file a browser can open, so it should not be the one sitting next to the slide
claiming to be.

Two separate kinds of waste come out of that, and they are worth separating
because only one of them is lossy.

**Most of the genes are not transcribed here.** The decoder answers for every
gene in the reference whether or not the tissue uses it, so a colon map carries
olfactory receptors and testis antigens at the same cost per value as COL1A1.
On the run above, 8,432 of 19,338 genes sit below a slide-wide mean of 0.05 —
and they are not merely uninteresting, they are noise being ranked against
signal in every differential test. This is the same floor the enrichment and
gradient panels apply, moved upstream to where it also saves the bytes.

**fp16 is finer than the model is.** DeepSpot-M predicts log1p expression from
a 224 px tile; its disagreement with held-out truth is a large fraction of the
value. Eleven bits of mantissa records the decoder's arithmetic, not the
biology. `subset_expression.py` stores one byte per value with a scale and zero
*per gene*, so 255 steps span the range that gene actually occupies rather than
a range set by whichever gene in the map was loudest. A gene that never goes
negative decodes code 0 to exactly 0, so an absent gene stays absent instead of
picking up a faint floor everywhere.

`--top` then keeps the most spatially variable of the survivors, which is the
selection that matters for browsing: a gene sitting flat at 2.0 across the whole
slide is highly expressed and draws an empty map. `--rank mean` asks for the
loudest instead, and `--genes` takes a list.

Measured against the fp16 map it came from, over 40,000 patches and 3,000 genes:

| | |
|---|---|
| Size | 11.9 GB → 0.93 GB |
| Worst decode error | 0.0126, against a value range of 6.4 — half a step |
| Values within half a step | 100.0000% |
| Differential AUC, mean shift | 0.000041 |
| Differential AUC, worst shift | 0.0009 |
| Rank agreement on AUC | ρ = 0.9988 |
| Top 25 genes preserved | 24 of 25 |

That last row is the honest limit. Genes whose AUCs differ by less than 0.0009
can swap places, so where a ranking is that close, read the AUCs rather than the
row numbers — which is true of the fp16 map too, just less visibly.

Quantised maps are also *faster* to work with than the fp16 ones they came from,
because a byte is what makes the memory layout tractable. The Mann-Whitney
counting sort bins by the stored value, so 256 bins replace 65,536 — small
enough that 64 genes can be tested together, and at one byte per value 64 genes
is exactly the cache line being fetched anyway. Each gene's group means then
fall out of the same sweep instead of needing their own pass over the column.
Measured on the 0.93 GB map above:

| | |
|---|---|
| Opening the file | 72 ms |
| Switching gene, 306,587 patches | 29 ms |
| Region enrichment, all 3,000 genes | 4.1 s |

The last of those was 28 s before the genes were blocked, which is the whole
reason the number is worth quoting: the saving is not the disk space.

Precision is a choice here, not an assumption: `--dtype float16` subsets the
genes and keeps the original precision, and either way the tool prints the error
it introduced. Keep the full map for anything leaving for scanpy —
`expression_to_anndata.py` reads both forms and decodes the quantised one on the
way out.

### Splitting one slide across several GPUs {#shards}

A strided run is the case that needs this: quarter-stride is sixteen times the
patches, and patch count is the only axis that costs. `--shard I/N` does the Ith
of N contiguous blocks, so a Slurm array can put each on its own GPU:

```bash
python scripts/predict_expression.py slide.svs --panel ibd-colon --stride 56 \
    --shard $SLURM_ARRAY_TASK_ID/4 --out "slide.shard${SLURM_ARRAY_TASK_ID}of4.bin"
python scripts/merge_shards.py slide.shard*of4.bin --out slide.expression.bin
```

Contiguous blocks rather than every Nth patch: every patch costs the same to
embed, so there is nothing to balance, and a block keeps each worker reading one
region of the slide instead of all four seeking across the whole of it.

Make the merge depend on the array — `sbatch --dependency=afterok:<jobid>` — so
it cannot run on a partial set. It refuses one anyway: every shard has to be
present exactly once, and they must agree on the genes, the patch size, the
stride and the slide, because a merge that quietly accepted mismatched pieces
would produce a file that opens, draws, and is wrong.

### Running it on a cluster {#cluster}

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

### Modules, not single genes {#modules}

Per-gene prediction from H&E is noisy. Averaging a marker set buys that back:
the independent part of each gene's error averages down while the shared signal
does not, and "where are the T cells" is usually the real question. The
difference is stark — a single gene reads as static where the same genes as a
module show mucosa, submucosa and muscle wall.

**Fourteen modules ship built in**, so a map is readable the moment it loads:
epithelium, mature colonocyte, goblet, crypt, antimicrobial, the calprotectin
axis, T cells, plasma cells, myeloid, cytokines that are also drug targets,
chemokines, stroma, vessels, neuroendocrine. Switch to **Signatures** to pick
one. The best-covered is selected automatically when a map arrives.

Each row shows how many of its genes the loaded map actually carries. A module
the map cannot cover with at least three is hidden rather than scored on a
couple and presented as if it meant something — so the same list serves an
eight-gene map and a whole transcriptome.

For more depth, `signatures/ibd-colon.json` carries **111 annotated cell types**
derived from 156,905 colonic cells across ulcerative colitis, Crohn's disease
and normal tissue. Load it with **Load more**. Derive your own for any tissue
and disease the CELLxGENE Census covers:

```bash
.venv-export/bin/pip install cellxgene-census
python scripts/signatures_from_cellxgene.py --tissue colon \
    --disease "ulcerative colitis" --disease "Crohn disease" --disease normal
```

Two things that query taught us, both worth knowing before you run your own.
Colonic IBD sits under `tissue_general == "colon"`; `"large intestine"` holds
none of it. And genes are ranked *after* mitochondrial and ribosomal
transcripts are removed, not before — those track dissociation stress and
sequencing depth rather than cell identity, and left in they top the ranking and
push real markers out.

Scores are standardised per gene across the patches before averaging, so they
are **relative to this slide**: they say where a cell type is concentrated here,
not how much of it there is compared with another slide. That also means a
module's difference is in standard deviations, where a single gene's is in the
model's own units.

### What is in this region? {#enrichment}

Draw round something — a calcified focus, a tumour nest, a lymphoid aggregate —
with any tool, select it, and ask one of two questions in the Virtual ST tab.

**Which cell types?** ranks the modules by how well they separate your region
from the rest of the slide. This is usually the question you actually have:
"CXCL13 is enriched here" is only useful to someone who already knows what
CXCL13 means, where "this is a lymphoid aggregate" is the finding itself. With
`signatures/ibd-colon.json` loaded you get all 111 cell types ranked instead of
the fourteen built-in modules.

**Which genes?** does the same against every gene in the map.

A box drawn over colonic mucosa returns Epithelium 0.87, Goblet 0.85,
Colonocyte 0.84 and Crypt 0.83, with Neutrophil depleted at 0.31 — which is
mucosa described as mucosa.

The ranking is by **AUC**: the probability a random inside patch exceeds a
random outside one. 0.5 is nothing, 1.0 is perfect separation. Export gives you
mean in, mean out, difference, AUC, p and q.

> The q-values are Benjamini-Hochberg, and they are **optimistic**. Neighbouring
> patches are near-copies of each other, so the effective sample size is well
> below the patch count and every test is anti-conservative. Rank by AUC; use q
> to filter obvious noise, not as evidence.

### Find the others like it {#similar}

Draw round one example, ask **Which genes?**, then **Save as a module**. The
ranked list is already a description of the thing you drew, so keeping it as a
module and scoring it over every patch says where else on the slide that
description fits — with nothing trained and nothing labelled.

Both directions are kept. A gene the region is *depleted* of describes it as
well as one it is full of — "no collagen here" is half of what makes a lymphoid
aggregate look like one — and the negative weight is what lets the score use it
that way.

Then **Find the top N% as objects** traces where the field clears a threshold
and commits those regions as real annotations: select them, drag their corners,
classify them, delete the ones that are wrong. A percentile rather than a fixed
value, because a module's score is standardised per slide — 0.8 means something
different on each one, where "the top 5% of this slide" asks the same question
everywhere.

Those accepted and rejected regions are exactly what the [prediction
loop](#predict) trains on, so the intended path is: find candidates this way,
judge them, then train a head over the patch embeddings and correct *that*.

> A caveat worth holding on to. Predicted expression is a function of the same
> pixels the morphology encoder sees, so a match here is not independent
> evidence — it means two regions look alike to a model trained on
> transcriptomics. That is a useful and interpretable way of looking alike, and
> it is still looking alike. Treat what comes back as candidates to judge, not
> as a result.

Only a **gene** enrichment can become a module; a module is a set of genes, and
a ranking of cell types is a ranking of other modules. The button says so.

### What changes along an axis? {#gradients}

Enrichment asks whether a region differs from the rest, which suits a thing with
a boundary. Much of mucosa has none — expression varies *along* an axis, crypt
base to luminal surface, mucosa to muscularis — and splitting that into inside
and outside throws away the ordering, which was the signal.

So press <kbd>A</kbd> and drag an arrow across the tissue. Each patch is
projected onto it and rank-correlated with how far along it sits, and the answer
is **signed by the direction you drew**: positive rises toward the arrowhead.
Reverse the arrow and every sign flips. That is why the axis is drawn with a
head — a plain line cannot tell you which claim you are reading.

**Corridor** sets how wide a band around the arrow counts, in patches. Without
it the projection would accept the whole slide and compute a gradient across
tissue the arrow never pointed at. Patches outside the band, or past either end,
are not counted.

Drawn from mucosa to wall on a colonic resection — 6 mm, 266 patches — this
returns Goblet −0.86, Colonocyte −0.84, Crypt −0.84 and Epithelium −0.83 all
falling, with Myeloid +0.77 and Vascular +0.73 rising. The genes behave the same
way: LCN2, FCGBP, SLC26A3, DMBT1 and KRT20 at the top, all epithelial, all
falling as you leave the mucosa.

Ranking is by the size of the correlation regardless of direction, so a gene
that falls ranks alongside one that rises — direction is in the sign, not the
position. Ranking a whole transcriptome takes a few seconds and says so while it
works.

**Plot it as a module** turns the ranking into a field. Each gene is weighted by
its own rho, so the score is high where the risers are high and the fallers are
low — which is the gradient itself, drawn. It covers the whole slide rather than
just the corridor, so you can see whether the same trend holds away from the
arrow you drew.

### Genes the model does not really express {#expression-floor}

Both gene analyses rank by a scale-free statistic. That is the point of a rank
test and also its trap: a gene the model predicts at 0.001 everywhere can
separate a region perfectly, or track an axis almost exactly, on the ordering of
noise alone — and it will outrank COL1A1.

So **Min. expression** drops genes whose mean across the slide is below a floor,
before anything is corrected. On a whole transcriptome roughly half the genes
sit below the 0.05 default, while the markers that matter are well clear of it:
COL1A1 2.3, EPCAM 0.69, CD3D 0.096. The ones known to be unreliable here are not
— AQP8 0.007, PYY 0.004, which is the [oncology training set](#spatial) showing
through.

Filtering happens before Benjamini-Hochberg, so the correction covers the genes
actually reported rather than thousands that were never going to be.

> The q-values here are weaker even than the region test's. Patches along one
> axis are immediate neighbours, so they are about as far from independent as
> patches get. Rank by rho; use q to filter noise, not as evidence.

### Reading it

Pick a gene from the list to colour the map. The scale is viridis, clipped to
the 2nd and 98th percentiles so one saturated patch — a fold, a pen mark —
cannot flatten everything else to the bottom of the range.

**Export CSV** writes a row per patch with its coordinates and every gene, so
the result goes back to R or scanpy.

> These values are **predicted from morphology, not measured**. They are a
> hypothesis to check against an assay, not a substitute for one. The panel says
> so, and it is worth repeating to anyone you show a map to.

## 13. Export {#export}

The Annotations panel exports GeoJSON in level-0 slide pixels, with QuPath's
`objectType` and `classification` fields, so it round-trips with QuPath in both
directions.

### Taking an expression map to scanpy {#anndata}

Slidecraft answers spatial questions on the slide. It does not do trajectory
inference, or differential testing across patients, or most of what the Python
stack already does well — so a map has to be able to leave.

**Export AnnData** in the Virtual ST tab writes `<slide>.anndata.zarr.zip`:

```python
import zipfile, anndata
zipfile.ZipFile("slide.anndata.zarr.zip").extractall("slide.zarr")
adata = anndata.read_zarr("slide.zarr")
```

What you get:

- `X` — patches by genes, float32, dense. Predicted expression has no zeros to
  speak of, so a sparse layout would store the same numbers plus two index
  arrays.
- `obs` — `x` and `y` of each patch in level-0 pixels, plus `in_tissue` when
  tissue has been detected. Patches are named by position, so two slides'
  tables concatenate without collision.
- `obsm["spatial"]` — patch **centres**, which is what `scanpy` and `squidpy`
  plotting expect.
- `uns` — the slide, the model, the pathway, and a line saying these are
  predictions rather than measurements. Provenance travels with the numbers or
  it is lost.

It is zarr inside a zip rather than `.h5ad` because writing HDF5 from scratch in
a browser is a great deal of code to get subtly wrong, where zarr is JSON beside
raw little-endian blocks.

**For a whole transcriptome, use the script instead.** 32,000 patches by 19,338
genes is 2.5 GB once widened to the float32 AnnData stores, which is more than a
tab should assemble — the button says so and disables itself. This streams it a
block of patches at a time:

```bash
python scripts/expression_to_anndata.py slide.expression.bin
python scripts/expression_to_anndata.py slide.expression.bin --genes EPCAM MUC2 COL1A1
python scripts/expression_to_anndata.py slide.expression.bin --spatialdata
```

`--genes` is usually what makes a whole transcriptome workable downstream.
`--spatialdata` additionally writes a SpatialData zarr with the patches as
points and the map as a table annotating them (`pip install spatialdata`).

It reads quantised maps too and decodes them on the way out, so a subset cut for
the browser can still leave for scanpy. For anything where the numbers
themselves are the result, export from the full fp16 map rather than from a
subset of it — the quantisation is small and measured, but there is no reason to
carry it into an analysis that has the original to hand.

---

## Troubleshooting {#troubleshooting}

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
