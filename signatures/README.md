# Derived signature sets

Cell-type signatures for scoring predicted expression, derived from the
[CELLxGENE Discover Census](https://chanzuckerberg.github.io/cellxgene-census/)
by `scripts/signatures_from_cellxgene.py`.

Load one through **Virtual ST → Load more** in the app. Slidecraft's own
fourteen modules are built in and need no file; these are here for depth — 111
annotated cell types where the built-ins have fourteen hand-grouped ones.

## `ibd-colon.json`

111 cell types, 40 genes each, from 156,905 cells of `tissue_general == colon`
across ulcerative colitis, Crohn's disease, IBD-unspecified and normal.

```bash
python scripts/signatures_from_cellxgene.py --tissue colon \
    --disease "ulcerative colitis" --disease "Crohn disease" \
    --disease "inflammatory bowel disease" --disease normal \
    --top 40 --per-type 2000 --census-version 2025-11-08
```

Two things worth knowing before reading a score from these.

**Colonic IBD lives under `tissue_general == "colon"`, not `"large intestine"`.**
The latter holds none of it — it is 54% colon adenocarcinoma and 39% normal.
Crohn's disease also has 145,504 cells under `small intestine`, which this query
deliberately excludes.

**Genes are ranked after technical ones are removed, not before.** Mitochondrial
and ribosomal transcripts track dissociation stress and sequencing depth rather
than cell identity, and they otherwise top a fold-change ranking and push real
markers out of the top 40 — which is what put `MT-RNR2` and `RPL17` at the head
of the T-cell signature on the first pass.

`ibd-colon.genes.txt` is the union of their genes, for exporting a model that
covers exactly them:

```bash
python scripts/export_deepspot.py --fp16 --genes $(cat signatures/ibd-colon.genes.txt)
```

Census data is CC-BY. Cite the CZI CELLxGENE Discover Census in work that uses
these.
