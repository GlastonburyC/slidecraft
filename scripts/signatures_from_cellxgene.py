"""
Derive cell-type signatures from the CELLxGENE Census, for scoring predicted
expression instead of reading single genes.

Why bother. Predicting one gene from H&E is noisy — per-gene correlations are
modest for most of the transcriptome, and a map of one gene inherits all of that
noise. A cell-type signature is a weighted average over tens of genes, so the
independent part of the error averages down while the shared biological signal
does not. It also answers the question people usually have, which is "where are
the T cells" rather than "what is CD3D doing".

    python scripts/signatures_from_cellxgene.py --tissue lung --top 40

Writes signatures.json for Slidecraft, and prints the union of their genes so
the model can be exported to cover exactly them:

    python scripts/export_deepspot.py --fp16 --genes $(cat signatures.genes.txt)

Requires the census client, which is a separate install:

    .venv-export/bin/pip install cellxgene-census

Census data is CC-BY; cite the CZI CELLxGENE Discover Census in anything that
uses these signatures.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Cells sampled per cell type. Means converge long before the full census, and
# pulling every cell of a common type in a big tissue is tens of GB.
PER_TYPE = 2000
# Cell types with fewer cells than this are dropped: a mean over a handful of
# cells is not a signature, it is a few cells with a label.
MIN_CELLS = 200

import re

# Genes that top a fold-change ranking without saying anything about cell type.
#
# Mitochondrial and ribosomal transcripts track dissociation stress and
# sequencing depth, so they separate protocols and tissue handling rather than
# biology. MALAT1 and NEAT1 are nuclear lncRNAs so abundant that single-nucleus
# data is largely made of them. Unnamed accessions cannot be matched to a
# prediction vocabulary at all.
#
# They are dropped before the ranking, not after, because leaving them in
# pushes real markers out of the top N -- which is what put MT-RNR2 and RPL17
# at the head of the T cell signature and left it saying nothing.
# Mitochondrial symbols all carry the hyphen (MT-ND1, MT-CO1). Matching a bare
# MT followed by a digit would take the metallothioneins with it -- MT1G, MT2A
# -- and those are real genes, induced in inflamed mucosa.
JUNK = re.compile(
    r"^(MT-|RP[SL][0-9]|MRP[SL][0-9]|ENSG[0-9]|LINC[0-9]|"
    r"MALAT1$|NEAT1$|XIST$|EEF1[AG][0-9]?$|TMSB4X$|B2M$|ACTB$|GAPDH$)"
)


def is_informative(symbol: str) -> bool:
    return not JUNK.match(symbol)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tissue", required=True, help="tissue_general, e.g. lung, breast, colon")
    ap.add_argument("--organism", default="Homo sapiens")
    ap.add_argument("--disease", action="append", default=[], metavar="NAME",
                    help="Restrict to these disease labels; repeatable. Omit for "
                         "every state. Exact Census strings, e.g. 'ulcerative "
                         "colitis', 'Crohn disease', 'inflammatory bowel disease', "
                         "'normal'. Note colonic IBD lives under "
                         "tissue_general 'colon', not 'large intestine'.")
    ap.add_argument("--top", type=int, default=40, help="Genes per signature")
    ap.add_argument("--per-type", type=int, default=PER_TYPE)
    ap.add_argument("--census-version", default="stable")
    ap.add_argument("--out", default="signatures.json")
    args = ap.parse_args()

    try:
        import cellxgene_census
        import numpy as np
    except ImportError as err:
        print(
            f"Missing dependency: {err}\n"
            "  .venv-export/bin/pip install cellxgene-census",
            file=sys.stderr,
        )
        return 1

    # Primary data only: the Census republishes the same cells across datasets,
    # and counting them twice biases every mean.
    obs_filter = f"tissue_general == '{args.tissue}' and is_primary_data == True"
    if args.disease:
        clause = " or ".join(f'disease == "{d}"' for d in args.disease)
        obs_filter += f" and ({clause})"

    organism_key = args.organism.lower().replace(" ", "_")
    rng = np.random.default_rng(0)

    print(f"Opening the Census ({args.census_version}) …")
    with cellxgene_census.open_soma(census_version=args.census_version) as census:
        print(f"Reading cell metadata for {args.tissue} …")
        obs = (
            census["census_data"][organism_key]
            .obs.read(value_filter=obs_filter,
                      column_names=["soma_joinid", "cell_type", "disease"])
            .concat()
            .to_pandas()
        )
        if len(obs) == 0:
            print(f"No cells match: {obs_filter}", file=sys.stderr)
            return 1
        print(f"{len(obs):,} cells, {obs['cell_type'].nunique()} cell types")
        for d, n in obs["disease"].value_counts().items():
            if n:
                print(f"  {n:>8,}  {d}")

        # Choose the cells FIRST, then fetch only those.
        #
        # Reading every cell of a tissue and sampling afterwards means holding
        # hundreds of thousands of cells by sixty thousand genes in memory to
        # keep a couple of thousand per type. Colon alone is half a million
        # cells. Sampling join ids is metadata-cheap, and the expression matrix
        # that comes back is the size of what is actually used.
        keep: list[np.ndarray] = []
        counts_per_type: dict[str, int] = {}
        for ct, group in obs.groupby("cell_type", observed=True):
            if len(group) < MIN_CELLS:
                continue
            ids = group["soma_joinid"].to_numpy()
            if len(ids) > args.per_type:
                ids = rng.choice(ids, args.per_type, replace=False)
            keep.append(ids)
            counts_per_type[str(ct)] = len(ids)

        if len(counts_per_type) < 2:
            print(f"Fewer than two cell types cleared {MIN_CELLS} cells. "
                  "Widen --disease, or lower the floor.", file=sys.stderr)
            return 1

        coords = np.sort(np.concatenate(keep))
        print(f"Fetching expression for {len(coords):,} sampled cells "
              f"across {len(counts_per_type)} types …")
        adata = cellxgene_census.get_anndata(
            census,
            organism=args.organism,
            obs_coords=coords,
            column_names={"obs": ["cell_type"]},
        )

    print(f"{adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # Counts to log-CPM, so cells sequenced to different depths are comparable.
    counts = adata.X
    totals = np.asarray(counts.sum(axis=1)).ravel()
    totals[totals == 0] = 1
    import scipy.sparse as sp

    scaled = sp.diags(1e4 / totals) @ counts if sp.issparse(counts) else counts * (1e4 / totals[:, None])
    logged = scaled.log1p() if sp.issparse(scaled) else np.log1p(scaled)

    cell_types = adata.obs["cell_type"].astype(str).to_numpy()
    genes = adata.var["feature_name"].astype(str).to_numpy() if "feature_name" in adata.var else adata.var_names.to_numpy()

    # The sampling already happened, against the join ids. These are the cells
    # that were asked for, so every one of them counts toward its type's mean.
    means: dict[str, np.ndarray] = {}
    for ct in sorted(set(cell_types)):
        idx = np.flatnonzero(cell_types == ct)
        if len(idx) == 0:
            continue
        means[ct] = np.asarray(logged[idx].mean(axis=0)).ravel()

    if len(means) < 2:
        print("Fewer than two cell types came back. Try a broader query.", file=sys.stderr)
        return 1
    print(f"{len(means)} cell types")

    stacked = np.vstack([means[ct] for ct in means])
    signatures = []
    for i, ct in enumerate(means):
        # Against the mean of the other types, not the grand mean: otherwise an
        # abundant type is compared largely against itself and looks unremarkable.
        others = np.delete(stacked, i, axis=0).mean(axis=0)
        lfc = means[ct] - others
        # A gene has to actually be expressed here, or a near-zero difference on
        # a near-zero gene reads as a large fold change.
        expressed = means[ct] > 0.1
        lfc = np.where(expressed, lfc, -np.inf)

        # Rank over informative genes only. Taking the top N first and filtering
        # afterwards would leave a signature short by however many technical
        # genes happened to win.
        order = np.argsort(lfc)[::-1]
        picked: list[tuple[str, float]] = []
        for j in order:
            if len(picked) >= args.top:
                break
            if not np.isfinite(lfc[j]) or lfc[j] <= 0:
                break
            if not is_informative(str(genes[j])):
                continue
            picked.append((genes[j], float(lfc[j])))
        if len(picked) < 5:
            continue
        signatures.append(
            {
                "name": ct,
                "cells": counts_per_type[ct],
                # Weighted by how specific each gene is, so a marker counts for
                # more than a gene that is merely somewhat enriched.
                "genes": [{"gene": g, "weight": round(w, 4)} for g, w in picked],
            }
        )

    out = pathlib.Path(args.out)
    out.write_text(
        json.dumps(
            {
                "source": (
                    f"CELLxGENE Census {args.census_version}, "
                    f"tissue_general={args.tissue}"
                    + (f", disease: {', '.join(args.disease)}" if args.disease else "")
                ),
                "organism": args.organism,
                "signatures": signatures,
            },
            indent=2,
        )
        + "\n"
    )

    union = sorted({g["gene"] for s in signatures for g in s["genes"]})
    gene_file = out.with_suffix("").with_suffix(".genes.txt")
    gene_file.write_text(" ".join(union) + "\n")

    print(f"Wrote {out}: {len(signatures)} signatures over {len(union)} distinct genes")
    print(f"Wrote {gene_file} — export the model to cover exactly these:")
    print(f"  python scripts/export_deepspot.py --fp16 --genes $(cat {gene_file})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
