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


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tissue", required=True, help="tissue_general, e.g. lung, breast, colon")
    ap.add_argument("--organism", default="Homo sapiens")
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

    print(f"Opening the Census ({args.census_version}) …")
    with cellxgene_census.open_soma(census_version=args.census_version) as census:
        # Primary data only: the Census contains the same cells re-published
        # across datasets, and counting them twice biases every mean.
        obs_filter = (
            f"tissue_general == '{args.tissue}' and is_primary_data == True"
        )
        print(f"Reading {args.tissue} …")
        adata = cellxgene_census.get_anndata(
            census,
            organism=args.organism,
            obs_value_filter=obs_filter,
            column_names={"obs": ["cell_type", "assay"]},
        )

    if adata.n_obs == 0:
        print(f"No cells for tissue_general == '{args.tissue}'.", file=sys.stderr)
        return 1
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

    rng = np.random.default_rng(0)
    means: dict[str, np.ndarray] = {}
    counts_per_type: dict[str, int] = {}
    for ct in sorted(set(cell_types)):
        idx = np.flatnonzero(cell_types == ct)
        if len(idx) < MIN_CELLS:
            continue
        if len(idx) > args.per_type:
            idx = rng.choice(idx, args.per_type, replace=False)
        block = logged[idx]
        means[ct] = np.asarray(block.mean(axis=0)).ravel()
        counts_per_type[ct] = len(idx)

    if len(means) < 2:
        print("Fewer than two cell types passed the minimum. Try a broader tissue.", file=sys.stderr)
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

        order = np.argsort(lfc)[::-1][: args.top]
        picked = [(genes[j], float(lfc[j])) for j in order if np.isfinite(lfc[j]) and lfc[j] > 0]
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
                "source": f"CELLxGENE Census {args.census_version}, tissue_general={args.tissue}",
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
