"""
Export DeepSpot-M to ONNX so it can predict spatial expression in the browser.

DeepSpot-M (Rätsch lab) predicts spatial gene expression from an H&E tile:
a frozen pathology encoder (Midnight) with LoRA adapters, feeding a
cross-attention decoder in which each gene queries the patch tokens. That last
part is what makes this usable in a browser at all — a gene is a query, so
asking for eight genes costs a fraction of asking for all 19,338, and the export
below bakes a chosen gene subset into the graph.

    python scripts/export_deepspot.py --genes EPCAM CD3D PTPRC COL1A1
    python scripts/export_deepspot.py --all          # every gene; a large file

Weights are NOT redistributed with Slidecraft and are not bundled by this
script's output. You fetch them yourself, under their own terms:

    model  https://huggingface.co/ratschlab/DeepSpotM   CC-BY-NC-SA-4.0
    code   https://github.com/ratschlab/DeepSpotM       PolyForm Noncommercial 1.0.0

Two things follow from that, and they are why this step is a script rather than
a download button in the app.

**The repository is gated.** You have to accept its conditions on HuggingFace
first — academic or public non-profit affiliation, no concurrent commercial role
and no commercially-funded research — and then authenticate:

    .venv-export/bin/hf auth login      # or: export HF_TOKEN=...

Slidecraft cannot accept those terms for you, so it cannot fetch the weights on
your behalf.

**They are PyTorch safetensors, not ONNX.** The browser runs ONNX Runtime, so a
conversion has to happen somewhere, and it needs PyTorch. That is this script,
run once. The .onnx it writes carries the same licence as the weights it came
from: non-commercial. If your work is commercial, this model is not available to
you and Slidecraft cannot change that.

Paper: DeepSpot-M, medRxiv 2026.06.19.26356060
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

DEFAULT_GENES = ["EPCAM", "PTPRC", "CD3D", "COL1A1", "MKI67", "VIM", "KRT19", "ACTA2"]

# The gene router draws its projections from one of five frozen biological
# embedding spaces -- DNA, RNA, protein, single-cell and text.
SOURCES = ["evo2", "orthrus", "prott5", "scgpt", "apertus"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="ratschlab/DeepSpotM", help="HuggingFace repo id")
    ap.add_argument("--source", default="scgpt", choices=SOURCES,
                    help="Which frozen gene-embedding pathway conditions the gene router. "
                         "The checkpoint carries all five and will not guess between them")
    ap.add_argument("--genes", nargs="*", default=DEFAULT_GENES,
                    help="Gene symbols to bake into the export")
    ap.add_argument("--all", action="store_true",
                    help="Export every gene the model knows (much larger and slower)")
    ap.add_argument("--out", default="deepspot-m.onnx")
    ap.add_argument("--token", default=None,
                    help="HuggingFace token; defaults to HF_TOKEN or a hf auth login")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fp16", action="store_true",
                    help="Halve the file: the encoder is a 1B-parameter ViT-g, so fp32 lands "
                         "around 4.4 GB — past ONNX's 2 GB single-file limit and past what a "
                         "32-bit wasm heap can hold at all")
    ap.add_argument("--backend", choices=["wasm", "webgpu"], default="webgpu",
                    help="Execution provider the browser should prefer. WebGPU is the only one "
                         "that makes a ViT-g tractable per patch; wasm is the safe fallback")
    args = ap.parse_args()

    try:
        import torch
        from deepspotm import DeepSpotM  # from the ratschlab/DeepSpotM repo
    except ImportError as err:
        print(
            f"Missing dependency: {err}\n"
            "Install the model's own package first:\n"
            "  pip install torch onnx\n"
            "  pip install git+https://github.com/ratschlab/DeepSpotM",
            file=sys.stderr,
        )
        return 1

    import os

    token = args.token or os.environ.get("HF_TOKEN")
    print(f"Loading {args.repo} (source {args.source}) …")
    try:
        kwargs = {"token": token} if token else {}
        model, image_processor = DeepSpotM.from_pretrained(
            args.repo, source=args.source, **kwargs
        )
    except Exception as err:  # noqa: BLE001 - the cause matters more than the type
        print(
            f"Could not load {args.repo}: {err}\n\n"
            "This repository is gated. Accept its conditions at\n"
            f"  https://huggingface.co/{args.repo}\n"
            "then run `.venv-export/bin/hf auth login`, or set HF_TOKEN.",
            file=sys.stderr,
        )
        return 1
    model.eval()

    gene_names = list(model.gene_names)
    genes = gene_names if args.all else list(args.genes)

    missing = [g for g in genes if g not in gene_names]
    if missing:
        print(f"Not in this model: {', '.join(missing)}", file=sys.stderr)
        return 1
    index = [gene_names.index(g) for g in genes]
    print(f"Exporting {len(genes)} gene{'s' if len(genes) != 1 else ''}.")

    class Subset(torch.nn.Module):
        """
        Wraps the model so the graph returns only the genes asked for.

        Slicing after a full forward pass would keep the whole decoder in the
        graph; selecting the gene queries is what actually makes the export
        small enough to run per patch in a browser.
        """

        def __init__(self, inner, idx):
            super().__init__()
            self.inner = inner
            self.register_buffer("idx", torch.tensor(idx, dtype=torch.long))

        def forward(self, pixel_values):
            expression, *_ = self.inner(pixel_values)
            return expression.index_select(-1, self.idx)

    wrapper = Subset(model, index).eval()

    # The processor is the authority on preprocessing; read it rather than
    # assuming ImageNet statistics, which is how a silently wrong export happens.
    ip = getattr(image_processor, "image_processor", image_processor)
    size = getattr(ip, "size", {}) or {}
    side = size.get("height") or size.get("shortest_edge") or 224
    mean = [float(v) for v in getattr(ip, "image_mean", [0.485, 0.456, 0.406])]
    std = [float(v) for v in getattr(ip, "image_std", [0.229, 0.224, 0.225])]

    if args.fp16:
        wrapper = wrapper.half()
        print("Exporting in fp16.")

    dtype = torch.float16 if args.fp16 else torch.float32
    dummy = torch.zeros(1, 3, side, side, dtype=dtype)
    out = pathlib.Path(args.out)
    print(f"Tracing at {side}x{side} → {out} …")
    torch.onnx.export(
        wrapper,
        dummy,
        str(out),
        input_names=["pixel_values"],
        output_names=["expression"],
        dynamic_axes={"pixel_values": {0: "batch"}, "expression": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )

    sidecar = {
        "id": f"deepspot-m-{len(genes)}",
        "name": f"DeepSpot-M ({len(genes)} genes)",
        "task": "virtual-spatial",
        "blurb": (
            "Predicts spatial gene expression from H&E. Frozen pathology encoder "
            "with a cross-attention gene decoder, so each gene is a query."
        ),
        "inputSize": side,
        # In 0-255 space, which is what the browser's normaliser works in.
        "mean": [m * 255 for m in mean],
        "std": [s * 255 for s in std],
        # The model was trained at ~20x; feeding it 40x tiles quietly changes
        # the scale it is reasoning about, so the reader resamples to this.
        "targetMpp": 0.5,
        "genes": genes,
        "licence": "CC-BY-NC-SA-4.0 (non-commercial)",
        "source": args.repo,
        "paper": "medRxiv 2026.06.19.26356060",
        "backend": args.backend,
        "precision": "fp16" if args.fp16 else "fp32",
    }
    side_path = out.with_suffix(".onnx.json")
    side_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    size_gb = out.stat().st_size / 1e9
    print(f"Wrote {out} ({size_gb:.2f} GB) and {side_path}")
    if size_gb > 2:
        print(
            "\nThat is past ONNX's 2 GB single-file limit and past what a 32-bit wasm heap\n"
            "can hold. Re-run with --fp16, or with fewer genes.",
            file=sys.stderr,
        )
    print("Import both in Slidecraft: Spatial → Import model…")
    print("Reminder: these weights are CC-BY-NC-SA-4.0. Non-commercial use only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
