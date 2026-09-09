#!/usr/bin/env python3
"""Export a gated Hugging Face histology encoder to ONNX for Slidecraft.

UNI, UNI2-h, Virchow2, CONCH and GigaPath are distributed as timm/PyTorch
checkpoints behind a licence gate. The browser cannot run those directly, so
this converts one to ONNX on your machine; the resulting file is then imported
through the app's "Import ONNX…" dialog.

The token never leaves your machine and is never written to the output.

    export HF_TOKEN=hf_...                     # your own token
    python scripts/export_onnx.py MahmoodLab/UNI --out uni.onnx

It prints the exact values to type into the import dialog, because getting the
input size or normalisation wrong produces plausible-looking nonsense rather
than an error.

Licence note: UNI and Virchow2 are CC-BY-NC-ND-4.0. "ND" is no-derivatives, and
whether a format conversion counts as a derivative is a question for you and
whoever owns the licence — this script does not decide it for you.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REQUIREMENTS = "pip install torch timm onnx huggingface_hub"


def die(message: str) -> "None":
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="Hugging Face repo id, e.g. MahmoodLab/UNI")
    ap.add_argument("--out", default=None, help="output .onnx path (default: <name>.onnx)")
    ap.add_argument("--size", type=int, default=224, help="square input size in pixels (default 224)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--token",
        default=None,
        help="HF token; prefer the HF_TOKEN environment variable so it stays out of your shell history",
    )
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        die("no token. Set HF_TOKEN=hf_... (preferred) or pass --token.")

    try:
        import torch
        import timm
    except ImportError:
        die(f"missing dependencies. {REQUIREMENTS}")

    out = args.out or f"{args.model.split('/')[-1].lower()}.onnx"

    print(f"loading {args.model} …", file=sys.stderr)
    try:
        model = timm.create_model(
            f"hf-hub:{args.model}",
            pretrained=True,
            num_classes=0,  # features only: we want embeddings, not logits
        )
    except Exception as exc:  # noqa: BLE001 - surface the hub's own message
        die(
            f"could not load {args.model}: {exc}\n"
            "If this is a gating error, accept the model's terms on its Hugging Face page first."
        )

    model.eval()

    # timm carries the preprocessing the weights were trained with; reading it
    # here is what keeps the app's normalisation honest rather than guessed.
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = [round(v * 255, 4) for v in cfg.get("mean", (0.485, 0.456, 0.406))]
    std = [round(v * 255, 4) for v in cfg.get("std", (0.229, 0.224, 0.225))]
    size = args.size or cfg.get("input_size", (3, 224, 224))[-1]

    dummy = torch.zeros(1, 3, size, size)
    with torch.no_grad():
        dim = int(model(dummy).shape[-1])

    print(f"exporting to {out} (input {size}x{size}, embedding dim {dim}) …", file=sys.stderr)
    torch.onnx.export(
        model,
        dummy,
        out,
        input_names=["pixel_values"],
        output_names=["embedding"],
        # Batch must be dynamic: the app encodes a whole patch grid in one call.
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )

    try:
        import onnx

        onnx.checker.check_model(out)
    except ImportError:
        print("note: `onnx` not installed, skipping validation", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        die(f"the exported graph did not validate: {exc}")

    meta = {
        "name": args.model.split("/")[-1],
        "inputSize": size,
        "dim": dim,
        "mean": mean,
        "std": std,
        "normalise": "custom" if mean != [123.675, 116.28, 103.53] else "imagenet",
        "source": args.model,
    }
    sidecar = f"{out}.json"
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    size_mb = os.path.getsize(out) / 1e6
    print(
        f"\nwrote {out} ({size_mb:.0f} MB) and {sidecar}\n\n"
        f"In Slidecraft: right-click the click-to-segment tool -> Import ONNX…\n"
        f"  Name        {meta['name']}\n"
        f"  Encoder     {out}\n"
        f"  Input px    {size}\n"
        f"  Normalise   {meta['normalise']}  (mean {mean}, std {std})\n\n"
        f"This is a feature extractor: it produces {dim}-dimensional embeddings for the\n"
        f"Phase 3 training head, not click-to-segment masks.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
