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
import pathlib
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
    # 18, not 17: torch traces these encoders at 18 and then down-converts,
    # and the conversion emits a Split node the ONNX checker rejects. Asking
    # for 18 up front skips a lossy step that produces an invalid graph.
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument(
        "--fp16",
        action="store_true",
        help=(
            "Halve the file. A ViT-H at fp32 is around 2.5 GB, which passes "
            "ONNX's 2 GB single-file limit and spills into a sidecar .onnx.data "
            "the browser loader cannot take — and would not fit a 32-bit wasm "
            "heap even if it could."
        ),
    )
    ap.add_argument(
        "--preset",
        choices=["auto", "plain", "uni2", "virchow2"],
        default="auto",
        help=(
            "How to build the model and read its embedding. UNI2-h needs a "
            "specific timm configuration, and Virchow2's embedding is a "
            "concatenation the default forward does not produce — neither is "
            "recoverable from the hub alone, so they are named here."
        ),
    )
    ap.add_argument(
        "--token",
        default=None,
        help="HF token; prefer the HF_TOKEN environment variable so it stays out of your shell history",
    )
    args = ap.parse_args()

    # A `hf auth login` already stores a token, and refusing to use it means
    # telling someone who has authenticated correctly that they have not. The
    # library reads its own store here; the token is never printed or written
    # into the output.
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:  # noqa: BLE001 - an old hub version simply has no store
            token = None
    if not token:
        die(
            "no credentials. Run `hf auth login`, or set HF_TOKEN=hf_..., or pass --token.\n"
            "You must also accept the model's terms on its Hugging Face page."
        )

    try:
        import torch
        import timm
    except ImportError:
        die(f"missing dependencies. {REQUIREMENTS}")

    out = args.out or f"{args.model.split('/')[-1].lower()}.onnx"

    preset = args.preset
    if preset == "auto":
        name = args.model.lower()
        preset = "uni2" if "uni2" in name else "virchow2" if "virchow2" in name else "plain"
        if preset != "plain":
            print(f"detected {preset} from the repo name", file=sys.stderr)

    kwargs: dict = {"pretrained": True, "num_classes": 0}
    if preset == "uni2":
        # Straight from the model card. UNI2-h is not a stock timm config: get
        # any of these wrong and the weights load into the wrong shape, or load
        # silently into a subtly different model.
        kwargs = {
            "pretrained": True,
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
    elif preset == "virchow2":
        # Virchow2 needs its own MLP and activation "for proper init"; without
        # them the model builds and produces numbers that are not embeddings.
        kwargs = {
            "pretrained": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        }

    print(f"loading {args.model} ({preset}) …", file=sys.stderr)
    try:
        model = timm.create_model(f"hf-hub:{args.model}", **kwargs)
    except Exception as exc:  # noqa: BLE001 - surface the hub's own message
        die(
            f"could not load {args.model}: {exc}\n"
            "If this is a gating error, accept the model's terms on its Hugging Face page first."
        )

    model.eval()

    class Virchow2Embedding(torch.nn.Module):
        """
        Virchow2's embedding is a concatenation, not the forward pass.

        The model returns 261 tokens: a class token, four register tokens, then
        256 patch tokens. The published representation is the class token
        concatenated with the mean of the patch tokens — 2560 dimensions, not
        1280. Exporting the bare forward gives tokens that look like a valid
        output and are not the embedding anything was benchmarked on, so the
        construction is baked into the graph here rather than left to callers.
        """

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            tokens = self.inner(pixel_values)
            class_token = tokens[:, 0]
            patch_tokens = tokens[:, 5:]  # skip the four register tokens
            return torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)

    # Kept before wrapping: resolve_data_config reads timm's pretrained_cfg,
    # which a wrapper does not carry. Without this the normalisation quietly
    # falls back to ImageNet defaults, which is the kind of wrong that shows up
    # as "the encoder just is not very good".
    base = model
    if preset == "virchow2":
        model = Virchow2Embedding(model).eval()

    # timm carries the preprocessing the weights were trained with; reading it
    # here is what keeps the app's normalisation honest rather than guessed.
    cfg = timm.data.resolve_data_config({}, model=base)
    mean = [round(v * 255, 4) for v in cfg.get("mean", (0.485, 0.456, 0.406))]
    std = [round(v * 255, 4) for v in cfg.get("std", (0.229, 0.224, 0.225))]
    size = args.size or cfg.get("input_size", (3, 224, 224))[-1]

    if args.fp16:
        model = model.half()
        print("exporting in fp16", file=sys.stderr)

    dtype = torch.float16 if args.fp16 else torch.float32
    dummy = torch.zeros(1, 3, size, size, dtype=dtype)
    with torch.no_grad():
        dim = int(model(dummy).shape[-1])

    if preset == "virchow2" and dim != 2560:
        die(f"expected a 2560-d Virchow2 embedding, got {dim}. The token layout has changed.")
    if preset == "uni2" and dim != 1536:
        die(f"expected a 1536-d UNI2 embedding, got {dim}. Check the timm kwargs.")

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

    # torch writes the weights into a sidecar .onnx.data whenever the model is
    # large, whatever its actual size. The app loads one file, so anything that
    # still fits ONNX's 2 GB single-file limit is folded back in — otherwise a
    # perfectly good export is unusable for the sake of a default.
    data_file = pathlib.Path(f"{out}.data")
    if data_file.exists():
        total = pathlib.Path(out).stat().st_size + data_file.stat().st_size
        if total < 2 * 1024**3:
            try:
                import onnx as _onnx

                print(f"folding {data_file.name} back into a single file …", file=sys.stderr)
                model_proto = _onnx.load(out, load_external_data=True)
                _onnx.save_model(model_proto, out, save_as_external_data=False)
                data_file.unlink()
            except Exception as exc:  # noqa: BLE001 - the sidecar is still valid
                print(f"could not consolidate ({exc}); keeping {data_file.name}", file=sys.stderr)
        else:
            print(
                f"weights stay in {data_file.name}: {total / 1024**3:.2f} GB is past ONNX's "
                "2 GB single-file limit. Re-run with --fp16 for a single file.",
                file=sys.stderr,
            )

    meta = {
        "name": args.model.split("/")[-1],
        "inputSize": size,
        "dim": dim,
        "mean": mean,
        "std": std,
        "normalise": "custom" if mean != [123.675, 116.28, 103.53] else "imagenet",
        "preset": preset,
        "task": "encode",
        "precision": "fp16" if args.fp16 else "fp32",
        # A ViT-H is minutes per patch on wasm and seconds on WebGPU; the
        # fallback still exists, and the app reports which one it got.
        "backend": "webgpu",
        "source": args.model,
    }
    sidecar = f"{out}.json"
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    size_mb = os.path.getsize(out) / 1e6
    print(
        f"\nwrote {out} ({size_mb:.0f} MB) and {sidecar}\n\n"
        f"In Slidecraft: Models -> Import ONNX…, giving it both the .onnx and its .onnx.json\n"
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
