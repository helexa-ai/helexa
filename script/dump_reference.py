#!/usr/bin/env python3
"""Capture numerical-reference fixtures from HF transformers (#15).

Runs the reference Python implementation of an architecture neuron
serves (today: qwen3_5) on a fixed input and dumps the tensors a
companion Rust test (crates/neuron/tests/numerical_reference.rs)
replays and compares against. The fixtures pin the README's
"implemented in this repository, ported against the HuggingFace
reference" claim to checked-in numbers.

Cases:
  text    — a fixed >64-token prompt (long enough that neuron's
            chunked delta-rule prefill path is exercised), dumping
            the token ids and the final-position logits.
  vision  — a deterministic synthetic 448x448 image (factor-aligned,
            so resize is the identity and pixel-level preprocessing
            parity is part of what the comparison validates) plus a
            short prompt, dumping the expanded token ids, the image
            PNG, the LM grid, the vision tower's post-merger output,
            and the final-position logits.

Fixture layout (one directory per model+case):
  manifest.json          — model id, case, token ids, shapes, versions
  <name>.f32             — raw little-endian f32 tensor data
  image.png              — (vision only) the input image

Usage (on a host with torch + transformers and the model snapshot):
  python3 script/dump_reference.py \
      --model-path /path/to/hf/snapshot --case text \
      --out crates/neuron/tests/fixtures/numerical/qwen3_5-0.8b-text

Regenerate fixtures whenever the pinned model snapshot or the
transformers reference implementation changes; record both versions
from the manifest in the commit message.
"""

import argparse
import json
import os
import struct
import sys

# ---------------------------------------------------------------------------
# Compat shim: transformers 5.9 constructs kernels-hub repository
# objects at import time without the revision/version that kernels
# 0.15 requires. The hub kernels are never used here
# (USE_HUB_KERNELS=NO below); the constructors just must not throw.
os.environ.setdefault("USE_HUB_KERNELS", "NO")
try:
    import kernels.layer.layer as _kl
    import kernels.layer.func as _kf

    def _patch(cls):
        orig = cls.__init__

        def patched(self, *a, **kw):
            if "revision" not in kw and "version" not in kw:
                kw["revision"] = "main"
            orig(self, *a, **kw)

        cls.__init__ = patched

    _patch(_kl.LayerRepository)
    _patch(_kf.FuncRepository)
except Exception:  # noqa: BLE001 — older/newer kernels may not need it
    pass
# ---------------------------------------------------------------------------

import torch  # noqa: E402

# Long enough (>64 tokens) that neuron's replay takes the chunked
# delta-rule prefill path; plain prose so the tokenization is stable.
TEXT_PROMPT = (
    "The helexa fleet serves near-frontier language models on consumer "
    "graphics cards. Each host runs a small daemon that discovers its "
    "hardware, loads the configured models, and answers OpenAI-compatible "
    "requests over the private mesh network. The gateway routes each "
    "request to the host that already holds the model, restores any "
    "cached prefix state, and streams the generated tokens back to the "
    "caller one chunk at a time. Operators care about three numbers: the "
    "time to the first token, the steady decode rate, and the time a "
    "cold model takes to become ready after a deploy. This paragraph "
    "exists only to be tokenized identically by two implementations."
)

VISION_PROMPT = "Describe this image in one sentence."


def write_f32(path, tensor):
    data = tensor.detach().to(torch.float32).cpu().contiguous().reshape(-1)
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{data.numel()}f", *data.tolist()))


def synthetic_image(size=448):
    """Deterministic, NON-periodic RGB pattern. Every patch must be
    unique: periodic patterns (checkerboards) make many patches exact
    duplicates, and attention over near-identical keys is
    ill-conditioned — tiny dtype rounding then amplifies chaotically
    and the fixture comparison drowns in noise. The x*y term breaks
    all translational symmetry while staying byte-deterministic."""
    from PIL import Image

    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            r = (x * 255) // size
            g = (y * 255) // size
            b = (x * y) % 251
            px[x, y] = (r, g, b)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="HF snapshot dir or repo id")
    ap.add_argument("--case", choices=["text", "vision"], required=True)
    ap.add_argument("--out", required=True, help="fixture directory to write")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="reference compute dtype. float32 (default) pins the math "
        "itself — the Rust replay compares f32-to-f32 and implementation "
        "bugs are not masked by (or blamed on) bf16 rounding chaos.",
    )
    ap.add_argument(
        "--processor-path",
        default=None,
        help="where to load the tokenizer/processor from (defaults to "
        "--model-path; pass the repo id with HF_HUB_CACHE pointed at a "
        "writable scratch dir when the local snapshot is missing "
        "preprocessor_config.json)",
    )
    args = ap.parse_args()

    import transformers
    from transformers import AutoProcessor, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
    )

    os.makedirs(args.out, exist_ok=True)
    manifest = {
        "model_path": args.model_path,
        "case": args.case,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "files": {},
    }

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    manifest["dtype"] = args.dtype
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path, dtype=dtype, device_map=args.device
    )
    model.eval()

    if args.case == "text":
        tok = AutoTokenizer.from_pretrained(args.processor_path or args.model_path)
        ids = tok(TEXT_PROMPT, return_tensors="pt").input_ids
        manifest["prompt"] = TEXT_PROMPT
        manifest["token_ids"] = ids[0].tolist()
        with torch.no_grad():
            logits = model(input_ids=ids.to(model.device)).logits[0, -1]
        write_f32(os.path.join(args.out, "logits.f32"), logits)
        manifest["files"]["logits"] = {"file": "logits.f32", "shape": [logits.shape[-1]]}
    else:
        processor = AutoProcessor.from_pretrained(args.processor_path or args.model_path)
        img = synthetic_image()
        img.save(os.path.join(args.out, "image.png"))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        manifest["prompt"] = VISION_PROMPT
        manifest["token_ids"] = inputs["input_ids"][0].tolist()
        manifest["image_grid_thw"] = inputs["image_grid_thw"][0].tolist()
        with torch.no_grad():
            visual_out = model.model.visual(
                inputs["pixel_values"].to(model.device, dtype),
                grid_thw=inputs["image_grid_thw"].to(model.device),
            )
            # transformers 5.x returns BaseModelOutputWithPooling:
            # pooler_output is the post-merger embedding the LM
            # splices (= neuron's VisionTower::forward output);
            # last_hidden_state is the pre-merger grid.
            if hasattr(visual_out, "pooler_output"):
                visual_out = visual_out.pooler_output
            logits = model(
                **{k: v.to(model.device) for k, v in inputs.items()}
            ).logits[0, -1]
        write_f32(os.path.join(args.out, "visual_out.f32"), visual_out)
        manifest["files"]["visual_out"] = {
            "file": "visual_out.f32",
            "shape": list(visual_out.shape),
        }
        write_f32(os.path.join(args.out, "logits.f32"), logits)
        manifest["files"]["logits"] = {"file": "logits.f32", "shape": [logits.shape[-1]]}

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote fixture: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
