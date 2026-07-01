#!/usr/bin/env python3
"""Synthesize a tiny qwen3_next parity fixture (#92).

Unlike script/dump_reference.py (which replays a real HF snapshot and
therefore needs the weights on disk), this builds a TINY random-weight
`Qwen3NextForCausalLM` from scratch, saves the checkpoint INTO the
fixture directory, runs the reference forward on fixed token ids, and
dumps the final-position logits. The whole fixture (weights included)
is a few hundred KB, so it is committed and the companion Rust test
(crates/neuron/tests/qwen3_next_parity.rs) runs everywhere — no env
var, no snapshot, CI included.

What this pins, exactly: neuron's qwen3_next wiring against upstream —
flat config normalisation, the `model.*` weight prefix, the
per-key-head-group de-interleave of the fused `in_proj_qkvz` /
`in_proj_ba` projections, hybrid layer interleaving, and the MoE block
(softmax→top-k→renorm routing, per-expert SwiGLU, shared expert +
sigmoid gate). The full-size 80B checkpoint differs only in dimensions.

The config mirrors the real 80B's *shape decisions* at doll-house
scale: interval-4 hybrid, 8 layers (so two full-attention layers),
every layer MoE (decoder_sparse_step 1), 16 experts / top-4 + shared
expert, partial rotary 0.25, 2 KV heads.

Usage (host with torch + transformers>=4.57, e.g. beast):
  python3 script/dump_qwen3_next_tiny.py \
      --out crates/neuron/tests/fixtures/numerical/qwen3_next-tiny

Regenerate whenever the transformers reference implementation changes;
record the transformers version from the manifest in the commit
message.
"""

import argparse
import json
import os
import struct

# ---------------------------------------------------------------------------
# Compat shim (same as dump_reference.py): transformers 5.9 constructs
# kernels-hub repository objects at import time without the
# revision/version that kernels 0.15 requires. The hub kernels are
# never used here; the constructors just must not throw.
os.environ.setdefault("USE_HUB_KERNELS", "NO")
try:
    import kernels.layer.func as _kf
    import kernels.layer.layer as _kl

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

# Fixed input: 96 token ids (> 64 so neuron's chunked delta-rule
# prefill path is exercised), deterministic arithmetic sequence folded
# into the tiny vocab.
TOKEN_IDS = [(7 * i + 3) % 512 for i in range(96)]
SEED = 92


def tiny_config():
    from transformers import Qwen3NextConfig

    return Qwen3NextConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        partial_rotary_factor=0.25,
        rope_theta=10000000,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        full_attention_interval=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_value_head_dim=16,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        moe_intermediate_size=32,
        norm_topk_prob=True,
        num_experts=16,
        num_experts_per_tok=4,
        shared_expert_intermediate_size=32,
    )


def write_f32(path, tensor):
    data = tensor.detach().to(torch.float32).cpu().contiguous().reshape(-1)
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{data.numel()}f", *data.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="fixture directory to write")
    args = ap.parse_args()

    import transformers
    from transformers import Qwen3NextForCausalLM

    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(SEED)
    cfg = tiny_config()
    model = Qwen3NextForCausalLM(cfg).to(torch.float32).eval()

    # Save the checkpoint into the fixture itself (config.json +
    # model.safetensors) — the Rust test loads neuron's implementation
    # from exactly these files.
    model.save_pretrained(args.out, safe_serialization=True)

    ids = torch.tensor([TOKEN_IDS], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids)
    logits = out.logits[0, -1]  # final position, (vocab,)

    write_f32(f"{args.out}/logits.f32", logits)
    manifest = {
        "case": "qwen3_next-tiny",
        "seed": SEED,
        "token_ids": TOKEN_IDS,
        "files": {"logits": {"file": "logits.f32", "shape": [cfg.vocab_size]}},
        "versions": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
    }
    with open(f"{args.out}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"fixture written to {args.out}")
    print(f"  transformers {transformers.__version__}, torch {torch.__version__}")
    print(f"  logits[:4] = {logits[:4].tolist()}")


if __name__ == "__main__":
    main()
