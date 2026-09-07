"""Generate a qwen4_exp parity fixture from the upstream reference (#323).

Writes, into --out:
  config.json          the tiny config, as both implementations parse it
  model.safetensors    the reference's own weights, with the PLE n-gram
                       table split into `split_ngram_parts` shards on
                       dim 0 the way the released checkpoint ships it
  expected.safetensors input_ids and the reference's logits

The weights are the reference's, so a name we cannot find is a finding
rather than a fixture bug. Norm weights are perturbed away from their
zero init on purpose: at exactly zero, `(1 + w)` and `w` differ, but
`(1 + w)` and `1` do not, and the whole point is to tell those apart.
"""

import argparse, json, os
import torch
from safetensors.torch import save_file
from transformers.models.qwen4_exp import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, help="the tiny config.json both implementations parse")
ap.add_argument("--out", required=True, help="fixture directory to write")
# Seed 2 is tie-free; 0, 1 and 4 are not. See the tie check below.
ap.add_argument("--seed", type=int, default=2)
ap.add_argument("--tokens", type=int, default=12)
args = ap.parse_args()

torch.manual_seed(args.seed)
torch.use_deterministic_algorithms(True)

raw = json.load(open(args.config))
cfg = Qwen4ExpConfig(**raw)
tc = cfg.text_config

model = Qwen4ExpForConditionalGeneration(cfg).eval()

# Every parameter gets a non-degenerate value. Zeros hide a transposed
# load. But *small* values hide more than that: silu(x) ~ x/2 near the
# origin, so with weights in +-0.1 the SwiGLU is close enough to
# symmetric that reading gate and up the wrong way round changes the
# logits by 2e-4 — inside any tolerance you would write. The scale here
# is chosen to put activations where silu is actually curved, which is
# what makes the ordering observable, while staying below the point
# where the MoE router saturates and top-k stops discriminating.
with torch.no_grad():
    for name, p in model.named_parameters():
        if "visual" in name or "vision" in name:
            continue
        p.copy_(torch.empty_like(p).uniform_(-0.8, 0.8))

ids = torch.randint(0, tc.vocab_size, (1, args.tokens), dtype=torch.long)
# Keep eos out of the prompt: it starts a new n-gram segment, and the
# first fixture should not conflate segment handling with everything
# else. A separate fixture exercises it deliberately.
ids[ids == tc.eos_token_id] = (tc.eos_token_id + 1) % tc.vocab_size

# The QSA selection must not rest on a tie.
#
# A block scores exactly 0.0 whenever relu clamps every head's dot
# product, which is common enough that a random fixture hits it. When
# the k-th and (k+1)-th scores are equal, which block gets dropped is
# decided by `topk`'s ordering of equal elements — and torch does not
# specify that, nor does its CPU kernel agree with its CUDA one. A
# fixture containing such a tie measures that unspecified detail rather
# than our arithmetic, so we refuse to emit one.
orig_topk = torch.Tensor.topk
ties = []
def _topk(self, k, *a, **kw):
    if self.dim() == 1 and self.numel() <= 64 and k < self.numel():
        ordered = sorted(self.detach().tolist(), reverse=True)
        if ordered[k - 1] == ordered[k]:
            ties.append((self.detach().tolist(), k))
    return orig_topk(self, k, *a, **kw)
torch.Tensor.topk = _topk
with torch.no_grad():
    out = model(input_ids=ids, use_cache=False)
torch.Tensor.topk = orig_topk

if ties:
    raise SystemExit(
        f"seed {args.seed} produces {len(ties)} QSA selection tie(s), e.g. "
        f"scores={[round(v, 6) for v in ties[0][0]]} k={ties[0][1]}. "
        "Pick another --seed: the fixture must isolate the arithmetic."
    )

os.makedirs(args.out, exist_ok=True)

sd = {k: v for k, v in model.state_dict().items() if "visual" not in k and "vision" not in k}

# Split the runtime n-gram embedding back into the checkpoint's shard
# layout: dim 0, ascending, per upstream's own note that "the
# checkpoints for it are sharded on dim0" and loading concatenates them.
parts = tc.split_ngram_parts
for key in [k for k in sd if k.endswith("ngram_embedding.weight")]:
    table = sd.pop(key)
    rows = table.shape[0]
    assert rows % parts == 0, f"{rows} rows do not divide into {parts} shards"
    per = rows // parts
    base = key[: -len(".weight")]
    for s in range(parts):
        sd[f"{base}.shard_{s}.weight"] = table[s * per : (s + 1) * per].clone()

save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(args.out, "model.safetensors"))
save_file(
    {"input_ids": ids.to(torch.int64), "logits": out.logits.to(torch.float32).contiguous()},
    os.path.join(args.out, "expected.safetensors"),
)
with open(os.path.join(args.out, "config.json"), "w") as f:
    json.dump(raw, f, indent=2)

print(f"tokens      {args.tokens}")
print(f"tensors     {len(sd)}")
print(f"logits      {tuple(out.logits.shape)}")
print(f"logits absmax {out.logits.abs().max().item():.6f}")
print(f"finite      {bool(torch.isfinite(out.logits).all())}")
