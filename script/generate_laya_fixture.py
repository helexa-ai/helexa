"""Generate a tiny Laya parity fixture from the upstream reference (#336).

Builds the `laya` SDK's own `DecisionModel` (laya/common.py) around a
tiny ModernBERT, then records its outputs for a right-padded batch.

Writes, into --out:
  rl_agent_config.json   the decision-model config, as neuron parses it
  encoder/config.json    the ModernBERT config, saved by transformers
  model.safetensors      the reference's own weights, float32
  tokenizer.json         a placeholder word-level tokenizer (the fixture
                         is scored on token ids; loading needs a file)
  expected.json          the batch (ids, markers, question types) and the
                         reference's per-row scorer logits and act softmax

The shape is chosen so every mechanism is exercised at toy size:

- four encoder layers with a global layer on each side of two sliding
  ones, and a sliding window (local_attention = 8, so +-4 positions)
  far shorter than the 40-token row, so a window bug changes the result;
- two different rotary bases for global and sliding layers, so using
  the wrong one for either kind is visible;
- rows of different lengths, so padding is present and must be masked;
- a one-option row, which takes the reference's single-marker branch
  of the act features;
- every question type, so the type embedding is indexed three ways.

Weights are initialised well above transformers' default scale so
attention is sharp rather than near-uniform, and norm weights and biases
are perturbed away from their ones/zeros init: at init a missing norm
and a present one are indistinguishable.

Usage, in a venv holding the `laya` SDK and CPU torch:

    python script/generate_laya_fixture.py \
        --out crates/neuron/src/harness/testdata/laya/tiny
"""

import argparse
import json
import os

import torch
from safetensors.torch import save_file
from transformers import ModernBertConfig, AutoModel

from laya.common import DecisionModel

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()
torch.manual_seed(args.seed)
os.makedirs(os.path.join(args.out, "encoder"), exist_ok=True)

ecfg = ModernBertConfig(
    vocab_size=97,
    hidden_size=32,
    num_hidden_layers=4,
    num_attention_heads=2,
    intermediate_size=48,
    max_position_embeddings=128,
    local_attention=8,
    global_attn_every_n_layers=3,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    cls_token_id=1,
    sep_token_id=2,
    norm_eps=1e-5,
    hidden_activation="gelu",
    # Well above the 0.02 default: at 0.02 attention is near-uniform and
    # a window or rotary bug moves the logits only in the sixth decimal.
    initializer_range=0.3,
    rope_parameters={
        "full_attention": {"rope_theta": 1000.0, "rope_type": "default"},
        "sliding_attention": {"rope_theta": 300.0, "rope_type": "default"},
    },
)
encoder = AutoModel.from_config(ecfg, attn_implementation="sdpa")
model = DecisionModel(encoder, head_layers=2, n_act=2).eval()

with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name or name.startswith("scorer.0"):
            p.add_(0.3 * torch.randn_like(p))
        elif name.startswith("type_emb"):
            p.mul_(3.0)

ecfg.save_pretrained(os.path.join(args.out, "encoder"))
rl_cfg = {
    "encoder": "tiny-modernbert",
    "head_layers": 2,
    "max_len": 64,
    "head_max_len": 24,
    "act_costs": {"escalate": 0.5},
    "amp_dtype": "bf16",
    "temperature": [1.0, 1.0, 1.0],
    "temperature_by_options": {},
}
with open(os.path.join(args.out, "rl_agent_config.json"), "w") as f:
    json.dump(rl_cfg, f, indent=1)
# A placeholder: the fixture is scored on token ids, but a checkpoint
# directory is loaded with its tokenizer, so one must exist.
with open(os.path.join(args.out, "tokenizer.json"), "w") as f:
    json.dump({"version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
               "normalizer": None, "pre_tokenizer": {"type": "Whitespace"},
               "post_processor": None, "decoder": None,
               "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "a": 1, "b": 2},
                         "unk_token": "[UNK]"}}, f)

state = {k: v.detach().contiguous().float() for k, v in model.state_dict().items()}
save_file(state, os.path.join(args.out, "model.safetensors"))

# (length, marker positions, qtype). Ids avoid 0 (pad) inside a row.
rows = [
    (40, [3, 9, 17], 0),
    (23, [4, 11], 1),
    (9, [2], 2),
    (31, [5, 7, 12, 20, 26], 0),
]
seqs = []
for n, markers, qtype in rows:
    ids = torch.randint(1, ecfg.vocab_size, (n,)).tolist()
    seqs.append({"input_ids": ids, "markers": markers, "qtype": qtype})

b = len(seqs)
L = max(len(s["input_ids"]) for s in seqs)
K = max(len(s["markers"]) for s in seqs)
ids = torch.zeros((b, L), dtype=torch.long)
att = torch.zeros((b, L), dtype=torch.long)
mpos = torch.zeros((b, K), dtype=torch.long)
mmask = torch.zeros((b, K), dtype=torch.bool)
for i, s in enumerate(seqs):
    n = len(s["input_ids"])
    ids[i, :n] = torch.tensor(s["input_ids"])
    att[i, :n] = 1
    k = len(s["markers"])
    mpos[i, :k] = torch.tensor(s["markers"])
    mmask[i, :k] = True
qtype = torch.tensor([s["qtype"] for s in seqs])

with torch.no_grad():
    logits, act = model(ids, att, mpos, mmask, qtype)
act = torch.softmax(act.float(), -1)

for i, s in enumerate(seqs):
    k = len(s["markers"])
    s["logits"] = logits[i, :k].tolist()
    s["act"] = act[i].tolist()

with open(os.path.join(args.out, "expected.json"), "w") as f:
    json.dump({"seed": args.seed, "seqs": seqs}, f, indent=1)
print("wrote", args.out, "params:", sum(v.numel() for v in state.values()))
