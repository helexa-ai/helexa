# Numerical-reference fixtures (#15)

Reference tensors captured from the HF `transformers` implementation
by [`script/dump_reference.py`](../../../../../script/dump_reference.py),
replayed and compared by
[`tests/numerical_reference.rs`](../../numerical_reference.rs). These
pin the README's "implemented in this repository, ported against the
HuggingFace reference" claim to checked-in numbers.

| fixture | model | case | dtype | compared by |
|---|---|---|---|---|
| `qwen3_5-0.8b-text` | Qwen/Qwen3.5-0.8B | text (>64-token prompt → chunked GDN prefill) | f32 | `text_logits_match_reference` |
| `qwen3_5-0.8b-vision` | Qwen/Qwen3.5-0.8B | 448×448 synthetic image + prompt | f32 | `vision_tower_and_logits_match_reference` |
| `qwen3_6-27b-text` | Qwen/Qwen3.6-27B | text | bf16 | manual (see below) |

## Running the comparison

On a host with the model snapshot (beast):

```sh
NEURON_REF_MODEL_PATH=/archive3/llm-cache/models--Qwen--Qwen3.5-0.8B/snapshots/<rev> \
    cargo test -p neuron --test numerical_reference -- --nocapture
```

Without `NEURON_REF_MODEL_PATH` the tests compile and self-skip, so CI
stays green without weights.

## Why f32 fixtures

f32-vs-f32 isolates implementation differences: observed agreement is
text max_abs 0.000 / cosine 1.000000, vision tower cosine 0.999998.
Cross-dtype comparisons drown in bf16 rounding chaos through the
27-layer tower (global cosine ~0.997, worst patch ~0.92, worst index
unstable across runs) — that is production-dtype noise, not
implementation error. The mutation check: rerunning with
`NEURON_VISION_LEGACY_POS=1` (the deliberately-wrong sequential
pos-embed lookup) collapses tower cosine to 0.75 / worst patch 0.28
and fails the test loudly.

## The 27B fixture

`qwen3_6-27b-text` is captured in bf16 on CPU (an f32 27B forward
needs ~108 GB; beast has 91 GB free). The automated tests run against
the 0.8B because both models execute the *same* arch modules — the
27B differs only in hyperparameters — and an apples-to-apples 27B
replay needs either TP=2 bf16 (idle GPUs, no neuron running) or a
bigger-RAM host. Manual procedure when wanted: stop neuron on beast,
replay the manifest's token ids through a TP=2 bf16 load, compare
argmax + cosine against `logits.f32` with bf16-calibrated tolerances.

## Regenerating

Regenerate whenever the pinned snapshot or the transformers reference
changes; record both versions (in each `manifest.json`) in the commit
message:

```sh
# on beast; processor files may be missing from neuron's snapshot —
# point the processor at the repo id with a scratch cache
SNAP=$(ls -d /archive3/llm-cache/models--Qwen--Qwen3.5-0.8B/snapshots/*/ | head -1)
HF_HUB_CACHE=/tmp/hf-ref-cache python3 script/dump_reference.py \
    --model-path "$SNAP" --processor-path Qwen/Qwen3.5-0.8B \
    --case text   --out crates/neuron/tests/fixtures/numerical/qwen3_5-0.8b-text
HF_HUB_CACHE=/tmp/hf-ref-cache python3 script/dump_reference.py \
    --model-path "$SNAP" --processor-path Qwen/Qwen3.5-0.8B \
    --case vision --out crates/neuron/tests/fixtures/numerical/qwen3_5-0.8b-vision
```
