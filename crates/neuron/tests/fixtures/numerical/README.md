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
| `qwen4_exp-tiny` | random weights, real architecture | text, 12 tokens | f32 | `qwen4_exp_logits_match_reference` |

## `qwen4_exp-tiny` is different, and runs in CI

The others replay a real snapshot and self-skip without
`NEURON_REF_MODEL_PATH`. This one is 44 KB of random weights in the
real architecture — 4 layers (3 linear-attention, 1 QSA), PLE on layer
1, 2 experts — generated *by* `transformers` itself via
[`script/generate_qwen4_exp_fixture.py`](../../../../../script/generate_qwen4_exp_fixture.py).
So it is checked in, needs no weights, and runs on every push.

That matters because the whole `qwen4_exp` architecture was ported from
a written spec (#308) and tested against our own reading of it. This is
the only thing in the tree that compares it to the implementation it
was ported from (#323).

Two properties of the fixture are load-bearing, and both were learned
by getting them wrong:

- **Weights are `uniform(-0.8, 0.8)`, not `uniform(-0.1, 0.1)`.** Near
  the origin `silu(x) ~ x/2`, so the SwiGLU is nearly symmetric and
  reading the fused expert's gate and up halves the wrong way round
  moved the logits by 2.4e-4 — inside any tolerance worth writing. At
  the larger scale the same mutation moves them by 2.7.
- **The generator refuses to emit a fixture containing a QSA selection
  tie.** A block scores exactly 0.0 whenever relu clamps every head, so
  ties are common with random weights; when the k-th and (k+1)-th
  scores are equal, which block is dropped is decided by `topk`'s
  ordering of equal elements, which torch does not specify and whose
  CPU and CUDA kernels need not agree. Seeds 0, 1 and 4 produce ties;
  seed 2 (the default) does not. See "the tie-break" below.

Regenerate with the reference from `transformers` **main** — the 5.9.0
release does not carry `qwen4_exp`:

```sh
python -m venv --system-site-packages .venv
.venv/bin/pip install --no-deps "transformers @ git+https://github.com/huggingface/transformers@main"
.venv/bin/pip install "tokenizers>=0.23.1,<0.24" "safetensors>=0.8.0"
.venv/bin/python script/generate_qwen4_exp_fixture.py \
    --config crates/neuron/tests/fixtures/numerical/qwen4_exp-tiny/config.json \
    --out crates/neuron/tests/fixtures/numerical/qwen4_exp-tiny
```

### The tie-break

Where the reference and this port genuinely differ: at an exact tie in
the QSA block scores, `topk` keeps one block and we keep another. Ours
is deterministic (lower block index wins); torch's is unspecified. A
tie means every head's dot product was clamped to zero for both blocks
— the indexer scored them equally worthless — so either choice is
defensible, and there is no stable target to match. The fixture avoids
the situation rather than pretending it is resolved.

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
