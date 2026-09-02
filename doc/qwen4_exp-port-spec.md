# `qwen4_exp` port spec

Reference specification for `Qwen/Qwen3.8-Flash-Next`, written so that
the decoder can be implemented without re-reading the 131 KB upstream
Python. Every dimension in this document was read from the checkpoint's
own safetensors headers or from the pinned reference source; nothing is
inferred from the model card.

Companion to issue #308 (child A of epic #307).

## Provenance

| what | pinned value |
|---|---|
| reference source | `huggingface/transformers` `src/transformers/models/qwen4_exp/`, commit `fc5c5bde8e656dad91cbf34e61940d984b1c7b91` ("Add Qwen4Exp model", #48337), read at `main` = `5f8ab9bb53ec9e0c9329153d18bd825ff1db80f9` |
| checkpoint | `Qwen/Qwen3.8-Flash-Next`, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| declared | `model_type: qwen4_exp`, `architectures: [Qwen4ExpForConditionalGeneration]`, `transformers_version: 5.8.0.dev0` |

The checkpoint ships **no remote code**, so the transformers package is
the only authority. Note that `modeling_qwen4_exp.py` has already been
amended once since the model landed (commit `83d46aa2`, "Make gated
delta rule more explicit") — re-diff before trusting this document for
the linear-attention path.

## Measured parameter split

Read from all 131 shard headers (`total_size` 359,999,963,128 B;
1,658 tensors; 1,655 BF16 + 3 I64). This **replaces** the estimate in
#307 — the shape of the estimate was right, the residual bucket was
understated because MTP was counted as dense.

| component | params | share | BF16 | tensors |
|---|---:|---:|---:|---:|
| routed experts | 120,795,955,200 | 67.11% | 241.6 GB | 96 |
| PLE n-gram tables | 51,200,245,760 | 28.44% | 102.4 GB | 128 |
| MTP head | 2,607,150,848 | 1.45% | 5.2 GB | 31 |
| linear attention | 2,086,510,464 | 1.16% | 4.2 GB | 324 |
| hyper-connections | 640,624,640 | 0.36% | 1.3 GB | 387 |
| `embed_tokens` | 635,699,200 | 0.35% | 1.3 GB | 1 |
| `lm_head` | 635,699,200 | 0.35% | 1.3 GB | 1 |
| full attention | 597,694,464 | 0.33% | 1.2 GB | 72 |
| vision tower | 448,931,056 | 0.25% | 0.9 GB | 333 |
| shared experts | 236,052,480 | 0.13% | 0.5 GB | 192 |
| MoE routers | 62,914,560 | 0.03% | 0.1 GB | 48 |
| PLE consumption (conv1d, projections, norms) | 32,839,683 | 0.02% | 0.1 GB | 7 |
| QSA indexers | 19,663,872 | 0.01% | 0.04 GB | 36 |
| **total** | **179,999,981,459** | | **360.0 GB** | **1,658** |

The offload plan in #307 survives contact with the measurement:

- **sparse-access, offloadable: 171.996 B (95.55%)** — routed experts
  (10 of 512 per token) plus PLE tables (a few rows per token, one
  layer).
- **always-resident text path: 4.94 B (2.74%)** — everything above
  minus MTP and vision. 9.9 GB at BF16.
- MTP (2.61 B) is only resident if speculative decoding is on (#313);
  vision (0.45 B) only if the vision tower is loaded (#314).

## Skeleton

```
hidden_size            2560          num_hidden_layers   48
head_dim               256           num_attention_heads 24
num_key_value_heads    2             vocab_size          248,320
max_position_embeddings              262,144
layer_types            36 linear_attention + 12 full_attention,
                       full_attention_interval 4 → layers 3,7,11,…,47
num_experts            512           num_experts_per_tok 10
moe_intermediate_size  640           shared_expert_intermediate_size 640
norm_topk_prob         true (default; absent from config.json)
hc_count               4             hc_lowrank          320
ple_layer_ids          [2]  (one-indexed → zero-indexed layer 1)
rms_norm_eps           1e-6          hidden_act          silu
output_gate_type       sigmoid       mamba_ssm_dtype     float32
```

`layer_types` is authoritative; `full_attention_interval` is a hint. The
same convention as `arch/qwen3_5/`.

### Top-level dataflow

```
input_ids ──► embed_tokens ──► x [B,T,2560]
                                 │
                    repeat(1,1,hc_count) ──► h [B,T,10240]     4 identical streams
                                 │
        ┌────────────────────────▼───────────────────────────────┐
        │ for layer in 0..48:                                    │
        │     if layer == 1:  h = h + PLE(h, input_ids)          │  [B,T,10240]
        │     h = decoder_layer(h)                               │
        └────────────────────────┬───────────────────────────────┘
                                 │
                 hyper_connection_mixer(h)   (use_combine=False)
                                 │
                                 ▼  [B,T,2560]
                             lm_head
```

Two things to note, because both differ from every arch we serve:

1. **The inter-layer residual is 10240 wide, not 2560.** Four parallel
   residual streams, initialised as four copies of the token embedding.
2. **There is no final `model.norm`.** The `hc_norm` inside
   `hyper_connection_mixer` is the only normalisation before `lm_head`.
   Do not go looking for a missing tensor — the checkpoint has
   `hyper_connection_mixer.{hc_norm,input_mix_weight_down,input_mix_weight_up}`
   and no `norm.weight`.

### Decoder layer

There is **no `input_layernorm` and no `post_attention_layernorm`**. The
hyper-connection block absorbs both roles: it normalises, mixes the four
streams down to one 2560 vector for the sublayer, and scatters the
sublayer's output back across the four streams.

```
h [B,T,10240]
  │
  ├─ (layer 1 only) h = h + PLE(h, ple_input_ids)
  │
  ├─ (x, h_saved, inj) = attn_hyper_connection(h)      x:[.,2560] inj:[.,4]
  │   x = linear_attn(x)  |  x = self_attn(x)          → [.,2560]
  │   h = h_saved + (x ⊗ inj).flatten(-2)              → [.,10240]
  │
  └─ (x, h_saved, inj) = mlp_hyper_connection(h)
      x = sparse_moe(x)                                → [.,2560]
      h = h_saved + (x ⊗ inj).flatten(-2)              → [.,10240]
```

`x ⊗ inj` is the outer product `x.unsqueeze(-2) * inj.unsqueeze(-1)`,
giving `[.,4,2560]` — the sublayer output scaled by a per-stream weight
before being added into each stream.

## Component specs

### 1. Hyper-connections (`Qwen4ExpTextGatedResidual`)

Tensors, per instance:

```
hc_norm.weight              (10240,)
input_mix_weight_down.weight  (320, 10240)
input_mix_weight_up.weight    (10240, 320)
block_inject_weight.weight    (4, 10240)     # absent on the final mixer
```

Instances: `layers.{i}.attn_hyper_connection`,
`layers.{i}.mlp_hyper_connection` for all 48 layers, plus
`model.language_model.hyper_connection_mixer` and
`mtp.hyper_connection_mixer` (both `use_combine=False`, i.e. no
`block_inject_weight`). 387 + 3 tensors.

Forward, exactly:

```
hn   = hc_norm(h)                                   # grouped RMSNorm, see §8
w    = silu(input_mix_weight_down(hn) / hc_count)   # [.,320]
w    = sigmoid(input_mix_weight_up(w))              # [.,10240]
w    = w.unflatten(-1, (4, 2560))
x    = (w * hn.unflatten(-1, (4, 2560))).mean(dim=-2)     # [.,2560]
inj  = 2 * sigmoid(block_inject_weight(hn) / hc_count)    # [.,4]
return x, h, inj                                    # note: h, NOT hn
```

Three details that are easy to get wrong and produce fluent-but-worse
output rather than a crash:

- The **division by `hc_count` (4)** happens inside both gates, before
  the nonlinearity.
- The mix uses `.mean()` over the four streams, not `.sum()`.
- The residual returned for the skip connection is the **un-normalised**
  `h`. Only the mixing and injection paths see `hn`.

### 2. PLE — per-layer embedding over hashed n-grams

Applied at zero-indexed **layer 1** (`ple_layer_ids: [2]` is
one-indexed; upstream computes
`config.ple_layer_ids.index(layer_idx + 1)`, and the checkpoint
confirms it — the tensors live at
`model.language_model.layers.1.ple.*`).

#### Table geometry

```
ngram_size            3        heads_per_ngram   8
ngram_heads = (ngram_size - 1) * heads_per_ngram = 16
ple_embed_dim         2560  →  head_dim_per_ngram = 2560 / 16 = 160
ngram_vocab_size_base 20,000,000
make_ngram_vocab_size_divisible_by 128
split_ngram_parts     128
```

Head *h* (0..15) gets vocab size = the (h+1)-th prime strictly greater
than 19,999,999; offsets are the running cumulative sum. Total padded up
to a multiple of 128 gives **320,001,536 rows × 160 dims**, sharded as
128 × `(2500012, 160)` tensors named
`ple_embedding.ngram_embedding.shard_{0..127}.weight`. The shards are a
**flat split of the concatenated table**, not a per-head split —
concatenate along dim 0 in shard-index order and the global row index
from the hash indexes straight into it.

Do **not** reimplement the prime search or the splitmix64 multiplier
derivation. All three derived quantities ship in the checkpoint as I64
buffers and should simply be loaded:

```
ple_embedding.ngram_heads_offsets      (16,)
ple_embedding.ngram_heads_vocab_sizes  (16,)
ple_embedding.layer_multipliers        (3,)
```

#### Hashing

Let `tok[s]` be the input ids shifted right by `s` (s = 0,1,2) and
`m = layer_multipliers`. For n-gram order *n* ∈ {2, 3}, occupying heads
`[(n-2)*8, (n-2)*8 + 8)`:

```
mixed = tok[0] * m[0]
for p in 1..n:  mixed ^= tok[p] * m[p]        # XOR, int64
row[head] = (mixed mod vocab_sizes[head]) + offsets[head]
```

so each token produces 16 row indices, gathers 16 × 160 values and
flattens to a 2560-wide embedding. Arithmetic is int64 and the
multipliers are ~10^13, so the products overflow int32 by a wide margin
— this must be done in 64-bit.

There is **no unigram head**: with `ngram_size = 3` the loop runs
n = 2,3 only. `layer_multipliers` has three entries because the trigram
needs three shift multipliers.

#### Shifting is segment-aware

`_shift_right_ignore_eos` does not shift across an EOS token. Within
each EOS-delimited segment, positions closer to the segment start than
`shift` get filled with `eos_token_id` (248044) instead of borrowing
the previous document's tokens. Additionally
`Qwen4ExpTextModel.forward` replaces padded positions in
`ple_input_ids` with EOS before the layer is called. Getting this wrong
is invisible on a single-turn prompt and wrong on anything batched or
multi-document.

#### Consumption

```
key_proj.weight    (10240, 2560)     value_proj.weight (2560, 2560)
norm_key.weight    (10240,)          norm_query.weight (10240,)
norm_conv.weight   (10240,)          conv1d.weight     (10240, 1, 4)
```

```
e     = ngram_lookup(input_ids)                             # [.,2560]
k     = norm_key(key_proj(e)).unflatten(-1, (4, 2560))
v     = value_proj(e)                                       # [.,2560]
q     = norm_query(h).unflatten(-1, (4, 2560))              # h is the 10240 stream
g     = (k * q).sum(-1, keepdim=True) / sqrt(2560)          # [.,4,1]
g     = sign(g) * sqrt(clamp_min(|g|, 1e-6))                # signed sqrt
gv    = sigmoid(g) * v.unsqueeze(-2)                        # [.,4,2560]
out   = gv.flatten(-2) + short_conv(norm_conv(gv.flatten(-2)))
```

`short_conv` is a **depthwise, dilated** Conv1d: `groups = 10240`,
`kernel_size = ple_conv_kernel_size = 4`, **`dilation = ngram_size = 3`**,
no bias, followed by SiLU. Its receptive field is
`(4 - 1) * 3 = 9` positions of left context, which is also the cached
conv-state length. The dilation is the part most likely to be dropped
by accident — it is not a plain causal conv.

The three norms are grouped RMSNorm (§8) over 10240 in groups of 2560.

#### Cache state

PLE needs two pieces of per-request state, both stored upstream as
extra "conv states" on layer 1:

| slot | shape | contents |
|---|---|---|
| `conv_states[1]` | `(B, 10240, 9)` | short-conv left context |
| `conv_states[2]` | `(B, 2)` | previous `ngram_size - 1` input **ids**, EOS-initialised |

The second is token ids, not activations. Reset both when a request's
KV cache is cleared.

### 3. QSA sparse attention indexer

On the 12 full-attention layers and the MTP layer.

```
indexer_n_heads 4   indexer_kv_heads 1   indexer_head_dim 128
indexer_budget 2048 indexer_compress_ratio 4
block_topk = indexer_budget / indexer_compress_ratio = 512

index_qk_proj.weight (640, 2560)   # 4*128 query + 1*128 key, fused
q_layernorm.weight   (128,)
k_layernorm.weight   (128,)
```

`indexer_budget` is in **tokens**; 512 *blocks* of 4 tokens are
selected. `index_qk_proj` is a fused projection: the first 512 output
channels are the 4 query heads, the last 128 are the single index key.

Per query position:

1. `q = q_layernorm(q); q = RoPE(q, pos)` — RoPE at the query's own
   position, using the same partial-rotary cos/sin as the main
   attention (64 of 128 dims rotated).
2. The raw index key (**pre-norm, pre-RoPE**) is appended to a
   **separate indexer KV cache**, one head × 128 dims.
3. Take the causally-visible key positions. Group them into
   `floor(n_visible / 4)` complete blocks of 4 **consecutive visible
   positions**.
4. Pool: `mean` over the 4 keys **in float32**, then `k_layernorm`, then
   RoPE at the **first token position of the block**.
5. Score: `relu(q @ pooled_k^T).sum(over the 4 index heads) / sqrt(128)`.
   The ReLU is before the head sum — a signed sum is a different model.
6. Select the top `min(512, n_blocks)` blocks; expand back to their
   4 token positions each.
7. **Unconditionally append the tail** — the `n_visible mod 4` positions
   that did not form a complete block are always attended, never
   scored.
8. The selection becomes a mask that is ANDed (bool) or added (float)
   onto the causal mask before the main attention.

Consequences for the port:

- **The full KV cache is still materialised for every position.** #307's
  24 KiB/token arithmetic holds. QSA bounds the *attention compute*, not
  the cache.
- **There is a second cache.** 12 layers × 1 head × 128 dims. At BF16
  that is 3 KiB/token on top of the main 24 KiB/token — #307's KV table
  omitted it. llama.cpp's own accounting on a 2-GPU box reports
  6144 MB main + 2304 MB indexer at the full 262,144 window, i.e. the
  indexer is a real ~27% surcharge, not a rounding error.
- Below the budget (fewer than 2048 visible tokens) selection is
  effectively a no-op: every complete block gets selected and the tail
  is appended, so the result is dense causal attention. **Short-prompt
  parity against a dense implementation is therefore a valid test**, and
  is the cheapest correctness gate available.
- The reference is a Python double loop over batch × query position. It
  is a correctness oracle only — a usable implementation must batch the
  block pooling and score in one pass, and gather rather than build a
  dense `[T, kv_len]` mask.

### 4. Full attention

```
q_proj.weight (12288, 2560)   # 24 heads × 256 × 2 — query and gate
k_proj.weight   (512, 2560)   # 2 kv heads × 256
v_proj.weight   (512, 2560)
o_proj.weight (2560, 6144)
q_norm.weight   (256,)  k_norm.weight (256,)
```

Mechanically identical to `arch/qwen3_5/full_attn.rs`: GQA 24:2, per-head
RMSNorm on q and k, a sigmoid output gate carried in the widened
`q_proj`, scaling `head_dim^-0.5`.

The gate split is **per head, not a flat halving**: reshape to
`(..., n_heads, 2 * head_dim)` and chunk on the last axis, so within
each head the first 256 channels are the query and the last 256 are the
gate. Splitting the 12288 vector in half is a different and wrong
permutation.

The gate is applied to the attention output **before** `o_proj`:
`attn_out = attn_out * sigmoid(gate)`.

### 5. Linear attention (GatedDeltaNet)

```
linear_num_key_heads 16   linear_key_head_dim   128
linear_num_value_heads 48 linear_value_head_dim 128
linear_conv_kernel_dim 4  → conv_dim = 16*128*2 + 48*128 = 10240

in_proj_qkv.weight (10240, 2560)   in_proj_z.weight (6144, 2560)
in_proj_a.weight      (48, 2560)   in_proj_b.weight   (48, 2560)
conv1d.weight    (10240, 1, 4)     A_log (48,)  dt_bias (48,)
norm.weight  (128,)                out_proj.weight (2560, 6144)
```

**Tensor names and layout are identical to Qwen3.6's split (non-fused)
form**, which `arch/qwen3_5/linear_attn.rs` already loads — it branches
on the presence of `in_proj_qkvz.weight` and takes the split path
otherwise. Same delta rule, same conv, same state shapes.

One real difference: the output gated RMSNorm's activation is
`config.output_gate_type or config.hidden_act`, and this config sets
**`output_gate_type: sigmoid`**. Our `Qwen3_5RmsNormGated` hardcodes
SiLU. This must become configurable, or the linear-attention output is
wrong on all 36 layers.

`mamba_ssm_dtype: float32` — upstream carries the recurrent state in
f32. We default the equivalent GatedDeltaNet state to a bf16 round-trip
(#284, `NEURON_GDN_STATE_F32` off). **A port must not silently inherit
our default against upstream's explicit choice**: default this arch to
f32 state and measure the cost, rather than assuming #284's tradeoff
transfers.

### 6. MoE

```
mlp.gate.weight            (512, 2560)
mlp.experts.gate_up_proj   (512, 1280, 2560)
mlp.experts.down_proj      (512, 2560, 640)
mlp.shared_expert.{gate,up}_proj.weight (640, 2560)
mlp.shared_expert.down_proj.weight      (2560, 640)
mlp.shared_expert_gate.weight (1, 2560)
```

```
shared = sigmoid(shared_expert_gate(x)) * shared_expert(x)
probs  = softmax(gate(x), dim=-1, dtype=f32)      # over all 512
w, idx = topk(probs, 10)
w      = w / w.sum(-1, keepdim=True)              # norm_topk_prob = true
out    = sum_k w_k * expert_{idx_k}(x)  +  shared
```

Routing arithmetic matches `arch/qwen3_5/moe.rs` exactly. The **storage
layout does not**: this checkpoint ships experts as fused 3D tensors,
where ours loads per-expert `mlp.experts.{i}.{gate,up,down}_proj`. In
`gate_up_proj` the first 640 output rows are the gate and the last 640
are the up projection. The fused layout is the better one for a grouped
GEMM; the loader needs a new branch, not a new algorithm.

Sparsity is extreme: 10 of 512, `moe_intermediate_size` only 640.
Per token, 10 × (1280 + 2560) × 2560 ≈ 98.3 M params ≈ 2.36 B across
48 layers — the number that drives every placement decision in #309.

### 7. Rotary — interleaved M-RoPE

```
rope_theta 1e7    partial_rotary_factor 0.25    mrope_section [11,11,10]
rotary_dim = head_dim * 0.25 = 64  →  32 inverse frequencies
```

`sum(mrope_section) = 32` = the number of frequencies. The interleave
assigns frequency index *i* to axis T, H or W:

```
freqs = freqs[0]                      # start from temporal
freqs[..., 1:33:3] = freqs_H[..., 1:33:3]     # 11 indices: 1,4,…,31
freqs[..., 2:30:3] = freqs_W[..., 2:30:3]     # 10 indices: 2,5,…,29
```

i.e. the layout is `T,H,W,T,H,W,…` with temporal keeping the remainder,
rather than three contiguous chunks. Rotation is `rotate_half` over the
**first 64 of 256** head dims; the remaining 192 pass through unrotated.

**This is byte-for-byte the configuration `arch/qwen3_5/rope.rs`
already implements** (`head_dim 256`, `partial_rotary_factor 0.25`,
`mrope_section [11,11,10]`, `mrope_interleaved true`) — the module's
own tests pin exactly these values. Verify the index sets against the
slices above, but expect this item to be free.

`rope_theta` differs between the main model (1e7) and the MTP layer's
config block (also 1e7 here — `mtp.rope_theta: 10000000`); do not
assume they always agree.

### 8. Norms

Two conventions coexist in the same file, and mixing them up is silent:

| variant | scale | used by |
|---|---|---|
| `Qwen4ExpTextRMSNorm` | `(1.0 + weight)` | everything except the GDN output |
| `Qwen4ExpTextRMSNormGated` | `weight` | GatedDeltaNet output gate only |

`arch/qwen3_5/rmsnorm.rs` already implements both with these exact
conventions — this is a solved problem in our tree.

**What is new is `group_size`.** `Qwen4ExpTextRMSNorm(dim, group_size)`
reshapes the last axis to `(-1, group_size)`, normalises each group
independently, and flattens back. The 10240-wide norms in the decoder
(`hc_norm`, `norm_key`, `norm_query`, `norm_conv`) use
`group_size = 2560`, i.e. four independent RMS normalisations, not one
over 10240. A single norm over the full 10240 vector compiles, runs,
and is wrong.

**`mtp.pre_fc_norm_hidden` is the exception, and this document had it
wrong.** It is 10240 wide and **ungrouped** — one normalisation over
the whole vector. vLLM distinguishes the two cases explicitly: its
hyper-connection uses `GroupedGemmaRMSNorm` with
`group_size = hidden_size`, and the MTP head uses a plain
`GemmaRMSNorm(hidden_size * hc_count)`. Both classes exist in that
tree, so the choice is deliberate rather than an oversight. Grouping it
would rescale the draft head's input in four independent pieces and
show up as a poor acceptance rate, not as an error. See §9.

Variance is computed in f32 and the `(1 + w)` shift is applied in f32
before the multiply — same reasoning as the note already in
`rmsnorm.rs`.

### 9. MTP head

```
mtp.pre_fc_norm_embedding.weight (2560,)        # plain RMSNorm
mtp.pre_fc_norm_hidden.weight    (10240,)       # grouped, group_size 2560
mtp.fc_embedding.weight       (2560, 2560)
mtp.fc_hidden.weight          (2560, 2560)
mtp.hyper_connection_mixer.*                    # use_combine = False
mtp.layers.0.*                                  # a complete full_attention
                                                # decoder layer: 512-expert
                                                # MoE, shared expert, QSA
                                                # indexer, both hyper-
                                                # connections
```

`mtp_num_hidden_layers: 1`, `mtp.hybrid: true`, `mtp.layer_types:
["full_attention"]`, `mtp_use_dedicated_embeddings: false` (so it reuses
`embed_tokens` and `lm_head`).

**Transformers does not implement the MTP head.**
`Qwen4ExpPreTrainedModel._keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`
— the weights are dropped on load.

**The wiring, settled against vLLM** (`vllm/models/qwen4_exp/nvidia/
mtp.py`, read 2026-09-02). vLLM calls this shape
`residual_linear_shared`, as against the `Linear(2H, H)` + repeat that
other MTP variants use:

```
e = fc_embedding(pre_fc_norm_embedding(embed(t)))         # [T, 2560]
h = pre_fc_norm_hidden(h.flatten(-2))                      # ONE norm over 10240
h = fc_hidden(h.view(T, hc, 2560))                         # per stream, shared matrix
h = e.unsqueeze(-2) + h                                    # embedding broadcast to all four
h = h.flatten(-2)                                          # [T, 10240] -> the decoder layer
```

Three things here are not guessable from the tensor shapes:

1. **`pre_fc_norm_hidden` is applied to the flattened 10240 and is
   ungrouped** — see §8, which this document previously got wrong.
2. **The hidden state is the *pre-final-mixer* multi-stream**, i.e. `h`
   after the last decoder layer and *before* `hyper_connection_mixer`
   collapses it. Not the 2560 the LM head sees. vLLM calls this
   "scheme A"; on later draft steps the head consumes its own previous
   multi-stream rather than the target's.
3. **The head emits two things per step**: the collapsed `[T, 2560]`
   for `lm_head`, and the pre-final-mixer `[T, 10240]` for the next
   draft step.

PLE is forced off in the draft layer while the HC stream count is kept
(`ngram_context=None`), which is consistent with `mtp.layers.0` shipping
no `ple.*` tensors.

The accept rule is standard (greedy prefix match, or the Leviathan
rejection scheme) and is not architecture-specific; this document
previously conflated it with the wiring above.

`mtp.layers.0` carries its own full 512-expert MoE — 2.61 B params,
5.2 GB at BF16. It is not free to keep resident and its placement is
part of #310's problem.

### 10. Vision tower

```
depth 27   hidden_size 1152   intermediate_size 4304   num_heads 16
patch_size 16   spatial_merge_size 2   temporal_patch_size 2
out_hidden_size 2560   num_position_embeddings 2304
hidden_act gelu_pytorch_tanh   deepstack_visual_indexes []
```

448.9 M params. Smaller than the 27B's tower and with deepstack merging
disabled. Out of scope for the text path (#314); `language_model_only`
is a config knob.

## Reuse boundary against `arch/qwen3_5/`

| piece | verdict |
|---|---|
| `rope.rs` | **reuse unchanged.** Identical `head_dim`/`partial_rotary_factor`/`mrope_section`/`mrope_interleaved`. Verify the interleave index sets, expect no change. |
| `rmsnorm.rs` | **extend.** Both `(1+w)` and gated conventions already correct. Add `group_size`; make the gated variant's activation configurable (sigmoid here, SiLU in Qwen3.6). |
| `linear_attn.rs` | **reuse, two deltas.** Names and layout already match the split form. Switch the output-gate activation to sigmoid; default the recurrent state to f32 per `mamba_ssm_dtype`. |
| `full_attn.rs` | **extend.** Same GQA + q/k norm + widened-`q_proj` output gate. Add the QSA indexer and its separate cache. |
| `moe.rs` | **extend loader only.** Routing arithmetic identical; add a fused-3D `experts.{gate_up,down}_proj` branch beside the per-expert one. |
| `mlp.rs` | **reuse** for the shared expert. |
| `decoder.rs` | **rewrite.** Hyper-connections replace both layernorms and the residual is 4× wide. |
| `mod.rs` (config, load, forward) | **new sibling arch.** Config field vocabulary overlaps heavily; the model-level plumbing (10240 stream, no final norm, PLE at layer 1, MTP) does not. |
| `snapshot.rs` | **extend.** Batched decode needs snapshot support for the new per-request state: PLE's two conv slots and the QSA indexer cache, on top of the existing conv + recurrent state. |
| `vision.rs` | **likely reuse with a new config.** Deferred (#314). |

Net: the linear-attention half of the model — 36 of 48 layers and the
piece that took the most work in `qwen3_5` — carries over close to
unchanged. The genuinely new maths is hyper-connections (§1), PLE (§2)
and QSA (§3).

## Correctness gates, in the order they should be built

Each of these fails loudly, unlike the model as a whole, which fails
fluently.

1. **Grouped RMSNorm** against a hand-computed 4×2560 case.
2. **Hyper-connection block** in isolation: feed a known `h`, check
   `x`, `inj` and the reconstructed `h` against a torch trace. Do this
   before stacking anything on it — a wrong stream mix still produces
   plausible text.
3. **PLE hashing** against the reference for a short id sequence,
   including one that crosses an EOS, in int64.
4. **QSA below budget** — a prompt under 2048 tokens must give
   bit-comparable logits to dense causal attention through the same
   layer.
5. **QSA above budget** against the reference double loop on a small
   synthetic case (short `indexer_budget`, so the selection is actually
   exercised).
6. Full-model logits parity on a short prompt against a reference trace.

## Open questions this document does not settle

- **The accept rule and proposal depth for the MTP head** (§9). Needs
  llama.cpp #27836 or vLLM, not transformers.
- **Whether QSA can engage flash-attn at all.** The selection is a
  gather, our flavour builds are per-sm, and flash-attn is only wired on
  some (#95). Measure; do not assume.
- **Whether the f32 recurrent state (§5) costs anything measurable**
  here, given 36 linear layers rather than Qwen3.6's proportion.
- **How `kv_budget_mb` should account for the indexer cache** (§3) — it
  is a second, differently-shaped allocation and `derive_limit` does not
  know about it. Feeds #310 and #315.
