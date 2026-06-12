# Qwen3.6-27B vision specification (Stage A0)

Sourced from beast's local cache on 2026-06-01:
`/archive3/llm-cache/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/`.

Single source of truth for Stages A–D of the vision plan in
`~/.claude/plans/foamy-twirling-catmull.md`. Umbrella issue:
[#3](https://git.lair.cafe/helexa/helexa/issues/3).

---

## Top-level shape

The model is a unified text+vision architecture (`Qwen3_5ForConditionalGeneration`,
`model_type: qwen3_5`) with three weight sections under a single safetensors
index. Counts from `model.safetensors.index.json`:

| Prefix | Tensors | Role |
|---|---|---|
| `model.language_model.*` | 850 | LM (currently loaded) |
| `model.visual.*` | 333 | Vision tower (currently filtered out at `arch/qwen3_5/mod.rs:228-230`) |
| `mtp.*` | 15 | Multi-token-prediction heads (filtered, out of scope) |
| `lm_head.weight` | 1 | LM head |

Vision tensors live in shards `model-00007-of-00015.safetensors` and
`model-00008-of-00015.safetensors` (2 of the 15 safetensors). Loading just
these two for vision-tower-only smoke tests is feasible.

## Vision tower architecture (`model.visual.*`)

From `config.json::vision_config`:

```
depth:                       27   (transformer blocks)
hidden_size:               1152   (vision token dim)
num_heads:                   16   (per-block self-attention)
intermediate_size:         4304   (MLP hidden)
patch_size:                  16   (16×16 spatial patches)
temporal_patch_size:          2   (video frame pairing; irrelevant for stills)
spatial_merge_size:           2   (2×2 spatial merge in the merger → 4 patches/LM token)
num_position_embeddings:   2304   (learned pos embed slots — max patch sequence length)
in_channels:                  3   (RGB)
hidden_act:    gelu_pytorch_tanh  (GELU with tanh approximation, not exact GELU)
out_hidden_size:           5120   (= LM hidden_size, merger output dim)
deepstack_visual_indexes:    []   (no deep-stack visual indexes)
```

### Module inventory (per-block and global)

Global:
- `model.visual.patch_embed.proj.{weight, bias}` — Conv2d (3 → 1152, kernel 16×16, stride 16). Turns image patches into tokens.
- `model.visual.pos_embed.weight` — Learned position embedding, shape `(2304, 1152)`.
- `model.visual.merger.{norm, linear_fc1, linear_fc2}` — The projector that merges 2×2 patches and projects to LM hidden_size (1152 → 5120). All weights have biases.

Per block (×27, named `model.visual.blocks.{0..26}`):
- `norm1.{weight, bias}` — **LayerNorm** before attention (with bias — not RmsNorm).
- `attn.qkv.{weight, bias}` — Fused QKV linear (1152 → 3·1152 = 3456).
- `attn.proj.{weight, bias}` — Attention output projection (1152 → 1152).
- `norm2.{weight, bias}` — LayerNorm before MLP.
- `mlp.linear_fc1.{weight, bias}` — MLP up-projection (1152 → 4304).
- `mlp.linear_fc2.{weight, bias}` — MLP down-projection (4304 → 1152).

Pattern matches a standard ViT block with **pre-norm** layout (norm → attn → residual, norm → MLP → residual). Activation between fc1/fc2 is GELU-tanh-approx per `hidden_act`. No attention masking inside the vision tower (all patches attend to each other).

### Forward signature (target)

```
VisionTower::forward(
    patches: Tensor [N, in_channels, patch_size, patch_size],  # CPU-preprocessed RGB float patches
    grid_thw: Option<(usize, usize, usize)>,                   # (t, h, w) patch grid for position lookup
) -> Tensor [N / (spatial_merge_size²), out_hidden_size]      # = (N/4, 5120) for static images
```

Note: the merger consumes 4 spatially-adjacent patches and emits 1 LM token. So an image producing 64×64 = 4096 patches yields 1024 LM-side image tokens.

## Image preprocessor (`preprocessor_config.json`)

```json
{
    "size": { "longest_edge": 16777216, "shortest_edge": 65536 },
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std":  [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast"
}
```

Reading:

- `image_mean = image_std = 0.5` → normalisation is simply `(x/255 - 0.5) / 0.5 = 2*x/255 - 1`, mapping `[0,255]` → `[-1, 1]`. No imagenet-style mean/std.
- `size.{shortest_edge, longest_edge}` are **pixel counts**, not edge lengths. The `Qwen2VLImageProcessorFast` recipe picks a resolution within `[65,536 = 256², 16,777,216 = 4096²]` total pixels, snapping `h` and `w` to multiples of `patch_size × spatial_merge_size = 32` pixels.
- Stage A ships **fixed resolution**: pick a target pixel count (e.g. 448×448 = 200,704 px → 28×28 patches → 14×14 LM tokens after merger). Variable resolution deferred to issue [#14](https://git.lair.cafe/helexa/helexa/issues/14).

## Chat template (`chat_template.jinja`)

Image insertion (lines 8–18 of the template):

```jinja
{%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
    ...
    {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
```

Per image, the template emits **one `<|image_pad|>` token** flanked by `<|vision_start|>` and `<|vision_end|>` sentinels. The runtime must:

1. Render the template (preserving the single `<|image_pad|>` per image).
2. For each image, replace its single `<|image_pad|>` with N copies, where N is the number of LM tokens that image produces after the vision tower + merger (= `patches / spatial_merge_size²`).
3. Tokenize the expanded string → `input_ids`.
4. At forward time, locate positions where `input_ids == image_token_id` (248056) and splice in the vision tower's merger output.

Token IDs (top of `config.json`):
- `vision_start_token_id`: 248053
- `vision_end_token_id`:   248054
- `image_token_id`:        248056
- `video_token_id`:        248057 (out of scope)
- `bos_token_id`:          248044
- `eos_token_id`:          248044, 248046 (per `generation_config.json`)

System messages cannot contain images (template raises). Other template-side details:
- `add_vision_id` (jinja arg, default false): emits `'Picture N: '` prefixes when true.
- `preserve_thinking` (jinja arg, default false): keeps `<think>` blocks from prior assistant turns in the rendered prompt.
- `enable_thinking` (jinja arg, default true): emits `<think>\n` (or skips it) at the end of the generation prompt.

The existing chat-template renderer in `crates/neuron/src/harness/chat_template.rs` already passes `MessageContent::Parts` to the Jinja context as a `Value::Array`; the template's `is iterable` branch (line 6 of the template) handles them. **The path is structurally in place** — Stage B just needs to do the `<|image_pad|>` expansion + token-position-aware splice.

## LM-side considerations

The LM's RoPE config uses **multi-axis RoPE (MRoPE)**:

```
rope_parameters: {
    mrope_interleaved: true,
    mrope_section: [11, 11, 10],         # text + height + width components
    partial_rotary_factor: 0.25,
    rope_theta: 10000000,
    rope_type: "default"
}
```

MRoPE encodes spatial position alongside text position so the LM attention layers can reason about image-token spatial structure. The LM's existing forward path *may or may not* already implement this — the qwen3_5 module's doc-comment notes "numerical correctness vs the reference Python is not yet validated." Verifying MRoPE behaviour in the language model is out of Stage A scope (vision tower only) but will be required in Stage B (LM splice) and is tracked under the numerical-validation issue [#15](https://git.lair.cafe/helexa/helexa/issues/15).

`max_position_embeddings = 262144` (256 K context), so context-length limits are not a constraint for vision.

## Iteration target decision

The vision tower has its own self-contained weight tree and is small (~333 tensors in 2 shards, hidden_size 1152 vs LM's 5120). For Stage A specifically (vision-tower-only smoke), we **don't need a smaller iteration model** — we can:

- Build the Rust `VisionTower` struct against the spec above.
- Run unit tests with random tensor weights matching the exact shapes → assert forward produces correct output shape with finite values.
- Optionally: a CUDA-integration test that loads just the 2 vision shards from beast's cache (or on a smaller GPU like quadbrat's Ampere) and runs encode on a real image. Doesn't require loading the 27B LM at all.

This sidesteps the "develop against a smaller VL model" question for Stage A. Stage B (LM splice → end-to-end chat with vision) is where iteration speed becomes pressing; revisit there. The default scope pick 2a (smaller iteration model) is therefore deferred to Stage B planning — issue [#13](https://git.lair.cafe/helexa/helexa/issues/13) covers deployment validation regardless.

## Concrete Stage A1+ inputs

- Add deps to `crates/neuron/Cargo.toml`:
  - `image = "0.25"`
  - `base64 = "0.22"`
- Stage A2 preprocessor target resolution (fixed): **448×448 → 28×28 patches → 14×14 = 196 image tokens per image**. This balances minimum-patch-count for cheap tests against the model's expected input range.
- Stage A3 module structure: one `VisionTower` struct holding `patch_embed: Conv2d`, `pos_embed: Embedding`, `blocks: Vec<VisionBlock>`, `merger: Merger`. `VisionBlock` carries `norm1`, `norm2`, `attn`, `mlp`. Hand-roll using candle primitives.
- Stage A4 weight loading: extend `Qwen3_5ForCausalLM::new()` to construct `Some(VisionTower::new(vb.pp("model.visual"), config))` when `vision_config` is present in the parsed config.
- Stage A5 worker job: `Job::EncodeImage { handle, patches: Vec<f32>, patch_shape: (usize, usize, usize, usize, usize), reply: oneshot<Result<Vec<f32>>> }`. Patch shape = `(N, C, T, H, W)` where T=1 for static images.

## What this doc does NOT settle (deferred to issues)

- Numerical correctness of `VisionTower` output vs Python transformers
  → issue [#15](https://git.lair.cafe/helexa/helexa/issues/15).
- Variable image resolution
  → issue [#14](https://git.lair.cafe/helexa/helexa/issues/14).
- TP-vision (multi-rank vision tower)
  → issue [#12](https://git.lair.cafe/helexa/helexa/issues/12).
- 27B production deployment
  → issue [#13](https://git.lair.cafe/helexa/helexa/issues/13).
