# ACE-Step 1.5 reference traces

Parity fixtures for neuron's ACE-Step port (#383, #384, #385), recorded by
`script/acestep-reference.py` from the upstream PyTorch pipeline:

- code: `ace-step/ACE-Step-1.5` @ `ca1e85fe9430179831e6bc6be790c332190a3866`
- weights: `ACE-Step/Ace-Step1.5` @ `19671f406d603126926c1b7e2adc169acbcade22`
  (`acestep-v15-turbo`, `acestep-5Hz-lm-1.7B`, `Qwen3-Embedding-0.6B`, `vae`)
- torch 2.10.0+cu128, transformers 4.57.6, pytorch_wavelets 1.3.0

Each `<case>__<trace>.safetensors` holds the tensors, and the matching
`.json` holds the metadata: the case inputs, decoded prompts, the sampler
settings `generate_audio` actually received, the timestep schedule, planner
prompts and outputs, per-tensor stats (shape, dtype, NaN/Inf counts, absmax,
mean, std), the first tensor to contain a NaN, drift against `cpu_f32`, wall
time and peak VRAM. `index.json` summarises every trace.

## Cases

| case | content | duration | planner |
|---|---|---|---|
| `instr_10s_noplan` | instrumental | 10 s | off |
| `vocal_en_30s_noplan` | English vocals | 30 s | off |
| `vocal_en_30s_plan` | English vocals | 30 s | on (1.7B, greedy) |
| `vocal_es_20s_noplan` | Spanish vocals | 20 s | off |
| `vocal_en_120s_noplan` | English vocals | 120 s | off |

"Planner off" turns off `thinking` and all three `use_cot_*` flags, so the
LM is never called. "Planner on" runs the LM greedily (`lm_temperature=0`),
with upstream's default CFG (2.0 for the code phase) and constrained
decoding.

## Traces

| trace | device | dtype | what it is |
|---|---|---|---|
| `cpu_f32` | CPU | float32 | the parity reference: deterministic and complete |
| `cuda_bf16` | RTX 4090 | bfloat16 | what upstream runs on Ampere and later |
| `cuda_f32` | RTX 4090 | float32 | true f32 on the GPU, TF32 off: the Pascal compute recipe |
| `cuda_f16` | RTX 4090 | float16 | upstream's own pre-Ampere path (fp16 + eager attention), forced |

All traces of a case start from the **same noise**: `cpu_f32` records it,
and the GPU traces load it instead of drawing their own. CUDA and CPU
`torch.Generator`s differ at the same seed. A port should likewise take
`noise` as an input when checking parity.

`cuda_f32` is loaded on the CPU and then moved to the GPU. Loading on CUDA
and casting to f32 afterwards is not equivalent. A CUDA load rounds every
RoPE `inv_freq` buffer and the float32 `silence_latent` to bf16, and a later
cast cannot restore them. That alone moved a single DiT forward by 11%
(rel L2) from the CPU result.

## Results

| case | `cuda_f32` vs CPU | `cuda_bf16` vs CPU | `cuda_f16` |
|---|---|---|---|
| `instr_10s_noplan` | 6e-6 | 0.08 | clean (0.018) |
| `vocal_es_20s_noplan` | 1e-5 | 0.17 | **NaN** |
| `vocal_en_30s_noplan` | 8e-4 | 0.41 | **NaN** |
| `vocal_en_120s_noplan` | 2e-4 | 0.63 | **NaN** |
| `vocal_en_30s_plan` | — | 1.01 (different plan) | — |

Values are the relative L2 error of `target_latents` against `cpu_f32`.

- **fp16 vocals NaN, instrumentals don't.** This is upstream issue #1055,
  reproduced on Ada. The first non-finite value is in the lyric encoder's
  last layer: `layers.7.mlp.down_proj` overflows float16 (the residual
  stream is already ~3.6k at layer 6). Keeping RoPE in f32 does not help.
  "[Instrumental]" is a short lyric that stays in range. Upstream raises
  rather than decoding; the trace records everything up to that point.
- **True f32 on the GPU matches the CPU** to about 1e-5 per forward, and
  1e-5 to 8e-4 end to end. The f32 recipe is numerically sound.
- **Upstream's bf16 path is far from f32**: 8% to 63% end to end, growing
  with length. Part of that is the bf16-rounded `inv_freq` and silence
  latent described above, not only bf16 arithmetic. A candle port that
  computes RoPE in f32 should be judged against `cpu_f32`/`cuda_f32`.
  `cuda_bf16` is upstream's behaviour, not a tolerance target.
- **The planner's code phase is fragile under precision change.** The CoT
  phase (164 tokens) is identical between `cpu_f32` and `cuda_bf16`, but
  the very first audio code differs, so the whole plan differs.

Peak VRAM on the 4090 (DiT, text encoder and VAE; plus the 1.7B LM where
loaded): bf16 with the LM loaded, 10.2 GiB; f32 without the LM, 12.6 GiB;
fp16 without the LM, 6.2–6.7 GiB.

## Tensors

`cpu_f32` and `cuda_f32` store float32; `cuda_bf16` and `cuda_f16` store
their own dtype, which is exact and half the size. Shapes are for batch 1,
and `T` is the number of 25 Hz latent frames.

| name | shape | what |
|---|---|---|
| `text_token_ids` | `[1, Lt]` | DiT text prompt, tokenized by the Qwen3-Embedding tokenizer (prompt text in the JSON) |
| `text_hidden_states` | `[1, Lt, 1024]` | Qwen3-Embedding-0.6B `last_hidden_state` |
| `lyric_token_ids` | `[1, Ll]` | lyric prompt (`# Languages\n<lang>\n\n# Lyric\n<lyrics><\|endoftext\|>`) |
| `lyric_embeds` | `[1, Ll, 1024]` | `embed_tokens` lookup only, with no transformer pass |
| `encoder_hidden_states` | `[1, Ll+1+Lt, 2048]` | packed condition: lyric-encoder output (8 layers), then timbre (1 token), then projected text |
| `encoder_attention_mask` | `[1, Ll+1+Lt]` | packed mask |
| `gen_chunk_masks`, `gen_is_covers` | | `generate_audio` inputs |
| `gen_precomputed_lm_hints_25Hz` | `[1, T, 64]` | planner cases: LM audio codes, detokenized to 25 Hz |
| `context_latents` | `[1, T, 128]` | src latents (silence, or LM hints) concatenated with chunk masks |
| `decoder_attention_mask` | `[1, T]` | |
| `noise` | `[1, T, 64]` | initial `xt` (CPU traces only; GPU traces reuse it) |
| `step{i}_xt`, `step{i}_vt` | `[1, T, 64]` | DiT input and predicted velocity at sampler step `i` |
| `target_latents` | `[1, T, 64]` | sampler output, after the final step's `x0` and DCW |
| `vae_input` | `[1, 64, T]` | what the VAE decoder received |
| `vae_waveform` | `[1, 2, 24000]` | VAE output, first 0.5 s (48 kHz stereo), before normalisation. The pipeline output is this × `normalisation_gain` (JSON) |
| `planner{n}_input_ids`, `planner{n}_output_ids` | | LM phase `n` (0 = CoT, 1 = codes): prompt ids (with CFG, conditional and unconditional rows, left-padded) and generated ids |
| `planner_audio_codes` | `[N]` | the 5 Hz audio codes the planner produced (N = 5 × seconds) |

The lyric encoder's own output is not stored. With an all-true mask it is
exactly the first `Ll` rows of `encoder_hidden_states`, and the JSON
records that this was checked (`lyric_encoder_out_is_encoder_prefix`).

Steps saved (`steps_saved` in the JSON):

- **CPU trace:** every step for the 10 s case; steps 0, 4 and 7 for the
  20–30 s cases; step 0 for the 120 s case.
- **GPU traces:** `step0_vt`, `step7_vt` and `target_latents`. The 120 s
  case keeps only `target_latents`. A step where a NaN first appears is
  always kept.
- `step0_xt` is never stored, because it is `noise`.
- The 120 s case omits `context_latents` (silence plus the chunk mask) and
  `vae_input` (`target_latents` transposed).

Every tensor that was dropped still has its stats in the JSON.

## Regenerating

In the ACE-Step venv, with the checkpoints already on disk (fetched through
your registry; the script refuses to download). Record `cpu_f32` first,
because the GPU traces load its noise:

```sh
python -I script/acestep-reference.py --src <ACE-Step-1.5 checkout> --trace cpu_f32 --out <this dir>
python -I script/acestep-reference.py --src ... --trace cuda_bf16 --out ...
python -I script/acestep-reference.py --src ... --trace cuda_f32 --out ...
python -I script/acestep-reference.py --src ... --trace cuda_f16 --out ...
```
