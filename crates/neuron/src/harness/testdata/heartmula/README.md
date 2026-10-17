# HeartMuLa reference traces

Parity fixtures for neuron's HeartMuLa port (#391 LM, #392 codec, #393
placement), recorded by `script/heartmula-reference.py` from heartlib's own
model code:

- code: `HeartMuLa/heartlib` @ `5858ca8f1ffe58d62be7c007ea55f1033c5ca456`
- weights:
  - `HeartMuLa/HeartMuLa-oss-3B-happy-new-year` @ `41f6fc68490e11dc43fdabaa6b5767946408c903`
  - `HeartMuLa/HeartCodec-oss-20260123` @ `f889dab0532cfa4bf459f2a3367eb6d346b8eeda`
  - `HeartMuLa/HeartMuLaGen` (`tokenizer.json`, `gen_config.json`)
- torch 2.10.0+cu128, transformers 4.57.0, torchtune 0.4.0

The MuQ-MuLan reference-audio encoder (CC-BY-NC) is never loaded. heartlib
builds the pipeline with `muq_mulan=None` and conditions on a zero
embedding (#389).

Each `<case>__<trace>.safetensors` holds the tensors, and the matching
`.json` holds the metadata:

- the case inputs and the prompts as tokenized, with token ids
- the sampler settings
- per-tensor stats (shape, dtype, NaN/Inf counts, absmax, mean)
- frame counts and wall time
- for GPU traces, their agreement with `cpu_f32`

`index.json` summarises every trace.

## Cases

| case | content | tags | duration |
|---|---|---|---|
| `instr_12s` | instrumental: the section-tag template the beta and `/music` use (`[Intro] [Verse] [Chorus] [Bridge] [Chorus] [Outro]`, blank lines between) | lo-fi hip hop | 12 s |
| `vocal_en_12s` | English lyrics | pop, female vocals | 12 s |
| `vocal_es_12s` | Spanish lyrics | latin ballad, male vocals | 12 s |

A 12 s request is 150 frames plus heartlib's first frame, so 151 frames.
No case stopped early on the audio EOS token.

## Traces

| trace | device | dtype | what it records |
|---|---|---|---|
| `cpu_f32` | CPU | float32 | the parity reference: LM greedy + sampled, codec on the greedy codes (`instr_12s`, `vocal_en_12s`) |
| `cuda_bf16` | RTX 4090 | bfloat16 weights + autocast | what heartlib runs: LM greedy + sampled |

**`cuda_f32` was deferred.** The f32 LM does not fit a 24 GB card: it has
3.94 B parameters (15.75 GB), and moving it to the GPU ran out of memory at
22.65 GB allocated, before the KV cache existed. A port judges f32 parity
against `cpu_f32`.

The codec is float32 in every heartlib configuration (a bf16 codec
audibly degrades output), so only `cpu_f32` records it, from that trace's
greedy codes.

## Tensors

**Shared by all traces:**

- `prompt_tokens [prompt_len, 9]`: the matrix the LM consumes.
  - Column 8 holds the text: tag ids, then a zero (the MuQ slot, at index
    `muq_idx`), then lyric ids.
  - Columns 0–7 are the audio codebooks, which are all zero in the prompt.
- `prompt_tokens_mask [prompt_len, 9]`: only the text column is live in the
  prompt.

**Per sampler** (`greedy`: cfg 1.5, top-k 1; `sampled`: cfg 1.5,
temperature 1.0, top-k 50):

- `lm_<sampler>_frames [8, frames]`: the generated codes, codebook-major.
- `lm_<sampler>_logits [2, 8, 8197]`: the logits each code was sampled
  from, after CFG and before temperature or top-k, for frames 0–1 and all
  8 codebooks.
- `lm_<sampler>_backbone_last_h [2, 2, 3072]`: the backbone's last hidden
  state at frames 0–1, conditional row then unconditional row.

**Codec, `cpu_f32` only:**

- `codec_codes [8, frames]`: the codes the codec was given.
- `codec_rvq_quantized [1, 372, 512]`: the RVQ sum of the codes, after
  they are tiled to one 29.76 s segment.
- `codec_rvq_cond`: the same after `cond_feature_emb`, before 2×
  upsampling.
- `codec_flow_mu [1, 744, 512]`: the flow-matching condition, after
  nearest 2× upsampling to 25 Hz.
- `codec_flow_noise [1, 744, 256]`: the initial latent noise. Inject this
  for parity.
- `codec_flow_t_span [11]`: the timesteps.
- `codec_flow_step1_v`: the CFG-combined velocity at step 1.
- `codec_flow_step5_x`: the flow state after step 5.
- `codec_flow_final`: the final latents.
- `codec_waveform [2, 144000]`: the first 3 s of the decoded 48 kHz
  stereo output. The full output is `frames / 12.5 × 48000` samples.

## Findings

- **No NaN or Inf anywhere**, in any trace. bf16 is numerically sound for
  the LM:
  - At frame 0, before any sampled token can differ, the bf16 codebook-0
    logits are within 0.44–0.70% relative L2 of f32, and the backbone's
    last hidden state within 0.86–1.0%.
  - The frame-0 codebook-0 argmax agrees in every case.
- **A free-running stream diverges within a few frames.** This is
  autoregressive amplification of near-ties, not a defect. Greedy bf16
  matches f32 token for token in frame 0 of both vocal cases (all eight
  codebooks), but differs from frame 2 (`vocal_es`) or frame 5
  (`vocal_en`). In `instr_12s` it differs from codebook 2 of frame 0.
  **So a port's LM parity test must be teacher-forced:** feed the recorded
  `cpu_f32` frames and compare the logits. Free-running greedy streams are
  only a smoke test.
- **Sampled traces are not comparable across devices.** `sample_topk`
  draws `torch.empty_like(probs).exponential_(1)` on the logits' device,
  and the CPU and CUDA generators produce different streams from the same
  seed. Sampled streams are reproducible per device. The RNG consumption
  order is below.
- **Run-to-run determinism.** `instr_12s` was recorded twice on each
  device:
  - `cuda_bf16` was bit-identical.
  - `cpu_f32` produced identical token streams (greedy and sampled). Its
    float tensors differed by ≤ 8e-7 relative L2, because the two runs used
    different thread counts (28 and 24) and so a different reduction order.
  - Judge a CPU port within about 1e-6, not bit-exactly.

## How heartlib generates

**Prompt.**

- Tags are lowercased and wrapped as `<tag>…</tag>`. Lyrics are lowercased.
- Both are tokenized with `tokenizer.json`, the Llama-3.2 tokenizer, and
  each is given BOS 128000 and EOS 128001 if missing.
- The prompt is `tags ‖ [MuQ slot] ‖ lyrics` along time.

**Embedding.**

- At every position, the input is the **sum** over the 9 streams of each
  stream's embedding, masked by `tokens_mask`:
  - text: `text_embeddings` [128256, 3072]
  - audio codebook *k*: `audio_embeddings[token + k × 8197]`, from one
    shared [65576, 3072] table
- With CFG, batch row 1 is the unconditional copy. Its **text** embedding
  is replaced by `unconditional_text_embedding` at every position. Audio
  embeddings are not replaced.
- The MuQ slot is overwritten with `muq_linear(zeros(512))`, which is that
  layer's bias. In the unconditional row it is overwritten with the
  unconditional embedding.

**Backbone.**

- torchtune `llama3_2`: 28 layers, 24 heads, 8 KV heads, dim 3072, MLP
  8192, RMSNorm eps 1e-5.
- RoPE base 500000 with Llama-3 scaling (`scale_factor` 32).
- Its token embedding and output head are replaced by identity.
- KV cache with `max_seq_len` 8192 at batch 2.

**Codebook 0.**

- `codebook0_head(last_h)` gives [8197] logits.
- CFG: `uncond + 1.5 × (cond − uncond)`.
- Then `sample_topk`: divide by temperature, mask everything below the
  top-k-th logit to −inf, then log-softmax and softmax.
- The draw is `argmax(probs / Exp(1))`. That is a Gumbel-style draw, so
  top-k 1 is exact greedy.
- The sampled token is repeated to both rows.

**Codebooks 1–7.**

- The local decoder is torchtune `llama3_2`: 3 layers, 8 heads, 4 KV
  heads, dim 3072, MLP 8192, KV cache length 8.
- It is reset every frame.
- Its input sequence is `[last_h, embed(c0)]`, projected by `projection`.
  At each later step the input is the previous codebook's embedding.
- Codebook *i*'s logits are `decoder_h[-1] @ audio_head[i−1]`, with the
  same CFG and sampler as codebook 0.

**Next frame.**

- The 8 codes are fed back with an empty text column (masked out).
- Position is `prompt_len + i`.
- Generation stops when any code of row 0 reaches ≥ 8193, the audio EOS.

**RNG consumption order:**

- Per frame, codebooks 0–7, one `exponential_` draw of shape [1, 8197]
  each, on the LM's device.
- A greedy run consumes the same draws.

**Codec** (`HeartCodec.detokenize`, f32; defaults: segment 29.76 s,
10 Euler steps, guidance 1.25):

1. Tile the codes until they cover a whole segment (372 frames at 12.5 Hz).
2. RVQ lookup: 8 quantizers, 8192 codes, codebook dim 32, output 512.
3. `cond_feature_emb` (Linear 512→512), then nearest 2× upsampling, giving
   744 frames at 25 Hz.
4. A first latent of `randn(1, 744, 256)` is drawn on the CPU generator.
   In the first segment it is multiplied by a zero mask, so it changes no
   value, but it does consume RNG.
5. The flow noise is a second `randn(1, 744, 256)` on the codec's device.
6. Euler from t = 0 to 1 in 10 steps.
   - Estimator input: `cat([x, incontext_x (zeros), mu], -1)` = 1024
     channels.
   - CFG pairs a zero-`mu` row with the real one:
     `v = v_u + 1.25 × (v_c − v_u)`.
7. The estimator is a LlamaTransformer:
   - `proj_in`: Conv1d k3, 1024→1536.
   - 24 blocks at dim 1536 (24 heads × 64), RoPE on the full head, with
     AdaLN-single timestep conditioning.
   - `norm_out` with a scale/shift table.
   - `connection_proj`: Conv1d k3 over `[input, s]`, 2560→3072.
   - 6 blocks at dim 3072 (24 heads × 128).
   - `proj_out`: Conv1d k3, 3072→256.
8. Each 256-dim latent is split into 2 halves of 128 dims, one per stereo
   channel.
9. ScalarModel decode:
   - A scalar quantizer: `round(9x) / 9`.
   - A Snake-activation conv decoder with transposed-conv upsampling
     5·4·4·4·3 = 960×, then a PostProcessor that repeats every sample
     `num_samples` = 2 times before a conv and PReLU, so 25 Hz becomes
     48 kHz.
10. Segments overlap-add with a linear crossfade (one segment for songs up
    to 29.76 s), and the output is trimmed to `frames / 12.5 × 48000`
    samples.

**Rates:** the LM runs at 12.5 Hz (80 ms per frame), the codec latents at
25 Hz, and the output is 48 kHz stereo.

## Regenerating

On a host with heartlib's venv and the checkpoints laid out as
`<ckpt>/{HeartMuLa-oss-3B,HeartCodec-oss,tokenizer.json,gen_config.json}`:

    python script/heartmula-reference.py --ckpt <ckpt> --trace cpu_f32 --out <dir>
    python script/heartmula-reference.py --ckpt <ckpt> --trace cuda_bf16 --out <dir>
    python script/heartmula-reference.py --compare --out <dir>

The CPU trace takes about 15 minutes on 28 cores, after the weights are in
the page cache. Pre-read the safetensors sequentially: memory-mapped page
faults read at about 10 MB/s.
