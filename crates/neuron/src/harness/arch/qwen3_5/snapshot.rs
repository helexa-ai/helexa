//! Cache-state snapshots for prefix KV caching (#11).
//!
//! A snapshot captures everything `clear_kv_cache` would destroy, at
//! one consistent token boundary:
//!
//! - full-attention layers: the `ConcatKvCache` k/v tensors,
//! - linear-attention layers: the GatedDeltaNet `conv_state` +
//!   `recurrent_state`,
//! - the model-level `rope_delta` position counter.
//!
//! The GatedDeltaNet recurrent state cannot be rewound to an earlier
//! token, so a snapshot is only reusable when its entire token
//! sequence is an exact prefix of an incoming prompt — matching policy
//! lives in `harness/prefix_cache.rs`; this module is just the state
//! capture.
//!
//! ## Copy semantics
//!
//! Attention k/v snapshots share storage with the live cache:
//! `ConcatKvCache::append` never mutates stored tensors in place (it
//! `cat`s into fresh allocations), so a shallow `Tensor` clone stays
//! valid after the live cache moves on. The GDN states are
//! **deep-copied** in both directions (`Tensor::copy`): the CUDA
//! delta-rule kernels update the recurrent-state buffer in place, and
//! `flatten`/`contiguous` on an already-contiguous tensor is a view —
//! a shared-storage snapshot would be corrupted by the next forward.

use candle_core::Tensor;

/// Per-layer captured state. Variant kind must match the layer's
/// `AttentionKind` on restore.
pub enum LayerKvSnapshot {
    /// `ConcatKvCache` contents. `None` when the cache was empty
    /// (a zero-token snapshot — valid but useless; the registry never
    /// stores one).
    Full(Option<(Tensor, Tensor)>),
    /// GatedDeltaNet state. Either tensor is `None` before the first
    /// forward touches it.
    Linear {
        conv_state: Option<Tensor>,
        recurrent_state: Option<Tensor>,
    },
}

/// One consistent cache snapshot of a `Qwen3_5Model` (or its TP
/// mirror `tp_qwen3_5::TpQwen3_5Model`, whose per-rank shard state
/// has the same shape) at a token boundary. Fields are `pub(crate)`
/// so the TP module can construct/consume the same type; holders
/// outside the harness only ever pass it back to `restore_kv_cache`.
pub struct KvCacheSnapshot {
    pub(crate) layers: Vec<LayerKvSnapshot>,
    pub(crate) rope_delta: i64,
}

impl KvCacheSnapshot {
    /// Number of layer snapshots held (test/diagnostic helper).
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Total bytes of tensor data held by this snapshot. Used for the
    /// prefix-cache VRAM budget. Attention k/v shares storage with the
    /// live cache at capture time, but the live cache is cleared or
    /// replaced before the next request, so counting the full size is
    /// the honest steady-state figure.
    pub fn size_bytes(&self) -> u64 {
        fn t_bytes(t: &Tensor) -> u64 {
            (t.elem_count() * t.dtype().size_in_bytes()) as u64
        }
        self.layers
            .iter()
            .map(|l| match l {
                LayerKvSnapshot::Full(Some((k, v))) => t_bytes(k) + t_bytes(v),
                LayerKvSnapshot::Full(None) => 0,
                LayerKvSnapshot::Linear {
                    conv_state,
                    recurrent_state,
                } => {
                    conv_state.as_ref().map(t_bytes).unwrap_or(0)
                        + recurrent_state.as_ref().map(t_bytes).unwrap_or(0)
                }
            })
            .sum()
    }
}

/// Batched cache state assembled from per-sequence snapshots (#98).
/// Install with `Qwen3_5Model::restore_kv_cache(&self.snapshot)`; the
/// forward then runs lockstep batched decode with
/// `forward_batch_decode` using `prefix_lens[i] + step` positions and
/// the padding mask from `Qwen3_5Model::batch_decode_mask`.
pub struct BatchedKvState {
    /// Per-layer `(B, …)` state: attention K/V right-padded along the
    /// sequence axis to `padded_len` and `cat`ed on dim 0; GDN
    /// conv/recurrent states `cat`ed on dim 0 (position-free).
    pub snapshot: KvCacheSnapshot,
    /// The uniform KV sequence length every row was padded to — the
    /// max prefix length in the batch. Decode appends start here.
    pub padded_len: usize,
    /// Each row's true prefix length. Columns `[prefix_lens[i],
    /// padded_len)` of row `i` are zero-padding and must stay masked.
    pub prefix_lens: Vec<usize>,
}

/// Assemble per-sequence snapshots into one batched cache state.
/// `seqs` pairs each snapshot with its true token length (the caller
/// tracks prompt token counts; attention K/V lengths are validated
/// against it). All snapshots must come from single-sequence (`B=1`)
/// prefills of the same model, with `rope_delta == 0` (text-only —
/// vision requests don't batch, #98 v1).
///
/// Keys are stored post-RoPE, so right-padding does not disturb
/// position correctness: a row's cached keys keep the rotation of
/// their true positions, the garbage columns are masked, and new
/// tokens rotate at `prefix_len + step` while landing at storage
/// column `padded_len + step`.
pub fn assemble_batch(seqs: &[(&KvCacheSnapshot, usize)]) -> candle_core::Result<BatchedKvState> {
    let Some((first, _)) = seqs.first() else {
        candle_core::bail!("assemble_batch: empty batch");
    };
    let n_layers = first.layers.len();
    let prefix_lens: Vec<usize> = seqs.iter().map(|&(_, len)| len).collect();
    let padded_len = *prefix_lens.iter().max().expect("non-empty");
    for (snap, len) in seqs {
        if snap.layers.len() != n_layers {
            candle_core::bail!(
                "assemble_batch: snapshot layer count mismatch ({} vs {n_layers})",
                snap.layers.len()
            );
        }
        if snap.rope_delta != 0 {
            candle_core::bail!(
                "assemble_batch: rope_delta {} != 0 — vision-positioned sequences cannot batch",
                snap.rope_delta
            );
        }
        if *len == 0 {
            candle_core::bail!("assemble_batch: zero-length sequence");
        }
    }

    let mut layers = Vec::with_capacity(n_layers);
    for li in 0..n_layers {
        layers.push(assemble_layer(seqs, li, padded_len)?);
    }
    Ok(BatchedKvState {
        snapshot: KvCacheSnapshot {
            layers,
            rope_delta: 0,
        },
        padded_len,
        prefix_lens,
    })
}

fn assemble_layer(
    seqs: &[(&KvCacheSnapshot, usize)],
    li: usize,
    padded_len: usize,
) -> candle_core::Result<LayerKvSnapshot> {
    match &seqs[0].0.layers[li] {
        LayerKvSnapshot::Full(_) => {
            let mut ks = Vec::with_capacity(seqs.len());
            let mut vs = Vec::with_capacity(seqs.len());
            for (row, (snap, len)) in seqs.iter().enumerate() {
                let LayerKvSnapshot::Full(Some((k, v))) = &snap.layers[li] else {
                    candle_core::bail!(
                        "assemble_batch: row {row} layer {li} is not a populated \
                         full-attention snapshot"
                    );
                };
                let (b, _h, s, _d) = k.dims4()?;
                if b != 1 {
                    candle_core::bail!(
                        "assemble_batch: row {row} layer {li} has batch dim {b}, want 1"
                    );
                }
                if s != *len {
                    candle_core::bail!(
                        "assemble_batch: row {row} layer {li} KV length {s} != declared \
                         sequence length {len}"
                    );
                }
                ks.push(pad_seq(k, padded_len)?);
                vs.push(pad_seq(v, padded_len)?);
            }
            let k = Tensor::cat(&ks, 0)?;
            let v = Tensor::cat(&vs, 0)?;
            Ok(LayerKvSnapshot::Full(Some((k, v))))
        }
        LayerKvSnapshot::Linear { .. } => {
            let mut convs = Vec::with_capacity(seqs.len());
            let mut recs = Vec::with_capacity(seqs.len());
            for (row, (snap, _)) in seqs.iter().enumerate() {
                let LayerKvSnapshot::Linear {
                    conv_state: Some(conv),
                    recurrent_state: Some(rec),
                } = &snap.layers[li]
                else {
                    candle_core::bail!(
                        "assemble_batch: row {row} layer {li} is not a populated \
                         linear-attention snapshot"
                    );
                };
                convs.push(conv.clone());
                recs.push(rec.clone());
            }
            Ok(LayerKvSnapshot::Linear {
                conv_state: Some(Tensor::cat(&convs, 0)?),
                recurrent_state: Some(Tensor::cat(&recs, 0)?),
            })
        }
    }
}

/// Right-pad a `(1, H, S, D)` K or V tensor with zeros along the
/// sequence axis to `padded_len`. Zero columns are inert: the padding
/// mask keeps every query from attending to them.
fn pad_seq(t: &Tensor, padded_len: usize) -> candle_core::Result<Tensor> {
    let (b, h, s, d) = t.dims4()?;
    if s == padded_len {
        return Ok(t.clone());
    }
    let pad = Tensor::zeros((b, h, padded_len - s, d), t.dtype(), t.device())?;
    Tensor::cat(&[t, &pad], 2)
}

#[cfg(test)]
mod tests {
    use super::super::{Qwen3_5Model, RopeParameters, TextConfig};
    use candle_core::{DType, Device, Tensor};
    use std::collections::HashMap;

    /// Tiny two-layer config covering both attention kinds.
    fn tiny_config() -> TextConfig {
        TextConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            max_position_embeddings: 64,
            rope_parameters: RopeParameters {
                rope_theta: 10000.0,
                partial_rotary_factor: 0.5,
                rope_type: None,
                mrope_section: Vec::new(),
                mrope_interleaved: false,
            },
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            attn_output_gate: true,
            layer_types: vec!["linear_attention".into(), "full_attention".into()],
            full_attention_interval: Some(4),
            hidden_act: "silu".into(),
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            decoder_sparse_step: 1,
            mlp_only_layers: Vec::new(),
            norm_topk_prob: false,
        }
    }

    /// Build a Qwen3_5Model from random weights written to a temp
    /// safetensors file — the same `ShardedVarBuilder` path the real
    /// loader uses.
    fn tiny_model(cfg: &TextConfig) -> Qwen3_5Model {
        let dev = Device::Cpu;
        let randn = |shape: &[usize]| Tensor::randn(0f32, 0.2f32, shape, &dev).unwrap();

        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads;
        let value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let nv = cfg.linear_num_value_heads;
        let hd = cfg.head_dim;
        let q_out = cfg.num_attention_heads * hd * 2;
        let kv_out = cfg.num_key_value_heads * hd;

        let mut t: HashMap<String, Tensor> = HashMap::new();
        let p = "model.language_model";
        t.insert(
            format!("{p}.embed_tokens.weight"),
            randn(&[cfg.vocab_size, h]),
        );
        t.insert(format!("{p}.norm.weight"), randn(&[h]));
        for (i, kind) in cfg.layer_types.iter().enumerate() {
            let lp = format!("{p}.layers.{i}");
            t.insert(format!("{lp}.input_layernorm.weight"), randn(&[h]));
            t.insert(format!("{lp}.post_attention_layernorm.weight"), randn(&[h]));
            t.insert(format!("{lp}.mlp.gate_proj.weight"), randn(&[inter, h]));
            t.insert(format!("{lp}.mlp.up_proj.weight"), randn(&[inter, h]));
            t.insert(format!("{lp}.mlp.down_proj.weight"), randn(&[h, inter]));
            match kind.as_str() {
                "linear_attention" => {
                    let ap = format!("{lp}.linear_attn");
                    t.insert(format!("{ap}.in_proj_qkv.weight"), randn(&[conv_dim, h]));
                    t.insert(format!("{ap}.in_proj_z.weight"), randn(&[value_dim, h]));
                    t.insert(format!("{ap}.in_proj_b.weight"), randn(&[nv, h]));
                    t.insert(format!("{ap}.in_proj_a.weight"), randn(&[nv, h]));
                    t.insert(format!("{ap}.out_proj.weight"), randn(&[h, value_dim]));
                    t.insert(
                        format!("{ap}.conv1d.weight"),
                        randn(&[conv_dim, 1, cfg.linear_conv_kernel_dim]),
                    );
                    t.insert(format!("{ap}.dt_bias"), randn(&[nv]));
                    t.insert(format!("{ap}.A_log"), randn(&[nv]));
                    t.insert(
                        format!("{ap}.norm.weight"),
                        randn(&[cfg.linear_value_head_dim]),
                    );
                }
                "full_attention" => {
                    let ap = format!("{lp}.self_attn");
                    t.insert(format!("{ap}.q_proj.weight"), randn(&[q_out, h]));
                    t.insert(format!("{ap}.k_proj.weight"), randn(&[kv_out, h]));
                    t.insert(format!("{ap}.v_proj.weight"), randn(&[kv_out, h]));
                    t.insert(
                        format!("{ap}.o_proj.weight"),
                        randn(&[h, cfg.num_attention_heads * hd]),
                    );
                    t.insert(format!("{ap}.q_norm.weight"), randn(&[hd]));
                    t.insert(format!("{ap}.k_norm.weight"), randn(&[hd]));
                }
                other => panic!("unexpected layer type {other}"),
            }
        }

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("model.safetensors");
        candle_core::safetensors::save(&t, &path).expect("save safetensors");
        // SAFETY: mmap of a file this test just wrote and nothing else
        // mutates — same justification as the real loader.
        let vb = unsafe {
            candle_nn::var_builder::ShardedSafeTensors::var_builder(
                std::slice::from_ref(&path),
                DType::F32,
                &dev,
            )
            .expect("build ShardedVarBuilder")
        };
        Qwen3_5Model::load(cfg, &vb, "model.language_model").expect("load tiny qwen3_5 model")
    }

    fn forward_tokens(model: &mut Qwen3_5Model, tokens: &[u32], offset: usize) -> Vec<f32> {
        let input = Tensor::new(tokens, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let hidden = model.forward(&input, offset).unwrap();
        // Last-position hidden row — what the lm_head would consume.
        let (_, l, _) = hidden.dims3().unwrap();
        hidden
            .narrow(1, l - 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    /// The gold test for #11: prefill a prefix, snapshot, perturb the
    /// live state with unrelated tokens, restore, prefill only the
    /// suffix — the result must match a fresh full prefill. Exercises
    /// attention KV, GDN conv/recurrent state, and offset bookkeeping
    /// in one pass; the perturbation step would corrupt a
    /// shared-storage (non-deep-copied) GDN snapshot.
    #[test]
    fn restore_then_suffix_matches_full_prefill() {
        let cfg = tiny_config();
        let mut model = tiny_model(&cfg);

        let prefix: &[u32] = &[1, 2, 3];
        let suffix: &[u32] = &[4, 5, 6];
        let full: Vec<u32> = prefix.iter().chain(suffix).copied().collect();

        model.clear_kv_cache();
        let h_full = forward_tokens(&mut model, &full, 0);

        model.clear_kv_cache();
        forward_tokens(&mut model, prefix, 0);
        let snap = model.snapshot_kv_cache().expect("snapshot");
        assert_eq!(snap.layer_count(), 2);
        assert!(snap.size_bytes() > 0);

        // Advance the live state past the snapshot boundary — a
        // different continuation, as a subsequent request would be.
        forward_tokens(&mut model, &[9, 8], prefix.len());

        model.restore_kv_cache(&snap).expect("restore");
        let h_restored = forward_tokens(&mut model, suffix, prefix.len());
        let diff = max_abs_diff(&h_full, &h_restored);
        assert!(diff < 1e-4, "restored-prefix forward diverged: {diff}");

        // The snapshot must survive restore + forward cycles (deep
        // copy of the in-place-mutated GDN state): restore again and
        // expect the identical result.
        model.restore_kv_cache(&snap).expect("second restore");
        let h_again = forward_tokens(&mut model, suffix, prefix.len());
        let diff = max_abs_diff(&h_restored, &h_again);
        assert!(diff < 1e-6, "second restore diverged: {diff}");
    }

    /// The gold test for #98 slice 1: ragged sequences decoded in one
    /// lockstep batch (assembled from per-sequence snapshots, per-row
    /// positions, padding mask) must match the same sequences decoded
    /// sequentially, hidden-state for hidden-state at every step.
    #[test]
    fn batched_decode_matches_sequential() {
        use candle_core::IndexOp;
        let cfg = tiny_config();
        let mut model = tiny_model(&cfg);

        let prompts: [&[u32]; 3] = [&[1, 2, 3], &[4, 5], &[7, 7, 2, 5, 6]];
        let steps: [&[u32]; 3] = [&[11, 12, 13, 14], &[9, 8, 7, 6], &[21, 22, 23, 24]];
        let n_steps = 4;

        // Sequential reference: each sequence decoded alone.
        let mut expected: Vec<Vec<Vec<f32>>> = Vec::new(); // [row][step]
        for (prompt, toks) in prompts.iter().zip(steps.iter()) {
            model.clear_kv_cache();
            forward_tokens(&mut model, prompt, 0);
            let mut per_step = Vec::new();
            for (t, tok) in toks.iter().enumerate() {
                per_step.push(forward_tokens(&mut model, &[*tok], prompt.len() + t));
            }
            expected.push(per_step);
        }

        // Batched: prefill each sequence alone, snapshot, assemble.
        let mut snaps = Vec::new();
        for prompt in prompts.iter() {
            model.clear_kv_cache();
            forward_tokens(&mut model, prompt, 0);
            snaps.push(model.snapshot_kv_cache().expect("snapshot"));
        }
        let seqs: Vec<(&super::KvCacheSnapshot, usize)> = snaps
            .iter()
            .zip(prompts.iter())
            .map(|(s, p)| (s, p.len()))
            .collect();
        let batch = super::assemble_batch(&seqs).expect("assemble");
        assert_eq!(batch.padded_len, 5);
        assert_eq!(batch.prefix_lens, vec![3, 2, 5]);
        model
            .restore_kv_cache(&batch.snapshot)
            .expect("install batched state");

        for t in 0..n_steps {
            let toks: Vec<u32> = steps.iter().map(|s| s[t]).collect();
            let input = Tensor::from_vec(toks, (3, 1), &Device::Cpu).unwrap();
            let positions: Vec<usize> = prompts.iter().map(|p| p.len() + t).collect();
            let total_len = batch.padded_len + t + 1;
            let mask = model
                .batch_decode_mask(&batch.prefix_lens, batch.padded_len, total_len)
                .expect("mask");
            assert!(mask.is_some(), "ragged batch must be masked");
            let h = model
                .forward_batch_decode(&input, &positions, mask.as_ref())
                .expect("batched step");
            assert_eq!(h.dims()[0], 3);
            for row in 0..3 {
                let got: Vec<f32> = h.i((row, 0, ..)).unwrap().to_vec1().unwrap();
                let diff = max_abs_diff(&expected[row][t], &got);
                assert!(diff < 1e-4, "row {row} step {t} diverged: {diff}");
            }
        }
    }

    /// Uniform-length batch: no padding → `batch_decode_mask` returns
    /// `None`, and unmasked lockstep decode still matches sequential.
    #[test]
    fn batched_decode_uniform_lengths_needs_no_mask() {
        use candle_core::IndexOp;
        let cfg = tiny_config();
        let mut model = tiny_model(&cfg);

        let prompts: [&[u32]; 2] = [&[1, 2, 3], &[6, 5, 4]];
        let toks = [13u32, 17];

        let mut expected = Vec::new();
        for (prompt, tok) in prompts.iter().zip(toks.iter()) {
            model.clear_kv_cache();
            forward_tokens(&mut model, prompt, 0);
            expected.push(forward_tokens(&mut model, &[*tok], prompt.len()));
        }

        let mut snaps = Vec::new();
        for prompt in prompts.iter() {
            model.clear_kv_cache();
            forward_tokens(&mut model, prompt, 0);
            snaps.push(model.snapshot_kv_cache().expect("snapshot"));
        }
        let seqs: Vec<(&super::KvCacheSnapshot, usize)> = snaps
            .iter()
            .zip(prompts.iter())
            .map(|(s, p)| (s, p.len()))
            .collect();
        let batch = super::assemble_batch(&seqs).expect("assemble");
        let mask = model
            .batch_decode_mask(&batch.prefix_lens, batch.padded_len, batch.padded_len + 1)
            .expect("mask");
        assert!(mask.is_none(), "uniform lengths must not build a mask");
        model.restore_kv_cache(&batch.snapshot).expect("install");

        let input = Tensor::from_vec(toks.to_vec(), (2, 1), &Device::Cpu).unwrap();
        let h = model
            .forward_batch_decode(&input, &[3, 3], None)
            .expect("step");
        for row in 0..2 {
            let got: Vec<f32> = h.i((row, 0, ..)).unwrap().to_vec1().unwrap();
            let diff = max_abs_diff(&expected[row], &got);
            assert!(diff < 1e-4, "row {row} diverged: {diff}");
        }
    }

    /// Mask geometry: `-inf` exactly on `[prefix_len, padded_len)` per
    /// row, zero elsewhere (including the decode columns past
    /// `padded_len`).
    #[test]
    fn batch_decode_mask_covers_only_padding_gap() {
        let model = tiny_model(&tiny_config());
        let m = model
            .batch_decode_mask(&[3, 5], 5, 7)
            .unwrap()
            .expect("ragged → mask");
        assert_eq!(m.dims(), &[2, 1, 1, 7]);
        let flat: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();
        let (row0, row1) = flat.split_at(7);
        for (j, &v) in row0.iter().enumerate() {
            if (3..5).contains(&j) {
                assert_eq!(v, f32::NEG_INFINITY, "row0 col {j} must be masked");
            } else {
                assert_eq!(v, 0.0, "row0 col {j} must be open");
            }
        }
        assert!(row1.iter().all(|&v| v == 0.0), "unpadded row must be open");
    }

    /// Restoring must fully replace the live state, not blend with it
    /// — a divergent continuation after restore equals the same
    /// continuation after a fresh prefill of the prefix.
    #[test]
    fn restore_replaces_live_state() {
        let cfg = tiny_config();
        let mut model = tiny_model(&cfg);

        let prefix: &[u32] = &[7, 7, 2, 5];
        let cont: &[u32] = &[11, 13];

        model.clear_kv_cache();
        forward_tokens(&mut model, prefix, 0);
        let h_fresh = forward_tokens(&mut model, cont, prefix.len());

        model.clear_kv_cache();
        forward_tokens(&mut model, prefix, 0);
        let snap = model.snapshot_kv_cache().expect("snapshot");
        forward_tokens(&mut model, &[3, 1, 4, 1, 5], prefix.len());
        model.restore_kv_cache(&snap).expect("restore");
        let h_restored = forward_tokens(&mut model, cont, prefix.len());

        let diff = max_abs_diff(&h_fresh, &h_restored);
        assert!(diff < 1e-5, "restore did not replace live state: {diff}");
    }
}
