//! The multi-token-prediction head — `qwen3_5`'s shipped draft model.
//!
//! `Qwen/Qwen3.8-27B` and `Qwen/Qwen3.5-0.8B` both carry a complete
//! extra decoder layer under `mtp.*` that nothing in this crate has
//! ever loaded: 15 tensors of 1199 in the 27B (810 MB BF16), 15 of 488
//! in the 0.8B (39 MB). It is the drafter speculative decoding needs
//! (#96), already in the weights we download.
//!
//! ```text
//! e = pre_fc_norm_embedding(embed(t))      [B, L, H]
//! h = pre_fc_norm_hidden(h)                [B, L, H]  (the target's)
//! x = fc(cat[e, h])                        [B, L, 2H] -> [B, L, H]
//! x = layer(x)                             one full-attention block
//! x = norm(x)                              -> the target's lm_head
//! ```
//!
//! Three things here cannot be recovered from the tensor shapes, and
//! all three were read from vLLM's `qwen3_next_mtp.py` rather than
//! inferred — the same discipline, and for the same reason, as the
//! sibling head in `arch/qwen4_exp/mtp.rs`:
//!
//! 1. **The concatenation is `[embedding, hidden]`**, not the reverse:
//!    `hidden_states = torch.cat([inputs_embeds, hidden_states], -1)`.
//!    `fc` is square in the sense that both halves are `hidden_size`
//!    wide, so swapping them loads, runs, and produces finite logits
//!    that are simply wrong. Pinned by value below.
//! 2. **Each half is normalised before the concatenation**, by its own
//!    norm — `pre_fc_norm_embedding` on the embedding,
//!    `pre_fc_norm_hidden` on the hidden state — never one norm over
//!    the joined 2H.
//! 3. **The hidden state is the target's *post-final-norm* output**,
//!    the same tensor its `lm_head` consumes. vLLM's
//!    `Qwen3NextModel::forward` returns `self.norm(...)` and the runner
//!    hands that to the head. Feeding the pre-norm residual instead is
//!    the other plausible reading, and it is wrong for this
//!    architecture (`qwen4_exp`'s head takes its pre-mixer stream —
//!    they do not generalise to each other).
//!
//! **The head has no `lm_head` or `embed_tokens` of its own.** The
//! checkpoint carries neither (`mtp_use_dedicated_embeddings: false`),
//! so both come from the target — which is what makes the resident cost
//! the 810 MB above rather than that plus a second embedding table.
//!
//! **The head keeps its own KV cache, and it is not the target's.** A
//! head fed only the newest token attends over an empty past, so it has
//! to walk the prompt alongside the target before it can draft. That
//! puts MTP prefill inside the speculative loop's cost, not just decode
//! — see #96.

use anyhow::{Context, Result};
use candle_core::{Module, Tensor};
use candle_nn::Linear;
use candle_nn::var_builder::ShardedVarBuilder;
use std::sync::Arc;

use super::TextConfig;
use super::decoder::Qwen3_5DecoderLayer;
use super::rmsnorm::Qwen3_5RmsNorm;
use super::rope::RotaryEmbedding;

pub struct MtpHead {
    pre_fc_norm_embedding: Qwen3_5RmsNorm,
    pre_fc_norm_hidden: Qwen3_5RmsNorm,
    /// `Linear(2H -> H)` over `cat[embedding, hidden]`.
    fc: Linear,
    layer: Qwen3_5DecoderLayer,
    /// Final norm before the *target's* lm_head.
    norm: Qwen3_5RmsNorm,
}

impl MtpHead {
    /// `vb` is the checkpoint root; the head hangs off `mtp.`.
    pub fn load(
        cfg: &TextConfig,
        rotary: Arc<RotaryEmbedding>,
        vb: &ShardedVarBuilder,
    ) -> Result<Self> {
        let mtp = vb.pp("mtp");
        let h = cfg.hidden_size;
        let fc_weight = mtp
            .pp("fc")
            .get((h, 2 * h), "weight")
            .context("load 'mtp.fc.weight'")?;
        Ok(Self {
            pre_fc_norm_embedding: Qwen3_5RmsNorm::load(
                &mtp.pp("pre_fc_norm_embedding"),
                h,
                cfg.rms_norm_eps,
            )
            .context("load mtp.pre_fc_norm_embedding")?,
            pre_fc_norm_hidden: Qwen3_5RmsNorm::load(
                &mtp.pp("pre_fc_norm_hidden"),
                h,
                cfg.rms_norm_eps,
            )
            .context("load mtp.pre_fc_norm_hidden")?,
            fc: Linear::new(fc_weight, None),
            // Stated, not looked up: the head's layer is outside
            // `layer_types`. Full attention, dense MLP — see
            // `Qwen3_5DecoderLayer::load_typed`.
            layer: Qwen3_5DecoderLayer::load_typed(
                cfg,
                rotary,
                "full_attention",
                false,
                &mtp.pp("layers").pp(0),
            )
            .context("load mtp.layers.0")?,
            norm: Qwen3_5RmsNorm::load(&mtp.pp("norm"), h, cfg.rms_norm_eps)
                .context("load mtp.norm")?,
        })
    }

    /// Whether this checkpoint carries a head at all.
    ///
    /// `mtp_num_hidden_layers` is the config's own answer and is 1 on
    /// the checkpoints that ship one. Absent or 0 means the weights
    /// have no `mtp.*` tensors and [`Self::load`] would fail.
    pub fn present_in(cfg: &TextConfig) -> bool {
        cfg.mtp_num_hidden_layers > 0
    }

    /// One draft step.
    ///
    /// `token_embed` is `embed_tokens(t)` for the token the draft is
    /// predicting *from*, `[B, L, H]`. `hidden` is the target's
    /// post-final-norm hidden state at the same positions, `[B, L, H]`
    /// — on a chained second step, this head's own previous output.
    /// The returned tensor is `[B, L, H]`, for the *target's* lm_head.
    pub fn forward(
        &mut self,
        token_embed: &Tensor,
        hidden: &Tensor,
        attn_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        anyhow::ensure!(
            token_embed.dims() == hidden.dims(),
            "mtp: embedding {:?} and hidden {:?} must have the same shape — \
             the head fuses them position-for-position",
            token_embed.dims(),
            hidden.dims()
        );
        let e = self.pre_fc_norm_embedding.forward(token_embed)?;
        let h = self.pre_fc_norm_hidden.forward(hidden)?;
        // [embedding, hidden] — vLLM's order. See the module note.
        let x = Tensor::cat(&[&e, &h], candle_core::D::Minus1)?;
        let x = self.fc.forward(&x)?;
        let x = self
            .layer
            .forward(&x, attn_mask, cos, sin)
            .context("mtp draft layer")?;
        Ok(self.norm.forward(&x)?)
    }

    /// Drop the draft layer's KV cache. Independent of the target's.
    pub fn clear_kv_cache(&mut self) {
        self.layer.clear_kv_cache();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    /// Build a `ShardedVarBuilder` over a synthetic `mtp.*` head, the
    /// way the sibling tests in this arch do: write a safetensors file
    /// and open it.
    fn tiny_head(
        dir: &std::path::Path,
        h: usize,
        inter: usize,
        heads: usize,
        kv: usize,
        head_dim: usize,
    ) -> ShardedVarBuilder<'static> {
        let dev = Device::Cpu;
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let zeros = |shape: &[usize]| Tensor::zeros(shape, DType::F32, &dev).unwrap();
        let randn = |shape: &[usize]| Tensor::randn(0f32, 0.1f32, shape, &dev).unwrap();

        t.insert("mtp.fc.weight".into(), randn(&[h, 2 * h]));
        t.insert("mtp.norm.weight".into(), zeros(&[h]));
        t.insert("mtp.pre_fc_norm_embedding.weight".into(), zeros(&[h]));
        t.insert("mtp.pre_fc_norm_hidden.weight".into(), zeros(&[h]));
        let l = "mtp.layers.0";
        t.insert(format!("{l}.input_layernorm.weight"), zeros(&[h]));
        t.insert(format!("{l}.post_attention_layernorm.weight"), zeros(&[h]));
        // Output-gated q_proj: 2 * heads * head_dim rows (attn_output_gate).
        t.insert(
            format!("{l}.self_attn.q_proj.weight"),
            randn(&[2 * heads * head_dim, h]),
        );
        t.insert(
            format!("{l}.self_attn.k_proj.weight"),
            randn(&[kv * head_dim, h]),
        );
        t.insert(
            format!("{l}.self_attn.v_proj.weight"),
            randn(&[kv * head_dim, h]),
        );
        t.insert(
            format!("{l}.self_attn.o_proj.weight"),
            randn(&[h, heads * head_dim]),
        );
        t.insert(format!("{l}.self_attn.q_norm.weight"), zeros(&[head_dim]));
        t.insert(format!("{l}.self_attn.k_norm.weight"), zeros(&[head_dim]));
        t.insert(format!("{l}.mlp.gate_proj.weight"), randn(&[inter, h]));
        t.insert(format!("{l}.mlp.up_proj.weight"), randn(&[inter, h]));
        t.insert(format!("{l}.mlp.down_proj.weight"), randn(&[h, inter]));

        let path = dir.join("mtp.safetensors");
        candle_core::safetensors::save(&t, &path).expect("write tiny mtp checkpoint");
        // SAFETY: mmaps a file this test just wrote and owns.
        unsafe {
            candle_nn::var_builder::ShardedSafeTensors::var_builder(
                std::slice::from_ref(&path),
                DType::F32,
                &dev,
            )
        }
        .expect("build ShardedVarBuilder")
    }

    fn tiny_cfg(h: usize, inter: usize, heads: usize, kv: usize, head_dim: usize) -> TextConfig {
        let raw = format!(
            r#"{{
                "model_type": "qwen3_next",
                "vocab_size": 32, "hidden_size": {h}, "intermediate_size": {inter},
                "num_hidden_layers": 1, "num_attention_heads": {heads},
                "num_key_value_heads": {kv}, "head_dim": {head_dim},
                "max_position_embeddings": 64, "rms_norm_eps": 1e-6,
                "attn_output_gate": true,
                "layer_types": ["full_attention"],
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false
            }}"#
        );
        super::super::Config::from_config_json(&raw)
            .expect("parse tiny qwen3_5 config")
            .text_config
    }

    /// The head loads from the `mtp.*` names the real checkpoints use
    /// and produces finite output of the target's hidden width — which
    /// is what the target's `lm_head` will consume.
    #[test]
    fn a_tiny_mtp_head_loads_and_drafts() {
        let (h, inter, heads, kv, head_dim) = (8usize, 16usize, 2usize, 1usize, 4usize);
        let dir = tempfile::tempdir().expect("tempdir");
        let vb = tiny_head(dir.path(), h, inter, heads, kv, head_dim);
        let cfg = tiny_cfg(h, inter, heads, kv, head_dim);
        assert!(MtpHead::present_in(&cfg));

        let dev = Device::Cpu;
        let rotary = Arc::new(RotaryEmbedding::new(DType::F32, &cfg, &dev).expect("rotary"));
        let mut head = MtpHead::load(&cfg, rotary.clone(), &vb).expect("load tiny mtp head");

        let (b, l) = (1usize, 3usize);
        let embed = Tensor::randn(0f32, 1.0, (b, l, h), &dev).unwrap();
        let hidden = Tensor::randn(0f32, 1.0, (b, l, h), &dev).unwrap();
        let positions: Vec<usize> = (0..l).collect();
        let (cos, sin) = rotary.cos_sin_at(&positions).expect("cos/sin");
        let out = head
            .forward(&embed, &hidden, None, &cos, &sin)
            .expect("draft step");
        assert_eq!(out.dims(), &[b, l, h]);
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            v.iter().all(|x| x.is_finite()),
            "draft hidden must be finite"
        );
    }

    /// `fc` consumes `cat[embedding, hidden]`, in that order.
    ///
    /// Both halves are `hidden_size` wide, so the reversed reading
    /// loads, runs, and returns finite numbers that are simply a
    /// different model. vLLM settles it —
    /// `torch.cat([inputs_embeds, hidden_states], dim=-1)` — and this
    /// pins it by value so a future edit cannot quietly swap them.
    ///
    /// The probe: `fc` reads only the embedding half (identity on the
    /// first H columns, zero on the second). With a zero embedding and
    /// a non-zero hidden state, the correct order yields zero and the
    /// reversed order does not.
    #[test]
    fn the_fc_input_is_embedding_then_hidden() {
        let dev = Device::Cpu;
        let h = 4usize;
        // fc = [I | 0] : takes the first H columns only.
        let mut w = vec![0f32; h * 2 * h];
        for i in 0..h {
            w[i * 2 * h + i] = 1.0;
        }
        let fc = Linear::new(Tensor::from_vec(w, (h, 2 * h), &dev).unwrap(), None);
        // Zero weight => (1 + w) = 1, a pure RMS normalisation, so the
        // norms cannot mask the ordering.
        let norm =
            || Qwen3_5RmsNorm::from_weight(Tensor::zeros(h, DType::F32, &dev).unwrap(), 1e-6, None);

        let e = Tensor::zeros((1, 1, h), DType::F32, &dev).unwrap();
        let hid = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, h), &dev).unwrap();

        let en = norm().forward(&e).unwrap();
        let hn = norm().forward(&hid).unwrap();

        let correct: Vec<f32> = fc
            .forward(&Tensor::cat(&[&en, &hn], candle_core::D::Minus1).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let reversed: Vec<f32> = fc
            .forward(&Tensor::cat(&[&hn, &en], candle_core::D::Minus1).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            correct.iter().all(|x| x.abs() < 1e-6),
            "with [embedding, hidden] the zero embedding must reach fc: {correct:?}"
        );
        assert!(
            reversed.iter().any(|x| x.abs() > 1e-3),
            "the reversed reading is what this test exists to catch: {reversed:?}"
        );
    }

    /// `forward_multi` must agree with `forward` at the position they
    /// share.
    ///
    /// The speculative verify pass (#96) reads the target's own token
    /// at every drafted position, so the whole scheme rests on those
    /// logits being the ones the model would have produced anyway. The
    /// two differ by one slice, and this pins that they differ by
    /// nothing else: same tokens, same offset, last row bit-identical.
    #[test]
    fn forward_multi_agrees_with_forward_at_the_last_position() {
        use super::super::{Config, Qwen3_5ForCausalLM};
        use candle_core::IndexOp;

        let (h, inter, heads, kv, head_dim, vocab) =
            (8usize, 16usize, 2usize, 1usize, 4usize, 32usize);
        let dev = Device::Cpu;
        let dir = tempfile::tempdir().expect("tempdir");
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let zeros = |shape: &[usize]| Tensor::zeros(shape, DType::F32, &dev).unwrap();
        let randn = |shape: &[usize]| Tensor::randn(0f32, 0.1f32, shape, &dev).unwrap();
        t.insert("model.embed_tokens.weight".into(), randn(&[vocab, h]));
        t.insert("lm_head.weight".into(), randn(&[vocab, h]));
        t.insert("model.norm.weight".into(), zeros(&[h]));
        let l = "model.layers.0";
        t.insert(format!("{l}.input_layernorm.weight"), zeros(&[h]));
        t.insert(format!("{l}.post_attention_layernorm.weight"), zeros(&[h]));
        t.insert(
            format!("{l}.self_attn.q_proj.weight"),
            randn(&[2 * heads * head_dim, h]),
        );
        t.insert(
            format!("{l}.self_attn.k_proj.weight"),
            randn(&[kv * head_dim, h]),
        );
        t.insert(
            format!("{l}.self_attn.v_proj.weight"),
            randn(&[kv * head_dim, h]),
        );
        t.insert(
            format!("{l}.self_attn.o_proj.weight"),
            randn(&[h, heads * head_dim]),
        );
        t.insert(format!("{l}.self_attn.q_norm.weight"), zeros(&[head_dim]));
        t.insert(format!("{l}.self_attn.k_norm.weight"), zeros(&[head_dim]));
        t.insert(format!("{l}.mlp.gate_proj.weight"), randn(&[inter, h]));
        t.insert(format!("{l}.mlp.up_proj.weight"), randn(&[inter, h]));
        t.insert(format!("{l}.mlp.down_proj.weight"), randn(&[h, inter]));
        let path = dir.path().join("model.safetensors");
        candle_core::safetensors::save(&t, &path).expect("write tiny checkpoint");
        // SAFETY: mmaps a file this test just wrote and owns.
        let vb = unsafe {
            candle_nn::var_builder::ShardedSafeTensors::var_builder(
                std::slice::from_ref(&path),
                DType::F32,
                &dev,
            )
        }
        .expect("build ShardedVarBuilder");

        let raw = format!(
            r#"{{
                "model_type": "qwen3_next",
                "vocab_size": {vocab}, "hidden_size": {h}, "intermediate_size": {inter},
                "num_hidden_layers": 1, "num_attention_heads": {heads},
                "num_key_value_heads": {kv}, "head_dim": {head_dim},
                "max_position_embeddings": 64, "rms_norm_eps": 1e-6,
                "attn_output_gate": true,
                "layer_types": ["full_attention"]
            }}"#
        );
        let cfg = Config::from_config_json(&raw).expect("parse tiny config");
        let mut model = Qwen3_5ForCausalLM::new(cfg, vb).expect("load tiny checkpoint");

        let input = Tensor::new(&[1u32, 5, 9, 2], &dev)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let multi = model.forward_multi(&input, 0).expect("forward_multi");
        assert_eq!(multi.dims(), &[1, 4, vocab], "one logits row per position");
        let last_of_multi: Vec<f32> = multi.i((0, 3, ..)).unwrap().to_vec1().unwrap();
        // A head that returned the same row L times would satisfy the
        // comparison below without carrying per-position information,
        // which is the whole point of the pass.
        let first_of_multi: Vec<f32> = multi.i((0, 0, ..)).unwrap().to_vec1().unwrap();
        assert_ne!(
            first_of_multi, last_of_multi,
            "positions must carry distinct logits, not a repeated row"
        );

        model.clear_kv_cache();
        let single = model.forward(&input, 0).expect("forward");
        assert_eq!(single.dims(), &[1, 1, vocab]);
        let single: Vec<f32> = single.flatten_all().unwrap().to_vec1().unwrap();

        assert_eq!(
            last_of_multi, single,
            "the last row of forward_multi is what forward returns; they differ by a slice \
             and must differ by nothing else"
        );
    }
}
