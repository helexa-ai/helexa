//! The multi-token-prediction head — `qwen4_exp`'s shipped draft model.
//!
//! The checkpoint carries a complete second decoder layer under `mtp.*`:
//! full attention with its own QSA indexer, its own 512-expert MoE, both
//! hyper-connections, and a final mixer. It reuses `embed_tokens` and
//! `lm_head` (`mtp_use_dedicated_embeddings: false`), so what is new
//! here is only how the target model's state and the next token's
//! embedding are fused into an input for it.
//!
//! ```text
//! e = fc_embedding(pre_fc_norm_embedding(embed(t)))   [T, 2560]
//! h = pre_fc_norm_hidden(h.flatten(-2))               ONE norm over 10240
//! h = fc_hidden(h.view(T, hc, 2560))                  per stream, shared matrix
//! h = e.unsqueeze(-2) + h                             embedding into every stream
//! h = h.flatten(-2)                                   [T, 10240] -> the layer
//! ```
//!
//! vLLM calls this `residual_linear_shared`, as against the
//! `Linear(2H, H)` + repeat other MTP variants use. Three parts of it
//! cannot be recovered from the tensor shapes, and all three were read
//! from `vllm/models/qwen4_exp/nvidia/mtp.py` rather than inferred:
//!
//! 1. **`pre_fc_norm_hidden` is ungrouped** — one normalisation over the
//!    whole 10240, unlike every other 10240-wide norm in this
//!    architecture. vLLM has a `GroupedGemmaRMSNorm` and deliberately
//!    does not use it here. Grouping it rescales the draft input in four
//!    independent pieces, which costs acceptance rate and reports
//!    nothing.
//! 2. **The hidden state is the *pre-final-mixer* multi-stream** — `h`
//!    after the target's last decoder layer, before its mixer collapses
//!    it. Not the 2560 the LM head sees.
//! 3. **The head emits both** — a collapsed stream for `lm_head` and the
//!    pre-final-mixer stream for its own next step. A head that returned
//!    only the first cannot be chained past one token.
//!
//! PLE is off in this layer while the stream count is kept, which the
//! absence of `mtp.layers.0.ple.*` in the checkpoint agrees with.
//!
//! **The head keeps its own KV and indexer caches, and they are not the
//! target's.** It therefore has to walk the prompt alongside the target
//! before it can draft: a head fed only the newest token is attending
//! over an empty past, and this architecture says so rather than
//! guessing, because the indexer's divergence check fires. That makes
//! prefill part of the speculative loop's cost, not just decode.
//!
//! See `doc/qwen4_exp-port-spec.md` §9 and #313.

use anyhow::{Context, Result};
use candle_core::quantized::GgmlDType;
use candle_core::{D, Module, Tensor};
use candle_nn::Linear;
use candle_nn::var_builder::ShardedVarBuilder;
use std::sync::Arc;

use crate::harness::arch::qwen3_5::rmsnorm::Qwen3_5RmsNorm;
use crate::harness::arch::qwen3_5::rope::RotaryEmbedding;

use super::config::TextConfig;
use super::decoder::DecoderLayer;
use super::hyper::HyperConnection;

/// One draft step's two outputs.
#[derive(Debug)]
pub struct MtpStep {
    /// `[B, L, hidden]` — the mixer's collapse, for `lm_head`.
    pub sample_hidden: Tensor,
    /// `[B, L, hidden * hc_count]` — pre-final-mixer, and the input to
    /// the *next* draft step. Chaining reads this, not the collapse.
    pub multi_stream: Tensor,
}

pub struct MtpHead {
    pre_fc_norm_embedding: Qwen3_5RmsNorm,
    /// Ungrouped over `hidden * hc_count` — see the module note.
    pre_fc_norm_hidden: Qwen3_5RmsNorm,
    fc_embedding: Linear,
    /// Shared across the streams: one `hidden -> hidden` matrix applied
    /// to each branch, not a `wide -> wide` one.
    fc_hidden: Linear,
    layer: DecoderLayer,
    mixer: HyperConnection,
    hidden_size: usize,
    hc_count: usize,
}

impl MtpHead {
    /// `vb` is the checkpoint root; the head hangs off `mtp.`.
    pub fn load(
        cfg: &TextConfig,
        rotary: Arc<RotaryEmbedding>,
        quant: Option<GgmlDType>,
        vb: &ShardedVarBuilder,
    ) -> Result<Self> {
        let mtp = vb.pp("mtp");
        let (h, hc) = (cfg.hidden_size, cfg.hc_count);
        Ok(Self {
            pre_fc_norm_embedding: Qwen3_5RmsNorm::load(
                &mtp.pp("pre_fc_norm_embedding"),
                h,
                cfg.rms_norm_eps,
            )?,
            // `load`, not `load_grouped`: the one 10240-wide norm in
            // this architecture that is not per-stream.
            pre_fc_norm_hidden: Qwen3_5RmsNorm::load(
                &mtp.pp("pre_fc_norm_hidden"),
                h * hc,
                cfg.rms_norm_eps,
            )?,
            fc_embedding: linear(&mtp, "fc_embedding", h, h)?,
            fc_hidden: linear(&mtp, "fc_hidden", h, h)?,
            layer: DecoderLayer::load_typed(
                cfg,
                rotary,
                "full_attention",
                false,
                quant,
                &mtp.pp("layers").pp(0),
            )
            .context("load mtp.layers.0")?,
            mixer: HyperConnection::load(
                &mtp.pp("hyper_connection_mixer"),
                h,
                hc,
                cfg.hc_lowrank,
                cfg.rms_norm_eps,
                false,
            )
            .context("load mtp.hyper_connection_mixer")?,
            hidden_size: h,
            hc_count: hc,
        })
    }

    /// One draft step.
    ///
    /// `token_embed` is `embed_tokens(t)` for the token being predicted
    /// *from*, `[B, L, hidden]`. `hidden` is the pre-final-mixer stream,
    /// `[B, L, hidden * hc_count]` — from the target model on the first
    /// step, from this head's own previous [`MtpStep::multi_stream`]
    /// afterwards.
    pub fn forward(
        &mut self,
        token_embed: &Tensor,
        hidden: &Tensor,
        causal_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        past_len: usize,
    ) -> Result<MtpStep> {
        let (b, l, wide) = hidden.dims3()?;
        anyhow::ensure!(
            wide == self.hidden_size * self.hc_count,
            "mtp: expected the pre-final-mixer stream ({} wide), got {wide} — \
             passing the collapsed hidden state here is the likely mistake",
            self.hidden_size * self.hc_count
        );

        let e = self
            .fc_embedding
            .forward(&self.pre_fc_norm_embedding.forward(token_embed)?)?;

        // Normalise the flattened stream, *then* split it: one
        // normalisation over the whole width.
        let h = self.pre_fc_norm_hidden.forward(hidden)?.reshape((
            b,
            l,
            self.hc_count,
            self.hidden_size,
        ))?;
        let h = self.fc_hidden.forward(&h)?;

        // The embedding enters every stream.
        let h = h
            .broadcast_add(&e.unsqueeze(D::Minus2)?)?
            .flatten_from(D::Minus2)?;

        let h = self
            .layer
            .forward(&h, None, causal_mask, cos, sin, past_len)
            .context("mtp draft layer")?;

        Ok(MtpStep {
            sample_hidden: self.mixer.collapse(&h)?,
            multi_stream: h,
        })
    }

    /// Drop the draft layer's caches. The head has its own KV and its
    /// own indexer cache, independent of the target model's.
    pub fn clear_kv_cache(&mut self) -> Result<()> {
        self.layer.clear_kv_cache()
    }
}

fn linear(vb: &ShardedVarBuilder, name: &str, in_dim: usize, out_dim: usize) -> Result<Linear> {
    let weight = vb
        .pp(name)
        .get((out_dim, in_dim), "weight")
        .with_context(|| format!("load '{}/{name}/weight'", vb.prefix()))?;
    Ok(Linear::new(weight, None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// `pre_fc_norm_hidden` normalises the whole 10240 at once, where
    /// every other norm this wide in the architecture normalises four
    /// streams of 2560 independently. The port spec asserted the
    /// opposite until vLLM settled it, and the difference is invisible
    /// downstream — it rescales the draft input and costs acceptance
    /// rate.
    ///
    /// So it is pinned by value, on an input the two readings cannot
    /// agree on: two streams whose magnitudes differ threefold. Grouped,
    /// each is normalised to its own scale and both come out all-ones.
    /// Ungrouped, the larger stream stays larger.
    #[test]
    fn the_hidden_pre_norm_is_ungrouped() {
        let dev = Device::Cpu;
        let (hidden, hc) = (4usize, 2usize);
        let wide = hidden * hc;
        // Zero weight: (1 + w) = 1, so this is a pure RMS normalisation.
        let norm =
            Qwen3_5RmsNorm::from_weight(Tensor::zeros(wide, DType::F32, &dev).unwrap(), 1e-6, None);
        let h = Tensor::from_vec(
            vec![1.0f32, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0],
            (1, 1, wide),
            &dev,
        )
        .unwrap();

        let got: Vec<f32> = norm
            .forward(&h)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // rms over all eight = sqrt((4*1 + 4*9)/8) = sqrt(5)
        let rms = 5.0f32.sqrt();
        let want = [
            1.0 / rms,
            1.0 / rms,
            1.0 / rms,
            1.0 / rms,
            3.0 / rms,
            3.0 / rms,
            3.0 / rms,
            3.0 / rms,
        ];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "got {got:?} want {want:?}");
        }
        // The grouped reading would flatten both streams to all-ones.
        assert!(
            (got[4] - 1.0).abs() > 1e-3,
            "this is the grouped reading, which the spec had and vLLM does not: {got:?}"
        );
    }
}
