//! `qwen4_exp`'s decoder layer — the composition, and the only part of
//! it that is new.
//!
//! Every other architecture we serve wraps a sublayer as
//! `x = x + sublayer(norm(x))` over one residual stream. This one
//! carries **four**, concatenated into a `hidden_size * hc_count`
//! tensor, and has **no `input_layernorm`, no
//! `post_attention_layernorm` and no final `model.norm`** anywhere in
//! the checkpoint. The `hc_norm` inside each hyper-connection does all
//! three jobs.
//!
//! ```text
//! h [B,T,10240]
//!   ├─ (layer 1 only)  h = h + PLE(h, gathered n-grams)
//!   ├─ attn_hyper_connection: split → linear_attn | self_attn → scatter
//!   └─ mlp_hyper_connection:  split → sparse_moe               → scatter
//! ```
//!
//! The spec's warning about this layer is worth repeating: a wrong
//! stream mix still produces fluent text. So the plumbing lives in
//! [`through`], which takes the sublayer as a closure and can be tested
//! against a known function without constructing an attention block —
//! and the two things it must get right are asserted there:
//!
//! 1. **The sublayer sees the mixed `hidden_size` vector**, not the
//!    four-stream tensor. It is 2560 wide, and the sublayers are built
//!    for exactly that.
//! 2. **The residual added back is the un-normalised `h`.** Only the
//!    mixing and injection paths see the normalised form.
//!
//! PLE is added into the stream *before* the attention
//! hyper-connection, not merged into it — the layer's own blocks then
//! run on the sum. See `doc/qwen4_exp-port-spec.md`.

use anyhow::{Result, anyhow};
use candle_core::{Module, Tensor};
use candle_nn::var_builder::ShardedVarBuilder;
use std::sync::Arc;

use crate::harness::arch::qwen3_5::linear_attn::GatedDeltaNet;
use crate::harness::arch::qwen3_5::moe::Qwen3_5MoeBlock;
use crate::harness::arch::qwen3_5::rope::RotaryEmbedding;

use super::config::TextConfig;
use super::full_attn::Attention;
use super::hyper::HyperConnection;
use super::ple::PleBlock;
use super::{linear_attn, moe};
use crate::harness::arch::snapshot::LayerKvSnapshot;

/// The attention slot: one layer in four is full attention with the QSA
/// indexer, the rest are the recurrent delta rule.
enum Mixer {
    Full(Box<Attention>),
    Linear(Box<GatedDeltaNet>),
}

pub struct DecoderLayer {
    attn_hc: HyperConnection,
    mlp_hc: HyperConnection,
    mixer: Mixer,
    /// Every layer's FFN is the MoE; there is no dense width in this
    /// config to build an alternative from.
    mlp: Qwen3_5MoeBlock,
    /// Present on exactly the layers `ple_layer_ids` names — layer 1,
    /// zero-indexed, for the shipped checkpoint.
    ple: Option<PleBlock>,
}

impl DecoderLayer {
    pub fn load(
        cfg: &TextConfig,
        rotary: Arc<RotaryEmbedding>,
        layer_idx: usize,
        vb: &ShardedVarBuilder,
    ) -> Result<Self> {
        let layer_type = cfg
            .layer_types
            .get(layer_idx)
            .map(String::as_str)
            .ok_or_else(|| {
                anyhow!(
                    "layer_types[{layer_idx}] missing (have {} entries)",
                    cfg.layer_types.len()
                )
            })?;
        let mixer = match layer_type {
            "full_attention" => {
                Mixer::Full(Box::new(Attention::load(cfg, rotary, &vb.pp("self_attn"))?))
            }
            "linear_attention" => {
                Mixer::Linear(Box::new(linear_attn::load(cfg, &vb.pp("linear_attn"))?))
            }
            other => anyhow::bail!(
                "unknown layer_type '{other}' for layer {layer_idx} (expected \
                 'full_attention' or 'linear_attention')"
            ),
        };

        let hc = |name: &str| {
            HyperConnection::load(
                &vb.pp(name),
                cfg.hidden_size,
                cfg.hc_count,
                cfg.hc_lowrank,
                cfg.rms_norm_eps,
                // Decoder hyper-connections always scatter back; only
                // the top-level mixer does not.
                true,
            )
        };

        let ple = if cfg.ple_layers().contains(&layer_idx) {
            Some(PleBlock::load(
                &vb.pp("ple"),
                cfg.hidden_size,
                cfg.hc_count,
                cfg.ple_conv_kernel_size,
                // The dilation is the n-gram size, not the kernel.
                cfg.ngram_size,
                cfg.rms_norm_eps,
            )?)
        } else {
            None
        };

        Ok(Self {
            attn_hc: hc("attn_hyper_connection")?,
            mlp_hc: hc("mlp_hyper_connection")?,
            mixer,
            mlp: moe::load(cfg, &vb.pp("mlp"))?,
            ple,
        })
    }

    /// `h` is the four-stream residual, `[B, L, hidden * hc_count]`.
    ///
    /// `ple_embed` is the gathered n-gram embedding for this step,
    /// `[B, L, hidden]` — required on the PLE layer and ignored
    /// everywhere else.
    pub fn forward(
        &mut self,
        h: &Tensor,
        ple_embed: Option<&Tensor>,
        causal_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        past_len: usize,
    ) -> Result<Tensor> {
        // Split the borrows so the sublayer closures can take `&mut`
        // while the hyper-connections are read.
        let Self {
            attn_hc,
            mlp_hc,
            mixer,
            mlp,
            ple,
        } = self;

        let h = match ple {
            Some(block) => {
                let e = ple_embed.ok_or_else(|| {
                    anyhow!(
                        "this layer carries a PLE block but no gathered n-gram \
                         embedding was supplied"
                    )
                })?;
                (h + block.forward(h, e)?)?
            }
            None => h.clone(),
        };

        let h = through(attn_hc, &h, |x| match mixer {
            Mixer::Full(attn) => attn.forward(x, causal_mask, cos, sin, past_len),
            // The recurrent path's causality is in its state lifecycle,
            // so it takes neither mask nor rotary.
            Mixer::Linear(net) => Ok(net.forward(x)?),
        })?;

        through(mlp_hc, &h, |x| Ok(mlp.forward(x)?))
    }

    pub fn clear_kv_cache(&mut self) -> Result<()> {
        match &mut self.mixer {
            Mixer::Full(attn) => attn.clear_kv_cache(),
            Mixer::Linear(net) => net.clear_kv_cache(),
        }
        if let Some(ple) = &mut self.ple {
            ple.clear_state();
        }
        Ok(())
    }

    /// Capture this layer's attention-side state.
    pub fn snapshot_kv(&self) -> candle_core::Result<LayerKvSnapshot> {
        Ok(match &self.mixer {
            Mixer::Full(attn) => {
                let (kv, indexer_keys) = attn.snapshot_kv();
                LayerKvSnapshot::FullSparse { kv, indexer_keys }
            }
            Mixer::Linear(net) => {
                let (conv_state, recurrent_state) = net.snapshot_state()?;
                LayerKvSnapshot::Linear {
                    conv_state,
                    recurrent_state,
                }
            }
        })
    }

    /// Replace this layer's attention-side state. A variant mismatch
    /// means the snapshot came from a different model, or from the same
    /// model with a different `layer_types` — either way, restoring it
    /// would put a recurrent state into an attention layer.
    pub fn restore_kv(&mut self, snap: &LayerKvSnapshot) -> candle_core::Result<()> {
        match (&mut self.mixer, snap) {
            (Mixer::Full(attn), LayerKvSnapshot::FullSparse { kv, indexer_keys }) => {
                attn.restore_kv(kv.as_ref(), indexer_keys.as_ref())
            }
            (
                Mixer::Linear(net),
                LayerKvSnapshot::Linear {
                    conv_state,
                    recurrent_state,
                },
            ) => net.restore_state(conv_state.as_ref(), recurrent_state.as_ref()),
            _ => candle_core::bail!(
                "restore_kv: snapshot layer kind does not match this layer's mixer kind"
            ),
        }
    }

    /// The PLE block's rolling conv context, if this layer carries one.
    pub fn snapshot_ple(&self) -> candle_core::Result<Option<Tensor>> {
        match &self.ple {
            Some(block) => block.snapshot_state(),
            None => Ok(None),
        }
    }

    /// Restore the PLE conv context. Refuses a snapshot that disagrees
    /// with this layer about whether PLE lives here at all.
    pub fn restore_ple(&mut self, conv_state: Option<&Tensor>) -> candle_core::Result<()> {
        match &mut self.ple {
            Some(block) => block.restore_state(conv_state),
            None if conv_state.is_none() => Ok(()),
            None => candle_core::bail!(
                "restore_ple: snapshot carries PLE state but this layer has no PLE block"
            ),
        }
    }

    pub fn has_ple(&self) -> bool {
        self.ple.is_some()
    }
}

/// Run one sublayer through a hyper-connection.
///
/// Split the four streams into the `hidden_size` vector the sublayer
/// expects, call it, and scatter the result back across the streams
/// with the per-stream injection weights. The tensor added back is the
/// **un-normalised** `h`; only the mixing and injection paths see the
/// normalised form.
fn through<F>(hc: &HyperConnection, h: &Tensor, sublayer: F) -> Result<Tensor>
where
    F: FnOnce(&Tensor) -> Result<Tensor>,
{
    let split = hc.split(h)?;
    let inject = split.inject.as_ref().ok_or_else(|| {
        anyhow!("a decoder hyper-connection must carry block_inject_weight to scatter back")
    })?;
    let out = sublayer(&split.input)?;
    Ok(hc.combine(&split.residual, &out, inject)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::cell::RefCell;

    /// With every projection zeroed the gate is exactly 0.5 and the
    /// injection exactly 1, so `through` reduces to
    /// `h + (sublayer(mean of normalised streams) broadcast to all)`.
    fn hc(hidden: usize, hc_count: usize) -> HyperConnection {
        HyperConnection::zeroed_for_test(hidden, hc_count, 2, true)
    }

    fn stream(values: &[f32]) -> Tensor {
        Tensor::from_vec(values.to_vec(), (1, 1, values.len()), &Device::Cpu).unwrap()
    }

    /// The sublayer is handed the mixed `hidden_size` vector, not the
    /// four-stream tensor. Feeding it the raw stream would be four
    /// times too wide and every sublayer in the model is built for the
    /// narrow one.
    #[test]
    fn the_sublayer_sees_one_stream_and_the_result_is_four() {
        let seen = RefCell::new(Vec::new());
        let h = stream(&[3.0, 4.0, 30.0, 40.0]);
        let out = through(&hc(2, 2), &h, |x| {
            seen.borrow_mut().extend_from_slice(x.dims());
            Ok(x.zeros_like()?)
        })
        .unwrap();

        assert_eq!(seen.into_inner(), vec![1, 1, 2], "sublayer input width");
        assert_eq!(out.dims(), &[1, 1, 4], "the residual stays four streams");
    }

    /// A sublayer that returns zero must leave the stream exactly as it
    /// was — not as its normalised form. This is the assertion that
    /// separates a correct residual from one that quietly re-normalises
    /// the model every layer.
    #[test]
    fn a_silent_sublayer_leaves_the_stream_untouched() {
        let h = stream(&[3.0, 4.0, 30.0, 40.0]);
        let out = through(&hc(2, 2), &h, |x| Ok(x.zeros_like()?)).unwrap();
        let got: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got, vec![3.0, 4.0, 30.0, 40.0]);
    }

    /// The sublayer's output reaches every stream, scaled by that
    /// stream's injection weight — 1.0 for all of them here.
    #[test]
    fn the_result_is_injected_into_every_stream() {
        let h = stream(&[3.0, 4.0, 30.0, 40.0]);
        let out = through(&hc(2, 2), &h, |x| Ok((x.ones_like()? * 7.0)?)).unwrap();
        let got: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got, vec![10.0, 11.0, 37.0, 47.0]);
    }

    /// A hyper-connection with no `block_inject_weight` is the
    /// top-level mixer, which collapses the streams instead of feeding
    /// a sublayer. Using one here would silently drop the residual.
    #[test]
    fn a_mixer_cannot_stand_in_for_a_decoder_hyper_connection() {
        let mixer = HyperConnection::zeroed_for_test(2, 2, 2, false);
        let h = stream(&[1.0, 1.0, 1.0, 1.0]);
        let err = through(&mixer, &h, |x| Ok(x.clone()))
            .unwrap_err()
            .to_string();
        assert!(err.contains("block_inject_weight"), "got: {err}");
    }

    #[test]
    fn ple_layers_are_the_zero_indexed_ones() {
        let cfg =
            super::super::config::Config::from_config_json(super::super::config::SHIPPED).unwrap();
        // The composition asks the config, and the config is one-indexed.
        assert_eq!(cfg.text_config.ple_layers(), vec![1]);
        assert!(!cfg.text_config.ple_layers().contains(&2));
    }

    #[test]
    fn zeroed_stream_keeps_its_dtype_and_shape() {
        let h = Tensor::zeros((2, 3, 8), DType::F32, &Device::Cpu).unwrap();
        let out = through(&hc(2, 4), &h, |x| Ok(x.clone())).unwrap();
        assert_eq!(out.dims(), &[2, 3, 8]);
    }
}
