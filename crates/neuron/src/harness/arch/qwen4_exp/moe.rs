//! `qwen4_exp`'s MoE FFN — every layer's FFN, since there is no dense
//! one.
//!
//! The routing is [`Qwen3_5MoeBlock`]'s, unchanged: softmax over all
//! 512 in f32, top 10, renormalise, add a sigmoid-gated shared expert.
//! What differs is **storage**. This checkpoint ships the routed
//! experts as two fused 3D tensors rather than 512 modules:
//!
//! ```text
//! mlp.experts.gate_up_proj   (512, 1280, 2560)   gate ++ up, stacked
//! mlp.experts.down_proj      (512, 2560,  640)
//! ```
//!
//! against our loader's `mlp.experts.{i}.{gate,up,down}_proj`. So this
//! module is a loader, not an algorithm: it slices the fused tensors
//! into the per-expert SwiGLUs the block already knows how to route.
//!
//! **In `gate_up_proj` the first `moe_intermediate_size` rows are the
//! gate and the last are the up projection.** Reading them the other
//! way round computes `silu(up) * gate` instead of `silu(gate) * up` —
//! a different function of the same weights, which is exactly the kind
//! of error that produces a model that talks and reasons slightly
//! worse. It is asserted below rather than commented.
//!
//! ## Why the fused tensors stay fused
//!
//! Slicing them into 512 per-expert modules copies **241.6 GB** at
//! BF16, 5.03 GB per layer. That fits in neither beast's 64 GB of VRAM
//! nor its 123 GB of host RAM, and no quantisation rescues it: #309
//! measured 59.4 GB of 63.7 used at 4.25 bpw with only *four* layers'
//! experts offloaded. A model that cannot be held cannot be loaded.
//!
//! So the block takes `Experts::Banked` and views one expert per
//! routed token instead. A slice along dim 0 of a contiguous tensor is
//! contiguous, and `x.matmul(w.t())` is what `Linear` does anyway, so
//! the views feed the same GEMMs with no copy. It is also the layout a
//! host-resident bank with a device-side gather wants (#318).
//!
//! See `doc/qwen4_exp-port-spec.md` §6.

use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_core::quantized::GgmlDType;
use candle_nn::Linear;
use candle_nn::var_builder::ShardedVarBuilder;

use crate::harness::arch::qwen3_5::mlp::Qwen3_5MLP;
use crate::harness::arch::qwen3_5::moe::{Experts, Qwen3_5MoeBlock};

use super::config::TextConfig;

/// `vb` should be `.pp(...)`-ed to the layer's `mlp` prefix.
///
/// `quant` quantises the routed experts in situ (#315). Without it the
/// experts stay banked at the checkpoint's precision, which is only
/// useful for a model small enough to hold — 120.8 B routed parameters
/// are 241.6 GB at BF16 and fit nowhere on this hardware, so a real
/// load of `Qwen3.8-Flash-Next` always passes one.
pub fn load(
    cfg: &TextConfig,
    quant: Option<GgmlDType>,
    vb: &ShardedVarBuilder,
) -> Result<Qwen3_5MoeBlock> {
    let (h, inter) = (cfg.hidden_size, cfg.moe_intermediate_size);
    anyhow::ensure!(
        cfg.num_experts > 0 && cfg.num_experts_per_tok > 0 && inter > 0,
        "MoE needs num_experts ({}), num_experts_per_tok ({}) and \
         moe_intermediate_size ({inter}) all > 0",
        cfg.num_experts,
        cfg.num_experts_per_tok,
    );

    let gate = Linear::new(
        vb.pp("gate")
            .get((cfg.num_experts, h), "weight")
            .with_context(|| format!("load '{}/gate/weight'", vb.prefix()))?,
        None,
    );

    let experts_vb = vb.pp("experts");
    let gate_up = experts_vb
        .get((cfg.num_experts, inter * 2, h), "gate_up_proj")
        .with_context(|| format!("load '{}/gate_up_proj'", experts_vb.prefix()))?;
    let down = experts_vb
        .get((cfg.num_experts, h, inter), "down_proj")
        .with_context(|| format!("load '{}/down_proj'", experts_vb.prefix()))?;
    check_fused_experts(&gate_up, &down, inter)?;
    // The fused pair is the peak: 5.03 GB per layer at BF16. Quantising
    // here and letting it drop at the end of this function is what keeps
    // the whole 241.6 GB from ever existing.
    let experts = match quant {
        Some(dtype) => Experts::quantize_banked(&gate_up, &down, inter, dtype)
            .with_context(|| format!("quantize experts to {dtype:?}"))?,
        None => Experts::Banked {
            gate_up,
            down,
            intermediate: inter,
        },
    };

    let (shared_expert, shared_expert_gate) = if cfg.shared_expert_intermediate_size > 0 {
        let shared = Qwen3_5MLP::load_with_dims(
            &vb.pp("shared_expert"),
            h,
            cfg.shared_expert_intermediate_size,
        )
        .context("load shared_expert")?;
        let gate_w = vb
            .pp("shared_expert_gate")
            .get((1, h), "weight")
            .with_context(|| format!("load '{}/shared_expert_gate/weight'", vb.prefix()))?;
        (Some(shared), Some(Linear::new(gate_w, None)))
    } else {
        (None, None)
    };

    Ok(Qwen3_5MoeBlock::from_parts(
        gate,
        experts,
        shared_expert,
        shared_expert_gate,
        cfg.num_experts_per_tok,
        cfg.norm_topk_prob,
    ))
}

/// Validate the fused pair before it is banked.
///
/// The block will take views of these per routed expert, so a
/// disagreement here surfaces as a shape error mid-forward on whichever
/// token first routes to the offending expert — which is a worse place
/// to find it than at load.
pub(crate) fn check_fused_experts(
    gate_up: &Tensor,
    down: &Tensor,
    intermediate: usize,
) -> Result<()> {
    let (experts, fused, _) = gate_up.dims3()?;
    anyhow::ensure!(
        fused == intermediate * 2,
        "gate_up_proj is {fused} wide, expected 2 x {intermediate}"
    );
    anyhow::ensure!(
        down.dims3()?.0 == experts,
        "down_proj holds {} experts, gate_up_proj holds {experts}",
        down.dims3()?.0
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// The loader validates the fused pair up front. Without this the
    /// first token to route to a bad expert fails mid-forward, which is
    /// a worse place to learn the checkpoint disagrees with the config.
    #[test]
    fn a_fused_pair_that_disagrees_with_the_config_is_rejected() {
        let dev = Device::Cpu;
        let gate_up = Tensor::zeros((2, 6, 4), DType::F32, &dev).unwrap();
        let down = Tensor::zeros((2, 4, 3), DType::F32, &dev).unwrap();
        assert!(check_fused_experts(&gate_up, &down, 3).is_ok());
        // 2 x 4 != 6
        assert!(check_fused_experts(&gate_up, &down, 4).is_err());
        // expert counts must agree
        let down_short = Tensor::zeros((1, 4, 3), DType::F32, &dev).unwrap();
        assert!(check_fused_experts(&gate_up, &down_short, 3).is_err());
    }
}
