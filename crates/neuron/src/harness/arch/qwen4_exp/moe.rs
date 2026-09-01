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
//! The fused layout is the better one for a grouped GEMM, and keeping
//! it would be the faster path; that is an algorithm change, and the
//! spec is explicit that this stage is not it. Sparsity is extreme —
//! 10 of 512 with an intermediate of 640 — so the routing, not the
//! GEMM shape, is what dominates today.
//!
//! See `doc/qwen4_exp-port-spec.md` §6.

use anyhow::{Context, Result};
use candle_core::{IndexOp, Tensor};
use candle_nn::Linear;
use candle_nn::var_builder::ShardedVarBuilder;

use crate::harness::arch::qwen3_5::mlp::Qwen3_5MLP;
use crate::harness::arch::qwen3_5::moe::Qwen3_5MoeBlock;

use super::config::TextConfig;

/// `vb` should be `.pp(...)`-ed to the layer's `mlp` prefix.
pub fn load(cfg: &TextConfig, vb: &ShardedVarBuilder) -> Result<Qwen3_5MoeBlock> {
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
    let experts = split_fused_experts(&gate_up, &down, inter)?;

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

/// Slice `(E, 2I, H)` and `(E, H, I)` into `E` SwiGLU experts.
///
/// Gate first, up second — see the module note.
pub(crate) fn split_fused_experts(
    gate_up: &Tensor,
    down: &Tensor,
    intermediate: usize,
) -> Result<Vec<Qwen3_5MLP>> {
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

    let mut out = Vec::with_capacity(experts);
    for e in 0..experts {
        let fused_e = gate_up.i(e)?;
        out.push(Qwen3_5MLP::from_weights(
            Linear::new(fused_e.narrow(0, 0, intermediate)?.contiguous()?, None),
            Linear::new(
                fused_e
                    .narrow(0, intermediate, intermediate)?
                    .contiguous()?,
                None,
            ),
            Linear::new(down.i(e)?.contiguous()?, None),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Module};

    /// One expert, two hidden dims, one intermediate. The gate row picks
    /// the first input channel and the up row picks the second, so
    /// swapping them swaps which channel is squashed by the SiLU — and
    /// the two answers differ by more than any tolerance.
    #[test]
    fn the_gate_is_the_first_half_of_the_fused_tensor() {
        let dev = Device::Cpu;
        // gate_up: [[gate row], [up row]] = [[1, 0], [0, 1]]
        let gate_up = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (1, 2, 2), &dev).unwrap();
        // down: (E=1, H=2, I=1)
        let down = Tensor::from_vec(vec![2.0f32, 3.0], (1, 2, 1), &dev).unwrap();

        let experts = split_fused_experts(&gate_up, &down, 1).unwrap();
        assert_eq!(experts.len(), 1);

        let x = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), &dev).unwrap();
        let got: Vec<f32> = experts[0]
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // gate(x) = 1, up(x) = 2  ->  silu(1) * 2  ->  down
        let silu1 = 1.0f32 / (1.0 + (-1.0f32).exp());
        let want = [2.0 * silu1 * 2.0, 3.0 * silu1 * 2.0];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "got {got:?} want {want:?}");
        }

        // The swapped reading: silu(2) * 1, which is a different model.
        let silu2 = 2.0f32 / (1.0 + (-2.0f32).exp());
        assert!(
            (got[0] - 2.0 * silu2).abs() > 1e-3,
            "this is silu(up) * gate, not silu(gate) * up: {got:?}"
        );
    }

    #[test]
    fn a_fused_tensor_of_the_wrong_width_is_rejected() {
        let dev = Device::Cpu;
        let gate_up = Tensor::zeros((2, 6, 4), DType::F32, &dev).unwrap();
        let down = Tensor::zeros((2, 4, 3), DType::F32, &dev).unwrap();
        assert!(split_fused_experts(&gate_up, &down, 3).is_ok());
        // 2 x 4 != 6
        assert!(split_fused_experts(&gate_up, &down, 4).is_err());
        // expert counts must agree
        let down_short = Tensor::zeros((1, 4, 3), DType::F32, &dev).unwrap();
        assert!(split_fused_experts(&gate_up, &down_short, 3).is_err());
    }
}
