//! SwiGLU MLP block for Qwen3-Next.
//!
//! Identical to plain Qwen3's MLP: `down(silu(gate(x)) * up(x))` with
//! no bias on any of the three projections.

use anyhow::{Context, Result};
use candle_core::{Module, Tensor};
use candle_nn::Linear;
use candle_nn::var_builder::ShardedVarBuilder;

use super::TextConfig;

pub struct Qwen3_5MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen3_5MLP {
    /// Construct directly from pre-built projections (MoE-block tests).
    #[cfg(test)]
    pub(crate) fn from_weights(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn load(cfg: &TextConfig, vb: &ShardedVarBuilder) -> Result<Self> {
        Self::load_with_dims(vb, cfg.hidden_size, cfg.intermediate_size)
    }

    /// Load with explicit dims — the MoE block (#92) reuses this SwiGLU
    /// shape for routed experts (`moe_intermediate_size`) and the shared
    /// expert (`shared_expert_intermediate_size`), both narrower than
    /// the dense `intermediate_size`.
    pub fn load_with_dims(
        vb: &ShardedVarBuilder,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self> {
        let gate_proj = load_linear_no_bias(vb, "gate_proj", hidden, intermediate)?;
        let up_proj = load_linear_no_bias(vb, "up_proj", hidden, intermediate)?;
        let down_proj = load_linear_no_bias(vb, "down_proj", intermediate, hidden)?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for Qwen3_5MLP {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let rhs = self.up_proj.forward(x)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

fn load_linear_no_bias(
    vb: &ShardedVarBuilder,
    name: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<Linear> {
    let weight = vb
        .pp(name)
        .get((out_dim, in_dim), "weight")
        .with_context(|| format!("load '{}/{name}/weight'", vb.prefix()))?;
    Ok(Linear::new(weight, None))
}
