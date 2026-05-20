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
    pub fn load(cfg: &TextConfig, vb: &ShardedVarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let gate_proj = load_linear_no_bias(vb, "gate_proj", h, i)?;
        let up_proj = load_linear_no_bias(vb, "up_proj", h, i)?;
        let down_proj = load_linear_no_bias(vb, "down_proj", i, h)?;
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
