//! SwiGLU MLP block for Qwen3-Next.
//!
//! Identical to plain Qwen3's MLP: `down(silu(gate(x)) * up(x))` with
//! no bias on any of the three projections.

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::var_builder::ShardedVarBuilder;

use super::TextConfig;
use super::proj::{LinearSource, Proj, load_proj};

pub struct Qwen3_5MLP {
    gate_proj: Proj,
    up_proj: Proj,
    down_proj: Proj,
}

impl Qwen3_5MLP {
    /// Construct directly from pre-built projections.
    ///
    /// Test-only again: `qwen4_exp` briefly built its 512 experts this
    /// way before the fused tensors stayed banked (`Experts::Banked`),
    /// which removed the last caller outside tests. The differential
    /// test that proves banking neutral still needs it, to express the
    /// same weights in the layout it is being compared against.
    #[cfg(test)]
    pub(crate) fn from_weights(
        gate_proj: candle_nn::Linear,
        up_proj: candle_nn::Linear,
        down_proj: candle_nn::Linear,
    ) -> Self {
        Self {
            gate_proj: gate_proj.into(),
            up_proj: up_proj.into(),
            down_proj: down_proj.into(),
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
        Self::load_with_source(vb, hidden, intermediate, None)
    }

    /// As [`Self::load_with_dims`], taking projections from `source`
    /// where it has them.
    pub fn load_with_source(
        vb: &ShardedVarBuilder,
        hidden: usize,
        intermediate: usize,
        source: Option<&dyn LinearSource>,
    ) -> Result<Self> {
        let gate_proj = load_proj(vb, "gate_proj", hidden, intermediate, source)?;
        let up_proj = load_proj(vb, "up_proj", hidden, intermediate, source)?;
        let down_proj = load_proj(vb, "down_proj", intermediate, hidden, source)?;
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
