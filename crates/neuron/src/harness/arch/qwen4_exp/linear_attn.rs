//! `qwen4_exp`'s linear-attention layer — 36 of 48.
//!
//! This is `qwen3_5`'s GatedDeltaNet with nothing changed. The tensor
//! names, the split (non-fused) `in_proj_*` layout, the delta rule, the
//! conv and the state shapes are all identical, which is why the
//! largest single piece of the Qwen3.6 port carries over untouched and
//! this module is a page rather than a file.
//!
//! What does not carry over is two settings, both of which our
//! defaults would get wrong in the same quiet way:
//!
//! 1. **`output_gate_type: sigmoid`.** Qwen3.6 leaves the field unset
//!    and the gated output norm falls back to SiLU. Inheriting that
//!    here applies the wrong nonlinearity on every one of these 36
//!    layers — fluent output, worse model.
//! 2. **`mamba_ssm_dtype: float32`.** We carry the recurrent state
//!    through a bf16 round-trip by default, deliberately (#284: the
//!    flag is off because the #283 measurements were taken against
//!    that baseline). This checkpoint states f32, and a port must not
//!    silently answer an upstream choice with our own. So the
//!    checkpoint wins here, and `NEURON_GDN_STATE_F32` remains the
//!    default only where the config is silent.
//!
//! See `doc/qwen4_exp-port-spec.md` §5.

use anyhow::Result;
use candle_nn::var_builder::ShardedVarBuilder;

use crate::harness::arch::qwen3_5::linear_attn::{
    GatedDeltaNet, GatedDeltaNetParams, gdn_state_f32,
};

use super::config::TextConfig;

/// `vb` should be `.pp(...)`-ed to the layer's `linear_attn` prefix.
pub fn load(cfg: &TextConfig, vb: &ShardedVarBuilder) -> Result<GatedDeltaNet> {
    GatedDeltaNet::load_with_params(params(cfg)?, vb)
}

/// The settings this architecture asks for, as opposed to the ones
/// `qwen3_5` would supply.
pub(crate) fn params(cfg: &TextConfig) -> Result<GatedDeltaNetParams> {
    Ok(GatedDeltaNetParams {
        hidden_size: cfg.hidden_size,
        num_value_heads: cfg.linear_num_value_heads,
        num_key_heads: cfg.linear_num_key_heads,
        key_head_dim: cfg.linear_key_head_dim,
        value_head_dim: cfg.linear_value_head_dim,
        conv_kernel_size: cfg.linear_conv_kernel_dim,
        rms_norm_eps: cfg.rms_norm_eps,
        output_gate: cfg.output_gate()?,
        // The checkpoint's choice where it makes one; ours only where
        // it is silent.
        state_f32: cfg.ssm_state_is_f32().unwrap_or_else(gdn_state_f32),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::arch::qwen3_5::rmsnorm::OutputGate;
    use crate::harness::arch::qwen4_exp::config::{Config, SHIPPED};

    fn shipped() -> Config {
        Config::from_config_json(SHIPPED).unwrap()
    }

    /// The layout `qwen3_5` already loads: 16 key heads and 48 value
    /// heads of 128, a conv of 4. If these ever diverge the reuse claim
    /// is void, so they are asserted rather than assumed.
    #[test]
    fn the_geometry_is_the_one_qwen3_5_already_loads() {
        let p = params(&shipped().text_config).unwrap();
        assert_eq!(p.hidden_size, 2560);
        assert_eq!(p.num_key_heads, 16);
        assert_eq!(p.num_value_heads, 48);
        assert_eq!(p.key_head_dim, 128);
        assert_eq!(p.value_head_dim, 128);
        assert_eq!(p.conv_kernel_size, 4);
        // conv_dim = 2 * 16 * 128 + 48 * 128 = 10240, the same width as
        // the four-stream residual, which is a coincidence worth not
        // reading anything into.
        assert_eq!(
            2 * p.num_key_heads * p.key_head_dim + p.num_value_heads * p.value_head_dim,
            10240
        );
    }

    /// The two deltas, together — this is the whole reason the module
    /// exists.
    #[test]
    fn the_checkpoint_overrides_both_of_our_defaults() {
        let p = params(&shipped().text_config).unwrap();
        assert_eq!(
            p.output_gate,
            OutputGate::Sigmoid,
            "inheriting hidden_act would give SiLU on all 36 layers"
        );
        assert!(
            p.state_f32,
            "mamba_ssm_dtype is float32; our #284 default is a bf16 round-trip"
        );
    }

    /// Where the checkpoint says nothing, our default stands rather
    /// than a guess — the fallback is the process-wide #284 flag, not a
    /// hardcoded answer.
    #[test]
    fn a_silent_checkpoint_leaves_our_default_in_place() {
        let json = SHIPPED.replace("\"mamba_ssm_dtype\": \"float32\",", "");
        let cfg = Config::from_config_json(&json).unwrap();
        assert_eq!(cfg.text_config.ssm_state_is_f32(), None);
        assert_eq!(params(&cfg.text_config).unwrap().state_f32, gdn_state_f32());
    }
}
