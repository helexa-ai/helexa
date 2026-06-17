//! Self-derived context/token limits (#67).
//!
//! The correct `limit{context,input,output}` for a deployment is not a
//! static fact an operator should memorise — it's a computed function of
//! things the neuron already knows better than any operator:
//!
//! - **model architecture** — `max_position_embeddings` and the
//!   KV-cost-per-token implied by the attention layout;
//! - **live free VRAM** on the tightest card the model occupies, after
//!   weights and an activation reserve;
//! - the **coherence/throughput trade-off** — "biggest that fits VRAM"
//!   is not "biggest that's usable": with no cross-request KV reuse every
//!   turn re-prefills the whole context, so there's a usable ceiling
//!   below the VRAM ceiling (it rises as prefix caching / #11 lands).
//!
//! This module is the arch-agnostic physics + policy. Each arch's load
//! path builds a [`ContextProfile`] (the physics) via
//! [`kv_bytes_per_token`]; [`derive_limit`] applies the policy against
//! live VRAM + a self-measured prefill rate + [`ContextLimitConfig`].
//! qwen3_5 is the only arch wired today; a future standard
//! full-attention model is the simpler case (`n_full_attn_layers =
//! n_layers`) and drops in by constructing a `ContextProfile`.

use cortex_core::harness::ModelLimit;

use crate::config::ContextLimitConfig;

/// Bytes of KV cache one token adds **per card**, counting only the
/// full-attention layers (linear/recurrent layers carry fixed-size
/// state, not a growing cache). Sharded across the TP world: per-rank
/// KV-head count is `n_kv_heads / world_size`.
///
/// `2 ×` accounts for K and V. Shared by the limit derivation here and
/// the per-rank load-time logging in the TP paths (and, in future, by
/// #65's length-aware pre-flight guard).
pub fn kv_bytes_per_token(
    n_full_attn_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    dtype_bytes: usize,
    world_size: u32,
) -> u64 {
    let per_rank_kv_heads = (n_kv_heads / world_size.max(1) as usize).max(1);
    (2 * n_full_attn_layers * per_rank_kv_heads * head_dim * dtype_bytes) as u64
}

/// Per-model physics needed to derive a context limit, captured at load
/// time (the arch config is consumed during model construction, so the
/// relevant numbers are snapshotted into this struct). Arch-agnostic:
/// the hybrid qwen3_5 case counts only its full-attention layers; a
/// standard transformer would pass `n_full_attn_layers = n_layers`.
#[derive(Debug, Clone, Copy)]
pub struct ContextProfile {
    /// The model's native context ceiling (quality wall).
    pub max_position_embeddings: usize,
    /// KV bytes added per token, per card — from [`kv_bytes_per_token`].
    pub kv_bytes_per_token_per_card: u64,
    /// Tensor-parallel world size the model is loaded with (1 = single GPU).
    pub world_size: u32,
}

/// Round a token count down to a clean boundary so the advertised limit
/// doesn't jitter by a handful of tokens as live VRAM / the throughput
/// EMA wobble between polls.
fn round_down(tokens: usize, granularity: usize) -> usize {
    if granularity == 0 {
        return tokens;
    }
    (tokens / granularity) * granularity
}

const CONTEXT_GRANULARITY: usize = 1024;

/// Derive `limit{context,input,output}` for a loaded model.
///
/// ```text
/// output  = output_reserve_tokens
/// vram_ceiling       = (free_tightest − activation_headroom − min_free_floor) / kv_bytes_per_token_per_card
/// throughput_ceiling = target_prefill_latency_secs × prefill_tok_per_sec
/// context = min(max_position_embeddings, vram_ceiling, throughput_ceiling) [clamped by `hard_ceiling` if set]
/// input   = context − output
/// ```
///
/// `free_tightest_mb` is the minimum free VRAM (MiB) across the model's
/// devices — the tightest card, which on a TP model is often a
/// non-leader rank. `prefill_tok_per_sec` is the model's self-measured
/// prefill rate (or a bootstrap estimate before the first sample).
/// `hard_ceiling` is an optional clamp-only backstop
/// (`NEURON_MAX_PROMPT_TOKENS` or a catalogue override); `None` = no clamp.
///
/// `reasoning`: `input = context − output` keeps a generation reserve
/// below the wall; `output` (the reserve) is a *sub-budget* of context,
/// matching opencode's compaction model.
pub fn derive_limit(
    profile: &ContextProfile,
    free_tightest_mb: u64,
    prefill_tok_per_sec: f64,
    hard_ceiling: Option<usize>,
    cfg: &ContextLimitConfig,
) -> ModelLimit {
    let output = cfg.output_reserve_tokens;

    // VRAM ceiling — what actually fits, from live free VRAM.
    let reserved_mb = cfg
        .activation_headroom_mb
        .saturating_add(cfg.min_free_floor_mb);
    let avail_bytes = free_tightest_mb
        .saturating_sub(reserved_mb)
        .saturating_mul(1024 * 1024);
    // `checked_div` yields `None` for a degenerate zero-KV profile (e.g.
    // no full-attention layers) → VRAM imposes no ceiling, the other
    // terms bind.
    let vram_ceiling = avail_bytes
        .checked_div(profile.kv_bytes_per_token_per_card)
        .map_or(usize::MAX, |t| t as usize);

    // Throughput ceiling — usable, not just fittable. Fall back to the
    // bootstrap estimate until the model has measured its own rate.
    let tok_per_sec = if prefill_tok_per_sec.is_finite() && prefill_tok_per_sec > 0.0 {
        prefill_tok_per_sec
    } else {
        cfg.bootstrap_prefill_tok_per_sec
    };
    let throughput_ceiling = (cfg.target_prefill_latency_secs * tok_per_sec).max(0.0) as usize;

    let mut context = profile
        .max_position_embeddings
        .min(vram_ceiling)
        .min(throughput_ceiling);
    if let Some(clamp) = hard_ceiling {
        context = context.min(clamp);
    }
    context = round_down(context, CONTEXT_GRANULARITY);

    let input = context.saturating_sub(output);
    ModelLimit {
        context,
        input: Some(input),
        output,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// beast Qwen3.6-27B: 16 full-attn layers, 4 kv heads, head_dim 256,
    /// f16 (2 B), TP=2 → 64 KiB/token total, 32 KiB/token/card.
    fn beast_profile() -> ContextProfile {
        let kv = kv_bytes_per_token(16, 4, 256, 2, 2);
        ContextProfile {
            max_position_embeddings: 262144,
            kv_bytes_per_token_per_card: kv,
            world_size: 2,
        }
    }

    #[test]
    fn kv_bytes_matches_hand_derivation() {
        // 2 × 16 × (4/2) × 256 × 2 = 32 KiB per card.
        assert_eq!(kv_bytes_per_token(16, 4, 256, 2, 2), 32 * 1024);
        // Single-GPU (world=1) doubles the per-card cost: 64 KiB.
        assert_eq!(kv_bytes_per_token(16, 4, 256, 2, 1), 64 * 1024);
    }

    #[test]
    fn throughput_ceiling_binds_pre_prefix_cache() {
        // ~850 tok/s × 120 s ≈ 102k → the coherence wall binds below the
        // VRAM ceiling on beast pre-#11. VRAM (~9.2 GB free) allows far
        // more, max_position_embeddings is 262144, so throughput wins.
        let cfg = ContextLimitConfig::default();
        let limit = derive_limit(&beast_profile(), 9254, 850.0, None, &cfg);
        // 120 × 850 = 102000 → rounded down to 1024 → 101376.
        assert_eq!(limit.context, 101376);
        assert_eq!(limit.output, 8192);
        assert_eq!(limit.input, Some(101376 - 8192));
        assert!(limit.input.unwrap() < limit.context);
    }

    #[test]
    fn faster_prefill_raises_the_limit() {
        // Prefix caching (#11) speeds effective prefill → ceiling rises,
        // eventually pinned by VRAM / max_position_embeddings.
        let cfg = ContextLimitConfig::default();
        let slow = derive_limit(&beast_profile(), 9254, 850.0, None, &cfg);
        let fast = derive_limit(&beast_profile(), 9254, 8500.0, None, &cfg);
        assert!(fast.context > slow.context);
    }

    #[test]
    fn tighter_vram_lowers_the_limit() {
        // Same model, less free VRAM → VRAM ceiling binds below throughput.
        let cfg = ContextLimitConfig::default();
        let roomy = derive_limit(&beast_profile(), 9254, 8500.0, None, &cfg);
        let tight = derive_limit(&beast_profile(), 2600, 8500.0, None, &cfg);
        assert!(tight.context < roomy.context);
    }

    #[test]
    fn hard_ceiling_clamps_only_downward() {
        let cfg = ContextLimitConfig::default();
        // A backstop below the derived value clamps it.
        let clamped = derive_limit(&beast_profile(), 9254, 8500.0, Some(49152), &cfg);
        assert_eq!(clamped.context, 49152);
        // A backstop above the derived value is a no-op.
        let unclamped = derive_limit(&beast_profile(), 9254, 850.0, Some(200000), &cfg);
        assert_eq!(unclamped.context, 101376);
    }

    #[test]
    fn zero_kv_cost_falls_back_to_other_ceilings() {
        // A degenerate profile (no full-attn layers) must not divide by
        // zero — VRAM ceiling becomes unbounded, others still apply.
        let profile = ContextProfile {
            max_position_embeddings: 32768,
            kv_bytes_per_token_per_card: 0,
            world_size: 1,
        };
        let cfg = ContextLimitConfig::default();
        let limit = derive_limit(&profile, 8000, 8500.0, None, &cfg);
        // max_position_embeddings (32768) binds below throughput (~1.02M).
        assert_eq!(limit.context, 32768);
    }
}
