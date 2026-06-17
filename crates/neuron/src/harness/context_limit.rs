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

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use cortex_core::harness::ModelLimit;

use crate::config::ContextLimitConfig;

/// EMA smoothing factor for the prefill-rate sample. Low enough that one
/// anomalous turn (a contended GPU, a cold cache) doesn't swing the
/// advertised limit, high enough to track a real shift (e.g. prefix
/// caching, #11, dropping effective prefill cost) within a few turns.
const PREFILL_EMA_ALPHA: f64 = 0.3;

/// Self-measured prefill throughput for one loaded model, as an
/// exponential moving average of tokens/sec (#67). Updated at the end of
/// each streaming request's prefill phase, read when deriving the
/// throughput ceiling. Lock-free: prefill is serialised per model (the
/// `inference_lock`), and the limit reader only needs a recent value.
/// Stores the f64 rate as raw bits; `0` means "no sample yet" → callers
/// fall back to the configured bootstrap estimate.
#[derive(Debug)]
pub struct PrefillRateEma {
    bits: AtomicU64,
}

impl PrefillRateEma {
    pub const fn new() -> Self {
        Self {
            bits: AtomicU64::new(0),
        }
    }

    /// Fold one prefill measurement (`prompt_tokens` processed in
    /// `elapsed`) into the EMA. No-op for degenerate inputs so a probe
    /// request or a clock blip can't poison the average.
    pub fn record(&self, prompt_tokens: usize, elapsed: Duration) {
        let secs = elapsed.as_secs_f64();
        if prompt_tokens == 0 || secs <= 0.0 {
            return;
        }
        let sample = prompt_tokens as f64 / secs;
        if !sample.is_finite() || sample <= 0.0 {
            return;
        }
        let prev = f64::from_bits(self.bits.load(Ordering::Acquire));
        let next = if prev > 0.0 {
            PREFILL_EMA_ALPHA * sample + (1.0 - PREFILL_EMA_ALPHA) * prev
        } else {
            sample
        };
        self.bits.store(next.to_bits(), Ordering::Release);
    }

    /// The current measured rate (tokens/sec), or `None` before the
    /// first sample lands.
    pub fn get(&self) -> Option<f64> {
        let v = f64::from_bits(self.bits.load(Ordering::Acquire));
        (v.is_finite() && v > 0.0).then_some(v)
    }
}

impl Default for PrefillRateEma {
    fn default() -> Self {
        Self::new()
    }
}

/// Bytes per element of the KV cache. qwen3_5 keeps K/V in the model's
/// f16/bf16 compute dtype regardless of weight quantisation (ISQ
/// quantises weights, not the cache), so this is 2 for every supported
/// load. Matches the per-rank logging math in the TP load paths.
pub const KV_CACHE_DTYPE_BYTES: usize = 2;

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

/// Build a [`ContextProfile`] from a qwen3_5 `config.json` on disk
/// (mirrors `VisionMeta::from_config_path`). Returns `None` for any other
/// `model_type` or an unparseable config — those arches fall back to the
/// static prompt cap with no advertised limit. `world_size` is the TP
/// degree the model is loaded with (1 = single GPU).
///
/// KV grows only on full-attention layers; `layer_types` is authoritative
/// (every entry is `"full_attention"` or `"linear_attention"`), with the
/// `full_attention_interval` hint as a fallback when the array is absent.
pub fn profile_from_qwen3_5_config(config_path: &Path, world_size: u32) -> Option<ContextProfile> {
    let text = std::fs::read_to_string(config_path).ok()?;
    let model_type = serde_json::from_str::<serde_json::Value>(&text)
        .ok()?
        .get("model_type")?
        .as_str()?
        .to_owned();
    if model_type != super::arch::qwen3_5::MODEL_TYPE {
        return None;
    }
    let cfg: super::arch::qwen3_5::Config = serde_json::from_str(&text).ok()?;
    let tc = &cfg.text_config;
    let n_full_attn_layers = {
        let counted = tc
            .layer_types
            .iter()
            .filter(|t| t.as_str() == "full_attention")
            .count();
        if counted > 0 {
            counted
        } else {
            // layer_types absent — derive from the interval hint.
            let interval = tc.full_attention_interval.unwrap_or(4).max(1);
            tc.num_hidden_layers / interval
        }
    };
    let kv_bytes_per_token_per_card = kv_bytes_per_token(
        n_full_attn_layers,
        tc.num_key_value_heads,
        tc.head_dim,
        KV_CACHE_DTYPE_BYTES,
        world_size,
    );
    Some(ContextProfile {
        max_position_embeddings: tc.max_position_embeddings,
        kv_bytes_per_token_per_card,
        world_size,
    })
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

    // VRAM ceiling — what actually fits, from live free VRAM. A zero
    // `free_tightest_mb` is the "unknown / no-context sentinel" (CPU
    // build, or a failed per-rank query) → VRAM imposes no ceiling, the
    // other terms bind, rather than collapsing the limit to zero.
    let vram_ceiling = if free_tightest_mb == 0 {
        usize::MAX
    } else {
        let reserved_mb = cfg
            .activation_headroom_mb
            .saturating_add(cfg.min_free_floor_mb);
        let avail_bytes = free_tightest_mb
            .saturating_sub(reserved_mb)
            .saturating_mul(1024 * 1024);
        // `checked_div` yields `None` for a degenerate zero-KV profile
        // (e.g. no full-attention layers) → VRAM imposes no ceiling.
        avail_bytes
            .checked_div(profile.kv_bytes_per_token_per_card)
            .map_or(usize::MAX, |t| t as usize)
    };

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
    fn prefill_ema_tracks_and_ignores_degenerate_samples() {
        let ema = PrefillRateEma::new();
        assert_eq!(ema.get(), None);
        // First real sample seeds the average exactly.
        ema.record(1000, Duration::from_secs(1));
        assert_eq!(ema.get(), Some(1000.0));
        // Degenerate inputs are ignored (no poisoning).
        ema.record(0, Duration::from_secs(1));
        ema.record(1000, Duration::from_secs(0));
        assert_eq!(ema.get(), Some(1000.0));
        // A faster sample pulls the EMA up but is smoothed (alpha 0.3):
        // 0.3*2000 + 0.7*1000 = 1300.
        ema.record(2000, Duration::from_secs(1));
        assert!((ema.get().unwrap() - 1300.0).abs() < 1e-6);
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
