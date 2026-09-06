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

/// EMA smoothing factor for a throughput sample. Low enough that one
/// anomalous turn (a contended GPU, a cold cache) doesn't swing the
/// advertised limit / published rate, high enough to track a real shift
/// (e.g. prefix caching, #11, dropping effective prefill cost) within a
/// few turns.
const RATE_EMA_ALPHA: f64 = 0.3;

/// Smallest prompt whose prefill time says anything about throughput.
///
/// A prefill has a fixed per-request cost — tokenize, launch, sync —
/// that dominates a short prompt completely. The deploy's own smoke
/// probe prefills 19 tokens in ~130 ms and reads as **145 tok/s** on a
/// host that measures ~1,400. Folding that in does not make the
/// estimate slightly worse; it replaces it.
///
/// That mattered because the fallback it displaces is the whole
/// cold-start guard: with no sample at all, `derive_limit` uses
/// `bootstrap_prefill_tok_per_sec` (800, ≈96k context at the default
/// 120 s target). One 19-token probe replaced it with 145 tok/s, and
/// the host advertised **17,408** — a sixth of what it can serve —
/// until real traffic happened to warm it (#295).
///
/// 512 excludes the smoke probe and the short auxiliary calls agent
/// harnesses make (observed at 19, 143 and 469 tokens) while admitting
/// ordinary prompts. At ~1,400 tok/s a 512-token prefill runs ~370 ms,
/// so fixed overhead is roughly a tenth of it — informative rather than
/// dominant.
const MIN_PREFILL_SAMPLE_TOKENS: usize = 512;

/// Fold one throughput sample (`tokens` in `elapsed`) into the EMA held in
/// `bits`. No-op for degenerate inputs so a probe request or a clock blip
/// can't poison the average.
fn fold_rate(bits: &AtomicU64, tokens: usize, elapsed: Duration) {
    let secs = elapsed.as_secs_f64();
    if tokens == 0 || secs <= 0.0 {
        return;
    }
    let sample = tokens as f64 / secs;
    if !sample.is_finite() || sample <= 0.0 {
        return;
    }
    let prev = f64::from_bits(bits.load(Ordering::Acquire));
    let next = if prev > 0.0 {
        RATE_EMA_ALPHA * sample + (1.0 - RATE_EMA_ALPHA) * prev
    } else {
        sample
    };
    bits.store(next.to_bits(), Ordering::Release);
}

/// Read a rate EMA, or `None` before the first sample lands.
fn read_rate(bits: &AtomicU64) -> Option<f64> {
    let v = f64::from_bits(bits.load(Ordering::Acquire));
    (v.is_finite() && v > 0.0).then_some(v)
}

/// Self-measured throughput for one loaded model, as exponential moving
/// averages of tokens/sec. Tracks the two phases the client can't tell
/// apart from chunk-arrival timing:
///
/// - **prefill** (#67) — prompt tokens/sec, read when deriving the context
///   throughput ceiling;
/// - **decode** (#137) — generation tokens/sec, the live throughput number
///   cortex publishes for capacity planning.
///
/// Updated at the end of each request's respective phase, read by the
/// context-limit deriver and by `/health`. Lock-free: each phase is
/// serialised per model and readers only need a recent value. Each rate is
/// stored as raw f64 bits; `0` means "no sample yet".
///
/// The [`PrefillRateEma`] alias preserves the pre-#137 name at the many
/// prefill call sites; the type now carries decode too.
#[derive(Debug)]
pub struct ThroughputEma {
    prefill_bits: AtomicU64,
    decode_bits: AtomicU64,
}

/// Legacy name for [`ThroughputEma`] — kept so the prefill call sites
/// (context-limit derivation, load paths) read unchanged.
pub type PrefillRateEma = ThroughputEma;

impl ThroughputEma {
    pub const fn new() -> Self {
        Self {
            prefill_bits: AtomicU64::new(0),
            decode_bits: AtomicU64::new(0),
        }
    }

    /// Fold one prefill measurement (`prompt_tokens` processed in
    /// `elapsed`) into the prefill EMA.
    ///
    /// Samples below [`MIN_PREFILL_SAMPLE_TOKENS`] are ignored: they
    /// measure fixed per-request cost rather than throughput, and the
    /// value they displace — the bootstrap estimate — is better than
    /// they are.
    pub fn record(&self, prompt_tokens: usize, elapsed: Duration) {
        if prompt_tokens < MIN_PREFILL_SAMPLE_TOKENS {
            return;
        }
        fold_rate(&self.prefill_bits, prompt_tokens, elapsed);
    }

    /// The current prefill rate (tokens/sec), or `None` before the first
    /// sample.
    pub fn get(&self) -> Option<f64> {
        read_rate(&self.prefill_bits)
    }

    /// Fold one decode measurement (`completion_tokens` generated in
    /// `elapsed`) into the decode EMA (#137).
    pub fn record_decode(&self, completion_tokens: usize, elapsed: Duration) {
        fold_rate(&self.decode_bits, completion_tokens, elapsed);
    }

    /// The current decode rate (tokens/sec), or `None` before the first
    /// sample (#137).
    pub fn decode(&self) -> Option<f64> {
        read_rate(&self.decode_bits)
    }
}

impl Default for ThroughputEma {
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
/// `2 ×` accounts for K and V. Shared by the limit derivation here, the
/// per-rank load-time logging in the TP paths, and #65's request-time
/// length-aware pre-flight guard (`candle::validate_request`).
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
    if model_type != super::arch::qwen3_5::MODEL_TYPE
        && model_type != super::arch::qwen3_5::MODEL_TYPE_NEXT
    {
        return None;
    }
    // `from_config_json` normalises both layouts (nested qwen3_5, flat
    // qwen3_next) — a plain serde parse would reject the flat family,
    // which is how Coder-Next went without an advertised limit (#126).
    let cfg = super::arch::qwen3_5::Config::from_config_json(&text).ok()?;
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

/// Default ceiling on what a request may name as its output budget.
///
/// Not derived from anything — a deliberate figure, and configurable
/// per host (`[context_limit] max_output_tokens`). The evidence behind
/// it: a reasoning model asked to build a small application front-loads
/// one long think block before it acts, measured at 27,504 tokens for a
/// frontier model on a task Qwen3.8-27B was cut off attempting at
/// 24,433 (#271, #278). A ceiling below that turns a capable model into
/// one that never reaches its first tool call, so the default sits
/// above the observed cost of that class of turn.
const DEFAULT_OUTPUT_CEILING: usize = 32_768;

/// Derive `limit{context,input,output}` for a loaded model.
///
/// ```text
/// output         = output_reserve_tokens          (default + KV planning)
/// output_ceiling = min(max_output_tokens, max_position_embeddings)
/// vram_ceiling       = (free_tightest − activation_headroom − min_free_floor) / kv_bytes_per_token_per_card
/// throughput_ceiling = target_prefill_latency_secs × prefill_tok_per_sec
/// kv_ceiling         = kv_budget_mb × 1 MiB / kv_bytes_per_token_per_card
/// context = min(max_position_embeddings, vram_ceiling, throughput_ceiling, kv_ceiling) [clamped by `hard_ceiling` if set]
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
/// `kv_budget_mb` is what admission will actually hand out
/// ([`kv_reservation_mb`](super::candle::kv_reservation_mb) prices every
/// request against it). Without this term the advertisement and the
/// admission gate answer "how long a prompt fits" independently, and
/// they disagree: measured on beast 2026-08-30, a 3689 MiB budget holds
/// 118,048 tokens at 32 KiB/token/card while the advertised context was
/// the model's full 131,072 — so a client that believed the
/// advertisement filled to it and got a hard `413
/// prompt_too_long_for_vram` with no compaction path. `0` means the
/// budget is not published yet (pre-load, or CPU) → no clamp, matching
/// the `free_tightest_mb == 0` sentinel above.
///
/// Note this bounds the *whole* sequence. A caller that intends to
/// request `output_ceiling` output must plan against `context − its own
/// declared output`, not against `input`: `input` is `context − output`
/// for the default reserve, which is the only output size it can know.
///
/// `reasoning`: `input = context − output` keeps a generation reserve
/// below the wall; `output` (the reserve) is a *sub-budget* of context,
/// matching opencode's compaction model.
pub fn derive_limit(
    profile: &ContextProfile,
    free_tightest_mb: u64,
    prefill_tok_per_sec: f64,
    hard_ceiling: Option<usize>,
    kv_budget_mb: u64,
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

    // Admission ceiling — what the KV gate will actually grant. Zero is
    // the "not published yet" sentinel (pre-load, CPU) and imposes no
    // ceiling, like `free_tightest_mb` above.
    let kv_ceiling = if kv_budget_mb == 0 || profile.kv_bytes_per_token_per_card == 0 {
        usize::MAX
    } else {
        (kv_budget_mb.saturating_mul(1024 * 1024) / profile.kv_bytes_per_token_per_card) as usize
    };

    let mut context = profile
        .max_position_embeddings
        .min(vram_ceiling)
        .min(throughput_ceiling)
        .min(kv_ceiling);
    if let Some(clamp) = hard_ceiling {
        context = context.min(clamp);
    }
    context = round_down(context, CONTEXT_GRANULARITY);

    // Observability (#126): every input and intermediate term, so a
    // surprising advertised limit is diagnosable from the journal
    // instead of re-deriving by hand.
    //
    // TRACE, not debug: this runs on every `GET /models` poll and every
    // input is fixed at load, so the line is byte-identical each time —
    // about 8,600 a day per model at a 10s poll. It is worth having when
    // reasoning about a context ceiling and worth nothing the rest of
    // the time, and at debug it buries the lines that are not routine
    // (#320).
    tracing::trace!(
        max_pos = profile.max_position_embeddings,
        kv_bytes_per_token_per_card = profile.kv_bytes_per_token_per_card,
        free_tightest_mb,
        prefill_tok_per_sec = tok_per_sec,
        vram_ceiling,
        throughput_ceiling,
        kv_budget_mb,
        kv_ceiling,
        ?hard_ceiling,
        context,
        "derive_limit"
    );

    let input = context.saturating_sub(output);
    // The ceiling a request may name, distinct from the reserve above
    // (#278). Bounded by the model's own position limit rather than by
    // the live context, because the advertisement is read once and
    // cached by clients: a ceiling that shrank with fleet VRAM would be
    // stale in whichever direction hurt — either rejecting what we said
    // we would serve, or leaving budget unclaimed. A request that will
    // not actually fit is refused at request time, with the numbers,
    // instead of being under-promised here.
    let output_ceiling = round_down(
        cfg.max_output_tokens
            .unwrap_or(DEFAULT_OUTPUT_CEILING)
            .min(profile.max_position_embeddings),
        CONTEXT_GRANULARITY,
    )
    .max(output);
    ModelLimit {
        context,
        input: Some(input),
        output,
        output_ceiling,
    }
}

#[cfg(test)]
mod tests {

    /// The deploy's smoke probe must not set the advertised context.
    ///
    /// 19 tokens in 130 ms reads as ~145 tok/s on a host that measures
    /// ~1,400. Before this guard that one sample displaced the
    /// bootstrap estimate and the host advertised 17,408 instead of
    /// ~96,000 — until real traffic happened along (#295).
    #[test]
    fn a_smoke_probe_does_not_poison_the_prefill_rate() {
        let ema = ThroughputEma::new();
        ema.record(19, Duration::from_millis(130));
        assert_eq!(
            ema.get(),
            None,
            "a 19-token prefill must leave the EMA unset so the bootstrap value stands"
        );
    }

    /// The short auxiliary calls agent harnesses make are the same
    /// shape as the probe and equally uninformative — observed at 143
    /// and 469 tokens against prompts three orders of magnitude larger.
    #[test]
    fn short_auxiliary_prompts_are_ignored_too() {
        let ema = ThroughputEma::new();
        for tokens in [143usize, 469] {
            ema.record(tokens, Duration::from_millis(200));
        }
        assert_eq!(ema.get(), None);
    }

    /// A real prompt still records, and still tracks a shift.
    #[test]
    fn representative_prefills_still_measure() {
        let ema = ThroughputEma::new();
        ema.record(1400, Duration::from_secs(1));
        let first = ema.get().expect("a 1400-token sample must record");
        assert!((first - 1400.0).abs() < 1.0, "got {first}");

        // A slower host is tracked, not ignored.
        for _ in 0..20 {
            ema.record(700, Duration::from_secs(1));
        }
        let after = ema.get().unwrap();
        assert!(after < first && after > 600.0, "got {after}");
    }

    /// Exactly at the threshold counts — the boundary is inclusive so
    /// the constant means what it says.
    #[test]
    fn the_threshold_itself_is_a_valid_sample() {
        let ema = ThroughputEma::new();
        ema.record(MIN_PREFILL_SAMPLE_TOKENS, Duration::from_secs(1));
        assert!(ema.get().is_some());
    }
    use super::*;

    /// The ceiling a client is told to plan against must not be the
    /// reserve. Advertising the reserve is what told a reasoning model's
    /// harness to cap itself below the cost of one think block (#278).
    #[test]
    fn the_advertised_ceiling_is_not_the_reserve() {
        let cfg = ContextLimitConfig::default();
        let profile = ContextProfile {
            max_position_embeddings: 131_072,
            kv_bytes_per_token_per_card: 65_536,
            world_size: 2,
        };
        let limit = derive_limit(&profile, 0, 0.0, None, 0, &cfg);
        assert_eq!(limit.output, cfg.output_reserve_tokens, "reserve unchanged");
        assert_eq!(limit.output_ceiling, DEFAULT_OUTPUT_CEILING);
        assert!(limit.output_ceiling > limit.output);
    }

    /// A model whose own position limit is below the default ceiling
    /// cannot generate past it, so the advertisement must not claim it.
    #[test]
    fn the_ceiling_never_exceeds_the_models_own_position_limit() {
        let cfg = ContextLimitConfig::default();
        let profile = ContextProfile {
            max_position_embeddings: 16_384,
            kv_bytes_per_token_per_card: 65_536,
            world_size: 1,
        };
        let limit = derive_limit(&profile, 0, 0.0, None, 0, &cfg);
        assert_eq!(limit.output_ceiling, 16_384);
    }

    /// The advertisement may not promise a context the KV gate will
    /// refuse. Numbers are the live beast measurement (2026-08-30): a
    /// 3689 MiB budget at 32 KiB/token/card holds 118,048 tokens, while
    /// the model's own position limit is 131,072. Before this clamp the
    /// node advertised 131,072, pi filled a session to 90,432 prompt
    /// tokens believing it, and the turn died on a hard 413.
    #[test]
    fn the_context_never_exceeds_what_admission_will_grant() {
        let cfg = ContextLimitConfig::default();
        let profile = ContextProfile {
            max_position_embeddings: 131_072,
            kv_bytes_per_token_per_card: 32_768,
            world_size: 2,
        };
        // Roomy VRAM and a fast card, so only the KV budget can bind.
        let limit = derive_limit(&profile, 60_000, 100_000.0, None, 3689, &cfg);
        let context = limit.context;
        assert!(
            context <= 118_048,
            "advertised {context} tokens against a budget holding 118,048"
        );
        assert_eq!(context, 117_760, "118,048 rounded down to the 1024 grid");

        // A caller planning against the advertised ceiling must fit too:
        // it subtracts its own declared output from `context`.
        let ceiling = limit.output_ceiling;
        assert!(context.saturating_sub(ceiling) <= 118_048 - ceiling);
    }

    /// Zero is "not published yet" (pre-load, or a CPU load), not "no
    /// room" — the same sentinel `free_tightest_mb` uses. Collapsing the
    /// limit to zero there would unadvertise every model on startup.
    #[test]
    fn an_unpublished_kv_budget_imposes_no_ceiling() {
        let cfg = ContextLimitConfig::default();
        let with_budget = derive_limit(&beast_profile(), 9254, 8500.0, None, 3689, &cfg);
        let without = derive_limit(&beast_profile(), 9254, 8500.0, None, 0, &cfg);
        assert!(
            without.context >= with_budget.context,
            "the sentinel must not bind tighter than a real budget"
        );
        assert!(without.context > 0);
    }

    /// The operator's number wins, which is what makes the default a
    /// starting point rather than a guess baked into the binary.
    #[test]
    fn a_configured_ceiling_overrides_the_default() {
        let cfg = ContextLimitConfig {
            max_output_tokens: Some(65_536),
            ..ContextLimitConfig::default()
        };
        let profile = ContextProfile {
            max_position_embeddings: 131_072,
            kv_bytes_per_token_per_card: 65_536,
            world_size: 2,
        };
        assert_eq!(
            derive_limit(&profile, 0, 0.0, None, 0, &cfg).output_ceiling,
            65_536
        );
    }

    /// The ceiling is deliberately *not* a function of live VRAM: it is
    /// read once and cached by clients, so it has to mean the same thing
    /// tomorrow. The live context still moves — that is what a request
    /// is judged against — but the contract does not.
    #[test]
    fn the_ceiling_holds_still_while_the_context_moves() {
        let cfg = ContextLimitConfig::default();
        let profile = ContextProfile {
            max_position_embeddings: 131_072,
            kv_bytes_per_token_per_card: 65_536,
            world_size: 2,
        };
        let roomy = derive_limit(&profile, 60_000, 5_000.0, None, 0, &cfg);
        // Tight enough that the VRAM term binds — the same shape as the
        // fleet advertising 109568 one hour and 131072 the next.
        let tight = derive_limit(&profile, 6_000, 5_000.0, None, 0, &cfg);
        assert!(
            tight.context < roomy.context,
            "the live window should shrink under VRAM pressure"
        );
        assert_eq!(
            tight.output_ceiling, roomy.output_ceiling,
            "the advertised ceiling must not move with fleet load"
        );
    }

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
        let limit = derive_limit(&beast_profile(), 9254, 850.0, None, 0, &cfg);
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
        let slow = derive_limit(&beast_profile(), 9254, 850.0, None, 0, &cfg);
        let fast = derive_limit(&beast_profile(), 9254, 8500.0, None, 0, &cfg);
        assert!(fast.context > slow.context);
    }

    #[test]
    fn tighter_vram_lowers_the_limit() {
        // Same model, less free VRAM → VRAM ceiling binds below throughput.
        let cfg = ContextLimitConfig::default();
        let roomy = derive_limit(&beast_profile(), 9254, 8500.0, None, 0, &cfg);
        let tight = derive_limit(&beast_profile(), 2600, 8500.0, None, 0, &cfg);
        assert!(tight.context < roomy.context);
    }

    #[test]
    fn hard_ceiling_clamps_only_downward() {
        let cfg = ContextLimitConfig::default();
        // A backstop below the derived value clamps it.
        let clamped = derive_limit(&beast_profile(), 9254, 8500.0, Some(49152), 0, &cfg);
        assert_eq!(clamped.context, 49152);
        // A backstop above the derived value is a no-op.
        let unclamped = derive_limit(&beast_profile(), 9254, 850.0, Some(200000), 0, &cfg);
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
    fn decode_ema_is_independent_of_prefill() {
        // #137: decode throughput tracks separately from prefill and both
        // ignore degenerate samples.
        let ema = ThroughputEma::new();
        assert_eq!(ema.decode(), None);
        ema.record_decode(50, Duration::from_secs(1));
        assert_eq!(ema.decode(), Some(50.0));
        // Recording prefill doesn't touch decode, and vice versa.
        ema.record(1000, Duration::from_secs(1));
        assert_eq!(ema.decode(), Some(50.0));
        assert_eq!(ema.get(), Some(1000.0));
        // Degenerate decode samples are ignored.
        ema.record_decode(0, Duration::from_secs(1));
        ema.record_decode(50, Duration::from_secs(0));
        assert_eq!(ema.decode(), Some(50.0));
        // Smoothing: 0.3*100 + 0.7*50 = 65.
        ema.record_decode(100, Duration::from_secs(1));
        assert!((ema.decode().unwrap() - 65.0).abs() < 1e-6);
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
        let limit = derive_limit(&profile, 8000, 8500.0, None, 0, &cfg);
        // max_position_embeddings (32768) binds below throughput (~1.02M).
        assert_eq!(limit.context, 32768);
    }
}
