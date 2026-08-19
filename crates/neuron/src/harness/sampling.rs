//! Sampling parameter resolution (#272).
//!
//! Three sources, in priority order:
//!
//! 1. what the **request** asked for,
//! 2. what the **model** published in `generation_config.json`,
//! 3. a built-in fallback.
//!
//! Before this, (2) did not exist and (3) was a hardcoded
//! `temperature = 0.7` with `top_p = None` — and `None` selects
//! `Sampling::All`, meaning no truncation at all. So a caller that
//! omitted sampling fields (the common case: the DeepSeek Harness sends
//! only `temperature`) got untruncated sampling across the full ~250k
//! vocabulary on a model whose authors published `top_k = 20,
//! top_p = 0.95, temperature = 1.0`.
//!
//! Picking "no truncation" for an absent field is not a neutral default,
//! it is the widest possible one. The neutral default is what the model
//! shipped with.
//!
//! Resolved **once** per request and threaded whole. The eight
//! `LogitsProcessor` construction sites this replaces could each have
//! disagreed about a default — the same shape of bug as #252, where two
//! config files described one model and drifted.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Fallback temperature when neither the request nor the model says.
/// Retained from the previous hardcoded value so a model without a
/// `generation_config.json` behaves exactly as it did before.
const FALLBACK_TEMPERATURE: f64 = 0.7;

/// Repetition penalty applied to recently-generated tokens before
/// sampling. 1.0 disables it; >1.0 makes recently-emitted tokens less
/// likely. mistral.rs and llama.cpp default to 1.1, which is enough to
/// stop small quantized models from degenerating into "Wait, no, no..."
/// loops without distorting normal output.
pub const DEFAULT_REPEAT_PENALTY: f32 = 1.1;

/// Number of recently-generated tokens fed into the repetition penalty.
/// Matches the candle quantized-qwen3 example default.
///
/// Defensible for chat and poorly suited to a 24k-token reasoning block,
/// where a derivation repeated 500 tokens later is well outside the
/// window — which is why it is now a per-request knob rather than a
/// constant.
pub const DEFAULT_REPEAT_LAST_N: usize = 64;

/// What a model published in its `generation_config.json`.
///
/// Every field optional: the file is a HuggingFace convention, not a
/// schema, and models populate different subsets of it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelGenerationDefaults {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
}

impl ModelGenerationDefaults {
    /// Read `generation_config.json` from a model snapshot directory.
    ///
    /// Absent or unparseable is not an error — we fall back exactly as
    /// before — but it is logged, because silently guessing the model's
    /// sampling is how the untruncated default went unnoticed.
    pub fn load_from_dir(dir: &Path) -> Self {
        let path = dir.join("generation_config.json");
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                tracing::debug!(
                    path = %path.display(),
                    error = %e,
                    "sampling: no generation_config.json; using built-in fallbacks"
                );
                return Self::default();
            }
        };
        match serde_json::from_str::<Self>(&text) {
            Ok(d) => {
                tracing::info!(
                    path = %path.display(),
                    temperature = ?d.temperature,
                    top_p = ?d.top_p,
                    top_k = ?d.top_k,
                    "sampling: model generation defaults loaded"
                );
                d
            }
            Err(e) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "sampling: generation_config.json did not parse; using built-in fallbacks"
                );
                Self::default()
            }
        }
    }
}

/// What the caller asked for. All optional — absent means "no opinion",
/// which defers to the model's published value.
#[derive(Debug, Clone, Default)]
pub struct RequestedSampling {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub seed: Option<u64>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<usize>,
}

/// Fully-resolved sampling for one request.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl SamplingParams {
    /// Resolve request over model defaults over fallback.
    ///
    /// `random_seed` is the per-request seed used when the caller did
    /// not pin one; it is passed in rather than generated here so this
    /// stays a pure function and the "same seed reproduces" property is
    /// testable without a clock.
    pub fn resolve(
        requested: &RequestedSampling,
        model: &ModelGenerationDefaults,
        random_seed: u64,
    ) -> Self {
        Self {
            temperature: requested
                .temperature
                .or(model.temperature)
                .unwrap_or(FALLBACK_TEMPERATURE),
            // No `unwrap_or` here on purpose: `None` still means "do not
            // truncate", but it is now only reachable when neither the
            // caller nor the model expressed a preference, rather than
            // whenever the caller stayed quiet.
            top_p: requested.top_p.or(model.top_p),
            top_k: requested.top_k.or(model.top_k),
            seed: requested.seed.unwrap_or(random_seed),
            repeat_penalty: requested.repeat_penalty.unwrap_or(DEFAULT_REPEAT_PENALTY),
            repeat_last_n: requested.repeat_last_n.unwrap_or(DEFAULT_REPEAT_LAST_N),
        }
    }

    /// The candle sampling strategy these parameters select.
    ///
    /// `temperature <= 0` is greedy regardless of the rest — a caller
    /// asking for determinism that way should not have it silently
    /// softened by a `top_k` the model happened to publish.
    pub fn to_sampling(&self) -> candle_transformers::generation::Sampling {
        use candle_transformers::generation::Sampling;
        if self.temperature <= 0.0 {
            return Sampling::ArgMax;
        }
        let temperature = self.temperature;
        match (self.top_k, self.top_p) {
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            (Some(k), None) => Sampling::TopK { k, temperature },
            (None, Some(p)) => Sampling::TopP { p, temperature },
            (None, None) => Sampling::All { temperature },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_transformers::generation::Sampling;

    fn qwen3_defaults() -> ModelGenerationDefaults {
        ModelGenerationDefaults {
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(20),
        }
    }

    /// The case that motivated #271: a caller that names no sampling
    /// parameters must get the model's published ones, not untruncated
    /// sampling at a hardcoded 0.7.
    #[test]
    fn a_silent_request_inherits_the_models_published_sampling() {
        let p = SamplingParams::resolve(&RequestedSampling::default(), &qwen3_defaults(), 42);
        assert_eq!(p.temperature, 1.0);
        assert_eq!(p.top_p, Some(0.95));
        assert_eq!(p.top_k, Some(20));
        assert_eq!(
            p.to_sampling(),
            Sampling::TopKThenTopP {
                k: 20,
                p: 0.95,
                temperature: 1.0
            },
            "the combination Qwen3 asks for must be expressible"
        );
    }

    /// The caller still wins. A harness that deliberately sets a low
    /// temperature must not have it overridden by the model's file.
    #[test]
    fn the_request_overrides_the_model() {
        let requested = RequestedSampling {
            temperature: Some(0.2),
            top_k: Some(5),
            ..Default::default()
        };
        let p = SamplingParams::resolve(&requested, &qwen3_defaults(), 42);
        assert_eq!(p.temperature, 0.2);
        assert_eq!(p.top_k, Some(5));
        // Untouched fields still fall through to the model.
        assert_eq!(p.top_p, Some(0.95));
    }

    /// A model with no `generation_config.json` must behave exactly as
    /// it did before this landed, so nothing regresses on the models
    /// that do not ship one.
    #[test]
    fn without_model_defaults_the_old_behaviour_is_preserved() {
        let p = SamplingParams::resolve(
            &RequestedSampling::default(),
            &ModelGenerationDefaults::default(),
            42,
        );
        assert_eq!(p.temperature, FALLBACK_TEMPERATURE);
        assert_eq!(p.top_p, None);
        assert_eq!(p.top_k, None);
        assert_eq!(p.to_sampling(), Sampling::All { temperature: 0.7 });
    }

    /// A pinned seed is honoured; an absent one takes the per-request
    /// random value. Reproducibility is the whole point — the text paths
    /// previously called `unix_subsec_nanos()` unconditionally, so
    /// `seed` was accepted and discarded.
    #[test]
    fn seed_is_pinned_when_given_and_random_otherwise() {
        let pinned = SamplingParams::resolve(
            &RequestedSampling {
                seed: Some(7),
                ..Default::default()
            },
            &ModelGenerationDefaults::default(),
            42,
        );
        assert_eq!(pinned.seed, 7);

        let unpinned = SamplingParams::resolve(
            &RequestedSampling::default(),
            &ModelGenerationDefaults::default(),
            42,
        );
        assert_eq!(unpinned.seed, 42);
    }

    /// Greedy beats every truncation setting: a caller asking for
    /// `temperature = 0` wants determinism, not a softened version of it.
    #[test]
    fn zero_temperature_is_greedy_even_with_model_top_k() {
        let requested = RequestedSampling {
            temperature: Some(0.0),
            ..Default::default()
        };
        let p = SamplingParams::resolve(&requested, &qwen3_defaults(), 42);
        assert_eq!(p.to_sampling(), Sampling::ArgMax);
    }

    /// Each truncation knob alone selects the matching strategy.
    #[test]
    fn single_knobs_select_their_own_strategy() {
        let only_k = SamplingParams::resolve(
            &RequestedSampling {
                top_k: Some(20),
                ..Default::default()
            },
            &ModelGenerationDefaults::default(),
            1,
        );
        assert_eq!(
            only_k.to_sampling(),
            Sampling::TopK {
                k: 20,
                temperature: 0.7
            }
        );

        let only_p = SamplingParams::resolve(
            &RequestedSampling {
                top_p: Some(0.9),
                ..Default::default()
            },
            &ModelGenerationDefaults::default(),
            1,
        );
        assert_eq!(
            only_p.to_sampling(),
            Sampling::TopP {
                p: 0.9,
                temperature: 0.7
            }
        );
    }

    /// Repetition penalty and its window are now per-request, defaulting
    /// to the values that were previously constants.
    #[test]
    fn repetition_knobs_default_to_the_former_constants() {
        let d = SamplingParams::resolve(
            &RequestedSampling::default(),
            &ModelGenerationDefaults::default(),
            1,
        );
        assert_eq!(d.repeat_penalty, DEFAULT_REPEAT_PENALTY);
        assert_eq!(d.repeat_last_n, DEFAULT_REPEAT_LAST_N);

        let tuned = SamplingParams::resolve(
            &RequestedSampling {
                repeat_penalty: Some(1.05),
                repeat_last_n: Some(512),
                ..Default::default()
            },
            &ModelGenerationDefaults::default(),
            1,
        );
        assert_eq!(tuned.repeat_penalty, 1.05);
        assert_eq!(tuned.repeat_last_n, 512);
    }

    #[test]
    fn absent_generation_config_is_not_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let d = ModelGenerationDefaults::load_from_dir(dir.path());
        assert!(d.temperature.is_none() && d.top_p.is_none() && d.top_k.is_none());
    }

    #[test]
    fn a_real_generation_config_parses() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Verbatim from Qwen/Qwen3.8-27B, extra fields included — the
        // reader must tolerate the ones it does not model.
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"bos_token_id":248044,"do_sample":true,"eos_token_id":[248046,248044],
                "pad_token_id":248044,"temperature":1.0,"top_k":20,"top_p":0.95}"#,
        )
        .expect("write");
        let d = ModelGenerationDefaults::load_from_dir(dir.path());
        assert_eq!(d.temperature, Some(1.0));
        assert_eq!(d.top_p, Some(0.95));
        assert_eq!(d.top_k, Some(20));
    }

    #[test]
    fn malformed_generation_config_falls_back_rather_than_failing() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("generation_config.json"), "{not json").expect("write");
        let d = ModelGenerationDefaults::load_from_dir(dir.path());
        assert!(d.temperature.is_none());
    }
}
