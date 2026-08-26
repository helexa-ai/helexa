//! Harness trait and supporting types for inference engine management.
//!
//! Defined in cortex-core so both cortex (control plane) and neuron
//! (node plane) share the type definitions. neuron provides the
//! runtime implementations.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Configuration for a harness instance on a neuron.
///
/// All current harnesses are in-process (candle); per-harness tuning
/// (cache paths, device policies, etc.) lives in dedicated config
/// blocks rather than on this struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessConfig {
    pub name: String,
}

/// Health status of a harness process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessHealth {
    pub name: String,
    pub running: bool,
    pub uptime_secs: Option<u64>,
}

/// Operator-set sampling defaults for one model (#283).
///
/// A model's `generation_config.json` is the model authors' statement of
/// intent, and honouring it (#272) is right by default. It is not always
/// right for the workload the operator serves: `Qwen/Qwen3.8-27B`
/// publishes `temperature = 1.0`, which measured a 20% structural-defect
/// rate on ~2k-token code generation against 0/60 at `<= 0.6`
/// (#283, p = 0.0031).
///
/// This is the operator's explicit, visible correction to that default —
/// deliberately *not* a heuristic that inspects the model and guesses.
/// A temperature guesser is a footgun; a number in a config file the
/// operator wrote is not.
///
/// Precedence is **request > operator > model > built-in fallback**: the
/// override replaces what the model published, but a caller that names a
/// value still wins, because an explicit API parameter that is silently
/// ignored is a contract break.
///
/// Every field is optional and overlays independently — setting
/// `temperature` alone leaves the model's `top_p`/`top_k` in force.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SamplingOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
}

impl SamplingOverride {
    /// True when the operator set nothing — an empty `[sampling]` table
    /// must behave exactly as no table at all.
    pub fn is_empty(&self) -> bool {
        self.temperature.is_none() && self.top_p.is_none() && self.top_k.is_none()
    }
}

/// Specification for loading a model through a harness.
///
/// Doubles as the `[[default_models]]` entry shape in `neuron.toml`, so
/// a field added here is available to both the load API and the
/// operator's per-host config without a second type to keep in step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub model_id: String,
    pub harness: String,
    pub quant: Option<String>,
    pub tensor_parallel: Option<u32>,
    pub devices: Option<Vec<u32>>,
    /// Operator sampling override (#283). Travels with the *model*, not
    /// the request: neuron is the only place sampling is resolved, so
    /// the override has to reach it whether the load came from cortex's
    /// catalogue or from this host's own `[[default_models]]`.
    ///
    /// `#[serde(default)]` so a cortex or neuron predating this field
    /// still round-trips the spec.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingOverride>,
    /// Whether prior assistant turns keep their reasoning when the chat
    /// template re-renders the conversation.
    ///
    /// This is the model's own `preserve_thinking` template control, not
    /// an invention of ours. `Qwen/Qwen3.8-27B` defaults it to *true* —
    /// the whole transcript's think blocks are replayed — and `false`
    /// keeps reasoning only for turns after the last user query, i.e.
    /// the turn in progress.
    ///
    /// `None` leaves the kwarg unset so the template's own default
    /// applies, which is the only safe default: the model's authors
    /// chose it, and overriding that fleet-wide on a hunch is how you
    /// ship a regression you cannot see.
    ///
    /// Exposed as operator config so the premise can be A/B'd on a real
    /// workload — full replay is the larger prompt and pushes against
    /// the prefix-cache budget and the throughput-derived context
    /// ceiling, but whether it *helps* the model is a measurement, not
    /// a belief. A request's own `chat_template_kwargs` still wins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preserve_thinking: Option<bool>,
}

/// Per-model token budget advertised by the catalogue or neuron.
///
/// `context` is the hard wall (the served max-seq-len).  `input` is the
/// compaction trigger — when set, opencode treats it as "usable context =
/// input − reserved".  When omitted, clients fall back to `context − output`.
/// `output` is the maximum number of generation tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLimit {
    /// Hard wall — served max-seq-len in tokens.
    pub context: usize,
    /// Compaction trigger / usable input budget.  When absent clients fall
    /// back to `context − output`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<usize>,
    /// The generation budget a request gets when it names none, and the
    /// figure `input = context − output` is derived from.
    ///
    /// A *reserve*, not a ceiling — a request may ask for more and be
    /// served it. Named `output` for the shape's history; the number a
    /// client should plan against is [`ModelLimit::output_ceiling`].
    pub output: usize,
    /// The largest output a request may name and be served (#278).
    ///
    /// Distinct from `output`, which is the default and the KV-planning
    /// reserve. Publishing the reserve as the ceiling meant a client
    /// that trusted the advertisement asked for a fraction of what the
    /// model would happily generate — on a reasoning model, often less
    /// than its own think block costs.
    ///
    /// `#[serde(default)]` so a neuron predating this field still
    /// deserializes; zero means "not advertised", and consumers fall
    /// back to `output`.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub output_ceiling: usize,
}

fn is_zero(v: &usize) -> bool {
    *v == 0
}

/// What each reasoning effort level costs, in reasoning tokens (#223).
///
/// OpenAI-shaped clients can only send `minimal|low|medium|high` — the
/// ladder is their entire vocabulary, and they cannot express a token
/// count. So the server picks the numbers, and has to say what they are:
/// a client choosing `low` with no idea whether that buys 2k tokens or
/// 20k is guessing, which is the failure #274 exists to end.
///
/// Ordered rungs rather than a map, so the ladder reads in the order a
/// client would climb it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningBudgetRung {
    /// The effort level as the caller spells it on the wire.
    ///
    /// Since #290 these are the rungs the **model's own chat template**
    /// accepts, discovered by rendering it, rather than a ladder the
    /// operator invented. `Qwen/Qwen3.8-27B` accepts exactly `low`,
    /// `medium` and `xhigh` and raises on anything else; we previously
    /// advertised `minimal`/`low`/`medium`/`high`, two of which the
    /// template rejects and which omitted the model's own default.
    ///
    /// Advertise only rungs the template accepts: an invented name is a
    /// name a caller can send and the model will reject. Note that a
    /// client need not read this at all — pi-ai takes its levels from a
    /// static `thinkingLevelMap` in the operator's config — so this is
    /// a statement of truth, not a control surface we can rely on.
    pub effort: String,
    /// Backstop reasoning-token cap for this rung, when the deployment
    /// sets one.
    ///
    /// `None` is the normal case and means "named rung, no cap": effort
    /// is expressed to the model through its template, and the number of
    /// tokens it then spends is the model's business. A cap is a safety
    /// net against a runaway think block (#223), not the mechanism by
    /// which effort is selected — enforcing effort by truncation is what
    /// #290 fixed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<usize>,
    /// Whether the model applies this rung when the caller names none.
    ///
    /// Taken from the template's own `|default(...)`. Without it a
    /// caller that says nothing cannot tell what it is getting — on
    /// Qwen3.8 that silently means `xhigh`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub default: bool,
}

/// Operator-set pricing, **USD per 1,000,000 tokens, as JSON numbers**
/// (`float`) — the models.dev/opencode `cost` convention, which is what
/// helexa's primary client reads. NOT per-token, NOT decimal strings (that
/// is OpenRouter's `pricing` shape, which helexa deliberately does not emit
/// — see #68). A client must not rescale by 10⁶.
///
/// `cost` is sourced from the operator's `models.toml` catalogue profile and
/// surfaced verbatim on `/v1/models`. The *absent* vs *zero* distinction is
/// intentional and load-bearing (#68):
/// - **`cost` absent** (the whole object omitted) — the model is **not
///   priced**: the operator has not declared a rate. Clients should treat
///   spend as unknown, not free.
/// - **`cost` present with `input`/`output` = `0.0`** — the model is
///   **intentionally free** (self-hosted, no charge). opencode renders `$0`.
///
/// Cache fields are optional — set them only when the backend supports a
/// prefix-cache discount tier (relevant once cache-token reporting, #64,
/// lands). The advertised rate here must equal the rate metering (#51) and
/// reconciliation (#58/#59) bill against; today both read this catalogue
/// value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCost {
    /// USD per 1M input (prompt) tokens.
    #[serde(default)]
    pub input: f64,
    /// USD per 1M output (completion) tokens.
    #[serde(default)]
    pub output: f64,
    /// USD per 1M cache-hit tokens (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<f64>,
    /// USD per 1M cache-write tokens (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<f64>,
}

/// A model as reported by a harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub harness: String,
    pub status: String,
    pub devices: Vec<u32>,
    pub vram_used_mb: Option<u64>,
    /// Modalities this loaded model supports. Today: `["text"]` for
    /// text-only checkpoints, `["text", "vision"]` for vision-capable
    /// ones (Stage B7). Clients like litellm / agent0 can gate
    /// `image_url` submission on the advertised set.
    ///
    /// Optional in the wire format so older clients that don't read
    /// it stay compatible. Default-empty for absent/older data, which
    /// callers can interpret as "text".
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,

    // ── Enrichment (issue #62) ────────────────────────────────
    /// Token budget advertised by the catalogue or discovered at load time.
    /// `None` when neither the catalogue nor the loaded model can provide it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<ModelLimit>,
    /// Operator-set pricing — see [`ModelCost`] for units and the
    /// absent (not priced) vs `0.0` (intentionally free) distinction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<ModelCost>,
    /// `true` when the model's tokenizer contains recognised tool-call
    /// marker tokens (`<tool_call>` / `<\/tool_call>` convention).
    #[serde(default)]
    pub tool_call: bool,
    /// `true` when the model's tokenizer contains recognised reasoning
    /// marker tokens (`<think>` / `<\/think>` or similar).
    #[serde(default)]
    pub reasoning: bool,
    /// The operator's `preserve_thinking` for this loaded model, or
    /// `None` when unset and the template's own default applies.
    ///
    /// Advertised so a run can be *stamped* with the value it ran
    /// under. This is an A/B knob whose whole purpose is comparison
    /// between runs, and a comparison whose arms cannot be told apart
    /// afterwards is not a comparison. Recovering it from deploy
    /// history is guesswork — a config sync can land mid-session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preserve_thinking: Option<bool>,

    /// Whether this node can actually serve a request for this model
    /// right now (#245).
    ///
    /// `status == "loaded"` only says the weights are resident. A node
    /// can hold a model and still reject every request — most commonly
    /// because something outside neuron has eaten the device's free VRAM
    /// and the prefill floor check will fail before any device work.
    /// That happened in production: a leaked desktop compositor took a
    /// whole tier down while the node answered every poll correctly and
    /// the fleet reported itself fully healthy.
    ///
    /// `None` means the node did not say — an older neuron, or a state
    /// it cannot evaluate. Callers must treat that as "assume servable",
    /// never as "unservable", or a version skew would evict the fleet
    /// from its own routing table.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub servable: Option<ModelServability>,
    /// The reasoning-effort ladder this deployment honours (#223).
    /// Absent when the model does not reason, or the harness has no
    /// budget to offer.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasoning_budget: Vec<ReasoningBudgetRung>,
}

/// Why a loaded model can or cannot be served right now (#245).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelServability {
    pub ok: bool,
    /// Machine-readable cause when `ok == false` — e.g.
    /// `"insufficient_vram"`, matching the error the request would get.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Human-facing detail, for an operator reading a health page.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// What an inference harness must do, from neuron's perspective.
///
/// All current harnesses are in-process — they share neuron's address
/// space and lifecycle. `start`/`stop` therefore default to no-ops; a
/// future process-supervising harness would override them.
#[async_trait]
pub trait Harness: Send + Sync {
    /// Human-readable name (e.g. "candle").
    fn name(&self) -> &str;

    /// Start the harness. Default no-op for in-process harnesses.
    async fn start(&self, _config: &HarnessConfig) -> Result<()> {
        Ok(())
    }

    /// Stop the harness. Default no-op for in-process harnesses.
    async fn stop(&self) -> Result<()> {
        Ok(())
    }

    /// Health check. Returns the harness process status.
    async fn health(&self) -> HarnessHealth;

    /// List models the harness knows about (loaded + unloaded).
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;

    /// Load a model with the given spec (quant, TP, device assignment).
    async fn load_model(&self, spec: &ModelSpec) -> Result<()>;

    /// Unload a model, freeing device memory.
    async fn unload_model(&self, model_id: &str) -> Result<()>;

    /// Return the URL where inference requests for this model should
    /// be sent. None if the model is not loaded.
    async fn inference_endpoint(&self, model_id: &str) -> Option<String>;
}
