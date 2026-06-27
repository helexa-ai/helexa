//! OpenAI-compatible request and response types.
//!
//! These are a subset sufficient for chat completions (streaming + non-streaming).
//! Fields not relevant to proxying are captured as `serde_json::Value` via
//! `#[serde(flatten)]` so we forward them without needing to enumerate every
//! extension field a backend might support.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Chat completion request ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// All other fields (tools, response_format, backend extensions, etc.)
    #[serde(flatten)]
    pub extra: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
    #[serde(flatten)]
    pub extra: Value,
}

/// Content can be a simple string or an array of content parts (for vision).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<Value>),
}

// ── Chat completion response (non-streaming) ─────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(flatten)]
    pub extra: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
    #[serde(flatten)]
    pub extra: Value,
}

// ── Streaming chunk ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: u64,
    // Lenient deserialization throughout: the gateway parses chunks
    // from arbitrary OpenAI-compatible upstreams, and some engines
    // omit fields on special frames (e.g. usage-only final chunks).
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(flatten)]
    pub extra: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Value,
    pub finish_reason: Option<String>,
    #[serde(flatten)]
    pub extra: Value,
}

// ── Usage ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    /// OpenAI-standard breakdown of `completion_tokens`. Optional and
    /// additive — clients that don't read it are unaffected. Carries
    /// `reasoning_tokens` for reasoning models (a sub-count of
    /// `completion_tokens`, never added into `total_tokens`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// OpenAI-standard breakdown of `prompt_tokens`. Populated once
    /// prompt caching lands (#11); `None` until then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// helexa extension (non-OpenAI): server-measured prefill/decode
    /// timing, so the bench harness can compute true prefill vs decode
    /// tok/s instead of inferring both from client-side SSE arrival
    /// (#85). Additive and optional — standard OpenAI clients ignore
    /// it; cortex forwards usage verbatim so it survives proxying.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub helexa_timing: Option<HelexaTiming>,
}

/// helexa extension carried on [`Usage::helexa_timing`]. Mirrors
/// neuron's internal `FinishTiming`. All fields are server-measured;
/// `prefill_tokens` is the prefill-rate denominator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelexaTiming {
    pub prefill_ms: u64,
    pub decode_ms: u64,
    pub prefill_tokens: u64,
}

/// Sub-counts of `Usage::completion_tokens`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    /// Tokens generated inside the model's reasoning span.
    pub reasoning_tokens: u64,
}

/// Sub-counts of `Usage::prompt_tokens`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    /// Prompt tokens served from cache (cache-read rate). Populated
    /// once prompt caching lands (#11).
    pub cached_tokens: u64,
}

// ── Models list response ─────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owned_by: Option<String>,
    /// Gateway extensions: which node(s) host this model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locations: Option<Vec<super::node::ModelLocation>>,
    #[serde(flatten)]
    pub extra: Value,
}
