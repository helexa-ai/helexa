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
    /// The deprecated spelling of the output cap. Still what most
    /// clients send; see [`ChatCompletionRequest::effective_max_tokens`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// OpenAI's current spelling of the output cap, which deprecated
    /// `max_tokens`. Newer SDKs and clients send only this one — it has
    /// to be a named field rather than falling into `extra`, or the cap
    /// is silently ignored and generation runs to the server default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// All other fields (tools, response_format, backend extensions, etc.)
    #[serde(flatten)]
    pub extra: Value,
}

impl ChatCompletionRequest {
    /// The output cap the caller asked for, under either spelling.
    ///
    /// `max_completion_tokens` wins when both are present — it is the
    /// current OpenAI field, and cortex's metering already reserves
    /// against it (`metering::requested_max_output`), so preferring it
    /// here keeps what we bill and what we generate in agreement.
    pub fn effective_max_tokens(&self) -> Option<u64> {
        self.max_completion_tokens.or(self.max_tokens)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// Absent on an assistant turn that carries only `tool_calls` —
    /// see [`MessageContent::Null`]. `#[serde(default)]` so a message
    /// with no `content` key at all deserializes rather than being
    /// rejected.
    #[serde(default)]
    pub content: MessageContent,
    #[serde(flatten)]
    pub extra: Value,
}

/// Content can be a simple string, an array of content parts (for
/// vision), or absent.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<Value>),
    /// `"content": null`, or no `content` key at all.
    ///
    /// This is the OpenAI-canonical shape for an assistant turn whose
    /// only payload is `tool_calls`, and agentic clients replay the
    /// assistant turn verbatim on the follow-up request that carries
    /// the tool result — so any client doing OpenAI-native tool
    /// calling sends it on its second turn. HF chat templates model it
    /// the same way (`content is none` renders as empty), so it maps
    /// straight through to the prompt.
    #[default]
    Null,
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
    /// OpenAI-standard breakdown of `prompt_tokens`, carrying
    /// `cached_tokens` (#269). Omitted when nothing was reused, so
    /// "absent" stays distinguishable from "measured as zero".
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
    /// Prompt tokens served from neuron's prefix KV cache (#11), so a
    /// caller can see the saving rather than assume none (#269). A
    /// sub-count of `prompt_tokens`, never added into `total_tokens`.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The follow-up request an OpenAI-native agentic client sends
    /// after running a tool: it replays its own assistant turn, which
    /// OpenAI emits with `"content": null` because the payload was
    /// only `tool_calls`. Rejecting this shape breaks every such
    /// client on its second turn.
    #[test]
    fn assistant_tool_call_turn_with_null_content_deserializes() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "model": "m",
                "messages": [
                    {"role": "user", "content": "list the files"},
                    {"role": "assistant", "content": null, "tool_calls": [
                        {"id": "call_0", "type": "function",
                         "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}}
                    ]},
                    {"role": "tool", "content": "a.txt", "tool_call_id": "call_0"}
                ]
            }"#,
        )
        .expect("null assistant content is a valid OpenAI request");

        assert!(matches!(req.messages[1].content, MessageContent::Null));
        // The tool calls survive in `extra`, where the chat template
        // reads them from.
        assert_eq!(
            req.messages[1].extra["tool_calls"][0]["function"]["name"],
            "bash"
        );
        assert_eq!(req.messages[2].extra["tool_call_id"], "call_0");
    }

    /// Some clients omit the key entirely rather than sending null.
    #[test]
    fn assistant_turn_with_no_content_key_deserializes() {
        let msg: ChatMessage = serde_json::from_str(
            r#"{"role": "assistant", "tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": "bash", "arguments": "{}"}}]}"#,
        )
        .expect("absent content is a valid OpenAI message");
        assert!(matches!(msg.content, MessageContent::Null));
    }

    /// Absent content round-trips back onto the wire as null, so a
    /// translated or re-serialized request stays OpenAI-shaped.
    #[test]
    fn null_content_round_trips() {
        let msg = ChatMessage {
            role: "assistant".into(),
            content: MessageContent::Null,
            extra: Value::Null,
        };
        let v = serde_json::to_value(&msg).expect("serialize");
        assert_eq!(v["content"], Value::Null);
        assert!(v.get("content").is_some(), "content key must be present");
    }

    /// Newer OpenAI SDKs send only `max_completion_tokens`; the cap
    /// must survive as a named field rather than falling into `extra`,
    /// where every generation loop would miss it.
    #[test]
    fn max_completion_tokens_is_the_effective_cap() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model": "m", "max_completion_tokens": 240,
                "messages": [{"role": "user", "content": "hi"}]}"#,
        )
        .expect("deserialize");
        assert_eq!(req.max_completion_tokens, Some(240));
        assert_eq!(req.max_tokens, None);
        assert_eq!(req.effective_max_tokens(), Some(240));
        assert!(
            req.extra.get("max_completion_tokens").is_none(),
            "must be a named field, not swept into extra"
        );
    }

    /// The legacy spelling keeps working on its own.
    #[test]
    fn legacy_max_tokens_still_caps() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model": "m", "max_tokens": 128,
                "messages": [{"role": "user", "content": "hi"}]}"#,
        )
        .expect("deserialize");
        assert_eq!(req.effective_max_tokens(), Some(128));
    }

    /// Clients that send both for compatibility must not be rejected,
    /// and the current field wins — matching what cortex meters
    /// against, so billed and generated caps agree.
    #[test]
    fn both_spellings_resolve_to_the_current_field() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model": "m", "max_tokens": 99, "max_completion_tokens": 256,
                "messages": [{"role": "user", "content": "hi"}]}"#,
        )
        .expect("sending both must not be an error");
        assert_eq!(req.effective_max_tokens(), Some(256));
    }

    /// Neither spelling present leaves the cap to the server default.
    #[test]
    fn absent_cap_is_none() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model": "m", "messages": [{"role": "user", "content": "hi"}]}"#,
        )
        .expect("deserialize");
        assert_eq!(req.effective_max_tokens(), None);
    }

    /// The string and array forms keep working unchanged.
    #[test]
    fn text_and_parts_content_still_deserialize() {
        let text: ChatMessage =
            serde_json::from_str(r#"{"role": "user", "content": "hi"}"#).expect("text");
        assert!(matches!(text.content, MessageContent::Text(ref t) if t == "hi"));

        let parts: ChatMessage = serde_json::from_str(
            r#"{"role": "user", "content": [{"type": "text", "text": "hi"}]}"#,
        )
        .expect("parts");
        assert!(matches!(parts.content, MessageContent::Parts(ref p) if p.len() == 1));
    }
}
