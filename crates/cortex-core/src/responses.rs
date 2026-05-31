//! OpenAI Responses API (`POST /v1/responses`) envelope types.
//!
//! This is OpenAI's newer chat surface, distinct from
//! `/v1/chat/completions` in three ways that matter for us:
//!
//! 1. **Input shape**. Instead of a `messages` array, the request
//!    carries `input` — either a plain string (single user turn)
//!    or an array of typed items (messages, function calls,
//!    function-call outputs, reasoning blocks, …).
//! 2. **Output shape**. The response carries a single `output`
//!    array of items, each typed. We always emit one
//!    `OutputItem::Message` containing the assistant's reply (plus,
//!    when we get there, separate `function_call` items).
//! 3. **Streaming events**. Where chat completions stream
//!    structurally-identical `chat.completion.chunk` frames over
//!    `data:` lines, Responses streams *named* events
//!    (`response.created`, `response.output_text.delta`,
//!    `response.completed`, …) over `event:` + `data:` SSE pairs.
//!    The wire projector in `neuron::wire::openai_responses` builds
//!    these from the same [`crate::openai`]-shaped
//!    `InferenceEvent` stream the chat projector consumes.
//!
//! Scope cuts for this first cut:
//!
//! - **`previous_response_id` is rejected at parse time**. Stateful
//!   chained conversations need a persistence layer we don't have.
//! - **Reasoning items are accepted-and-ignored** (no Qwen3
//!   `<think>` routing yet). Audio and embedded resources are
//!   rejected as unsupported.
//! - **Tool calls** (function_call / function_call_output) are
//!   carried as round-trip types but the candle harness doesn't
//!   emit them yet — wired so the surface is in place for the
//!   day we add proper tool-call extraction.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Request ──────────────────────────────────────────────────────────

/// Body of a `POST /v1/responses` request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: ResponsesInput,
    /// System-prompt-style instructions. The Responses API
    /// separates these from input so a caller doesn't have to
    /// build a `system` message item by hand.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Chained-conversation identifier. We don't store responses
    /// server-side yet; if this is `Some`, the handler returns 400.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    /// Catch-all for anything we don't model yet (tools, tool_choice,
    /// reasoning, response_format, …). Lets a client send a
    /// forward-compatible request without our parser rejecting it.
    #[serde(flatten)]
    pub extra: Value,
}

/// `input` is either a single string or an array of typed items.
/// `#[serde(untagged)]` so the wire shape `"input": "hi"` and
/// `"input": [{...}]` both deserialize.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<ResponsesInputItem>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesInputItem {
    /// A user / assistant / system turn.
    Message {
        role: String,
        content: ResponsesMessageContent,
    },
    /// Assistant emitted a tool call. Round-trip only — neuron
    /// doesn't synthesise these yet.
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    /// User is feeding a tool result back into the model.
    FunctionCallOutput { call_id: String, output: String },
    /// Reasoning items emitted by o-series models. Accepted but
    /// not forwarded to the model — neuron's candle path doesn't
    /// surface reasoning separately yet.
    Reasoning {
        #[serde(default)]
        content: Vec<Value>,
    },
}

/// Inside a `Message` item, content is either a plain string or an
/// array of typed parts. Mirrors the chat-completions Parts shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesMessageContent {
    Text(String),
    Parts(Vec<ResponsesContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesContentPart {
    /// Plain text inside a user / system turn.
    InputText { text: String },
    /// An image. `image_url` is either a remote URL or a
    /// `data:image/png;base64,…` URI; the request translator just
    /// forwards the string.
    InputImage {
        image_url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    /// Returned text inside an assistant turn — only relevant when
    /// the caller is feeding an assistant turn back in to continue
    /// a conversation manually (no `previous_response_id`).
    OutputText {
        text: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        annotations: Vec<Value>,
    },
}

// ── Response (non-streaming) ─────────────────────────────────────────

/// Body of a `POST /v1/responses` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesResponse {
    pub id: String,
    /// Always `"response"`.
    pub object: String,
    pub created_at: u64,
    /// `"completed"`, `"incomplete"`, or — for the initial event of
    /// a streaming response — `"in_progress"`.
    pub status: String,
    pub model: String,
    pub output: Vec<ResponsesOutputItem>,
    /// Populated on completion; `None` while streaming.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponsesUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesOutputItem {
    Message {
        id: String,
        /// Always `"assistant"` for model output.
        role: String,
        /// Output content parts. We always emit a single
        /// `OutputText` today; multi-part output would land here
        /// once we have e.g. image generation.
        content: Vec<ResponsesOutputContent>,
        /// Item-level status. `"in_progress"` while streaming the
        /// content parts, `"completed"` when done.
        #[serde(default = "default_item_status")]
        status: String,
    },
    /// Reserved for the day tool-call extraction lands. The wire
    /// shape mirrors `ResponsesInputItem::FunctionCall`.
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        #[serde(default = "default_item_status")]
        status: String,
    },
}

fn default_item_status() -> String {
    "completed".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesOutputContent {
    OutputText {
        text: String,
        /// Citations / inline annotations. Empty today; reserved
        /// for the day we wire in web search / file search.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        annotations: Vec<Value>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

// ── Streaming event names ────────────────────────────────────────────

/// Event names the SSE projector emits, hoisted as constants so
/// the projector and the wire shape stay in sync without
/// string-typos. The strings are dictated by OpenAI's published
/// Responses API.
pub mod events {
    pub const CREATED: &str = "response.created";
    /// Fired between `response.created` and the first output-item
    /// event. Marks "request validated, model is generating" —
    /// some clients use it to differentiate the "warming up" state
    /// from "streaming tokens" in their UI.
    pub const IN_PROGRESS: &str = "response.in_progress";
    pub const OUTPUT_ITEM_ADDED: &str = "response.output_item.added";
    pub const CONTENT_PART_ADDED: &str = "response.content_part.added";
    pub const OUTPUT_TEXT_DELTA: &str = "response.output_text.delta";
    pub const OUTPUT_TEXT_DONE: &str = "response.output_text.done";
    pub const CONTENT_PART_DONE: &str = "response.content_part.done";
    pub const OUTPUT_ITEM_DONE: &str = "response.output_item.done";
    pub const COMPLETED: &str = "response.completed";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_input_string_form() {
        let raw = r#"{"model": "m", "input": "hello"}"#;
        let req: ResponsesRequest = serde_json::from_str(raw).unwrap();
        match req.input {
            ResponsesInput::Text(s) => assert_eq!(s, "hello"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn deserialises_input_items_form() {
        let raw = r#"{
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": "hi"}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(raw).unwrap();
        match req.input {
            ResponsesInput::Items(items) => {
                assert_eq!(items.len(), 1);
                match &items[0] {
                    ResponsesInputItem::Message { role, content } => {
                        assert_eq!(role, "user");
                        match content {
                            ResponsesMessageContent::Text(t) => assert_eq!(t, "hi"),
                            other => panic!("expected Text content, got {other:?}"),
                        }
                    }
                    other => panic!("expected Message item, got {other:?}"),
                }
            }
            other => panic!("expected Items, got {other:?}"),
        }
    }

    #[test]
    fn deserialises_input_with_image() {
        let raw = r#"{
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "what is this"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAA="}
                ]}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(raw).unwrap();
        let items = match req.input {
            ResponsesInput::Items(i) => i,
            other => panic!("expected Items, got {other:?}"),
        };
        let parts = match &items[0] {
            ResponsesInputItem::Message {
                content: ResponsesMessageContent::Parts(p),
                ..
            } => p,
            other => panic!("expected Parts, got {other:?}"),
        };
        assert_eq!(parts.len(), 2);
        assert!(matches!(
            &parts[0],
            ResponsesContentPart::InputText { text } if text == "what is this"
        ));
        assert!(matches!(
            &parts[1],
            ResponsesContentPart::InputImage { image_url, .. }
                if image_url == "data:image/png;base64,AAA="
        ));
    }

    #[test]
    fn unknown_fields_round_trip_via_extra() {
        let raw = r#"{
            "model": "m",
            "input": "hi",
            "tools": [{"type": "web_search"}],
            "reasoning": {"effort": "medium"}
        }"#;
        let req: ResponsesRequest = serde_json::from_str(raw).unwrap();
        assert!(req.extra.get("tools").is_some());
        assert!(req.extra.get("reasoning").is_some());
    }

    #[test]
    fn response_round_trips_through_serde() {
        let r = ResponsesResponse {
            id: "resp_1".into(),
            object: "response".into(),
            created_at: 1700,
            status: "completed".into(),
            model: "m".into(),
            output: vec![ResponsesOutputItem::Message {
                id: "msg_1".into(),
                role: "assistant".into(),
                content: vec![ResponsesOutputContent::OutputText {
                    text: "hi there".into(),
                    annotations: vec![],
                }],
                status: "completed".into(),
            }],
            usage: Some(ResponsesUsage {
                input_tokens: 5,
                output_tokens: 3,
                total_tokens: 8,
            }),
        };
        let json = serde_json::to_string(&r).unwrap();
        let parsed: ResponsesResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "resp_1");
        assert_eq!(parsed.output.len(), 1);
    }
}
