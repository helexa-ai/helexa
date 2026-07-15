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

/// `input` is either a single string or an array of items.
/// `#[serde(untagged)]` so the wire shape `"input": "hi"` and
/// `"input": [{...}]` both deserialize.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<ResponsesInputElement>),
}

/// One element of an `input` array.
///
/// OpenAI's Responses API accepts three shapes here, and real clients
/// use all of them — most notably agent-zero (via litellm), which
/// sends the bare "easy message" form. We must tolerate every shape,
/// because `input` is an `#[serde(untagged)]` array: a single element
/// that matches no variant fails the *entire* request with a 422
/// (`did not match any variant of untagged enum ResponsesInput`).
///
/// 1. [`Self::Typed`] — an item carrying an explicit `"type"`
///    discriminant (`message`, `function_call`, `function_call_output`,
///    `reasoning`).
/// 2. [`Self::EasyMessage`] — a bare `{role, content}` with **no**
///    `type` field. This is OpenAI's `EasyInputMessage` and what
///    litellm emits for every turn. `content` is optional so an
///    assistant turn carrying only tool calls (`content: null`) still
///    parses.
/// 3. [`Self::Other`] — anything else, captured as raw JSON and
///    dropped during translation. This is the forward-compat escape
///    hatch that mirrors [`ResponsesRequest::extra`] at the item
///    level: an unmodeled item type can never again reject the whole
///    request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInputElement {
    Typed(ResponsesInputItem),
    EasyMessage {
        role: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<ResponsesMessageContent>,
    },
    Other(Value),
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
    /// User is feeding a tool result back into the model. `output`
    /// is a `Value` because OpenAI allows it to be either a plain
    /// string or an array of content parts; the translator renders
    /// either form to text rather than losing the tool result.
    FunctionCallOutput { call_id: String, output: Value },
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
    /// Any content-part type we don't model (e.g. `refusal`, audio).
    /// Captured as a unit so an unknown part can't reject the whole
    /// request; dropped during translation.
    #[serde(other)]
    Unknown,
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
    /// OpenAI-standard breakdown of `output_tokens`. Optional and
    /// additive. Carries `reasoning_tokens` for reasoning models (a
    /// sub-count of `output_tokens`, never added into `total_tokens`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
    /// OpenAI-standard breakdown of `input_tokens`. Populated once
    /// prompt caching lands (#11); `None` until then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
}

/// Sub-counts of `ResponsesUsage::output_tokens`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    /// Tokens generated inside the model's reasoning span.
    pub reasoning_tokens: u64,
}

/// Sub-counts of `ResponsesUsage::input_tokens`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputTokensDetails {
    /// Input tokens served from cache (cache-read rate). Populated
    /// once prompt caching lands (#11).
    pub cached_tokens: u64,
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
    pub const FUNCTION_CALL_ARGUMENTS_DELTA: &str = "response.function_call_arguments.delta";
    pub const FUNCTION_CALL_ARGUMENTS_DONE: &str = "response.function_call_arguments.done";
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
                    ResponsesInputElement::Typed(ResponsesInputItem::Message { role, content }) => {
                        assert_eq!(role, "user");
                        match content {
                            ResponsesMessageContent::Text(t) => assert_eq!(t, "hi"),
                            other => panic!("expected Text content, got {other:?}"),
                        }
                    }
                    other => panic!("expected typed Message item, got {other:?}"),
                }
            }
            other => panic!("expected Items, got {other:?}"),
        }
    }

    #[test]
    fn deserialises_bare_easy_message_without_type() {
        // The shape agent-zero (via litellm) actually sends: `input`
        // items are bare `{role, content}` with NO `type` field. This
        // is the exact payload that was returning 422.
        let raw = r#"{
            "model": "Qwen/Qwen3.6-27B",
            "store": true,
            "tools": [{"type": "function", "name": "x", "description": "d", "parameters": {}}],
            "input": [
                {"role": "system", "content": "you are helpful"},
                {"role": "assistant", "content": "{\"tool_name\":\"response\"}"},
                {"role": "user", "content": "hi"}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(raw).unwrap();
        let items = match req.input {
            ResponsesInput::Items(i) => i,
            other => panic!("expected Items, got {other:?}"),
        };
        assert_eq!(items.len(), 3);
        for el in &items {
            assert!(
                matches!(el, ResponsesInputElement::EasyMessage { .. }),
                "expected EasyMessage, got {el:?}"
            );
        }
        // `tools` / `store` ride through `extra`, not `input`.
        assert!(req.extra.get("tools").is_some());
        assert_eq!(req.extra.get("store"), Some(&Value::Bool(true)));
    }

    #[test]
    fn tolerates_null_content_and_unknown_item_types() {
        // An assistant turn carrying only tool calls has `content: null`;
        // and a future/unmodeled item type must not 422 the request.
        let raw = r#"{
            "model": "m",
            "input": [
                {"role": "assistant", "content": null},
                {"type": "item_reference", "id": "abc"},
                {"type": "function_call_output", "call_id": "c1",
                 "output": [{"type": "output_text", "text": "result"}]},
                {"role": "user", "content": "go"}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(raw).unwrap();
        let items = match req.input {
            ResponsesInput::Items(i) => i,
            other => panic!("expected Items, got {other:?}"),
        };
        assert_eq!(items.len(), 4);
        assert!(matches!(
            &items[0],
            ResponsesInputElement::EasyMessage { content: None, .. }
        ));
        assert!(matches!(&items[1], ResponsesInputElement::Other(_)));
        assert!(matches!(
            &items[2],
            ResponsesInputElement::Typed(ResponsesInputItem::FunctionCallOutput { .. })
        ));
        assert!(matches!(
            &items[3],
            ResponsesInputElement::EasyMessage { .. }
        ));
    }

    #[test]
    fn tolerates_unknown_content_part_type() {
        // A `refusal` (or any unmodeled) content part must parse, not 422.
        let raw = r#"{
            "model": "m",
            "input": [
                {"role": "assistant", "content": [
                    {"type": "refusal", "refusal": "no"},
                    {"type": "output_text", "text": "ok"}
                ]}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(raw).unwrap();
        let items = match req.input {
            ResponsesInput::Items(i) => i,
            other => panic!("expected Items, got {other:?}"),
        };
        let parts = match &items[0] {
            ResponsesInputElement::EasyMessage {
                content: Some(ResponsesMessageContent::Parts(p)),
                ..
            } => p,
            other => panic!("expected EasyMessage with Parts, got {other:?}"),
        };
        assert_eq!(parts.len(), 2);
        assert!(matches!(&parts[0], ResponsesContentPart::Unknown));
        assert!(matches!(&parts[1], ResponsesContentPart::OutputText { .. }));
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
            ResponsesInputElement::Typed(ResponsesInputItem::Message {
                content: ResponsesMessageContent::Parts(p),
                ..
            }) => p,
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
                output_tokens_details: None,
                input_tokens_details: None,
            }),
        };
        let json = serde_json::to_string(&r).unwrap();
        let parsed: ResponsesResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "resp_1");
        assert_eq!(parsed.output.len(), 1);
    }
}
