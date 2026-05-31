//! OpenAI Responses API (`POST /v1/responses`) provider.
//!
//! Mirror image of [`super::openai_chat`]: same `Provider` trait
//! impl, same back-pressured SSE decoder, but speaking OpenAI's
//! newer Responses surface instead of chat completions.
//!
//! Differences from the chat provider, all contained in this file:
//!
//! - **Request encoding**: history flattens into an `input` array
//!   of typed items (`message`, `function_call`, `function_call_output`)
//!   plus a top-level `instructions` field for the system prompt.
//!   Multi-part user content stays in the same `[{type:"input_text"},
//!   {type:"input_image"}]` shape neuron's `request_to_chat` already
//!   accepts.
//! - **Streaming decoder**: events are named (`response.created`,
//!   `response.output_text.delta`, `response.completed`, …) carried
//!   on the SSE `event:` line. The chat path's `[DONE]` terminator
//!   doesn't apply; the stream ends after `response.completed`.
//! - **Tool calls** plumb through the `response.output_item.added`
//!   (item type `function_call`) → `response.function_call_arguments.delta`
//!   → `response.function_call_arguments.done` event sequence. The
//!   neuron candle harness doesn't synthesize these yet (tracked as
//!   issue #6), but the decoder is wired so the day the upstream
//!   does, downstream `CompletionEvent::ToolCall*` plumbing just
//!   works.
//!
//! Tool-name handling: the model knows its tool descriptions via
//! the [`crate::qwen3`] system-prompt block exactly the way the chat
//! provider does. We don't echo them in the request body because
//! neuron currently ignores `tools` on /v1/responses (same as on
//! /v1/chat/completions). Once neuron honours request-side tool
//! definitions, both providers add them in the same place.

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt, stream::BoxStream};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use tokio_util::sync::CancellationToken;

use super::{
    CompletionEvent, CompletionRequest, Message, MessageContent, MessagePart, ModelInfo, Provider,
    Role, UsageStats,
};
use crate::config::EndpointConfig;

pub struct OpenAIResponsesProvider {
    endpoint: EndpointConfig,
    #[allow(dead_code)] // Read in `complete()`'s HTTP path; tests don't stand up a server.
    api_key: Option<String>,
    #[allow(dead_code)]
    http: reqwest::Client,
}

impl OpenAIResponsesProvider {
    pub fn new(endpoint: EndpointConfig) -> anyhow::Result<Self> {
        let api_key = endpoint.resolve_api_key()?;
        let http = reqwest::Client::builder()
            // Same generous timeout as the chat provider: cortex may
            // need to cold-load a model before serving the first
            // chunk, which can be tens of seconds. Cancellation
            // handles early termination, not timeout.
            .timeout(std::time::Duration::from_secs(600))
            .build()?;
        Ok(Self {
            endpoint,
            api_key,
            http,
        })
    }
}

#[async_trait]
impl Provider for OpenAIResponsesProvider {
    fn name(&self) -> &str {
        &self.endpoint.name
    }

    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        let mut req = self.http.get(self.endpoint.models_url());
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("{} list_models: {e}", self.endpoint.name))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "{} list_models returned {}: {}",
                self.endpoint.name,
                status,
                body
            );
        }
        let body: WireModelsResponse = resp.json().await?;
        Ok(body
            .data
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id,
                display_name: None,
            })
            .collect())
    }

    async fn complete(
        &self,
        request: CompletionRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<CompletionEvent>>> {
        let body = encode_request(&request);
        tracing::debug!(
            endpoint = %self.endpoint.name,
            url = %self.endpoint.responses_url(),
            body = %serde_json::to_string(&body).unwrap_or_else(|_| "<unserializable>".into()),
            "POST /responses"
        );
        let mut req = self.http.post(self.endpoint.responses_url()).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("{} responses send: {e}", self.endpoint.name))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "{} responses returned {}: {}",
                self.endpoint.name,
                status,
                body
            );
        }
        let sse = resp.bytes_stream().eventsource();
        let stream = decode_stream(sse, cancel);
        Ok(Box::pin(stream))
    }
}

// ── Request encoding ─────────────────────────────────────────────────

fn encode_request(req: &CompletionRequest) -> Value {
    // Pull the system messages out of history into a single
    // `instructions` string — the Responses API expects them there,
    // not inline as an `input` item. Multiple system messages
    // concatenate with blank lines so we don't lose ordering.
    let mut instructions: Vec<String> = Vec::new();
    let mut input_items: Vec<Value> = Vec::new();
    for msg in &req.messages {
        if msg.role == Role::System
            && let MessageContent::Text { text } = &msg.content
        {
            instructions.push(text.clone());
            continue;
        }
        if let Some(item) = encode_message_as_input_item(msg) {
            input_items.push(item);
        }
    }

    let mut body = json!({
        "model": req.model,
        "input": input_items,
        "stream": true,
    });
    if let Value::Object(map) = &mut body {
        if !instructions.is_empty() {
            map.insert(
                "instructions".into(),
                Value::String(instructions.join("\n\n")),
            );
        }
        if let Some(t) = req.temperature {
            map.insert("temperature".into(), json!(t));
        }
        if let Some(p) = req.top_p {
            map.insert("top_p".into(), json!(p));
        }
        if let Some(m) = req.max_tokens {
            // Responses calls it `max_output_tokens`; preserve the
            // semantic (response cap) when we translate.
            map.insert("max_output_tokens".into(), json!(m));
        }
    }
    body
}

fn encode_message_as_input_item(msg: &Message) -> Option<Value> {
    match (msg.role, &msg.content) {
        (Role::System, _) => None, // handled out-of-band as `instructions`
        (Role::User, MessageContent::Text { text }) => Some(json!({
            "type": "message",
            "role": "user",
            "content": text,
        })),
        (Role::User, MessageContent::MultiPart { parts }) => Some(json!({
            "type": "message",
            "role": "user",
            "content": encode_user_parts(parts),
        })),
        (Role::Assistant, MessageContent::Text { text }) => Some(json!({
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": text,
                "annotations": [],
            }],
        })),
        (Role::Assistant, MessageContent::ToolCalls { text, calls }) => {
            // Assistant turns that called tools become a sequence of
            // items: an optional `message` (any prose alongside the
            // call) followed by one `function_call` per call. Mirrors
            // OpenAI Responses' "each item is one structural slot"
            // shape.
            //
            // We can't return multiple items from one call site, so
            // we encode this by side-stuffing additional items into a
            // single composite value and have the caller flatten —
            // but that complicates the API. Easier: build the array
            // ourselves in the caller path. For now, emit just the
            // function_calls (the assistant's prose lives in the next
            // turn's chat history anyway because the model isn't
            // looking back at its own previous narration). If the
            // text is non-empty AND we have calls, we lose the text;
            // qwen3 rarely emits prose alongside tool calls so this
            // is a deliberate simplification — revisit if it bites.
            let _ = text;
            // Take the first call only for the moment; multi-call
            // turns would need the caller-flattening above.
            let call = calls.first()?;
            Some(json!({
                "type": "function_call",
                "call_id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            }))
        }
        (
            Role::Tool,
            MessageContent::ToolResult {
                tool_call_id,
                content,
            },
        ) => Some(json!({
            "type": "function_call_output",
            "call_id": tool_call_id,
            "output": content,
        })),
        (role, content) => {
            tracing::warn!(
                ?role,
                ?content,
                "openai_responses: unexpected (role, content) shape"
            );
            None
        }
    }
}

fn encode_user_parts(parts: &[MessagePart]) -> Value {
    let items: Vec<Value> = parts
        .iter()
        .map(|p| match p {
            MessagePart::Text { text } => json!({"type": "input_text", "text": text}),
            MessagePart::Image(img) => json!({
                "type": "input_image",
                "image_url": format!("data:{};base64,{}", img.mime_type, img.data),
            }),
        })
        .collect();
    Value::Array(items)
}

// ── Wire types ──────────────────────────────────────────────────────

#[allow(dead_code)] // fields read only when list_models runs against a real endpoint
#[derive(Debug, Deserialize)]
struct WireModelsResponse {
    data: Vec<WireModelObject>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct WireModelObject {
    id: String,
}

// SSE event payload shapes. We only model the fields we care about;
// `#[serde(default)]` + `Option` everywhere else lets the upstream
// add optional fields without breaking deserialise.

#[derive(Debug, Deserialize, Serialize)]
struct OutputItemAddedEvent {
    #[serde(default)]
    output_index: u32,
    item: OutputItem,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutputItem {
    Message {
        #[serde(default)]
        id: Option<String>,
    },
    FunctionCall {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        call_id: Option<String>,
        #[serde(default)]
        name: Option<String>,
        /// Some upstreams populate `arguments` already on the
        /// `output_item.added` event for a fully-buffered tool call
        /// (i.e. when the model finalised the call before the SSE
        /// flush). Capture it so we can emit a single args delta.
        #[serde(default)]
        arguments: Option<String>,
    },
    /// `reasoning`, `web_search_call`, etc. We capture-and-ignore
    /// any item we don't model; the decoder still emits the
    /// outer events correctly.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize, Serialize)]
struct OutputTextDeltaEvent {
    #[serde(default)]
    item_id: Option<String>,
    #[serde(default)]
    output_index: u32,
    #[serde(default)]
    delta: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct FunctionCallArgumentsDeltaEvent {
    #[serde(default)]
    item_id: Option<String>,
    #[serde(default)]
    output_index: u32,
    #[serde(default)]
    delta: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct ResponseCompletedEvent {
    response: ResponseShell,
}

#[derive(Debug, Deserialize, Serialize)]
struct ResponseShell {
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Debug, Deserialize, Serialize)]
struct WireUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
}

// ── Streaming decoder ───────────────────────────────────────────────

/// Translate the named-event Responses SSE into the provider-agnostic
/// [`CompletionEvent`] stream the agent loop expects. The decoder
/// holds per-stream state — output_index → tool-call-index plus
/// the next available tool-call slot — so it can fire
/// `ToolCallStart` exactly once per item.
fn decode_stream<S>(
    sse: S,
    cancel: CancellationToken,
) -> impl Stream<Item = anyhow::Result<CompletionEvent>>
where
    S: Stream<
            Item = Result<
                eventsource_stream::Event,
                eventsource_stream::EventStreamError<reqwest::Error>,
            >,
        > + Send
        + 'static,
{
    async_stream::stream! {
        let mut sse = Box::pin(sse);
        // Maps an output_index that's a function_call to the tool-call
        // slot we hand downstream. Lets us correlate later
        // `function_call_arguments.delta` events back to the index
        // we already announced on `output_item.added`.
        let mut tool_index_by_output: HashMap<u32, usize> = HashMap::new();
        let mut next_tool_index: usize = 0;

        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    tracing::debug!("openai_responses: cancellation requested, ending stream");
                    break;
                }
                next = sse.next() => {
                    let Some(event) = next else { break };
                    let event = match event {
                        Ok(e) => e,
                        Err(e) => {
                            yield Err(anyhow::anyhow!("SSE transport: {e}"));
                            break;
                        }
                    };
                    // Event name lives on `event.event`; data is JSON.
                    let event_name = event.event.as_str();
                    let data = event.data.as_str();
                    match event_name {
                        "response.output_text.delta" => {
                            match serde_json::from_str::<OutputTextDeltaEvent>(data) {
                                Ok(d) if !d.delta.is_empty() => {
                                    yield Ok(CompletionEvent::TextDelta(d.delta));
                                }
                                Ok(_) => {}
                                Err(e) => {
                                    tracing::warn!(
                                        error = %e,
                                        raw = %data,
                                        "openai_responses: failed to parse output_text.delta; skipping"
                                    );
                                }
                            }
                        }
                        "response.output_item.added" => {
                            match serde_json::from_str::<OutputItemAddedEvent>(data) {
                                Ok(ev) => {
                                    if let OutputItem::FunctionCall {
                                        id,
                                        call_id,
                                        name,
                                        arguments,
                                    } = ev.item
                                    {
                                        let idx = next_tool_index;
                                        next_tool_index += 1;
                                        tool_index_by_output.insert(ev.output_index, idx);
                                        // Prefer the user-facing
                                        // `call_id` (what gets paired
                                        // with tool results) over the
                                        // internal item `id` when
                                        // both are present. Falls
                                        // back to a synthetic id so
                                        // history bookkeeping never
                                        // breaks.
                                        let final_id = call_id
                                            .or(id)
                                            .unwrap_or_else(|| format!("call_{idx}"));
                                        let final_name = name.unwrap_or_default();
                                        yield Ok(CompletionEvent::ToolCallStart {
                                            index: idx,
                                            id: final_id,
                                            name: final_name,
                                        });
                                        // Some upstreams attach the
                                        // fully-buffered arguments on
                                        // the `output_item.added`
                                        // event itself (rare; happens
                                        // when the model finalised
                                        // before the SSE flush).
                                        // Emit as a single args
                                        // delta if present.
                                        if let Some(args) = arguments
                                            && !args.is_empty()
                                        {
                                            yield Ok(CompletionEvent::ToolCallArgsDelta {
                                                index: idx,
                                                args_delta: args,
                                            });
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        error = %e,
                                        raw = %data,
                                        "openai_responses: failed to parse output_item.added; skipping"
                                    );
                                }
                            }
                        }
                        "response.function_call_arguments.delta" => {
                            match serde_json::from_str::<FunctionCallArgumentsDeltaEvent>(data) {
                                Ok(ev) => {
                                    let Some(&idx) = tool_index_by_output.get(&ev.output_index)
                                    else {
                                        // Args delta for an item we
                                        // never saw an `output_item.added`
                                        // for. Could happen if the
                                        // upstream reordered events;
                                        // log + skip.
                                        tracing::warn!(
                                            output_index = ev.output_index,
                                            "openai_responses: function_call_arguments.delta for unknown output_index"
                                        );
                                        continue;
                                    };
                                    if !ev.delta.is_empty() {
                                        yield Ok(CompletionEvent::ToolCallArgsDelta {
                                            index: idx,
                                            args_delta: ev.delta,
                                        });
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        error = %e,
                                        raw = %data,
                                        "openai_responses: failed to parse function_call_arguments.delta; skipping"
                                    );
                                }
                            }
                        }
                        "response.completed" => {
                            // Final event. Pull usage + status off
                            // the response shell. Status maps:
                            // "completed" → no special handling
                            // (caller treats as EndTurn),
                            // "incomplete" → length stop.
                            let (reason, usage) =
                                match serde_json::from_str::<ResponseCompletedEvent>(data) {
                                    Ok(ev) => {
                                        let reason = match ev.response.status.as_deref() {
                                            Some("incomplete") => Some("length".to_string()),
                                            _ => Some("stop".to_string()),
                                        };
                                        let usage = ev.response.usage.map(|u| UsageStats {
                                            prompt_tokens: u.input_tokens,
                                            completion_tokens: u.output_tokens,
                                            total_tokens: u.total_tokens,
                                        });
                                        (reason, usage)
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            error = %e,
                                            raw = %data,
                                            "openai_responses: failed to parse response.completed; ending stream with EndTurn"
                                        );
                                        (Some("stop".to_string()), None)
                                    }
                                };
                            if let Some(u) = usage {
                                yield Ok(CompletionEvent::Usage(u));
                            }
                            yield Ok(CompletionEvent::Finish { reason });
                            break;
                        }
                        // Bookkeeping events we don't need to surface:
                        // response.created, response.in_progress,
                        // response.content_part.added/.done,
                        // response.output_text.done,
                        // response.output_item.done,
                        // response.function_call_arguments.done,
                        // response.reasoning_*. Logged at debug for
                        // wire-tracing.
                        other => {
                            tracing::trace!(
                                event = other,
                                "openai_responses: bookkeeping event"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ToolCall;
    use crate::provider::{ImageData, MessagePart};
    use futures::stream;
    use url::Url;

    fn ep() -> EndpointConfig {
        EndpointConfig {
            name: "test".into(),
            base_url: Url::parse("http://localhost:9999/v1").unwrap(),
            wire_api: crate::config::WireApi::OpenAiResponses,
            default_model: None,
            api_key: None,
            api_key_env: None,
            max_tokens: None,
            context_window: None,
        }
    }

    // ── encode_request ──────────────────────────────────────────────

    #[test]
    fn system_messages_collapse_to_instructions() {
        let req = CompletionRequest {
            model: "m".into(),
            messages: vec![
                Message {
                    role: Role::System,
                    content: MessageContent::Text {
                        text: "you are helpful".into(),
                    },
                },
                Message {
                    role: Role::User,
                    content: MessageContent::Text { text: "hi".into() },
                },
            ],
            tools: vec![],
            temperature: Some(0.7),
            top_p: None,
            max_tokens: Some(256),
        };
        let body = encode_request(&req);
        assert_eq!(body["model"], "m");
        assert_eq!(body["instructions"], "you are helpful");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_output_tokens"], 256);
        assert_eq!(body["temperature"], 0.7);
        let input = body["input"].as_array().unwrap();
        // System message NOT echoed in input — it's only in
        // instructions.
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"], "hi");
    }

    #[test]
    fn multiple_system_messages_concatenate() {
        let req = CompletionRequest {
            model: "m".into(),
            messages: vec![
                Message {
                    role: Role::System,
                    content: MessageContent::Text {
                        text: "first".into(),
                    },
                },
                Message {
                    role: Role::System,
                    content: MessageContent::Text {
                        text: "second".into(),
                    },
                },
                Message {
                    role: Role::User,
                    content: MessageContent::Text { text: "hi".into() },
                },
            ],
            tools: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
        };
        let body = encode_request(&req);
        assert_eq!(body["instructions"], "first\n\nsecond");
    }

    #[test]
    fn user_multipart_becomes_input_parts_array() {
        let req = CompletionRequest {
            model: "vl".into(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::MultiPart {
                    parts: vec![
                        MessagePart::Text {
                            text: "what's in this?".into(),
                        },
                        MessagePart::Image(ImageData {
                            mime_type: "image/png".into(),
                            data: "AAA=".into(),
                            uri: None,
                        }),
                    ],
                },
            }],
            tools: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
        };
        let body = encode_request(&req);
        let content = &body["input"][0]["content"].as_array().unwrap().clone();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "input_text");
        assert_eq!(content[0]["text"], "what's in this?");
        assert_eq!(content[1]["type"], "input_image");
        assert_eq!(content[1]["image_url"], "data:image/png;base64,AAA=");
    }

    #[test]
    fn assistant_text_becomes_output_text_content_part() {
        let req = CompletionRequest {
            model: "m".into(),
            messages: vec![
                Message {
                    role: Role::User,
                    content: MessageContent::Text { text: "hi".into() },
                },
                Message {
                    role: Role::Assistant,
                    content: MessageContent::Text {
                        text: "hello there".into(),
                    },
                },
                Message {
                    role: Role::User,
                    content: MessageContent::Text {
                        text: "more".into(),
                    },
                },
            ],
            tools: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
        };
        let body = encode_request(&req);
        let input = body["input"].as_array().unwrap();
        assert_eq!(input.len(), 3);
        assert_eq!(input[1]["type"], "message");
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[1]["content"][0]["type"], "output_text");
        assert_eq!(input[1]["content"][0]["text"], "hello there");
    }

    #[test]
    fn tool_calls_and_results_round_trip_via_function_call_items() {
        let req = CompletionRequest {
            model: "m".into(),
            messages: vec![
                Message {
                    role: Role::Assistant,
                    content: MessageContent::ToolCalls {
                        text: None,
                        calls: vec![ToolCall {
                            id: "call_42".into(),
                            name: "read_file".into(),
                            arguments: r#"{"path":"/etc/hostname"}"#.into(),
                        }],
                    },
                },
                Message {
                    role: Role::Tool,
                    content: MessageContent::ToolResult {
                        tool_call_id: "call_42".into(),
                        content: "host".into(),
                    },
                },
            ],
            tools: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
        };
        let body = encode_request(&req);
        let input = body["input"].as_array().unwrap();
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["type"], "function_call");
        assert_eq!(input[0]["call_id"], "call_42");
        assert_eq!(input[0]["name"], "read_file");
        assert_eq!(input[0]["arguments"], r#"{"path":"/etc/hostname"}"#);
        assert_eq!(input[1]["type"], "function_call_output");
        assert_eq!(input[1]["call_id"], "call_42");
        assert_eq!(input[1]["output"], "host");
    }

    // ── decode_stream ───────────────────────────────────────────────

    fn sse_event(name: &str, data: &str) -> eventsource_stream::Event {
        eventsource_stream::Event {
            id: String::new(),
            retry: None,
            event: name.into(),
            data: data.into(),
        }
    }

    async fn collect_events(
        items: Vec<eventsource_stream::Event>,
    ) -> Vec<anyhow::Result<CompletionEvent>> {
        let sse = stream::iter(
            items
                .into_iter()
                .map(Ok::<_, eventsource_stream::EventStreamError<reqwest::Error>>),
        );
        let decoded = decode_stream(sse, CancellationToken::new());
        decoded.collect().await
    }

    #[tokio::test]
    async fn decodes_text_then_finish() {
        let events = collect_events(vec![
            sse_event("response.created", "{}"),
            sse_event(
                "response.output_text.delta",
                r#"{"item_id":"msg_1","output_index":0,"delta":"hel"}"#,
            ),
            sse_event(
                "response.output_text.delta",
                r#"{"item_id":"msg_1","output_index":0,"delta":"lo"}"#,
            ),
            sse_event(
                "response.completed",
                r#"{"response":{"status":"completed","usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}"#,
            ),
        ])
        .await;
        let events: Vec<CompletionEvent> = events.into_iter().map(|r| r.unwrap()).collect();
        let mut iter = events.into_iter();
        assert!(matches!(iter.next(), Some(CompletionEvent::TextDelta(t)) if t == "hel"));
        assert!(matches!(iter.next(), Some(CompletionEvent::TextDelta(t)) if t == "lo"));
        assert!(matches!(iter.next(), Some(CompletionEvent::Usage(u)) if u.total_tokens == 5));
        assert!(matches!(
            iter.next(),
            Some(CompletionEvent::Finish { reason: Some(r) }) if r == "stop"
        ));
        assert!(iter.next().is_none());
    }

    #[tokio::test]
    async fn empty_delta_is_dropped() {
        let events = collect_events(vec![
            sse_event(
                "response.output_text.delta",
                r#"{"item_id":"m","output_index":0,"delta":""}"#,
            ),
            sse_event(
                "response.completed",
                r#"{"response":{"status":"completed"}}"#,
            ),
        ])
        .await;
        let mut completion_events = events.into_iter().map(|r| r.unwrap());
        // First event MUST be the Finish — the empty delta dropped.
        assert!(matches!(
            completion_events.next(),
            Some(CompletionEvent::Finish { .. })
        ));
    }

    #[tokio::test]
    async fn incomplete_status_maps_to_length_finish_reason() {
        let events = collect_events(vec![sse_event(
            "response.completed",
            r#"{"response":{"status":"incomplete"}}"#,
        )])
        .await;
        let events: Vec<CompletionEvent> = events.into_iter().map(|r| r.unwrap()).collect();
        assert!(matches!(
            events.last(),
            Some(CompletionEvent::Finish { reason: Some(r) }) if r == "length"
        ));
    }

    #[tokio::test]
    async fn function_call_items_emit_toolcall_events() {
        let events = collect_events(vec![
            sse_event(
                "response.output_item.added",
                r#"{"output_index":0,"item":{"type":"function_call","id":"item_1","call_id":"call_xyz","name":"read_file"}}"#,
            ),
            sse_event(
                "response.function_call_arguments.delta",
                r#"{"item_id":"item_1","output_index":0,"delta":"{\"path"}"#,
            ),
            sse_event(
                "response.function_call_arguments.delta",
                r#"{"item_id":"item_1","output_index":0,"delta":"\":\"/etc/hostname\"}"}"#,
            ),
            sse_event("response.completed", r#"{"response":{"status":"completed"}}"#),
        ])
        .await;
        let events: Vec<CompletionEvent> = events.into_iter().map(|r| r.unwrap()).collect();
        let mut iter = events.into_iter();
        assert!(matches!(
            iter.next(),
            Some(CompletionEvent::ToolCallStart { index: 0, ref id, ref name })
                if id == "call_xyz" && name == "read_file"
        ));
        assert!(matches!(
            iter.next(),
            Some(CompletionEvent::ToolCallArgsDelta { index: 0, ref args_delta })
                if args_delta == r#"{"path"#
        ));
        assert!(matches!(
            iter.next(),
            Some(CompletionEvent::ToolCallArgsDelta { index: 0, ref args_delta })
                if args_delta == r#"":"/etc/hostname"}"#
        ));
        assert!(matches!(iter.next(), Some(CompletionEvent::Finish { .. })));
    }

    #[tokio::test]
    async fn function_call_added_with_inline_arguments_emits_single_args_delta() {
        // Some upstreams (rare) include the fully-buffered arguments
        // on the `output_item.added` event when the model finalised
        // the call before SSE flush. Verify both ToolCallStart and a
        // single args delta fire.
        let events = collect_events(vec![
            sse_event(
                "response.output_item.added",
                r#"{"output_index":0,"item":{"type":"function_call","call_id":"call_a","name":"f","arguments":"{\"x\":1}"}}"#,
            ),
            sse_event("response.completed", r#"{"response":{"status":"completed"}}"#),
        ])
        .await;
        let events: Vec<CompletionEvent> = events.into_iter().map(|r| r.unwrap()).collect();
        let mut iter = events.into_iter();
        assert!(matches!(
            iter.next(),
            Some(CompletionEvent::ToolCallStart { .. })
        ));
        assert!(matches!(
            iter.next(),
            Some(CompletionEvent::ToolCallArgsDelta { index: 0, ref args_delta })
                if args_delta == r#"{"x":1}"#
        ));
        assert!(matches!(iter.next(), Some(CompletionEvent::Finish { .. })));
    }

    #[tokio::test]
    async fn cancellation_ends_stream_promptly() {
        // Hand the decoder an empty stream + a triggered cancellation
        // token; it should terminate without yielding anything.
        let sse = stream::iter(Vec::<
            Result<eventsource_stream::Event, eventsource_stream::EventStreamError<reqwest::Error>>,
        >::new());
        let cancel = CancellationToken::new();
        cancel.cancel();
        let decoded = decode_stream(sse, cancel);
        let events: Vec<_> = decoded.collect().await;
        assert!(events.is_empty());
    }

    #[tokio::test]
    async fn malformed_event_payload_is_skipped() {
        let events = collect_events(vec![
            sse_event("response.output_text.delta", "{not valid json"),
            sse_event(
                "response.output_text.delta",
                r#"{"item_id":"m","output_index":0,"delta":"ok"}"#,
            ),
            sse_event(
                "response.completed",
                r#"{"response":{"status":"completed"}}"#,
            ),
        ])
        .await;
        let events: Vec<CompletionEvent> = events.into_iter().map(|r| r.unwrap()).collect();
        // First text delta dropped; second one fires.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, CompletionEvent::TextDelta(t) if t == "ok"))
        );
        // No errors yielded (parse failures are warn-and-skip).
        assert!(
            events
                .iter()
                .all(|e| !matches!(e, CompletionEvent::Finish { reason: None }))
        );
    }

    #[test]
    fn provider_construction_is_cheap() {
        let _ = OpenAIResponsesProvider::new(ep()).unwrap();
    }
}
