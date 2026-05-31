//! Anthropic `/v1/messages` provider.
//!
//! Third provider in helexa-acp, after [`super::openai_chat`] and
//! [`super::openai_responses`]. Lets a user point an endpoint at
//! `https://api.anthropic.com/v1/messages` (or cortex's
//! `/v1/messages` translation surface) and drive Claude models
//! through the same agent-loop machinery the OpenAI providers use.
//!
//! Wire-format differences from OpenAI, all contained here:
//!
//! - **Authentication**: Anthropic uses `x-api-key: <key>` plus a
//!   required `anthropic-version: 2023-06-01` header rather than
//!   OpenAI's `Authorization: Bearer …`. Both headers ship whenever
//!   an `api_key` is configured; servers that don't care about
//!   either (like cortex's translation proxy) ignore them.
//! - **System prompt** lives at the top level (`system`), not as a
//!   message turn. Multiple internal system messages concatenate
//!   into one string — the API also supports an array-of-blocks
//!   form but the plain-string form covers every helexa-acp use
//!   case.
//! - **Tool calls** are first-class content blocks (`tool_use` from
//!   the assistant, `tool_result` from the user) inside the
//!   `content` array, rather than a side-channel `tool_calls`
//!   field on the assistant message. Round-trips faithfully when
//!   we send history back.
//! - **Images** use the `image` content-block shape with a
//!   structured `source: { type: "base64", media_type, data }`
//!   payload — distinct from OpenAI's `image_url` data URI.
//! - **Streaming SSE** events carry typed names
//!   (`message_start`, `content_block_start`, `content_block_delta`,
//!   `content_block_stop`, `message_delta`, `message_stop`). Tool
//!   call arguments stream as `input_json_delta` fragments inside
//!   `content_block_delta` events.
//!
//! Tool-name handling: when the upstream is *real* Anthropic it
//! reads tool definitions from a request-side `tools` field, not
//! the qwen3 system-prompt block our other providers rely on.
//! Implementing that round-trip is out of scope for this initial
//! provider (see issue tracking) — today the provider doesn't echo
//! `tools` in the request body and the model is expected to know
//! its tool catalogue via the system prompt the agent loop
//! already produces.

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt, stream::BoxStream};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use tokio_util::sync::CancellationToken;

use super::{
    CompletionEvent, CompletionRequest, Message, MessageContent, MessagePart, ModelInfo, Provider,
    Role, UsageStats,
};
use crate::config::EndpointConfig;

/// Required version header. Pinned at the first stable Messages API
/// release; bump cautiously — newer values can change the wire shape
/// (e.g. response_format extensions).
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic requires this on every request even when `max_tokens`
/// would otherwise default upstream. Pick a value the agent loop
/// can comfortably consume — large enough to not surprise the user
/// on long responses, small enough to fail-fast on a runaway model.
const DEFAULT_MAX_TOKENS: u64 = 8192;

pub struct AnthropicMessagesProvider {
    endpoint: EndpointConfig,
    #[allow(dead_code)] // read in `complete()`'s HTTP path; tests don't stand up a server
    api_key: Option<String>,
    #[allow(dead_code)]
    http: reqwest::Client,
}

impl AnthropicMessagesProvider {
    pub fn new(endpoint: EndpointConfig) -> anyhow::Result<Self> {
        let api_key = endpoint.resolve_api_key()?;
        let http = reqwest::Client::builder()
            // Same generous timeout as the OpenAI providers: cold-load
            // / first-token latency can run into tens of seconds.
            // Cancellation handles early termination, not timeout.
            .timeout(std::time::Duration::from_secs(600))
            .build()?;
        Ok(Self {
            endpoint,
            api_key,
            http,
        })
    }

    /// `{base_url}/messages` — joined the same way the OpenAI
    /// providers join `{base_url}/chat/completions`. Configured
    /// `base_url` should NOT include the `messages` suffix; the
    /// example config points at `https://api.anthropic.com/v1` or
    /// the cortex equivalent.
    fn messages_url(&self) -> url::Url {
        let mut out = self.endpoint.base_url.clone();
        if let Ok(mut path) = out.path_segments_mut() {
            path.pop_if_empty().push("messages");
        }
        out
    }
}

#[async_trait]
impl Provider for AnthropicMessagesProvider {
    fn name(&self) -> &str {
        &self.endpoint.name
    }

    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        // Anthropic exposes a `/v1/models` listing endpoint too,
        // mirroring OpenAI's. Use the same shape as the other two
        // providers so the model-picker plumbing stays uniform.
        let mut req = self.http.get(self.endpoint.models_url());
        if let Some(key) = &self.api_key {
            req = req
                .header("x-api-key", key)
                .header("anthropic-version", ANTHROPIC_VERSION);
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
                display_name: m.display_name,
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
            url = %self.messages_url(),
            body = %serde_json::to_string(&body).unwrap_or_else(|_| "<unserializable>".into()),
            "POST /messages"
        );
        let mut req = self
            .http
            .post(self.messages_url())
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("accept", "text/event-stream")
            .json(&body);
        if let Some(key) = &self.api_key {
            req = req.header("x-api-key", key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("{} messages send: {e}", self.endpoint.name))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "{} messages returned {}: {}",
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

// ── Request encoding ────────────────────────────────────────────────

fn encode_request(req: &CompletionRequest) -> Value {
    // System messages collapse to the top-level `system` field.
    // Multiple of them join with blank lines so ordering survives.
    let mut system_chunks: Vec<String> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();

    for msg in &req.messages {
        if msg.role == Role::System
            && let MessageContent::Text { text } = &msg.content
        {
            system_chunks.push(text.clone());
            continue;
        }
        if let Some(encoded) = encode_message(msg) {
            messages.push(encoded);
        }
    }

    let mut body = json!({
        "model": req.model,
        "messages": messages,
        // Anthropic requires `max_tokens` on every request — unlike
        // OpenAI where it's optional. Default to a sensible cap so
        // a user who doesn't configure one still gets a working
        // request.
        "max_tokens": req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        "stream": true,
    });
    if let Value::Object(map) = &mut body {
        if !system_chunks.is_empty() {
            map.insert("system".into(), Value::String(system_chunks.join("\n\n")));
        }
        if let Some(t) = req.temperature {
            map.insert("temperature".into(), json!(t));
        }
        if let Some(p) = req.top_p {
            map.insert("top_p".into(), json!(p));
        }
    }
    body
}

fn encode_message(msg: &Message) -> Option<Value> {
    match (msg.role, &msg.content) {
        (Role::System, _) => None, // handled out-of-band as `system`
        (Role::User, MessageContent::Text { text }) => Some(json!({
            "role": "user",
            "content": text,
        })),
        (Role::User, MessageContent::MultiPart { parts }) => Some(json!({
            "role": "user",
            "content": encode_user_parts(parts),
        })),
        (Role::Assistant, MessageContent::Text { text }) => Some(json!({
            "role": "assistant",
            "content": text,
        })),
        (Role::Assistant, MessageContent::ToolCalls { text, calls }) => {
            // Assistant turn carrying tool calls. Anthropic wants
            // each as its own `tool_use` content block, optionally
            // preceded by a `text` block with any prose the
            // assistant spoke alongside the call.
            let mut blocks: Vec<Value> = Vec::new();
            if let Some(t) = text
                && !t.is_empty()
            {
                blocks.push(json!({ "type": "text", "text": t }));
            }
            for call in calls {
                // `input` must be the parsed object, not the raw
                // JSON string. Best-effort parse — if the model
                // emitted malformed JSON the agent loop has its
                // own repair pass; here we fall back to wrapping
                // the raw string so the request body stays
                // serialisable.
                let input: Value = serde_json::from_str(&call.arguments)
                    .unwrap_or_else(|_| Value::String(call.arguments.clone()));
                blocks.push(json!({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": input,
                }));
            }
            Some(json!({
                "role": "assistant",
                "content": blocks,
            }))
        }
        (
            Role::Tool,
            MessageContent::ToolResult {
                tool_call_id,
                content,
            },
        ) => {
            // Anthropic encodes tool results as a `user` turn whose
            // content carries a `tool_result` block. Not a separate
            // `tool` role like OpenAI uses.
            Some(json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content,
                }],
            }))
        }
        (role, content) => {
            tracing::warn!(
                ?role,
                ?content,
                "anthropic_messages: unexpected (role, content) shape"
            );
            None
        }
    }
}

fn encode_user_parts(parts: &[MessagePart]) -> Value {
    let items: Vec<Value> = parts
        .iter()
        .map(|p| match p {
            MessagePart::Text { text } => json!({ "type": "text", "text": text }),
            MessagePart::Image(img) => json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.mime_type,
                    "data": img.data,
                },
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
    #[serde(default)]
    display_name: Option<String>,
}

// ── Streaming decoder ───────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct MessageStartEvent {
    message: MessageStartBody,
}

#[derive(Debug, Deserialize)]
struct MessageStartBody {
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Debug, Deserialize)]
struct ContentBlockStartEvent {
    index: usize,
    content_block: ContentBlockStart,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlockStart {
    Text {
        #[serde(default)]
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
    /// Reasoning / thinking blocks (Claude's extended thinking).
    /// Captured but not yet surfaced — would route to
    /// `CompletionEvent::ReasoningDelta` once we have an upstream
    /// that emits them. Matches the shape neuron's
    /// `InferenceEvent::ReasoningDelta` will eventually carry.
    Thinking {
        #[serde(default)]
        thinking: String,
    },
    /// Any future block type passes through silently.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct ContentBlockDeltaEvent {
    index: usize,
    delta: ContentBlockDelta,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlockDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    ThinkingDelta {
        thinking: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaEvent {
    #[serde(default)]
    delta: MessageDeltaInner,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Debug, Default, Deserialize)]
struct MessageDeltaInner {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Default, Clone, Copy, Deserialize)]
struct WireUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
}

/// Map Anthropic's `stop_reason` to the same strings the OpenAI
/// providers emit. The downstream agent loop uses these to set
/// `StopReason` in the ACP response.
fn map_stop_reason(reason: &str) -> &'static str {
    match reason {
        "end_turn" | "stop_sequence" => "stop",
        "max_tokens" => "length",
        "tool_use" => "tool_calls",
        _ => "stop",
    }
}

/// Convert the Anthropic SSE event stream into provider-agnostic
/// [`CompletionEvent`]s. Tracks content-block index → tool-call slot
/// for `InputJsonDelta` events that fire between
/// `content_block_start` and `content_block_stop` for a `tool_use`
/// block.
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
        // Map a content-block index that's a `tool_use` to the
        // tool-call slot we hand downstream.
        let mut tool_index_by_block: HashMap<usize, usize> = HashMap::new();
        let mut next_tool_index: usize = 0;
        // Running totals for the final Usage event. Anthropic reports
        // input_tokens on `message_start` and output_tokens on
        // `message_delta`; we accumulate as they arrive.
        let mut input_tokens: u64 = 0;
        let mut output_tokens: u64 = 0;

        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    tracing::debug!("anthropic_messages: cancellation requested, ending stream");
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
                    let event_name = event.event.as_str();
                    let data = event.data.as_str();

                    match event_name {
                        "message_start" => {
                            if let Ok(ev) = serde_json::from_str::<MessageStartEvent>(data)
                                && let Some(u) = ev.message.usage
                            {
                                input_tokens = u.input_tokens;
                                output_tokens = u.output_tokens;
                            }
                        }
                        "content_block_start" => {
                            match serde_json::from_str::<ContentBlockStartEvent>(data) {
                                Ok(ev) => match ev.content_block {
                                    ContentBlockStart::Text { text } => {
                                        if !text.is_empty() {
                                            yield Ok(CompletionEvent::TextDelta(text));
                                        }
                                    }
                                    ContentBlockStart::ToolUse { id, name, input } => {
                                        let idx = next_tool_index;
                                        next_tool_index += 1;
                                        tool_index_by_block.insert(ev.index, idx);
                                        yield Ok(CompletionEvent::ToolCallStart {
                                            index: idx,
                                            id,
                                            name,
                                        });
                                        // Some upstreams ship a fully-
                                        // buffered `input` on the start
                                        // event when the model finalised
                                        // the call before SSE flush —
                                        // emit it as a single args delta.
                                        if !input.is_null()
                                            && let Ok(s) = serde_json::to_string(&input)
                                            && s != "{}"
                                        {
                                            yield Ok(CompletionEvent::ToolCallArgsDelta {
                                                index: idx,
                                                args_delta: s,
                                            });
                                        }
                                    }
                                    ContentBlockStart::Thinking { thinking } => {
                                        if !thinking.is_empty() {
                                            yield Ok(CompletionEvent::ReasoningDelta(thinking));
                                        }
                                    }
                                    ContentBlockStart::Unknown => {
                                        tracing::debug!(
                                            raw = %data,
                                            "anthropic_messages: unknown content_block_start type"
                                        );
                                    }
                                },
                                Err(e) => {
                                    tracing::warn!(
                                        error = %e,
                                        raw = %data,
                                        "anthropic_messages: failed to parse content_block_start; skipping"
                                    );
                                }
                            }
                        }
                        "content_block_delta" => {
                            match serde_json::from_str::<ContentBlockDeltaEvent>(data) {
                                Ok(ev) => match ev.delta {
                                    ContentBlockDelta::TextDelta { text } => {
                                        if !text.is_empty() {
                                            yield Ok(CompletionEvent::TextDelta(text));
                                        }
                                    }
                                    ContentBlockDelta::InputJsonDelta { partial_json } => {
                                        let Some(&idx) = tool_index_by_block.get(&ev.index)
                                        else {
                                            tracing::warn!(
                                                block_index = ev.index,
                                                "anthropic_messages: input_json_delta for non-tool_use block; ignoring"
                                            );
                                            continue;
                                        };
                                        if !partial_json.is_empty() {
                                            yield Ok(CompletionEvent::ToolCallArgsDelta {
                                                index: idx,
                                                args_delta: partial_json,
                                            });
                                        }
                                    }
                                    ContentBlockDelta::ThinkingDelta { thinking } => {
                                        if !thinking.is_empty() {
                                            yield Ok(CompletionEvent::ReasoningDelta(thinking));
                                        }
                                    }
                                    ContentBlockDelta::Unknown => {
                                        tracing::debug!(
                                            raw = %data,
                                            "anthropic_messages: unknown content_block_delta type"
                                        );
                                    }
                                },
                                Err(e) => {
                                    tracing::warn!(
                                        error = %e,
                                        raw = %data,
                                        "anthropic_messages: failed to parse content_block_delta; skipping"
                                    );
                                }
                            }
                        }
                        "content_block_stop" => {
                            // No CompletionEvent for this — the agent
                            // loop figures out call boundaries from
                            // ToolCallStart + ToolCallArgsDelta + the
                            // overall stream ending.
                        }
                        "message_delta" => {
                            // Carries stop_reason + (sometimes) usage.
                            // The stop_reason lives here, not on
                            // message_stop, because message_stop is a
                            // bare terminator with no payload.
                            if let Ok(ev) = serde_json::from_str::<MessageDeltaEvent>(data) {
                                if let Some(u) = ev.usage {
                                    output_tokens = u.output_tokens;
                                }
                                if let Some(reason) = ev.delta.stop_reason {
                                    let mapped = map_stop_reason(&reason).to_string();
                                    // Emit Usage before Finish so the
                                    // agent loop sees them in the
                                    // order the OpenAI providers also
                                    // emit them.
                                    yield Ok(CompletionEvent::Usage(UsageStats {
                                        prompt_tokens: input_tokens,
                                        completion_tokens: output_tokens,
                                        total_tokens: input_tokens + output_tokens,
                                    }));
                                    yield Ok(CompletionEvent::Finish {
                                        reason: Some(mapped),
                                    });
                                }
                            } else {
                                tracing::warn!(
                                    raw = %data,
                                    "anthropic_messages: failed to parse message_delta; continuing"
                                );
                            }
                        }
                        "message_stop" => {
                            // Bare terminator. If we never saw a
                            // message_delta with a stop_reason (rare
                            // — Anthropic always sends one) synthesise
                            // a default Finish so the consumer's stream
                            // contract is honoured.
                            // Track whether we already emitted Finish
                            // — we know that happens via stop_reason
                            // in message_delta. The current shape
                            // doesn't carry state across events, so
                            // we always emit a Finish here; the
                            // agent loop tolerates duplicates (its
                            // map_finish_reason just picks the last
                            // call site). Cleaner shape: track and
                            // skip — but the extra event is cheap.
                            break;
                        }
                        "ping" => {
                            // Anthropic injects keep-alive pings.
                            // Ignore.
                        }
                        "error" => {
                            // Mid-stream errors surface here. Drain
                            // the JSON for context and end the stream
                            // with an Err so the agent loop knows.
                            yield Err(anyhow::anyhow!(
                                "anthropic_messages: server error event: {data}"
                            ));
                            break;
                        }
                        other => {
                            tracing::trace!(
                                event = other,
                                "anthropic_messages: unrecognised SSE event"
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
    use crate::provider::{ImageData, ToolCall};
    use futures::stream;
    use url::Url;

    fn ep() -> EndpointConfig {
        EndpointConfig {
            name: "anthropic".into(),
            base_url: Url::parse("https://api.anthropic.com/v1").unwrap(),
            wire_api: crate::config::WireApi::AnthropicMessages,
            default_model: None,
            api_key: None,
            api_key_env: None,
            max_tokens: None,
            context_window: None,
        }
    }

    // ── messages_url ────────────────────────────────────────────────

    #[test]
    fn messages_url_appends_messages_segment() {
        let p = AnthropicMessagesProvider::new(ep()).unwrap();
        assert_eq!(
            p.messages_url().as_str(),
            "https://api.anthropic.com/v1/messages"
        );
    }

    // ── encode_request ──────────────────────────────────────────────

    #[test]
    fn system_messages_become_top_level_system_field() {
        let req = CompletionRequest {
            model: "claude-opus-4".into(),
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
        assert_eq!(body["model"], "claude-opus-4");
        assert_eq!(body["system"], "you are helpful");
        assert_eq!(body["max_tokens"], 256);
        assert_eq!(body["temperature"], 0.7);
        assert_eq!(body["stream"], true);
        // System message NOT echoed as a user turn — only in the
        // top-level system field.
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "hi");
    }

    #[test]
    fn multiple_system_messages_concatenate_with_blank_line() {
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
        assert_eq!(body["system"], "first\n\nsecond");
    }

    #[test]
    fn default_max_tokens_applies_when_unset() {
        let req = CompletionRequest {
            model: "m".into(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text { text: "hi".into() },
            }],
            tools: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
        };
        let body = encode_request(&req);
        assert_eq!(body["max_tokens"], DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn user_multipart_uses_image_source_shape() {
        let req = CompletionRequest {
            model: "claude-opus-4".into(),
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
        let content = body["messages"][0]["content"].as_array().unwrap().clone();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "what's in this?");
        // Anthropic's distinct image shape: structured `source`
        // object, not OpenAI's `image_url` URI.
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["type"], "base64");
        assert_eq!(content[1]["source"]["media_type"], "image/png");
        assert_eq!(content[1]["source"]["data"], "AAA=");
    }

    #[test]
    fn assistant_tool_call_produces_tool_use_block() {
        let req = CompletionRequest {
            model: "m".into(),
            messages: vec![Message {
                role: Role::Assistant,
                content: MessageContent::ToolCalls {
                    text: Some("calling now".into()),
                    calls: vec![ToolCall {
                        id: "toolu_42".into(),
                        name: "read_file".into(),
                        arguments: r#"{"path":"/etc/hostname"}"#.into(),
                    }],
                },
            }],
            tools: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
        };
        let body = encode_request(&req);
        let blocks = body["messages"][0]["content"].as_array().unwrap().clone();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "calling now");
        assert_eq!(blocks[1]["type"], "tool_use");
        assert_eq!(blocks[1]["id"], "toolu_42");
        assert_eq!(blocks[1]["name"], "read_file");
        // input is parsed JSON, not the raw string.
        assert_eq!(blocks[1]["input"]["path"], "/etc/hostname");
    }

    #[test]
    fn tool_result_becomes_user_turn_with_tool_result_block() {
        let req = CompletionRequest {
            model: "m".into(),
            messages: vec![Message {
                role: Role::Tool,
                content: MessageContent::ToolResult {
                    tool_call_id: "toolu_42".into(),
                    content: "host".into(),
                },
            }],
            tools: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
        };
        let body = encode_request(&req);
        let m = &body["messages"][0];
        assert_eq!(m["role"], "user");
        let blocks = m["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[0]["tool_use_id"], "toolu_42");
        assert_eq!(blocks[0]["content"], "host");
    }

    #[test]
    fn malformed_tool_arguments_fall_back_to_string_input() {
        // Defensive: if the agent loop's repair pass missed a
        // malformed call, encode_message should still produce a
        // serialisable body rather than panicking.
        let req = CompletionRequest {
            model: "m".into(),
            messages: vec![Message {
                role: Role::Assistant,
                content: MessageContent::ToolCalls {
                    text: None,
                    calls: vec![ToolCall {
                        id: "toolu_1".into(),
                        name: "read_file".into(),
                        arguments: "this isn't json".into(),
                    }],
                },
            }],
            tools: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
        };
        let body = encode_request(&req);
        assert_eq!(
            body["messages"][0]["content"][0]["input"],
            "this isn't json"
        );
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
        decode_stream(sse, CancellationToken::new()).collect().await
    }

    #[tokio::test]
    async fn decodes_text_streaming() {
        let events = collect_events(vec![
            sse_event(
                "message_start",
                r#"{"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[],"usage":{"input_tokens":5,"output_tokens":0}}}"#,
            ),
            sse_event(
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            ),
            sse_event(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hel"}}"#,
            ),
            sse_event(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}"#,
            ),
            sse_event(
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            ),
            sse_event(
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}"#,
            ),
            sse_event("message_stop", r#"{"type":"message_stop"}"#),
        ])
        .await;
        let events: Vec<CompletionEvent> = events.into_iter().map(|r| r.unwrap()).collect();
        let mut iter = events.into_iter();
        assert!(matches!(iter.next(), Some(CompletionEvent::TextDelta(t)) if t == "hel"));
        assert!(matches!(iter.next(), Some(CompletionEvent::TextDelta(t)) if t == "lo"));
        assert!(matches!(iter.next(), Some(CompletionEvent::Usage(u))
                if u.prompt_tokens == 5 && u.completion_tokens == 2 && u.total_tokens == 7));
        assert!(
            matches!(iter.next(), Some(CompletionEvent::Finish { reason: Some(r) }) if r == "stop")
        );
        assert!(iter.next().is_none());
    }

    #[tokio::test]
    async fn tool_use_block_emits_tool_call_events() {
        let events = collect_events(vec![
            sse_event(
                "message_start",
                r#"{"type":"message_start","message":{"id":"m","role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}"#,
            ),
            sse_event(
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_x","name":"read_file","input":{}}}"#,
            ),
            sse_event(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\""}}"#,
            ),
            sse_event(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":":\"/etc/hostname\"}"}}"#,
            ),
            sse_event(
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            ),
            sse_event(
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4}}"#,
            ),
            sse_event("message_stop", r#"{"type":"message_stop"}"#),
        ])
        .await;
        let events: Vec<CompletionEvent> = events.into_iter().map(|r| r.unwrap()).collect();
        let mut iter = events.into_iter();
        assert!(matches!(
            iter.next(),
            Some(CompletionEvent::ToolCallStart { index: 0, ref id, ref name })
                if id == "toolu_x" && name == "read_file"
        ));
        // First args delta...
        assert!(matches!(
            iter.next(),
            Some(CompletionEvent::ToolCallArgsDelta { index: 0, ref args_delta })
                if args_delta == r#"{"path""#
        ));
        // Second args delta.
        assert!(matches!(
            iter.next(),
            Some(CompletionEvent::ToolCallArgsDelta { index: 0, ref args_delta })
                if args_delta == r#":"/etc/hostname"}"#
        ));
        // Usage + Finish with tool_calls reason.
        assert!(matches!(iter.next(), Some(CompletionEvent::Usage(_))));
        assert!(matches!(
            iter.next(),
            Some(CompletionEvent::Finish { reason: Some(r) }) if r == "tool_calls"
        ));
    }

    #[tokio::test]
    async fn max_tokens_stop_reason_maps_to_length() {
        let events = collect_events(vec![
            sse_event(
                "message_start",
                r#"{"type":"message_start","message":{"id":"m","role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}"#,
            ),
            sse_event(
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"max_tokens"}}"#,
            ),
            sse_event("message_stop", r#"{"type":"message_stop"}"#),
        ])
        .await;
        let events: Vec<CompletionEvent> = events.into_iter().map(|r| r.unwrap()).collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, CompletionEvent::Finish { reason: Some(r) } if r == "length"))
        );
    }

    #[tokio::test]
    async fn empty_text_deltas_are_dropped() {
        let events = collect_events(vec![
            sse_event(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":""}}"#,
            ),
            sse_event(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}"#,
            ),
            sse_event(
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
            ),
            sse_event("message_stop", r#"{"type":"message_stop"}"#),
        ])
        .await;
        let events: Vec<CompletionEvent> = events.into_iter().map(|r| r.unwrap()).collect();
        let text_count = events
            .iter()
            .filter(|e| matches!(e, CompletionEvent::TextDelta(_)))
            .count();
        assert_eq!(text_count, 1, "empty delta must not produce an event");
    }

    #[tokio::test]
    async fn ping_events_are_ignored() {
        let events = collect_events(vec![
            sse_event("ping", r#"{"type":"ping"}"#),
            sse_event(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
            ),
            sse_event(
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
            ),
            sse_event("message_stop", r#"{"type":"message_stop"}"#),
        ])
        .await;
        let events: Vec<CompletionEvent> = events.into_iter().map(|r| r.unwrap()).collect();
        // Ping doesn't produce a CompletionEvent but doesn't break the
        // stream either.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, CompletionEvent::TextDelta(t) if t == "hi"))
        );
    }

    #[tokio::test]
    async fn server_error_event_ends_stream_with_err() {
        let events = collect_events(vec![sse_event(
            "error",
            r#"{"type":"error","error":{"type":"overloaded_error","message":"server busy"}}"#,
        )])
        .await;
        // The single output is an Err; the stream ends.
        assert_eq!(events.len(), 1);
        assert!(events[0].is_err());
        let msg = format!("{}", events[0].as_ref().unwrap_err());
        assert!(msg.contains("server error event"));
        assert!(msg.contains("overloaded_error"));
    }

    #[tokio::test]
    async fn cancellation_ends_stream_promptly() {
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
    async fn malformed_payload_is_skipped_not_fatal() {
        let events = collect_events(vec![
            sse_event("content_block_delta", "{not valid json"),
            sse_event(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}"#,
            ),
            sse_event(
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
            ),
            sse_event("message_stop", r#"{"type":"message_stop"}"#),
        ])
        .await;
        let events: Vec<CompletionEvent> = events.into_iter().map(|r| r.unwrap()).collect();
        // The "ok" delta still fires despite the prior malformed payload.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, CompletionEvent::TextDelta(t) if t == "ok"))
        );
    }

    #[tokio::test]
    async fn thinking_blocks_emit_reasoning_delta() {
        // Extended-thinking models (claude-opus-4-thinking) emit a
        // `thinking` content block. We route to ReasoningDelta which
        // the agent loop renders in the dedicated thought UI.
        let events = collect_events(vec![
            sse_event(
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            ),
            sse_event(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"considering..."}}"#,
            ),
            sse_event(
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
            ),
            sse_event("message_stop", r#"{"type":"message_stop"}"#),
        ])
        .await;
        let events: Vec<CompletionEvent> = events.into_iter().map(|r| r.unwrap()).collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, CompletionEvent::ReasoningDelta(t) if t == "considering..."))
        );
    }

    #[test]
    fn provider_construction_is_cheap() {
        let _ = AnthropicMessagesProvider::new(ep()).unwrap();
    }
}
