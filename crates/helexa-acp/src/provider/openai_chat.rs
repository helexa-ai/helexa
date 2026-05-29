//! OpenAI `/v1/chat/completions` provider.
//!
//! Covers cortex, LM Studio, Ollama (compat mode), OpenRouter, and
//! OpenAI itself. The wire format is well-documented and stable;
//! tool calls follow the `tools` request param + `tool_calls`
//! response delta convention shared by every reasonably-modern
//! OpenAI-compatible server.

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt, stream::BoxStream};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use super::{
    CompletionEvent, CompletionRequest, Message, MessageContent, ModelInfo, Provider, Role,
    ToolSpec, UsageStats,
};
use crate::config::EndpointConfig;

// Several fields and types in this module are only used through the
// async HTTP path in `complete()` and `list_models()`. Tests don't
// stand up a mock HTTP server (we'd be over-engineering for the
// payoff), so clippy's dead-code pass under `--tests` flags them.
// Each `allow(dead_code)` below names exactly what's exercised only
// at runtime, with a one-line rationale so the next reader can tell
// it's intentional.
pub struct OpenAIChatProvider {
    endpoint: EndpointConfig,
    /// Read by `list_models` and `complete` (bearer auth header).
    #[allow(dead_code)]
    api_key: Option<String>,
    /// Read by `list_models` and `complete` (request builder).
    #[allow(dead_code)]
    http: reqwest::Client,
}

impl OpenAIChatProvider {
    pub fn new(endpoint: EndpointConfig) -> anyhow::Result<Self> {
        let api_key = endpoint.resolve_api_key()?;
        let http = reqwest::Client::builder()
            // Generous timeout: cortex may need to cold-load a model
            // before serving the first chunk, which can be tens of
            // seconds. We rely on cancellation for early termination,
            // not on timeout.
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
impl Provider for OpenAIChatProvider {
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
        // Diagnostics for "the model isn't using tools" issues:
        // at debug level we log the full body so an operator can
        // confirm `tools` is in the request and inspect message
        // shapes. Stays at debug because chat history can be large.
        tracing::debug!(
            endpoint = %self.endpoint.name,
            url = %self.endpoint.chat_completions_url(),
            body = %serde_json::to_string(&body).unwrap_or_else(|_| "<unserializable>".into()),
            "POST /chat/completions"
        );
        let mut req = self
            .http
            .post(self.endpoint.chat_completions_url())
            .json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("{} chat_completion send: {e}", self.endpoint.name))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "{} chat_completion returned {}: {}",
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ToolCall;
    use futures::stream;
    use url::Url;

    fn ep() -> EndpointConfig {
        EndpointConfig {
            name: "test".into(),
            base_url: Url::parse("http://localhost:9999/v1").unwrap(),
            wire_api: crate::config::WireApi::OpenAiChat,
            default_model: None,
            api_key: None,
            api_key_env: None,
            max_tokens: None,
            context_window: None,
        }
    }

    #[test]
    fn encodes_text_only_request() {
        let req = CompletionRequest {
            model: "helexa/large".into(),
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
        assert_eq!(body["model"], "helexa/large");
        assert_eq!(body["stream"], true);
        assert_eq!(body["temperature"], 0.7);
        assert_eq!(body["max_tokens"], 256);
        assert!(body.get("top_p").is_none(), "absent options are omitted");
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "hi");
        assert!(body.get("tools").is_none(), "empty tools omitted");
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn encodes_tool_call_round_trip() {
        let req = CompletionRequest {
            model: "x".into(),
            messages: vec![
                Message {
                    role: Role::Assistant,
                    content: MessageContent::ToolCalls {
                        text: Some("calling read_file".into()),
                        calls: vec![ToolCall {
                            id: "call_1".into(),
                            name: "read_file".into(),
                            arguments: "{\"path\":\"/tmp/a.txt\"}".into(),
                        }],
                    },
                },
                Message {
                    role: Role::Tool,
                    content: MessageContent::ToolResult {
                        tool_call_id: "call_1".into(),
                        content: "file contents".into(),
                    },
                },
            ],
            tools: vec![ToolSpec {
                name: "read_file".into(),
                description: "Read a file".into(),
                parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            }],
            temperature: None,
            top_p: None,
            max_tokens: None,
        };
        let body = encode_request(&req);
        // Tool defs flow through as a courtesy to any future
        // strict-OpenAI backend; today's Qwen3 path puts them in
        // the prompt instead.
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools[0]["function"]["name"], "read_file");

        // Qwen3 wire shape for the assistant turn: tool calls are
        // inline in `content` as `<tool_call>{…}</tool_call>` blocks,
        // *not* in a structured `tool_calls` field.
        let asst = &body["messages"][0];
        assert_eq!(asst["role"], "assistant");
        assert!(
            asst.get("tool_calls").is_none(),
            "tool_calls should not be set"
        );
        let content = asst["content"].as_str().expect("content is a string");
        assert!(
            content.starts_with("calling read_file\n<tool_call>"),
            "content was: {content}"
        );
        assert!(content.contains(r#""name":"read_file""#));
        assert!(content.contains(r#""path":"/tmp/a.txt""#));
        assert!(content.ends_with("</tool_call>"));

        // Qwen3 wire shape for the tool result: a user-role turn
        // wrapped in `<tool_response>`. No `role: "tool"`.
        let tool = &body["messages"][1];
        assert_eq!(tool["role"], "user");
        assert!(tool.get("tool_call_id").is_none());
        let tool_content = tool["content"].as_str().expect("content is a string");
        assert_eq!(
            tool_content,
            "<tool_response>\nfile contents\n</tool_response>"
        );
    }

    /// Build a fake eventsource stream from canned SSE `data:` lines.
    fn fake_sse(
        lines: Vec<&'static str>,
    ) -> impl Stream<
        Item = std::result::Result<
            eventsource_stream::Event,
            eventsource_stream::EventStreamError<reqwest::Error>,
        >,
    > {
        stream::iter(lines.into_iter().map(|data| {
            Ok(eventsource_stream::Event {
                event: "message".into(),
                data: data.into(),
                id: String::new(),
                retry: None,
            })
        }))
    }

    #[tokio::test]
    async fn decodes_text_then_finish() {
        let sse = fake_sse(vec![
            r#"{"choices":[{"delta":{"content":"hel"},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
            r#"{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}"#,
            "[DONE]",
        ]);
        let stream = decode_stream(sse, CancellationToken::new());
        let events: Vec<_> = stream.collect().await;
        let events: Vec<_> = events.into_iter().map(|r| r.unwrap()).collect();

        assert!(matches!(&events[0], CompletionEvent::TextDelta(s) if s == "hel"));
        assert!(matches!(&events[1], CompletionEvent::TextDelta(s) if s == "lo"));
        assert!(
            matches!(&events[2], CompletionEvent::Finish { reason } if reason.as_deref() == Some("stop"))
        );
        assert!(matches!(&events[3], CompletionEvent::Usage(u) if u.total_tokens == 7));
        assert_eq!(events.len(), 4);
    }

    #[tokio::test]
    async fn decodes_qwen3_inline_think_block_to_reasoning_deltas() {
        // Qwen3-shaped output: a `<think>…</think>` block lives
        // inside `delta.content`. The decoder should route bytes
        // inside the block to ReasoningDelta and the surrounding
        // content to TextDelta. Marker boundaries split across
        // chunks to exercise the parser's prefix-hold logic.
        let sse = fake_sse(vec![
            r#"{"choices":[{"delta":{"content":"<thi"}}]}"#,
            r#"{"choices":[{"delta":{"content":"nk>internal reasoning</thi"}}]}"#,
            r#"{"choices":[{"delta":{"content":"nk>visible answer"}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
            "[DONE]",
        ]);
        let events: Vec<_> = decode_stream(sse, CancellationToken::new())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let text: String = events
            .iter()
            .filter_map(|e| match e {
                CompletionEvent::TextDelta(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        let reasoning: String = events
            .iter()
            .filter_map(|e| match e {
                CompletionEvent::ReasoningDelta(r) => Some(r.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "visible answer");
        assert_eq!(reasoning, "internal reasoning");
        assert!(matches!(
            events.last(),
            Some(CompletionEvent::Finish { reason }) if reason.as_deref() == Some("stop")
        ));
    }

    #[tokio::test]
    async fn decodes_qwen3_inline_tool_call_from_content_stream() {
        // Qwen3-shaped output: `<tool_call>{…}</tool_call>` inside
        // ordinary `delta.content`, split across multiple chunks at
        // arbitrary byte boundaries.
        let sse = fake_sse(vec![
            r#"{"choices":[{"delta":{"content":"sure, let me read it.\n<too"}}]}"#,
            r#"{"choices":[{"delta":{"content":"l_call>\n{\"name\":\"read_file\","}}]}"#,
            r#"{"choices":[{"delta":{"content":"\"arguments\":{\"path\":\"/etc/hostname\"}}\n</tool_call>"}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
            "[DONE]",
        ]);
        let events: Vec<_> = decode_stream(sse, CancellationToken::new())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // Concatenated text deltas should equal the leading prose
        // (everything before `<tool_call>`).
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                CompletionEvent::TextDelta(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "sure, let me read it.\n");
        // Exactly one structured tool call.
        assert!(matches!(
            events.iter().find(|e| matches!(e, CompletionEvent::ToolCallStart { .. })),
            Some(CompletionEvent::ToolCallStart { index: 0, name, .. }) if name == "read_file"
        ));
        let args: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                CompletionEvent::ToolCallArgsDelta { args_delta, .. } => Some(args_delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(args.len(), 1);
        assert!(args[0].contains(r#""path":"/etc/hostname""#));
        // Finish reason still propagates.
        assert!(matches!(
            events.last(),
            Some(CompletionEvent::Finish { reason }) if reason.as_deref() == Some("stop")
        ));
    }

    #[tokio::test]
    async fn decodes_tool_call_progressively() {
        let sse = fake_sse(vec![
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read_file"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"pa"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"/tmp/a\"}"}}]}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            "[DONE]",
        ]);
        let events: Vec<_> = decode_stream(sse, CancellationToken::new())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        assert!(matches!(
            &events[0],
            CompletionEvent::ToolCallStart { index: 0, id, name }
            if id == "c1" && name == "read_file"
        ));
        assert!(matches!(
            &events[1],
            CompletionEvent::ToolCallArgsDelta { index: 0, args_delta }
            if args_delta == "{\"pa"
        ));
        assert!(matches!(
            &events[2],
            CompletionEvent::ToolCallArgsDelta { index: 0, args_delta }
            if args_delta == "th\":\"/tmp/a\"}"
        ));
        assert!(matches!(
            &events[3],
            CompletionEvent::Finish { reason } if reason.as_deref() == Some("tool_calls")
        ));
    }

    #[tokio::test]
    async fn cancellation_ends_stream() {
        let sse = fake_sse(vec![
            r#"{"choices":[{"delta":{"content":"hello"}}]}"#,
            // These chunks should NOT be consumed once we cancel.
            r#"{"choices":[{"delta":{"content":" world"}}]}"#,
        ]);
        let cancel = CancellationToken::new();
        cancel.cancel(); // pre-cancel so the first select! arm wins.
        let events: Vec<_> = decode_stream(sse, cancel).collect().await;
        assert!(events.is_empty(), "cancelled stream yields nothing");
    }

    #[tokio::test]
    async fn skips_malformed_chunks() {
        let sse = fake_sse(vec![
            r#"{"choices":[{"delta":{"content":"before"}}]}"#,
            r#"not valid json"#,
            r#"{"choices":[{"delta":{"content":"after"}}]}"#,
            "[DONE]",
        ]);
        let events: Vec<_> = decode_stream(sse, CancellationToken::new())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        // The bad chunk is skipped with a warn; the bracketing
        // chunks both come through.
        assert!(matches!(&events[0], CompletionEvent::TextDelta(s) if s == "before"));
        assert!(matches!(&events[1], CompletionEvent::TextDelta(s) if s == "after"));
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn provider_construction_is_cheap() {
        // Ensures construction doesn't accidentally make any HTTP calls
        // — important because helexa-acp builds a provider per
        // configured endpoint at startup, before the editor has
        // necessarily connected.
        let p = OpenAIChatProvider::new(ep()).expect("construction");
        assert_eq!(p.name(), "test");
    }
}

// ── Request encoding ────────────────────────────────────────────────

fn encode_request(req: &CompletionRequest) -> Value {
    let messages: Vec<Value> = req.messages.iter().map(encode_message).collect();
    let mut body = json!({
        "model": req.model,
        "messages": messages,
        "stream": true,
    });
    if let Value::Object(map) = &mut body {
        if let Some(t) = req.temperature {
            map.insert("temperature".into(), json!(t));
        }
        if let Some(p) = req.top_p {
            map.insert("top_p".into(), json!(p));
        }
        if let Some(m) = req.max_tokens {
            map.insert("max_tokens".into(), json!(m));
        }
        if !req.tools.is_empty() {
            map.insert("tools".into(), encode_tools(&req.tools));
        }
        // Some servers (cortex via neuron, OpenAI) report usage at the
        // end of the stream only when explicitly requested.
        map.insert("stream_options".into(), json!({ "include_usage": true }));
    }
    body
}

fn encode_message(m: &Message) -> Value {
    match (m.role, &m.content) {
        (Role::System, MessageContent::Text { text }) => {
            json!({"role": "system", "content": text})
        }
        (Role::User, MessageContent::Text { text }) => json!({"role": "user", "content": text}),
        (Role::Assistant, MessageContent::Text { text }) => {
            json!({"role": "assistant", "content": text})
        }
        // Qwen3 wire shape: assistant turns that called tools come
        // back to the model with `<tool_call>{…}</tool_call>` blocks
        // inline in `content`, *not* via the structured `tool_calls`
        // field. Using the OpenAI shape here would invisibly drop
        // the tool calls from the model's context the next round,
        // because neuron's chat template only renders `content`.
        (Role::Assistant, MessageContent::ToolCalls { text, calls }) => {
            json!({
                "role": "assistant",
                "content": crate::qwen3::render_assistant_with_tool_calls(text.as_deref(), calls),
            })
        }
        // Qwen3 convention: tool results live in a *user* turn
        // wrapped in `<tool_response>…</tool_response>`. The model
        // wasn't trained on a separate `role: "tool"`.
        (
            Role::Tool,
            MessageContent::ToolResult {
                tool_call_id: _,
                content,
            },
        ) => json!({
            "role": "user",
            "content": crate::qwen3::render_tool_response(content),
        }),
        (role, content) => {
            tracing::warn!(
                ?role,
                ?content,
                "encode_message: unexpected (role, content) shape"
            );
            json!({"role": role_str(role), "content": content_as_text(content)})
        }
    }
}

fn role_str(r: Role) -> &'static str {
    match r {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn content_as_text(c: &MessageContent) -> String {
    match c {
        MessageContent::Text { text } => text.clone(),
        MessageContent::ToolCalls { text, .. } => text.clone().unwrap_or_default(),
        MessageContent::ToolResult { content, .. } => content.clone(),
    }
}

fn encode_tools(tools: &[ToolSpec]) -> Value {
    let arr: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            })
        })
        .collect();
    Value::Array(arr)
}

// ── Response decoding ───────────────────────────────────────────────

// Both types are deserialised through `list_models()`. Tests don't
// exercise that path (no mock HTTP server), so clippy --tests reports
// them as dead; in real use they're hit on every Zed model-picker
// refresh.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct WireModelsResponse {
    data: Vec<WireModelObject>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct WireModelObject {
    id: String,
}

#[derive(Debug, Deserialize)]
struct WireChunk {
    #[serde(default)]
    choices: Vec<WireChunkChoice>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Debug, Deserialize)]
struct WireChunkChoice {
    #[serde(default)]
    delta: WireDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct WireDelta {
    #[serde(default)]
    content: Option<String>,
    /// Some servers expose chain-of-thought text via this field
    /// (mirroring OpenAI's reasoning-model schema). When present we
    /// surface it as `ReasoningDelta`.
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<WireToolCallDelta>,
}

#[derive(Debug, Deserialize)]
struct WireToolCallDelta {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<WireFunctionDelta>,
}

#[derive(Debug, Default, Deserialize)]
struct WireFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct WireUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

/// Convert the eventsource-stream byte SSE into provider-agnostic
/// events. Bails the stream on the first parse failure with a logged
/// warning — partial state is preferable to silently corrupting a
/// conversation by skipping bad events.
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
        // Track which (index) tool calls we've already announced. The
        // For structured OpenAI tool calls (the canonical wire
        // format) we still want to dedupe ToolCallStart events per
        // index — only the first chunk for a given index carries the
        // id and name. This stays alongside the qwen3 text-stream
        // parser below; backends that *do* emit structured
        // tool_calls (a future strict-OpenAI endpoint) just keep
        // working without going through the Qwen3 path.
        let mut announced: std::collections::HashSet<usize> = Default::default();

        // Qwen3 wire path: tool calls come through `delta.content` as
        // literal `<tool_call>{…}</tool_call>` blocks. The parser
        // splits content into plain-text passthrough and
        // structured tool-call events, holding back only the suffix
        // bytes that could be the start of a marker.
        let mut qwen_parser = crate::qwen3::ToolCallParser::new();
        // Same shape, second stage: take the plain-text events out
        // of the tool-call parser and split off `<think>…</think>`
        // blocks into ReasoningDelta so Zed can render them in its
        // dedicated thought UI rather than the message pane.
        let mut think_parser = crate::qwen3::ThinkParser::new();

        let mut sse = Box::pin(sse);
        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    tracing::debug!("openai_chat: cancellation requested, ending stream");
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
                    let data = event.data;
                    if data == "[DONE]" {
                        break;
                    }
                    let chunk: WireChunk = match serde_json::from_str(&data) {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                raw = %data,
                                "openai_chat: failed to parse SSE chunk; skipping"
                            );
                            continue;
                        }
                    };
                    for choice in chunk.choices {
                        if let Some(text) = choice.delta.content
                            && !text.is_empty()
                        {
                            for ev in qwen_parser.feed(&text) {
                                match ev {
                                    crate::qwen3::ParserEvent::Text(t) if !t.is_empty() => {
                                        for tev in think_parser.feed(&t) {
                                            match tev {
                                                crate::qwen3::ThinkEvent::Text(s)
                                                    if !s.is_empty() =>
                                                {
                                                    yield Ok(CompletionEvent::TextDelta(s));
                                                }
                                                crate::qwen3::ThinkEvent::Reasoning(s)
                                                    if !s.is_empty() =>
                                                {
                                                    yield Ok(CompletionEvent::ReasoningDelta(s));
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                    crate::qwen3::ParserEvent::Text(_) => {}
                                    crate::qwen3::ParserEvent::Start { index, name } => {
                                        yield Ok(CompletionEvent::ToolCallStart {
                                            index,
                                            id: format!("call_{index}"),
                                            name,
                                        });
                                    }
                                    crate::qwen3::ParserEvent::Args { index, args_json } => {
                                        yield Ok(CompletionEvent::ToolCallArgsDelta {
                                            index,
                                            args_delta: args_json,
                                        });
                                    }
                                    crate::qwen3::ParserEvent::Malformed { raw } => {
                                        tracing::warn!(raw = %raw, "qwen3: malformed <tool_call> block; surfacing as Failed tool card");
                                        yield Ok(CompletionEvent::MalformedToolCall { raw });
                                    }
                                }
                            }
                        }
                        if let Some(reasoning) = choice.delta.reasoning_content
                            && !reasoning.is_empty()
                        {
                            yield Ok(CompletionEvent::ReasoningDelta(reasoning));
                        }
                        // Pass-through for backends that *do* emit
                        // structured tool_calls (a future strict
                        // OpenAI endpoint). Today cortex never
                        // populates this, so this branch stays cold.
                        for tc in choice.delta.tool_calls {
                            let idx = tc.index;
                            if announced.insert(idx) {
                                let id = tc.id.unwrap_or_default();
                                let name = tc
                                    .function
                                    .as_ref()
                                    .and_then(|f| f.name.clone())
                                    .unwrap_or_default();
                                yield Ok(CompletionEvent::ToolCallStart {
                                    index: idx,
                                    id,
                                    name,
                                });
                            }
                            if let Some(f) = tc.function
                                && let Some(args) = f.arguments
                                && !args.is_empty()
                            {
                                yield Ok(CompletionEvent::ToolCallArgsDelta {
                                    index: idx,
                                    args_delta: args,
                                });
                            }
                        }
                        if let Some(reason) = choice.finish_reason {
                            // Flush any tail bytes from the qwen
                            // parser before announcing the finish so
                            // the agent's stop-reason logic sees the
                            // complete picture (in particular, any
                            // trailing <tool_call> block that
                            // arrived without a close tag).
                            for ev in qwen_parser.finish() {
                                match ev {
                                    crate::qwen3::ParserEvent::Text(t) if !t.is_empty() => {
                                        for tev in think_parser.feed(&t) {
                                            match tev {
                                                crate::qwen3::ThinkEvent::Text(s)
                                                    if !s.is_empty() =>
                                                {
                                                    yield Ok(CompletionEvent::TextDelta(s));
                                                }
                                                crate::qwen3::ThinkEvent::Reasoning(s)
                                                    if !s.is_empty() =>
                                                {
                                                    yield Ok(CompletionEvent::ReasoningDelta(s));
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                    crate::qwen3::ParserEvent::Text(_) => {}
                                    crate::qwen3::ParserEvent::Start { index, name } => {
                                        yield Ok(CompletionEvent::ToolCallStart {
                                            index,
                                            id: format!("call_{index}"),
                                            name,
                                        });
                                    }
                                    crate::qwen3::ParserEvent::Args { index, args_json } => {
                                        yield Ok(CompletionEvent::ToolCallArgsDelta {
                                            index,
                                            args_delta: args_json,
                                        });
                                    }
                                    crate::qwen3::ParserEvent::Malformed { raw } => {
                                        tracing::warn!(raw = %raw, "qwen3: unterminated <tool_call> at stream end");
                                        yield Ok(CompletionEvent::MalformedToolCall { raw });
                                    }
                                }
                            }
                            // Flush the think parser too — any
                            // unclosed <think> at stream end becomes
                            // a final ReasoningDelta rather than
                            // being lost.
                            for tev in think_parser.finish() {
                                match tev {
                                    crate::qwen3::ThinkEvent::Text(s) if !s.is_empty() => {
                                        yield Ok(CompletionEvent::TextDelta(s));
                                    }
                                    crate::qwen3::ThinkEvent::Reasoning(s) if !s.is_empty() => {
                                        yield Ok(CompletionEvent::ReasoningDelta(s));
                                    }
                                    _ => {}
                                }
                            }
                            yield Ok(CompletionEvent::Finish { reason: Some(reason) });
                        }
                    }
                    if let Some(u) = chunk.usage {
                        yield Ok(CompletionEvent::Usage(UsageStats {
                            prompt_tokens: u.prompt_tokens,
                            completion_tokens: u.completion_tokens,
                            total_tokens: u.total_tokens,
                        }));
                    }
                }
            }
        }
    }
}
