//! Translation between OpenAI and Anthropic request/response envelopes.
//!
//! This is a stateless transformation — no context is carried between requests.

use crate::anthropic::{
    AnthropicContent, AnthropicUsage, ContentBlock, MessagesRequest, MessagesResponse, SystemPrompt,
};
use crate::openai::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageContent, Usage,
};
use serde_json::{Value, json};

/// Convert an Anthropic Messages request into an OpenAI ChatCompletion request.
///
/// This is the request half of the round trip Claude Code (and any
/// Anthropic-native client pointed at cortex via `ANTHROPIC_BASE_URL`)
/// exercises. The non-obvious work here is **tool translation**: the
/// Anthropic and OpenAI tool shapes differ, and neuron feeds whatever
/// `tools` array it receives straight into the HF chat template, which
/// iterates the OpenAI shape (`tool.function.name`,
/// `tool.function.parameters`). If we forwarded Anthropic-shaped tools
/// (`{name, description, input_schema}`) verbatim the template would
/// render empty/garbage definitions and the model would improvise an
/// unparseable tool-call format — exactly the
/// `<tool_use_name>…</tool_use_name>` text that leaks through to the
/// client. So we reshape here:
///
/// - tool **definitions**: `{name, description, input_schema}` →
///   `{type:"function", function:{name, description, parameters}}`
/// - `tool_choice`: Anthropic `{type:"auto"|"any"|"tool", name}` →
///   OpenAI `"auto"|"required"|{type:"function",function:{name}}`
/// - assistant `tool_use` content blocks → an OpenAI assistant message
///   carrying `tool_calls` (with `arguments` JSON-stringified)
/// - user `tool_result` content blocks → standalone `role:"tool"`
///   messages keyed by `tool_call_id`
pub fn anthropic_to_openai(req: MessagesRequest) -> ChatCompletionRequest {
    // Collect ALL system content into a single leading system message.
    // The top-level `system` field PLUS any `role:"system"` turns inside
    // `messages` (Claude Code injects extra system-role messages beyond
    // the top-level one) are merged into one message at index 0.
    //
    // This is load-bearing: most chat templates — Qwen3.6's among them —
    // hard-reject a system message anywhere but the start
    // (`raise_exception('System message must be at the beginning.')`),
    // and on that render error neuron silently falls back to a
    // template that renders NO tools at all, so the model gets zero
    // tool-format guidance and improvises an unparseable tool syntax —
    // tool calling breaks entirely. Merging keeps every system
    // instruction while satisfying the template.
    let mut system_parts: Vec<String> = Vec::new();
    if let Some(system) = req.system {
        system_parts.push(match system {
            SystemPrompt::Text(t) => t,
            SystemPrompt::Blocks(blocks) => system_blocks_to_text(&blocks),
        });
    }

    // Translate the conversation. A single Anthropic message can fan out
    // into several OpenAI messages (tool results split into their own
    // `role:"tool"` turns); `role:"system"` turns are pulled into the
    // accumulator above rather than emitted mid-stream.
    let mut convo: Vec<ChatMessage> = Vec::new();
    for msg in req.messages {
        if msg.role == "system" {
            system_parts.push(anthropic_content_to_text(msg.content));
            continue;
        }
        push_translated_message(&mut convo, &msg.role, msg.content);
    }

    let mut messages = Vec::new();
    if !system_parts.is_empty() {
        messages.push(ChatMessage {
            role: "system".into(),
            content: MessageContent::Text(system_parts.join("\n\n")),
            extra: Value::Null,
        });
    }
    messages.extend(convo);

    // Reshape `tools` / `tool_choice` (carried over from the request's
    // flattened `extra`) into the OpenAI shape neuron's chat template
    // expects. Computed-then-inserted to avoid borrowing `obj` across
    // the mutation.
    let mut extra = req.extra;
    if let Value::Object(obj) = &mut extra {
        let tools = obj.get("tools").and_then(anthropic_tools_to_openai);
        if let Some(tools) = tools {
            obj.insert("tools".into(), tools);
        }
        let tool_choice = obj
            .get("tool_choice")
            .and_then(anthropic_tool_choice_to_openai);
        if let Some(tc) = tool_choice {
            obj.insert("tool_choice".into(), tc);
        }
    }

    ChatCompletionRequest {
        model: req.model,
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: Some(req.max_tokens),
        stream: req.stream,
        extra,
    }
}

/// Translate one Anthropic message into one-or-more OpenAI messages,
/// appending them to `out`.
fn push_translated_message(out: &mut Vec<ChatMessage>, role: &str, content: AnthropicContent) {
    let blocks = match content {
        AnthropicContent::Text(t) => {
            out.push(ChatMessage {
                role: role.into(),
                content: MessageContent::Text(t),
                extra: Value::Null,
            });
            return;
        }
        AnthropicContent::Blocks(blocks) => blocks,
    };

    let mut text_segments: Vec<String> = Vec::new();
    let mut parts: Vec<Value> = Vec::new();
    let mut has_nontext_part = false;
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut tool_msgs: Vec<ChatMessage> = Vec::new();

    for block in blocks {
        match block.block_type.as_str() {
            "text" => {
                let t = block
                    .data
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                parts.push(json!({ "type": "text", "text": t }));
                text_segments.push(t);
            }
            "tool_use" => {
                // Anthropic `input` is a JSON object; OpenAI wants the
                // arguments as a JSON *string*.
                let id = block
                    .data
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("toolu_unknown");
                let name = block
                    .data
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let input = block
                    .data
                    .get("input")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": input.to_string(),
                    }
                }));
            }
            "tool_result" => {
                let tool_use_id = block
                    .data
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .unwrap_or("toolu_unknown");
                tool_msgs.push(ChatMessage {
                    role: "tool".into(),
                    content: MessageContent::Text(tool_result_content_to_string(&block.data)),
                    extra: json!({ "tool_call_id": tool_use_id }),
                });
            }
            "image" => {
                if let Some(part) = anthropic_image_to_openai(&block.data) {
                    parts.push(part);
                    has_nontext_part = true;
                }
            }
            _ => {
                // Unknown block kind: preserve it as a JSON part rather
                // than silently dropping it.
                parts.push(serde_json::to_value(&block).unwrap_or(Value::Null));
                has_nontext_part = true;
            }
        }
    }

    // Tool results become standalone `role:"tool"` turns and must
    // precede any residual content from the same Anthropic message.
    out.append(&mut tool_msgs);

    if !tool_calls.is_empty() {
        // An assistant turn that invoked tools. OpenAI carries the calls
        // in `tool_calls`; the visible text (if any) stays in `content`.
        out.push(ChatMessage {
            role: role.into(),
            content: MessageContent::Text(text_segments.join("")),
            extra: json!({ "tool_calls": tool_calls }),
        });
    } else if has_nontext_part {
        // Mixed content (images): forward as OpenAI content parts.
        out.push(ChatMessage {
            role: role.into(),
            content: MessageContent::Parts(parts),
            extra: Value::Null,
        });
    } else if !text_segments.is_empty() {
        out.push(ChatMessage {
            role: role.into(),
            content: MessageContent::Text(text_segments.join("")),
            extra: Value::Null,
        });
    }
    // else: the message was only tool_result blocks — already emitted
    // as `role:"tool"` turns above, nothing residual to add.
}

/// Extract plain text from an Anthropic `tool_result` block's `content`
/// (a string, or an array of `{type:"text", text}` blocks).
fn tool_result_content_to_string(data: &Value) -> String {
    match data.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => arr
            .iter()
            .map(|b| {
                if b.get("type").and_then(Value::as_str) == Some("text") {
                    b.get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string()
                } else {
                    b.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

/// Convert an Anthropic image block's `data` (`{source:{...}}`) into an
/// OpenAI `image_url` content part.
fn anthropic_image_to_openai(data: &Value) -> Option<Value> {
    let source = data.get("source")?;
    match source
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("base64")
    {
        "base64" => {
            let media = source
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("image/png");
            let b64 = source
                .get("data")
                .and_then(Value::as_str)
                .unwrap_or_default();
            Some(json!({
                "type": "image_url",
                "image_url": { "url": format!("data:{media};base64,{b64}") }
            }))
        }
        "url" => {
            let url = source
                .get("url")
                .and_then(Value::as_str)
                .unwrap_or_default();
            Some(json!({ "type": "image_url", "image_url": { "url": url } }))
        }
        _ => None,
    }
}

/// Reshape an Anthropic `tools` array into the OpenAI function-tool
/// shape. Returns `None` if the value isn't an array (left untouched).
fn anthropic_tools_to_openai(tools: &Value) -> Option<Value> {
    let arr = tools.as_array()?;
    let converted = arr
        .iter()
        .map(|t| {
            // Already OpenAI-shaped (a client mixing conventions, or a
            // re-translation): pass through unchanged.
            if t.get("type").and_then(Value::as_str) == Some("function")
                && t.get("function").is_some()
            {
                return t.clone();
            }
            let mut function = serde_json::Map::new();
            function.insert("name".into(), t.get("name").cloned().unwrap_or(Value::Null));
            if let Some(desc) = t.get("description") {
                function.insert("description".into(), desc.clone());
            }
            function.insert(
                "parameters".into(),
                t.get("input_schema")
                    .cloned()
                    .unwrap_or_else(|| json!({ "type": "object" })),
            );
            json!({ "type": "function", "function": Value::Object(function) })
        })
        .collect();
    Some(Value::Array(converted))
}

/// Map an Anthropic `tool_choice` to the OpenAI form.
fn anthropic_tool_choice_to_openai(tc: &Value) -> Option<Value> {
    match tc.get("type").and_then(Value::as_str)? {
        "auto" => Some(json!("auto")),
        "any" => Some(json!("required")),
        "none" => Some(json!("none")),
        "tool" => {
            let name = tc.get("name").and_then(Value::as_str).unwrap_or_default();
            Some(json!({ "type": "function", "function": { "name": name } }))
        }
        _ => None,
    }
}

/// Flatten Anthropic system content blocks (`[{type:"text", text}]`)
/// into a single string.
fn system_blocks_to_text(blocks: &[Value]) -> String {
    let joined = blocks
        .iter()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|b| b.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n");
    if joined.is_empty() {
        // Unusual shape — don't lose it.
        serde_json::to_string(blocks).unwrap_or_default()
    } else {
        joined
    }
}

/// Flatten an Anthropic message's content into plain text. Used to fold
/// `role:"system"` conversation turns into the leading system message;
/// non-text blocks (rare in a system turn) are JSON-stringified rather
/// than dropped.
fn anthropic_content_to_text(content: AnthropicContent) -> String {
    match content {
        AnthropicContent::Text(t) => t,
        AnthropicContent::Blocks(blocks) => blocks
            .iter()
            .map(|b| {
                if b.block_type == "text" {
                    b.data
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string()
                } else {
                    serde_json::to_value(b)
                        .ok()
                        .map(|v| v.to_string())
                        .unwrap_or_default()
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Convert an OpenAI ChatCompletion response into an Anthropic Messages response.
pub fn openai_to_anthropic(resp: ChatCompletionResponse) -> MessagesResponse {
    let choice = resp.choices.into_iter().next();

    let (content_text, stop_reason) = match choice {
        Some(c) => {
            let text = match c.message.content {
                MessageContent::Text(t) => t,
                MessageContent::Parts(parts) => serde_json::to_string(&parts).unwrap_or_default(),
            };
            let stop = c.finish_reason.map(|r| map_stop_reason(&r));
            (text, stop)
        }
        None => (String::new(), None),
    };

    let usage = resp.usage.unwrap_or(Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    });

    MessagesResponse {
        id: resp.id,
        response_type: "message".into(),
        role: "assistant".into(),
        content: vec![ContentBlock {
            block_type: "text".into(),
            data: json!({ "text": content_text }),
        }],
        model: resp.model,
        stop_reason,
        usage: AnthropicUsage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        },
        extra: Value::Null,
    }
}

// ── Streaming SSE translation (#24) ──────────────────────────────────

/// Map an OpenAI `finish_reason` to an Anthropic `stop_reason`.
pub fn map_stop_reason(openai: &str) -> String {
    match openai {
        "stop" => "end_turn".to_string(),
        "length" => "max_tokens".to_string(),
        "tool_calls" => "tool_use".to_string(),
        other => other.to_string(),
    }
}

/// Stateful OpenAI-SSE → Anthropic-SSE event translator.
///
/// Feed each parsed OpenAI [`crate::openai::ChatCompletionChunk`] to
/// [`on_chunk`](Self::on_chunk) and call [`finish`](Self::finish) on
/// `[DONE]` (or upstream EOF); both return ordered
/// `(event_name, payload)` pairs ready to be framed as
/// `event: <name>\ndata: <payload>\n\n`. The translation is stateless
/// across requests — one instance per stream — and never buffers
/// content: every text delta maps to a `content_block_delta`
/// immediately.
///
/// Event sequence produced (per Anthropic's streaming spec):
/// `message_start` → `content_block_start` / `content_block_delta`* /
/// `content_block_stop` (text and `tool_use` blocks, indexed) →
/// `message_delta` (stop_reason + output usage) → `message_stop`.
#[derive(Debug, Default)]
pub struct AnthropicStreamTranslator {
    started: bool,
    finished: bool,
    /// Index of the currently-open content block, with its kind.
    open_block: Option<(u32, OpenBlock)>,
    next_index: u32,
    stop_reason: Option<String>,
    usage: Option<Usage>,
    /// Visible text deltas counted as an output-token estimate for
    /// streams whose upstream never sends a usage frame (neuron emits
    /// one chunk per token, so this is exact there).
    text_deltas: u64,
}

#[derive(Debug, PartialEq, Eq)]
enum OpenBlock {
    Text,
    ToolUse,
}

impl AnthropicStreamTranslator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn on_chunk(&mut self, chunk: &crate::openai::ChatCompletionChunk) -> Vec<(String, Value)> {
        let mut out = Vec::new();
        if !self.started {
            self.started = true;
            out.push((
                "message_start".to_string(),
                json!({
                    "type": "message_start",
                    "message": {
                        // Upstream ids are opaque to Anthropic clients;
                        // prefix for shape-compatibility with msg_* ids.
                        "id": format!("msg_{}", chunk.id),
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": chunk.model,
                        "stop_reason": null,
                        "stop_sequence": null,
                        // Input tokens are unknown until (if ever) a
                        // usage frame arrives; corrected in
                        // message_delta. Anthropic clients sum deltas.
                        "usage": { "input_tokens": 0, "output_tokens": 0 }
                    }
                }),
            ));
        }

        if let Some(usage) = &chunk.usage {
            self.usage = Some(usage.clone());
        }

        for choice in &chunk.choices {
            if let Some(text) = choice.delta.get("content").and_then(Value::as_str)
                && !text.is_empty()
            {
                self.ensure_text_block(&mut out);
                self.text_deltas += 1;
                let index = self.open_block.as_ref().map(|(i, _)| *i).unwrap_or(0);
                out.push((
                    "content_block_delta".to_string(),
                    json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": { "type": "text_delta", "text": text }
                    }),
                ));
            }

            if let Some(calls) = choice.delta.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    let name = call
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str);
                    let arguments = call
                        .get("function")
                        .and_then(|f| f.get("arguments"))
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if let Some(name) = name {
                        // A named entry begins a new tool_use block.
                        self.close_open_block(&mut out);
                        let id = call
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("toolu_unknown");
                        let index = self.next_index;
                        self.next_index += 1;
                        self.open_block = Some((index, OpenBlock::ToolUse));
                        out.push((
                            "content_block_start".to_string(),
                            json!({
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": id,
                                    "name": name,
                                    "input": {}
                                }
                            }),
                        ));
                    }
                    if !arguments.is_empty()
                        && let Some((index, OpenBlock::ToolUse)) = &self.open_block
                    {
                        out.push((
                            "content_block_delta".to_string(),
                            json!({
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": arguments
                                }
                            }),
                        ));
                    }
                }
            }

            if let Some(reason) = &choice.finish_reason {
                self.stop_reason = Some(map_stop_reason(reason));
            }
        }
        out
    }

    /// Close the stream: emits the trailing block-stop, message_delta
    /// (stop_reason + output usage) and message_stop. Idempotent.
    pub fn finish(&mut self) -> Vec<(String, Value)> {
        let mut out = Vec::new();
        if self.finished || !self.started {
            self.finished = true;
            return out;
        }
        self.finished = true;
        self.close_open_block(&mut out);
        let output_tokens = self
            .usage
            .as_ref()
            .map(|u| u.completion_tokens)
            .unwrap_or(self.text_deltas);
        let mut usage = json!({ "output_tokens": output_tokens });
        if let Some(u) = &self.usage {
            usage["input_tokens"] = json!(u.prompt_tokens);
        }
        out.push((
            "message_delta".to_string(),
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": self.stop_reason.as_deref().unwrap_or("end_turn"),
                    "stop_sequence": null
                },
                "usage": usage
            }),
        ));
        out.push((
            "message_stop".to_string(),
            json!({ "type": "message_stop" }),
        ));
        out
    }

    fn ensure_text_block(&mut self, out: &mut Vec<(String, Value)>) {
        match &self.open_block {
            Some((_, OpenBlock::Text)) => {}
            _ => {
                self.close_open_block(out);
                let index = self.next_index;
                self.next_index += 1;
                self.open_block = Some((index, OpenBlock::Text));
                out.push((
                    "content_block_start".to_string(),
                    json!({
                        "type": "content_block_start",
                        "index": index,
                        "content_block": { "type": "text", "text": "" }
                    }),
                ));
            }
        }
    }

    fn close_open_block(&mut self, out: &mut Vec<(String, Value)>) {
        if let Some((index, _)) = self.open_block.take() {
            out.push((
                "content_block_stop".to_string(),
                json!({ "type": "content_block_stop", "index": index }),
            ));
        }
    }
}

#[cfg(test)]
mod stream_tests {
    use super::*;
    use crate::openai::{ChatCompletionChunk, ChunkChoice};

    fn chunk(delta: Value, finish: Option<&str>) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: "abc123".into(),
            object: "chat.completion.chunk".into(),
            created: 1,
            model: "Qwen/Qwen3-8B".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason: finish.map(String::from),
                extra: Value::Null,
            }],
            usage: None,
            extra: Value::Null,
        }
    }

    fn names(events: &[(String, Value)]) -> Vec<&str> {
        events.iter().map(|(n, _)| n.as_str()).collect()
    }

    #[test]
    fn text_stream_produces_full_anthropic_sequence() {
        let mut t = AnthropicStreamTranslator::new();
        let mut all = Vec::new();
        all.extend(t.on_chunk(&chunk(json!({"role": "assistant"}), None)));
        all.extend(t.on_chunk(&chunk(json!({"content": "Hel"}), None)));
        all.extend(t.on_chunk(&chunk(json!({"content": "lo"}), None)));
        all.extend(t.on_chunk(&chunk(json!({}), Some("stop"))));
        all.extend(t.finish());

        assert_eq!(
            names(&all),
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        // message_start carries role/model; deltas carry the text.
        assert_eq!(all[0].1["message"]["model"], "Qwen/Qwen3-8B");
        assert_eq!(all[2].1["delta"]["text"], "Hel");
        assert_eq!(all[3].1["delta"]["text"], "lo");
        // stop → end_turn; without a usage frame the output count
        // falls back to the delta count (engine-exact for neuron's
        // one-chunk-per-token streams).
        let md = &all[5].1;
        assert_eq!(md["delta"]["stop_reason"], "end_turn");
        assert_eq!(md["usage"]["output_tokens"], 2);
    }

    #[test]
    fn length_maps_to_max_tokens_and_missing_finish_defaults_to_end_turn() {
        let mut t = AnthropicStreamTranslator::new();
        t.on_chunk(&chunk(json!({"content": "x"}), Some("length")));
        let fin = t.finish();
        assert_eq!(fin[1].1["delta"]["stop_reason"], "max_tokens");

        let mut t2 = AnthropicStreamTranslator::new();
        t2.on_chunk(&chunk(json!({"content": "x"}), None));
        let fin2 = t2.finish();
        assert_eq!(fin2[1].1["delta"]["stop_reason"], "end_turn");
    }

    #[test]
    fn tool_call_becomes_tool_use_block() {
        let mut t = AnthropicStreamTranslator::new();
        let mut all = Vec::new();
        all.extend(t.on_chunk(&chunk(json!({"content": "Let me check."}), None)));
        all.extend(t.on_chunk(&chunk(
            json!({"tool_calls": [{
                "index": 0,
                "id": "call_7",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Brno\"}"}
            }]}),
            None,
        )));
        all.extend(t.on_chunk(&chunk(json!({}), Some("tool_calls"))));
        all.extend(t.finish());

        assert_eq!(
            names(&all),
            vec![
                "message_start",
                "content_block_start", // text
                "content_block_delta", // text delta
                "content_block_stop",  // text closed by tool block
                "content_block_start", // tool_use
                "content_block_delta", // input_json_delta
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let tool_start = &all[4].1;
        assert_eq!(tool_start["content_block"]["type"], "tool_use");
        assert_eq!(tool_start["content_block"]["id"], "call_7");
        assert_eq!(tool_start["content_block"]["name"], "get_weather");
        assert_eq!(tool_start["index"], 1);
        assert_eq!(all[5].1["delta"]["partial_json"], "{\"city\":\"Brno\"}");
        assert_eq!(all[7].1["delta"]["stop_reason"], "tool_use");
    }

    #[test]
    fn usage_frame_feeds_message_delta() {
        let mut t = AnthropicStreamTranslator::new();
        t.on_chunk(&chunk(json!({"content": "hi"}), Some("stop")));
        let mut usage_chunk = chunk(json!({}), None);
        usage_chunk.choices.clear();
        usage_chunk.usage = Some(crate::openai::Usage {
            prompt_tokens: 225,
            completion_tokens: 42,
            total_tokens: 267,
        });
        t.on_chunk(&usage_chunk);
        let fin = t.finish();
        let md = &fin[1].1;
        assert_eq!(md["usage"]["output_tokens"], 42);
        assert_eq!(md["usage"]["input_tokens"], 225);
    }

    #[test]
    fn finish_is_idempotent_and_silent_without_start() {
        let mut t = AnthropicStreamTranslator::new();
        assert!(t.finish().is_empty(), "no events for an empty stream");
        assert!(t.finish().is_empty());

        let mut t2 = AnthropicStreamTranslator::new();
        t2.on_chunk(&chunk(json!({"content": "x"}), None));
        assert!(!t2.finish().is_empty());
        assert!(t2.finish().is_empty(), "second finish must emit nothing");
    }
}

#[cfg(test)]
mod request_tests {
    use super::*;
    use crate::openai::MessageContent;

    fn req(value: Value) -> MessagesRequest {
        serde_json::from_value(value).expect("valid MessagesRequest")
    }

    #[test]
    fn tool_definitions_reshape_to_openai_function_shape() {
        let r = req(json!({
            "model": "Qwen/Qwen3.6-27B",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "read the file"}],
            "tools": [{
                "name": "Read",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            }]
        }));
        let openai = anthropic_to_openai(r);
        let tools = openai
            .extra
            .get("tools")
            .and_then(Value::as_array)
            .expect("tools array");
        assert_eq!(tools.len(), 1);
        let t = &tools[0];
        assert_eq!(t["type"], "function");
        assert_eq!(t["function"]["name"], "Read");
        assert_eq!(t["function"]["description"], "Read a file");
        // input_schema is renamed to parameters, contents preserved.
        assert_eq!(
            t["function"]["parameters"]["properties"]["path"]["type"],
            "string"
        );
        assert!(t["function"].get("input_schema").is_none());
    }

    #[test]
    fn tool_choice_maps_each_variant() {
        let mk = |tc: Value| {
            let r = req(json!({
                "model": "m", "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": tc
            }));
            anthropic_to_openai(r)
                .extra
                .get("tool_choice")
                .cloned()
                .unwrap()
        };
        assert_eq!(mk(json!({"type": "auto"})), json!("auto"));
        assert_eq!(mk(json!({"type": "any"})), json!("required"));
        assert_eq!(mk(json!({"type": "none"})), json!("none"));
        assert_eq!(
            mk(json!({"type": "tool", "name": "Read"})),
            json!({"type": "function", "function": {"name": "Read"}})
        );
    }

    #[test]
    fn assistant_tool_use_block_becomes_openai_tool_calls() {
        let r = req(json!({
            "model": "m", "max_tokens": 8,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me read it."},
                    {"type": "tool_use", "id": "toolu_1", "name": "Read",
                     "input": {"path": "/etc/hosts"}}
                ]
            }]
        }));
        let openai = anthropic_to_openai(r);
        // One assistant message carrying both the text and the call.
        let m = openai.messages.last().expect("a message");
        assert_eq!(m.role, "assistant");
        match &m.content {
            MessageContent::Text(t) => assert_eq!(t, "Let me read it."),
            other => panic!("expected text content, got {other:?}"),
        }
        let calls = m
            .extra
            .get("tool_calls")
            .and_then(Value::as_array)
            .expect("tool_calls");
        assert_eq!(calls[0]["id"], "toolu_1");
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "Read");
        // arguments is a JSON *string*, not an object.
        assert_eq!(
            calls[0]["function"]["arguments"],
            "{\"path\":\"/etc/hosts\"}"
        );
    }

    #[test]
    fn user_tool_result_block_becomes_role_tool_message() {
        let r = req(json!({
            "model": "m", "max_tokens": 8,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1",
                     "content": "127.0.0.1 localhost"}
                ]
            }]
        }));
        let openai = anthropic_to_openai(r);
        assert_eq!(openai.messages.len(), 1);
        let m = &openai.messages[0];
        assert_eq!(m.role, "tool");
        assert_eq!(m.extra["tool_call_id"], "toolu_1");
        match &m.content {
            MessageContent::Text(t) => assert_eq!(t, "127.0.0.1 localhost"),
            other => panic!("expected text content, got {other:?}"),
        }
    }

    #[test]
    fn tool_result_with_block_array_content_is_flattened() {
        let r = req(json!({
            "model": "m", "max_tokens": 8,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t",
                     "content": [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}]}
                ]
            }]
        }));
        let openai = anthropic_to_openai(r);
        match &openai.messages[0].content {
            MessageContent::Text(t) => assert_eq!(t, "line1line2"),
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[test]
    fn tool_result_then_text_emits_tool_turn_first() {
        // A user turn that carries a tool result *and* a follow-up
        // question must yield the tool message before the user text.
        let r = req(json!({
            "model": "m", "max_tokens": 8,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": "ok"},
                    {"type": "text", "text": "now what?"}
                ]
            }]
        }));
        let openai = anthropic_to_openai(r);
        assert_eq!(openai.messages.len(), 2);
        assert_eq!(openai.messages[0].role, "tool");
        assert_eq!(openai.messages[1].role, "user");
        match &openai.messages[1].content {
            MessageContent::Text(t) => assert_eq!(t, "now what?"),
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[test]
    fn system_blocks_flatten_to_text_not_json() {
        let r = req(json!({
            "model": "m", "max_tokens": 8,
            "system": [{"type": "text", "text": "You are helpful."}],
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let openai = anthropic_to_openai(r);
        let sys = &openai.messages[0];
        assert_eq!(sys.role, "system");
        match &sys.content {
            MessageContent::Text(t) => assert_eq!(t, "You are helpful."),
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[test]
    fn system_role_messages_merge_into_one_leading_system() {
        // Claude Code's shape: a top-level `system` PLUS a `role:"system"`
        // turn inside `messages` (which Qwen3.6's template rejects unless
        // it's first). Both must merge into a single leading system msg.
        let r = req(json!({
            "model": "m", "max_tokens": 8,
            "system": "TOP LEVEL SYSTEM",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "INJECTED SYSTEM"},
                {"role": "user", "content": "do it"}
            ],
            "tools": [{"name": "noop", "input_schema": {"type": "object"}}]
        }));
        let openai = anthropic_to_openai(r);
        // Exactly one system message, at index 0, merging both parts.
        let systems: Vec<usize> = openai
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == "system")
            .map(|(i, _)| i)
            .collect();
        assert_eq!(systems, vec![0], "one system message, at the front");
        match &openai.messages[0].content {
            MessageContent::Text(t) => {
                assert!(t.contains("TOP LEVEL SYSTEM"));
                assert!(t.contains("INJECTED SYSTEM"));
            }
            other => panic!("expected text, got {other:?}"),
        }
        // The two real user turns survive, in order, after the system.
        let roles: Vec<&str> = openai.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["system", "user", "user"]);
    }

    #[test]
    fn already_openai_shaped_tools_pass_through() {
        let r = req(json!({
            "model": "m", "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}]
        }));
        let openai = anthropic_to_openai(r);
        let tools = openai.extra.get("tools").and_then(Value::as_array).unwrap();
        assert_eq!(tools[0]["function"]["name"], "x");
    }
}
