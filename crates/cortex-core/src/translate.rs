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
pub fn anthropic_to_openai(req: MessagesRequest) -> ChatCompletionRequest {
    let mut messages = Vec::new();

    // Anthropic `system` field becomes a system message.
    if let Some(system) = req.system {
        let content = match system {
            SystemPrompt::Text(t) => t,
            SystemPrompt::Blocks(blocks) => serde_json::to_string(&blocks).unwrap_or_default(),
        };
        messages.push(ChatMessage {
            role: "system".into(),
            content: MessageContent::Text(content),
            extra: Value::Null,
        });
    }

    // Convert message roles and content.
    for msg in req.messages {
        let content = match msg.content {
            AnthropicContent::Text(t) => MessageContent::Text(t),
            AnthropicContent::Blocks(blocks) => {
                // For simple text-only blocks, extract the text.
                // For mixed content (images, etc.), pass as parts.
                if blocks.len() == 1 && blocks[0].block_type == "text" {
                    let text = blocks[0]
                        .data
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    MessageContent::Text(text)
                } else {
                    MessageContent::Parts(blocks.into_iter().map(|b| json!(b)).collect())
                }
            }
        };
        messages.push(ChatMessage {
            role: msg.role,
            content,
            extra: Value::Null,
        });
    }

    ChatCompletionRequest {
        model: req.model,
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: Some(req.max_tokens),
        stream: req.stream,
        extra: req.extra,
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
