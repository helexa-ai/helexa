//! Qwen3 tool-call wire conventions.
//!
//! Qwen3 (and the Hermes-derived chat templates it inherits) wires tool
//! use entirely through the prompt and the model's text output —
//! nothing on the server cares about the OpenAI `tools` API field.
//! This module owns both sides of that convention so the rest of
//! helexa-acp can stay generic.
//!
//! **System prompt** — a `# Tools` block is appended to the system
//! message describing every available function. Models trained on
//! this template recognise it and emit calls as
//! `<tool_call>{"name":"…","arguments":{…}}</tool_call>` inside the
//! normal content stream.
//!
//! **Streaming parse** — [`ToolCallParser`] is a small state machine
//! fed SSE content chunks. It emits a sequence of
//! [`ParserEvent`]s — plain text outside tool calls; `Start` + `Args`
//! events for each `<tool_call>` block. Marker detection is split-safe:
//! a chunk that ends with `<tool` is buffered until the next chunk
//! arrives, so even a one-byte-at-a-time stream produces the same
//! events as a single-buffer reparse would.
//!
//! **Multi-turn replay** — when helexa-acp re-sends the conversation
//! after a tool dispatch, the assistant turn that called the tool and
//! the tool result need to go back to the model in Qwen3 wire shape:
//! the assistant turn carries `<tool_call>` blocks inline in its
//! content, and the tool result rides in a user turn wrapped in
//! `<tool_response>…</tool_response>`. [`render_assistant_with_tool_calls`]
//! and [`render_tool_response`] handle those.

use serde_json::json;

use crate::provider::{ToolCall, ToolSpec};

/// One opening marker. Length 11.
const TOOL_CALL_OPEN: &str = "<tool_call>";
/// One closing marker. Length 12.
const TOOL_CALL_CLOSE: &str = "</tool_call>";

/// Reasoning open. Length 7.
const THINK_OPEN: &str = "<think>";
/// Reasoning close. Length 8.
const THINK_CLOSE: &str = "</think>";

// ── System-prompt-side rendering ────────────────────────────────────

/// Append-this-to-the-system-prompt block describing the available
/// tools in Qwen3's expected format. Returns the empty string if
/// `tools` is empty (no separator, no `# Tools` header — keeps the
/// prompt clean when tools are absent for any reason).
pub fn render_tool_block(tools: &[ToolSpec]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    out.push_str("\n\n# Tools\n\n");
    out.push_str(
        "You may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n",
    );
    out.push_str("<tools>\n");
    for spec in tools {
        // Each entry is one JSON object on its own line — newline-
        // delimited, no commas between entries. This is the format
        // Qwen3's training tokenisation expects.
        let entry = json!({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }
        });
        out.push_str(&serde_json::to_string(&entry).unwrap_or_default());
        out.push('\n');
    }
    out.push_str("</tools>\n\n");
    out.push_str(
        "For each function call, return a json object with function name \
         and arguments within <tool_call></tool_call> XML tags:\n\
         <tool_call>\n\
         {\"name\": <function-name>, \"arguments\": <args-json-object>}\n\
         </tool_call>",
    );
    out
}

// ── Multi-turn replay rendering ─────────────────────────────────────

/// Build the assistant-turn content the model expects when we replay
/// a turn that included tool calls. Format: any visible text first,
/// then one `<tool_call>{json}</tool_call>` block per call, joined by
/// newlines.
pub fn render_assistant_with_tool_calls(text: Option<&str>, calls: &[ToolCall]) -> String {
    let mut out = String::new();
    if let Some(t) = text
        && !t.is_empty()
    {
        out.push_str(t);
        if !calls.is_empty() {
            out.push('\n');
        }
    }
    for (i, call) in calls.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        // The arguments field on a `ToolCall` is a JSON-encoded
        // string; we want it inlined as an object inside the
        // tool_call body. Best-effort parse; if it isn't valid JSON,
        // pass the raw string through wrapped in quotes so the
        // emission stays well-formed.
        let args_value: serde_json::Value = serde_json::from_str(&call.arguments)
            .unwrap_or_else(|_| serde_json::Value::String(call.arguments.clone()));
        let body = json!({ "name": call.name, "arguments": args_value });
        out.push_str(TOOL_CALL_OPEN);
        out.push('\n');
        out.push_str(&serde_json::to_string(&body).unwrap_or_default());
        out.push('\n');
        out.push_str(TOOL_CALL_CLOSE);
    }
    out
}

/// Wrap a tool-result string in the Qwen3 `<tool_response>` block
/// that goes inside a `user` role message on the next turn.
pub fn render_tool_response(content: &str) -> String {
    format!("<tool_response>\n{content}\n</tool_response>")
}

// ── Streaming parser ────────────────────────────────────────────────

/// Events produced by [`ToolCallParser`]. Distinct from the
/// `CompletionEvent` enum because the parser is provider-agnostic —
/// the caller decides how to translate these into
/// `CompletionEvent::ToolCall*` and `TextDelta`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParserEvent {
    /// Plain text that lives outside any tool_call block.
    Text(String),
    /// Beginning of a tool call. The index increments per call within
    /// the same parser lifetime.
    Start { index: usize, name: String },
    /// JSON-encoded arguments for the most recent `Start`. Always
    /// follows its `Start` immediately; never split across multiple
    /// `Args` events for a single call (the parser buffers the whole
    /// `<tool_call>` body before emitting).
    Args { index: usize, args_json: String },
    /// Parser encountered a malformed `<tool_call>` body. Emitted so
    /// the agent can log and continue rather than crashing the
    /// conversation.
    Malformed { raw: String },
}

/// Streaming parser for Qwen3 tool calls embedded in the model's text
/// output. Feed it chunks via [`feed`](Self::feed); call
/// [`finish`](Self::finish) at end-of-stream to drain any trailing
/// buffered bytes.
///
/// Design notes:
///
/// - Markers (`<tool_call>` / `</tool_call>`) can be split across
///   chunks at any byte. The parser holds back exactly as much suffix
///   as could be the start of the marker it's currently looking for,
///   and no more.
/// - JSON inside a tool_call is held in a separate buffer until the
///   closing marker arrives. We don't try to stream-parse JSON; the
///   bodies are tiny (one function call) and assembling first
///   yields a much simpler implementation.
/// - Index is monotonic across the parser's lifetime — one
///   conversation turn can contain multiple `<tool_call>` blocks and
///   each gets its own index.
#[derive(Debug, Default)]
pub struct ToolCallParser {
    /// Unprocessed input bytes carried over between feeds.
    buffer: String,
    /// True while we're between `<tool_call>` and `</tool_call>`.
    in_tool_call: bool,
    /// Bytes accumulated inside the current `<tool_call>` block.
    tool_call_buf: String,
    /// Next tool-call index to assign.
    next_index: usize,
}

impl ToolCallParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn feed(&mut self, chunk: &str) -> Vec<ParserEvent> {
        self.buffer.push_str(chunk);
        self.drain()
    }

    /// End-of-stream: emit anything still in the buffers. An
    /// unterminated tool_call is reported as `Malformed` so the
    /// caller can decide what to surface to the user.
    pub fn finish(&mut self) -> Vec<ParserEvent> {
        let mut events = self.drain();
        if self.in_tool_call {
            let raw = std::mem::take(&mut self.tool_call_buf) + &std::mem::take(&mut self.buffer);
            events.push(ParserEvent::Malformed { raw });
            self.in_tool_call = false;
        } else if !self.buffer.is_empty() {
            events.push(ParserEvent::Text(std::mem::take(&mut self.buffer)));
        }
        events
    }

    fn drain(&mut self) -> Vec<ParserEvent> {
        let mut events = Vec::new();
        loop {
            if self.in_tool_call {
                if let Some(end) = self.buffer.find(TOOL_CALL_CLOSE) {
                    let body = &self.buffer[..end];
                    self.tool_call_buf.push_str(body);
                    self.buffer.drain(..end + TOOL_CALL_CLOSE.len());
                    self.emit_completed_tool_call(&mut events);
                    self.in_tool_call = false;
                } else {
                    // Hold back exactly the suffix that could be the
                    // start of `</tool_call>`. Everything before it
                    // is safely part of the call body.
                    let hold = longest_marker_prefix_suffix(&self.buffer, TOOL_CALL_CLOSE);
                    let safe = self.buffer.len() - hold;
                    if safe > 0 {
                        self.tool_call_buf.push_str(&self.buffer[..safe]);
                        self.buffer.drain(..safe);
                    }
                    return events;
                }
            } else if let Some(start) = self.buffer.find(TOOL_CALL_OPEN) {
                let text = &self.buffer[..start];
                if !text.is_empty() {
                    events.push(ParserEvent::Text(text.to_string()));
                }
                self.buffer.drain(..start + TOOL_CALL_OPEN.len());
                self.in_tool_call = true;
            } else {
                let hold = longest_marker_prefix_suffix(&self.buffer, TOOL_CALL_OPEN);
                let safe = self.buffer.len() - hold;
                if safe > 0 {
                    let text: String = self.buffer.drain(..safe).collect();
                    events.push(ParserEvent::Text(text));
                }
                return events;
            }
        }
    }

    fn emit_completed_tool_call(&mut self, events: &mut Vec<ParserEvent>) {
        let body = std::mem::take(&mut self.tool_call_buf);
        let trimmed = body.trim();
        let parsed: Result<ToolCallBody, _> = serde_json::from_str(trimmed);
        match parsed {
            Ok(call) => {
                let index = self.next_index;
                self.next_index += 1;
                let name = call.name;
                let args_json =
                    serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_string());
                events.push(ParserEvent::Start { index, name });
                events.push(ParserEvent::Args { index, args_json });
            }
            Err(_) => {
                events.push(ParserEvent::Malformed { raw: body });
            }
        }
    }
}

/// Returns the length of the longest suffix of `haystack` that is a
/// proper prefix of `needle`. Used to decide how many trailing bytes
/// to hold back when scanning for `needle`: anything that could
/// possibly be the start of `needle` is held; everything else is
/// safe to emit.
fn longest_marker_prefix_suffix(haystack: &str, needle: &str) -> usize {
    // Try prefixes of needle from longest to shortest; the first one
    // that matches as a suffix of haystack wins. O(|needle|^2) which
    // is fine — both markers are < 20 chars.
    let max = needle.len().min(haystack.len());
    for n in (1..=max).rev() {
        if !haystack.is_char_boundary(haystack.len() - n) || !needle.is_char_boundary(n) {
            continue;
        }
        if haystack.ends_with(&needle[..n]) {
            return n;
        }
    }
    0
}

#[derive(Debug, serde::Deserialize)]
struct ToolCallBody {
    name: String,
    // The model is supposed to emit a JSON object here; in practice
    // some Qwen3 variants stringify it. Deserialize-as-value handles
    // both.
    #[serde(default)]
    arguments: serde_json::Value,
}

// ── Think-block parser ──────────────────────────────────────────────

/// Events from [`ThinkParser`]. Plain text outside any `<think>`
/// block stays `Text`; bytes between `<think>` and `</think>` become
/// `Reasoning` so the agent can route them to a thought-channel
/// notification (Zed surfaces these in a dedicated UI affordance
/// rather than the main message pane).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThinkEvent {
    Text(String),
    Reasoning(String),
}

/// Streaming parser for Qwen3-style inline reasoning. Same
/// chunk-boundary discipline as [`ToolCallParser`]: hold back only
/// the suffix that could be the start of the marker we're scanning
/// for. Markers (`<think>`, `</think>`) never nest; a stray
/// `</think>` outside a block is emitted as text (the model
/// occasionally writes the tag conversationally).
#[derive(Debug, Default)]
pub struct ThinkParser {
    buffer: String,
    in_think: bool,
}

impl ThinkParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn feed(&mut self, chunk: &str) -> Vec<ThinkEvent> {
        self.buffer.push_str(chunk);
        self.drain()
    }

    /// Flush any buffered tail at end-of-stream. If we end mid-think
    /// (no closing tag arrived), emit what we have as reasoning so
    /// the partial thought isn't silently dropped.
    pub fn finish(&mut self) -> Vec<ThinkEvent> {
        let mut events = self.drain();
        if !self.buffer.is_empty() {
            let raw = std::mem::take(&mut self.buffer);
            if self.in_think {
                events.push(ThinkEvent::Reasoning(raw));
            } else {
                events.push(ThinkEvent::Text(raw));
            }
        }
        self.in_think = false;
        events
    }

    fn drain(&mut self) -> Vec<ThinkEvent> {
        let mut events = Vec::new();
        loop {
            if self.in_think {
                if let Some(end) = self.buffer.find(THINK_CLOSE) {
                    let body = self.buffer[..end].to_string();
                    if !body.is_empty() {
                        events.push(ThinkEvent::Reasoning(body));
                    }
                    self.buffer.drain(..end + THINK_CLOSE.len());
                    self.in_think = false;
                } else {
                    let hold = longest_marker_prefix_suffix(&self.buffer, THINK_CLOSE);
                    let safe = self.buffer.len() - hold;
                    if safe > 0 {
                        let r: String = self.buffer.drain(..safe).collect();
                        events.push(ThinkEvent::Reasoning(r));
                    }
                    return events;
                }
            } else if let Some(start) = self.buffer.find(THINK_OPEN) {
                let text = self.buffer[..start].to_string();
                if !text.is_empty() {
                    events.push(ThinkEvent::Text(text));
                }
                self.buffer.drain(..start + THINK_OPEN.len());
                self.in_think = true;
            } else {
                let hold = longest_marker_prefix_suffix(&self.buffer, THINK_OPEN);
                let safe = self.buffer.len() - hold;
                if safe > 0 {
                    let t: String = self.buffer.drain(..safe).collect();
                    events.push(ThinkEvent::Text(t));
                }
                return events;
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.to_string(),
            description: format!("desc of {name}"),
            parameters: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        }
    }

    // ── render_tool_block ───────────────────────────────────────────

    #[test]
    fn empty_tools_renders_empty() {
        assert_eq!(render_tool_block(&[]), "");
    }

    #[test]
    fn tool_block_contains_hermes_markers_and_each_function() {
        let block = render_tool_block(&[tool("read_file"), tool("write_file")]);
        assert!(block.contains("# Tools"));
        assert!(block.contains("<tools>"));
        assert!(block.contains("</tools>"));
        assert!(block.contains("\"name\":\"read_file\""));
        assert!(block.contains("\"name\":\"write_file\""));
        assert!(block.contains("<tool_call>"));
        assert!(block.contains("</tool_call>"));
    }

    // ── render_assistant_with_tool_calls ────────────────────────────

    #[test]
    fn renders_pure_text_when_no_calls() {
        let out = render_assistant_with_tool_calls(Some("hi"), &[]);
        assert_eq!(out, "hi");
    }

    #[test]
    fn renders_text_then_tool_call_block() {
        let calls = vec![ToolCall {
            id: "call_0".into(),
            name: "read_file".into(),
            arguments: r#"{"path":"/etc/hostname"}"#.into(),
        }];
        let out = render_assistant_with_tool_calls(Some("reading"), &calls);
        assert!(out.starts_with("reading\n<tool_call>"));
        assert!(out.contains(r#""name":"read_file""#));
        assert!(out.contains(r#""path":"/etc/hostname""#));
        assert!(out.ends_with("</tool_call>"));
    }

    #[test]
    fn multiple_calls_separated_by_newlines() {
        let calls = vec![
            ToolCall {
                id: "call_0".into(),
                name: "a".into(),
                arguments: "{}".into(),
            },
            ToolCall {
                id: "call_1".into(),
                name: "b".into(),
                arguments: "{}".into(),
            },
        ];
        let out = render_assistant_with_tool_calls(None, &calls);
        assert_eq!(out.matches("<tool_call>").count(), 2);
        assert_eq!(out.matches("</tool_call>").count(), 2);
    }

    #[test]
    fn invalid_arguments_json_is_wrapped_as_string() {
        let calls = vec![ToolCall {
            id: "call_0".into(),
            name: "x".into(),
            arguments: "not even json".into(),
        }];
        let out = render_assistant_with_tool_calls(None, &calls);
        // Wrapped as JSON string rather than breaking the envelope.
        assert!(out.contains(r#""arguments":"not even json""#));
    }

    // ── render_tool_response ────────────────────────────────────────

    #[test]
    fn tool_response_wraps_content() {
        let out = render_tool_response("hello world");
        assert_eq!(out, "<tool_response>\nhello world\n</tool_response>");
    }

    // ── longest_marker_prefix_suffix ────────────────────────────────

    #[test]
    fn marker_prefix_suffix_returns_longest_match() {
        assert_eq!(longest_marker_prefix_suffix("foo<tool", "<tool_call>"), 5);
        assert_eq!(longest_marker_prefix_suffix("foo<", "<tool_call>"), 1);
        assert_eq!(longest_marker_prefix_suffix("foo<bar", "<tool_call>"), 0);
        assert_eq!(longest_marker_prefix_suffix("foo", "<tool_call>"), 0);
        assert_eq!(longest_marker_prefix_suffix("", "<tool_call>"), 0);
        // Exact prefix length matches.
        assert_eq!(
            longest_marker_prefix_suffix("foo<tool_call", "<tool_call>"),
            10
        );
    }

    // ── ToolCallParser ──────────────────────────────────────────────

    fn drive(parser: &mut ToolCallParser, chunks: &[&str]) -> Vec<ParserEvent> {
        let mut events = Vec::new();
        for c in chunks {
            events.extend(parser.feed(c));
        }
        events.extend(parser.finish());
        events
    }

    #[test]
    fn plain_text_passes_through() {
        let mut p = ToolCallParser::new();
        let events = drive(&mut p, &["hello ", "world"]);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], ParserEvent::Text("hello ".to_string()));
        assert_eq!(events[1], ParserEvent::Text("world".to_string()));
    }

    #[test]
    fn single_complete_tool_call() {
        let mut p = ToolCallParser::new();
        let input =
            r#"before <tool_call>{"name":"read_file","arguments":{"path":"/x"}}</tool_call> after"#;
        let events = drive(&mut p, &[input]);
        // "before " (text) → Start → Args → " after" (text)
        assert_eq!(events[0], ParserEvent::Text("before ".to_string()));
        assert!(matches!(
            &events[1],
            ParserEvent::Start { index: 0, name } if name == "read_file"
        ));
        assert!(matches!(
            &events[2],
            ParserEvent::Args { index: 0, args_json } if args_json.contains(r#""path":"/x""#)
        ));
        assert_eq!(events[3], ParserEvent::Text(" after".to_string()));
    }

    #[test]
    fn open_marker_split_across_chunks_is_buffered() {
        // The chunk boundary lands inside the opening marker.
        let chunks = [
            "before <",
            "tool_call>",
            r#"{"name":"a","arguments":{}}"#,
            "</tool_call> after",
        ];
        let mut p = ToolCallParser::new();
        let events = drive(&mut p, &chunks);
        // Despite the split, we get exactly:
        // - "before " as text (the "<" suffix was held)
        // - Start { name: "a" }
        // - Args
        // - " after"
        let texts: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                ParserEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts.join(""), "before  after");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParserEvent::Start { name, .. } if name == "a"))
        );
        assert!(events.iter().any(|e| matches!(e, ParserEvent::Args { .. })));
    }

    #[test]
    fn close_marker_split_across_chunks() {
        let chunks = [
            r#"<tool_call>{"name":"a","arguments":{}}<"#,
            "/tool_",
            "call>tail",
        ];
        let mut p = ToolCallParser::new();
        let events = drive(&mut p, &chunks);
        // Tail should arrive as text after the call is fully parsed.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParserEvent::Start { name, .. } if name == "a"))
        );
        let last_text = events.iter().rev().find_map(|e| match e {
            ParserEvent::Text(t) => Some(t.as_str()),
            _ => None,
        });
        assert_eq!(last_text, Some("tail"));
    }

    #[test]
    fn one_byte_at_a_time_produces_same_events_as_one_chunk() {
        let input = r#"a<tool_call>{"name":"f","arguments":{"k":1}}</tool_call>b"#;

        let mut single = ToolCallParser::new();
        let single_events = drive(&mut single, &[input]);

        let chunks: Vec<String> = input.chars().map(|c| c.to_string()).collect();
        let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
        let mut byte = ToolCallParser::new();
        let byte_events = drive(&mut byte, &chunk_refs);

        // Concatenated text equals on both paths.
        let text = |evs: &[ParserEvent]| -> String {
            evs.iter()
                .filter_map(|e| match e {
                    ParserEvent::Text(t) => Some(t.as_str()),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(text(&single_events), text(&byte_events));
        // Both paths see exactly one Start and one Args, with the
        // same name and arguments payload.
        let starts: Vec<&str> = byte_events
            .iter()
            .filter_map(|e| match e {
                ParserEvent::Start { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(starts, vec!["f"]);
        let args: Vec<&str> = byte_events
            .iter()
            .filter_map(|e| match e {
                ParserEvent::Args { args_json, .. } => Some(args_json.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(args.len(), 1);
        assert!(args[0].contains(r#""k":1"#));
    }

    #[test]
    fn multiple_tool_calls_get_distinct_indices() {
        let input = concat!(
            "lead ",
            r#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#,
            " mid ",
            r#"<tool_call>{"name":"b","arguments":{}}</tool_call>"#,
            " tail",
        );
        let mut p = ToolCallParser::new();
        let events = drive(&mut p, &[input]);
        let starts: Vec<(usize, String)> = events
            .iter()
            .filter_map(|e| match e {
                ParserEvent::Start { index, name } => Some((*index, name.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(starts, vec![(0, "a".into()), (1, "b".into())]);
    }

    #[test]
    fn malformed_tool_call_does_not_crash() {
        let mut p = ToolCallParser::new();
        let events = drive(&mut p, &[r#"x<tool_call>not valid json</tool_call>y"#]);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParserEvent::Malformed { .. }))
        );
        // Bracketing text still flows.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParserEvent::Text(t) if t == "x"))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParserEvent::Text(t) if t == "y"))
        );
    }

    #[test]
    fn unterminated_tool_call_is_reported_on_finish() {
        let mut p = ToolCallParser::new();
        let events = drive(&mut p, &[r#"x<tool_call>{"name":"a""#]);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ParserEvent::Malformed { .. }))
        );
    }

    // ── ThinkParser ─────────────────────────────────────────────────

    fn drive_think(parser: &mut ThinkParser, chunks: &[&str]) -> Vec<ThinkEvent> {
        let mut events = Vec::new();
        for c in chunks {
            events.extend(parser.feed(c));
        }
        events.extend(parser.finish());
        events
    }

    #[test]
    fn think_plain_text_passes_through() {
        let mut p = ThinkParser::new();
        let events = drive_think(&mut p, &["hello ", "world"]);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], ThinkEvent::Text("hello ".into()));
        assert_eq!(events[1], ThinkEvent::Text("world".into()));
    }

    #[test]
    fn think_splits_text_reasoning_text() {
        let mut p = ThinkParser::new();
        let events = drive_think(&mut p, &["before <think>thinking now</think> after"]);
        assert_eq!(events[0], ThinkEvent::Text("before ".into()));
        assert_eq!(events[1], ThinkEvent::Reasoning("thinking now".into()));
        assert_eq!(events[2], ThinkEvent::Text(" after".into()));
    }

    #[test]
    fn think_open_marker_split_across_chunks() {
        let mut p = ThinkParser::new();
        let events = drive_think(&mut p, &["pre <", "think>middle</think> post"]);
        let texts: String = events
            .iter()
            .filter_map(|e| match e {
                ThinkEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        let reasoning: String = events
            .iter()
            .filter_map(|e| match e {
                ThinkEvent::Reasoning(r) => Some(r.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, "pre  post");
        assert_eq!(reasoning, "middle");
    }

    #[test]
    fn think_close_marker_split_across_chunks() {
        let mut p = ThinkParser::new();
        let events = drive_think(&mut p, &["a<think>b<", "/think>c"]);
        let reasoning: String = events
            .iter()
            .filter_map(|e| match e {
                ThinkEvent::Reasoning(r) => Some(r.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning, "b");
        let last_text = events.iter().rev().find_map(|e| match e {
            ThinkEvent::Text(t) => Some(t.as_str()),
            _ => None,
        });
        assert_eq!(last_text, Some("c"));
    }

    #[test]
    fn think_one_byte_at_a_time_matches_single_chunk() {
        let input = "x<think>internal</think>y";
        let mut single = ThinkParser::new();
        let single_events = drive_think(&mut single, &[input]);

        let chunks: Vec<String> = input.chars().map(|c| c.to_string()).collect();
        let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
        let mut byte = ThinkParser::new();
        let byte_events = drive_think(&mut byte, &chunk_refs);

        let text = |evs: &[ThinkEvent]| -> (String, String) {
            let mut t = String::new();
            let mut r = String::new();
            for e in evs {
                match e {
                    ThinkEvent::Text(s) => t.push_str(s),
                    ThinkEvent::Reasoning(s) => r.push_str(s),
                }
            }
            (t, r)
        };
        assert_eq!(text(&single_events), text(&byte_events));
        assert_eq!(text(&byte_events), ("xy".into(), "internal".into()));
    }

    #[test]
    fn think_empty_block_emits_no_reasoning_event() {
        let mut p = ThinkParser::new();
        let events = drive_think(&mut p, &["<think></think>real"]);
        // No Reasoning event for an empty <think></think>; just the
        // trailing text.
        assert!(
            !events.iter().any(|e| matches!(e, ThinkEvent::Reasoning(_))),
            "events: {events:?}"
        );
        assert_eq!(events[0], ThinkEvent::Text("real".into()));
    }

    #[test]
    fn think_unterminated_block_flushes_as_reasoning_on_finish() {
        let mut p = ThinkParser::new();
        let events = drive_think(&mut p, &["x<think>thinking but no close"]);
        assert_eq!(events[0], ThinkEvent::Text("x".into()));
        let reasoning: String = events
            .iter()
            .filter_map(|e| match e {
                ThinkEvent::Reasoning(r) => Some(r.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning, "thinking but no close");
    }

    #[test]
    fn think_bare_close_marker_passes_through_as_text() {
        // Model emits </think> with no preceding <think>. Treat the
        // bare close as ordinary text — the agent doesn't try to
        // retroactively reclassify earlier deltas.
        let mut p = ThinkParser::new();
        let events = drive_think(&mut p, &["hello </think> world"]);
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                ThinkEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "hello </think> world");
        assert!(!events.iter().any(|e| matches!(e, ThinkEvent::Reasoning(_))));
    }

    #[test]
    fn quoted_lt_inside_args_does_not_trigger_marker() {
        // Sanity: a string value that happens to contain "<tool" is
        // not a marker. (Our marker search is on the literal byte
        // sequence "<tool_call>" / "</tool_call>", so this would
        // only break if a literal "</tool_call>" appeared in args
        // — which the model has no reason to emit.)
        let input = r#"<tool_call>{"name":"f","arguments":{"q":"why <tool emit?"}}</tool_call>"#;
        let mut p = ToolCallParser::new();
        let events = drive(&mut p, &[input]);
        let starts: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                ParserEvent::Start { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(starts, vec!["f"]);
    }
}
