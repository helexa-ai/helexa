//! Format-agnostic inference event stream.
//!
//! The candle harness emits a sequence of these for every streaming
//! request. Wire-format projections in sibling modules
//! ([`super::openai_chat`], the eventual `openai_responses` /
//! `anthropic_messages` projections) read this stream and produce
//! the chunks / events their HTTP clients expect.
//!
//! Design notes:
//!
//! - [`Start`] carries no token of its own. It only signals "the
//!   model has accepted the prompt and is about to begin emitting
//!   text". OpenAI chat materialises this as a `role: assistant`
//!   chunk; OpenAI Responses as the `response.created` +
//!   `response.output_item.added` pair; Anthropic as
//!   `message_start`. All three of those would otherwise have to
//!   peek at the *first* token to know when to emit, which couples
//!   the wire layer to the producer's pacing.
//! - [`TextDelta`] is *visible* output. Reasoning / `<think>`
//!   blocks go through a future [`ReasoningDelta`] variant once
//!   the harness learns to split them (today they pass through as
//!   plain text inside `TextDelta`; helexa-acp picks them apart on
//!   the consumer side).
//! - [`Finish`] is the only place a stream is allowed to end
//!   cleanly. Projections rely on this to emit final usage
//!   bookkeeping; absence means the producer crashed and the
//!   consumer should treat the stream as truncated.
//!
//! [`Start`]: InferenceEvent::Start
//! [`TextDelta`]: InferenceEvent::TextDelta
//! [`Finish`]: InferenceEvent::Finish

/// One unit of output from the inference loop.
///
/// Producers send these on an `mpsc::Sender<InferenceEvent>`;
/// projection layers in sibling modules consume them and emit
/// wire-format-specific frames downstream.
#[derive(Debug, Clone)]
pub enum InferenceEvent {
    /// The producer has accepted the prompt and is about to emit
    /// the first token. Sent at most once per stream.
    Start,
    /// A piece of visible assistant text. Multiple deltas
    /// concatenate into the complete reply.
    TextDelta(String),
    /// Reasoning / scratchpad text the model emitted inside a
    /// `<think>` block (or equivalent). The harness routes
    /// content between marker tokens here so wire projectors can
    /// decide what to do with it (chat completions drops by
    /// default; Responses API has a dedicated event family).
    ReasoningDelta(String),
    /// A tool call has been parsed out of a `<tool_call>{json}</tool_call>`
    /// block. Carries the parsed name + arguments JSON string
    /// (Anthropic / OpenAI projectors emit their own wire shape
    /// from this).
    ///
    /// `index` is the call slot — incremented per tool call in a
    /// turn so wire formats that order calls by index
    /// (OpenAI chat completions) can correlate.
    ToolCall {
        index: usize,
        id: String,
        name: String,
        /// Complete JSON arguments string. The model could in
        /// principle stream these token-by-token, but our
        /// extraction buffers the whole block until `</tool_call>`
        /// arrives and emits exactly one event per call.
        arguments: String,
    },
    /// The stream is complete. Carries the reason so wire formats
    /// that use it (OpenAI's `finish_reason`, Anthropic's
    /// `stop_reason`) can render it without re-parsing — plus the token
    /// counts, so the streaming projectors can emit a `usage` chunk
    /// (clients like opencode track context / trigger compaction off
    /// it; without it they show "0 tokens" and overflow the cap).
    Finish {
        reason: FinishReason,
        prompt_tokens: u32,
        completion_tokens: u32,
    },
}

/// Why a stream stopped. Stays small on purpose — anything that
/// doesn't map cleanly to one of these collapses to [`Stop`].
///
/// Mappings to wire formats:
///
/// | variant | OpenAI `finish_reason` | OpenAI Responses `status` | Anthropic `stop_reason` |
/// |---------|------------------------|---------------------------|-------------------------|
/// | `Stop`  | `"stop"`               | `"completed"`             | `"end_turn"`            |
/// | `Length`| `"length"`             | `"incomplete"`            | `"max_tokens"`          |
/// | `ToolCalls` | `"tool_calls"`     | `"completed"`             | `"tool_use"`            |
///
/// [`Stop`]: FinishReason::Stop
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Model emitted EOS naturally.
    Stop,
    /// Hit `max_tokens` before EOS.
    Length,
    /// Stopped because the model called a tool and is waiting for
    /// the result. Emitted by the streaming candle loops once a
    /// `<tool_call>` block parses into a structured tool call, so
    /// Anthropic clients receive `stop_reason: tool_use`.
    ToolCalls,
}

impl FinishReason {
    /// String form used by OpenAI chat completions and OpenAI
    /// completions. Wire modules can call this directly or do their
    /// own mapping for non-string formats.
    pub fn as_openai_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
        }
    }
}

/// Open/close token IDs for the reasoning marker a loaded model uses
/// (or `None` for non-reasoning models). The harness reads this once
/// at load time from the tokenizer's added-tokens table, then the
/// inference loop checks `next_token` against the pair to flip
/// between [`InferenceEvent::TextDelta`] and
/// [`InferenceEvent::ReasoningDelta`].
///
/// `open` and `close` text are kept alongside the IDs so wire
/// projectors that want to re-emit the literal markers (the
/// opt-in `include_thinking` path on chat completions) don't have
/// to reach back into the tokenizer for the strings.
#[derive(Debug, Clone)]
pub struct ReasoningTokenPair {
    pub open_id: u32,
    pub close_id: u32,
    pub open_text: String,
    pub close_text: String,
}

/// Known reasoning-marker conventions. Each is a `(open, close)`
/// pair of literal token strings. Each modern reasoning model
/// declares its markers in the tokenizer's `added_tokens` table;
/// at load time we probe for whichever pair the loaded tokenizer
/// has and stash both IDs.
///
/// Ordering matters only for tie-breaking when a model declares
/// multiple pairs (shouldn't happen in practice); the first hit
/// wins.
const KNOWN_REASONING_MARKERS: &[(&str, &str)] = &[
    // Qwen3, DeepSeek-R1, gpt-oss, and most other open-weight
    // reasoning models.
    ("<think>", "</think>"),
    // Mistral Magistral.
    ("[THINK]", "[/THINK]"),
    // Some older derivatives; harmless to probe.
    ("<thought>", "</thought>"),
    ("<reasoning>", "</reasoning>"),
];

/// Open/close token IDs for the model's tool-call marker
/// convention (or `None` for models that don't emit structured
/// tool calls). Same shape as [`ReasoningTokenPair`]: probed once
/// at load time, consumed by the inference loop to switch between
/// "emit visible deltas" and "buffer JSON for the next tool
/// call".
#[derive(Debug, Clone)]
pub struct ToolCallTokenPair {
    pub open_id: u32,
    pub close_id: u32,
    pub open_text: String,
    pub close_text: String,
}

/// Tool-call marker conventions. Open-weight tool-use models
/// converged on `<tool_call>` / `</tool_call>` (Qwen3-Coder /
/// -Instruct, the Hermes function-call format, DeepSeek-Coder,
/// gpt-oss). The pair lives alongside the reasoning markers in
/// the same `added_tokens` table.
const KNOWN_TOOL_CALL_MARKERS: &[(&str, &str)] = &[("<tool_call>", "</tool_call>")];

/// Probe a tokenizer for known tool-call marker pairs. Mirrors
/// [`detect_reasoning_token_pair`] — both open AND close must
/// resolve for the pair to be returned. `None` means the model
/// doesn't emit structured tool calls (or its tokenizer split
/// the markers across tokens).
pub fn detect_tool_call_token_pair<F>(token_to_id: F) -> Option<ToolCallTokenPair>
where
    F: Fn(&str) -> Option<u32>,
{
    for (open_text, close_text) in KNOWN_TOOL_CALL_MARKERS {
        let open_id = token_to_id(open_text);
        let close_id = token_to_id(close_text);
        if let (Some(open_id), Some(close_id)) = (open_id, close_id) {
            return Some(ToolCallTokenPair {
                open_id,
                close_id,
                open_text: (*open_text).into(),
                close_text: (*close_text).into(),
            });
        }
    }
    None
}

/// Inspect a tokenizer for known reasoning-marker pairs and return
/// the first match. The tokenizer types this trait is defined over
/// just need to expose `token_to_id(&str) -> Option<u32>` so this
/// stays decoupled from the candle crate — the production caller
/// passes a `tokenizers::Tokenizer`, but tests can fake one.
///
/// Returns `None` when no known marker pair is fully declared
/// (both open AND close token ids must resolve). That's the
/// pass-through case — non-reasoning models, or reasoning models
/// whose tokenizer split the markers across multiple tokens (rare
/// in practice; modern reasoning tokenizers list them as
/// `added_tokens`).
pub fn detect_reasoning_token_pair<F>(token_to_id: F) -> Option<ReasoningTokenPair>
where
    F: Fn(&str) -> Option<u32>,
{
    for (open_text, close_text) in KNOWN_REASONING_MARKERS {
        let open_id = token_to_id(open_text);
        let close_id = token_to_id(close_text);
        if let (Some(open_id), Some(close_id)) = (open_id, close_id) {
            return Some(ReasoningTokenPair {
                open_id,
                close_id,
                open_text: (*open_text).into(),
                close_text: (*close_text).into(),
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn lookup<'a>(map: &'a HashMap<&'static str, u32>) -> impl Fn(&str) -> Option<u32> + 'a {
        |s| map.get(s).copied()
    }

    #[test]
    fn detects_qwen3_style_think_markers() {
        let mut m = HashMap::new();
        m.insert("<think>", 151648);
        m.insert("</think>", 151649);
        let pair = detect_reasoning_token_pair(lookup(&m)).expect("pair detected");
        assert_eq!(pair.open_id, 151648);
        assert_eq!(pair.close_id, 151649);
        assert_eq!(pair.open_text, "<think>");
        assert_eq!(pair.close_text, "</think>");
    }

    #[test]
    fn detects_mistral_magistral_markers() {
        let mut m = HashMap::new();
        m.insert("[THINK]", 100);
        m.insert("[/THINK]", 101);
        let pair = detect_reasoning_token_pair(lookup(&m)).expect("pair detected");
        assert_eq!(pair.open_text, "[THINK]");
    }

    #[test]
    fn returns_none_when_only_open_marker_present() {
        // A pathological tokenizer that has `<think>` but not
        // `</think>` shouldn't half-detect. Pass-through.
        let mut m = HashMap::new();
        m.insert("<think>", 1);
        assert!(detect_reasoning_token_pair(lookup(&m)).is_none());
    }

    #[test]
    fn returns_none_for_non_reasoning_tokenizer() {
        let m: HashMap<&'static str, u32> = HashMap::new();
        assert!(detect_reasoning_token_pair(lookup(&m)).is_none());
    }

    #[test]
    fn detects_tool_call_markers() {
        let mut m = HashMap::new();
        m.insert("<tool_call>", 151657);
        m.insert("</tool_call>", 151658);
        let pair = detect_tool_call_token_pair(lookup(&m)).expect("pair detected");
        assert_eq!(pair.open_id, 151657);
        assert_eq!(pair.close_id, 151658);
        assert_eq!(pair.open_text, "<tool_call>");
        assert_eq!(pair.close_text, "</tool_call>");
    }

    #[test]
    fn returns_none_for_non_tool_use_tokenizer() {
        let m: HashMap<&'static str, u32> = HashMap::new();
        assert!(detect_tool_call_token_pair(lookup(&m)).is_none());
    }

    #[test]
    fn first_match_wins_when_multiple_pairs_declared() {
        // Hypothetical tokenizer with both Qwen-style AND Mistral-style
        // markers — the `<think>` pair is earlier in the convention
        // table so it wins.
        let mut m = HashMap::new();
        m.insert("<think>", 1);
        m.insert("</think>", 2);
        m.insert("[THINK]", 3);
        m.insert("[/THINK]", 4);
        let pair = detect_reasoning_token_pair(lookup(&m)).unwrap();
        assert_eq!(pair.open_id, 1);
        assert_eq!(pair.close_id, 2);
    }
}
