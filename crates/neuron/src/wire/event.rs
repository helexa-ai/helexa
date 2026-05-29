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
    /// `<think>` block (or equivalent). Producers that don't
    /// surface reasoning separately use [`TextDelta`] for
    /// everything; future split lives here.
    ///
    /// Not yet emitted by the candle harness — present so future
    /// stages (qwen3 `<think>` routing, OpenAI o-series reasoning)
    /// have a typed home without breaking the existing
    /// projections.
    #[allow(dead_code)]
    ReasoningDelta(String),
    /// The stream is complete. Carries the reason so wire formats
    /// that use it (OpenAI's `finish_reason`, Anthropic's
    /// `stop_reason`) can render it without re-parsing.
    Finish { reason: FinishReason },
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
    /// the result. Not yet emitted by the candle harness —
    /// reserved for the day tool-call extraction lands.
    #[allow(dead_code)]
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
