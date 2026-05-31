//! Wire-format projection layer.
//!
//! The candle harness produces a single, format-agnostic stream of
//! [`InferenceEvent`]s. Each wire format (OpenAI chat completions,
//! OpenAI Responses, Anthropic messages, …) lives in its own module
//! under `wire::` and projects that event stream into the chunks /
//! events its HTTP clients expect.
//!
//! The benefit over translating *between* wire shapes (OpenAI chat
//! → Anthropic, etc.) is that we never have to reason about a
//! wire-N → wire-M conversion: every translation is wire-N ↔ the
//! internal event currency, and the projections are independent. A
//! new wire format adds a new file under `wire::`; nothing else
//! needs to know about it.
//!
//! Today: [`openai_chat`]. Stage 2 adds `openai_responses`. Stage 3
//! could add a native Anthropic projection that replaces the
//! gateway-side translation.

pub mod event;
pub mod openai_chat;
pub mod openai_responses;

pub use event::{FinishReason, InferenceEvent, ReasoningTokenPair, detect_reasoning_token_pair};
