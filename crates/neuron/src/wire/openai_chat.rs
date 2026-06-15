//! OpenAI chat completions projection.
//!
//! Reads [`InferenceEvent`]s from a receiver and produces
//! [`ChatCompletionChunk`]s in the shape `POST /v1/chat/completions`
//! clients expect on its streaming SSE response. The HTTP handler in
//! [`crate::api`] wraps the resulting receiver in axum's
//! `Sse::new(...)` adapter; nothing in this module touches HTTP
//! framing or `data:` lines.
//!
//! Per the OpenAI streaming spec, three chunk shapes appear:
//!
//! 1. **Role chunk** — `delta: { "role": "assistant" }`, no content,
//!    sent once at stream start. We emit this on [`InferenceEvent::Start`].
//! 2. **Content chunks** — `delta: { "content": "<text>" }`, one per
//!    [`InferenceEvent::TextDelta`].
//! 3. **Final chunk** — empty `delta`, `finish_reason` populated.
//!    Emitted on [`InferenceEvent::Finish`].
//!
//! `usage` stays `None` on every chunk; the legacy candle paths
//! never surfaced usage on the streaming endpoint and we keep that
//! behaviour bit-for-bit so existing clients see no diff.
//!
//! Back-pressure: the projection task awaits both `rx.recv()` and
//! `tx.send()`. A slow consumer fills the output channel → the
//! task blocks on send → it stops reading from the input → the
//! producer blocks on its own send. The bounded channels
//! propagate without us writing any logic.

use cortex_core::openai::{ChatCompletionChunk, ChunkChoice, Usage};
use serde_json::json;
use tokio::sync::mpsc;

use super::event::{FinishReason, InferenceEvent, ReasoningTokenPair};

/// Output channel buffer size. Mirrors the input side's bound; one
/// event maps to at most one chunk, so equal capacity keeps the
/// two ends in sync without surprising memory growth.
const CHUNK_CHANNEL_CAPACITY: usize = 32;

/// Per-stream config for the chat projector. Used by the
/// production handler to thread per-request choices (currently:
/// whether to surface reasoning content) into the projection
/// without bloating the function signature.
#[derive(Debug, Clone, Default)]
pub struct ChatProjectionConfig {
    /// When `true`, reasoning content is re-wrapped with the
    /// model's literal open/close markers and emitted as content
    /// deltas — preserving the on-the-wire shape that
    /// reasoning-aware clients like helexa-acp's `ThinkParser`
    /// expect.
    ///
    /// When `false` (the default), [`InferenceEvent::ReasoningDelta`]s
    /// are dropped entirely so consumers that don't know about
    /// reasoning (Zed's commit-message generator, any vanilla
    /// OpenAI client) don't have model-internal scratchpad
    /// material leaking into their UI. The chat-completions wire
    /// format has no slot for reasoning, so the default chooses
    /// the safer-for-naïve-clients behaviour.
    pub include_thinking: bool,
    /// Open/close marker strings to re-emit when `include_thinking`
    /// is set. Sourced from the loaded model's
    /// [`ReasoningTokenPair`]; `None` for non-reasoning models or
    /// when the caller doesn't have the pair handy (in which case
    /// `include_thinking` becomes equivalent to dropping reasoning
    /// because there's nothing to wrap).
    pub reasoning_markers: Option<ReasoningTokenPair>,
}

/// Project an [`InferenceEvent`] receiver into a
/// [`ChatCompletionChunk`] receiver. Spawns one tokio task that
/// owns the input receiver for the stream's lifetime and exits
/// when either side closes.
///
/// `id`, `created`, and `model_id` are stamped into every emitted
/// chunk so the receiver can stay generic (decoupled from
/// per-request metadata).
pub fn project_chat_stream(
    rx: mpsc::Receiver<InferenceEvent>,
    id: String,
    created: u64,
    model_id: String,
) -> mpsc::Receiver<ChatCompletionChunk> {
    // Default config: include_thinking off, no marker rewrap.
    project_chat_stream_with(rx, id, created, model_id, ChatProjectionConfig::default())
}

/// Same as [`project_chat_stream`] but with a per-stream config
/// (currently controlling reasoning surfacing). Production
/// callers that need the opt-in path call this directly; the
/// shorter wrapper above stays as the no-config convenience.
pub fn project_chat_stream_with(
    mut rx: mpsc::Receiver<InferenceEvent>,
    id: String,
    created: u64,
    model_id: String,
    config: ChatProjectionConfig,
) -> mpsc::Receiver<ChatCompletionChunk> {
    let (tx, out_rx) = mpsc::channel::<ChatCompletionChunk>(CHUNK_CHANNEL_CAPACITY);

    tokio::spawn(async move {
        // Track whether the previous event was inside a reasoning
        // block — used to decide when to emit the literal close
        // marker on the include_thinking re-wrap path. When this
        // flips from true → false (a TextDelta or Finish lands
        // after one or more ReasoningDeltas), we emit the close
        // marker exactly once.
        let mut was_in_reasoning = false;

        while let Some(event) = rx.recv().await {
            // Close-marker insertion: if we're leaving a reasoning
            // chain, emit the literal close marker before the
            // current event.
            if was_in_reasoning && !matches!(event, InferenceEvent::ReasoningDelta(_)) {
                if let Some(marker) = config
                    .include_thinking
                    .then_some(())
                    .and(config.reasoning_markers.as_ref())
                {
                    let chunk = content_chunk(&id, created, &model_id, &marker.close_text);
                    if tx.send(chunk).await.is_err() {
                        return;
                    }
                }
                was_in_reasoning = false;
            }

            let chunks = match event {
                InferenceEvent::Start => vec![role_chunk(&id, created, &model_id)],
                InferenceEvent::TextDelta(text) => {
                    if text.is_empty() {
                        // DecodeStream is buffering a multi-byte
                        // codepoint; don't bother sending an empty
                        // chunk downstream.
                        continue;
                    }
                    vec![content_chunk(&id, created, &model_id, &text)]
                }
                InferenceEvent::ReasoningDelta(text) => {
                    if !config.include_thinking {
                        // Default path — reasoning has no slot in
                        // chat completions, so it's dropped. Naïve
                        // clients (Zed commit-message generator,
                        // any vanilla OpenAI client) get clean
                        // output.
                        continue;
                    }
                    let Some(markers) = config.reasoning_markers.as_ref() else {
                        // Caller asked to include thinking but
                        // didn't supply markers — best we can do
                        // is emit the content as visible text.
                        // Skip the wrap entirely.
                        if text.is_empty() {
                            continue;
                        }
                        let chunk = content_chunk(&id, created, &model_id, &text);
                        if tx.send(chunk).await.is_err() {
                            return;
                        }
                        continue;
                    };
                    // First chunk of a reasoning block → open
                    // marker prelude. Subsequent reasoning deltas
                    // in the same block reuse `was_in_reasoning`
                    // to skip the prelude.
                    let mut chunks = Vec::new();
                    if !was_in_reasoning {
                        chunks.push(content_chunk(&id, created, &model_id, &markers.open_text));
                    }
                    if !text.is_empty() {
                        chunks.push(content_chunk(&id, created, &model_id, &text));
                    }
                    was_in_reasoning = true;
                    chunks
                }
                InferenceEvent::ToolCall {
                    index,
                    id: call_id,
                    name,
                    arguments,
                } => {
                    // OpenAI streaming shape for tool calls:
                    // `delta.tool_calls[]` with id + function.name
                    // on the first chunk per index, then
                    // function.arguments deltas. We have the
                    // complete arguments buffered already, so one
                    // delta carries everything.
                    vec![tool_call_chunk(
                        &id, created, &model_id, index, &call_id, &name, &arguments,
                    )]
                }
                InferenceEvent::Finish {
                    reason,
                    prompt_tokens,
                    completion_tokens,
                } => {
                    // The finish_reason chunk, then an OpenAI-style
                    // usage-only chunk (`choices: []`, `usage` populated).
                    // Clients (opencode) read this to track context size;
                    // cortex's Anthropic translator also picks `usage` up
                    // for its `message_delta`.
                    vec![
                        final_chunk(&id, created, &model_id, reason),
                        usage_chunk(&id, created, &model_id, prompt_tokens, completion_tokens),
                    ]
                }
            };
            for chunk in chunks {
                if tx.send(chunk).await.is_err() {
                    // Consumer hung up; nothing more to do.
                    return;
                }
            }
        }
    });

    out_rx
}

fn role_chunk(id: &str, created: u64, model_id: &str) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.into(),
        object: "chat.completion.chunk".into(),
        created,
        model: model_id.into(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: json!({ "role": "assistant" }),
            finish_reason: None,
            extra: serde_json::Value::Object(Default::default()),
        }],
        usage: None,
        extra: serde_json::Value::Object(Default::default()),
    }
}

fn content_chunk(id: &str, created: u64, model_id: &str, text: &str) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.into(),
        object: "chat.completion.chunk".into(),
        created,
        model: model_id.into(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: json!({ "content": text }),
            finish_reason: None,
            extra: serde_json::Value::Object(Default::default()),
        }],
        usage: None,
        extra: serde_json::Value::Object(Default::default()),
    }
}

/// OpenAI chat streaming shape for a tool call. One chunk per
/// call slot, carrying id + name + the complete arguments JSON.
/// Mirrors the format real OpenAI emits on the streaming path,
/// minus the per-token arguments-streaming complication (we have
/// the whole buffer already after the model finishes the
/// `<tool_call>...</tool_call>` block).
fn tool_call_chunk(
    id: &str,
    created: u64,
    model_id: &str,
    index: usize,
    call_id: &str,
    name: &str,
    arguments: &str,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.into(),
        object: "chat.completion.chunk".into(),
        created,
        model: model_id.into(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: json!({
                "tool_calls": [{
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    }
                }],
            }),
            finish_reason: None,
            extra: serde_json::Value::Object(Default::default()),
        }],
        usage: None,
        extra: serde_json::Value::Object(Default::default()),
    }
}

fn final_chunk(
    id: &str,
    created: u64,
    model_id: &str,
    reason: FinishReason,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.into(),
        object: "chat.completion.chunk".into(),
        created,
        model: model_id.into(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: serde_json::Value::Object(Default::default()),
            finish_reason: Some(reason.as_openai_str().to_string()),
            extra: serde_json::Value::Object(Default::default()),
        }],
        usage: None,
        extra: serde_json::Value::Object(Default::default()),
    }
}

/// OpenAI-style trailing usage chunk: empty `choices`, populated
/// `usage`. Mirrors what `stream_options: {include_usage: true}`
/// produces. Emitted unconditionally — clients that don't read usage
/// ignore the empty-choices chunk; clients that do (opencode, and
/// cortex's Anthropic translator) get the token counts they need to
/// track context.
fn usage_chunk(
    id: &str,
    created: u64,
    model_id: &str,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.into(),
        object: "chat.completion.chunk".into(),
        created,
        model: model_id.into(),
        choices: Vec::new(),
        usage: Some(Usage {
            prompt_tokens: prompt_tokens as u64,
            completion_tokens: completion_tokens as u64,
            total_tokens: (prompt_tokens + completion_tokens) as u64,
        }),
        extra: serde_json::Value::Object(Default::default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drain the projection's output into a Vec for assertion.
    async fn collect(mut rx: mpsc::Receiver<ChatCompletionChunk>) -> Vec<ChatCompletionChunk> {
        let mut out = Vec::new();
        while let Some(chunk) = rx.recv().await {
            out.push(chunk);
        }
        out
    }

    #[tokio::test]
    async fn empty_event_stream_yields_no_chunks() {
        let (tx, rx) = mpsc::channel::<InferenceEvent>(4);
        drop(tx);
        let out = collect(project_chat_stream(rx, "id-1".into(), 1700, "m".into())).await;
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn start_text_finish_produces_role_content_finish_and_usage() {
        let (tx, rx) = mpsc::channel::<InferenceEvent>(4);
        let out_rx = project_chat_stream(rx, "id-1".into(), 1700, "m".into());

        tx.send(InferenceEvent::Start).await.unwrap();
        tx.send(InferenceEvent::TextDelta("hello".into()))
            .await
            .unwrap();
        tx.send(InferenceEvent::Finish {
            reason: FinishReason::Stop,
            prompt_tokens: 0,
            completion_tokens: 0,
        })
        .await
        .unwrap();
        drop(tx);

        let out = collect(out_rx).await;
        assert_eq!(out.len(), 4); // role, content, finish, usage
        assert_eq!(out[0].choices[0].delta["role"], "assistant");
        assert_eq!(out[1].choices[0].delta["content"], "hello");
        assert_eq!(out[2].choices[0].finish_reason.as_deref(), Some("stop"));
        // Trailing usage-only chunk: empty choices, usage populated.
        assert!(out[3].choices.is_empty());
        assert!(out[3].usage.is_some());
        // Every chunk carries the stamped metadata.
        for chunk in &out {
            assert_eq!(chunk.id, "id-1");
            assert_eq!(chunk.created, 1700);
            assert_eq!(chunk.model, "m");
            assert_eq!(chunk.object, "chat.completion.chunk");
        }
    }

    #[tokio::test]
    async fn empty_text_delta_is_dropped() {
        let (tx, rx) = mpsc::channel::<InferenceEvent>(4);
        let out_rx = project_chat_stream(rx, "id".into(), 1, "m".into());
        tx.send(InferenceEvent::TextDelta(String::new()))
            .await
            .unwrap();
        drop(tx);
        let out = collect(out_rx).await;
        assert!(out.is_empty(), "empty deltas must not produce chunks");
    }

    #[tokio::test]
    async fn finish_length_maps_to_openai_string() {
        let (tx, rx) = mpsc::channel::<InferenceEvent>(4);
        let out_rx = project_chat_stream(rx, "id".into(), 1, "m".into());
        tx.send(InferenceEvent::Finish {
            reason: FinishReason::Length,
            prompt_tokens: 0,
            completion_tokens: 0,
        })
        .await
        .unwrap();
        drop(tx);
        let out = collect(out_rx).await;
        assert_eq!(out.len(), 2); // finish, usage
        assert_eq!(out[0].choices[0].finish_reason.as_deref(), Some("length"));
        assert!(out[1].usage.is_some(), "usage chunk emitted after finish");
    }

    #[tokio::test]
    async fn reasoning_delta_is_dropped_in_chat_projection() {
        let (tx, rx) = mpsc::channel::<InferenceEvent>(4);
        let out_rx = project_chat_stream(rx, "id".into(), 1, "m".into());
        tx.send(InferenceEvent::ReasoningDelta("<think>".into()))
            .await
            .unwrap();
        tx.send(InferenceEvent::TextDelta("real".into()))
            .await
            .unwrap();
        drop(tx);
        let out = collect(out_rx).await;
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].choices[0].delta["content"], "real");
    }

    fn pair() -> ReasoningTokenPair {
        ReasoningTokenPair {
            open_id: 0,
            close_id: 1,
            open_text: "<think>".into(),
            close_text: "</think>".into(),
        }
    }

    #[tokio::test]
    async fn include_thinking_rewraps_reasoning_with_literal_markers() {
        let (tx, rx) = mpsc::channel::<InferenceEvent>(8);
        let out_rx = project_chat_stream_with(
            rx,
            "id".into(),
            1,
            "m".into(),
            ChatProjectionConfig {
                include_thinking: true,
                reasoning_markers: Some(pair()),
            },
        );
        tx.send(InferenceEvent::ReasoningDelta("first ".into()))
            .await
            .unwrap();
        tx.send(InferenceEvent::ReasoningDelta("second".into()))
            .await
            .unwrap();
        tx.send(InferenceEvent::TextDelta("answer".into()))
            .await
            .unwrap();
        tx.send(InferenceEvent::Finish {
            reason: FinishReason::Stop,
            prompt_tokens: 0,
            completion_tokens: 0,
        })
        .await
        .unwrap();
        drop(tx);
        let out = collect(out_rx).await;
        // Expected sequence: open marker → reasoning content (2 chunks)
        // → close marker → visible answer → final chunk.
        let contents: Vec<&str> = out
            .iter()
            .filter_map(|c| {
                c.choices
                    .first()
                    .and_then(|ch| ch.delta["content"].as_str())
            })
            .collect();
        assert_eq!(
            contents,
            vec!["<think>", "first ", "second", "</think>", "answer"]
        );
        assert_eq!(
            out.iter()
                .find_map(|c| c.choices.first().and_then(|ch| ch.finish_reason.as_deref())),
            Some("stop")
        );
    }

    #[tokio::test]
    async fn include_thinking_closes_marker_at_finish_when_no_trailing_text() {
        // Edge case: stream ends inside a reasoning block (model
        // hit max_tokens mid-thought, no visible answer ever).
        // The Finish event still triggers the close marker so the
        // stream is balanced.
        let (tx, rx) = mpsc::channel::<InferenceEvent>(4);
        let out_rx = project_chat_stream_with(
            rx,
            "id".into(),
            1,
            "m".into(),
            ChatProjectionConfig {
                include_thinking: true,
                reasoning_markers: Some(pair()),
            },
        );
        tx.send(InferenceEvent::ReasoningDelta("thinking...".into()))
            .await
            .unwrap();
        tx.send(InferenceEvent::Finish {
            reason: FinishReason::Length,
            prompt_tokens: 0,
            completion_tokens: 0,
        })
        .await
        .unwrap();
        drop(tx);
        let out = collect(out_rx).await;
        let contents: Vec<&str> = out
            .iter()
            .filter_map(|c| {
                c.choices
                    .first()
                    .and_then(|ch| ch.delta["content"].as_str())
            })
            .collect();
        assert_eq!(contents, vec!["<think>", "thinking...", "</think>"]);
        assert_eq!(
            out.iter()
                .find_map(|c| c.choices.first().and_then(|ch| ch.finish_reason.as_deref())),
            Some("length")
        );
    }

    #[tokio::test]
    async fn include_thinking_without_markers_emits_content_directly() {
        // Defensive: if the caller asks for thinking but the
        // model declared no markers, we still emit the content
        // rather than dropping it. Better to leak than to lose.
        let (tx, rx) = mpsc::channel::<InferenceEvent>(4);
        let out_rx = project_chat_stream_with(
            rx,
            "id".into(),
            1,
            "m".into(),
            ChatProjectionConfig {
                include_thinking: true,
                reasoning_markers: None,
            },
        );
        tx.send(InferenceEvent::ReasoningDelta("raw".into()))
            .await
            .unwrap();
        tx.send(InferenceEvent::Finish {
            reason: FinishReason::Stop,
            prompt_tokens: 0,
            completion_tokens: 0,
        })
        .await
        .unwrap();
        drop(tx);
        let out = collect(out_rx).await;
        let contents: Vec<&str> = out
            .iter()
            .filter_map(|c| {
                c.choices
                    .first()
                    .and_then(|ch| ch.delta["content"].as_str())
            })
            .collect();
        assert_eq!(contents, vec!["raw"]);
    }

    #[tokio::test]
    async fn include_thinking_off_drops_reasoning_even_with_markers() {
        // Default behaviour even when markers happen to be
        // configured. The flag is the gate, not the marker
        // presence.
        let (tx, rx) = mpsc::channel::<InferenceEvent>(4);
        let out_rx = project_chat_stream_with(
            rx,
            "id".into(),
            1,
            "m".into(),
            ChatProjectionConfig {
                include_thinking: false,
                reasoning_markers: Some(pair()),
            },
        );
        tx.send(InferenceEvent::ReasoningDelta("hidden".into()))
            .await
            .unwrap();
        tx.send(InferenceEvent::TextDelta("visible".into()))
            .await
            .unwrap();
        tx.send(InferenceEvent::Finish {
            reason: FinishReason::Stop,
            prompt_tokens: 0,
            completion_tokens: 0,
        })
        .await
        .unwrap();
        drop(tx);
        let out = collect(out_rx).await;
        let contents: Vec<&str> = out
            .iter()
            .filter_map(|c| {
                c.choices
                    .first()
                    .and_then(|ch| ch.delta["content"].as_str())
            })
            .collect();
        assert_eq!(contents, vec!["visible"]);
    }
    #[tokio::test]
    async fn finish_emits_a_usage_chunk() {
        let (tx, rx) = mpsc::channel::<InferenceEvent>(4);
        let out_rx = project_chat_stream(rx, "id".into(), 1, "m".into());
        tx.send(InferenceEvent::TextDelta("hello".into()))
            .await
            .unwrap();
        tx.send(InferenceEvent::Finish {
            reason: FinishReason::Stop,
            prompt_tokens: 42,
            completion_tokens: 5,
        })
        .await
        .unwrap();
        drop(tx);
        let out = collect(out_rx).await;
        // Last chunk is usage-only: empty choices, populated usage.
        let last = out.last().unwrap();
        assert!(last.choices.is_empty(), "usage chunk has no choices");
        let u = last.usage.as_ref().expect("usage present on final chunk");
        assert_eq!(u.prompt_tokens, 42);
        assert_eq!(u.completion_tokens, 5);
        assert_eq!(u.total_tokens, 47);
    }
}
