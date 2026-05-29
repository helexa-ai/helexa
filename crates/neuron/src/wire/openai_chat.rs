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

use cortex_core::openai::{ChatCompletionChunk, ChunkChoice};
use serde_json::json;
use tokio::sync::mpsc;

use super::event::{FinishReason, InferenceEvent};

/// Output channel buffer size. Mirrors the input side's bound; one
/// event maps to at most one chunk, so equal capacity keeps the
/// two ends in sync without surprising memory growth.
const CHUNK_CHANNEL_CAPACITY: usize = 32;

/// Project an [`InferenceEvent`] receiver into a
/// [`ChatCompletionChunk`] receiver. Spawns one tokio task that
/// owns the input receiver for the stream's lifetime and exits
/// when either side closes.
///
/// `id`, `created`, and `model_id` are stamped into every emitted
/// chunk so the receiver can stay generic (decoupled from
/// per-request metadata).
pub fn project_chat_stream(
    mut rx: mpsc::Receiver<InferenceEvent>,
    id: String,
    created: u64,
    model_id: String,
) -> mpsc::Receiver<ChatCompletionChunk> {
    let (tx, out_rx) = mpsc::channel::<ChatCompletionChunk>(CHUNK_CHANNEL_CAPACITY);

    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
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
                InferenceEvent::ReasoningDelta(_) => {
                    // Reasoning isn't representable in OpenAI chat
                    // streaming today. The o-series uses a separate
                    // `summary` event but it's gated by the
                    // Responses API; chat-completions just drops it.
                    continue;
                }
                InferenceEvent::Finish { reason } => {
                    vec![final_chunk(&id, created, &model_id, reason)]
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
    async fn start_text_finish_produces_three_chunks() {
        let (tx, rx) = mpsc::channel::<InferenceEvent>(4);
        let out_rx = project_chat_stream(rx, "id-1".into(), 1700, "m".into());

        tx.send(InferenceEvent::Start).await.unwrap();
        tx.send(InferenceEvent::TextDelta("hello".into()))
            .await
            .unwrap();
        tx.send(InferenceEvent::Finish {
            reason: FinishReason::Stop,
        })
        .await
        .unwrap();
        drop(tx);

        let out = collect(out_rx).await;
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].choices[0].delta["role"], "assistant");
        assert_eq!(out[1].choices[0].delta["content"], "hello");
        assert_eq!(out[2].choices[0].finish_reason.as_deref(), Some("stop"));
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
        })
        .await
        .unwrap();
        drop(tx);
        let out = collect(out_rx).await;
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].choices[0].finish_reason.as_deref(), Some("length"));
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
}
