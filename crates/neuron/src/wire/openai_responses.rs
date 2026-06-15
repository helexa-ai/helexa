//! OpenAI Responses API projection.
//!
//! Two responsibilities:
//!
//! 1. **Translate request shape**: [`request_to_chat`] flattens
//!    [`ResponsesRequest`]'s typed `input` items + `instructions`
//!    into the [`ChatCompletionRequest`] the candle harness already
//!    knows how to run. The Responses-specific shape stops at this
//!    function — everything downstream is the same chat path the
//!    `/v1/chat/completions` route exercises.
//!
//! 2. **Project event stream**: [`project_responses_stream`] reads
//!    [`InferenceEvent`]s from the harness and emits the named SSE
//!    events the Responses API client expects
//!    (`response.created`, `response.output_text.delta`,
//!    `response.completed`, …) along with their JSON payloads.
//!    The HTTP handler in [`crate::api`] reads
//!    `(event_name, data)` tuples off the receiver and stamps them
//!    onto axum SSE frames.
//!
//! Scope cuts (carried over from [`cortex_core::responses`]):
//!
//! - `previous_response_id` is rejected by [`request_to_chat`]
//!   with [`TranslateError::ChainedConversationNotSupported`].
//! - `Reasoning` input items are dropped (no equivalent in chat).
//! - `FunctionCall` / `FunctionCallOutput` items round-trip but the
//!   harness never emits tool calls today; the synthesis paths are
//!   in place so the surface is ready when it does.

use cortex_core::openai::{ChatCompletionRequest, ChatMessage, MessageContent};
use cortex_core::responses::{
    ResponsesContentPart, ResponsesInput, ResponsesInputItem, ResponsesMessageContent,
    ResponsesOutputContent, ResponsesOutputItem, ResponsesRequest, ResponsesResponse,
    ResponsesUsage, events,
};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use super::event::{FinishReason, InferenceEvent};

/// Per-request metadata that has to be stamped into every emitted
/// event. The projector spawns a task that owns one of these.
#[derive(Debug, Clone)]
pub struct ResponseMeta {
    pub response_id: String,
    pub created_at: u64,
    pub model_id: String,
    /// Item id used inside `output[0]` (the message). All
    /// `content_part.*` and `output_text.*` events reference this
    /// so the consumer knows which item the delta belongs to.
    pub message_item_id: String,
}

/// Reasons [`request_to_chat`] refuses a request.
#[derive(Debug, thiserror::Error)]
pub enum TranslateError {
    #[error(
        "previous_response_id is not supported on this neuron; chained \
         conversations require server-side state we don't store yet"
    )]
    ChainedConversationNotSupported,
}

/// Flatten a [`ResponsesRequest`] into the chat-completions shape
/// the candle harness already knows how to drive. Keeps the
/// Responses-specific machinery contained to a single function so
/// the harness stays format-agnostic.
///
/// Semantics:
///
/// - `instructions` (if set) becomes a leading `system` message.
/// - `input: "<string>"` becomes a single `user` message.
/// - `input: [items]` flattens each item:
///   - `Message { role, content }` → one `ChatMessage`.
///   - `FunctionCall` → an `assistant` turn whose `extra.tool_calls`
///     carries the call (chat-completions-shaped). The harness
///     doesn't act on tool_calls today, but the shape stays
///     consistent with what chat would expect.
///   - `FunctionCallOutput` → a `tool` role message with the
///     output text. Matches OpenAI's chat convention.
///   - `Reasoning` items are dropped (no equivalent in chat).
/// - Text parts within an array `content` collapse to a single
///   string; image parts get rendered as a chat-style content
///   array `[{type:"text"}, {type:"image_url"}]` so the chat
///   handler's existing vision path applies.
pub fn request_to_chat(req: ResponsesRequest) -> Result<ChatCompletionRequest, TranslateError> {
    if req.previous_response_id.is_some() {
        return Err(TranslateError::ChainedConversationNotSupported);
    }

    let mut messages: Vec<ChatMessage> = Vec::new();

    if let Some(instructions) = req.instructions
        && !instructions.is_empty()
    {
        messages.push(ChatMessage {
            role: "system".into(),
            content: MessageContent::Text(instructions),
            extra: Value::Object(Default::default()),
        });
    }

    match req.input {
        ResponsesInput::Text(text) => {
            messages.push(ChatMessage {
                role: "user".into(),
                content: MessageContent::Text(text),
                extra: Value::Object(Default::default()),
            });
        }
        ResponsesInput::Items(items) => {
            for item in items {
                if let Some(msg) = input_item_to_chat(item) {
                    messages.push(msg);
                }
            }
        }
    }

    Ok(ChatCompletionRequest {
        model: req.model,
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_output_tokens,
        stream: Some(req.stream),
        extra: Value::Object(Default::default()),
    })
}

fn input_item_to_chat(item: ResponsesInputItem) -> Option<ChatMessage> {
    match item {
        ResponsesInputItem::Message { role, content } => Some(ChatMessage {
            role,
            content: message_content_to_chat(content),
            extra: Value::Object(Default::default()),
        }),
        ResponsesInputItem::FunctionCall {
            call_id,
            name,
            arguments,
        } => {
            // Express the call in chat-completions shape via
            // `extra.tool_calls`. The harness ignores it today but
            // the shape is consistent for the day it doesn't.
            let mut extra = serde_json::Map::new();
            extra.insert(
                "tool_calls".into(),
                json!([{
                    "id": call_id,
                    "type": "function",
                    "function": { "name": name, "arguments": arguments },
                }]),
            );
            Some(ChatMessage {
                role: "assistant".into(),
                content: MessageContent::Text(String::new()),
                extra: Value::Object(extra),
            })
        }
        ResponsesInputItem::FunctionCallOutput { call_id, output } => {
            let mut extra = serde_json::Map::new();
            extra.insert("tool_call_id".into(), Value::String(call_id));
            Some(ChatMessage {
                role: "tool".into(),
                content: MessageContent::Text(output),
                extra: Value::Object(extra),
            })
        }
        // Reasoning items don't have a chat-completions equivalent
        // we can faithfully forward. Silently drop — the alternative
        // is rejecting a well-formed request, which is worse UX.
        ResponsesInputItem::Reasoning { .. } => None,
    }
}

fn message_content_to_chat(content: ResponsesMessageContent) -> MessageContent {
    match content {
        ResponsesMessageContent::Text(s) => MessageContent::Text(s),
        ResponsesMessageContent::Parts(parts) => {
            // Collapse to a string when every part is text; emit
            // the chat content-array shape only when an image is
            // present (some upstreams treat the array form as a
            // vision-only signal and reject it for text-only
            // models).
            let has_image = parts
                .iter()
                .any(|p| matches!(p, ResponsesContentPart::InputImage { .. }));
            if !has_image {
                let joined = parts
                    .into_iter()
                    .filter_map(|p| match p {
                        ResponsesContentPart::InputText { text }
                        | ResponsesContentPart::OutputText { text, .. } => Some(text),
                        ResponsesContentPart::InputImage { .. } => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n");
                return MessageContent::Text(joined);
            }
            let mut out: Vec<Value> = Vec::with_capacity(parts.len());
            for p in parts {
                match p {
                    ResponsesContentPart::InputText { text }
                    | ResponsesContentPart::OutputText { text, .. } => {
                        out.push(json!({ "type": "text", "text": text }));
                    }
                    ResponsesContentPart::InputImage { image_url, .. } => {
                        out.push(json!({
                            "type": "image_url",
                            "image_url": { "url": image_url },
                        }));
                    }
                }
            }
            MessageContent::Parts(out)
        }
    }
}

// ── Streaming projection ─────────────────────────────────────────────

/// One frame the projector emits. The HTTP handler maps each into
/// an axum `Sse::Event` with both an `event:` name and a `data:`
/// JSON payload — Responses, unlike chat completions, uses named
/// SSE events.
#[derive(Debug, Clone)]
pub struct ResponseStreamFrame {
    pub event_name: &'static str,
    pub data: Value,
}

/// Project an [`InferenceEvent`] receiver into a stream of
/// [`ResponseStreamFrame`]s. The emitted sequence per stream is:
///
/// 1. `response.created` — shell with `status: "in_progress"`.
/// 2. `response.output_item.added` — empty message item.
/// 3. `response.content_part.added` — empty `output_text` part.
/// 4. `response.output_text.delta` × N — token-by-token text.
/// 5. `response.output_text.done` — full accumulated text.
/// 6. `response.content_part.done` — full part payload.
/// 7. `response.output_item.done` — full message item.
/// 8. `response.completed` — final response with `status:"completed"`.
///
/// Empty TextDeltas (the harness's incomplete-UTF-8 buffering) are
/// dropped. `ReasoningDelta`s have no representation in the
/// Responses API spec we model yet, so they're dropped too.
pub fn project_responses_stream(
    rx: mpsc::Receiver<InferenceEvent>,
    meta: ResponseMeta,
) -> mpsc::Receiver<ResponseStreamFrame> {
    let (tx, out_rx) = mpsc::channel::<ResponseStreamFrame>(64);
    tokio::spawn(async move {
        run_projection(rx, meta, tx).await;
    });
    out_rx
}

async fn run_projection(
    mut rx: mpsc::Receiver<InferenceEvent>,
    meta: ResponseMeta,
    tx: mpsc::Sender<ResponseStreamFrame>,
) {
    let mut accumulated = String::new();
    let mut finish: Option<FinishReason> = None;
    let mut emitted_start = false;

    while let Some(event) = rx.recv().await {
        match event {
            InferenceEvent::Start => {
                emitted_start = true;
                if !emit_start_frames(&tx, &meta).await {
                    return;
                }
            }
            InferenceEvent::TextDelta(text) => {
                if text.is_empty() {
                    continue;
                }
                accumulated.push_str(&text);
                let frame = ResponseStreamFrame {
                    event_name: events::OUTPUT_TEXT_DELTA,
                    data: json!({
                        "item_id": meta.message_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": text,
                    }),
                };
                if tx.send(frame).await.is_err() {
                    return;
                }
            }
            InferenceEvent::ReasoningDelta(_) => {
                // No representation in our Responses model yet.
                // Stage where it'd land: a `response.reasoning_*`
                // event family alongside `response.output_text.*`.
            }
            InferenceEvent::ToolCall { .. } => {
                // Responses-side tool-call routing not wired yet
                // (would emit response.function_call_arguments.*
                // events). Drop for now; the chat-completions
                // projector handles tool calls. Future work
                // tracked in #7 alongside the in_progress event.
            }
            InferenceEvent::Finish { reason, .. } => {
                finish = Some(reason);
            }
        }
    }

    // Producers can drop without ever sending Start (e.g. early
    // poisoned-model error). Synthesize the open frames so the
    // consumer at least sees a coherent shell before completed.
    if !emitted_start && !emit_start_frames(&tx, &meta).await {
        return;
    }

    let reason = finish.unwrap_or(FinishReason::Stop);
    let _ = emit_finish_frames(&tx, &meta, &accumulated, reason).await;
}

async fn emit_start_frames(tx: &mpsc::Sender<ResponseStreamFrame>, meta: &ResponseMeta) -> bool {
    let shell = response_shell(meta, "in_progress", &[], None);
    let frames = [
        ResponseStreamFrame {
            event_name: events::CREATED,
            data: json!({ "response": shell.clone() }),
        },
        // `response.in_progress` carries the same shell as
        // `response.created` — both report the "in_progress"
        // status and both are payload-light bookkeeping events.
        // The distinction is meaningful to clients that
        // differentiate "request validated" from "model is
        // generating" in their UI (loading spinner vs streaming
        // spinner). OpenAI's own Responses SSE emits them as a
        // pair; matching the wire shape avoids subtle client
        // breakage.
        ResponseStreamFrame {
            event_name: events::IN_PROGRESS,
            data: json!({ "response": shell }),
        },
        ResponseStreamFrame {
            event_name: events::OUTPUT_ITEM_ADDED,
            data: json!({
                "output_index": 0,
                "item": empty_message_item(&meta.message_item_id),
            }),
        },
        ResponseStreamFrame {
            event_name: events::CONTENT_PART_ADDED,
            data: json!({
                "item_id": meta.message_item_id,
                "output_index": 0,
                "content_index": 0,
                "part": { "type": "output_text", "text": "", "annotations": [] },
            }),
        },
    ];
    for frame in frames {
        if tx.send(frame).await.is_err() {
            return false;
        }
    }
    true
}

async fn emit_finish_frames(
    tx: &mpsc::Sender<ResponseStreamFrame>,
    meta: &ResponseMeta,
    full_text: &str,
    reason: FinishReason,
) -> bool {
    let status = finish_to_status(reason);
    let full_part = json!({
        "type": "output_text",
        "text": full_text,
        "annotations": [],
    });
    let full_item = json!({
        "type": "message",
        "id": meta.message_item_id,
        "role": "assistant",
        "content": [full_part.clone()],
        "status": status,
    });
    let frames = [
        ResponseStreamFrame {
            event_name: events::OUTPUT_TEXT_DONE,
            data: json!({
                "item_id": meta.message_item_id,
                "output_index": 0,
                "content_index": 0,
                "text": full_text,
            }),
        },
        ResponseStreamFrame {
            event_name: events::CONTENT_PART_DONE,
            data: json!({
                "item_id": meta.message_item_id,
                "output_index": 0,
                "content_index": 0,
                "part": full_part,
            }),
        },
        ResponseStreamFrame {
            event_name: events::OUTPUT_ITEM_DONE,
            data: json!({
                "output_index": 0,
                "item": full_item.clone(),
            }),
        },
        ResponseStreamFrame {
            event_name: events::COMPLETED,
            data: json!({
                "response": response_shell(meta, status, &[full_item], None)
            }),
        },
    ];
    for frame in frames {
        if tx.send(frame).await.is_err() {
            return false;
        }
    }
    true
}

fn response_shell(
    meta: &ResponseMeta,
    status: &str,
    output: &[Value],
    usage: Option<&ResponsesUsage>,
) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("id".into(), Value::String(meta.response_id.clone()));
    obj.insert("object".into(), Value::String("response".into()));
    obj.insert("created_at".into(), json!(meta.created_at));
    obj.insert("status".into(), Value::String(status.into()));
    obj.insert("model".into(), Value::String(meta.model_id.clone()));
    obj.insert("output".into(), Value::Array(output.to_vec()));
    if let Some(u) = usage {
        obj.insert(
            "usage".into(),
            json!({
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "total_tokens": u.total_tokens,
            }),
        );
    }
    Value::Object(obj)
}

fn empty_message_item(item_id: &str) -> Value {
    json!({
        "type": "message",
        "id": item_id,
        "role": "assistant",
        "content": [],
        "status": "in_progress",
    })
}

fn finish_to_status(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop | FinishReason::ToolCalls => "completed",
        FinishReason::Length => "incomplete",
    }
}

// ── Non-streaming helpers ────────────────────────────────────────────

/// Collect a chat-completions response into a non-streaming
/// [`ResponsesResponse`]. Used by the `/v1/responses` handler when
/// the request doesn't set `stream: true`.
pub fn build_response(
    meta: &ResponseMeta,
    full_text: String,
    reason: FinishReason,
    usage: Option<ResponsesUsage>,
) -> ResponsesResponse {
    let status = finish_to_status(reason).to_string();
    ResponsesResponse {
        id: meta.response_id.clone(),
        object: "response".into(),
        created_at: meta.created_at,
        status: status.clone(),
        model: meta.model_id.clone(),
        output: vec![ResponsesOutputItem::Message {
            id: meta.message_item_id.clone(),
            role: "assistant".into(),
            content: vec![ResponsesOutputContent::OutputText {
                text: full_text,
                annotations: vec![],
            }],
            status,
        }],
        usage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortex_core::openai::MessageContent;

    fn meta() -> ResponseMeta {
        ResponseMeta {
            response_id: "resp_1".into(),
            created_at: 1700,
            model_id: "m".into(),
            message_item_id: "msg_1".into(),
        }
    }

    // ── request translator ──────────────────────────────────────────

    #[test]
    fn translates_text_input_to_single_user_message() {
        let req = ResponsesRequest {
            model: "m".into(),
            input: ResponsesInput::Text("hi".into()),
            instructions: None,
            stream: false,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            previous_response_id: None,
            extra: Value::Object(Default::default()),
        };
        let chat = request_to_chat(req).unwrap();
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "user");
        assert!(matches!(
            &chat.messages[0].content,
            MessageContent::Text(t) if t == "hi"
        ));
    }

    #[test]
    fn instructions_become_leading_system_message() {
        let req = ResponsesRequest {
            model: "m".into(),
            input: ResponsesInput::Text("hi".into()),
            instructions: Some("you are helpful".into()),
            stream: false,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            previous_response_id: None,
            extra: Value::Object(Default::default()),
        };
        let chat = request_to_chat(req).unwrap();
        assert_eq!(chat.messages.len(), 2);
        assert_eq!(chat.messages[0].role, "system");
        assert!(matches!(
            &chat.messages[0].content,
            MessageContent::Text(t) if t == "you are helpful"
        ));
        assert_eq!(chat.messages[1].role, "user");
    }

    #[test]
    fn rejects_previous_response_id() {
        let req = ResponsesRequest {
            model: "m".into(),
            input: ResponsesInput::Text("hi".into()),
            instructions: None,
            stream: false,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            previous_response_id: Some("resp_prev".into()),
            extra: Value::Object(Default::default()),
        };
        assert!(matches!(
            request_to_chat(req),
            Err(TranslateError::ChainedConversationNotSupported)
        ));
    }

    #[test]
    fn translates_input_items_to_chat_messages() {
        let req = ResponsesRequest {
            model: "m".into(),
            input: ResponsesInput::Items(vec![
                ResponsesInputItem::Message {
                    role: "user".into(),
                    content: ResponsesMessageContent::Text("first".into()),
                },
                ResponsesInputItem::Message {
                    role: "assistant".into(),
                    content: ResponsesMessageContent::Text("reply".into()),
                },
                ResponsesInputItem::Message {
                    role: "user".into(),
                    content: ResponsesMessageContent::Text("second".into()),
                },
            ]),
            instructions: None,
            stream: false,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            previous_response_id: None,
            extra: Value::Object(Default::default()),
        };
        let chat = request_to_chat(req).unwrap();
        assert_eq!(chat.messages.len(), 3);
        let roles: Vec<&str> = chat.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["user", "assistant", "user"]);
    }

    #[test]
    fn image_input_translates_to_chat_parts_array() {
        let req = ResponsesRequest {
            model: "m".into(),
            input: ResponsesInput::Items(vec![ResponsesInputItem::Message {
                role: "user".into(),
                content: ResponsesMessageContent::Parts(vec![
                    ResponsesContentPart::InputText {
                        text: "what is this?".into(),
                    },
                    ResponsesContentPart::InputImage {
                        image_url: "data:image/png;base64,AAA=".into(),
                        detail: None,
                    },
                ]),
            }]),
            instructions: None,
            stream: false,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            previous_response_id: None,
            extra: Value::Object(Default::default()),
        };
        let chat = request_to_chat(req).unwrap();
        let parts = match &chat.messages[0].content {
            MessageContent::Parts(p) => p.clone(),
            other => panic!("expected Parts, got {other:?}"),
        };
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,AAA=");
    }

    #[test]
    fn multiple_images_translate_in_order_and_tolerate_detail() {
        // C2: a Responses request carrying several InputImage parts
        // (with `detail` set) must translate to a chat Parts array that
        // preserves image order and the `image_url.url` shape the chat
        // vision path (`extract_images_from_request`) walks. The
        // `detail` hint has no chat-completions analogue we forward, so
        // it's dropped — but it must not break translation.
        let req = ResponsesRequest {
            model: "m".into(),
            input: ResponsesInput::Items(vec![ResponsesInputItem::Message {
                role: "user".into(),
                content: ResponsesMessageContent::Parts(vec![
                    ResponsesContentPart::InputText {
                        text: "compare these".into(),
                    },
                    ResponsesContentPart::InputImage {
                        image_url: "data:image/png;base64,FIRST".into(),
                        detail: Some("high".into()),
                    },
                    ResponsesContentPart::InputImage {
                        image_url: "data:image/png;base64,SECOND".into(),
                        detail: None,
                    },
                ]),
            }]),
            instructions: None,
            stream: false,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            previous_response_id: None,
            extra: Value::Object(Default::default()),
        };
        let chat = request_to_chat(req).unwrap();
        let parts = match &chat.messages[0].content {
            MessageContent::Parts(p) => p.clone(),
            other => panic!("expected Parts, got {other:?}"),
        };
        // text + two images, in input order.
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,FIRST");
        assert_eq!(parts[2]["image_url"]["url"], "data:image/png;base64,SECOND");
        // `detail` is not forwarded into the chat image_url object.
        assert!(parts[1]["image_url"].get("detail").is_none());
    }

    #[test]
    fn text_only_parts_collapse_to_string() {
        let req = ResponsesRequest {
            model: "m".into(),
            input: ResponsesInput::Items(vec![ResponsesInputItem::Message {
                role: "user".into(),
                content: ResponsesMessageContent::Parts(vec![
                    ResponsesContentPart::InputText {
                        text: "first".into(),
                    },
                    ResponsesContentPart::InputText {
                        text: "second".into(),
                    },
                ]),
            }]),
            instructions: None,
            stream: false,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            previous_response_id: None,
            extra: Value::Object(Default::default()),
        };
        let chat = request_to_chat(req).unwrap();
        assert!(matches!(
            &chat.messages[0].content,
            MessageContent::Text(t) if t == "first\n\nsecond"
        ));
    }

    #[test]
    fn reasoning_items_are_silently_dropped() {
        let req = ResponsesRequest {
            model: "m".into(),
            input: ResponsesInput::Items(vec![
                ResponsesInputItem::Reasoning { content: vec![] },
                ResponsesInputItem::Message {
                    role: "user".into(),
                    content: ResponsesMessageContent::Text("hi".into()),
                },
            ]),
            instructions: None,
            stream: false,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            previous_response_id: None,
            extra: Value::Object(Default::default()),
        };
        let chat = request_to_chat(req).unwrap();
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "user");
    }

    // ── streaming projector ─────────────────────────────────────────

    async fn collect(mut rx: mpsc::Receiver<ResponseStreamFrame>) -> Vec<ResponseStreamFrame> {
        let mut out = Vec::new();
        while let Some(f) = rx.recv().await {
            out.push(f);
        }
        out
    }

    #[tokio::test]
    async fn full_stream_emits_expected_event_sequence() {
        let (tx, rx) = mpsc::channel::<InferenceEvent>(8);
        let out = project_responses_stream(rx, meta());

        tx.send(InferenceEvent::Start).await.unwrap();
        tx.send(InferenceEvent::TextDelta("hel".into()))
            .await
            .unwrap();
        tx.send(InferenceEvent::TextDelta("lo".into()))
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

        let frames = collect(out).await;
        let names: Vec<&str> = frames.iter().map(|f| f.event_name).collect();
        assert_eq!(
            names,
            vec![
                events::CREATED,
                events::IN_PROGRESS,
                events::OUTPUT_ITEM_ADDED,
                events::CONTENT_PART_ADDED,
                events::OUTPUT_TEXT_DELTA,
                events::OUTPUT_TEXT_DELTA,
                events::OUTPUT_TEXT_DONE,
                events::CONTENT_PART_DONE,
                events::OUTPUT_ITEM_DONE,
                events::COMPLETED,
            ]
        );

        // The two deltas should carry the right text. Indices
        // shifted by one after IN_PROGRESS inserted between
        // CREATED and OUTPUT_ITEM_ADDED.
        assert_eq!(frames[4].data["delta"], "hel");
        assert_eq!(frames[5].data["delta"], "lo");

        // The done event has the full accumulated text.
        assert_eq!(frames[6].data["text"], "hello");

        // Completed event carries the full message item.
        let completed = &frames[9].data["response"];
        assert_eq!(completed["status"], "completed");
        let output = completed["output"].as_array().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["content"][0]["text"], "hello");
    }

    #[tokio::test]
    async fn length_finish_maps_to_incomplete_status() {
        let (tx, rx) = mpsc::channel::<InferenceEvent>(8);
        let out = project_responses_stream(rx, meta());
        tx.send(InferenceEvent::Start).await.unwrap();
        tx.send(InferenceEvent::Finish {
            reason: FinishReason::Length,
            prompt_tokens: 0,
            completion_tokens: 0,
        })
        .await
        .unwrap();
        drop(tx);
        let frames = collect(out).await;
        let completed = frames
            .iter()
            .find(|f| f.event_name == events::COMPLETED)
            .unwrap();
        assert_eq!(completed.data["response"]["status"], "incomplete");
    }

    #[tokio::test]
    async fn synthesises_start_frames_when_producer_skips_start() {
        // A producer that drops without sending Start (poisoned
        // model, immediate disconnect, …) should still produce a
        // coherent stream — the projector synthesises the
        // mandatory header frames before COMPLETED so the
        // consumer never sees an output_text.done without a
        // matching content_part.added.
        let (tx, rx) = mpsc::channel::<InferenceEvent>(8);
        let out = project_responses_stream(rx, meta());
        drop(tx);
        let frames = collect(out).await;
        let names: Vec<&str> = frames.iter().map(|f| f.event_name).collect();
        assert!(names.contains(&events::CREATED));
        assert!(names.contains(&events::COMPLETED));
        assert!(names.contains(&events::OUTPUT_TEXT_DONE));
    }

    #[tokio::test]
    async fn empty_text_deltas_are_dropped() {
        let (tx, rx) = mpsc::channel::<InferenceEvent>(8);
        let out = project_responses_stream(rx, meta());
        tx.send(InferenceEvent::Start).await.unwrap();
        tx.send(InferenceEvent::TextDelta(String::new()))
            .await
            .unwrap();
        tx.send(InferenceEvent::TextDelta("real".into()))
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
        let frames = collect(out).await;
        let delta_count = frames
            .iter()
            .filter(|f| f.event_name == events::OUTPUT_TEXT_DELTA)
            .count();
        assert_eq!(delta_count, 1, "empty delta must not produce a frame");
    }

    // ── non-streaming builder ───────────────────────────────────────

    #[test]
    fn build_response_produces_completed_message_with_usage() {
        let r = build_response(
            &meta(),
            "hello".into(),
            FinishReason::Stop,
            Some(ResponsesUsage {
                input_tokens: 5,
                output_tokens: 1,
                total_tokens: 6,
            }),
        );
        assert_eq!(r.status, "completed");
        match &r.output[0] {
            ResponsesOutputItem::Message {
                role,
                content,
                status,
                ..
            } => {
                assert_eq!(role, "assistant");
                assert_eq!(status, "completed");
                match &content[0] {
                    ResponsesOutputContent::OutputText { text, .. } => {
                        assert_eq!(text, "hello");
                    }
                }
            }
            other => panic!("expected Message, got {other:?}"),
        }
        let u = r.usage.unwrap();
        assert_eq!(u.total_tokens, 6);
    }

    #[test]
    fn build_response_length_yields_incomplete_status() {
        let r = build_response(&meta(), "trunc".into(), FinishReason::Length, None);
        assert_eq!(r.status, "incomplete");
    }
}
