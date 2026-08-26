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
//!
//! Tool calling is wired end-to-end (#158): `tools` definitions are
//! normalized into the chat-wrapped shape and forwarded to the
//! harness, harness [`InferenceEvent::ToolCall`]s project into the
//! `response.function_call_arguments.*` event family, and
//! `FunctionCall` / `FunctionCallOutput` input items round-trip back
//! in as assistant `tool_calls` / tool-role messages.

use cortex_core::openai::{ChatCompletionRequest, ChatMessage, MessageContent};
use cortex_core::responses::{
    InputTokensDetails, OutputTokensDetails, ResponsesContentPart, ResponsesInput,
    ResponsesInputElement, ResponsesInputItem, ResponsesMessageContent, ResponsesOutputContent,
    ResponsesOutputItem, ResponsesRequest, ResponsesResponse, ResponsesUsage, events,
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
            // A reasoning item describes the assistant turn that
            // follows it, so it is held until that turn arrives rather
            // than becoming a message of its own (#277).
            let mut pending_reasoning: Option<String> = None;
            for element in items {
                let msg = match element {
                    ResponsesInputElement::Typed(ResponsesInputItem::Reasoning {
                        content,
                        summary,
                    }) => {
                        if let Some(text) = reasoning_item_text(&content, &summary) {
                            // Two reasoning items with no assistant turn
                            // between them: keep both, in order, rather
                            // than letting the second overwrite the first.
                            match &mut pending_reasoning {
                                Some(existing) => {
                                    existing.push_str("\n\n");
                                    existing.push_str(&text);
                                }
                                slot => *slot = Some(text),
                            }
                        }
                        None
                    }
                    ResponsesInputElement::Typed(item) => input_item_to_chat(item),
                    // Bare `{role, content}` (OpenAI EasyInputMessage —
                    // what litellm/agent-zero emit). `content: null`
                    // (e.g. an assistant turn carrying only tool calls)
                    // collapses to an empty string so the turn is kept.
                    ResponsesInputElement::EasyMessage { role, content } => Some(ChatMessage {
                        role,
                        content: content
                            .map(message_content_to_chat)
                            .unwrap_or_else(|| MessageContent::Text(String::new())),
                        extra: Value::Object(Default::default()),
                    }),
                    // Forward-compat: an item shape we don't model.
                    // Counted and logged rather than vanishing silently —
                    // a shape we drop must be discoverable from the
                    // outside (#277), which is how this class of defect
                    // stayed invisible for so long.
                    ResponsesInputElement::Other(other) => {
                        // Resolved before the macro: `tracing`'s own
                        // `Value` trait is in scope inside it, and would
                        // shadow `serde_json::Value` here.
                        let item_type = other
                            .get("type")
                            .and_then(|t| t.as_str())
                            .unwrap_or("<untyped>")
                            .to_string();
                        tracing::warn!(
                            %item_type,
                            "responses: dropping an input item shape this build does not model"
                        );
                        None
                    }
                };
                if let Some(mut msg) = msg {
                    if msg.role == "assistant"
                        && let Some(text) = pending_reasoning.take()
                    {
                        attach_reasoning(&mut msg, text);
                    }
                    messages.push(msg);
                }
            }
            // Reasoning with no assistant turn after it — the exact
            // shape of a turn truncated mid-think and replayed with
            // "continue". It is the whole payload of that turn, so it
            // becomes an assistant turn of its own rather than being
            // discarded for lack of a message to ride on.
            if let Some(text) = pending_reasoning.take() {
                let mut msg = ChatMessage {
                    role: "assistant".into(),
                    content: MessageContent::Text(String::new()),
                    extra: Value::Object(Default::default()),
                };
                attach_reasoning(&mut msg, text);
                messages.push(msg);
            }
        }
    }

    // Carry the caller's extension fields across the hop (#277).
    //
    // This used to start from an empty map and insert exactly one key,
    // which meant every extension a caller sent to `/v1/responses` was
    // discarded — including `chat_template_kwargs`, the only route by
    // which `enable_thinking` can reach the template, which is why
    // `/no_think` worked on chat/completions and not here (#223).
    // Keys the chat path does not model are inert, so forwarding them
    // costs nothing and stops the next control needing its own plumbing.
    let mut extra = match req.extra {
        Value::Object(map) => map,
        _ => serde_json::Map::new(),
    };
    // Tool definitions are the exception: Responses flattens the
    // function fields and the chat path (#158) reads the nested shape
    // for both the Jinja render and argument coercion, so this key is
    // rewritten rather than forwarded.
    match extra.get("tools").and_then(Value::as_array) {
        Some(tools) => {
            let normalized: Vec<Value> = tools.iter().filter_map(normalize_tool).collect();
            if normalized.is_empty() {
                extra.remove("tools");
            } else {
                extra.insert("tools".into(), Value::Array(normalized));
            }
        }
        None => {
            extra.remove("tools");
        }
    }

    // A caller that asked for no reasoning must get none (#223).
    apply_reasoning_effort(&mut extra);

    Ok(ChatCompletionRequest {
        model: req.model,
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        seed: req.seed,
        repetition_penalty: req.repetition_penalty,
        repeat_last_n: req.repeat_last_n,
        max_tokens: req.max_output_tokens,
        // Responses spells the cap `max_output_tokens`, already
        // resolved onto the legacy field above.
        max_completion_tokens: None,
        stream: Some(req.stream),
        extra: Value::Object(extra),
    })
}

/// Normalize one Responses-format tool definition into the
/// chat-completions wrapped shape the chat templates were written
/// against. Responses flattens the function fields onto the tool
/// object (`{type:"function", name, description, parameters}`);
/// chat nests them (`{type:"function", function:{…}}`). Hosted tool
/// types (web_search, file_search, …) have no server-side
/// implementation here and are dropped rather than rejected.
fn normalize_tool(tool: &Value) -> Option<Value> {
    if tool.get("function").is_some() {
        // Already chat-wrapped — pass through untouched.
        return Some(tool.clone());
    }
    if tool.get("type").and_then(Value::as_str) != Some("function") {
        return None;
    }
    let name = tool.get("name").and_then(Value::as_str)?;
    let mut func = serde_json::Map::new();
    func.insert("name".into(), Value::String(name.into()));
    for field in ["description", "parameters"] {
        if let Some(v) = tool.get(field)
            && !v.is_null()
        {
            func.insert(field.into(), v.clone());
        }
    }
    Some(json!({ "type": "function", "function": Value::Object(func) }))
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
            // `output` is either a plain string or an array of content
            // parts. Render a string as-is; anything else to compact
            // JSON so the tool result text reaches the model intact.
            let output_text = match output {
                Value::String(s) => s,
                other => other.to_string(),
            };
            let mut extra = serde_json::Map::new();
            extra.insert("tool_call_id".into(), Value::String(call_id));
            Some(ChatMessage {
                role: "tool".into(),
                content: MessageContent::Text(output_text),
                extra: Value::Object(extra),
            })
        }
        // Handled before this function is reached: a reasoning item
        // belongs to the assistant turn that follows it, so the caller
        // holds it and attaches it there (#277).
        ResponsesInputItem::Reasoning { .. } => None,
    }
}

/// Map the Responses reasoning control onto the template kwarg neuron
/// already honours (#223).
///
/// `reasoning: {"effort": "none"}` is how a Responses client says "do
/// not think" — it is what pi-ai emits for thinking-level *off*. The
/// chat-completions path couples generation to surfacing via
/// `default_enable_thinking`; this surface never did, so the switch had
/// no route at all and a client asking for no reasoning got a full think
/// block anyway. Observed live: a harness set thinking off and received
/// 32,768 tokens of reasoning and no answer.
///
/// Only `none` and `off` map. `minimal` / `low` deliberately do not:
/// they ask for *less* thinking, not none, and the template's switch is
/// binary — silently turning "minimal" into "off" would be the same
/// class of lie this whole area has been full of. They pass through and
/// the model thinks as usual.
///
/// An explicit `chat_template_kwargs.enable_thinking` from the caller
/// wins, matching `default_enable_thinking`'s precedence: the more
/// specific control is the one the caller reached for.
fn apply_reasoning_effort(extra: &mut serde_json::Map<String, Value>) {
    if extra
        .get("chat_template_kwargs")
        .and_then(|k| k.get("enable_thinking"))
        .is_some()
    {
        return;
    }
    let effort = extra
        .get("reasoning")
        .and_then(|r| r.get("effort"))
        .and_then(|e| e.as_str())
        .map(|e| e.trim().to_ascii_lowercase());
    if !matches!(effort.as_deref(), Some("none") | Some("off")) {
        return;
    }
    let kwargs = extra
        .entry("chat_template_kwargs")
        .or_insert_with(|| json!({}));
    if !kwargs.is_object() {
        *kwargs = json!({});
    }
    if let Some(kw) = kwargs.as_object_mut() {
        kw.insert("enable_thinking".into(), Value::Bool(false));
    }
}

/// The text of a replayed reasoning item, under either spelling.
///
/// `summary` is what neuron emits (`summary_text` parts) and `content`
/// is what OpenAI's o-series emits (`reasoning_text` parts); clients
/// replay the completed item verbatim, so the field we read has to be
/// the field the far end wrote. Parts are joined with a blank line,
/// matching how pi-ai flattens the same item for display.
///
/// Returns `None` for an item with no usable text — an
/// encrypted-content-only item, or an empty summary — so an assistant
/// turn is not annotated with an empty think block.
fn reasoning_item_text(content: &[Value], summary: &[Value]) -> Option<String> {
    fn join(parts: &[Value]) -> String {
        parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
    let text = match join(summary) {
        s if !s.is_empty() => s,
        _ => join(content),
    };
    (!text.is_empty()).then_some(text)
}

/// Hang a turn's reasoning on the assistant message it belongs to.
///
/// `reasoning_content` is the field HF chat templates read — Qwen3.8's
/// renders it back inside `<think>` — and `render_chat_template`
/// forwards message extras verbatim, so this is the whole of what the
/// round-trip needs on our side.
fn attach_reasoning(msg: &mut ChatMessage, text: String) {
    if !msg.extra.is_object() {
        msg.extra = Value::Object(Default::default());
    }
    if let Some(obj) = msg.extra.as_object_mut() {
        obj.insert("reasoning_content".into(), Value::String(text));
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
                        ResponsesContentPart::InputImage { .. } | ResponsesContentPart::Unknown => {
                            None
                        }
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
                    ResponsesContentPart::Unknown => {}
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
///
/// Every frame's data object is stamped with `"type"` (the event
/// name, duplicated from the SSE `event:` field) and a monotonic
/// `"sequence_number"`, matching OpenAI's wire shape. Official SDKs
/// and SDK-derived clients (ZeroClaw, #156) dispatch on `data.type`
/// and never read the SSE `event:` line — without the in-payload
/// tag they discard the entire stream.
pub fn project_responses_stream(
    rx: mpsc::Receiver<InferenceEvent>,
    meta: ResponseMeta,
) -> mpsc::Receiver<ResponseStreamFrame> {
    let (raw_tx, mut raw_rx) = mpsc::channel::<ResponseStreamFrame>(64);
    let (tx, out_rx) = mpsc::channel::<ResponseStreamFrame>(64);
    tokio::spawn(async move {
        run_projection(rx, meta, raw_tx).await;
    });
    tokio::spawn(async move {
        let mut sequence_number: u64 = 0;
        while let Some(mut frame) = raw_rx.recv().await {
            if let Value::Object(data) = &mut frame.data {
                data.insert("type".into(), Value::String(frame.event_name.into()));
                data.insert("sequence_number".into(), json!(sequence_number));
            }
            sequence_number += 1;
            if tx.send(frame).await.is_err() {
                return;
            }
        }
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
    let mut usage: Option<ResponsesUsage> = None;
    let mut emitted_start = false;
    // Completed `function_call` output items, re-emitted inside the
    // final `response.completed` payload so clients that only read
    // the terminal response still see the calls.
    let mut tool_items: Vec<Value> = Vec::new();
    // Reasoning item state (#267) and whether the message item's
    // opening frames have gone out. The message is opened lazily,
    // because its `output_index` depends on whether a reasoning item
    // claimed index 0 first.
    let mut reasoning = ReasoningTracker::default();
    let mut message_open = false;

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
                // Thinking is over the moment visible text starts.
                if !close_reasoning(&tx, &meta, &mut reasoning).await {
                    return;
                }
                if !message_open {
                    if !emit_message_open_frames(&tx, &meta, reasoning.message_index()).await {
                        return;
                    }
                    message_open = true;
                }
                accumulated.push_str(&text);
                let frame = ResponseStreamFrame {
                    event_name: events::OUTPUT_TEXT_DELTA,
                    data: json!({
                        "item_id": meta.message_item_id,
                        "output_index": reasoning.message_index(),
                        "content_index": 0,
                        "delta": text,
                    }),
                };
                if tx.send(frame).await.is_err() {
                    return;
                }
            }
            InferenceEvent::ReasoningDelta(text) => {
                // Stream the think block as its own output item (#267).
                // This used to be dropped on the floor, which meant a
                // model that reasoned for minutes emitted no events at
                // all and clients closed the stream as dead — dsh
                // reports `pi-ai stream idle timeout after 300000ms`.
                // Its watchdog resets on any parsed event, so surfacing
                // reasoning is what keeps a thinking model alive on the
                // wire; SSE comment keep-alives cannot, because idle
                // timers count events, not comments.
                if text.is_empty() {
                    continue;
                }
                if !reasoning.open {
                    if reasoning.item.is_some() {
                        // Reasoning resumed after visible text. Our
                        // model has one reasoning item per response, so
                        // fold it into the existing summary rather than
                        // opening a second item out of order.
                        continue;
                    }
                    if !emit_reasoning_open_frames(&tx, &meta).await {
                        return;
                    }
                    reasoning.open = true;
                }
                reasoning.text.push_str(&text);
                let frame = ResponseStreamFrame {
                    event_name: events::REASONING_SUMMARY_TEXT_DELTA,
                    data: json!({
                        "item_id": reasoning_item_id(&meta),
                        "output_index": 0,
                        "summary_index": 0,
                        "delta": text,
                    }),
                };
                if tx.send(frame).await.is_err() {
                    return;
                }
            }
            InferenceEvent::ToolCall {
                index,
                id,
                name,
                arguments,
            } => {
                if !close_reasoning(&tx, &meta, &mut reasoning).await {
                    return;
                }
                if !message_open {
                    if !emit_message_open_frames(&tx, &meta, reasoning.message_index()).await {
                        return;
                    }
                    message_open = true;
                }
                // function_call items follow the message item, which
                // itself follows any reasoning item — so the base is the
                // message's index, not a hardcoded 0.
                let output_index = reasoning.message_index() + 1 + index as u64;
                let item_id = format!(
                    "fc_{}_{index}",
                    meta.response_id.trim_start_matches("resp_")
                );
                let (id, name, arguments) = (id.as_str(), name.as_str(), arguments.as_str());
                let item_id = item_id.as_str();
                let full_item = json!({
                    "type": "function_call",
                    "id": item_id,
                    "call_id": id,
                    "name": name,
                    "arguments": arguments,
                    "status": "completed",
                });
                // The harness buffers a whole call and emits exactly
                // one ToolCall event, so the arguments "stream" is a
                // single delta. Emitting the full added → delta →
                // done → item.done family anyway matches OpenAI's
                // wire shape; SDK clients dedupe across the last
                // three, so no double-fire.
                let frames = [
                    ResponseStreamFrame {
                        event_name: events::OUTPUT_ITEM_ADDED,
                        data: json!({
                            "output_index": output_index,
                            "item": {
                                "type": "function_call",
                                "id": item_id,
                                "call_id": id,
                                "name": name,
                                "arguments": "",
                                "status": "in_progress",
                            },
                        }),
                    },
                    ResponseStreamFrame {
                        event_name: events::FUNCTION_CALL_ARGUMENTS_DELTA,
                        data: json!({
                            "item_id": item_id,
                            "output_index": output_index,
                            "delta": arguments,
                        }),
                    },
                    ResponseStreamFrame {
                        event_name: events::FUNCTION_CALL_ARGUMENTS_DONE,
                        data: json!({
                            "item_id": item_id,
                            "output_index": output_index,
                            "call_id": id,
                            "name": name,
                            "arguments": arguments,
                        }),
                    },
                    ResponseStreamFrame {
                        event_name: events::OUTPUT_ITEM_DONE,
                        data: json!({
                            "output_index": output_index,
                            "item": full_item.clone(),
                        }),
                    },
                ];
                for frame in frames {
                    if tx.send(frame).await.is_err() {
                        return;
                    }
                }
                tool_items.push(full_item);
            }
            InferenceEvent::Finish {
                reason,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                cached_tokens,
                // Responses-side `helexa_timing` surfacing not wired yet;
                // the bench harness reads timing off the chat path (#85).
                timing: _,
            } => {
                finish = Some(reason);
                // Surface usage on the streaming `response.completed`
                // frame — clients (opencode) track context/spend off it.
                // reasoning_tokens is an additive sub-count of
                // output_tokens (omitted for non-reasoning models).
                usage = Some(ResponsesUsage {
                    input_tokens: prompt_tokens as u64,
                    output_tokens: completion_tokens as u64,
                    total_tokens: (prompt_tokens + completion_tokens) as u64,
                    output_tokens_details: (reasoning_tokens > 0).then_some(OutputTokensDetails {
                        reasoning_tokens: reasoning_tokens as u64,
                    }),
                    // Prefix-cache reuse (#269). Omitted at zero so a
                    // client sees unchanged JSON when nothing was
                    // cached.
                    input_tokens_details: (cached_tokens > 0).then_some(InputTokensDetails {
                        cached_tokens: cached_tokens as u64,
                    }),
                });
            }
        }
    }

    // Producers can drop without ever sending Start (e.g. early
    // poisoned-model error). Synthesize the open frames so the
    // consumer at least sees a coherent shell before completed.
    if !emitted_start && !emit_start_frames(&tx, &meta).await {
        return;
    }

    // No `Finish` means the producer died rather than finished — the
    // poisoned-model path named above, an OOM, a dropped worker. This
    // used to default to `FinishReason::Stop`, which told the caller a
    // crash was a complete answer: pi-ai maps `status: "completed"` to
    // `stopReason: "stop"` and treats whatever partial text arrived as
    // the model's considered reply.
    //
    // Decided here, before the message item is opened, so a failed
    // stream does not leave an item open that nothing will ever close.
    let Some(reason) = finish else {
        let _ = tx
            .send(ResponseStreamFrame {
                event_name: events::FAILED,
                data: json!({
                    "response": failed_response_shell(
                        &meta,
                        "server_error",
                        "inference ended before the model finished",
                    )
                }),
            })
            .await;
        return;
    };

    // A stream can end while still thinking — a reasoning model that
    // exhausts max_output_tokens mid-block never emits visible text.
    // Close the item so the client sees a finished response rather than
    // a reasoning item left open forever.
    if !close_reasoning(&tx, &meta, &mut reasoning).await {
        return;
    }
    // The message item is opened lazily, so a response that produced no
    // text (tool-call-only turns, or the case above) has not announced
    // it yet. The finish frames reference it, so it must exist first.
    if !message_open && !emit_message_open_frames(&tx, &meta, reasoning.message_index()).await {
        return;
    }

    let _ = emit_finish_frames(
        &tx,
        &meta,
        FinishContext {
            full_text: &accumulated,
            reason,
            usage: usage.as_ref(),
            tool_items: &tool_items,
            message_index: reasoning.message_index(),
            reasoning_item: reasoning.item.as_ref(),
        },
    )
    .await;
}

/// The `response` payload for a [`events::FAILED`] frame.
///
/// pi-ai renders this as `` `${error.code}: ${error.message}` `` and
/// throws, which is the correct outcome — the request genuinely failed
/// — and is strictly better than a silent success.
fn failed_response_shell(meta: &ResponseMeta, code: &str, message: &str) -> Value {
    json!({
        "id": meta.response_id,
        "object": "response",
        "created_at": meta.created_at,
        "status": "failed",
        "model": meta.model_id,
        "output": [],
        "error": { "code": code, "message": message },
    })
}

/// Streaming state for a reasoning model's think block (#267).
///
/// The block is its own output item, ahead of the message, so the
/// message's `output_index` depends on whether reasoning happened —
/// which is only known once the first token arrives. This tracks that
/// decision and the accumulated summary text.
#[derive(Default)]
struct ReasoningTracker {
    /// Frames opening the reasoning item have been emitted.
    open: bool,
    /// Everything the model has thought so far, for the `*.done` frames
    /// and the terminal `completed` payload.
    text: String,
    /// The finished reasoning item, replayed inside `response.completed`
    /// so a client that only reads the terminal response still sees it.
    item: Option<Value>,
}

impl ReasoningTracker {
    /// `output_index` the message item takes: 1 when a reasoning item
    /// occupies 0, else 0.
    fn message_index(&self) -> u64 {
        u64::from(self.open || self.item.is_some())
    }
}

fn reasoning_item_id(meta: &ResponseMeta) -> String {
    format!("rs_{}", meta.response_id.trim_start_matches("resp_"))
}

/// Open the reasoning item and its summary part, once.
async fn emit_reasoning_open_frames(
    tx: &mpsc::Sender<ResponseStreamFrame>,
    meta: &ResponseMeta,
) -> bool {
    let item_id = reasoning_item_id(meta);
    let frames = [
        ResponseStreamFrame {
            event_name: events::OUTPUT_ITEM_ADDED,
            data: json!({
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": item_id,
                    "summary": [],
                    "status": "in_progress",
                },
            }),
        },
        ResponseStreamFrame {
            event_name: events::REASONING_SUMMARY_PART_ADDED,
            data: json!({
                "item_id": item_id,
                "output_index": 0,
                "summary_index": 0,
                "part": { "type": "summary_text", "text": "" },
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

/// Close the reasoning item, if one was opened, and record it for the
/// terminal payload. Idempotent — every path that emits a non-reasoning
/// item calls this first, and only the first call does anything.
async fn close_reasoning(
    tx: &mpsc::Sender<ResponseStreamFrame>,
    meta: &ResponseMeta,
    reasoning: &mut ReasoningTracker,
) -> bool {
    if !reasoning.open {
        return true;
    }
    reasoning.open = false;
    let item_id = reasoning_item_id(meta);
    let full_item = json!({
        "type": "reasoning",
        "id": item_id,
        "summary": [{ "type": "summary_text", "text": reasoning.text }],
        "status": "completed",
    });
    let frames = [
        ResponseStreamFrame {
            event_name: events::REASONING_SUMMARY_TEXT_DONE,
            data: json!({
                "item_id": item_id,
                "output_index": 0,
                "summary_index": 0,
                "text": reasoning.text,
            }),
        },
        ResponseStreamFrame {
            event_name: events::REASONING_SUMMARY_PART_DONE,
            data: json!({
                "item_id": item_id,
                "output_index": 0,
                "summary_index": 0,
                "part": { "type": "summary_text", "text": reasoning.text },
            }),
        },
        ResponseStreamFrame {
            event_name: events::OUTPUT_ITEM_DONE,
            data: json!({
                "output_index": 0,
                "item": full_item.clone(),
            }),
        },
    ];
    reasoning.item = Some(full_item);
    for frame in frames {
        if tx.send(frame).await.is_err() {
            return false;
        }
    }
    true
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
    ];
    for frame in frames {
        if tx.send(frame).await.is_err() {
            return false;
        }
    }
    true
}

/// Open the message item, once, at `output_index`.
///
/// Deferred rather than emitted with the start frames because the
/// message is no longer guaranteed to be `output[0]`: a reasoning model
/// puts its think block in an item ahead of it, and an item's index has
/// to be right the first time it is announced.
async fn emit_message_open_frames(
    tx: &mpsc::Sender<ResponseStreamFrame>,
    meta: &ResponseMeta,
    output_index: u64,
) -> bool {
    let frames = [
        ResponseStreamFrame {
            event_name: events::OUTPUT_ITEM_ADDED,
            data: json!({
                "output_index": output_index,
                "item": empty_message_item(&meta.message_item_id),
            }),
        },
        ResponseStreamFrame {
            event_name: events::CONTENT_PART_ADDED,
            data: json!({
                "item_id": meta.message_item_id,
                "output_index": output_index,
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

/// Everything the terminal frames need that isn't stream identity.
/// Grouped rather than passed loose: the reasoning item (#267) pushed
/// the argument list past the point where positional `&str`/`u64`
/// parameters are safe to read at a call site.
struct FinishContext<'a> {
    full_text: &'a str,
    reason: FinishReason,
    usage: Option<&'a ResponsesUsage>,
    tool_items: &'a [Value],
    /// Where the message item sits — 1 when a reasoning item took 0.
    message_index: u64,
    /// The completed reasoning item, replayed inside `completed`.
    reasoning_item: Option<&'a Value>,
}

async fn emit_finish_frames(
    tx: &mpsc::Sender<ResponseStreamFrame>,
    meta: &ResponseMeta,
    ctx: FinishContext<'_>,
) -> bool {
    let FinishContext {
        full_text,
        reason,
        usage,
        tool_items,
        message_index,
        reasoning_item,
    } = ctx;
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
    // Terminal output array, in wire order: the reasoning item (when
    // the model thought), then the message, then any completed
    // function_call items. Order matters — a client reconstructing the
    // response from `completed` alone should see what the stream showed.
    let mut output_items = Vec::with_capacity(2 + tool_items.len());
    if let Some(item) = reasoning_item {
        output_items.push(item.clone());
    }
    output_items.push(full_item.clone());
    output_items.extend(tool_items.iter().cloned());
    let frames = [
        ResponseStreamFrame {
            event_name: events::OUTPUT_TEXT_DONE,
            data: json!({
                "item_id": meta.message_item_id,
                "output_index": message_index,
                "content_index": 0,
                "text": full_text,
            }),
        },
        ResponseStreamFrame {
            event_name: events::CONTENT_PART_DONE,
            data: json!({
                "item_id": meta.message_item_id,
                "output_index": message_index,
                "content_index": 0,
                "part": full_part,
            }),
        },
        ResponseStreamFrame {
            event_name: events::OUTPUT_ITEM_DONE,
            data: json!({
                "output_index": message_index,
                "item": full_item.clone(),
            }),
        },
        ResponseStreamFrame {
            // `response.incomplete` when we stopped short, so a client
            // keyed on the event name rather than on `response.status`
            // still sees a terminal frame instead of timing out.
            event_name: if status == "incomplete" {
                events::INCOMPLETE
            } else {
                events::COMPLETED
            },
            data: json!({
                "response": response_shell(meta, status, &output_items, usage)
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
    // Why we stopped short. A client cannot distinguish an honest
    // truncation from a protocol fault without this, and pi-ai treats
    // the difference as continue-vs-halt.
    if status == "incomplete" {
        obj.insert(
            "incomplete_details".into(),
            json!({ "reason": "max_output_tokens" }),
        );
    }
    obj.insert("model".into(), Value::String(meta.model_id.clone()));
    obj.insert("output".into(), Value::Array(output.to_vec()));
    if let Some(u) = usage {
        let mut usage_obj = serde_json::Map::new();
        usage_obj.insert("input_tokens".into(), json!(u.input_tokens));
        usage_obj.insert("output_tokens".into(), json!(u.output_tokens));
        usage_obj.insert("total_tokens".into(), json!(u.total_tokens));
        // Additive detail objects — only emitted when populated, so
        // older clients see the unchanged three-field usage shape.
        if let Some(d) = &u.output_tokens_details {
            usage_obj.insert(
                "output_tokens_details".into(),
                json!({ "reasoning_tokens": d.reasoning_tokens }),
            );
        }
        if let Some(d) = &u.input_tokens_details {
            usage_obj.insert(
                "input_tokens_details".into(),
                json!({ "cached_tokens": d.cached_tokens }),
            );
        }
        obj.insert("usage".into(), Value::Object(usage_obj));
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
        // Same contract as the streaming shell: a truncated response
        // says so, or the caller cannot tell it from a fault.
        incomplete_details: matches!(reason, FinishReason::Length)
            .then(cortex_core::responses::IncompleteDetails::max_output_tokens),
        usage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortex_core::openai::MessageContent;

    /// Wrap typed items as `input` elements. Most translator tests
    /// exercise the typed path; the bare easy-message and unknown-item
    /// paths have dedicated tests below.
    fn typed_items(items: Vec<ResponsesInputItem>) -> ResponsesInput {
        ResponsesInput::Items(
            items
                .into_iter()
                .map(ResponsesInputElement::Typed)
                .collect(),
        )
    }

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
            top_k: None,
            seed: None,
            repetition_penalty: None,
            repeat_last_n: None,
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
            top_k: None,
            seed: None,
            repetition_penalty: None,
            repeat_last_n: None,
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
            top_k: None,
            seed: None,
            repetition_penalty: None,
            repeat_last_n: None,
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
            input: typed_items(vec![
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
            top_k: None,
            seed: None,
            repetition_penalty: None,
            repeat_last_n: None,
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
            input: typed_items(vec![ResponsesInputItem::Message {
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
            top_k: None,
            seed: None,
            repetition_penalty: None,
            repeat_last_n: None,
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
            input: typed_items(vec![ResponsesInputItem::Message {
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
            top_k: None,
            seed: None,
            repetition_penalty: None,
            repeat_last_n: None,
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
            input: typed_items(vec![ResponsesInputItem::Message {
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
            top_k: None,
            seed: None,
            repetition_penalty: None,
            repeat_last_n: None,
            previous_response_id: None,
            extra: Value::Object(Default::default()),
        };
        let chat = request_to_chat(req).unwrap();
        assert!(matches!(
            &chat.messages[0].content,
            MessageContent::Text(t) if t == "first\n\nsecond"
        ));
    }

    /// An item carrying no readable text — encrypted-content-only, or
    /// an empty summary — must not manufacture an empty assistant turn
    /// with a blank think block.
    #[test]
    fn a_reasoning_item_with_no_text_adds_no_turn() {
        let req = ResponsesRequest {
            model: "m".into(),
            input: typed_items(vec![
                ResponsesInputItem::Reasoning {
                    content: vec![],
                    summary: vec![],
                },
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
            top_k: None,
            seed: None,
            repetition_penalty: None,
            repeat_last_n: None,
            previous_response_id: None,
            extra: Value::Object(Default::default()),
        };
        let chat = request_to_chat(req).unwrap();
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "user");
    }

    /// The shape neuron itself emits — `summary` with a `summary_text`
    /// part — replayed by a client that round-trips the completed item
    /// verbatim, which is what pi-ai/DSH does.
    #[test]
    fn replayed_reasoning_rides_on_the_assistant_turn_it_belongs_to() {
        let raw = r#"{
            "model": "Qwen/Qwen3.8-27B",
            "input": [
                {"type": "message", "role": "user", "content": "build it"},
                {"type": "reasoning", "id": "rs_1", "status": "completed",
                 "summary": [{"type": "summary_text", "text": "the plan so far"}]},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "partial answer"}]},
                {"type": "message", "role": "user", "content": "continue"}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(raw).expect("parse");
        let chat = request_to_chat(req).expect("translate");
        assert_eq!(chat.messages.len(), 3);
        assert_eq!(chat.messages[1].role, "assistant");
        assert_eq!(
            chat.messages[1].extra.get("reasoning_content"),
            Some(&Value::String("the plan so far".into())),
            "the assistant turn must carry the thinking that produced it"
        );
    }

    /// OpenAI's o-series spelling: the text lives in `content` as
    /// `reasoning_text` parts. Same item, different field.
    #[test]
    fn the_content_spelling_of_a_reasoning_item_is_read_too() {
        let raw = r#"{
            "model": "m",
            "input": [
                {"type": "reasoning", "id": "rs_1",
                 "content": [{"type": "reasoning_text", "text": "step one"},
                             {"type": "reasoning_text", "text": "step two"}]},
                {"type": "message", "role": "assistant", "content": "done"}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(raw).expect("parse");
        let chat = request_to_chat(req).expect("translate");
        assert_eq!(
            chat.messages[0].extra.get("reasoning_content"),
            Some(&Value::String("step one\n\nstep two".into()))
        );
    }

    /// The case that motivated #277: a turn truncated mid-think, then
    /// `continue`. There is no assistant *message* to hang the
    /// reasoning on — the reasoning is the entire turn — so it must
    /// still reach the model, or the model restarts from nothing.
    #[test]
    fn a_truncated_turn_replayed_as_reasoning_alone_still_reaches_the_model() {
        let raw = r#"{
            "model": "Qwen/Qwen3.8-27B",
            "input": [
                {"type": "message", "role": "user", "content": "build a game"},
                {"type": "reasoning", "id": "rs_1", "status": "incomplete",
                 "summary": [{"type": "summary_text", "text": "24k tokens of design"}]},
                {"type": "message", "role": "user", "content": "continue"}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(raw).expect("parse");
        let chat = request_to_chat(req).expect("translate");
        let assistant: Vec<_> = chat
            .messages
            .iter()
            .filter(|m| m.role == "assistant")
            .collect();
        assert_eq!(
            assistant.len(),
            1,
            "the truncated turn must survive as a turn"
        );
        assert_eq!(
            assistant[0].extra.get("reasoning_content"),
            Some(&Value::String("24k tokens of design".into()))
        );
    }

    /// Consecutive reasoning items with no assistant turn between them
    /// accumulate rather than the later one erasing the earlier.
    #[test]
    fn consecutive_reasoning_items_accumulate() {
        let raw = r#"{
            "model": "m",
            "input": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "first"}]},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "second"}]},
                {"type": "message", "role": "assistant", "content": "ok"}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(raw).expect("parse");
        let chat = request_to_chat(req).expect("translate");
        assert_eq!(
            chat.messages[0].extra.get("reasoning_content"),
            Some(&Value::String("first\n\nsecond".into()))
        );
    }

    /// The control a Responses client actually reaches for (#223): pi-ai
    /// emits `reasoning: {"effort": "none"}` for thinking-level *off*.
    /// Before this it went nowhere, and a harness that asked for no
    /// thinking got 32,768 tokens of it and no answer.
    #[test]
    fn reasoning_effort_none_turns_thinking_off() {
        for effort in ["none", "off", "NONE", " off "] {
            let raw =
                format!(r#"{{"model":"m","input":"hi","reasoning":{{"effort":"{effort}"}}}}"#);
            let req: ResponsesRequest = serde_json::from_str(&raw).expect("parse");
            let chat = request_to_chat(req).expect("translate");
            assert_eq!(
                chat.extra
                    .get("chat_template_kwargs")
                    .and_then(|k| k.get("enable_thinking")),
                Some(&Value::Bool(false)),
                "effort {effort:?} must reach the template as enable_thinking=false"
            );
        }
    }

    /// `minimal` and `low` ask for *less* thinking, not none. The
    /// template switch is binary, so mapping them to off would be a
    /// silent substitution — they pass through untouched instead.
    #[test]
    fn a_reduced_effort_is_not_silently_treated_as_none() {
        for effort in ["minimal", "low", "medium", "high"] {
            let raw =
                format!(r#"{{"model":"m","input":"hi","reasoning":{{"effort":"{effort}"}}}}"#);
            let req: ResponsesRequest = serde_json::from_str(&raw).expect("parse");
            let chat = request_to_chat(req).expect("translate");
            assert!(
                chat.extra
                    .get("chat_template_kwargs")
                    .and_then(|k| k.get("enable_thinking"))
                    .is_none(),
                "effort {effort:?} must not be rewritten to off"
            );
        }
    }

    /// The more specific control wins, matching `default_enable_thinking`.
    #[test]
    fn an_explicit_template_kwarg_beats_the_effort_mapping() {
        let raw = r#"{"model":"m","input":"hi",
                      "reasoning":{"effort":"none"},
                      "chat_template_kwargs":{"enable_thinking":true}}"#;
        let req: ResponsesRequest = serde_json::from_str(raw).expect("parse");
        let chat = request_to_chat(req).expect("translate");
        assert_eq!(
            chat.extra
                .get("chat_template_kwargs")
                .and_then(|k| k.get("enable_thinking")),
            Some(&Value::Bool(true))
        );
    }

    /// A request with no reasoning control is untouched — the model
    /// thinks by default, which is the Responses-native expectation.
    #[test]
    fn no_reasoning_control_leaves_thinking_alone() {
        let raw = r#"{"model":"m","input":"hi"}"#;
        let req: ResponsesRequest = serde_json::from_str(raw).expect("parse");
        let chat = request_to_chat(req).expect("translate");
        assert!(chat.extra.get("chat_template_kwargs").is_none());
    }

    /// End to end, on the rendered prompt: Qwen3's template emits an
    /// empty think block when `enable_thinking` is false, and opens one
    /// otherwise. This asserts the caller's control reaches that branch.
    #[test]
    fn reasoning_effort_none_reaches_the_template_branch() {
        const QWEN_THINK_SWITCH: &str = "{%- if enable_thinking is defined and enable_thinking is false -%}\
<think>\n\n</think>\n\n\
{%- else -%}\
<think>\n\
{%- endif -%}";
        let raw = r#"{"model":"m","input":"hi","reasoning":{"effort":"none"}}"#;
        let req: ResponsesRequest = serde_json::from_str(raw).expect("parse");
        let chat = request_to_chat(req).expect("translate");
        let kwargs = chat
            .extra
            .get("chat_template_kwargs")
            .cloned()
            .unwrap_or(Value::Null);
        let out = crate::harness::chat_template::render_chat_template(
            QWEN_THINK_SWITCH,
            &chat.messages,
            &Value::Null,
            &kwargs,
        )
        .expect("render");
        assert!(
            out.contains("<think>\n\n</think>"),
            "the closed think block is what tells the model not to reason; got {out:?}"
        );
    }

    /// #223's root cause: `chat_template_kwargs` is how `enable_thinking`
    /// reaches the template, and the translator used to rebuild `extra`
    /// from an allowlist of one key, so it never arrived.
    #[test]
    fn extension_fields_survive_the_hop() {
        let raw = r#"{
            "model": "m",
            "input": "hello",
            "chat_template_kwargs": {"enable_thinking": false},
            "some_future_control": 7
        }"#;
        let req: ResponsesRequest = serde_json::from_str(raw).expect("parse");
        let chat = request_to_chat(req).expect("translate");
        assert_eq!(
            chat.extra
                .get("chat_template_kwargs")
                .and_then(|k| k.get("enable_thinking")),
            Some(&Value::Bool(false)),
            "the thinking switch must reach the chat path"
        );
        assert_eq!(chat.extra.get("some_future_control"), Some(&Value::from(7)));
    }

    /// The end-to-end claim, asserted where it actually matters: on the
    /// rendered prompt. A translator test proves we set a field; this
    /// proves the model sees the thinking, through the same
    /// `reasoning_content` branch Qwen3.8's real template uses.
    ///
    /// The same shape as the #179 system-prompt tests, and for the same
    /// reason — the contract is what reaches the model, not what the
    /// intermediate struct holds.
    #[test]
    fn replayed_reasoning_reaches_the_rendered_prompt() {
        // The assistant branch of Qwen3.8's chat_template, reduced to
        // the part under test.
        const QWEN_REASONING_LIKE: &str = "{%- for message in messages -%}\
{%- if message.role == 'assistant' -%}\
<|im_start|>assistant\n<think>\n{{ message.reasoning_content }}\n</think>\n\n{{ message.content }}<|im_end|>\n\
{%- else -%}\
<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n\
{%- endif -%}\
{%- endfor -%}";

        let raw = r#"{
            "model": "Qwen/Qwen3.8-27B",
            "input": [
                {"type": "message", "role": "user", "content": "build a game"},
                {"type": "reasoning", "id": "rs_1",
                 "summary": [{"type": "summary_text", "text": "the design so far"}]},
                {"type": "message", "role": "user", "content": "continue"}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(raw).expect("parse");
        let chat = request_to_chat(req).expect("translate");
        let prompt = crate::harness::chat_template::render_chat_template(
            QWEN_REASONING_LIKE,
            &chat.messages,
            &Value::Null,
            &Value::Null,
        )
        .expect("render");
        assert!(
            prompt.contains("<think>\nthe design so far\n</think>"),
            "the replayed reasoning must reach the prompt; got:\n{prompt}"
        );
    }

    /// Forwarding `extra` wholesale must not smuggle the Responses-shaped
    /// tool array through — the chat path reads the nested spelling.
    #[test]
    fn tools_are_still_rewritten_not_forwarded_raw() {
        let raw = r#"{
            "model": "m",
            "input": "hi",
            "tools": [{"type": "function", "name": "run_code",
                       "parameters": {"type": "object"}}]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(raw).expect("parse");
        let chat = request_to_chat(req).expect("translate");
        let tools = chat
            .extra
            .get("tools")
            .and_then(Value::as_array)
            .expect("tools");
        assert_eq!(
            tools[0]["function"]["name"],
            Value::String("run_code".into())
        );
        assert!(
            tools[0].get("name").is_none(),
            "the flat spelling must not survive"
        );
    }

    #[test]
    fn bare_easy_messages_translate_like_typed_messages() {
        // The agent-zero / litellm shape: bare `{role, content}` items
        // with no `type`. Deserialize from raw JSON (not hand-built)
        // so this exercises the real parse path end to end.
        let raw = r#"{
            "model": "Qwen/Qwen3.6-27B",
            "store": true,
            "input": [
                {"role": "system", "content": "be terse"},
                {"role": "assistant", "content": "{\"tool_name\":\"response\"}"},
                {"role": "user", "content": "alpha"}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(raw).unwrap();
        let chat = request_to_chat(req).unwrap();
        let roles: Vec<&str> = chat.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["system", "assistant", "user"]);
        assert!(matches!(
            &chat.messages[2].content,
            MessageContent::Text(t) if t == "alpha"
        ));
    }

    #[test]
    fn null_content_and_unknown_items_survive_translation() {
        // An assistant turn with `content: null` is kept (empty text);
        // an unmodeled item type is dropped, not rejected.
        let raw = r#"{
            "model": "m",
            "input": [
                {"role": "assistant", "content": null},
                {"type": "item_reference", "id": "x"},
                {"role": "user", "content": "go"}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(raw).unwrap();
        let chat = request_to_chat(req).unwrap();
        // assistant(null) kept, item_reference dropped, user kept.
        let roles: Vec<&str> = chat.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["assistant", "user"]);
        assert!(matches!(
            &chat.messages[0].content,
            MessageContent::Text(t) if t.is_empty()
        ));
    }

    #[test]
    fn function_call_output_array_renders_to_text() {
        // OpenAI allows `function_call_output.output` to be an array of
        // content parts; the tool result must reach the model as text.
        let raw = r#"{
            "model": "m",
            "input": [
                {"type": "function_call_output", "call_id": "c1",
                 "output": [{"type": "output_text", "text": "42"}]}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(raw).unwrap();
        let chat = request_to_chat(req).unwrap();
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "tool");
        match &chat.messages[0].content {
            MessageContent::Text(t) => assert!(t.contains("42"), "got {t:?}"),
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[test]
    fn tools_normalize_to_chat_wrapped_shape() {
        // Responses-flat function tools wrap into the chat shape the
        // templates expect; already-wrapped tools pass through; hosted
        // tool types we can't serve are dropped (#158).
        let raw = r#"{
            "model": "m",
            "input": "hi",
            "tools": [
                {"type": "function", "name": "shell", "description": "run a command",
                 "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                 "strict": false},
                {"type": "function", "function": {"name": "wrapped", "parameters": {}}},
                {"type": "web_search"}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(raw).unwrap();
        let chat = request_to_chat(req).unwrap();
        let tools = chat.extra.get("tools").unwrap().as_array().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "shell");
        assert_eq!(tools[0]["function"]["description"], "run a command");
        assert_eq!(
            tools[0]["function"]["parameters"]["properties"]["command"]["type"],
            "string"
        );
        assert_eq!(tools[1]["function"]["name"], "wrapped");
    }

    #[test]
    fn absent_tools_leave_extra_empty() {
        let raw = r#"{"model": "m", "input": "hi"}"#;
        let req: ResponsesRequest = serde_json::from_str(raw).unwrap();
        let chat = request_to_chat(req).unwrap();
        assert!(chat.extra.get("tools").is_none());
    }

    // ── streaming projector ─────────────────────────────────────────

    async fn collect(mut rx: mpsc::Receiver<ResponseStreamFrame>) -> Vec<ResponseStreamFrame> {
        let mut out = Vec::new();
        while let Some(f) = rx.recv().await {
            out.push(f);
        }
        out
    }

    /// The regression #267 was filed for: a reasoning model must put
    /// events on the wire while it thinks.
    ///
    /// These deltas used to be dropped, so a model that reasoned for
    /// minutes emitted nothing at all and clients closed the stream as
    /// dead — dsh reports `pi-ai stream idle timeout after 300000ms`
    /// and retries, so a task whose thinking outlasts the client's idle
    /// timeout could never finish. Its watchdog resets on any parsed
    /// event, which is why emitting these fixes it and why SSE comment
    /// keep-alives could not: idle timers count events, not comments.
    /// The Responses surface reports cache reuse too (#269), in its own
    /// spelling: `input_tokens_details.cached_tokens`.
    ///
    /// This is the surface dsh uses, and the one that showed
    /// `Cache hit 0%` against a 42.1K-token context while neuron was
    /// reusing thousands of tokens per prefill.
    #[tokio::test]
    async fn cached_tokens_reach_the_completed_usage() {
        let (tx, rx) = mpsc::channel::<InferenceEvent>(8);
        let out = project_responses_stream(rx, meta());

        tx.send(InferenceEvent::Start).await.unwrap();
        tx.send(InferenceEvent::TextDelta("ok".into()))
            .await
            .unwrap();
        tx.send(InferenceEvent::Finish {
            reason: FinishReason::Stop,
            prompt_tokens: 7413,
            completion_tokens: 12,
            reasoning_tokens: 0,
            cached_tokens: 2068,
            timing: None,
        })
        .await
        .unwrap();
        drop(tx);

        let frames = collect(out).await;
        let usage = &frames.last().unwrap().data["response"]["usage"];
        assert_eq!(usage["input_tokens_details"]["cached_tokens"], 2068);
        assert_eq!(usage["input_tokens"], 7413);
        // Sub-count, not additive — a cache hit must not inflate the total.
        assert_eq!(usage["total_tokens"], 7425);
    }

    #[tokio::test]
    async fn reasoning_projects_a_full_item_lifecycle_before_the_message() {
        let (tx, rx) = mpsc::channel::<InferenceEvent>(16);
        let out = project_responses_stream(rx, meta());

        tx.send(InferenceEvent::Start).await.unwrap();
        tx.send(InferenceEvent::ReasoningDelta("think".into()))
            .await
            .unwrap();
        tx.send(InferenceEvent::ReasoningDelta("ing".into()))
            .await
            .unwrap();
        tx.send(InferenceEvent::TextDelta("answer".into()))
            .await
            .unwrap();
        tx.send(InferenceEvent::Finish {
            reason: FinishReason::Stop,
            prompt_tokens: 0,
            completion_tokens: 0,
            reasoning_tokens: 2,
            cached_tokens: 0,
            timing: None,
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
                events::REASONING_SUMMARY_PART_ADDED,
                events::REASONING_SUMMARY_TEXT_DELTA,
                events::REASONING_SUMMARY_TEXT_DELTA,
                events::REASONING_SUMMARY_TEXT_DONE,
                events::REASONING_SUMMARY_PART_DONE,
                events::OUTPUT_ITEM_DONE,
                events::OUTPUT_ITEM_ADDED,
                events::CONTENT_PART_ADDED,
                events::OUTPUT_TEXT_DELTA,
                events::OUTPUT_TEXT_DONE,
                events::CONTENT_PART_DONE,
                events::OUTPUT_ITEM_DONE,
                events::COMPLETED,
            ]
        );

        // The reasoning item owns output_index 0, so the message shifts
        // to 1 — an item's index has to be right when it is announced,
        // which is why the message is opened lazily.
        let reasoning_added = &frames[2];
        assert_eq!(reasoning_added.data["output_index"], 0);
        assert_eq!(reasoning_added.data["item"]["type"], "reasoning");
        let message_added = &frames[9];
        assert_eq!(message_added.data["output_index"], 1);
        assert_eq!(message_added.data["item"]["type"], "message");
        assert_eq!(frames[11].data["output_index"], 1, "text delta index");

        // Accumulated thinking, not just the last fragment.
        assert_eq!(frames[6].data["text"], "thinking");

        // The terminal payload replays both items, in wire order, so a
        // client reading only `completed` sees what the stream showed.
        let output = frames.last().unwrap().data["response"]["output"]
            .as_array()
            .expect("output array");
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["summary"][0]["text"], "thinking");
        assert_eq!(output[1]["type"], "message");
    }

    /// A model can exhaust its budget mid-thought and never produce
    /// visible text. The reasoning item must still be closed and the
    /// message still opened, or the client waits on a response that
    /// never resolves.
    #[tokio::test]
    async fn reasoning_only_stream_still_closes_cleanly() {
        let (tx, rx) = mpsc::channel::<InferenceEvent>(16);
        let out = project_responses_stream(rx, meta());

        tx.send(InferenceEvent::Start).await.unwrap();
        tx.send(InferenceEvent::ReasoningDelta("pondering".into()))
            .await
            .unwrap();
        tx.send(InferenceEvent::Finish {
            reason: FinishReason::Length,
            prompt_tokens: 0,
            completion_tokens: 0,
            reasoning_tokens: 1,
            cached_tokens: 0,
            timing: None,
        })
        .await
        .unwrap();
        drop(tx);

        let frames = collect(out).await;
        let names: Vec<&str> = frames.iter().map(|f| f.event_name).collect();
        assert!(names.contains(&events::REASONING_SUMMARY_PART_DONE));
        assert!(
            names.contains(&events::CONTENT_PART_ADDED),
            "message item must still be opened so the finish frames refer to something"
        );
        // This is the live failure in miniature: the model spent its
        // whole budget thinking and produced almost no visible text.
        // The stream must still close, and must say *why* it stopped —
        // this frame used to be named `response.completed` while
        // carrying `status: "incomplete"`, an inconsistency that cost a
        // real agentic session its run.
        assert_eq!(names.last(), Some(&events::INCOMPLETE));
        let last = frames.last().unwrap();
        assert_eq!(last.data["response"]["status"], "incomplete");
        assert_eq!(
            last.data["response"]["incomplete_details"]["reason"],
            "max_output_tokens"
        );
    }

    /// Tool calls sit after the message, which sits after any reasoning
    /// item — so their index is relative to the message, not a
    /// hardcoded 0.
    #[tokio::test]
    async fn tool_call_index_shifts_past_a_reasoning_item() {
        let (tx, rx) = mpsc::channel::<InferenceEvent>(16);
        let out = project_responses_stream(rx, meta());

        tx.send(InferenceEvent::Start).await.unwrap();
        tx.send(InferenceEvent::ReasoningDelta("plan".into()))
            .await
            .unwrap();
        tx.send(InferenceEvent::ToolCall {
            index: 0,
            id: "call_1".into(),
            name: "bash".into(),
            arguments: "{}".into(),
        })
        .await
        .unwrap();
        tx.send(InferenceEvent::Finish {
            reason: FinishReason::ToolCalls,
            prompt_tokens: 0,
            completion_tokens: 0,
            reasoning_tokens: 1,
            cached_tokens: 0,
            timing: None,
        })
        .await
        .unwrap();
        drop(tx);

        let frames = collect(out).await;
        let call_added = frames
            .iter()
            .find(|f| {
                f.event_name == events::OUTPUT_ITEM_ADDED
                    && f.data["item"]["type"] == "function_call"
            })
            .expect("function_call item announced");
        // reasoning=0, message=1, first call=2.
        assert_eq!(call_added.data["output_index"], 2);
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
            reasoning_tokens: 0,
            cached_tokens: 0,
            timing: None,
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
    async fn tool_call_projects_function_call_event_family() {
        // A harness ToolCall becomes the OpenAI function_call event
        // family, and the completed response carries the item (#158).
        let (tx, rx) = mpsc::channel::<InferenceEvent>(8);
        let out = project_responses_stream(rx, meta());
        tx.send(InferenceEvent::Start).await.unwrap();
        tx.send(InferenceEvent::ToolCall {
            index: 0,
            id: "call_1".into(),
            name: "shell".into(),
            arguments: r#"{"command":"ls"}"#.into(),
        })
        .await
        .unwrap();
        tx.send(InferenceEvent::Finish {
            reason: FinishReason::ToolCalls,
            prompt_tokens: 0,
            completion_tokens: 0,
            reasoning_tokens: 0,
            cached_tokens: 0,
            timing: None,
        })
        .await
        .unwrap();
        drop(tx);
        let frames = collect(out).await;

        let added = frames
            .iter()
            .filter(|f| f.event_name == events::OUTPUT_ITEM_ADDED)
            .find(|f| f.data["item"]["type"] == "function_call")
            .expect("function_call output_item.added");
        assert_eq!(added.data["output_index"], 1);
        assert_eq!(added.data["item"]["call_id"], "call_1");
        assert_eq!(added.data["item"]["name"], "shell");
        assert_eq!(added.data["item"]["status"], "in_progress");
        let item_id = added.data["item"]["id"].as_str().unwrap().to_string();

        let delta = frames
            .iter()
            .find(|f| f.event_name == events::FUNCTION_CALL_ARGUMENTS_DELTA)
            .expect("arguments.delta");
        assert_eq!(delta.data["item_id"], item_id.as_str());
        assert_eq!(delta.data["delta"], r#"{"command":"ls"}"#);

        let done = frames
            .iter()
            .find(|f| f.event_name == events::FUNCTION_CALL_ARGUMENTS_DONE)
            .expect("arguments.done");
        assert_eq!(done.data["arguments"], r#"{"command":"ls"}"#);
        assert_eq!(done.data["call_id"], "call_1");
        assert_eq!(done.data["name"], "shell");

        let item_done = frames
            .iter()
            .filter(|f| f.event_name == events::OUTPUT_ITEM_DONE)
            .find(|f| f.data["item"]["type"] == "function_call")
            .expect("function_call output_item.done");
        assert_eq!(item_done.data["item"]["status"], "completed");
        assert_eq!(item_done.data["item"]["arguments"], r#"{"command":"ls"}"#);

        // ToolCalls finish maps to a completed response whose output
        // carries the message item plus the function_call item.
        let completed = frames
            .iter()
            .find(|f| f.event_name == events::COMPLETED)
            .unwrap();
        assert_eq!(completed.data["response"]["status"], "completed");
        let output = completed.data["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["type"], "message");
        assert_eq!(output[1]["type"], "function_call");
        assert_eq!(output[1]["call_id"], "call_1");
    }

    #[tokio::test]
    async fn every_frame_carries_in_payload_type_and_sequence_number() {
        // OpenAI-SDK-style clients (ZeroClaw, #156) dispatch on
        // `data.type`, never the SSE `event:` line. Every frame must
        // duplicate its event name into the payload and carry a
        // 0-based monotonic sequence_number.
        let (tx, rx) = mpsc::channel::<InferenceEvent>(8);
        let out = project_responses_stream(rx, meta());
        tx.send(InferenceEvent::Start).await.unwrap();
        tx.send(InferenceEvent::TextDelta("hi".into()))
            .await
            .unwrap();
        tx.send(InferenceEvent::Finish {
            reason: FinishReason::Stop,
            prompt_tokens: 0,
            completion_tokens: 0,
            reasoning_tokens: 0,
            cached_tokens: 0,
            timing: None,
        })
        .await
        .unwrap();
        drop(tx);
        let frames = collect(out).await;
        assert!(!frames.is_empty());
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(
                frame.data["type"], frame.event_name,
                "frame {i} missing in-payload type"
            );
            assert_eq!(
                frame.data["sequence_number"], i as u64,
                "frame {i} sequence_number mismatch"
            );
        }
    }

    #[tokio::test]
    async fn completed_frame_carries_usage_with_reasoning_detail() {
        let (tx, rx) = mpsc::channel::<InferenceEvent>(8);
        let out = project_responses_stream(rx, meta());
        tx.send(InferenceEvent::Start).await.unwrap();
        tx.send(InferenceEvent::Finish {
            reason: FinishReason::Stop,
            prompt_tokens: 30,
            completion_tokens: 12,
            reasoning_tokens: 4,
            cached_tokens: 0,
            timing: None,
        })
        .await
        .unwrap();
        drop(tx);
        let frames = collect(out).await;
        let completed = frames
            .iter()
            .find(|f| f.event_name == events::COMPLETED)
            .unwrap();
        let usage = &completed.data["response"]["usage"];
        assert_eq!(usage["input_tokens"], 30);
        assert_eq!(usage["output_tokens"], 12);
        // reasoning_tokens is a sub-count of output_tokens, not summed
        // into total_tokens.
        assert_eq!(usage["total_tokens"], 42);
        assert_eq!(usage["output_tokens_details"]["reasoning_tokens"], 4);
        // Deferred cache detail is absent until #11.
        assert!(usage.get("input_tokens_details").is_none());
    }

    #[tokio::test]
    async fn completed_frame_omits_reasoning_detail_for_non_reasoning() {
        let (tx, rx) = mpsc::channel::<InferenceEvent>(8);
        let out = project_responses_stream(rx, meta());
        tx.send(InferenceEvent::Start).await.unwrap();
        tx.send(InferenceEvent::Finish {
            reason: FinishReason::Stop,
            prompt_tokens: 8,
            completion_tokens: 3,
            reasoning_tokens: 0,
            cached_tokens: 0,
            timing: None,
        })
        .await
        .unwrap();
        drop(tx);
        let frames = collect(out).await;
        let completed = frames
            .iter()
            .find(|f| f.event_name == events::COMPLETED)
            .unwrap();
        let usage = &completed.data["response"]["usage"];
        assert_eq!(usage["output_tokens"], 3);
        assert!(usage.get("output_tokens_details").is_none());
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
            reasoning_tokens: 0,
            cached_tokens: 0,
            timing: None,
        })
        .await
        .unwrap();
        drop(tx);
        let frames = collect(out).await;
        // The terminal frame is `response.incomplete`, not
        // `response.completed` — a client keyed on the event name must
        // still see an end to the stream.
        assert!(
            !frames.iter().any(|f| f.event_name == events::COMPLETED),
            "a truncated response must not announce itself as completed"
        );
        let terminal = frames
            .iter()
            .find(|f| f.event_name == events::INCOMPLETE)
            .expect("a terminal frame is required");
        assert_eq!(terminal.data["response"]["status"], "incomplete");
        // The field that decides whether a client recovers or halts.
        assert_eq!(
            terminal.data["response"]["incomplete_details"]["reason"], "max_output_tokens",
            "pi-ai maps `incomplete` with no reason to stopReason:error \
             and ends the agent loop; with max_output_tokens it maps to \
             stopReason:length, fails truncated tool calls safely, and \
             takes another turn"
        );
    }

    /// The converse: a normal finish must stay `response.completed`
    /// and carry no reason, or every client sees phantom truncation.
    #[tokio::test]
    async fn a_normal_finish_carries_no_incomplete_reason() {
        let (tx, rx) = mpsc::channel::<InferenceEvent>(8);
        let out = project_responses_stream(rx, meta());
        tx.send(InferenceEvent::Start).await.unwrap();
        tx.send(InferenceEvent::Finish {
            reason: FinishReason::Stop,
            prompt_tokens: 0,
            completion_tokens: 0,
            reasoning_tokens: 0,
            cached_tokens: 0,
            timing: None,
        })
        .await
        .unwrap();
        drop(tx);
        let frames = collect(out).await;
        let completed = frames
            .iter()
            .find(|f| f.event_name == events::COMPLETED)
            .expect("a normal finish stays `response.completed`");
        assert_eq!(completed.data["response"]["status"], "completed");
        assert!(
            completed.data["response"]
                .get("incomplete_details")
                .is_none(),
            "a completed response must not carry an incomplete reason"
        );
    }

    /// A producer that drops without ever sending `Finish` did not
    /// finish — it died (poisoned model, OOM, dropped worker). The
    /// stream must still terminate, and must terminate as a *failure*.
    ///
    /// This previously asserted `COMPLETED`, encoding the defect: a
    /// crashed inference was announced as a complete response, so a
    /// client took whatever partial text had arrived to be the model's
    /// considered answer instead of retrying.
    #[tokio::test]
    async fn a_producer_that_dies_terminates_the_stream_as_failed() {
        let (tx, rx) = mpsc::channel::<InferenceEvent>(8);
        let out = project_responses_stream(rx, meta());
        drop(tx);
        let frames = collect(out).await;
        let names: Vec<&str> = frames.iter().map(|f| f.event_name).collect();
        // The shell still opens, so the failure refers to a response.
        assert!(names.contains(&events::CREATED));
        assert!(
            !names.contains(&events::COMPLETED),
            "a crash must never be announced as a completed response"
        );
        let failed = frames
            .iter()
            .find(|f| f.event_name == events::FAILED)
            .expect("the stream must terminate, or the client waits out its idle timeout");
        assert_eq!(failed.data["response"]["status"], "failed");
        assert_eq!(failed.data["response"]["error"]["code"], "server_error");
        assert!(
            failed.data["response"]["error"]["message"]
                .as_str()
                .is_some_and(|m| !m.is_empty()),
            "pi-ai renders `code: message`, so an empty message helps nobody"
        );
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
            reasoning_tokens: 0,
            cached_tokens: 0,
            timing: None,
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
                output_tokens_details: None,
                input_tokens_details: None,
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
