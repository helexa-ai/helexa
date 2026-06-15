//! Streaming Anthropic SSE translation (#24).
//!
//! The `/v1/messages` handler translates the request envelope to
//! OpenAI before proxying (see `cortex_core::translate`); this module
//! completes the round trip for `stream: true` — the upstream OpenAI
//! SSE stream is re-framed, event by event, into Anthropic's
//! `message_start` / `content_block_*` / `message_delta` /
//! `message_stop` sequence as it arrives. True streaming: each
//! upstream chunk is translated and forwarded immediately; nothing is
//! buffered beyond the current SSE event's bytes.
//!
//! The translation state machine itself is pure and lives in
//! [`cortex_core::translate::AnthropicStreamTranslator`]; this module
//! owns the wire concerns — splitting the upstream byte stream into
//! SSE events, parsing `data:` payloads, and framing the translated
//! events as `event: <name>\ndata: <json>\n\n`.

use axum::body::Body;
use axum::http::StatusCode;
use axum::response::Response;
use bytes::Bytes;
use cortex_core::openai::ChatCompletionChunk;
use cortex_core::translate::AnthropicStreamTranslator;
use futures::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

/// Forward the translated OpenAI request to the upstream node and
/// return the response translated to Anthropic SSE framing.
pub async fn stream_translated(
    client: &reqwest::Client,
    endpoint: &str,
    openai_body: axum::body::Bytes,
    model_id: &str,
    node_name: &str,
) -> Response {
    let url = format!("{endpoint}/v1/chat/completions");
    tracing::info!(
        handler = "anthropic_messages",
        model = %model_id,
        node = %node_name,
        url = %url,
        "proxying streaming request (anthropic SSE translation)"
    );

    let upstream = match client
        .post(&url)
        .header("content-type", "application/json")
        .body(openai_body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                handler = "anthropic_messages",
                node = %node_name,
                url = %url,
                error = %e,
                "anthropic stream: upstream request failed"
            );
            return anthropic_error(StatusCode::BAD_GATEWAY, "upstream request failed");
        }
    };

    let status = upstream.status();
    if !status.is_success() {
        tracing::warn!(
            handler = "anthropic_messages",
            node = %node_name,
            url = %url,
            status = status.as_u16(),
            "anthropic stream: upstream returned non-2xx"
        );
        return anthropic_error(
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            "upstream returned an error",
        );
    }

    // Bounded channel: a slow client back-pressures the pump task,
    // which back-pressures the upstream read — same propagation
    // discipline as neuron's own projectors.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::convert::Infallible>>(32);
    let node = node_name.to_string();
    let model = model_id.to_string();
    tokio::spawn(async move {
        let mut upstream = upstream.bytes_stream();
        let mut translator = AnthropicStreamTranslator::new();
        let mut buf: Vec<u8> = Vec::new();
        let mut done = false;
        // Wire-debug accounting for the stream summary emitted at the
        // end: did the model emit a structured tool call, what was the
        // final finish_reason, and how many upstream frames did we see.
        let mut saw_tool_call = false;
        let mut last_finish: Option<String> = None;
        let mut frames = 0u64;

        'outer: while let Some(block) = upstream.next().await {
            let block = match block {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(node = %node, error = %e, "anthropic stream: upstream read failed mid-stream");
                    break;
                }
            };
            buf.extend_from_slice(&block);
            // SSE events are separated by a blank line.
            while let Some(pos) = find_event_boundary(&buf) {
                let event: Vec<u8> = buf.drain(..pos + 2).collect();
                let text = String::from_utf8_lossy(&event);
                for line in text.lines() {
                    let Some(data) = line.strip_prefix("data:") else {
                        continue;
                    };
                    let data = data.trim();
                    if data == "[DONE]" {
                        done = true;
                        if !send_frames(&tx, translator.finish()).await {
                            break 'outer;
                        }
                        continue;
                    }
                    tracing::trace!(node = %node, frame = %data, "anthropic stream: upstream frame");
                    let Ok(chunk) = serde_json::from_str::<ChatCompletionChunk>(data) else {
                        tracing::debug!(node = %node, "anthropic stream: unparsable upstream frame skipped");
                        continue;
                    };
                    frames += 1;
                    if chunk
                        .choices
                        .iter()
                        .any(|c| c.delta.get("tool_calls").is_some())
                    {
                        saw_tool_call = true;
                    }
                    if let Some(fr) = chunk.choices.iter().find_map(|c| c.finish_reason.clone()) {
                        last_finish = Some(fr);
                    }
                    if !send_frames(&tx, translator.on_chunk(&chunk)).await {
                        break 'outer;
                    }
                }
            }
        }
        // Upstream ended without [DONE] (error or truncation): still
        // close the Anthropic event sequence so clients aren't left
        // with an unterminated message.
        if !done {
            let _ = send_frames(&tx, translator.finish()).await;
        }
        // Stream summary: the streaming counterpart to the non-streaming
        // handler's "upstream response" line. `upstream_tool_calls =
        // false` on a tools-bearing request is the fingerprint of the
        // model improvising an unparsed tool-call format.
        tracing::debug!(
            wire = "anthropic",
            model = %model,
            node = %node,
            frames,
            upstream_tool_calls = saw_tool_call,
            finish_reason = ?last_finish,
            terminated = done,
            "anthropic stream complete"
        );
    });

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap_or_else(|_| {
            anthropic_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to build response",
            )
        })
}

/// `\n\n` boundary of the first complete SSE event in `buf`, if any.
fn find_event_boundary(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\n\n")
}

/// Render translated events as SSE frames and send them. Returns
/// `false` when the client has gone away (receiver dropped).
async fn send_frames(
    tx: &tokio::sync::mpsc::Sender<Result<Bytes, std::convert::Infallible>>,
    events: Vec<(String, serde_json::Value)>,
) -> bool {
    for (name, payload) in events {
        let frame = format!("event: {name}\ndata: {payload}\n\n");
        if tx.send(Ok(Bytes::from(frame))).await.is_err() {
            return false;
        }
    }
    true
}

/// Anthropic-shaped error body (`{"type":"error","error":{...}}`).
fn anthropic_error(status: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({
        "type": "error",
        "error": { "type": "api_error", "message": message }
    });
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("static error response must build")
}
