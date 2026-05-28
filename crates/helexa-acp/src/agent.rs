//! ACP agent loop — text-only (Stage 2).
//!
//! Handlers:
//!
//! | ACP method        | Behaviour                                                  |
//! |-------------------|------------------------------------------------------------|
//! | `initialize`      | echo client's protocol version, advertise capabilities     |
//! | `session/new`     | mint a session id, register state, return it               |
//! | `session/prompt`  | flatten user blocks → history, stream provider → updates   |
//! | `session/cancel`  | fire the session's cancellation token                      |
//! | (anything else)   | "not implemented yet" error                                |
//!
//! Stage 3 adds tool calls; Stage 4 wires `session/set_model`; Stage 5
//! flips on image content. Stage 2 deliberately answers the model-picker
//! and session-modes fields with `None` so editors render a single model
//! / single mode UI.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use agent_client_protocol::schema::{
    AgentCapabilities, CancelNotification, ContentBlock, InitializeRequest, InitializeResponse,
    NewSessionRequest, NewSessionResponse, PromptCapabilities, PromptRequest, PromptResponse,
    SessionId, SessionNotification, SessionUpdate, StopReason, TextContent,
};
use agent_client_protocol::{Agent as AgentRole, Client, ConnectionTo, Dispatch, Stdio};
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::config::{Config, parse_model_selector};
use crate::prompt::build_system_prompt;
use crate::provider::{
    CompletionEvent, CompletionRequest, Message, MessageContent, Provider, Role,
};
use crate::session::{self, SessionState, SessionStore};

/// Public entry point. Wraps an `Arc<AgentInner>` so handlers can clone
/// it cheaply into every closure.
pub struct Agent {
    inner: Arc<AgentInner>,
}

struct AgentInner {
    /// Every successfully-built provider, indexed positionally. We look
    /// providers up by name (`endpoint:` prefix) rather than by index.
    providers: Vec<Arc<dyn Provider>>,
    /// Name of the endpoint used when a request omits the
    /// `endpoint:model` prefix.
    default_endpoint_name: String,
    /// Default model for the default endpoint, if configured. Required
    /// for Stage 2 because session/set_model lands in Stage 4 — a
    /// session with no model can't prompt anything.
    default_model: Option<String>,
    sessions: SessionStore,
    system_prompt_path: Option<PathBuf>,
    /// Monotonic counter for minting session ids. The wire format is
    /// `hxa-{n}` — short, debuggable, and the protocol doesn't require
    /// UUIDs for session ids (it only requires them for message ids
    /// behind an unstable flag).
    next_session_id: AtomicU64,
}

impl Agent {
    /// Construct an agent from a validated [`Config`] and the providers
    /// that were successfully built for each endpoint.
    pub fn new(cfg: &Config, providers: Vec<Arc<dyn Provider>>) -> anyhow::Result<Self> {
        if providers.is_empty() {
            anyhow::bail!("no usable providers");
        }
        let default = cfg.default_endpoint();
        // The default endpoint's provider must have built successfully —
        // otherwise we can't honour `model = "bare-model-id"` requests.
        // (If only a non-default endpoint is usable, the operator should
        // promote it to `default_endpoint` in the TOML.)
        if !providers.iter().any(|p| p.name() == default.name) {
            anyhow::bail!(
                "default endpoint '{}' has no usable provider — check config",
                default.name
            );
        }
        Ok(Self {
            inner: Arc::new(AgentInner {
                providers,
                default_endpoint_name: default.name.clone(),
                default_model: default.default_model.clone(),
                sessions: session::new_store(),
                system_prompt_path: cfg.system_prompt_path.clone(),
                next_session_id: AtomicU64::new(1),
            }),
        })
    }

    /// Run the agent against an ACP transport (typically [`Stdio`]).
    /// Returns when the transport closes or a handler errors.
    pub async fn serve(self, transport: Stdio) -> agent_client_protocol::Result<()> {
        let inner = self.inner;
        AgentRole
            .builder()
            .name("helexa-acp")
            .on_receive_request(
                async move |req: InitializeRequest, responder, _cx| {
                    responder.respond(initialize_response(&req))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let inner = inner.clone();
                    async move |req: NewSessionRequest, responder, _cx| {
                        let result = handle_new_session(&inner, req).await;
                        match result {
                            Ok(resp) => responder.respond(resp),
                            Err(e) => responder.respond_with_internal_error(format!("{e:#}")),
                        }
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let inner = inner.clone();
                    async move |req: PromptRequest, responder, cx: ConnectionTo<Client>| {
                        spawn_prompt(inner.clone(), cx, req, responder)
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_notification(
                {
                    let inner = inner.clone();
                    async move |notif: CancelNotification, _cx: ConnectionTo<Client>| {
                        handle_cancel(&inner, notif).await;
                        Ok(())
                    }
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .on_receive_dispatch(
                async move |message: Dispatch, cx: ConnectionTo<Client>| {
                    tracing::warn!(method = ?message.method(), "unhandled ACP message");
                    message.respond_with_error(
                        agent_client_protocol::util::internal_error("not implemented yet"),
                        cx,
                    )
                },
                agent_client_protocol::on_receive_dispatch!(),
            )
            .connect_to(transport)
            .await
    }
}

fn initialize_response(req: &InitializeRequest) -> InitializeResponse {
    // Stage 2: text-only prompts. Image / audio / embedded resources
    // flip on in later stages.
    let prompt_caps = PromptCapabilities::default();
    InitializeResponse::new(req.protocol_version)
        .agent_capabilities(AgentCapabilities::new().prompt_capabilities(prompt_caps))
}

async fn handle_new_session(
    inner: &AgentInner,
    req: NewSessionRequest,
) -> anyhow::Result<NewSessionResponse> {
    if !req.cwd.is_absolute() {
        anyhow::bail!("session cwd must be absolute, got {}", req.cwd.display());
    }
    let model_id = inner
        .default_model
        .clone()
        .ok_or_else(|| anyhow::anyhow!(
            "default endpoint '{}' has no default_model — set one in config or wait for Stage 4 set_model",
            inner.default_endpoint_name
        ))?;

    let n = inner.next_session_id.fetch_add(1, Ordering::Relaxed);
    let session_id = SessionId::new(format!("hxa-{n}"));
    let cwd_display = req.cwd.display().to_string();
    let log_model = model_id.clone();
    let state = SessionState::new(req.cwd, model_id);
    session::insert(&inner.sessions, session_id.clone(), state).await;

    tracing::info!(
        session_id = %session_id.0,
        model_id = %log_model,
        cwd = %cwd_display,
        "session created"
    );
    Ok(NewSessionResponse::new(session_id))
}

async fn handle_cancel(inner: &AgentInner, notif: CancelNotification) {
    let Some(state) = session::get(&inner.sessions, &notif.session_id).await else {
        tracing::debug!(session_id = %notif.session_id.0, "cancel for unknown session, ignoring");
        return;
    };
    let cancel = state.lock().await.cancel.clone();
    tracing::info!(session_id = %notif.session_id.0, "cancellation requested");
    cancel.cancel();
}

/// Kick the prompt off on a spawned task so the event loop is free to
/// dispatch the matching `session/cancel`. The handler itself returns
/// `Ok(())` immediately (= `Handled::Yes`); the spawned task is what
/// eventually consumes `responder`.
fn spawn_prompt(
    inner: Arc<AgentInner>,
    cx: ConnectionTo<Client>,
    req: PromptRequest,
    responder: agent_client_protocol::Responder<PromptResponse>,
) -> agent_client_protocol::Result<()> {
    let task_cx = cx.clone();
    cx.spawn(async move {
        if let Err(e) = drive_prompt(inner, task_cx, req, responder).await {
            // `drive_prompt` already consumed the responder on the
            // error paths it produces; this branch only fires if the
            // task itself errored before reaching responder.respond.
            // Log and swallow — propagating the error would tear down
            // the whole connection, which is too violent for one
            // failed prompt.
            tracing::error!(error = %format!("{e:#}"), "prompt task failed");
        }
        Ok(())
    })?;
    Ok(())
}

async fn drive_prompt(
    inner: Arc<AgentInner>,
    cx: ConnectionTo<Client>,
    req: PromptRequest,
    responder: agent_client_protocol::Responder<PromptResponse>,
) -> anyhow::Result<()> {
    let session_id = req.session_id.clone();
    let Some(session_arc) = session::get(&inner.sessions, &session_id).await else {
        let _ =
            responder.respond_with_internal_error(format!("unknown session id {}", session_id.0));
        return Ok(());
    };

    // Snapshot the inputs to the upstream call under the session
    // lock, then drop the lock before any `await` that touches the
    // network. We *also* install a fresh cancellation token so
    // `session/cancel` can fire only this prompt.
    let (mut history, model_id, cwd, cancel) = {
        let mut state = session_arc.lock().await;
        let cancel = CancellationToken::new();
        state.cancel = cancel.clone();
        let user_text = flatten_prompt(&req.prompt);
        state.history.push(Message {
            role: Role::User,
            content: MessageContent::Text(user_text),
        });
        (
            state.history.clone(),
            state.model_id.clone(),
            state.cwd.clone(),
            cancel,
        )
    };

    let system_prompt = build_system_prompt(&cwd, inner.system_prompt_path.as_deref())
        .map_err(|e| anyhow::anyhow!("build system prompt: {e:#}"))?;

    let (provider, local_model) =
        match resolve_provider(&inner.providers, &inner.default_endpoint_name, &model_id) {
            Ok(pair) => pair,
            Err(e) => {
                let _ = responder.respond_with_internal_error(format!("{e:#}"));
                return Ok(());
            }
        };

    tracing::info!(
        session_id = %session_id.0,
        endpoint = %provider.name(),
        model = %local_model,
        history_turns = history.len(),
        "sending prompt upstream"
    );

    let mut messages = Vec::with_capacity(history.len() + 1);
    messages.push(Message {
        role: Role::System,
        content: MessageContent::Text(system_prompt),
    });
    messages.append(&mut history);

    let completion_req = CompletionRequest {
        model: local_model,
        messages,
        tools: vec![],
        temperature: None,
        top_p: None,
        max_tokens: None,
    };

    let stream_result = provider.complete(completion_req, cancel.clone()).await;
    let mut stream = match stream_result {
        Ok(s) => s,
        Err(e) => {
            let _ = responder
                .respond_with_internal_error(format!("{} complete: {e:#}", provider.name()));
            return Ok(());
        }
    };

    let mut assistant_text = String::new();
    let mut stop_reason = StopReason::EndTurn;

    while let Some(event) = stream.next().await {
        let event = match event {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(error = %format!("{e:#}"), "stream error; ending turn");
                break;
            }
        };
        match event {
            CompletionEvent::TextDelta(t) => {
                assistant_text.push_str(&t);
                send_chunk(
                    &cx,
                    &session_id,
                    SessionUpdate::AgentMessageChunk(text_chunk(t)),
                );
            }
            CompletionEvent::ReasoningDelta(t) => {
                send_chunk(
                    &cx,
                    &session_id,
                    SessionUpdate::AgentThoughtChunk(text_chunk(t)),
                );
            }
            CompletionEvent::Finish { reason } => {
                stop_reason = map_finish_reason(reason.as_deref());
            }
            // Stage 2 ignores tool calls and usage. Tool calls land in
            // Stage 3; usage telemetry isn't in the (non-unstable)
            // PromptResponse, so there's nothing to attach it to today.
            CompletionEvent::ToolCallStart { .. }
            | CompletionEvent::ToolCallArgsDelta { .. }
            | CompletionEvent::Usage(_) => {}
        }
    }

    // If cancellation fired, override whatever finish reason we got
    // (or didn't get). Per spec: a `session/cancel` MUST result in
    // `StopReason::Cancelled`, regardless of partial output.
    if cancel.is_cancelled() {
        stop_reason = StopReason::Cancelled;
    }

    // Re-acquire the lock just long enough to persist the assistant
    // turn (even partial output, so future turns have the context).
    {
        let mut state = session_arc.lock().await;
        if !assistant_text.is_empty() {
            state.history.push(Message {
                role: Role::Assistant,
                content: MessageContent::Text(assistant_text),
            });
        }
    }

    let _ = responder.respond(PromptResponse::new(stop_reason));
    Ok(())
}

fn send_chunk(cx: &ConnectionTo<Client>, session_id: &SessionId, update: SessionUpdate) {
    let notif = SessionNotification::new(session_id.clone(), update);
    if let Err(e) = cx.send_notification(notif) {
        tracing::warn!(error = %format!("{e:#}"), "failed to forward session update");
    }
}

fn text_chunk(text: String) -> agent_client_protocol::schema::ContentChunk {
    use agent_client_protocol::schema::ContentChunk;
    ContentChunk::new(ContentBlock::Text(TextContent::new(text)))
}

fn map_finish_reason(reason: Option<&str>) -> StopReason {
    match reason {
        Some("length") => StopReason::MaxTokens,
        Some("refusal") => StopReason::Refusal,
        // "stop", "tool_calls" (no tools in Stage 2 — degrade to
        // EndTurn so we don't surface a bogus reason), missing, or
        // anything else → EndTurn.
        _ => StopReason::EndTurn,
    }
}

/// Pure helper — turn a prompt's ContentBlocks into the user-message
/// text that goes into history. Lifted out so unit tests don't need a
/// running runtime.
fn flatten_prompt(blocks: &[ContentBlock]) -> String {
    let mut out = String::new();
    for block in blocks {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        match block {
            ContentBlock::Text(t) => out.push_str(&t.text),
            ContentBlock::ResourceLink(link) => {
                // Stage 2 has no fs access; surface the link as a
                // textual reference so the model at least knows it
                // was asked about something.
                out.push_str(&format!("[resource link: {}]", link.uri));
            }
            // Image / Audio / Resource: not advertised in
            // PromptCapabilities for Stage 2; a well-behaved client
            // shouldn't send these. If one does, drop and warn.
            other => {
                tracing::warn!(?other, "ignoring unsupported content block in Stage 2");
            }
        }
    }
    out
}

/// Pure helper — pick which provider handles a session's `model_id`.
/// Returns the matching provider plus the endpoint-local model id
/// (i.e. with any `endpoint:` prefix stripped).
fn resolve_provider(
    providers: &[Arc<dyn Provider>],
    default_endpoint: &str,
    model_id: &str,
) -> anyhow::Result<(Arc<dyn Provider>, String)> {
    let (endpoint_hint, local_model) = parse_model_selector(model_id);
    let target_endpoint = endpoint_hint.unwrap_or(default_endpoint);
    let provider = providers
        .iter()
        .find(|p| p.name() == target_endpoint)
        .ok_or_else(|| anyhow::anyhow!("no provider for endpoint '{target_endpoint}'"))?;
    Ok((provider.clone(), local_model.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::ResourceLink;
    use async_trait::async_trait;
    use futures::stream::BoxStream;

    // ── flatten_prompt ──────────────────────────────────────────────

    #[test]
    fn flatten_empty_prompt_is_empty() {
        assert_eq!(flatten_prompt(&[]), "");
    }

    #[test]
    fn flatten_joins_text_blocks_with_blank_line() {
        let blocks = vec![
            ContentBlock::Text(TextContent::new("first")),
            ContentBlock::Text(TextContent::new("second")),
        ];
        assert_eq!(flatten_prompt(&blocks), "first\n\nsecond");
    }

    #[test]
    fn flatten_resource_link_becomes_reference_line() {
        let blocks = vec![ContentBlock::ResourceLink(ResourceLink::new(
            "readme",
            "file:///tmp/x",
        ))];
        assert_eq!(flatten_prompt(&blocks), "[resource link: file:///tmp/x]");
    }

    // ── resolve_provider ────────────────────────────────────────────

    /// Minimal Provider stub; just records its name. The trait methods
    /// aren't exercised by resolve_provider so we leave them
    /// unimplemented.
    struct StubProvider(&'static str);

    #[async_trait]
    impl Provider for StubProvider {
        fn name(&self) -> &str {
            self.0
        }
        async fn list_models(&self) -> anyhow::Result<Vec<crate::provider::ModelInfo>> {
            unimplemented!()
        }
        async fn complete(
            &self,
            _request: CompletionRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, anyhow::Result<CompletionEvent>>> {
            unimplemented!()
        }
    }

    fn providers() -> Vec<Arc<dyn Provider>> {
        vec![
            Arc::new(StubProvider("helexa")),
            Arc::new(StubProvider("openrouter")),
        ]
    }

    #[test]
    fn bare_model_routes_to_default() {
        let (p, m) = resolve_provider(&providers(), "helexa", "helexa/large").unwrap();
        assert_eq!(p.name(), "helexa");
        assert_eq!(m, "helexa/large");
    }

    #[test]
    fn prefixed_model_routes_by_endpoint() {
        let (p, m) =
            resolve_provider(&providers(), "helexa", "openrouter:anthropic/claude-opus-4").unwrap();
        assert_eq!(p.name(), "openrouter");
        assert_eq!(m, "anthropic/claude-opus-4");
    }

    #[test]
    fn unknown_endpoint_errors() {
        // `Arc<dyn Provider>` doesn't impl Debug, which rules out
        // `.unwrap_err()` (it requires T: Debug). Pattern-match instead.
        match resolve_provider(&providers(), "helexa", "ghost:gpt-9") {
            Ok(_) => panic!("expected error for unknown endpoint"),
            Err(e) => assert!(format!("{e}").contains("ghost")),
        }
    }

    // ── map_finish_reason ───────────────────────────────────────────

    #[test]
    fn maps_known_finish_reasons() {
        assert!(matches!(
            map_finish_reason(Some("length")),
            StopReason::MaxTokens
        ));
        assert!(matches!(
            map_finish_reason(Some("refusal")),
            StopReason::Refusal
        ));
        assert!(matches!(
            map_finish_reason(Some("stop")),
            StopReason::EndTurn
        ));
        assert!(matches!(
            map_finish_reason(Some("tool_calls")),
            StopReason::EndTurn
        ));
        assert!(matches!(map_finish_reason(None), StopReason::EndTurn));
    }
}
