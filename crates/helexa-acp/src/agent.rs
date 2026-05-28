//! ACP agent loop with tools and session modes (Stage 3).
//!
//! Handlers:
//!
//! | ACP method            | Behaviour                                                   |
//! |-----------------------|-------------------------------------------------------------|
//! | `initialize`          | echo protocol version, advertise capabilities               |
//! | `session/new`         | mint id, register state, advertise [Default, Bypass] modes  |
//! | `session/prompt`      | tool-call loop: stream → dispatch tools → re-enter, repeat  |
//! | `session/cancel`      | fire the session's cancellation token                       |
//! | `session/set_mode`    | mutate the session's mode (gated vs. bypass-permissions)    |
//! | (anything else)       | "not implemented yet" error                                 |
//!
//! Stage 4 wires `session/set_model`; Stage 5 flips on image content.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use agent_client_protocol::schema::{
    AgentCapabilities, CancelNotification, ContentBlock, InitializeRequest, InitializeResponse,
    NewSessionRequest, NewSessionResponse, PromptCapabilities, PromptRequest, PromptResponse,
    SessionId, SessionMode, SessionModeId, SessionModeState, SessionNotification, SessionUpdate,
    SetSessionModeRequest, SetSessionModeResponse, StopReason, TextContent,
};
use agent_client_protocol::{Agent as AgentRole, Client, ConnectionTo, Dispatch, Stdio};
use futures::StreamExt;
use std::collections::BTreeMap;
use tokio_util::sync::CancellationToken;

use crate::config::{Config, parse_model_selector};
use crate::prompt::build_system_prompt;
use crate::provider::{
    CompletionEvent, CompletionRequest, Message, MessageContent, Provider, Role, ToolCall,
};
use crate::session::{self, MODE_BYPASS, MODE_DEFAULT, SessionState, SessionStore};
use crate::tool_runner::{AcpClientOps, ToolCallEvent, dispatch_tool_call};
use crate::tools;

/// Maximum number of provider→tool→provider round-trips per
/// `session/prompt` request. Bound exists to keep a runaway model
/// from looping forever; the spec maps this to
/// [`StopReason::MaxTurnRequests`].
const MAX_TOOL_ROUNDS: usize = 25;

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
            .on_receive_request(
                {
                    let inner = inner.clone();
                    async move |req: SetSessionModeRequest, responder, _cx| {
                        match handle_set_session_mode(&inner, req).await {
                            Ok(()) => responder.respond(SetSessionModeResponse::new()),
                            Err(e) => responder.respond_with_internal_error(format!("{e:#}")),
                        }
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
    Ok(NewSessionResponse::new(session_id).modes(default_mode_state()))
}

/// The two modes every Stage 3 session advertises. Stage 7 may grow
/// this list (e.g. "plan" for plan-only output, "ask" for read-only),
/// but Default + Bypass cover the two operationally distinct
/// permission policies.
fn default_mode_state() -> SessionModeState {
    SessionModeState::new(
        SessionModeId::new(MODE_DEFAULT),
        vec![
            SessionMode::new(SessionModeId::new(MODE_DEFAULT), "Default")
                .description("Prompt for permission before writes or shell commands."),
            SessionMode::new(SessionModeId::new(MODE_BYPASS), "Bypass Permissions")
                .description("Auto-allow all tool calls. Use with care."),
        ],
    )
}

async fn handle_set_session_mode(
    inner: &AgentInner,
    req: SetSessionModeRequest,
) -> anyhow::Result<()> {
    let Some(state) = session::get(&inner.sessions, &req.session_id).await else {
        anyhow::bail!("unknown session id {}", req.session_id.0);
    };
    let accepted = req.mode_id.0.as_ref() == MODE_DEFAULT || req.mode_id.0.as_ref() == MODE_BYPASS;
    if !accepted {
        anyhow::bail!(
            "unknown mode '{}' — must be one of: {}, {}",
            req.mode_id.0,
            MODE_DEFAULT,
            MODE_BYPASS
        );
    }
    state.lock().await.mode_id = req.mode_id.clone();
    tracing::info!(
        session_id = %req.session_id.0,
        mode = %req.mode_id.0,
        "session mode changed"
    );
    Ok(())
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

    // Snapshot the inputs under the session lock, then drop the lock
    // before any `await` that touches the network. `mode_id` is
    // refreshed between tool rounds (the user can toggle modes
    // mid-turn).
    let (existing_history, model_id, cwd, cancel, mut mode_id) = {
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
            state.mode_id.clone(),
        )
    };

    let tool_specs = tools::all_tools();
    let system_prompt = build_system_prompt(&cwd, inner.system_prompt_path.as_deref(), &tool_specs)
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
        mode = %mode_id.0,
        history_turns = existing_history.len(),
        "sending prompt upstream"
    );

    let ops = AcpClientOps::new(cx.clone());

    // `messages` is the rolling conversation we send to the provider
    // each round. We seed it with the system prompt + the snapshot
    // (which includes the new user turn) and grow it with each
    // round's assistant turn + tool-result turns.
    let mut messages: Vec<Message> = Vec::with_capacity(existing_history.len() + 1);
    messages.push(Message {
        role: Role::System,
        content: MessageContent::Text(system_prompt),
    });
    messages.extend(existing_history);

    // Whatever new turns this prompt generates beyond the user's
    // input — we persist these to session.history at the end so
    // future prompts see them.
    let mut new_turns: Vec<Message> = Vec::new();

    let mut stop_reason = StopReason::EndTurn;

    for round in 0..MAX_TOOL_ROUNDS {
        if cancel.is_cancelled() {
            stop_reason = StopReason::Cancelled;
            break;
        }

        // Tool descriptions reach the model via the Qwen3 `# Tools`
        // block in the system prompt, not via the OpenAI `tools`
        // request field — cortex/neuron pass that field through to
        // the encoder unread, and including it would double-describe
        // tools once a strict-OpenAI backend lands. Leave empty.
        let completion_req = CompletionRequest {
            model: local_model.clone(),
            messages: messages.clone(),
            tools: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
        };

        let mut stream = match provider.complete(completion_req, cancel.clone()).await {
            Ok(s) => s,
            Err(e) => {
                let _ = responder
                    .respond_with_internal_error(format!("{} complete: {e:#}", provider.name()));
                return Ok(());
            }
        };

        let mut assistant_text = String::new();
        let mut finish_reason: Option<String> = None;
        // `BTreeMap` keyed by the provider's tool-call index keeps
        // insertion order while allowing arg deltas to mutate any
        // bucket — `ToolCallStart` may arrive interleaved with
        // `ToolCallArgsDelta` for different indices.
        let mut tool_buckets: BTreeMap<usize, ToolCallBucket> = BTreeMap::new();

        while let Some(event) = stream.next().await {
            let event = match event {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!(error = %format!("{e:#}"), "stream error; ending round");
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
                CompletionEvent::ToolCallStart { index, id, name } => {
                    tool_buckets.insert(
                        index,
                        ToolCallBucket {
                            id,
                            name,
                            arguments: String::new(),
                        },
                    );
                }
                CompletionEvent::ToolCallArgsDelta { index, args_delta } => {
                    tool_buckets
                        .entry(index)
                        .or_default()
                        .arguments
                        .push_str(&args_delta);
                }
                CompletionEvent::Finish { reason } => finish_reason = reason,
                CompletionEvent::Usage(_) => {}
            }
        }

        if cancel.is_cancelled() {
            stop_reason = StopReason::Cancelled;
            // Persist any partial text so the next turn has context.
            if !assistant_text.is_empty() {
                new_turns.push(Message {
                    role: Role::Assistant,
                    content: MessageContent::Text(assistant_text),
                });
            }
            break;
        }

        let has_tool_calls = !tool_buckets.is_empty();

        if !has_tool_calls {
            // Terminal turn: just text. Save and finish.
            if !assistant_text.is_empty() {
                new_turns.push(Message {
                    role: Role::Assistant,
                    content: MessageContent::Text(assistant_text),
                });
            }
            stop_reason = map_finish_reason(finish_reason.as_deref());
            break;
        }

        // Assistant turn carrying the tool calls.
        let calls: Vec<ToolCall> = tool_buckets
            .values()
            .map(|b| ToolCall {
                id: b.id.clone(),
                name: b.name.clone(),
                arguments: b.arguments.clone(),
            })
            .collect();
        let assistant_turn = Message {
            role: Role::Assistant,
            content: MessageContent::ToolCalls {
                text: (!assistant_text.is_empty()).then_some(assistant_text),
                calls,
            },
        };
        new_turns.push(assistant_turn.clone());
        messages.push(assistant_turn);

        // Refresh the mode in case the user toggled it during the
        // streaming above (cheap — one mutex acquisition).
        mode_id = session_arc.lock().await.mode_id.clone();

        // Dispatch every tool call sequentially. Parallelism is
        // tempting but would require Zed to handle interleaved
        // permission prompts; serial is friendlier.
        for bucket in tool_buckets.into_values() {
            if cancel.is_cancelled() {
                stop_reason = StopReason::Cancelled;
                break;
            }
            let event = ToolCallEvent {
                id: bucket.id,
                name: bucket.name,
                arguments: bucket.arguments,
            };
            let result =
                dispatch_tool_call(&ops, &session_id, &mode_id, &cwd, event, &cancel).await;
            let result_turn = Message {
                role: Role::Tool,
                content: MessageContent::ToolResult {
                    tool_call_id: result.tool_call_id,
                    content: result.content,
                },
            };
            new_turns.push(result_turn.clone());
            messages.push(result_turn);
        }

        if cancel.is_cancelled() {
            stop_reason = StopReason::Cancelled;
            break;
        }

        if round + 1 == MAX_TOOL_ROUNDS {
            tracing::warn!(
                session_id = %session_id.0,
                rounds = MAX_TOOL_ROUNDS,
                "hit MAX_TOOL_ROUNDS, returning MaxTurnRequests"
            );
            stop_reason = StopReason::MaxTurnRequests;
        }
    }

    {
        let mut state = session_arc.lock().await;
        state.history.extend(new_turns);
    }

    let _ = responder.respond(PromptResponse::new(stop_reason));
    Ok(())
}

/// Accumulator for one streamed tool call: the OpenAI wire format
/// sends `id` + `name` once (in the first chunk for that index) and
/// then argument bytes piecemeal. We gather them all before
/// dispatching.
#[derive(Debug, Default)]
struct ToolCallBucket {
    id: String,
    name: String,
    arguments: String,
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
