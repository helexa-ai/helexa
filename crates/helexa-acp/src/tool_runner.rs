//! Execute the LLM's tool calls against the editor client.
//!
//! Each tool call goes through `dispatch_tool_call`, which:
//!
//! 1. Emits `SessionUpdate::ToolCall { status: Pending }` so the
//!    editor can show "agent is about to do X".
//! 2. If the tool is gated (write / edit / bash) and the session mode
//!    is the default, asks the user via `session/request_permission`.
//!    Bypass mode skips this step.
//! 3. Executes the tool by calling the appropriate ACP client method
//!    (`fs/read_text_file`, `fs/write_text_file`, `terminal/create` +
//!    `terminal/wait_for_exit` + `terminal/output` + `terminal/release`)
//!    or, for `list_dir`, a local `std::fs` call.
//! 4. Emits `SessionUpdate::ToolCallUpdate { status: Completed | Failed }`
//!    with the result content (text, diff, or error).
//! 5. Returns a [`ToolResult`] string that the agent loop folds back
//!    into the model's conversation history.
//!
//! Client-side ACP calls are abstracted behind the [`ClientOps`]
//! trait. Production wires it to a real `ConnectionTo<Client>`; tests
//! pass in a recording fake.

use std::path::{Path, PathBuf};

use agent_client_protocol::schema::{
    ContentBlock, CreateTerminalRequest, Diff, KillTerminalRequest, PermissionOption,
    PermissionOptionId, PermissionOptionKind, ReadTextFileRequest, ReleaseTerminalRequest,
    RequestPermissionOutcome, RequestPermissionRequest, SessionId, SessionModeId,
    SessionNotification, SessionUpdate, TerminalExitStatus, TerminalId, TerminalOutputRequest,
    TerminalOutputResponse, TextContent, ToolCall, ToolCallContent, ToolCallId, ToolCallStatus,
    ToolCallUpdate, ToolCallUpdateFields, ToolKind, WaitForTerminalExitRequest,
    WriteTextFileRequest,
};
use agent_client_protocol::{Client, ConnectionTo, util::internal_error};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio_util::sync::CancellationToken;

use crate::path_util::expand_path;
use crate::session::{MODE_BYPASS, MODE_DEFAULT, MODE_PLAN};
use crate::store;
use crate::tools::{BASH, EDIT_FILE, LIST_DIR, READ_FILE, WRITE_FILE};

/// Accumulated state of a single tool call streamed from the
/// provider. The agent loop gathers `ToolCallStart` + N
/// `ToolCallArgsDelta` events into one of these before dispatch.
#[derive(Debug, Clone)]
pub struct ToolCallEvent {
    /// Provider-assigned id (e.g. OpenAI's `call_…`). Used as the
    /// `tool_call_id` in both the assistant turn and the tool-result
    /// turn we feed back to the model.
    pub id: String,
    pub name: String,
    /// Concatenated JSON argument bytes. Parsed lazily by the runner.
    pub arguments: String,
}

/// What the runner sends back to the agent loop after a dispatch.
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// Echoes [`ToolCallEvent::id`] so the agent can build the
    /// `MessageContent::ToolResult { tool_call_id, … }` history entry.
    pub tool_call_id: String,
    /// Human/agent-readable result text. Always non-empty: errors are
    /// stringified so the model can react to them.
    pub content: String,
    /// True for failures (so the agent can decide whether to stop on
    /// repeated tool errors — currently unused but worth surfacing).
    #[allow(dead_code)]
    pub is_error: bool,
}

/// Client-side ACP RPCs the runner needs. Real wiring lives in
/// [`AcpClientOps`]; tests use a recording fake.
#[async_trait]
pub trait ClientOps: Send + Sync {
    async fn read_text_file(
        &self,
        session: &SessionId,
        path: PathBuf,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> anyhow::Result<String>;

    async fn write_text_file(
        &self,
        session: &SessionId,
        path: PathBuf,
        content: String,
    ) -> anyhow::Result<()>;

    async fn request_permission(
        &self,
        session: &SessionId,
        tool_call: ToolCallUpdate,
        options: Vec<PermissionOption>,
    ) -> anyhow::Result<RequestPermissionOutcome>;

    async fn create_terminal(
        &self,
        session: &SessionId,
        command: String,
        args: Vec<String>,
        cwd: Option<PathBuf>,
    ) -> anyhow::Result<TerminalId>;

    async fn wait_for_terminal_exit(
        &self,
        session: &SessionId,
        terminal: &TerminalId,
    ) -> anyhow::Result<TerminalExitStatus>;

    async fn terminal_output(
        &self,
        session: &SessionId,
        terminal: &TerminalId,
    ) -> anyhow::Result<TerminalOutputResponse>;

    async fn kill_terminal(&self, session: &SessionId, terminal: &TerminalId)
    -> anyhow::Result<()>;

    async fn release_terminal(
        &self,
        session: &SessionId,
        terminal: &TerminalId,
    ) -> anyhow::Result<()>;

    /// Fire-and-forget. Failures are logged inside the impl, not
    /// propagated — losing a `session/update` is non-fatal.
    fn send_session_update(&self, session: &SessionId, update: SessionUpdate);
}

/// Production wrapper around a live ACP connection.
pub struct AcpClientOps {
    cx: ConnectionTo<Client>,
}

impl AcpClientOps {
    pub fn new(cx: ConnectionTo<Client>) -> Self {
        Self { cx }
    }
}

#[async_trait]
impl ClientOps for AcpClientOps {
    async fn read_text_file(
        &self,
        session: &SessionId,
        path: PathBuf,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> anyhow::Result<String> {
        let mut req = ReadTextFileRequest::new(session.clone(), path);
        req = req.line(line).limit(limit);
        let resp = self
            .cx
            .send_request(req)
            .block_task()
            .await
            .map_err(|e| anyhow::anyhow!("fs/read_text_file: {e}"))?;
        Ok(resp.content)
    }

    async fn write_text_file(
        &self,
        session: &SessionId,
        path: PathBuf,
        content: String,
    ) -> anyhow::Result<()> {
        let req = WriteTextFileRequest::new(session.clone(), path, content);
        self.cx
            .send_request(req)
            .block_task()
            .await
            .map_err(|e| anyhow::anyhow!("fs/write_text_file: {e}"))?;
        Ok(())
    }

    async fn request_permission(
        &self,
        session: &SessionId,
        tool_call: ToolCallUpdate,
        options: Vec<PermissionOption>,
    ) -> anyhow::Result<RequestPermissionOutcome> {
        let req = RequestPermissionRequest::new(session.clone(), tool_call, options);
        let resp = self
            .cx
            .send_request(req)
            .block_task()
            .await
            .map_err(|e| anyhow::anyhow!("session/request_permission: {e}"))?;
        Ok(resp.outcome)
    }

    async fn create_terminal(
        &self,
        session: &SessionId,
        command: String,
        args: Vec<String>,
        cwd: Option<PathBuf>,
    ) -> anyhow::Result<TerminalId> {
        let mut req = CreateTerminalRequest::new(session.clone(), command).args(args);
        req = req.cwd(cwd);
        let resp = self
            .cx
            .send_request(req)
            .block_task()
            .await
            .map_err(|e| anyhow::anyhow!("terminal/create: {e}"))?;
        Ok(resp.terminal_id)
    }

    async fn wait_for_terminal_exit(
        &self,
        session: &SessionId,
        terminal: &TerminalId,
    ) -> anyhow::Result<TerminalExitStatus> {
        let req = WaitForTerminalExitRequest::new(session.clone(), terminal.clone());
        let resp = self
            .cx
            .send_request(req)
            .block_task()
            .await
            .map_err(|e| anyhow::anyhow!("terminal/wait_for_exit: {e}"))?;
        Ok(resp.exit_status)
    }

    async fn terminal_output(
        &self,
        session: &SessionId,
        terminal: &TerminalId,
    ) -> anyhow::Result<TerminalOutputResponse> {
        let req = TerminalOutputRequest::new(session.clone(), terminal.clone());
        let resp = self
            .cx
            .send_request(req)
            .block_task()
            .await
            .map_err(|e| anyhow::anyhow!("terminal/output: {e}"))?;
        Ok(resp)
    }

    async fn kill_terminal(
        &self,
        session: &SessionId,
        terminal: &TerminalId,
    ) -> anyhow::Result<()> {
        let req = KillTerminalRequest::new(session.clone(), terminal.clone());
        self.cx
            .send_request(req)
            .block_task()
            .await
            .map_err(|e| anyhow::anyhow!("terminal/kill: {e}"))?;
        Ok(())
    }

    async fn release_terminal(
        &self,
        session: &SessionId,
        terminal: &TerminalId,
    ) -> anyhow::Result<()> {
        let req = ReleaseTerminalRequest::new(session.clone(), terminal.clone());
        self.cx
            .send_request(req)
            .block_task()
            .await
            .map_err(|e| anyhow::anyhow!("terminal/release: {e}"))?;
        Ok(())
    }

    fn send_session_update(&self, session: &SessionId, update: SessionUpdate) {
        let notif = SessionNotification::new(session.clone(), update);
        if let Err(e) = self.cx.send_notification(notif) {
            tracing::warn!(error = %internal_error(format!("{e}")), "session/update notification dropped");
        }
    }
}

// ── Tool argument shapes ─────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ReadFileArgs {
    path: PathBuf,
    #[serde(default)]
    line: Option<u32>,
    #[serde(default)]
    limit: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct WriteFileArgs {
    path: PathBuf,
    content: String,
}

#[derive(Debug, Deserialize)]
struct EditFileArgs {
    path: PathBuf,
    old_text: String,
    new_text: String,
}

#[derive(Debug, Deserialize)]
struct ListDirArgs {
    path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct BashArgs {
    command: String,
    #[serde(default)]
    cwd: Option<PathBuf>,
}

// ── Dispatch ─────────────────────────────────────────────────────────

/// Tools whose default-mode behaviour is to ask the user first.
pub fn is_gated(tool_name: &str) -> bool {
    matches!(tool_name, WRITE_FILE | EDIT_FILE | BASH)
}

/// Map a tool name to the [`ToolKind`] icon hint Zed uses.
fn tool_kind(name: &str) -> ToolKind {
    match name {
        READ_FILE | LIST_DIR => ToolKind::Read,
        WRITE_FILE | EDIT_FILE => ToolKind::Edit,
        BASH => ToolKind::Execute,
        _ => ToolKind::Other,
    }
}

/// Human-readable one-line title shown next to the tool-call card.
fn tool_title(name: &str, args_value: &serde_json::Value) -> String {
    fn path(args: &serde_json::Value) -> &str {
        args.get("path").and_then(|v| v.as_str()).unwrap_or("?")
    }
    match name {
        READ_FILE => format!("Read {}", path(args_value)),
        WRITE_FILE => format!("Write {}", path(args_value)),
        EDIT_FILE => format!("Edit {}", path(args_value)),
        LIST_DIR => format!("List {}", path(args_value)),
        BASH => {
            let cmd = args_value
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let snippet = if cmd.len() > 60 {
                format!("{}…", &cmd[..60])
            } else {
                cmd.to_string()
            };
            format!("Run: {snippet}")
        }
        other => format!("Tool: {other}"),
    }
}

/// Run a single tool call. Always returns a [`ToolResult`] — failures
/// are reported as `is_error = true` strings, not Err.
pub async fn dispatch_tool_call(
    ops: &dyn ClientOps,
    session_id: &SessionId,
    mode: &SessionModeId,
    session_cwd: &Path,
    call: ToolCallEvent,
    cancel: &CancellationToken,
) -> ToolResult {
    let tool_call_id = ToolCallId::new(call.id.clone());

    // Parse args once, up front. If the model produced invalid JSON
    // we surface that to it so it can retry rather than to the user.
    let args_value: serde_json::Value = match serde_json::from_str(&call.arguments) {
        Ok(v) => v,
        Err(e) => {
            let msg = format!("tool '{}' had invalid JSON arguments: {e}", call.name);
            let init = ToolCall::new(tool_call_id.clone(), tool_title(&call.name, &json!({})))
                .kind(tool_kind(&call.name))
                .status(ToolCallStatus::Failed)
                .content(vec![ToolCallContent::Content(
                    agent_client_protocol::schema::Content::new(ContentBlock::Text(
                        TextContent::new(msg.clone()),
                    )),
                )])
                .raw_input(serde_json::Value::String(call.arguments.clone()));
            ops.send_session_update(session_id, SessionUpdate::ToolCall(init));
            return ToolResult {
                tool_call_id: call.id,
                content: msg,
                is_error: true,
            };
        }
    };

    let title = tool_title(&call.name, &args_value);
    let kind = tool_kind(&call.name);
    let initial = ToolCall::new(tool_call_id.clone(), title)
        .kind(kind)
        .status(ToolCallStatus::Pending)
        .raw_input(args_value.clone());
    ops.send_session_update(session_id, SessionUpdate::ToolCall(initial));

    if cancel.is_cancelled() {
        return finish_failed(
            ops,
            session_id,
            &tool_call_id,
            &call.id,
            "cancelled before tool ran",
        );
    }

    // ── Plan-mode gate ───────────────────────────────────────────────
    // Plan mode is the most restrictive: bash is disabled outright,
    // writes are confined to the plan directory, and there is no
    // permission prompt (writes inside plan_dir auto-allow because
    // writing the plan IS the whole purpose). Reads pass through.
    if mode.0.as_ref() == MODE_PLAN {
        let plan_dir = store::plan_dir_for(session_cwd);
        match call.name.as_str() {
            BASH => {
                return finish_failed(
                    ops,
                    session_id,
                    &tool_call_id,
                    &call.id,
                    "plan mode: shell execution is disabled. Switch to Default or \
                     Bypass Permissions to run commands.",
                );
            }
            WRITE_FILE | EDIT_FILE => {
                let path = args_value
                    .get("path")
                    .and_then(|v| v.as_str())
                    .map(|s| expand_path(std::path::Path::new(s)));
                let inside_plan_dir = match (path.as_deref(), plan_dir.as_deref()) {
                    (Some(p), Some(pd)) => p.starts_with(pd),
                    _ => false,
                };
                if !inside_plan_dir {
                    let plan_dir_str = plan_dir
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "<unresolved>".to_string());
                    let attempted = path
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "<missing path>".to_string());
                    return finish_failed(
                        ops,
                        session_id,
                        &tool_call_id,
                        &call.id,
                        &format!(
                            "plan mode: writes are restricted to {plan_dir_str}; \
                             refused write to {attempted}."
                        ),
                    );
                }
                // Inside plan_dir: skip the permission prompt and
                // fall through to execution.
            }
            _ => {
                // read_file, list_dir: allowed without prompt.
            }
        }
    } else if is_gated(&call.name) && mode.0.as_ref() != MODE_BYPASS {
        // Default mode (or any non-bypass id): always ask. The user's
        // "Allow" decision is per-call here; we don't carry over an
        // "Allow always" across calls — that's a Stage 7 polish item
        // (persisted permission grants).
        let _ = mode.0.as_ref() == MODE_DEFAULT; // explicit acknowledgement that's our intent
        let options = vec![
            PermissionOption::new(
                PermissionOptionId::new("allow_once"),
                "Allow",
                PermissionOptionKind::AllowOnce,
            ),
            PermissionOption::new(
                PermissionOptionId::new("reject_once"),
                "Reject",
                PermissionOptionKind::RejectOnce,
            ),
        ];
        let permission_call =
            ToolCallUpdate::new(tool_call_id.clone(), ToolCallUpdateFields::new());
        match ops
            .request_permission(session_id, permission_call, options)
            .await
        {
            Ok(RequestPermissionOutcome::Selected(sel))
                if sel.option_id.0.as_ref().starts_with("allow") => {}
            Ok(RequestPermissionOutcome::Selected(_)) => {
                return finish_failed(
                    ops,
                    session_id,
                    &tool_call_id,
                    &call.id,
                    "user rejected the action",
                );
            }
            Ok(RequestPermissionOutcome::Cancelled) => {
                return finish_failed(
                    ops,
                    session_id,
                    &tool_call_id,
                    &call.id,
                    "permission request cancelled",
                );
            }
            // `RequestPermissionOutcome` is `#[non_exhaustive]`. If a
            // future protocol version adds a new variant, treat it
            // conservatively as "did not explicitly allow" rather
            // than letting the call through.
            Ok(_) => {
                return finish_failed(
                    ops,
                    session_id,
                    &tool_call_id,
                    &call.id,
                    "unknown permission outcome",
                );
            }
            Err(e) => {
                return finish_failed(
                    ops,
                    session_id,
                    &tool_call_id,
                    &call.id,
                    &format!("permission request failed: {e:#}"),
                );
            }
        }
    }

    // ── In-progress update ───────────────────────────────────────────
    ops.send_session_update(
        session_id,
        SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
            tool_call_id.clone(),
            ToolCallUpdateFields::new().status(ToolCallStatus::InProgress),
        )),
    );

    // ── Execute ──────────────────────────────────────────────────────
    let outcome: Result<(String, Vec<ToolCallContent>), String> = match call.name.as_str() {
        READ_FILE => exec_read_file(ops, session_id, &args_value).await,
        WRITE_FILE => exec_write_file(ops, session_id, &args_value).await,
        EDIT_FILE => exec_edit_file(ops, session_id, &args_value).await,
        LIST_DIR => exec_list_dir(&args_value),
        BASH => exec_bash(ops, session_id, session_cwd, &args_value, cancel).await,
        other => Err(format!("unknown tool '{other}'")),
    };

    match outcome {
        Ok((result_text, content)) => {
            // Log a snippet of what we'll feed back to the model.
            // This is the single most useful log line for "the model
            // says the tool failed but I gave it a result" debugging.
            let snippet: String = result_text.chars().take(200).collect();
            tracing::debug!(
                tool = %call.name,
                result_bytes = result_text.len(),
                snippet = %snippet,
                "tool completed; folding result back into history"
            );
            ops.send_session_update(
                session_id,
                SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                    tool_call_id.clone(),
                    ToolCallUpdateFields::new()
                        .status(ToolCallStatus::Completed)
                        .content(content),
                )),
            );
            ToolResult {
                tool_call_id: call.id,
                content: result_text,
                is_error: false,
            }
        }
        Err(msg) => {
            tracing::debug!(tool = %call.name, error = %msg, "tool failed");
            finish_failed(ops, session_id, &tool_call_id, &call.id, &msg)
        }
    }
}

fn finish_failed(
    ops: &dyn ClientOps,
    session_id: &SessionId,
    tool_call_id: &ToolCallId,
    raw_id: &str,
    message: &str,
) -> ToolResult {
    ops.send_session_update(
        session_id,
        SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
            tool_call_id.clone(),
            ToolCallUpdateFields::new()
                .status(ToolCallStatus::Failed)
                .content(vec![ToolCallContent::Content(
                    agent_client_protocol::schema::Content::new(ContentBlock::Text(
                        TextContent::new(message.to_string()),
                    )),
                )]),
        )),
    );
    ToolResult {
        tool_call_id: raw_id.to_string(),
        content: format!("ERROR: {message}"),
        is_error: true,
    }
}

// ── Per-tool executors ──────────────────────────────────────────────

async fn exec_read_file(
    ops: &dyn ClientOps,
    session_id: &SessionId,
    args_value: &serde_json::Value,
) -> Result<(String, Vec<ToolCallContent>), String> {
    let args: ReadFileArgs =
        serde_json::from_value(args_value.clone()).map_err(|e| format!("read_file: {e}"))?;
    let path = expand_path(&args.path);

    // Try the editor's filesystem first. Zed will show the open
    // buffer (if any) and respect any per-workspace mount points
    // / overlays it has configured.
    let acp_result = ops
        .read_text_file(session_id, path.clone(), args.line, args.limit)
        .await;
    let content = match acp_result {
        Ok(c) => c,
        Err(e) => {
            // ACP failures on read are almost always Zed's
            // workspace-boundary check (read paths outside the
            // session cwd are refused). Fall back to local
            // std::fs so the agent can still pull in shared
            // material like `~/git/architecture/generic.md` that
            // sits outside the active project. The user-process
            // file permissions still apply — this is not a
            // sandbox escape, just a way around Zed's
            // workspace-only default.
            tracing::warn!(
                path = %path.display(),
                error = %format!("{e:#}"),
                "fs/read_text_file failed; falling back to local std::fs"
            );
            let raw = std::fs::read_to_string(&path).map_err(|fs_err| {
                format!("read_file: ACP returned {e:#}; local fallback also failed: {fs_err}")
            })?;
            apply_line_limit(&raw, args.line, args.limit)
        }
    };

    let blocks = vec![ToolCallContent::Content(
        agent_client_protocol::schema::Content::new(ContentBlock::Text(TextContent::new(
            content.clone(),
        ))),
    )];
    Ok((content, blocks))
}

/// Slice a file's contents the same way ACP's `fs/read_text_file`
/// does (1-based line, optional line count). Used by the local-fs
/// fallback in `exec_read_file` so out-of-workspace reads honour
/// the same `line`/`limit` args the model passed.
fn apply_line_limit(content: &str, line: Option<u32>, limit: Option<u32>) -> String {
    if line.is_none() && limit.is_none() {
        return content.to_string();
    }
    let start = line.unwrap_or(1).max(1) as usize - 1;
    let count = limit.map(|l| l as usize).unwrap_or(usize::MAX);
    content
        .lines()
        .skip(start)
        .take(count)
        .collect::<Vec<_>>()
        .join("\n")
}

async fn exec_write_file(
    ops: &dyn ClientOps,
    session_id: &SessionId,
    args_value: &serde_json::Value,
) -> Result<(String, Vec<ToolCallContent>), String> {
    let args: WriteFileArgs =
        serde_json::from_value(args_value.clone()).map_err(|e| format!("write_file: {e}"))?;
    let path = expand_path(&args.path);
    // Best-effort read of the existing file so Zed can render a diff.
    // Failure here just means we render the write as an additive diff
    // — not a fatal error, the actual write below still runs.
    let old_text = ops
        .read_text_file(session_id, path.clone(), None, None)
        .await
        .ok();
    ops.write_text_file(session_id, path.clone(), args.content.clone())
        .await
        .map_err(|e| format!("write_file: {e:#}"))?;
    let mut diff = Diff::new(path.clone(), args.content.clone());
    if let Some(old) = old_text {
        diff = diff.old_text(old);
    }
    let summary = format!("wrote {} ({} bytes)", path.display(), args.content.len());
    Ok((summary, vec![ToolCallContent::Diff(diff)]))
}

async fn exec_edit_file(
    ops: &dyn ClientOps,
    session_id: &SessionId,
    args_value: &serde_json::Value,
) -> Result<(String, Vec<ToolCallContent>), String> {
    let args: EditFileArgs =
        serde_json::from_value(args_value.clone()).map_err(|e| format!("edit_file: {e}"))?;
    let path = expand_path(&args.path);
    let original = ops
        .read_text_file(session_id, path.clone(), None, None)
        .await
        .map_err(|e| format!("edit_file: read {}: {e:#}", path.display()))?;
    let occurrences = original.matches(args.old_text.as_str()).count();
    if occurrences == 0 {
        return Err(format!(
            "edit_file: old_text not found in {}",
            path.display()
        ));
    }
    if occurrences > 1 {
        return Err(format!(
            "edit_file: old_text appears {occurrences} times in {} — make it unique",
            path.display()
        ));
    }
    let new_content = original.replacen(args.old_text.as_str(), args.new_text.as_str(), 1);
    ops.write_text_file(session_id, path.clone(), new_content.clone())
        .await
        .map_err(|e| format!("edit_file: write {}: {e:#}", path.display()))?;
    let diff = Diff::new(path.clone(), new_content.clone()).old_text(original);
    let summary = format!("edited {} ({} bytes)", path.display(), new_content.len());
    Ok((summary, vec![ToolCallContent::Diff(diff)]))
}

fn exec_list_dir(args_value: &serde_json::Value) -> Result<(String, Vec<ToolCallContent>), String> {
    let args: ListDirArgs =
        serde_json::from_value(args_value.clone()).map_err(|e| format!("list_dir: {e}"))?;
    let path = expand_path(&args.path);
    let entries =
        std::fs::read_dir(&path).map_err(|e| format!("list_dir: read {}: {e}", path.display()))?;
    let mut lines: Vec<String> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let kind = match entry.file_type() {
            Ok(t) if t.is_dir() => 'd',
            Ok(t) if t.is_symlink() => 'l',
            Ok(_) => 'f',
            Err(_) => '?',
        };
        lines.push(format!("{kind} {name}"));
    }
    lines.sort();
    let body = lines.join("\n");
    let blocks = vec![ToolCallContent::Content(
        agent_client_protocol::schema::Content::new(ContentBlock::Text(TextContent::new(
            body.clone(),
        ))),
    )];
    Ok((body, blocks))
}

async fn exec_bash(
    ops: &dyn ClientOps,
    session_id: &SessionId,
    session_cwd: &Path,
    args_value: &serde_json::Value,
    cancel: &CancellationToken,
) -> Result<(String, Vec<ToolCallContent>), String> {
    let args: BashArgs =
        serde_json::from_value(args_value.clone()).map_err(|e| format!("bash: {e}"))?;
    // Expand the cwd if the model passed one; otherwise inherit the
    // session cwd verbatim. Don't expand the command string — sh
    // already handles `~` and `$HOME` inside the command line, and
    // pre-expanding would break the more interesting cases
    // (`echo ~`, `cd ~`, …).
    let cwd = match args.cwd {
        Some(c) => expand_path(&c),
        None => session_cwd.to_path_buf(),
    };

    tracing::debug!(
        command = %args.command,
        cwd = %cwd.display(),
        "bash: terminal/create"
    );

    let terminal = ops
        .create_terminal(
            session_id,
            "sh".to_string(),
            vec!["-c".to_string(), args.command.clone()],
            Some(cwd),
        )
        .await
        .map_err(|e| format!("bash: terminal/create: {e:#}"))?;

    // Wait for completion. If cancelled, ask the client to kill the
    // process. We still try to release the terminal afterwards.
    let exit = tokio::select! {
        biased;
        _ = cancel.cancelled() => {
            let _ = ops.kill_terminal(session_id, &terminal).await;
            let _ = ops.release_terminal(session_id, &terminal).await;
            return Err("bash: cancelled".to_string());
        }
        res = ops.wait_for_terminal_exit(session_id, &terminal) => {
            res.map_err(|e| format!("bash: terminal/wait_for_exit: {e:#}"))?
        }
    };
    tracing::debug!(
        terminal_id = %terminal.0,
        exit_code = ?exit.exit_code,
        signal = ?exit.signal,
        "bash: terminal exited"
    );

    let output_resp = ops
        .terminal_output(session_id, &terminal)
        .await
        .map_err(|e| format!("bash: terminal/output: {e:#}"))?;
    // Critical diagnostic: what did Zed actually buffer for us?
    // `len()` here is bytes including any control chars; the snippet
    // is the first 200 chars so we don't dump multi-megabyte file
    // contents into the journal at debug level.
    let snippet: String = output_resp.output.chars().take(200).collect();
    tracing::debug!(
        terminal_id = %terminal.0,
        output_bytes = output_resp.output.len(),
        truncated = output_resp.truncated,
        snippet = %snippet,
        "bash: terminal/output"
    );
    let _ = ops.release_terminal(session_id, &terminal).await;

    let summary = render_bash_result(&exit, &output_resp);
    let blocks = vec![ToolCallContent::Content(
        agent_client_protocol::schema::Content::new(ContentBlock::Text(TextContent::new(
            summary.clone(),
        ))),
    )];
    Ok((summary, blocks))
}

fn render_bash_result(exit: &TerminalExitStatus, output: &TerminalOutputResponse) -> String {
    let mut out = String::new();
    match (exit.exit_code, exit.signal.as_deref()) {
        (Some(0), _) => out.push_str("exit 0\n"),
        (Some(code), _) => out.push_str(&format!("exit {code}\n")),
        (None, Some(sig)) => out.push_str(&format!("terminated by signal {sig}\n")),
        (None, None) => out.push_str("exit ?\n"),
    }
    if output.truncated {
        out.push_str("(output truncated)\n");
    }
    out.push_str(&output.output);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Recording fake. Captures every outbound op so tests can
    /// assert on what the runner did.
    #[derive(Default)]
    struct FakeClient {
        events: Mutex<Vec<FakeEvent>>,
        /// Canned response for read_text_file.
        read_responses: Mutex<std::collections::HashMap<PathBuf, anyhow::Result<String>>>,
        /// Canned response for request_permission.
        permission: Mutex<Option<RequestPermissionOutcome>>,
    }

    // Fields are read only through the `{:?}` formatter in
    // `events()`; clippy's dead-code pass doesn't notice that, so
    // we suppress the warning at the enum level. The payloads stay
    // typed (vs. `String`-everything) so a test failure surfaces
    // useful detail in the Debug output.
    #[allow(dead_code)]
    #[derive(Debug)]
    enum FakeEvent {
        Read(PathBuf),
        Write(PathBuf, String),
        RequestPermission,
        CreateTerminal(String, Vec<String>),
        WaitForExit,
        TerminalOutput,
        KillTerminal,
        ReleaseTerminal,
        Update(String),
    }

    impl FakeClient {
        fn set_read(&self, path: PathBuf, body: anyhow::Result<String>) {
            self.read_responses.lock().unwrap().insert(path, body);
        }
        fn set_permission(&self, outcome: RequestPermissionOutcome) {
            *self.permission.lock().unwrap() = Some(outcome);
        }
        fn events(&self) -> Vec<String> {
            self.events
                .lock()
                .unwrap()
                .iter()
                .map(|e| format!("{e:?}"))
                .collect()
        }
    }

    #[async_trait]
    impl ClientOps for FakeClient {
        async fn read_text_file(
            &self,
            _session: &SessionId,
            path: PathBuf,
            _line: Option<u32>,
            _limit: Option<u32>,
        ) -> anyhow::Result<String> {
            self.events
                .lock()
                .unwrap()
                .push(FakeEvent::Read(path.clone()));
            self.read_responses
                .lock()
                .unwrap()
                .remove(&path)
                .unwrap_or_else(|| Err(anyhow::anyhow!("no canned read for {}", path.display())))
        }
        async fn write_text_file(
            &self,
            _session: &SessionId,
            path: PathBuf,
            content: String,
        ) -> anyhow::Result<()> {
            self.events
                .lock()
                .unwrap()
                .push(FakeEvent::Write(path, content));
            Ok(())
        }
        async fn request_permission(
            &self,
            _session: &SessionId,
            _tc: ToolCallUpdate,
            _options: Vec<PermissionOption>,
        ) -> anyhow::Result<RequestPermissionOutcome> {
            self.events
                .lock()
                .unwrap()
                .push(FakeEvent::RequestPermission);
            self.permission
                .lock()
                .unwrap()
                .clone()
                .ok_or_else(|| anyhow::anyhow!("no canned permission outcome"))
        }
        async fn create_terminal(
            &self,
            _session: &SessionId,
            command: String,
            args: Vec<String>,
            _cwd: Option<PathBuf>,
        ) -> anyhow::Result<TerminalId> {
            self.events
                .lock()
                .unwrap()
                .push(FakeEvent::CreateTerminal(command, args));
            Ok(TerminalId::new("t1"))
        }
        async fn wait_for_terminal_exit(
            &self,
            _session: &SessionId,
            _terminal: &TerminalId,
        ) -> anyhow::Result<TerminalExitStatus> {
            self.events.lock().unwrap().push(FakeEvent::WaitForExit);
            Ok(TerminalExitStatus::new().exit_code(0u32))
        }
        async fn terminal_output(
            &self,
            _session: &SessionId,
            _terminal: &TerminalId,
        ) -> anyhow::Result<TerminalOutputResponse> {
            self.events.lock().unwrap().push(FakeEvent::TerminalOutput);
            Ok(TerminalOutputResponse::new("ok\n", false))
        }
        async fn kill_terminal(
            &self,
            _session: &SessionId,
            _terminal: &TerminalId,
        ) -> anyhow::Result<()> {
            self.events.lock().unwrap().push(FakeEvent::KillTerminal);
            Ok(())
        }
        async fn release_terminal(
            &self,
            _session: &SessionId,
            _terminal: &TerminalId,
        ) -> anyhow::Result<()> {
            self.events.lock().unwrap().push(FakeEvent::ReleaseTerminal);
            Ok(())
        }
        fn send_session_update(&self, _session: &SessionId, update: SessionUpdate) {
            let tag = match update {
                SessionUpdate::ToolCall(_) => "tool_call".to_string(),
                SessionUpdate::ToolCallUpdate(u) => format!(
                    "tool_call_update:{:?}",
                    u.fields.status.unwrap_or(ToolCallStatus::Pending)
                ),
                _ => "other".to_string(),
            };
            self.events.lock().unwrap().push(FakeEvent::Update(tag));
        }
    }

    fn sid() -> SessionId {
        SessionId::new("s1")
    }
    fn mode_default() -> SessionModeId {
        SessionModeId::new(MODE_DEFAULT)
    }
    fn mode_bypass() -> SessionModeId {
        SessionModeId::new(MODE_BYPASS)
    }
    fn mode_plan() -> SessionModeId {
        SessionModeId::new(MODE_PLAN)
    }

    fn make_call(name: &str, args: serde_json::Value) -> ToolCallEvent {
        ToolCallEvent {
            id: "call_1".to_string(),
            name: name.to_string(),
            arguments: args.to_string(),
        }
    }

    #[tokio::test]
    async fn read_file_is_not_gated_in_default_mode() {
        let fake = FakeClient::default();
        fake.set_read(PathBuf::from("/tmp/x"), Ok("hello".to_string()));
        let res = dispatch_tool_call(
            &fake,
            &sid(),
            &mode_default(),
            Path::new("/tmp"),
            make_call(READ_FILE, json!({"path": "/tmp/x"})),
            &CancellationToken::new(),
        )
        .await;
        assert!(!res.is_error, "result: {}", res.content);
        assert_eq!(res.content, "hello");
        let events = fake.events();
        // Pending ToolCall → Read → InProgress update → Completed update
        assert!(!events.iter().any(|e| e == "RequestPermission"));
        assert!(events.iter().any(|e| e.starts_with("Read")));
    }

    #[tokio::test]
    async fn write_file_gated_in_default_mode_and_asks_permission() {
        let fake = FakeClient::default();
        fake.set_permission(RequestPermissionOutcome::Selected(
            agent_client_protocol::schema::SelectedPermissionOutcome::new("allow_once"),
        ));
        // The pre-write read fails; we tolerate that.
        let res = dispatch_tool_call(
            &fake,
            &sid(),
            &mode_default(),
            Path::new("/tmp"),
            make_call(WRITE_FILE, json!({"path": "/tmp/y", "content": "hi"})),
            &CancellationToken::new(),
        )
        .await;
        assert!(!res.is_error, "result: {}", res.content);
        let events = fake.events();
        assert!(events.iter().any(|e| e == "RequestPermission"));
        assert!(events.iter().any(|e| e.starts_with("Write")));
    }

    #[tokio::test]
    async fn bypass_mode_skips_permission_prompt() {
        let fake = FakeClient::default();
        let res = dispatch_tool_call(
            &fake,
            &sid(),
            &mode_bypass(),
            Path::new("/tmp"),
            make_call(WRITE_FILE, json!({"path": "/tmp/y", "content": "hi"})),
            &CancellationToken::new(),
        )
        .await;
        assert!(!res.is_error, "result: {}", res.content);
        let events = fake.events();
        assert!(
            !events.iter().any(|e| e == "RequestPermission"),
            "bypass mode must not prompt: {events:?}"
        );
        assert!(events.iter().any(|e| e.starts_with("Write")));
    }

    #[tokio::test]
    async fn rejected_permission_returns_error() {
        let fake = FakeClient::default();
        fake.set_permission(RequestPermissionOutcome::Selected(
            agent_client_protocol::schema::SelectedPermissionOutcome::new("reject_once"),
        ));
        let res = dispatch_tool_call(
            &fake,
            &sid(),
            &mode_default(),
            Path::new("/tmp"),
            make_call(BASH, json!({"command": "rm -rf /"})),
            &CancellationToken::new(),
        )
        .await;
        assert!(res.is_error, "expected error: {}", res.content);
        assert!(res.content.contains("reject"));
    }

    #[tokio::test]
    async fn bash_runs_through_terminal_lifecycle() {
        let fake = FakeClient::default();
        let res = dispatch_tool_call(
            &fake,
            &sid(),
            &mode_bypass(),
            Path::new("/tmp"),
            make_call(BASH, json!({"command": "echo ok"})),
            &CancellationToken::new(),
        )
        .await;
        assert!(!res.is_error, "result: {}", res.content);
        assert!(res.content.contains("exit 0"));
        assert!(res.content.contains("ok"));
        let events = fake.events();
        let sequence: Vec<&str> = events.iter().map(|s| s.as_str()).collect();
        // create → wait_for_exit → output → release
        let create = sequence
            .iter()
            .position(|e| e.starts_with("CreateTerminal"))
            .expect("CreateTerminal event");
        let wait = sequence
            .iter()
            .position(|e| e == &"WaitForExit")
            .expect("WaitForExit event");
        let out = sequence
            .iter()
            .position(|e| e == &"TerminalOutput")
            .expect("TerminalOutput event");
        let release = sequence
            .iter()
            .position(|e| e == &"ReleaseTerminal")
            .expect("ReleaseTerminal event");
        assert!(create < wait && wait < out && out < release);
    }

    #[tokio::test]
    async fn edit_file_rejects_ambiguous_match() {
        let fake = FakeClient::default();
        fake.set_read(PathBuf::from("/tmp/dup"), Ok("foo bar foo".to_string()));
        let res = dispatch_tool_call(
            &fake,
            &sid(),
            &mode_bypass(),
            Path::new("/tmp"),
            make_call(
                EDIT_FILE,
                json!({"path": "/tmp/dup", "old_text": "foo", "new_text": "baz"}),
            ),
            &CancellationToken::new(),
        )
        .await;
        assert!(res.is_error, "expected error, got {}", res.content);
        assert!(res.content.contains("2 times") || res.content.contains("appears"));
    }

    #[test]
    fn gated_set_matches_spec() {
        assert!(!is_gated(READ_FILE));
        assert!(!is_gated(LIST_DIR));
        assert!(is_gated(WRITE_FILE));
        assert!(is_gated(EDIT_FILE));
        assert!(is_gated(BASH));
    }

    // ── Plan-mode gating ────────────────────────────────────────────

    #[tokio::test]
    async fn plan_mode_refuses_bash() {
        let fake = FakeClient::default();
        let res = dispatch_tool_call(
            &fake,
            &sid(),
            &mode_plan(),
            Path::new("/tmp"),
            make_call(BASH, json!({"command": "ls"})),
            &CancellationToken::new(),
        )
        .await;
        assert!(res.is_error, "expected error: {}", res.content);
        assert!(
            res.content.contains("plan mode"),
            "expected plan-mode error, got: {}",
            res.content
        );
        let events = fake.events();
        assert!(
            !events.iter().any(|e| e.starts_with("CreateTerminal")),
            "bash must not run in plan mode: {events:?}"
        );
        assert!(
            !events.iter().any(|e| e == "RequestPermission"),
            "plan mode must not prompt for bash: {events:?}"
        );
    }

    #[tokio::test]
    async fn plan_mode_refuses_write_outside_plan_dir() {
        let fake = FakeClient::default();
        let res = dispatch_tool_call(
            &fake,
            &sid(),
            &mode_plan(),
            Path::new("/home/me/proj"),
            make_call(
                WRITE_FILE,
                json!({"path": "/home/me/proj/src/main.rs", "content": "fn main() {}"}),
            ),
            &CancellationToken::new(),
        )
        .await;
        assert!(res.is_error, "expected error: {}", res.content);
        assert!(
            res.content.contains("plan mode") && res.content.contains("/home/me/proj/src/main.rs"),
            "expected refusal naming attempted path, got: {}",
            res.content
        );
        let events = fake.events();
        assert!(
            !events.iter().any(|e| e.starts_with("Write")),
            "no write must happen for refused path: {events:?}"
        );
    }

    #[tokio::test]
    async fn plan_mode_allows_write_inside_plan_dir_without_permission() {
        // Skip if we can't resolve a plan dir in this environment
        // (would happen with no HOME / XDG_DATA_HOME — neither
        // realistic in CI nor for an interactive run).
        let Some(plan_dir) = store::plan_dir_for(Path::new("/home/me/proj")) else {
            eprintln!("skipping: plan_dir unresolvable in this env");
            return;
        };
        let target = plan_dir.join("01-overview.md");

        let fake = FakeClient::default();
        let res = dispatch_tool_call(
            &fake,
            &sid(),
            &mode_plan(),
            Path::new("/home/me/proj"),
            make_call(
                WRITE_FILE,
                json!({"path": target.to_str().unwrap(), "content": "# Overview"}),
            ),
            &CancellationToken::new(),
        )
        .await;
        assert!(
            !res.is_error,
            "expected success writing inside plan dir, got: {}",
            res.content
        );
        let events = fake.events();
        assert!(
            !events.iter().any(|e| e == "RequestPermission"),
            "plan mode must not prompt for in-plan-dir writes: {events:?}"
        );
        assert!(
            events.iter().any(|e| e.starts_with("Write")),
            "expected write to land: {events:?}"
        );
    }

    #[tokio::test]
    async fn plan_mode_allows_read_anywhere() {
        let fake = FakeClient::default();
        fake.set_read(PathBuf::from("/etc/hostname"), Ok("host".into()));
        let res = dispatch_tool_call(
            &fake,
            &sid(),
            &mode_plan(),
            Path::new("/home/me/proj"),
            make_call(READ_FILE, json!({"path": "/etc/hostname"})),
            &CancellationToken::new(),
        )
        .await;
        assert!(!res.is_error, "result: {}", res.content);
        assert_eq!(res.content, "host");
        let events = fake.events();
        assert!(
            !events.iter().any(|e| e == "RequestPermission"),
            "reads in plan mode must not prompt: {events:?}"
        );
    }

    // ── Path expansion + local read fallback ────────────────────────

    #[tokio::test]
    // We must hold the env-mutation lock across the await — releasing
    // it would let another test mutate HOME mid-dispatch and lose
    // the very thing we're testing for. The clippy lint is the
    // correct *default*; this is the documented exception.
    #[allow(clippy::await_holding_lock)]
    async fn read_file_expands_tilde_before_dispatch() {
        // HOME mutation is process-global; serialise tests that
        // touch it under a single std::sync::Mutex.
        use std::sync::Mutex;
        static LOCK: Mutex<()> = Mutex::new(());
        let _g = LOCK.lock().unwrap();
        let prior = std::env::var("HOME").ok();
        unsafe {
            std::env::set_var("HOME", "/home/me");
        }

        let fake = FakeClient::default();
        // The fake's canned-read map is keyed on the expanded path,
        // not the literal `~/...` — if expansion didn't happen the
        // lookup would miss and ACP would error → fallback to
        // local-fs (which also misses → final error). So a success
        // path here proves expansion ran before dispatch.
        fake.set_read(PathBuf::from("/home/me/notes.md"), Ok("body".into()));
        let res = dispatch_tool_call(
            &fake,
            &sid(),
            &mode_default(),
            Path::new("/tmp"),
            make_call(READ_FILE, json!({"path": "~/notes.md"})),
            &CancellationToken::new(),
        )
        .await;

        unsafe {
            match prior {
                Some(p) => std::env::set_var("HOME", p),
                None => std::env::remove_var("HOME"),
            }
        }

        assert!(!res.is_error, "result: {}", res.content);
        assert_eq!(res.content, "body");
    }

    #[tokio::test]
    async fn read_file_falls_back_to_local_fs_when_acp_errors() {
        // ACP read errors → local std::fs reads succeed for a file
        // we control. Use a temp file under CARGO_TARGET_TMPDIR.
        let tmpdir = std::env::var("CARGO_TARGET_TMPDIR")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        std::fs::create_dir_all(&tmpdir).unwrap();
        let pid = std::process::id();
        let target = tmpdir.join(format!("helexa-acp-fallback-{pid}.txt"));
        std::fs::write(&target, "line 1\nline 2\nline 3\n").unwrap();

        let fake = FakeClient::default();
        // No canned read → ACP returns Err. Fallback path should
        // pick up the local file.
        let res = dispatch_tool_call(
            &fake,
            &sid(),
            &mode_default(),
            Path::new("/tmp"),
            make_call(READ_FILE, json!({"path": target.to_str().unwrap()})),
            &CancellationToken::new(),
        )
        .await;
        let _ = std::fs::remove_file(&target);
        assert!(!res.is_error, "expected fallback success: {}", res.content);
        assert!(
            res.content.contains("line 1") && res.content.contains("line 3"),
            "unexpected fallback content: {}",
            res.content
        );
    }

    #[tokio::test]
    async fn read_file_fallback_honours_line_and_limit() {
        let tmpdir = std::env::var("CARGO_TARGET_TMPDIR")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        std::fs::create_dir_all(&tmpdir).unwrap();
        let pid = std::process::id();
        let target = tmpdir.join(format!("helexa-acp-fallback-slice-{pid}.txt"));
        std::fs::write(&target, "a\nb\nc\nd\ne\n").unwrap();

        let fake = FakeClient::default();
        let res = dispatch_tool_call(
            &fake,
            &sid(),
            &mode_default(),
            Path::new("/tmp"),
            make_call(
                READ_FILE,
                json!({"path": target.to_str().unwrap(), "line": 2, "limit": 2}),
            ),
            &CancellationToken::new(),
        )
        .await;
        let _ = std::fs::remove_file(&target);
        assert!(!res.is_error, "result: {}", res.content);
        assert_eq!(res.content, "b\nc");
    }

    #[tokio::test]
    async fn read_file_fallback_failure_surfaces_combined_error() {
        let fake = FakeClient::default();
        let res = dispatch_tool_call(
            &fake,
            &sid(),
            &mode_default(),
            Path::new("/tmp"),
            make_call(
                READ_FILE,
                json!({"path": "/definitely/not/a/real/path/xyz"}),
            ),
            &CancellationToken::new(),
        )
        .await;
        assert!(res.is_error, "expected error: {}", res.content);
        // The error message should cite BOTH the ACP failure and
        // the local-fs failure so the model knows what happened.
        assert!(
            res.content.contains("local fallback also failed"),
            "expected combined error message, got: {}",
            res.content
        );
    }

    #[test]
    fn apply_line_limit_basic_slice() {
        let body = "a\nb\nc\nd\ne";
        assert_eq!(apply_line_limit(body, None, None), body);
        assert_eq!(apply_line_limit(body, Some(1), Some(2)), "a\nb");
        assert_eq!(apply_line_limit(body, Some(3), None), "c\nd\ne");
        assert_eq!(apply_line_limit(body, None, Some(2)), "a\nb");
    }
}
