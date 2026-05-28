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

use crate::session::{MODE_BYPASS, MODE_DEFAULT};
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

    // ── Permission gate ──────────────────────────────────────────────
    if is_gated(&call.name) && mode.0.as_ref() != MODE_BYPASS {
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
    let content = ops
        .read_text_file(session_id, args.path, args.line, args.limit)
        .await
        .map_err(|e| format!("read_file: {e:#}"))?;
    let blocks = vec![ToolCallContent::Content(
        agent_client_protocol::schema::Content::new(ContentBlock::Text(TextContent::new(
            content.clone(),
        ))),
    )];
    Ok((content, blocks))
}

async fn exec_write_file(
    ops: &dyn ClientOps,
    session_id: &SessionId,
    args_value: &serde_json::Value,
) -> Result<(String, Vec<ToolCallContent>), String> {
    let args: WriteFileArgs =
        serde_json::from_value(args_value.clone()).map_err(|e| format!("write_file: {e}"))?;
    // Best-effort read of the existing file so Zed can render a diff.
    // Failure here just means we render the write as an additive diff
    // — not a fatal error, the actual write below still runs.
    let old_text = ops
        .read_text_file(session_id, args.path.clone(), None, None)
        .await
        .ok();
    ops.write_text_file(session_id, args.path.clone(), args.content.clone())
        .await
        .map_err(|e| format!("write_file: {e:#}"))?;
    let mut diff = Diff::new(args.path.clone(), args.content.clone());
    if let Some(old) = old_text {
        diff = diff.old_text(old);
    }
    let summary = format!(
        "wrote {} ({} bytes)",
        args.path.display(),
        args.content.len()
    );
    Ok((summary, vec![ToolCallContent::Diff(diff)]))
}

async fn exec_edit_file(
    ops: &dyn ClientOps,
    session_id: &SessionId,
    args_value: &serde_json::Value,
) -> Result<(String, Vec<ToolCallContent>), String> {
    let args: EditFileArgs =
        serde_json::from_value(args_value.clone()).map_err(|e| format!("edit_file: {e}"))?;
    let original = ops
        .read_text_file(session_id, args.path.clone(), None, None)
        .await
        .map_err(|e| format!("edit_file: read {}: {e:#}", args.path.display()))?;
    let occurrences = original.matches(args.old_text.as_str()).count();
    if occurrences == 0 {
        return Err(format!(
            "edit_file: old_text not found in {}",
            args.path.display()
        ));
    }
    if occurrences > 1 {
        return Err(format!(
            "edit_file: old_text appears {occurrences} times in {} — make it unique",
            args.path.display()
        ));
    }
    let new_content = original.replacen(args.old_text.as_str(), args.new_text.as_str(), 1);
    ops.write_text_file(session_id, args.path.clone(), new_content.clone())
        .await
        .map_err(|e| format!("edit_file: write {}: {e:#}", args.path.display()))?;
    let diff = Diff::new(args.path.clone(), new_content.clone()).old_text(original);
    let summary = format!(
        "edited {} ({} bytes)",
        args.path.display(),
        new_content.len()
    );
    Ok((summary, vec![ToolCallContent::Diff(diff)]))
}

fn exec_list_dir(args_value: &serde_json::Value) -> Result<(String, Vec<ToolCallContent>), String> {
    let args: ListDirArgs =
        serde_json::from_value(args_value.clone()).map_err(|e| format!("list_dir: {e}"))?;
    let entries = std::fs::read_dir(&args.path)
        .map_err(|e| format!("list_dir: read {}: {e}", args.path.display()))?;
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
    let cwd = args.cwd.unwrap_or_else(|| session_cwd.to_path_buf());

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
}
