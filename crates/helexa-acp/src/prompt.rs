//! System prompt assembly.
//!
//! The system message has two parts:
//!
//! 1. A short human-readable preamble (working directory, style
//!    instructions). Either the built-in [`DEFAULT_PROMPT`] or a
//!    user-supplied file at `HELEXA_ACP_SYSTEM_PROMPT_PATH` /
//!    `system_prompt_path`. `{cwd}` is substituted in both.
//! 2. A `# Tools` block in Qwen3 Hermes format (see [`crate::qwen3`])
//!    describing the available functions. This is what makes the
//!    model actually call them — neuron/cortex don't honour the
//!    OpenAI `tools` API field, so the tool list has to live in the
//!    prompt itself.

use agent_client_protocol::schema::SessionModeId;
use anyhow::Context;
use std::path::Path;

use crate::provider::ToolSpec;
use crate::qwen3;
use crate::session::MODE_PLAN;

const DEFAULT_PROMPT: &str = "\
You are helexa-acp, a coding assistant working inside an editor.

Working directory: {cwd}

Use the tools described below whenever the user's request involves
looking at or modifying files, or running commands. Do not ask the
user to paste file contents you could read yourself. All file paths
must be absolute. Writes and shell commands may prompt the user for
permission depending on the session mode.

Be concise; the user is reading your output in an editor pane.";

/// Build the system prompt for a session.
///
/// - `cwd`: session working directory (substituted for `{cwd}` in
///   the preamble — both the default and any user-supplied template).
/// - `override_path`: path to a user-supplied template, already
///   resolved by [`crate::config::Config`]. The `# Tools` block is
///   appended *after* the user's template so a custom preamble
///   still gets the tool descriptions the model needs.
/// - `tools`: the tools to advertise. Empty list → no `# Tools`
///   block is appended at all.
/// - `mode`: current session mode. When the mode is [`MODE_PLAN`]
///   a plan-mode addendum describing the restrictions and the
///   completion menu is appended *after* the `# Tools` block so it
///   is the last thing the model reads before user input.
/// - `plan_dir`: resolved plan directory for the cwd. Only consulted
///   when `mode == MODE_PLAN`. `None` means the plan directory could
///   not be resolved (no `HOME` / `XDG_DATA_HOME`) — the addendum
///   still renders but with a placeholder so the model knows to
///   surface the error to the user rather than guess a path.
pub fn build_system_prompt(
    cwd: &Path,
    override_path: Option<&Path>,
    tools: &[ToolSpec],
    mode: &SessionModeId,
    plan_dir: Option<&Path>,
) -> anyhow::Result<String> {
    let template = match override_path {
        Some(path) => std::fs::read_to_string(path)
            .with_context(|| format!("read system prompt from {}", path.display()))?,
        None => DEFAULT_PROMPT.to_string(),
    };
    let mut prompt = template.replace("{cwd}", &cwd.display().to_string());
    prompt.push_str(&qwen3::render_tool_block(tools));
    if mode.0.as_ref() == MODE_PLAN {
        prompt.push_str(&render_plan_mode_block(plan_dir));
    }
    Ok(prompt)
}

/// Plan-mode instruction block. Tells the model:
///
/// 1. Where it may write — only inside `plan_dir`.
/// 2. What it may *not* do — bash is disabled; writes outside
///    `plan_dir` are refused by the runtime.
/// 3. How to finish — emit the 3-option menu so the user can
///    switch modes and either kick off implementation (with or
///    without permission prompts) or keep iterating on the plan.
fn render_plan_mode_block(plan_dir: Option<&Path>) -> String {
    let plan_path = plan_dir
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "<plan directory could not be resolved — tell the user>".to_string());
    format!(
        "\n\n# Plan mode\n\
         \n\
         You are in **plan mode**. Your task is to draft a written\n\
         implementation plan for the user; you must NOT modify any\n\
         project files or run shell commands.\n\
         \n\
         Rules in plan mode:\n\
         \n\
         - `read_file` and `list_dir` are unrestricted — use them to\n\
           explore the codebase as needed.\n\
         - `write_file` and `edit_file` are allowed ONLY under the\n\
           plan directory: `{plan_path}`. The runtime will refuse any\n\
           write outside it.\n\
         - `bash` is disabled. Do not call it.\n\
         \n\
         Write the plan as one or more Markdown files under\n\
         `{plan_path}`. Use descriptive filenames\n\
         (`01-overview.md`, `02-data-model.md`, etc.). It is fine to\n\
         iterate — overwrite the file when you refine a section.\n\
         \n\
         When the plan is complete, do NOT begin implementation.\n\
         Instead, end your turn with this menu, verbatim, so the\n\
         user can choose how to proceed:\n\
         \n\
         ---\n\
         **Plan complete.** To proceed, switch the session mode in\n\
         the agent dropdown and send a follow-up message:\n\
         \n\
         1. **Bypass Permissions** — implement the plan now, skipping\n\
            per-tool permission prompts.\n\
         2. **Default** — implement the plan now, prompting before\n\
            each write or shell command.\n\
         3. **Plan** (stay here) — refine the plan; reply with the\n\
            change you want and I will revise it.\n\
         ---\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{MODE_DEFAULT, MODE_PLAN};
    use std::io::Write;

    fn default_mode() -> SessionModeId {
        SessionModeId::new(MODE_DEFAULT)
    }
    fn plan_mode() -> SessionModeId {
        SessionModeId::new(MODE_PLAN)
    }

    #[test]
    fn default_prompt_substitutes_cwd() {
        let prompt =
            build_system_prompt(Path::new("/home/me/proj"), None, &[], &default_mode(), None)
                .unwrap();
        assert!(
            prompt.contains("/home/me/proj"),
            "cwd not interpolated: {prompt}"
        );
        assert!(prompt.contains("helexa-acp"));
        assert!(
            !prompt.contains("{cwd}"),
            "left-over placeholder in default prompt"
        );
        // With no tools, the # Tools block is absent.
        assert!(!prompt.contains("# Tools"));
        // Default mode does not get the plan-mode addendum.
        assert!(!prompt.contains("# Plan mode"));
    }

    #[test]
    fn tools_are_appended_in_hermes_format() {
        let spec = ToolSpec {
            name: "read_file".into(),
            description: "Read a file.".into(),
            parameters: serde_json::json!({"type":"object","properties":{}, "required":[]}),
        };
        let prompt =
            build_system_prompt(Path::new("/x"), None, &[spec], &default_mode(), None).unwrap();
        assert!(prompt.contains("# Tools"));
        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains("\"name\":\"read_file\""));
        assert!(prompt.contains("<tool_call>"));
    }

    #[test]
    fn override_path_is_read_and_templated() {
        let mut tmp = tempfile_in_target("prompt.txt");
        tmp.write_all(b"custom prompt for {cwd} only").unwrap();
        tmp.flush().unwrap();

        let path = tmp.path().to_path_buf();
        drop(tmp);

        let prompt = build_system_prompt(
            Path::new("/etc"),
            Some(path.as_path()),
            &[],
            &default_mode(),
            None,
        )
        .expect("read override");
        assert_eq!(prompt, "custom prompt for /etc only");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_override_path_errors() {
        let err = build_system_prompt(
            Path::new("/tmp"),
            Some(Path::new("/definitely/not/a/real/path")),
            &[],
            &default_mode(),
            None,
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("read system prompt"));
    }

    #[test]
    fn plan_mode_addendum_includes_plan_dir_and_menu() {
        let plan_dir = Path::new("/home/me/.local/share/helexa-acp/plans/proj-deadbeef");
        let prompt = build_system_prompt(
            Path::new("/home/me/proj"),
            None,
            &[],
            &plan_mode(),
            Some(plan_dir),
        )
        .unwrap();
        assert!(prompt.contains("# Plan mode"));
        assert!(
            prompt.contains(plan_dir.to_str().unwrap()),
            "plan dir not interpolated: {prompt}"
        );
        // The 3-option menu must be present so the model emits it verbatim.
        assert!(prompt.contains("Bypass Permissions"));
        assert!(prompt.contains("**Default**"));
        assert!(prompt.contains("3. **Plan**"));
        // Bash disabled instruction must be present.
        assert!(prompt.contains("`bash` is disabled"));
    }

    #[test]
    fn plan_mode_addendum_handles_unresolved_plan_dir() {
        let prompt =
            build_system_prompt(Path::new("/home/me/proj"), None, &[], &plan_mode(), None).unwrap();
        assert!(prompt.contains("# Plan mode"));
        assert!(prompt.contains("could not be resolved"));
    }

    /// Tiny temp-file helper that doesn't pull in the `tempfile` crate.
    /// Writes under `target/` so it's cleaned up by `cargo clean`.
    fn tempfile_in_target(name: &str) -> TempHandle {
        let base = std::env::var("CARGO_TARGET_TMPDIR")
            .ok()
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let _ = std::fs::create_dir_all(&base);
        let pid = std::process::id();
        let path = base.join(format!("helexa-acp-{pid}-{name}"));
        let file = std::fs::File::create(&path).expect("create temp file");
        TempHandle { file, path }
    }

    struct TempHandle {
        file: std::fs::File,
        path: std::path::PathBuf,
    }

    impl TempHandle {
        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Write for TempHandle {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.file.write(buf)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.file.flush()
        }
    }
}
