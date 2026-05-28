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

use anyhow::Context;
use std::path::Path;

use crate::provider::ToolSpec;
use crate::qwen3;

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
pub fn build_system_prompt(
    cwd: &Path,
    override_path: Option<&Path>,
    tools: &[ToolSpec],
) -> anyhow::Result<String> {
    let template = match override_path {
        Some(path) => std::fs::read_to_string(path)
            .with_context(|| format!("read system prompt from {}", path.display()))?,
        None => DEFAULT_PROMPT.to_string(),
    };
    let mut prompt = template.replace("{cwd}", &cwd.display().to_string());
    prompt.push_str(&qwen3::render_tool_block(tools));
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn default_prompt_substitutes_cwd() {
        let prompt = build_system_prompt(Path::new("/home/me/proj"), None, &[]).unwrap();
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
    }

    #[test]
    fn tools_are_appended_in_hermes_format() {
        let spec = ToolSpec {
            name: "read_file".into(),
            description: "Read a file.".into(),
            parameters: serde_json::json!({"type":"object","properties":{}, "required":[]}),
        };
        let prompt = build_system_prompt(Path::new("/x"), None, &[spec]).unwrap();
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

        let prompt = build_system_prompt(Path::new("/etc"), Some(path.as_path()), &[])
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
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("read system prompt"));
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
