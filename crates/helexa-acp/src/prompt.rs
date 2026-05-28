//! System prompt assembly.
//!
//! The built-in prompt tells the model the working directory and
//! enumerates the tools it actually has — without this, models trained
//! to "be safe when you don't know your environment" tend to refuse
//! tool use and ask the user to paste content instead. Override with
//! `HELEXA_ACP_SYSTEM_PROMPT_PATH` (env) or `system_prompt_path`
//! (TOML); the literal token `{cwd}` in a user-supplied file is
//! substituted with the session's working directory.

use anyhow::Context;
use std::path::Path;

const DEFAULT_PROMPT: &str = "\
You are helexa-acp, a coding assistant working inside an editor.

Working directory: {cwd}

You have the following tools. Call them whenever the user's request
involves looking at or modifying files, or running commands — do not
ask the user to paste file contents you could read yourself.

- read_file(path, line?, limit?) — Read a text file's contents.
- write_file(path, content) — Create or overwrite a file.
- edit_file(path, old_text, new_text) — Replace one unique substring
  in a file. Fails if old_text is not unique; call multiple times for
  multiple edits.
- list_dir(path) — List a directory's entries.
- bash(command, cwd?) — Run a shell command via `sh -c`. Returns
  combined stdout+stderr and the exit status.

All file paths must be absolute. Writes and shell commands may
prompt the user for permission depending on the session mode.

Be concise; the user is reading your output in an editor pane.";

/// Build the system prompt for a session.
///
/// `cwd` is the session's working directory (substituted for `{cwd}`
/// in both the default prompt and any user-supplied template).
/// `override_path` is the user's `system_prompt_path` (TOML) or
/// `HELEXA_ACP_SYSTEM_PROMPT_PATH` (env) value, already resolved by
/// [`crate::config::Config`].
pub fn build_system_prompt(cwd: &Path, override_path: Option<&Path>) -> anyhow::Result<String> {
    let template = match override_path {
        Some(path) => std::fs::read_to_string(path)
            .with_context(|| format!("read system prompt from {}", path.display()))?,
        None => DEFAULT_PROMPT.to_string(),
    };
    Ok(template.replace("{cwd}", &cwd.display().to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn default_prompt_substitutes_cwd() {
        let prompt = build_system_prompt(Path::new("/home/me/proj"), None).unwrap();
        assert!(
            prompt.contains("/home/me/proj"),
            "cwd not interpolated: {prompt}"
        );
        assert!(prompt.contains("helexa-acp"));
        assert!(
            !prompt.contains("{cwd}"),
            "left-over placeholder in default prompt"
        );
    }

    #[test]
    fn override_path_is_read_and_templated() {
        let mut tmp = tempfile_in_target("prompt.txt");
        tmp.write_all(b"custom prompt for {cwd} only").unwrap();
        tmp.flush().unwrap();

        let path = tmp.path().to_path_buf();
        drop(tmp);

        let prompt =
            build_system_prompt(Path::new("/etc"), Some(path.as_path())).expect("read override");
        assert_eq!(prompt, "custom prompt for /etc only");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_override_path_errors() {
        let err = build_system_prompt(
            Path::new("/tmp"),
            Some(Path::new("/definitely/not/a/real/path")),
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
