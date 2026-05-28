//! System prompt assembly.
//!
//! Stage 2 ships a small built-in prompt aimed at coding assistance:
//! it tells the model the working directory and reminds it that no
//! tools are available yet. Users who want something different point
//! `HELEXA_ACP_SYSTEM_PROMPT_PATH` (env) or `system_prompt_path` (TOML)
//! at a file and we read that verbatim. The literal token `{cwd}` in
//! a user-supplied file is substituted with the session's working
//! directory so editor templates can include it without templating.

use anyhow::Context;
use std::path::Path;

const DEFAULT_PROMPT: &str = "\
You are helexa-acp, a coding assistant.

Working directory: {cwd}

Stage 2 build: you have no tools available — answer with text only.
When you need to refer to files or directories, describe paths
relative to the working directory above. Be concise; the user is
reading your output in an editor pane.";

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
