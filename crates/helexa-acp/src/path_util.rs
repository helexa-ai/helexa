//! Path expansion shared across every tool that takes a path.
//!
//! Models often emit shell-style paths like `~/git/repo/file.rs` or
//! `$HOME/notes.md`. ACP's `fs/read_text_file` and friends — and our
//! own local `std::fs` reads — both want a real absolute path; the
//! `~` / `$HOME` forms reach them as literal strings and the open
//! fails. The tool schemas already document "absolute path" but in
//! practice the model slips up often enough that handling it
//! server-side is the difference between "works" and "the agent is
//! brittle".
//!
//! Scope is deliberately small:
//!
//! - `~` and `~/` (current user only — `~user` lookups would require
//!   pulling in passwd parsing).
//! - `$HOME` and `$HOME/`.
//!
//! Any other shell variable (`$PWD`, `${HOME}`, …) passes through
//! unchanged. The shell already expands them inside `bash` tool
//! commands; for the file-tool argument fields, we deliberately
//! limit the set so the behaviour is predictable.
//!
//! Falls back to the input path verbatim when `HOME` is unset
//! (stripped-down container env). That preserves the "no surprise
//! mutations" rule — never invent a path the caller didn't ask for.

use std::path::{Path, PathBuf};

/// Process-global lock for tests that mutate `HOME`. Anyone in the
/// crate touching `HOME` must hold this for the duration of the
/// read-modify-restore window — otherwise concurrent `cargo test`
/// workers race and flake.
///
/// Only built into the test binaries. Production code never mutates
/// env vars.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Expand `~`, `~/`, `$HOME`, and `$HOME/` prefixes against the
/// current user's home directory. All other inputs pass through
/// unchanged.
///
/// Returns the input verbatim if `HOME` isn't set in the env.
pub fn expand_path(input: &Path) -> PathBuf {
    let Some(s) = input.to_str() else {
        return input.to_path_buf();
    };
    let Ok(home) = std::env::var("HOME") else {
        return input.to_path_buf();
    };
    let home = PathBuf::from(home);
    if s == "~" || s == "$HOME" {
        return home;
    }
    if let Some(rest) = s.strip_prefix("~/") {
        return home.join(rest);
    }
    if let Some(rest) = s.strip_prefix("$HOME/") {
        return home.join(rest);
    }
    input.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Set HOME for the duration of the test. Tests using this run
    /// serially under the crate-wide [`ENV_LOCK`] because env
    /// mutation isn't thread-safe — `cargo test` parallel workers
    /// would race without it.
    fn with_home<F: FnOnce()>(home: &str, body: F) {
        let _g = ENV_LOCK.lock().unwrap();
        let prior = std::env::var("HOME").ok();
        // SAFETY: tests touch process-global env. The mutex
        // serialises access; sub-threads in other test modules
        // touching HOME aren't expected (none in this crate).
        unsafe {
            std::env::set_var("HOME", home);
        }
        body();
        unsafe {
            match prior {
                Some(p) => std::env::set_var("HOME", p),
                None => std::env::remove_var("HOME"),
            }
        }
    }

    #[test]
    fn expands_tilde_slash() {
        with_home("/home/me", || {
            assert_eq!(
                expand_path(Path::new("~/git/repo/file.rs")),
                PathBuf::from("/home/me/git/repo/file.rs")
            );
        });
    }

    #[test]
    fn expands_bare_tilde() {
        with_home("/home/me", || {
            assert_eq!(expand_path(Path::new("~")), PathBuf::from("/home/me"));
        });
    }

    #[test]
    fn expands_dollar_home_slash() {
        with_home("/home/me", || {
            assert_eq!(
                expand_path(Path::new("$HOME/notes.md")),
                PathBuf::from("/home/me/notes.md")
            );
        });
    }

    #[test]
    fn expands_bare_dollar_home() {
        with_home("/home/me", || {
            assert_eq!(expand_path(Path::new("$HOME")), PathBuf::from("/home/me"));
        });
    }

    #[test]
    fn absolute_path_passes_through() {
        with_home("/home/me", || {
            assert_eq!(
                expand_path(Path::new("/etc/hostname")),
                PathBuf::from("/etc/hostname")
            );
        });
    }

    #[test]
    fn relative_path_passes_through() {
        with_home("/home/me", || {
            assert_eq!(
                expand_path(Path::new("src/main.rs")),
                PathBuf::from("src/main.rs")
            );
        });
    }

    #[test]
    fn tilde_user_form_not_expanded() {
        // ~other is shell sugar for /home/other and would require
        // passwd parsing to resolve. Out of scope — pass it
        // through and let the open fail with a clear error.
        with_home("/home/me", || {
            assert_eq!(
                expand_path(Path::new("~other/x")),
                PathBuf::from("~other/x")
            );
        });
    }

    #[test]
    fn no_home_env_passes_through() {
        // Share the same crate-wide lock as `with_home` — otherwise
        // a parallel test setting HOME races this clear-and-assert
        // window.
        let _g = ENV_LOCK.lock().unwrap();
        let prior = std::env::var("HOME").ok();
        // SAFETY: serialised by LOCK above.
        unsafe {
            std::env::remove_var("HOME");
        }
        assert_eq!(
            expand_path(Path::new("~/git/repo")),
            PathBuf::from("~/git/repo")
        );
        unsafe {
            if let Some(p) = prior {
                std::env::set_var("HOME", p);
            }
        }
    }

    #[test]
    fn dollar_other_var_not_expanded() {
        with_home("/home/me", || {
            assert_eq!(
                expand_path(Path::new("$PWD/file")),
                PathBuf::from("$PWD/file")
            );
            assert_eq!(
                expand_path(Path::new("${HOME}/file")),
                PathBuf::from("${HOME}/file")
            );
        });
    }
}
