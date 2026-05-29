//! On-disk session persistence for `session/load` support.
//!
//! Storage layout:
//!
//! ```text
//! $XDG_DATA_HOME/helexa-acp/sessions/{session_id}.json
//! ```
//!
//! (Fallback to `~/.local/share/helexa-acp/sessions/` when
//! `$XDG_DATA_HOME` is unset.) One JSON file per session. Writes
//! happen at the end of every `session/prompt` round through
//! [`save`], using tempfile-plus-rename so a crash mid-write can't
//! corrupt the store. Reads happen on `session/load` via [`load`].
//!
//! No compaction, no rotation: files accumulate until the user
//! cleans them up. That's deliberate — disk is cheap, and the
//! resume-on-restart workflow matters more than tidiness. The
//! [`SESSIONS_DIRNAME`] subdirectory is created lazily on first
//! save so an unprivileged install path never errors at startup.

use std::path::PathBuf;
use std::time::SystemTime;

use agent_client_protocol::schema::SessionId;
use serde::{Deserialize, Serialize};

use crate::provider::Message;

const APP_DIRNAME: &str = "helexa-acp";
const SESSIONS_DIRNAME: &str = "sessions";
const PLANS_DIRNAME: &str = "plans";

/// The shape persisted to disk for one session. Only what we can't
/// rebuild from the running config goes in here: the conversation
/// history, the mode toggle, the model id, and the cwd-at-creation.
///
/// `created_at` / `updated_at` are seconds-since-epoch — cheap to
/// compare, no third-party time crate, and stable across runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedSession {
    pub session_id: String,
    pub cwd: PathBuf,
    pub model_id: String,
    pub mode_id: String,
    pub history: Vec<Message>,
    pub created_at: u64,
    pub updated_at: u64,
}

/// Resolve the directory that holds session JSON files. Honors
/// `$XDG_DATA_HOME`; falls back to `~/.local/share/helexa-acp/sessions/`.
/// Returns `None` if neither is resolvable (no `HOME` set — possible
/// in stripped-down container environments).
pub fn sessions_dir() -> Option<PathBuf> {
    let base = std::env::var("XDG_DATA_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".local").join("share"))
        })?;
    Some(base.join(APP_DIRNAME).join(SESSIONS_DIRNAME))
}

/// Atomic save into the default sessions directory.
pub fn save(session: &PersistedSession) -> anyhow::Result<()> {
    let dir = sessions_dir()
        .ok_or_else(|| anyhow::anyhow!("can't resolve XDG_DATA_HOME or HOME for session store"))?;
    save_to_dir(&dir, session)
}

/// Load from the default sessions directory.
pub fn load(session_id: &SessionId) -> anyhow::Result<PersistedSession> {
    let dir = sessions_dir()
        .ok_or_else(|| anyhow::anyhow!("can't resolve XDG_DATA_HOME or HOME for session store"))?;
    load_from_dir(&dir, session_id)
}

/// Atomic save into an explicit directory. Writes to
/// `{id}.json.tmp` then renames over `{id}.json`. Creates the
/// target directory if it doesn't exist. Split from [`save`] so
/// unit tests can target a per-test scratch dir without mutating
/// process-global env vars.
pub fn save_to_dir(dir: &std::path::Path, session: &PersistedSession) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir).map_err(|e| anyhow::anyhow!("create {}: {e}", dir.display()))?;
    let safe = sanitize_id(&session.session_id);
    let final_path = dir.join(format!("{safe}.json"));
    let tmp_path = dir.join(format!("{safe}.json.tmp"));
    let json = serde_json::to_string_pretty(session)?;
    std::fs::write(&tmp_path, json)
        .map_err(|e| anyhow::anyhow!("write {}: {e}", tmp_path.display()))?;
    std::fs::rename(&tmp_path, &final_path)
        .map_err(|e| anyhow::anyhow!("rename → {}: {e}", final_path.display()))?;
    Ok(())
}

/// Load from an explicit directory. Returns a friendly error
/// message when the session id has no file on disk so the caller
/// can map it to a clean ACP error response.
pub fn load_from_dir(
    dir: &std::path::Path,
    session_id: &SessionId,
) -> anyhow::Result<PersistedSession> {
    let safe = sanitize_id(session_id.0.as_ref());
    let path = dir.join(format!("{safe}.json"));
    let bytes = std::fs::read(&path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            anyhow::anyhow!("no persisted session at {}", path.display())
        } else {
            anyhow::anyhow!("read {}: {e}", path.display())
        }
    })?;
    let session: PersistedSession = serde_json::from_slice(&bytes)
        .map_err(|e| anyhow::anyhow!("parse {}: {e}", path.display()))?;
    Ok(session)
}

/// List all persisted sessions, optionally filtered by `cwd`. Used
/// by the `session/list` handler so a client (Zed) can find the
/// session that belongs to the workspace it's reopening.
///
/// `filter_cwd = None` returns every session on disk. `Some(path)`
/// returns only sessions whose persisted `cwd` is exactly equal.
///
/// Files that fail to parse are skipped with a warning rather than
/// aborting the whole list — one corrupt session shouldn't make
/// the resume picker unusable.
pub fn list(filter_cwd: Option<&std::path::Path>) -> anyhow::Result<Vec<PersistedSession>> {
    let dir = sessions_dir()
        .ok_or_else(|| anyhow::anyhow!("can't resolve XDG_DATA_HOME or HOME for session store"))?;
    list_in_dir(&dir, filter_cwd)
}

/// Explicit-dir variant for tests, mirroring [`save_to_dir`] /
/// [`load_from_dir`].
pub fn list_in_dir(
    dir: &std::path::Path,
    filter_cwd: Option<&std::path::Path>,
) -> anyhow::Result<Vec<PersistedSession>> {
    let read = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(anyhow::anyhow!("read_dir {}: {e}", dir.display())),
    };
    let mut out = Vec::new();
    for entry in read.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        match std::fs::read(&path).and_then(|bytes| {
            serde_json::from_slice::<PersistedSession>(&bytes).map_err(std::io::Error::other)
        }) {
            Ok(session) => {
                if let Some(want) = filter_cwd
                    && session.cwd != want
                {
                    continue;
                }
                out.push(session);
            }
            Err(e) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "store: skipping unparseable session file"
                );
            }
        }
    }
    // Most-recent first by updated_at.
    out.sort_by_key(|s| std::cmp::Reverse(s.updated_at));
    Ok(out)
}

/// Seconds-since-epoch, saturating to 0 if the system clock is
/// behind epoch (which shouldn't happen but the type system
/// requires a fallible read).
pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Root directory for plan-mode artefacts. Mirrors [`sessions_dir`]
/// but under `…/helexa-acp/plans/` so plans and conversation
/// transcripts are siblings, not nested.
pub fn plans_root() -> Option<PathBuf> {
    sessions_dir().and_then(|s| s.parent().map(|p| p.join(PLANS_DIRNAME)))
}

/// Per-project plan directory:
/// `$XDG_DATA_HOME/helexa-acp/plans/<project-id>/`. The id derives
/// from the session's cwd so plans for the same project survive
/// across cwd-changes (a `/home/foo/git/bar` ↔ symlinked
/// `/srv/checkout/bar` would technically diverge, accepted as a
/// won't-fix corner case).
pub fn plan_dir_for(cwd: &std::path::Path) -> Option<PathBuf> {
    plans_root().map(|root| root.join(project_id_for(cwd)))
}

/// Deterministic, human-readable project identifier. Format:
/// `<basename>-<8-hex>` where the 8-hex suffix is FNV-1a of the
/// full path. Basename keeps the path skim-readable when poking
/// around `$XDG_DATA_HOME` by hand; the hash suffix disambiguates
/// repos that share a final path component (e.g. multiple
/// `/.../checkout/beat` checkouts).
///
/// FNV-1a rather than `std::collections::hash::DefaultHasher`
/// because the latter (SipHash) reseeds per process, so it'd give
/// us a different project_id on every run.
pub fn project_id_for(cwd: &std::path::Path) -> String {
    let basename = cwd
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let sanitised: String = basename
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let hash = fnv1a_32(cwd.to_string_lossy().as_bytes());
    format!("{sanitised}-{hash:08x}")
}

/// FNV-1a (32-bit). Deterministic, no third-party crate. Used for
/// project ids only — not cryptographic.
fn fnv1a_32(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in bytes {
        h ^= u32::from(*b);
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// Format seconds-since-epoch as an ISO 8601 / RFC 3339 string
/// (`YYYY-MM-DDTHH:MM:SSZ`) for `SessionInfo.updated_at`. Returns
/// `None` for values outside the representable range, in which
/// case the caller should omit the field.
pub fn unix_to_iso8601(secs: u64) -> Option<String> {
    use chrono::TimeZone;
    let dt = chrono::Utc.timestamp_opt(secs as i64, 0).single()?;
    Some(dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
}

/// Strip anything that isn't a safe filename character so a
/// mischievous (or just unconventional) session id can't escape
/// the sessions directory.
fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{MessageContent, Role};

    /// Unique scratch dir per test invocation. We use this dir
    /// directly with the `*_to_dir` / `*_from_dir` functions so
    /// the tests never mutate `$XDG_DATA_HOME` — that env var
    /// would race across the parallel test harness.
    fn unique_dir() -> PathBuf {
        let base = std::env::var("CARGO_TARGET_TMPDIR")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let pid = std::process::id();
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let dir = base.join(format!("helexa-acp-store-test-{pid}-{nanos}"));
        std::fs::create_dir_all(&dir).expect("create test dir");
        dir
    }

    fn sample(id: &str) -> PersistedSession {
        PersistedSession {
            session_id: id.into(),
            cwd: PathBuf::from("/home/me/proj"),
            model_id: "Qwen/Qwen3.6-27B".into(),
            mode_id: "default".into(),
            history: vec![
                Message {
                    role: Role::User,
                    content: MessageContent::Text {
                        text: "hello".into(),
                    },
                },
                Message {
                    role: Role::Assistant,
                    content: MessageContent::Text { text: "hi".into() },
                },
            ],
            created_at: 1_700_000_000,
            updated_at: 1_700_000_001,
        }
    }

    #[test]
    fn round_trip_save_then_load() {
        let dir = unique_dir();
        save_to_dir(&dir, &sample("hxa-1")).expect("save");
        let loaded = load_from_dir(&dir, &SessionId::new("hxa-1")).expect("load");
        assert_eq!(loaded.session_id, "hxa-1");
        assert_eq!(loaded.cwd, PathBuf::from("/home/me/proj"));
        assert_eq!(loaded.history.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_missing_session_errors_with_not_found_message() {
        let dir = unique_dir();
        let err = load_from_dir(&dir, &SessionId::new("nope")).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("no persisted session"),
            "want NotFound, got: {msg}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_overwrites_existing_atomically() {
        let dir = unique_dir();
        save_to_dir(&dir, &sample("hxa-1")).expect("save");
        let mut updated = sample("hxa-1");
        updated.history.push(Message {
            role: Role::User,
            content: MessageContent::Text {
                text: "third turn".into(),
            },
        });
        updated.updated_at = 1_700_000_500;
        save_to_dir(&dir, &updated).expect("re-save");
        let loaded = load_from_dir(&dir, &SessionId::new("hxa-1")).expect("load");
        assert_eq!(loaded.history.len(), 3);
        assert_eq!(loaded.updated_at, 1_700_000_500);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_then_load_preserves_tool_calls_and_results() {
        use crate::provider::ToolCall;
        let dir = unique_dir();
        let mut session = sample("hxa-2");
        session.history.push(Message {
            role: Role::Assistant,
            content: MessageContent::ToolCalls {
                text: Some("calling".into()),
                calls: vec![ToolCall {
                    id: "call_0".into(),
                    name: "read_file".into(),
                    arguments: r#"{"path":"/etc/hostname"}"#.into(),
                }],
            },
        });
        session.history.push(Message {
            role: Role::Tool,
            content: MessageContent::ToolResult {
                tool_call_id: "call_0".into(),
                content: "host".into(),
            },
        });
        save_to_dir(&dir, &session).expect("save");
        let loaded = load_from_dir(&dir, &SessionId::new("hxa-2")).expect("load");
        assert_eq!(loaded.history.len(), 4);
        match &loaded.history[2].content {
            MessageContent::ToolCalls { calls, .. } => {
                assert_eq!(calls[0].name, "read_file");
            }
            other => panic!("expected ToolCalls, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_filters_by_cwd_and_sorts_recent_first() {
        let dir = unique_dir();
        let mut a = sample("a");
        a.cwd = PathBuf::from("/home/me/proj-x");
        a.updated_at = 1_700_000_010;
        let mut b = sample("b");
        b.cwd = PathBuf::from("/home/me/proj-x");
        b.updated_at = 1_700_000_020;
        let mut c = sample("c");
        c.cwd = PathBuf::from("/home/me/elsewhere");
        c.updated_at = 1_700_000_030;
        save_to_dir(&dir, &a).unwrap();
        save_to_dir(&dir, &b).unwrap();
        save_to_dir(&dir, &c).unwrap();

        let proj_x = PathBuf::from("/home/me/proj-x");
        let list = list_in_dir(&dir, Some(&proj_x)).unwrap();
        let ids: Vec<&str> = list.iter().map(|s| s.session_id.as_str()).collect();
        // Filtered to proj-x; b before a because b is more recent.
        assert_eq!(ids, vec!["b", "a"]);

        let all = list_in_dir(&dir, None).unwrap();
        assert_eq!(all.len(), 3);
        // Global list still sorted recent-first across all cwds.
        assert_eq!(all[0].session_id, "c");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_returns_empty_for_missing_dir() {
        let dir = unique_dir().join("does-not-exist");
        let list = list_in_dir(&dir, None).unwrap();
        assert!(list.is_empty());
    }

    #[test]
    fn list_skips_unparseable_files() {
        let dir = unique_dir();
        save_to_dir(&dir, &sample("good")).unwrap();
        std::fs::write(dir.join("garbage.json"), b"{not valid json").unwrap();
        let list = list_in_dir(&dir, None).unwrap();
        // Garbage skipped; good survives.
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].session_id, "good");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn iso8601_formats_unix_seconds() {
        // 2024-01-01T00:00:00Z is 1704067200 unix seconds.
        assert_eq!(
            unix_to_iso8601(1_704_067_200),
            Some("2024-01-01T00:00:00Z".into())
        );
        assert_eq!(unix_to_iso8601(0), Some("1970-01-01T00:00:00Z".into()));
    }

    #[test]
    fn sanitize_id_rejects_path_traversal() {
        // `../../etc/passwd` — 6 non-alnum chars before "etc"
        // (`.`, `.`, `/`, `.`, `.`, `/`), one between, none
        // after, none before nothing. Every disallowed char
        // collapses to `_`.
        assert_eq!(sanitize_id("../../etc/passwd"), "______etc_passwd");
        assert_eq!(sanitize_id("ok-name_42"), "ok-name_42");
    }
}
