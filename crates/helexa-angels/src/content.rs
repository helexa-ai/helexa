//! Round content: manifests and documents, loaded from disk.
//!
//! Content lives in a directory outside any web root and outside this
//! repository. `helexa/helexa` is open source, so a business plan
//! committed here would be a business plan published — the separation is
//! not tidiness, it is the point.
//!
//! Layout:
//!
//! ```text
//! <content.dir>/
//!   VERSION                 # optional; stamped into every access record
//!   tt-eap-2026/
//!     round.toml            # metadata, framing, document order
//!     overview.md
//!     hardware.md
//!     ...
//! ```
//!
//! Markdown is rendered to HTML here and inserted with `|safe`, so raw
//! HTML in a document is passed through. That is deliberate and bounded:
//! these files are operator-authored and reachable only by someone with
//! filesystem access to the server. They are never visitor input — the one
//! place visitor text is rendered (the interest form) goes through
//! ordinary escaping.

use crate::error::{AngelsError, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// `round.toml` — a round's own description of itself.
///
/// Framing is per-round because rounds differ in kind, not just in
/// content: this one is an early-access programme in which the investor
/// buys hardware they own outright, and a later one may be something else
/// entirely. Nothing here assumes equity, shares, or priced packages.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RoundManifest {
    pub slug: String,
    pub title: String,
    #[serde(default = "default_framing")]
    pub framing_label: String,
    #[serde(default = "default_status")]
    pub status: String,
    #[serde(default = "default_true")]
    pub auto_grant: bool,
    #[serde(default)]
    pub summary: String,
    /// Shown above the document list — the standing legal note for this
    /// round's framing.
    #[serde(default)]
    pub disclaimer: String,
    #[serde(default)]
    pub documents: Vec<DocumentEntry>,
    /// Optional call to action rendered at the foot of the round index.
    #[serde(default)]
    pub cta_label: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DocumentEntry {
    pub slug: String,
    pub title: String,
    pub file: String,
    #[serde(default)]
    pub summary: String,
}

fn default_framing() -> String {
    "Early Access Programme".into()
}
fn default_status() -> String {
    "draft".into()
}
fn default_true() -> bool {
    true
}

/// The content tree on disk.
#[derive(Clone)]
pub struct Content {
    root: PathBuf,
}

impl Content {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { root: dir.into() }
    }

    /// A version tag for the whole tree, stamped into every access record
    /// so "which version of the plan did this investor see?" is
    /// answerable — which matters when the terms discussed in a meeting
    /// differ from the terms on the page.
    pub fn version(&self) -> String {
        std::fs::read_to_string(self.root.join("VERSION"))
            .map(|s| s.trim().to_string())
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unversioned".into())
    }

    /// Every round manifest present on disk.
    pub fn rounds(&self) -> Vec<RoundManifest> {
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            tracing::warn!(dir = %self.root.display(), "content directory unreadable");
            return Vec::new();
        };
        let mut out = Vec::new();
        for entry in entries.flatten() {
            if !entry.path().is_dir() {
                continue;
            }
            match self.manifest_at(&entry.path()) {
                Ok(m) => out.push(m),
                Err(e) => {
                    tracing::warn!(path = %entry.path().display(), error = %e, "skipping round")
                }
            }
        }
        out.sort_by(|a, b| a.slug.cmp(&b.slug));
        out
    }

    /// One round's manifest.
    pub fn manifest(&self, slug: &str) -> Result<RoundManifest> {
        let dir = self.round_dir(slug)?;
        self.manifest_at(&dir).map_err(|_| AngelsError::NotFound)
    }

    fn manifest_at(&self, dir: &Path) -> anyhow::Result<RoundManifest> {
        let raw = std::fs::read_to_string(dir.join("round.toml"))?;
        Ok(toml::from_str::<RoundManifest>(&raw)?)
    }

    /// Render one document to HTML.
    pub fn document(&self, slug: &str, doc_slug: &str) -> Result<(DocumentEntry, String)> {
        let manifest = self.manifest(slug)?;
        let entry = manifest
            .documents
            .iter()
            .find(|d| d.slug == doc_slug)
            .cloned()
            .ok_or(AngelsError::NotFound)?;
        let dir = self.round_dir(slug)?;
        let path = safe_join(&dir, &entry.file).ok_or(AngelsError::NotFound)?;
        let md = std::fs::read_to_string(path).map_err(|_| AngelsError::NotFound)?;
        Ok((entry, markdown_to_html(&md)))
    }

    /// The round's disclaimer, rendered.
    pub fn disclaimer(&self, slug: &str, file: &str) -> Option<String> {
        if file.is_empty() {
            return None;
        }
        let dir = self.round_dir(slug).ok()?;
        let path = safe_join(&dir, file)?;
        std::fs::read_to_string(path)
            .ok()
            .map(|s| markdown_to_html(&s))
    }

    fn round_dir(&self, slug: &str) -> Result<PathBuf> {
        safe_join(&self.root, slug).ok_or(AngelsError::NotFound)
    }
}

/// Join a caller-supplied component onto a base directory, refusing
/// anything that could escape it.
///
/// Round and document slugs reach this from the URL. Without the check, a
/// request for `/r/../../etc/passwd` would be a file-disclosure bug in a
/// service whose entire job is not disclosing files.
fn safe_join(base: &Path, component: &str) -> Option<PathBuf> {
    if component.is_empty()
        || component.contains("..")
        || component.contains('\0')
        || component.starts_with('/')
        || component.contains('\\')
    {
        return None;
    }
    // Permit only what a slug or filename legitimately needs.
    if !component
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/'))
    {
        return None;
    }
    let joined = base.join(component);
    // Belt and braces: the result must still sit under the base once the
    // OS has had its say about symlinks.
    let base_c = base.canonicalize().ok()?;
    match joined.canonicalize() {
        Ok(real) if real.starts_with(&base_c) => Some(real),
        _ => None,
    }
}

/// Markdown → HTML, with the extensions a business plan actually uses.
pub fn markdown_to_html(md: &str) -> String {
    use pulldown_cmark::{Options, Parser, html};
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_FOOTNOTES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_SMART_PUNCTUATION);
    let parser = Parser::new_ext(md, opts);
    let mut out = String::with_capacity(md.len() * 3 / 2);
    html::push_html(&mut out, parser);
    out
}

/// Reconcile the `rounds` table with the manifests on disk.
///
/// Disk is the source of truth for everything an operator edits — title,
/// framing, status, auto-grant — so publishing a round is `status = "open"`
/// in `round.toml` and a restart, not a database edit. The table exists so
/// grants and invites have something to reference and so a listing can be
/// produced without walking the filesystem.
///
/// Rounds are never deleted here: a manifest removed from disk leaves its
/// grants and access history intact, which is what you want when the
/// question later is "who saw the plan we withdrew?"
pub async fn sync_rounds(
    pool: &sqlx::postgres::PgPool,
    content: &Content,
) -> std::result::Result<usize, sqlx::Error> {
    let manifests = content.rounds();
    let version = content.version();
    for m in &manifests {
        sqlx::query(
            "INSERT INTO rounds (slug, title, framing_label, status, auto_grant, content_version, opened_at, closed_at) \
             VALUES ($1, $2, $3, $4, $5, $6, \
                     CASE WHEN $4 = 'open' THEN now() END, \
                     CASE WHEN $4 = 'closed' THEN now() END) \
             ON CONFLICT (slug) DO UPDATE SET \
               title = EXCLUDED.title, \
               framing_label = EXCLUDED.framing_label, \
               status = EXCLUDED.status, \
               auto_grant = EXCLUDED.auto_grant, \
               content_version = EXCLUDED.content_version, \
               opened_at = COALESCE(rounds.opened_at, EXCLUDED.opened_at), \
               closed_at = CASE WHEN EXCLUDED.status = 'closed' \
                                THEN COALESCE(rounds.closed_at, now()) ELSE NULL END",
        )
        .bind(&m.slug)
        .bind(&m.title)
        .bind(&m.framing_label)
        .bind(&m.status)
        .bind(m.auto_grant)
        .bind(&version)
        .execute(pool)
        .await?;
    }
    Ok(manifests.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_renders_tables_and_emphasis() {
        let html = markdown_to_html("| a | b |\n|---|---|\n| 1 | 2 |\n\n**bold**");
        assert!(html.contains("<table>"), "{html}");
        assert!(html.contains("<strong>bold</strong>"), "{html}");
    }

    #[test]
    fn path_traversal_is_refused() {
        let base = std::env::temp_dir();
        assert!(safe_join(&base, "../etc/passwd").is_none());
        assert!(safe_join(&base, "/etc/passwd").is_none());
        assert!(safe_join(&base, "..").is_none());
        assert!(safe_join(&base, "").is_none());
        assert!(safe_join(&base, "a\0b").is_none());
        assert!(safe_join(&base, "..\\windows").is_none());
        // Odd characters that have no business in a slug.
        assert!(safe_join(&base, "round;rm -rf").is_none());
    }

    #[test]
    fn safe_join_resolves_a_real_child() {
        let dir = std::env::temp_dir().join(format!("angels-test-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("round-a")).unwrap();
        let joined = safe_join(&dir, "round-a");
        assert!(joined.is_some(), "a legitimate child must resolve");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn manifest_parses_with_defaults() {
        let m: RoundManifest = toml::from_str(
            r#"
            slug = "tt-eap-2026"
            title = "Tenstorrent Early Access Programme"
            [[documents]]
            slug = "overview"
            title = "At a glance"
            file = "overview.md"
            "#,
        )
        .unwrap();
        assert_eq!(m.slug, "tt-eap-2026");
        // A round is a draft until it explicitly says otherwise — a
        // half-written plan must never be reachable by accident.
        assert_eq!(m.status, "draft");
        assert_eq!(m.framing_label, "Early Access Programme");
        assert_eq!(m.documents.len(), 1);
    }

    #[test]
    fn missing_version_file_degrades_to_a_marker_not_a_panic() {
        let c = Content::new("/nonexistent/angels/content");
        assert_eq!(c.version(), "unversioned");
        assert!(c.rounds().is_empty());
    }
}
