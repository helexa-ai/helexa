//! SSRF-guarded page fetch + readability extraction.
//!
//! Redirects are followed manually: every hop re-runs the URL checks
//! and per-hop DNS validation, and the connection is pinned to the
//! validated address via reqwest's `resolve()` so the client can't be
//! re-pointed at a private target between validation and connect.

use std::time::Duration;

use reqwest::redirect::Policy;
use thiserror::Error;
use url::Url;

use crate::config::ToolsConfig;
use crate::ssrf::{self, SsrfDenied};

const USER_AGENT: &str = concat!(
    "helexa-tools/",
    env!("CARGO_PKG_VERSION"),
    " (+https://helexa.ai; web_search grounding fetcher)"
);

#[derive(Debug, Error)]
pub enum FetchError {
    #[error("invalid url: {0}")]
    BadUrl(#[from] url::ParseError),
    #[error(transparent)]
    Denied(#[from] SsrfDenied),
    #[error("too many redirects")]
    TooManyRedirects,
    #[error("upstream returned status {0}")]
    UpstreamStatus(u16),
    #[error("unsupported content-type '{0}' (text/html or text/plain only)")]
    UnsupportedContentType(String),
    #[error("fetch failed: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("could not extract readable content: {0}")]
    Extraction(String),
}

impl FetchError {
    /// HTTP status the API surfaces for this failure.
    pub fn status(&self) -> u16 {
        match self {
            FetchError::BadUrl(_) | FetchError::Denied(_) => 400,
            FetchError::UnsupportedContentType(_) => 415,
            FetchError::TooManyRedirects
            | FetchError::UpstreamStatus(_)
            | FetchError::Transport(_)
            | FetchError::Extraction(_) => 502,
        }
    }
}

#[derive(Debug, serde::Serialize)]
pub struct Page {
    /// Final URL after redirects.
    pub url: String,
    pub title: String,
    /// Extracted article text (readability), truncated to the
    /// configured budget.
    pub text: String,
    pub truncated: bool,
}

pub async fn fetch_page(raw_url: &str, cfg: &ToolsConfig) -> Result<Page, FetchError> {
    let mut url = Url::parse(raw_url)?;
    for _hop in 0..=cfg.max_redirects {
        ssrf::check_url(&url)?;
        let pinned = ssrf::resolve_validated(&url).await?;
        let host = url.host_str().expect("checked").to_string();
        let client = reqwest::Client::builder()
            .redirect(Policy::none())
            .resolve(&host, pinned)
            .timeout(Duration::from_secs(cfg.fetch_timeout_secs))
            .user_agent(USER_AGENT)
            .build()?;

        let resp = client.get(url.clone()).send().await?;
        let status = resp.status();
        if status.is_redirection() {
            let location = resp
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or(FetchError::UpstreamStatus(status.as_u16()))?;
            url = url.join(location)?;
            continue;
        }
        if !status.is_success() {
            return Err(FetchError::UpstreamStatus(status.as_u16()));
        }

        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        let is_html = content_type.starts_with("text/html")
            || content_type.starts_with("application/xhtml+xml")
            || content_type.is_empty();
        let is_plain = content_type.starts_with("text/plain");
        if !is_html && !is_plain {
            return Err(FetchError::UnsupportedContentType(content_type));
        }

        let body = read_capped(resp, cfg.max_body_bytes).await?;
        if is_plain {
            let (text, truncated) = truncate_chars(&body, cfg.max_text_chars);
            return Ok(Page {
                url: url.to_string(),
                title: String::new(),
                text,
                truncated,
            });
        }
        return extract_article(&body, &url, cfg);
    }
    Err(FetchError::TooManyRedirects)
}

/// Read at most `cap` bytes of the response body, decoding as UTF-8
/// (lossy) — a page bigger than the cap is cut, not rejected.
async fn read_capped(resp: reqwest::Response, cap: usize) -> Result<String, FetchError> {
    let mut buf: Vec<u8> = Vec::with_capacity(64 * 1024);
    let mut resp = resp;
    while let Some(chunk) = resp.chunk().await? {
        let remaining = cap.saturating_sub(buf.len());
        if remaining == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn extract_article(html: &str, url: &Url, cfg: &ToolsConfig) -> Result<Page, FetchError> {
    let mut readability = dom_smoothie::Readability::new(html, Some(url.as_str()), None)
        .map_err(|e| FetchError::Extraction(e.to_string()))?;
    let article = readability
        .parse()
        .map_err(|e| FetchError::Extraction(e.to_string()))?;
    let (text, truncated) = truncate_chars(article.text_content.trim(), cfg.max_text_chars);
    Ok(Page {
        url: url.to_string(),
        title: article.title,
        text,
        truncated,
    })
}

fn truncate_chars(s: &str, max_chars: usize) -> (String, bool) {
    if s.chars().count() <= max_chars {
        (s.to_string(), false)
    } else {
        (s.chars().take(max_chars).collect(), true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ToolsConfig {
        ToolsConfig::default()
    }

    #[tokio::test]
    async fn denies_internal_targets_before_any_io() {
        for bad in [
            "http://10.3.0.1/",
            "http://beast.hanzalova.internal:13131/models",
            "file:///etc/passwd",
            "http://169.254.169.254/latest/meta-data/",
        ] {
            let err = fetch_page(bad, &cfg()).await.unwrap_err();
            assert_eq!(err.status(), 400, "{bad} should be a 400, got {err}");
        }
    }

    #[test]
    fn extracts_article_text() {
        let html = r#"<!doctype html><html><head><title>Test Article</title></head>
            <body><nav>menu menu menu</nav>
            <article><h1>Test Article</h1>
            <p>First paragraph of the article body with enough words to be
            considered content by the readability scorer. It keeps going and
            going with meaningful sentences about an interesting topic.</p>
            <p>Second paragraph, also substantial, adding further detail so
            the extractor keeps the article container as the top candidate.</p>
            </article>
            <footer>copyright</footer></body></html>"#;
        let url = Url::parse("https://example.com/article").unwrap();
        let page = extract_article(html, &url, &cfg()).expect("extract");
        assert_eq!(page.title, "Test Article");
        assert!(page.text.contains("First paragraph"));
        assert!(page.text.contains("Second paragraph"));
        assert!(!page.text.contains("menu menu"));
        assert!(!page.truncated);
    }

    #[test]
    fn truncates_to_char_budget() {
        let (t, cut) = truncate_chars(&"x".repeat(50), 10);
        assert_eq!(t.len(), 10);
        assert!(cut);
        // Multi-byte safety: chars, not bytes.
        let (t, cut) = truncate_chars(&"é".repeat(50), 10);
        assert_eq!(t.chars().count(), 10);
        assert!(cut);
    }
}
