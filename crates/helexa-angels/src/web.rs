//! HTTP surface.
//!
//! Every response is assembled here, on the server. An unauthenticated
//! request can reach exactly three things: the sign-in page, the
//! registration page, and the privacy note. There is no bundle to scrape
//! and no API that hands content to anything holding a token — which is
//! the entire reason this service exists separately from the helexa.ai
//! SPA.

use crate::auth::{self, Session};
use crate::error::{AngelsError, Result};
use crate::state::AppState;
use crate::templates;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::get;
use axum::{Form, Router};
use minijinja::value::Value;
use serde::Deserialize;
use std::collections::BTreeMap;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(portal))
        .route("/signin", get(signin_page).post(signin_submit))
        .route("/register", get(register_page).post(register_submit))
        .route("/signout", get(signout))
        .route("/account", get(account))
        .route("/privacy", get(privacy))
        .route("/health", get(health))
}

/// Liveness only — deliberately says nothing about rounds, grants, or
/// whether the content tree is loaded. It is reachable unauthenticated,
/// so it must not become a status oracle.
async fn health() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

/// Build a template context from the state's base plus per-page extras.
fn ctx(state: &AppState, extra: Vec<(&str, Value)>) -> Value {
    let mut map: BTreeMap<String, Value> = BTreeMap::new();
    for (k, v) in state.base_context() {
        map.insert(k.to_string(), Value::from(v));
    }
    for (k, v) in extra {
        map.insert(k.to_string(), v);
    }
    Value::from_object(map)
}

#[derive(Debug, Deserialize, Default)]
pub struct AuthQuery {
    /// Where to go after signing in, preserved across the auth detour.
    pub next: Option<String>,
    pub error: Option<String>,
    pub notice: Option<String>,
}

impl AuthQuery {
    /// Re-encode as a query string for the tab links, so `?next=` is not
    /// lost when the visitor switches between sign in and register.
    fn qs(&self) -> String {
        match self.next.as_deref().filter(|s| is_safe_next(s)) {
            Some(n) => format!("?next={}", urlencode(n)),
            None => String::new(),
        }
    }
}

/// Only ever redirect within this site.
///
/// An unchecked `next` is an open redirect: a link to
/// `angels.helexa.ai/signin?next=https://evil.example` would bounce a
/// signed-in investor off-site, and the address bar would have said
/// `helexa` the whole way. Requiring a single leading slash (and rejecting
/// `//host`, which browsers read as protocol-relative) confines it.
fn is_safe_next(next: &str) -> bool {
    next.starts_with('/') && !next.starts_with("//") && !next.contains("://")
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

// ── Portal ──────────────────────────────────────────────────────────

async fn portal(State(state): State<AppState>, headers: HeaderMap) -> Result<Response> {
    let Some(session) = auth::resolve_session(&state.pool, &headers, &state.config.session).await
    else {
        return Ok(Redirect::to("/signin").into_response());
    };

    let rounds = crate::grants::rounds_for_user(&state.pool, session.user_id).await?;
    let body = templates::render(
        "portal.html",
        ctx(
            &state,
            vec![
                ("user_email", Value::from(session.email)),
                ("rounds", Value::from_serialize(&rounds)),
            ],
        ),
    )?;
    Ok(Html(body).into_response())
}

async fn account(State(state): State<AppState>, headers: HeaderMap) -> Result<Response> {
    let Some(session) = auth::resolve_session(&state.pool, &headers, &state.config.session).await
    else {
        return Ok(Redirect::to("/signin?next=/account").into_response());
    };
    let rounds = crate::grants::rounds_for_user(&state.pool, session.user_id).await?;
    let body = templates::render(
        "account.html",
        ctx(
            &state,
            vec![
                ("user_email", Value::from(session.email)),
                ("rounds", Value::from_serialize(&rounds)),
            ],
        ),
    )?;
    Ok(Html(body).into_response())
}

async fn privacy(State(state): State<AppState>, headers: HeaderMap) -> Result<Response> {
    let session = auth::resolve_session(&state.pool, &headers, &state.config.session).await;
    let body = templates::render(
        "privacy.html",
        ctx(
            &state,
            vec![
                (
                    "user_email",
                    session.map(|s| Value::from(s.email)).unwrap_or_default(),
                ),
                ("retention_months", Value::from(crate::RETENTION_MONTHS)),
            ],
        ),
    )?;
    Ok(Html(body).into_response())
}

// ── Sign in / register ──────────────────────────────────────────────

fn auth_page(
    state: &AppState,
    q: &AuthQuery,
    tab: &str,
    invite_label: Option<String>,
) -> Result<Response> {
    let (heading, subheading) = match (tab, invite_label.as_deref()) {
        ("register", Some(_)) => (
            "Create your account",
            "You've been invited to review confidential material. Create an \
             account and it will be attached to you.",
        ),
        ("register", None) => (
            "Create your account",
            "One account covers this portal and helexa.ai itself.",
        ),
        (_, Some(_)) => (
            "Sign in to continue",
            "You've been invited to review confidential material. Sign in \
             and it will be attached to your account.",
        ),
        _ => (
            "Sign in",
            "This portal holds confidential material prepared for named \
             recipients.",
        ),
    };

    let body = templates::render(
        "signin.html",
        ctx(
            state,
            vec![
                ("tab", Value::from(tab)),
                ("qs", Value::from(q.qs())),
                ("heading", Value::from(heading)),
                ("subheading", Value::from(subheading)),
                (
                    "invite_label",
                    invite_label.map(Value::from).unwrap_or_default(),
                ),
                (
                    "error",
                    q.error.clone().map(Value::from).unwrap_or_default(),
                ),
                (
                    "notice",
                    q.notice.clone().map(Value::from).unwrap_or_default(),
                ),
            ],
        ),
    )?;
    Ok(Html(body).into_response())
}

async fn signin_page(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<AuthQuery>,
) -> Result<Response> {
    // Already signed in — skip the form.
    if auth::resolve_session(&state.pool, &headers, &state.config.session)
        .await
        .is_some()
    {
        let dest = q.next.filter(|n| is_safe_next(n)).unwrap_or("/".into());
        return Ok(Redirect::to(&dest).into_response());
    }
    let label = crate::invites::pending_invite_label(&state, &headers).await;
    auth_page(&state, &q, "signin", label)
}

async fn register_page(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<AuthQuery>,
) -> Result<Response> {
    let label = crate::invites::pending_invite_label(&state, &headers).await;
    auth_page(&state, &q, "register", label)
}

#[derive(Debug, Deserialize)]
pub struct Credentials {
    pub email: String,
    pub password: String,
}

async fn signin_submit(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<AuthQuery>,
    Form(form): Form<Credentials>,
) -> Result<Response> {
    let session = match auth::verify_credentials(&state.pool, &form.email, &form.password).await {
        Ok(s) => s,
        Err(_) => {
            // One message for every failure mode — see verify_credentials.
            let dest = format!(
                "/signin?error={}{}",
                urlencode(
                    "That email address and password don't match, or the address isn't confirmed yet."
                ),
                q.next
                    .as_deref()
                    .filter(|n| is_safe_next(n))
                    .map(|n| format!("&next={}", urlencode(n)))
                    .unwrap_or_default()
            );
            return Ok(Redirect::to(&dest).into_response());
        }
    };

    complete_signin(&state, &headers, session, q.next.as_deref()).await
}

/// Issue the session, redeem any pending invite, and land the visitor.
async fn complete_signin(
    state: &AppState,
    headers: &HeaderMap,
    session: Session,
    next: Option<&str>,
) -> Result<Response> {
    let raw = auth::issue_session(
        &state.pool,
        session.user_id,
        state.config.session.ttl_secs,
        auth::client_ip(headers).as_deref(),
        auth::user_agent(headers).as_deref(),
    )
    .await?;

    // An invitation waiting in the pending cookie becomes a grant now that
    // we know who the visitor is. This is the moment an anonymous, freely
    // forwardable code turns into a named, revocable, audited grant.
    let landed = crate::invites::redeem_pending(state, headers, session.user_id).await;

    let mut out = HeaderMap::new();
    out = auth::with_cookie(
        out,
        auth::session_cookie(
            &state.config.session.cookie_name,
            &raw,
            state.config.session.ttl_secs,
            state.config.session.secure,
        ),
    );
    // The invite has been consumed either way; do not leave it lying about.
    out = auth::with_cookie(
        out,
        auth::clearing_cookie("angels_invite", state.config.session.secure),
    );

    let dest = next
        .filter(|n| is_safe_next(n))
        .map(|n| n.to_string())
        .or(landed)
        .unwrap_or_else(|| "/".into());

    Ok((StatusCode::SEE_OTHER, out, [(header::LOCATION, dest)]).into_response())
}

async fn register_submit(
    State(state): State<AppState>,
    Form(form): Form<Credentials>,
) -> Result<Response> {
    if form.password.chars().count() < 8 {
        return Ok(Redirect::to(&format!(
            "/register?error={}",
            urlencode("Please choose a password of at least 8 characters.")
        ))
        .into_response());
    }

    // Delegated to helexa-upstream rather than reimplemented here: it owns
    // password policy, argon2 parameters, the verification email, the
    // unverified-signup reaper and fingerprinting. Two implementations
    // writing one `users` table is a defect waiting to happen.
    match crate::upstream::register(&state, &form.email, &form.password).await {
        Ok(()) => Ok(Redirect::to(&format!(
            "/signin?notice={}",
            urlencode(
                "Check your email — we've sent you a link to confirm the address. \
                 Once confirmed, sign in here."
            )
        ))
        .into_response()),
        Err(e) => {
            tracing::warn!(error = %e, "registration via upstream failed");
            Ok(Redirect::to(&format!(
                "/register?error={}",
                urlencode(
                    "We couldn't create that account. If you already have a helexa \
                     account, sign in instead."
                )
            ))
            .into_response())
        }
    }
}

async fn signout(State(state): State<AppState>, headers: HeaderMap) -> Result<Response> {
    auth::destroy_session(&state.pool, &headers, &state.config.session.cookie_name).await;
    let out = auth::with_cookie(
        HeaderMap::new(),
        auth::clearing_cookie(
            &state.config.session.cookie_name,
            state.config.session.secure,
        ),
    );
    Ok((StatusCode::SEE_OTHER, out, [(header::LOCATION, "/signin")]).into_response())
}

/// Not found, rendered as a page rather than an empty 404.
pub async fn fallback() -> Response {
    AngelsError::NotFound.into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_off_site_redirects() {
        assert!(is_safe_next("/r/tt-eap-2026"));
        assert!(is_safe_next("/account"));
        // Protocol-relative: browsers treat //evil.example as a host.
        assert!(!is_safe_next("//evil.example"));
        assert!(!is_safe_next("https://evil.example"));
        assert!(!is_safe_next("javascript:alert(1)"));
        assert!(!is_safe_next("evil.example"));
    }

    #[test]
    fn query_string_round_trips_only_safe_next() {
        let q = AuthQuery {
            next: Some("/r/tt-eap-2026".into()),
            ..Default::default()
        };
        assert_eq!(q.qs(), "?next=/r/tt-eap-2026");

        let hostile = AuthQuery {
            next: Some("https://evil.example".into()),
            ..Default::default()
        };
        assert_eq!(hostile.qs(), "", "an unsafe next must not survive");
    }

    #[test]
    fn urlencode_escapes_delimiters() {
        assert_eq!(urlencode("a b&c=d"), "a%20b%26c%3Dd");
        assert_eq!(urlencode("/r/x"), "/r/x");
    }
}
