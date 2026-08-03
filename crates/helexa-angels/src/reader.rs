//! The reading surface: round contents and documents.
//!
//! Every path here is gated on an **active grant** and records what was
//! read before returning it. The order matters — the audit row is written
//! for denials too, because an account probing rounds it has no grant on
//! is exactly the signal worth having.

use crate::audit::{self, Access};
use crate::auth;
use crate::error::{AngelsError, Result};
use crate::grants;
use crate::state::AppState;
use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::{Html, IntoResponse, Redirect, Response};
use minijinja::value::Value;

/// Timestamp for the watermark. Deliberately minute-resolution UTC: it
/// identifies a reading session without implying more precision than the
/// record actually supports.
fn now_stamp() -> String {
    chrono::Utc::now().format("%Y-%m-%d %H:%M UTC").to_string()
}

/// `GET /r/{slug}` — a round's contents page.
pub async fn round_index(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(slug): Path<String>,
) -> Result<Response> {
    let Some(session) = auth::resolve_session(&state.pool, &headers, &state.config.session).await
    else {
        return Ok(Redirect::to(&format!("/signin?next=/r/{slug}")).into_response());
    };

    let ip = auth::client_ip(&headers);
    let ua = auth::user_agent(&headers);

    if !grants::has_access(&state.pool, session.user_id, &slug).await? {
        audit::record(
            &state.pool,
            Access {
                user_id: Some(session.user_id),
                user_email: Some(&session.email),
                round_slug: Some(&slug),
                document_slug: None,
                content_version: None,
                kind: "denied",
                ip: ip.as_deref(),
                user_agent: ua.as_deref(),
            },
        )
        .await;
        return Err(AngelsError::Forbidden);
    }

    let manifest = state.content.manifest(&slug)?;
    let version = state.content.version();
    let disclaimer = state.content.disclaimer(&slug, &manifest.disclaimer);

    audit::record(
        &state.pool,
        Access {
            user_id: Some(session.user_id),
            user_email: Some(&session.email),
            round_slug: Some(&slug),
            document_slug: Some("(contents)"),
            content_version: Some(&version),
            kind: "view",
            ip: ip.as_deref(),
            user_agent: ua.as_deref(),
        },
    )
    .await;

    let body = crate::templates::render(
        "round.html",
        crate::web::ctx(
            &state,
            vec![
                ("user_email", Value::from(session.email)),
                ("round", Value::from_serialize(&manifest)),
                ("content_version", Value::from(version)),
                ("now", Value::from(now_stamp())),
                (
                    "disclaimer",
                    disclaimer.map(Value::from).unwrap_or_default(),
                ),
            ],
        ),
    )?;
    Ok(Html(body).into_response())
}

/// `GET /r/{slug}/{doc}` — one document.
pub async fn document(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((slug, doc_slug)): Path<(String, String)>,
) -> Result<Response> {
    let Some(session) = auth::resolve_session(&state.pool, &headers, &state.config.session).await
    else {
        return Ok(Redirect::to(&format!("/signin?next=/r/{slug}/{doc_slug}")).into_response());
    };

    let ip = auth::client_ip(&headers);
    let ua = auth::user_agent(&headers);

    if !grants::has_access(&state.pool, session.user_id, &slug).await? {
        audit::record(
            &state.pool,
            Access {
                user_id: Some(session.user_id),
                user_email: Some(&session.email),
                round_slug: Some(&slug),
                document_slug: Some(&doc_slug),
                content_version: None,
                kind: "denied",
                ip: ip.as_deref(),
                user_agent: ua.as_deref(),
            },
        )
        .await;
        return Err(AngelsError::Forbidden);
    }

    let manifest = state.content.manifest(&slug)?;
    let (entry, body_html) = state.content.document(&slug, &doc_slug)?;
    let version = state.content.version();

    // Neighbours, so a long plan reads as a document rather than a set of
    // disconnected pages.
    let idx = manifest.documents.iter().position(|d| d.slug == doc_slug);
    let prev = idx
        .and_then(|i| i.checked_sub(1))
        .and_then(|i| manifest.documents.get(i));
    let next = idx.and_then(|i| manifest.documents.get(i + 1));

    audit::record(
        &state.pool,
        Access {
            user_id: Some(session.user_id),
            user_email: Some(&session.email),
            round_slug: Some(&slug),
            document_slug: Some(&doc_slug),
            content_version: Some(&version),
            kind: "view",
            ip: ip.as_deref(),
            user_agent: ua.as_deref(),
        },
    )
    .await;

    let rendered = crate::templates::render(
        "document.html",
        crate::web::ctx(
            &state,
            vec![
                ("user_email", Value::from(session.email)),
                ("round", Value::from_serialize(&manifest)),
                ("doc", Value::from_serialize(&entry)),
                ("body", Value::from(body_html)),
                ("content_version", Value::from(version)),
                ("now", Value::from(now_stamp())),
                ("prev", prev.map(Value::from_serialize).unwrap_or_default()),
                ("next", next.map(Value::from_serialize).unwrap_or_default()),
            ],
        ),
    )?;
    Ok(Html(rendered).into_response())
}
