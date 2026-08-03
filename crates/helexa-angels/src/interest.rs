//! Expressions of interest.
//!
//! No payments here, deliberately: taking six figures through a web form
//! is a different project with its own compliance surface. This records
//! intent and routes it to a human.
//!
//! The three questions asked are the three commercial axes, and all of
//! them are the **investor's** decision:
//!
//! 1. **who purchases** — the investor directly from Tenstorrent, or
//!    Bears Lairs EOOD on their behalf once funds are received;
//! 2. **who hosts** — helexa, the Bears Lair datacentre, or the
//!    investor's own premises;
//! 3. **who covers maintenance and running costs.**
//!
//! Contracts are bespoke per investor to reflect the combination chosen,
//! so what is captured here is a starting position for a conversation and
//! not an order. The form says so, because a form that reads like a
//! checkout implies terms nobody has agreed.

use crate::audit::{self, Access};
use crate::auth;
use crate::error::{AngelsError, Result};
use crate::grants;
use crate::state::AppState;
use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::{Form, extract::Query};
use minijinja::value::Value;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct InterestForm {
    #[serde(default)]
    pub package_ref: String,
    #[serde(default)]
    pub purchaser: String,
    #[serde(default)]
    pub hosting_choice: String,
    #[serde(default)]
    pub running_costs: String,
    #[serde(default)]
    pub message: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct InterestQuery {
    pub sent: Option<String>,
}

/// `GET /r/{slug}/interest`
pub async fn form(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(slug): Path<String>,
    Query(q): Query<InterestQuery>,
) -> Result<Response> {
    let Some(session) = auth::resolve_session(&state.pool, &headers, &state.config.session).await
    else {
        return Ok(Redirect::to(&format!("/signin?next=/r/{slug}/interest")).into_response());
    };
    if !grants::has_access(&state.pool, session.user_id, &slug).await? {
        return Err(AngelsError::Forbidden);
    }

    let manifest = state.content.manifest(&slug)?;
    let body = crate::templates::render(
        "interest.html",
        crate::web::ctx(
            &state,
            vec![
                ("user_email", Value::from(session.email)),
                ("round", Value::from_serialize(&manifest)),
                ("sent", Value::from(q.sent.is_some())),
            ],
        ),
    )?;
    Ok(Html(body).into_response())
}

/// `POST /r/{slug}/interest`
pub async fn submit(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(slug): Path<String>,
    Form(form): Form<InterestForm>,
) -> Result<Response> {
    let Some(session) = auth::resolve_session(&state.pool, &headers, &state.config.session).await
    else {
        return Err(AngelsError::Unauthenticated);
    };
    if !grants::has_access(&state.pool, session.user_id, &slug).await? {
        return Err(AngelsError::Forbidden);
    }

    // Persist first. The notification is best-effort, and an investor
    // must never be told their submission failed because a relay was
    // down — the row is the record, the mail is a convenience.
    sqlx::query(
        "INSERT INTO interest \
           (user_id, round_slug, package_ref, purchaser, hosting_choice, running_costs, message) \
         VALUES ($1, $2, $3, $4, $5, $6, $7)",
    )
    .bind(session.user_id)
    .bind(&slug)
    .bind(trim_opt(&form.package_ref))
    .bind(trim_opt(&form.purchaser))
    .bind(trim_opt(&form.hosting_choice))
    .bind(trim_opt(&form.running_costs))
    .bind(trim_opt(&form.message))
    .execute(&state.pool)
    .await?;

    audit::record(
        &state.pool,
        Access {
            user_id: Some(session.user_id),
            user_email: Some(&session.email),
            round_slug: Some(&slug),
            document_slug: Some("(interest)"),
            content_version: None,
            kind: "view",
            ip: auth::client_ip(&headers).as_deref(),
            user_agent: auth::user_agent(&headers).as_deref(),
        },
    )
    .await;

    let body = format!(
        "Expression of interest — {slug}\n\n\
         From:            {}\n\
         Package:         {}\n\
         Purchaser:       {}\n\
         Hosting:         {}\n\
         Running costs:   {}\n\n\
         Message:\n{}\n\n\
         --\n\
         Contracts are bespoke per investor; this is a starting position, \
         not an order.",
        session.email,
        blank(&form.package_ref),
        blank(&form.purchaser),
        blank(&form.hosting_choice),
        blank(&form.running_costs),
        blank(&form.message),
    );

    if let Err(e) = state
        .notifier
        .send(
            &state.config.email.notify_to,
            &format!("[angels] interest from {}", session.email),
            &body,
        )
        .await
    {
        tracing::error!(error = %e, round = %slug, "interest notification failed — the row is stored, chase it with `helexa-angels interest`");
    }

    Ok(Redirect::to(&format!("/r/{slug}/interest?sent=1")).into_response())
}

fn trim_opt(s: &str) -> Option<String> {
    let t = s.trim();
    if t.is_empty() {
        None
    } else {
        // Bound what a form can push into the database and the mail body.
        Some(t.chars().take(4000).collect())
    }
}

fn blank(s: &str) -> &str {
    let t = s.trim();
    if t.is_empty() { "—" } else { t }
}

/// Operator listing.
pub async fn list(
    pool: &sqlx::postgres::PgPool,
    round: Option<&str>,
) -> std::result::Result<Vec<(String, String, String, String, String)>, sqlx::Error> {
    use sqlx::Row;
    let rows = sqlx::query(
        "SELECT to_char(i.submitted_at, 'YYYY-MM-DD HH24:MI') AS at, \
                u.email::text AS who, \
                coalesce(i.package_ref, '-') AS package, \
                coalesce(i.hosting_choice, '-') AS hosting, \
                i.state \
         FROM interest i JOIN public.users u ON u.id = i.user_id \
         WHERE ($1::text IS NULL OR i.round_slug = $1) \
         ORDER BY i.submitted_at DESC",
    )
    .bind(round)
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|r| {
            (
                r.get("at"),
                r.get("who"),
                r.get("package"),
                r.get("hosting"),
                r.get("state"),
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_fields_render_as_a_dash() {
        assert_eq!(blank("  "), "—");
        assert_eq!(blank("Galaxy"), "Galaxy");
    }

    #[test]
    fn empty_input_stores_null_not_empty_string() {
        assert_eq!(trim_opt("   "), None);
        assert_eq!(trim_opt(" Galaxy "), Some("Galaxy".to_string()));
    }

    #[test]
    fn oversized_input_is_bounded() {
        let huge = "x".repeat(10_000);
        assert_eq!(trim_opt(&huge).map(|s| s.len()), Some(4000));
    }
}
