//! Invitation codes.
//!
//! A reusable code is a **distribution** mechanism, not a security
//! boundary. It will be forwarded — to a spouse, an accountant, a
//! co-investor — and the design assumes so rather than pretending
//! otherwise. What the code buys is a chance to identify yourself;
//! confidentiality then rests on two things it cannot bypass:
//!
//! 1. documents are served only to an authenticated user holding a grant;
//! 2. every view is attributed and logged.
//!
//! Someone who forwards a code produces another *named* account, not
//! anonymous access. That is strictly better than an unguessable content
//! URL, which produces anonymous access and cannot be revoked once shared.

use crate::auth;
use crate::crypto;
use crate::error::Result;
use crate::state::AppState;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use sqlx::Row;
use sqlx::postgres::PgPool;
use uuid::Uuid;

pub struct Invite {
    pub id: Uuid,
    pub round_slug: String,
    pub label: String,
    pub round_title: String,
    pub auto_grant: bool,
}

/// Look up a live invite by its plaintext code.
///
/// Returns `None` for unknown, revoked, expired and exhausted codes
/// alike — the caller must not be able to tell them apart, or the portal
/// becomes an oracle for probing which codes exist.
pub async fn lookup(pool: &PgPool, code: &str) -> Option<Invite> {
    let row = sqlx::query(
        "SELECT i.id, i.round_slug, i.label, r.title, r.auto_grant \
         FROM invites i JOIN rounds r ON r.slug = i.round_slug \
         WHERE i.code_hash = $1 \
           AND i.revoked_at IS NULL \
           AND (i.expires_at IS NULL OR i.expires_at > now()) \
           AND (i.max_uses IS NULL OR i.used_count < i.max_uses) \
           AND r.status <> 'draft'",
    )
    .bind(crypto::sha256(code))
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()?;

    Some(Invite {
        id: row.get("id"),
        round_slug: row.get("round_slug"),
        label: row.get("label"),
        round_title: row.get("title"),
        auto_grant: row.get("auto_grant"),
    })
}

/// `GET /i/{code}` — the entry point an operator actually sends.
///
/// Signed in already: redeem immediately. Not signed in: stash the code in
/// a short-lived cookie and send them to sign in, so they are not asked to
/// find the link again afterwards.
pub async fn enter(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(code): Path<String>,
) -> Result<Response> {
    let Some(invite) = lookup(&state.pool, &code).await else {
        // Identical to any other dead link. Nothing here confirms whether
        // a code ever existed.
        return Ok(crate::error::AngelsError::NotFound.into_response());
    };

    if let Some(session) = auth::resolve_session(&state.pool, &headers, &state.config.session).await
    {
        let dest = redeem(&state, &invite, session.user_id).await?;
        return Ok((StatusCode::SEE_OTHER, [(header::LOCATION, dest)]).into_response());
    }

    let out = auth::with_cookie(
        HeaderMap::new(),
        auth::pending_invite_cookie(&code, state.config.session.secure),
    );
    Ok((
        StatusCode::SEE_OTHER,
        out,
        [(header::LOCATION, "/signin".to_string())],
    )
        .into_response())
}

/// Turn an invite into a grant for a now-known user, and count the use.
async fn redeem(state: &AppState, invite: &Invite, user_id: Uuid) -> Result<String> {
    let landed = crate::grants::upsert(
        &state.pool,
        user_id,
        &invite.round_slug,
        Some(invite.id),
        invite.auto_grant,
    )
    .await?;

    let _ = sqlx::query("UPDATE invites SET used_count = used_count + 1 WHERE id = $1")
        .bind(invite.id)
        .execute(&state.pool)
        .await;

    tracing::info!(
        round = %invite.round_slug,
        state = %landed,
        "invite redeemed"
    );

    Ok(match landed.as_str() {
        "active" => format!("/r/{}", invite.round_slug),
        // pending (awaiting approval) or revoked — the portal explains.
        _ => "/".to_string(),
    })
}

/// Redeem whatever invite was waiting in the pending cookie, if any.
/// Returns where to send the visitor next.
pub async fn redeem_pending(
    state: &AppState,
    headers: &HeaderMap,
    user_id: Uuid,
) -> Option<String> {
    let code = auth::cookie_value(headers, "angels_invite")?;
    let invite = lookup(&state.pool, &code).await?;
    redeem(state, &invite, user_id).await.ok()
}

/// The round title behind a pending invite, so the sign-in page can say
/// what the visitor is signing in *for*.
pub async fn pending_invite_label(state: &AppState, headers: &HeaderMap) -> Option<String> {
    let code = auth::cookie_value(headers, "angels_invite")?;
    lookup(&state.pool, &code).await.map(|i| i.round_title)
}

/// Mint a code. The plaintext is returned once, here; only its hash is
/// stored, so a database disclosure does not yield working invitations.
pub async fn mint(
    pool: &PgPool,
    round_slug: &str,
    label: &str,
    max_uses: Option<i32>,
    expires_days: Option<i64>,
) -> Result<String> {
    let code = crypto::generate_invite_code();
    let expires = expires_days.map(|d| chrono::Utc::now() + chrono::Duration::days(d));
    sqlx::query(
        "INSERT INTO invites (code_hash, label, round_slug, max_uses, expires_at) \
         VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(crypto::sha256(&code))
    .bind(label)
    .bind(round_slug)
    .bind(max_uses)
    .bind(expires)
    .execute(pool)
    .await?;
    Ok(code)
}

/// Stop a code issuing further grants. Existing grants are untouched —
/// revoking a code and revoking a person are different acts.
pub async fn revoke(pool: &PgPool, label: &str) -> Result<u64> {
    let res = sqlx::query(
        "UPDATE invites SET revoked_at = now() WHERE label = $1 AND revoked_at IS NULL",
    )
    .bind(label)
    .execute(pool)
    .await?;
    Ok(res.rows_affected())
}

/// Operator listing. Never shows a code — we do not hold the plaintext.
pub async fn list(pool: &PgPool) -> Result<Vec<(String, String, String, i32)>> {
    let rows = sqlx::query(
        "SELECT label, round_slug, \
                CASE WHEN revoked_at IS NOT NULL THEN 'revoked' \
                     WHEN expires_at IS NOT NULL AND expires_at < now() THEN 'expired' \
                     WHEN max_uses IS NOT NULL AND used_count >= max_uses THEN 'exhausted' \
                     ELSE 'live' END AS status, \
                used_count \
         FROM invites ORDER BY created_at DESC",
    )
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|r| {
            (
                r.get("label"),
                r.get("round_slug"),
                r.get("status"),
                r.get("used_count"),
            )
        })
        .collect())
}
