//! Round entitlement.
//!
//! Access attaches to a **user**, never to a URL or a code. That is the
//! property the whole confidentiality model rests on: an unguessable
//! content link, once forwarded, grants anonymous access forever and
//! cannot be withdrawn; a grant names one person, is revocable, and leaves
//! a trail.

use crate::error::Result;
use serde::Serialize;
use sqlx::Row;
use sqlx::postgres::PgPool;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize)]
pub struct RoundSummary {
    pub slug: String,
    pub title: String,
    pub framing_label: String,
    pub status: String,
    pub state: String,
    pub granted_at: String,
}

/// Rounds this user may see. `pending` grants are included so an
/// awaiting-approval visitor is told that, rather than being shown an
/// empty portal that looks like a mistake.
pub async fn rounds_for_user(pool: &PgPool, user_id: Uuid) -> Result<Vec<RoundSummary>> {
    let rows = sqlx::query(
        "SELECT r.slug, r.title, r.framing_label, r.status, g.state, \
                to_char(g.granted_at, 'YYYY-MM-DD') AS granted_at \
         FROM grants g JOIN rounds r ON r.slug = g.round_slug \
         WHERE g.user_id = $1 AND g.state IN ('active', 'pending') \
           AND r.status <> 'draft' \
         ORDER BY g.granted_at DESC",
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| RoundSummary {
            slug: r.get("slug"),
            title: r.get("title"),
            framing_label: r.get("framing_label"),
            status: r.get("status"),
            state: r.get("state"),
            granted_at: r.get("granted_at"),
        })
        .collect())
}

/// Whether this user may read this round's documents right now.
///
/// Deliberately strict: the grant must be `active` (not `pending`, not
/// `revoked`) and the round must not be a draft. A draft round is one
/// whose content is still being written, and half-written material is
/// exactly what should not reach an investor.
pub async fn has_access(pool: &PgPool, user_id: Uuid, round_slug: &str) -> Result<bool> {
    let row = sqlx::query(
        "SELECT 1 AS ok FROM grants g JOIN rounds r ON r.slug = g.round_slug \
         WHERE g.user_id = $1 AND g.round_slug = $2 \
           AND g.state = 'active' AND r.status IN ('open', 'closed')",
    )
    .bind(user_id)
    .bind(round_slug)
    .fetch_optional(pool)
    .await?;
    Ok(row.is_some())
}

/// Create (or re-activate) a grant. Idempotent: opening the same invite
/// twice is ordinary behaviour, not an error.
///
/// A previously **revoked** grant is deliberately NOT resurrected by
/// re-using an invite — revocation is an operator decision, and a code
/// still circulating must not undo it.
pub async fn upsert(
    pool: &PgPool,
    user_id: Uuid,
    round_slug: &str,
    invite_id: Option<Uuid>,
    active: bool,
) -> Result<String> {
    let state = if active { "active" } else { "pending" };
    let row = sqlx::query(
        "INSERT INTO grants (user_id, round_slug, invite_id, state) \
         VALUES ($1, $2, $3, $4) \
         ON CONFLICT (user_id, round_slug) DO UPDATE \
           SET state = CASE WHEN grants.state = 'revoked' THEN 'revoked' \
                            ELSE EXCLUDED.state END \
         RETURNING state",
    )
    .bind(user_id)
    .bind(round_slug)
    .bind(invite_id)
    .bind(state)
    .fetch_one(pool)
    .await?;
    Ok(row.get("state"))
}

/// Withdraw one person's access.
pub async fn revoke(pool: &PgPool, email: &str, round_slug: &str) -> Result<u64> {
    let res = sqlx::query(
        "UPDATE grants SET state = 'revoked', revoked_at = now() \
         WHERE round_slug = $1 \
           AND user_id = (SELECT id FROM public.users WHERE email = $2)",
    )
    .bind(round_slug)
    .bind(email)
    .execute(pool)
    .await?;
    Ok(res.rows_affected())
}

/// Approve a pending grant (used when a round runs with `auto_grant`
/// disabled).
pub async fn approve(pool: &PgPool, email: &str, round_slug: &str, by: &str) -> Result<u64> {
    let res = sqlx::query(
        "UPDATE grants SET state = 'active', approved_by = $3 \
         WHERE round_slug = $1 AND state = 'pending' \
           AND user_id = (SELECT id FROM public.users WHERE email = $2)",
    )
    .bind(round_slug)
    .bind(email)
    .bind(by)
    .execute(pool)
    .await?;
    Ok(res.rows_affected())
}

/// Everyone holding a grant on a round, for the operator listing.
pub async fn holders(pool: &PgPool, round_slug: &str) -> Result<Vec<(String, String, String)>> {
    let rows = sqlx::query(
        "SELECT u.email::text AS email, g.state, \
                to_char(g.granted_at, 'YYYY-MM-DD HH24:MI') AS granted_at \
         FROM grants g JOIN public.users u ON u.id = g.user_id \
         WHERE g.round_slug = $1 ORDER BY g.granted_at",
    )
    .bind(round_slug)
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|r| (r.get("email"), r.get("state"), r.get("granted_at")))
        .collect())
}
