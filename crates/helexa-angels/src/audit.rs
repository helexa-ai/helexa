//! The access record.
//!
//! This is the requirement the architecture was chosen to satisfy: knowing
//! which named person read which document, and which version of it.
//!
//! Because the portal is server-rendered, one document view is one server
//! request, so this table records reading rather than fetching. An SPA
//! against a JSON API could pull every document once and re-render them
//! offline all week; the log would show a single view. That difference is
//! why the confidentiality requirement drove the architecture rather than
//! just the routing.
//!
//! The email is denormalised alongside `user_id` on purpose: the record of
//! how confidential material was handled must survive the deletion of the
//! account that read it.

use sqlx::postgres::PgPool;
use uuid::Uuid;

pub struct Access<'a> {
    pub user_id: Option<Uuid>,
    pub user_email: Option<&'a str>,
    pub round_slug: Option<&'a str>,
    pub document_slug: Option<&'a str>,
    pub content_version: Option<&'a str>,
    pub kind: &'a str,
    pub ip: Option<&'a str>,
    pub user_agent: Option<&'a str>,
}

/// Record an access. Never fails a request: a page the investor is
/// entitled to read must not 500 because the audit insert had a hiccup.
/// A failure here is loud in the log instead.
pub async fn record(pool: &PgPool, a: Access<'_>) {
    let res = sqlx::query(
        "INSERT INTO access_log \
           (user_id, user_email, round_slug, document_slug, content_version, kind, ip, user_agent) \
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
    )
    .bind(a.user_id)
    .bind(a.user_email)
    .bind(a.round_slug)
    .bind(a.document_slug)
    .bind(a.content_version)
    .bind(a.kind)
    .bind(a.ip)
    .bind(a.user_agent)
    .execute(pool)
    .await;

    if let Err(e) = res {
        tracing::error!(error = %e, kind = a.kind, "FAILED TO RECORD ACCESS — audit gap");
    }
}

/// Who read what, most recent first — the operator's answer to "has
/// anyone actually looked at this?"
pub async fn recent(
    pool: &PgPool,
    round_slug: Option<&str>,
    limit: i64,
) -> Result<Vec<(String, String, String, String)>, sqlx::Error> {
    let rows = sqlx::query(
        "SELECT to_char(viewed_at, 'YYYY-MM-DD HH24:MI') AS at, \
                coalesce(user_email, '(deleted account)') AS who, \
                coalesce(round_slug, '-') AS round, \
                coalesce(document_slug, '-') AS doc \
         FROM access_log \
         WHERE ($1::text IS NULL OR round_slug = $1) AND kind <> 'denied' \
         ORDER BY viewed_at DESC LIMIT $2",
    )
    .bind(round_slug)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    use sqlx::Row;
    Ok(rows
        .into_iter()
        .map(|r| (r.get("at"), r.get("who"), r.get("round"), r.get("doc")))
        .collect())
}

/// Delete access records older than the retention window.
///
/// These rows identify people, so keeping them forever is neither
/// necessary nor defensible. The window is stated in the portal's privacy
/// note; the two must agree.
pub async fn prune(pool: &PgPool, months: i64) -> Result<u64, sqlx::Error> {
    let res =
        sqlx::query("DELETE FROM access_log WHERE viewed_at < now() - make_interval(months => $1)")
            .bind(months as i32)
            .execute(pool)
            .await?;
    Ok(res.rows_affected())
}
