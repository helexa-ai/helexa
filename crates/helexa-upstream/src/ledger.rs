//! The allocation ledger: reserve → settle/release with the no-overshoot
//! guarantee enforced by a row-locked transaction.
//!
//! Each reserve takes `SELECT … FOR UPDATE` on the account (and key) row, so
//! concurrent reserves from many cortexes serialize and `spent + reserved`
//! can never exceed the effective cap. The `accounts_no_overshoot` CHECK is
//! the DB-level backstop. Settle/release are idempotent (they only act on a
//! reservation still in `open`).
//!
//! Per-key effective cap = `min(resolved key cap, remaining account
//! allocation)`. The key cap is resolved from its `limit_kind`:
//! `hardcap` → the value verbatim; `percent` → that % of the account's
//! `allocation_total`.
//!
//! Cap-window semantics: this module implements **Balance** (non-resetting)
//! caps. Rolling-window key sub-caps (and the `RateLimited` rejection that
//! rides them) land with the authz API (B2); today an over-cap is always
//! `InsufficientQuota`.

use sqlx::postgres::PgPool;
use uuid::Uuid;

/// A bearer key resolved to its principal + a budget snapshot.
#[derive(Debug, Clone)]
pub struct ResolvedPrincipal {
    pub account_id: Uuid,
    pub key_id: Uuid,
    /// Effective per-key absolute cap (the key sub-cap; the account cap
    /// still binds at reserve time).
    pub hard_cap: i64,
    pub key_spent: i64,
    pub key_reserved: i64,
}

/// Resolve a key by its `sha256` hash to its principal, or `None` when the
/// key is unknown/archived **or its account is deactivated** (the silent
/// abuse flag — indistinguishable from an unknown key, by design: no clue).
pub async fn resolve_key(
    pool: &PgPool,
    key_hash: &[u8],
) -> Result<Option<ResolvedPrincipal>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT k.id AS key_id, k.account_id, k.limit_kind, k.limit_value, \
                k.key_spent, k.key_reserved, a.allocation_total \
         FROM api_keys k JOIN accounts a ON a.id = k.account_id \
         WHERE k.key_hash = $1 AND k.status = 'active' AND a.status = 'active'",
    )
    .bind(key_hash)
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|r| {
        let total: i64 = sqlx::Row::get(&r, "allocation_total");
        let limit_kind: String = sqlx::Row::get(&r, "limit_kind");
        let limit_value: i64 = sqlx::Row::get(&r, "limit_value");
        ResolvedPrincipal {
            account_id: sqlx::Row::get(&r, "account_id"),
            key_id: sqlx::Row::get(&r, "key_id"),
            hard_cap: resolve_abs_cap(&limit_kind, limit_value, total),
            key_spent: sqlx::Row::get(&r, "key_spent"),
            key_reserved: sqlx::Row::get(&r, "key_reserved"),
        }
    }))
}

/// Per-key budget snapshot `(hard_cap, spent, reserved)`, or `None` if the
/// key/account isn't an active pair.
pub async fn snapshot(
    pool: &PgPool,
    account_id: Uuid,
    key_id: Uuid,
) -> Result<Option<(i64, i64, i64)>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT k.limit_kind, k.limit_value, k.key_spent, k.key_reserved, a.allocation_total \
         FROM api_keys k JOIN accounts a ON a.id = k.account_id \
         WHERE k.id = $1 AND k.account_id = $2 AND k.status = 'active' AND a.status = 'active'",
    )
    .bind(key_id)
    .bind(account_id)
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|r| {
        let total: i64 = sqlx::Row::get(&r, "allocation_total");
        let limit_kind: String = sqlx::Row::get(&r, "limit_kind");
        let limit_value: i64 = sqlx::Row::get(&r, "limit_value");
        let cap = resolve_abs_cap(&limit_kind, limit_value, total);
        (
            cap,
            sqlx::Row::get::<i64, _>(&r, "key_spent"),
            sqlx::Row::get::<i64, _>(&r, "key_reserved"),
        )
    }))
}

/// Release every `open` reservation older than `max_age_secs`, returning
/// each one's reserved tokens to its account and key in a single statement.
/// The lost-settle self-heal. Returns the number swept.
pub async fn sweep_stale(pool: &PgPool, max_age_secs: i64) -> Result<u64, sqlx::Error> {
    // Data-modifying CTEs: release stale rows, then fold their reserved sums
    // back into accounts and api_keys. All in one atomic statement.
    let result = sqlx::query(
        "WITH stale AS ( \
             UPDATE reservations SET state = 'released', settled_at = now() \
             WHERE state = 'open' AND created_at < now() - make_interval(secs => $1) \
             RETURNING account_id, key_id, reserved \
         ), acct AS ( \
             UPDATE accounts a SET allocation_reserved = allocation_reserved - s.total \
             FROM (SELECT account_id, SUM(reserved) AS total FROM stale GROUP BY account_id) s \
             WHERE a.id = s.account_id \
         ) \
         UPDATE api_keys k SET key_reserved = key_reserved - s.total \
         FROM (SELECT key_id, SUM(reserved) AS total FROM stale GROUP BY key_id) s \
         WHERE k.id = s.key_id",
    )
    .bind(max_age_secs as f64)
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}

/// Resolve a key's per-key cap to an absolute token count.
///
/// `percent` is `floor(allocation_total * limit_value / 100)`; `hardcap` is
/// `limit_value` verbatim. Computed in i128 to avoid overflow, floored at 0.
pub fn resolve_abs_cap(limit_kind: &str, limit_value: i64, allocation_total: i64) -> i64 {
    let cap = match limit_kind {
        "percent" => (allocation_total as i128 * limit_value as i128) / 100,
        _ => limit_value as i128, // "hardcap" (and any unknown → treat as absolute)
    };
    cap.clamp(0, i64::MAX as i128) as i64
}

#[derive(Debug, thiserror::Error)]
pub enum LedgerError {
    #[error("account not found")]
    AccountNotFound,
    #[error("api key not found or not active")]
    KeyNotFound,
    /// Account balance or a Balance-window key sub-cap is exhausted.
    #[error("insufficient quota: requested {requested}, available {available}")]
    InsufficientQuota { requested: i64, available: i64 },
    #[error(transparent)]
    Db(#[from] sqlx::Error),
}

/// Reserve `max_tokens` against `account_id`/`key_id`. Returns the
/// reservation id (the `BIGSERIAL`, mapped to the cortex `Reservation.id`).
pub async fn reserve(
    pool: &PgPool,
    account_id: Uuid,
    key_id: Uuid,
    max_tokens: i64,
) -> Result<i64, LedgerError> {
    let mut tx = pool.begin().await?;

    // Lock the account row — serializes concurrent reserves on this account.
    let acct = sqlx::query(
        "SELECT allocation_total, allocation_spent, allocation_reserved \
         FROM accounts WHERE id = $1 AND status = 'active' FOR UPDATE",
    )
    .bind(account_id)
    .fetch_optional(&mut *tx)
    .await?;
    let Some(acct) = acct else {
        return Err(LedgerError::AccountNotFound);
    };
    let total: i64 = sqlx::Row::get(&acct, "allocation_total");
    let spent: i64 = sqlx::Row::get(&acct, "allocation_spent");
    let reserved: i64 = sqlx::Row::get(&acct, "allocation_reserved");
    let account_avail = total - spent - reserved;

    // Lock the key row and resolve its absolute sub-cap.
    let key = sqlx::query(
        "SELECT limit_kind, limit_value, key_spent, key_reserved \
         FROM api_keys WHERE id = $1 AND account_id = $2 AND status = 'active' FOR UPDATE",
    )
    .bind(key_id)
    .bind(account_id)
    .fetch_optional(&mut *tx)
    .await?;
    let Some(key) = key else {
        return Err(LedgerError::KeyNotFound);
    };
    let limit_kind: String = sqlx::Row::get(&key, "limit_kind");
    let limit_value: i64 = sqlx::Row::get(&key, "limit_value");
    let key_spent: i64 = sqlx::Row::get(&key, "key_spent");
    let key_reserved: i64 = sqlx::Row::get(&key, "key_reserved");
    let key_cap = resolve_abs_cap(&limit_kind, limit_value, total);
    let key_avail = key_cap - key_spent - key_reserved;

    let available = account_avail.min(key_avail).max(0);
    if max_tokens > available {
        // tx rolls back on drop
        return Err(LedgerError::InsufficientQuota {
            requested: max_tokens,
            available,
        });
    }

    let id: i64 = sqlx::Row::get(
        &sqlx::query(
            "INSERT INTO reservations (account_id, key_id, reserved, state) \
             VALUES ($1, $2, $3, 'open') RETURNING id",
        )
        .bind(account_id)
        .bind(key_id)
        .bind(max_tokens)
        .fetch_one(&mut *tx)
        .await?,
        "id",
    );
    sqlx::query("UPDATE accounts SET allocation_reserved = allocation_reserved + $1 WHERE id = $2")
        .bind(max_tokens)
        .bind(account_id)
        .execute(&mut *tx)
        .await?;
    sqlx::query("UPDATE api_keys SET key_reserved = key_reserved + $1 WHERE id = $2")
        .bind(max_tokens)
        .bind(key_id)
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;
    Ok(id)
}

/// Settle a reservation with the actual tokens used (clamped to
/// `[0, reserved]`). Idempotent: a second settle (or settle after release)
/// is a no-op.
pub async fn settle(
    pool: &PgPool,
    reservation_id: i64,
    actual_tokens: i64,
) -> Result<(), LedgerError> {
    let mut tx = pool.begin().await?;
    let row = sqlx::query(
        "UPDATE reservations SET state = 'settled', settled_at = now(), \
         actual = LEAST(GREATEST($2, 0), reserved) \
         WHERE id = $1 AND state = 'open' \
         RETURNING reserved, account_id, key_id, actual",
    )
    .bind(reservation_id)
    .bind(actual_tokens)
    .fetch_optional(&mut *tx)
    .await?;
    let Some(row) = row else {
        return Ok(()); // already settled/released, or unknown → idempotent no-op
    };
    let reserved: i64 = sqlx::Row::get(&row, "reserved");
    let actual: i64 = sqlx::Row::get(&row, "actual");
    let account_id: Uuid = sqlx::Row::get(&row, "account_id");
    let key_id: Uuid = sqlx::Row::get(&row, "key_id");

    sqlx::query(
        "UPDATE accounts SET allocation_reserved = allocation_reserved - $1, \
         allocation_spent = allocation_spent + $2 WHERE id = $3",
    )
    .bind(reserved)
    .bind(actual)
    .bind(account_id)
    .execute(&mut *tx)
    .await?;
    sqlx::query(
        "UPDATE api_keys SET key_reserved = key_reserved - $1, key_spent = key_spent + $2 WHERE id = $3",
    )
    .bind(reserved)
    .bind(actual)
    .bind(key_id)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;
    Ok(())
}

/// Release a reservation, returning its full reserved amount to the
/// allocation. Idempotent.
pub async fn release(pool: &PgPool, reservation_id: i64) -> Result<(), LedgerError> {
    let mut tx = pool.begin().await?;
    let row = sqlx::query(
        "UPDATE reservations SET state = 'released', settled_at = now() \
         WHERE id = $1 AND state = 'open' \
         RETURNING reserved, account_id, key_id",
    )
    .bind(reservation_id)
    .fetch_optional(&mut *tx)
    .await?;
    let Some(row) = row else {
        return Ok(());
    };
    let reserved: i64 = sqlx::Row::get(&row, "reserved");
    let account_id: Uuid = sqlx::Row::get(&row, "account_id");
    let key_id: Uuid = sqlx::Row::get(&row, "key_id");

    sqlx::query("UPDATE accounts SET allocation_reserved = allocation_reserved - $1 WHERE id = $2")
        .bind(reserved)
        .bind(account_id)
        .execute(&mut *tx)
        .await?;
    sqlx::query("UPDATE api_keys SET key_reserved = key_reserved - $1 WHERE id = $2")
        .bind(reserved)
        .bind(key_id)
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::resolve_abs_cap;

    #[test]
    fn hardcap_is_verbatim() {
        assert_eq!(resolve_abs_cap("hardcap", 50_000, 1_000_000), 50_000);
    }

    #[test]
    fn percent_is_fraction_of_allocation() {
        assert_eq!(resolve_abs_cap("percent", 25, 1_000_000), 250_000);
        assert_eq!(resolve_abs_cap("percent", 100, 1_000_000), 1_000_000);
        // floor
        assert_eq!(resolve_abs_cap("percent", 33, 10), 3);
    }

    #[test]
    fn percent_does_not_overflow_on_large_allocation() {
        // total * value would overflow i64 if not widened to i128.
        let cap = resolve_abs_cap("percent", 100, i64::MAX);
        assert_eq!(cap, i64::MAX);
    }

    #[test]
    fn negative_or_zero_clamps_to_zero() {
        assert_eq!(resolve_abs_cap("hardcap", -5, 100), 0);
        assert_eq!(resolve_abs_cap("percent", 0, 1_000_000), 0);
    }
}
