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
