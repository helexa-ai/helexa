//! Single-use top-up codes (#B5) — the second half of the hybrid allocation
//! model. Each code grants `value` tokens to the account that redeems it,
//! raising `accounts.allocation_total`. Minting codes is operator/CLI side
//! (the future faucet bot calls the same `mint` path); redemption is a
//! `/web/v1` action.
//!
//! Security: only `sha256(code)` is stored. Redemption is **timing-safe and
//! single-use** — a conditional `UPDATE … WHERE redeemed_by IS NULL` does
//! the claim atomically (concurrent double-redeem → exactly one winner), and
//! a not-found code and an already-redeemed code return the **same** generic
//! failure with the same code path (no oracle for "valid but spent").

use crate::config_store::{self, defaults, keys};
use crate::crypto::{random_token, sha256};
use sqlx::Row;
use sqlx::postgres::PgPool;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum TopUpError {
    /// Code unknown OR already redeemed — deliberately indistinguishable.
    #[error("invalid or already-redeemed code")]
    Invalid,
    #[error(transparent)]
    Db(#[from] sqlx::Error),
}

/// Why a self-service top-up was refused. Each maps to a message the
/// account holder can act on — unlike redemption, there is nothing to
/// keep ambiguous here: the caller is authenticated and asking about
/// their own account.
#[derive(Debug, thiserror::Error)]
pub enum AutoTopUpError {
    #[error("self-service top-ups are currently disabled")]
    Disabled,
    #[error("allocation is only {used_pct}% used; top-ups unlock at {threshold_pct}%")]
    BelowThreshold { used_pct: i64, threshold_pct: i64 },
    #[error("this account has had all {max} self-service top-ups")]
    LimitReached { max: i64 },
    #[error("try again in {retry_after_secs}s")]
    Cooldown { retry_after_secs: i64 },
    #[error(transparent)]
    Db(#[from] sqlx::Error),
}

/// What a self-service top-up granted.
#[derive(Debug, serde::Serialize)]
pub struct AutoTopUp {
    /// The code that was issued and immediately redeemed — recorded so the
    /// grant is auditable in `top_up_codes` exactly like a minted one.
    pub code: String,
    pub value: i64,
    pub allocation_total: i64,
    /// Self-service top-ups this account has now used, and its ceiling.
    pub used_count: i64,
    pub max_count: i64,
}

/// Whether this account may request a top-up right now, and why not.
///
/// Read-only: the dashboard calls this to decide whether to offer the
/// button, and `auto_grant` re-checks it before granting, so a stale UI
/// cannot talk the server into an extra grant.
pub async fn auto_eligibility(
    pool: &PgPool,
    account_id: Uuid,
) -> Result<Result<(), AutoTopUpError>, sqlx::Error> {
    if !config_store::get_bool(pool, keys::TOPUP_AUTO_ENABLED, defaults::TOPUP_AUTO_ENABLED).await {
        return Ok(Err(AutoTopUpError::Disabled));
    }
    let threshold = config_store::get_i64(
        pool,
        keys::TOPUP_AUTO_THRESHOLD_PCT,
        defaults::TOPUP_AUTO_THRESHOLD_PCT,
    )
    .await;
    let max_count = config_store::get_i64(
        pool,
        keys::TOPUP_AUTO_MAX_PER_ACCOUNT,
        defaults::TOPUP_AUTO_MAX_PER_ACCOUNT,
    )
    .await;
    let cooldown = config_store::get_i64(
        pool,
        keys::TOPUP_AUTO_COOLDOWN_SECS,
        defaults::TOPUP_AUTO_COOLDOWN_SECS,
    )
    .await;

    let row = sqlx::query(
        "SELECT allocation_total, allocation_spent, allocation_reserved FROM accounts WHERE id = $1",
    )
    .bind(account_id)
    .fetch_one(pool)
    .await?;
    let total: i64 = row.get("allocation_total");
    let spent: i64 = row.get("allocation_spent");
    let reserved: i64 = row.get("allocation_reserved");

    // An account with no allocation at all is 100% used, not a divide by
    // zero — it is exactly the case self-service top-ups exist for.
    let used_pct = if total <= 0 {
        100
    } else {
        (((spent + reserved) as i128 * 100) / total as i128) as i64
    };
    if used_pct < threshold {
        return Ok(Err(AutoTopUpError::BelowThreshold {
            used_pct,
            threshold_pct: threshold,
        }));
    }

    let (used_count, last_at) = auto_history(pool, account_id).await?;
    if used_count >= max_count {
        return Ok(Err(AutoTopUpError::LimitReached { max: max_count }));
    }
    if let Some(last) = last_at {
        let elapsed = (chrono::Utc::now() - last).num_seconds();
        if elapsed < cooldown {
            return Ok(Err(AutoTopUpError::Cooldown {
                retry_after_secs: cooldown - elapsed,
            }));
        }
    }
    Ok(Ok(()))
}

/// How many self-service top-ups this account has had, and when the last
/// one landed.
async fn auto_history(
    pool: &PgPool,
    account_id: Uuid,
) -> Result<(i64, Option<chrono::DateTime<chrono::Utc>>), sqlx::Error> {
    let row = sqlx::query(
        "SELECT count(*) AS n, max(redeemed_at) AS last_at FROM top_up_codes \
         WHERE source = 'auto' AND redeemed_by = $1",
    )
    .bind(account_id)
    .fetch_one(pool)
    .await?;
    Ok((row.get("n"), row.get("last_at")))
}

/// Issue and immediately redeem a self-service top-up.
///
/// Eligibility is re-checked here rather than trusted from the caller, and
/// the whole grant is one transaction: the code row and the allocation
/// increase land together or not at all.
pub async fn auto_grant(pool: &PgPool, account_id: Uuid) -> Result<AutoTopUp, AutoTopUpError> {
    // Re-check rather than trust the caller: the dashboard's copy of
    // eligibility may be stale by the time the button is pressed.
    auto_eligibility(pool, account_id).await??;
    let value = config_store::get_i64(
        pool,
        keys::TOPUP_AUTO_GRANT_TOKENS,
        defaults::TOPUP_AUTO_GRANT_TOKENS,
    )
    .await;
    let max_count = config_store::get_i64(
        pool,
        keys::TOPUP_AUTO_MAX_PER_ACCOUNT,
        defaults::TOPUP_AUTO_MAX_PER_ACCOUNT,
    )
    .await;

    let raw = format!("helexa-topup-{}", random_token());
    let mut tx = pool.begin().await?;

    // Insert already-redeemed: this code is never handed out unredeemed,
    // so there is no window in which it could be claimed by anyone else.
    // The `WHERE NOT EXISTS` re-counts inside the transaction, so two
    // concurrent requests cannot both slip past the per-account ceiling.
    let inserted = sqlx::query(
        "INSERT INTO top_up_codes (code_hash, value, denomination, source, redeemed_by, redeemed_at) \
         SELECT $1, $2, 'auto', 'auto', $3, now() \
         WHERE (SELECT count(*) FROM top_up_codes WHERE source = 'auto' AND redeemed_by = $3) < $4 \
         RETURNING value",
    )
    .bind(sha256(&raw))
    .bind(value)
    .bind(account_id)
    .bind(max_count)
    .fetch_optional(&mut *tx)
    .await?;
    if inserted.is_none() {
        // Lost the race against a concurrent request for the last slot.
        return Err(AutoTopUpError::LimitReached { max: max_count });
    }

    let allocation_total: i64 = sqlx::query(
        "UPDATE accounts SET allocation_total = allocation_total + $1 WHERE id = $2 \
         RETURNING allocation_total",
    )
    .bind(value)
    .bind(account_id)
    .fetch_one(&mut *tx)
    .await?
    .get("allocation_total");
    tx.commit().await?;

    let (used_count, _) = auto_history(pool, account_id).await?;
    Ok(AutoTopUp {
        code: raw,
        value,
        allocation_total,
        used_count,
        max_count,
    })
}

/// Redeem `raw_code` for `account_id`, raising the account's
/// `allocation_total` by the code's value. Returns the new total.
pub async fn redeem(pool: &PgPool, account_id: Uuid, raw_code: &str) -> Result<i64, TopUpError> {
    let mut tx = pool.begin().await?;
    // Atomic single-use claim. `redeemed_by IS NULL` is the guarantee: under
    // concurrent redemption exactly one UPDATE touches the row.
    let claimed = sqlx::query(
        "UPDATE top_up_codes SET redeemed_by = $1, redeemed_at = now() \
         WHERE code_hash = $2 AND redeemed_by IS NULL RETURNING value",
    )
    .bind(account_id)
    .bind(sha256(raw_code))
    .fetch_optional(&mut *tx)
    .await?;
    let Some(row) = claimed else {
        // Not found or already redeemed — same path, same error.
        return Err(TopUpError::Invalid);
    };
    let value: i64 = row.get("value");
    let new_total: i64 = sqlx::query(
        "UPDATE accounts SET allocation_total = allocation_total + $1 WHERE id = $2 \
         RETURNING allocation_total",
    )
    .bind(value)
    .bind(account_id)
    .fetch_one(&mut *tx)
    .await?
    .get("allocation_total");
    tx.commit().await?;
    Ok(new_total)
}

/// Mint `count` codes each worth `value` tokens, optionally tagged with a
/// `denomination` label. Returns the raw codes (shown once — only their
/// hash is stored). The CLI prints these; the future faucet bot calls this.
pub async fn mint(
    pool: &PgPool,
    value: i64,
    count: u32,
    denomination: Option<&str>,
) -> Result<Vec<String>, sqlx::Error> {
    let mut codes = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let raw = format!("helexa-topup-{}", random_token());
        sqlx::query(
            "INSERT INTO top_up_codes (code_hash, value, denomination) VALUES ($1, $2, $3)",
        )
        .bind(sha256(&raw))
        .bind(value)
        .bind(denomination)
        .execute(pool)
        .await?;
        codes.push(raw);
    }
    Ok(codes)
}
