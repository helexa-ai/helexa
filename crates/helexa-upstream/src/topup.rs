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
