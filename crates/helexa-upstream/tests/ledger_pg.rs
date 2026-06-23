//! Integration tests for the allocation ledger against a real PostgreSQL.
//!
//! Gated on `UPSTREAM_TEST_DATABASE_URL` — when unset (CI's generic runner,
//! local builds without a DB), every test logs a skip and returns, so
//! `cargo test --workspace` stays green without Postgres. Point the env var
//! at a throwaway database to exercise the no-overshoot guarantee and
//! settle/release idempotency:
//!
//!   UPSTREAM_TEST_DATABASE_URL=postgres://helexa:helexa@localhost/helexa_test \
//!     cargo test -p helexa-upstream --test ledger_pg

use helexa_upstream::db::connect_and_migrate;
use helexa_upstream::ledger::{self, LedgerError};
use sqlx::Executor;
use sqlx::Row;
use sqlx::postgres::PgPool;
use uuid::Uuid;

/// Returns a migrated pool, or `None` (with a skip log) when the env var is
/// unset.
async fn pool_or_skip(test: &str) -> Option<PgPool> {
    let Ok(url) = std::env::var("UPSTREAM_TEST_DATABASE_URL") else {
        eprintln!("skipping {test}: UPSTREAM_TEST_DATABASE_URL not set");
        return None;
    };
    Some(
        connect_and_migrate(&url, 16)
            .await
            .expect("connect + migrate"),
    )
}

/// Seed a verified user + account (with `total` allocation) + an active key
/// (percent=100 so the account cap binds). Returns (account_id, key_id).
async fn seed(pool: &PgPool, total: i64) -> (Uuid, Uuid) {
    let user_id: Uuid = pool
        .fetch_one(
            sqlx::query(
                "INSERT INTO users (email, password_hash, email_verified) \
                 VALUES ($1, 'x', true) RETURNING id",
            )
            .bind(format!("u-{}@test.local", Uuid::new_v4())),
        )
        .await
        .unwrap()
        .get("id");
    let account_id: Uuid = pool
        .fetch_one(
            sqlx::query(
                "INSERT INTO accounts (owner_user_id, allocation_total) \
                 VALUES ($1, $2) RETURNING id",
            )
            .bind(user_id)
            .bind(total),
        )
        .await
        .unwrap()
        .get("id");
    let key_id: Uuid = pool
        .fetch_one(
            sqlx::query(
                "INSERT INTO api_keys (account_id, key_hash, key_prefix, limit_kind, limit_value) \
                 VALUES ($1, $2, 'sk-test', 'percent', 100) RETURNING id",
            )
            .bind(account_id)
            .bind(Uuid::new_v4().as_bytes().to_vec()),
        )
        .await
        .unwrap()
        .get("id");
    (account_id, key_id)
}

async fn account_cols(pool: &PgPool, account_id: Uuid) -> (i64, i64) {
    let row = pool
        .fetch_one(
            sqlx::query("SELECT allocation_spent, allocation_reserved FROM accounts WHERE id = $1")
                .bind(account_id),
        )
        .await
        .unwrap();
    (row.get("allocation_spent"), row.get("allocation_reserved"))
}

#[tokio::test]
async fn concurrent_reserves_never_overshoot() {
    let Some(pool) = pool_or_skip("concurrent_reserves_never_overshoot").await else {
        return;
    };
    // Allocation admits exactly 5 reservations of 100 (cap 500).
    let (account_id, key_id) = seed(&pool, 500).await;

    let mut handles = Vec::new();
    for _ in 0..20 {
        let pool = pool.clone();
        handles.push(tokio::spawn(async move {
            ledger::reserve(&pool, account_id, key_id, 100).await
        }));
    }
    let mut ok = 0;
    let mut quota = 0;
    for h in handles {
        match h.await.unwrap() {
            Ok(_) => ok += 1,
            Err(LedgerError::InsufficientQuota { .. }) => quota += 1,
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
    assert_eq!(ok, 5, "exactly 5 reserves of 100 fit in a 500 allocation");
    assert_eq!(quota, 15);

    let (spent, reserved) = account_cols(&pool, account_id).await;
    assert_eq!(spent, 0);
    assert_eq!(reserved, 500, "reserved exactly the cap, never over");
}

#[tokio::test]
async fn settle_is_idempotent_and_reconciles_spend() {
    let Some(pool) = pool_or_skip("settle_is_idempotent_and_reconciles_spend").await else {
        return;
    };
    let (account_id, key_id) = seed(&pool, 1000).await;
    let rid = ledger::reserve(&pool, account_id, key_id, 400)
        .await
        .unwrap();

    // Settle actual=150 (< reserved 400): spent=150, reserved back to 0.
    ledger::settle(&pool, rid, 150).await.unwrap();
    let (spent, reserved) = account_cols(&pool, account_id).await;
    assert_eq!((spent, reserved), (150, 0));

    // Second settle is a no-op.
    ledger::settle(&pool, rid, 999).await.unwrap();
    let (spent2, reserved2) = account_cols(&pool, account_id).await;
    assert_eq!((spent2, reserved2), (150, 0), "settle is idempotent");
}

#[tokio::test]
async fn release_returns_reservation_and_is_idempotent() {
    let Some(pool) = pool_or_skip("release_returns_reservation_and_is_idempotent").await else {
        return;
    };
    let (account_id, key_id) = seed(&pool, 1000).await;
    let rid = ledger::reserve(&pool, account_id, key_id, 300)
        .await
        .unwrap();
    assert_eq!(account_cols(&pool, account_id).await, (0, 300));

    ledger::release(&pool, rid).await.unwrap();
    assert_eq!(account_cols(&pool, account_id).await, (0, 0));
    // Idempotent; settle-after-release also a no-op.
    ledger::release(&pool, rid).await.unwrap();
    ledger::settle(&pool, rid, 100).await.unwrap();
    assert_eq!(account_cols(&pool, account_id).await, (0, 0));
}

#[tokio::test]
async fn hardcap_key_subcap_binds_below_account() {
    let Some(pool) = pool_or_skip("hardcap_key_subcap_binds_below_account").await else {
        return;
    };
    // Account has 1000 but the key is hard-capped at 200.
    let user_id: Uuid = pool
        .fetch_one(
            sqlx::query(
                "INSERT INTO users (email, password_hash, email_verified) \
                 VALUES ($1, 'x', true) RETURNING id",
            )
            .bind(format!("u-{}@test.local", Uuid::new_v4())),
        )
        .await
        .unwrap()
        .get("id");
    let account_id: Uuid = pool
        .fetch_one(
            sqlx::query(
                "INSERT INTO accounts (owner_user_id, allocation_total) VALUES ($1, 1000) RETURNING id",
            )
            .bind(user_id),
        )
        .await
        .unwrap()
        .get("id");
    let key_id: Uuid = pool
        .fetch_one(
            sqlx::query(
                "INSERT INTO api_keys (account_id, key_hash, key_prefix, limit_kind, limit_value) \
                 VALUES ($1, $2, 'sk-test', 'hardcap', 200) RETURNING id",
            )
            .bind(account_id)
            .bind(Uuid::new_v4().as_bytes().to_vec()),
        )
        .await
        .unwrap()
        .get("id");

    ledger::reserve(&pool, account_id, key_id, 200)
        .await
        .unwrap();
    match ledger::reserve(&pool, account_id, key_id, 1).await {
        Err(LedgerError::InsufficientQuota { available, .. }) => assert_eq!(available, 0),
        other => panic!("expected InsufficientQuota, got {other:?}"),
    }
}
