//! Integration test for the served-usage report (#58): the idempotent,
//! monotonic upsert and the reconcile rollup. Gated on
//! UPSTREAM_TEST_DATABASE_URL (skips cleanly when unset).

use helexa_upstream::config::{ClientToken, UpstreamConfig};
use helexa_upstream::db::connect_and_migrate;
use helexa_upstream::email::EmailSender;
use helexa_upstream::reconcile::reconcile;
use helexa_upstream::state::AppState;
use serde_json::{Value, json};
use sqlx::Row;
use sqlx::postgres::PgPool;
use uuid::Uuid;

const CLIENT_TOKEN: &str = "su-test-token";
const OPERATOR: &str = "op-su-test";

async fn spawn_or_skip(test: &str) -> Option<(String, PgPool)> {
    let Ok(url) = std::env::var("UPSTREAM_TEST_DATABASE_URL") else {
        eprintln!("skipping {test}: UPSTREAM_TEST_DATABASE_URL not set");
        return None;
    };
    let pool = connect_and_migrate(&url, 16).await.expect("migrate");
    let mut config = UpstreamConfig {
        server: Default::default(),
        db: helexa_upstream::config::DbSettings {
            url,
            max_connections: 16,
        },
        grant: Default::default(),
        abuse: Default::default(),
        client_auth: Default::default(),
        authz: Default::default(),
        auth: Default::default(),
        email: Default::default(),
    };
    config.client_auth.tokens.push(ClientToken {
        token: CLIENT_TOKEN.into(),
        operator_id: OPERATOR.into(),
    });
    let email = EmailSender::from_config(&config.email).unwrap();
    let state = AppState::new(pool.clone(), config, email);
    let app = helexa_upstream::build_app(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Some((format!("http://{addr}"), pool))
}

async fn report(base: &str, rows: Value) -> u16 {
    reqwest::Client::new()
        .post(format!("{base}/authz/v1/served-usage"))
        .bearer_auth(CLIENT_TOKEN)
        .json(&json!({ "rows": rows }))
        .send()
        .await
        .unwrap()
        .status()
        .as_u16()
}

async fn stored(pool: &PgPool, account: Uuid, key: Uuid) -> i64 {
    sqlx::query(
        "SELECT served_tokens FROM served_usage WHERE operator_id = $1 AND account_id = $2 AND key_id = $3",
    )
    .bind(OPERATOR)
    .bind(account)
    .bind(key)
    .fetch_one(pool)
    .await
    .unwrap()
    .get("served_tokens")
}

#[tokio::test]
async fn served_usage_upsert_is_monotonic_and_reconciles() {
    let Some((base, pool)) = spawn_or_skip("served_usage_upsert_is_monotonic_and_reconciles").await
    else {
        return;
    };
    let account = Uuid::new_v4();
    let key = Uuid::new_v4();
    let period = "2026-06-23";
    let row = |n: i64| json!([{"account_id": account, "key_id": key, "period": period, "served_tokens": n}]);

    // First report.
    assert_eq!(report(&base, row(100)).await, 204);
    assert_eq!(stored(&pool, account, key).await, 100);

    // Re-send a higher absolute value → advances.
    assert_eq!(report(&base, row(250)).await, 204);
    assert_eq!(stored(&pool, account, key).await, 250);

    // A lower value (e.g. a restarted cortex) must NOT regress (GREATEST).
    assert_eq!(report(&base, row(50)).await, 204);
    assert_eq!(stored(&pool, account, key).await, 250);

    // Re-sending the same value is idempotent.
    assert_eq!(report(&base, row(250)).await, 204);
    assert_eq!(stored(&pool, account, key).await, 250);

    // Reconcile rolls it up and stamps reconciled_at; a second run is empty.
    let rollup = reconcile(&pool).await.unwrap();
    let mine = rollup
        .iter()
        .find(|r| r.operator_id == OPERATOR)
        .expect("operator in rollup");
    assert!(mine.total_served_tokens >= 250);
    let again = reconcile(&pool).await.unwrap();
    assert!(
        again.iter().all(|r| r.operator_id != OPERATOR),
        "already reconciled"
    );
}
