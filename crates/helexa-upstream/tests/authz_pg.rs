//! Integration tests for the `/authz/v1` surface against a real Postgres,
//! driving the built axum app over HTTP. Gated on `UPSTREAM_TEST_DATABASE_URL`
//! (skips cleanly when unset, so CI stays green without a DB):
//!
//!   UPSTREAM_TEST_DATABASE_URL=postgres://helexa:helexa@localhost/helexa_test \
//!     cargo test -p helexa-upstream --test authz_pg

use helexa_upstream::config::{ClientToken, UpstreamConfig};
use helexa_upstream::crypto::sha256;
use helexa_upstream::db::connect_and_migrate;
use helexa_upstream::state::AppState;
use serde_json::{Value, json};
use sqlx::Executor;
use sqlx::Row;
use sqlx::postgres::PgPool;
use uuid::Uuid;

const CLIENT_TOKEN: &str = "test-operator-token";

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
        features: Default::default(),
    };
    config.client_auth.tokens.push(ClientToken {
        token: CLIENT_TOKEN.into(),
        operator_id: "op-test".into(),
    });

    let email = helexa_upstream::email::EmailSender::from_config(&config.email).unwrap();
    let state = AppState::new(pool.clone(), config, email);
    let app = helexa_upstream::build_app(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Some((format!("http://{addr}"), pool))
}

/// Seed an account with `total` allocation and an active key with raw value
/// `raw` (percent=100). Optionally deactivate the account. Returns
/// (account_id, key_id).
async fn seed_key(pool: &PgPool, total: i64, raw: &str, deactivated: bool) -> (Uuid, Uuid) {
    let user_id: Uuid = pool
        .fetch_one(
            sqlx::query(
                "INSERT INTO users (email, password_hash, email_verified) VALUES ($1,'x',true) RETURNING id",
            )
            .bind(format!("u-{}@t.local", Uuid::new_v4())),
        )
        .await
        .unwrap()
        .get("id");
    let status = if deactivated { "deactivated" } else { "active" };
    let account_id: Uuid = pool
        .fetch_one(
            sqlx::query(
                "INSERT INTO accounts (owner_user_id, allocation_total, status) VALUES ($1,$2,$3) RETURNING id",
            )
            .bind(user_id)
            .bind(total)
            .bind(status),
        )
        .await
        .unwrap()
        .get("id");
    let key_id: Uuid = pool
        .fetch_one(
            sqlx::query(
                "INSERT INTO api_keys (account_id, key_hash, key_prefix, limit_kind, limit_value) \
                 VALUES ($1,$2,'sk-test','percent',100) RETURNING id",
            )
            .bind(account_id)
            .bind(sha256(raw)),
        )
        .await
        .unwrap()
        .get("id");
    (account_id, key_id)
}

fn client() -> reqwest::Client {
    reqwest::Client::new()
}

async fn post(
    c: &reqwest::Client,
    url: String,
    body: Value,
    bearer: Option<&str>,
) -> reqwest::Response {
    let mut req = c.post(url).json(&body);
    if let Some(b) = bearer {
        req = req.bearer_auth(b);
    }
    req.send().await.unwrap()
}

#[tokio::test]
async fn resolve_reserve_settle_round_trip() {
    let Some((base, pool)) = spawn_or_skip("resolve_reserve_settle_round_trip").await else {
        return;
    };
    let raw = format!("sk-{}", Uuid::new_v4());
    let (account_id, key_id) = seed_key(&pool, 1000, &raw, false).await;
    let c = client();

    // resolve
    let r = post(
        &c,
        format!("{base}/authz/v1/resolve"),
        json!({"api_key": raw}),
        Some(CLIENT_TOKEN),
    )
    .await;
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["principal"]["account_id"], account_id.to_string());
    assert_eq!(body["principal"]["key_id"], key_id.to_string());
    assert_eq!(body["snapshot"]["hard_cap"], 1000);

    // reserve 400
    let r = post(
        &c,
        format!("{base}/authz/v1/reserve"),
        json!({"account_id": account_id, "key_id": key_id, "max_tokens": 400}),
        Some(CLIENT_TOKEN),
    )
    .await;
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    let rid = body["reservation_id"].as_i64().expect("granted");

    // settle 150
    let r = post(
        &c,
        format!("{base}/authz/v1/settle"),
        json!({"reservation_id": rid, "actual_tokens": 150}),
        Some(CLIENT_TOKEN),
    )
    .await;
    assert_eq!(r.status(), 204);

    // snapshot reflects spend
    let r = post(
        &c,
        format!("{base}/authz/v1/snapshot"),
        json!({"account_id": account_id, "key_id": key_id}),
        Some(CLIENT_TOKEN),
    )
    .await;
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["spent"], 150);
    assert_eq!(body["reserved"], 0);
}

#[tokio::test]
async fn over_cap_reserve_is_rejected_not_errored() {
    let Some((base, pool)) = spawn_or_skip("over_cap_reserve_is_rejected_not_errored").await else {
        return;
    };
    let raw = format!("sk-{}", Uuid::new_v4());
    let (account_id, key_id) = seed_key(&pool, 100, &raw, false).await;
    let c = client();
    let r = post(
        &c,
        format!("{base}/authz/v1/reserve"),
        json!({"account_id": account_id, "key_id": key_id, "max_tokens": 999}),
        Some(CLIENT_TOKEN),
    )
    .await;
    assert_eq!(r.status(), 200, "budget refusal is an authoritative 200");
    let body: Value = r.json().await.unwrap();
    assert!(body["reservation_id"].is_null());
    assert_eq!(body["rejected"]["kind"], "insufficient_quota");
    assert_eq!(body["rejected"]["available"], 100);
}

#[tokio::test]
async fn deactivated_account_resolves_as_invalid_no_clue() {
    let Some((base, pool)) = spawn_or_skip("deactivated_account_resolves_as_invalid_no_clue").await
    else {
        return;
    };
    let raw = format!("sk-{}", Uuid::new_v4());
    seed_key(&pool, 1000, &raw, true).await; // deactivated
    let c = client();
    let r = post(
        &c,
        format!("{base}/authz/v1/resolve"),
        json!({"api_key": raw}),
        Some(CLIENT_TOKEN),
    )
    .await;
    // Indistinguishable from an unknown key.
    assert_eq!(r.status(), 401);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["error"]["code"], "invalid_api_key");
}

#[tokio::test]
async fn missing_client_auth_is_401_before_db() {
    let Some((base, pool)) = spawn_or_skip("missing_client_auth_is_401_before_db").await else {
        return;
    };
    let raw = format!("sk-{}", Uuid::new_v4());
    seed_key(&pool, 1000, &raw, false).await;
    let c = client();
    // No bearer → rejected by client_auth.
    let r = post(
        &c,
        format!("{base}/authz/v1/resolve"),
        json!({"api_key": raw}),
        None,
    )
    .await;
    assert_eq!(r.status(), 401);
    // Wrong bearer → also rejected.
    let r = post(
        &c,
        format!("{base}/authz/v1/resolve"),
        json!({"api_key": raw}),
        Some("wrong"),
    )
    .await;
    assert_eq!(r.status(), 401);
}
