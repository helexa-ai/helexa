//! Integration tests for the `/web/v1` account API + the silent fingerprint
//! abuse policy, driving the built app over HTTP against a real Postgres.
//! Gated on `UPSTREAM_TEST_DATABASE_URL` (skips cleanly when unset).

use helexa_upstream::config::{ClientToken, UpstreamConfig};
use helexa_upstream::crypto::sha256;
use helexa_upstream::db::connect_and_migrate;
use helexa_upstream::email::EmailSender;
use helexa_upstream::state::AppState;
use serde_json::{Value, json};
use sqlx::Executor;
use sqlx::Row;
use sqlx::postgres::PgPool;

const CLIENT_TOKEN: &str = "web-test-operator-token";

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
        email: Default::default(), // Log transport
        features: Default::default(),
    };
    config.client_auth.tokens.push(ClientToken {
        token: CLIENT_TOKEN.into(),
        operator_id: "op-web".into(),
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

fn unique_email() -> String {
    format!("u-{}@test.local", uuid::Uuid::new_v4())
}

async fn post(url: String, body: Value, bearer: Option<&str>) -> reqwest::Response {
    let c = reqwest::Client::new();
    let mut req = c.post(url).json(&body);
    if let Some(b) = bearer {
        req = req.bearer_auth(b);
    }
    req.send().await.unwrap()
}

#[tokio::test]
async fn verify_endpoint_consumes_token_once() {
    let Some((base, pool)) = spawn_or_skip("verify_endpoint_consumes_token_once").await else {
        return;
    };
    let email = unique_email();
    // Register, then mint a verify token directly (the raw token is only in
    // the email; here we insert a known one to drive the endpoint).
    assert_eq!(
        post(
            format!("{base}/web/v1/register"),
            json!({"email": email, "password": "password123"}),
            None
        )
        .await
        .status(),
        202
    );
    let user_id: uuid::Uuid = pool
        .fetch_one(sqlx::query("SELECT id FROM users WHERE email = $1").bind(&email))
        .await
        .unwrap()
        .get("id");
    let raw = "verify-raw-token-xyz";
    pool.execute(
        sqlx::query(
            "INSERT INTO email_tokens (token_hash, user_id, kind, expires_at) \
             VALUES ($1, $2, 'verify', now() + interval '1 hour')",
        )
        .bind(sha256(raw))
        .bind(user_id),
    )
    .await
    .unwrap();

    assert_eq!(
        post(format!("{base}/web/v1/verify"), json!({"token": raw}), None)
            .await
            .status(),
        200
    );
    // Consumed → second attempt fails.
    assert_eq!(
        post(format!("{base}/web/v1/verify"), json!({"token": raw}), None)
            .await
            .status(),
        400
    );

    let verified: bool = pool
        .fetch_one(sqlx::query("SELECT email_verified FROM users WHERE id = $1").bind(user_id))
        .await
        .unwrap()
        .get("email_verified");
    assert!(verified);
}

#[tokio::test]
async fn account_lifecycle_and_key_resolves_then_archives() {
    let Some((base, pool)) =
        spawn_or_skip("account_lifecycle_and_key_resolves_then_archives").await
    else {
        return;
    };
    let email = unique_email();
    post(
        format!("{base}/web/v1/register"),
        json!({"email": email, "password": "password123"}),
        None,
    )
    .await;
    // Bypass the email step for the login/key portion.
    pool.execute(
        sqlx::query("UPDATE users SET email_verified = true WHERE email = $1").bind(&email),
    )
    .await
    .unwrap();

    // login → session JWT
    let r = post(
        format!("{base}/web/v1/login"),
        json!({"email": email, "password": "password123"}),
        None,
    )
    .await;
    assert_eq!(r.status(), 200);
    let token = r.json::<Value>().await.unwrap()["token"]
        .as_str()
        .unwrap()
        .to_string();

    // create key (raw shown once)
    let r = post(
        format!("{base}/web/v1/keys"),
        json!({"label": "laptop"}),
        Some(&token),
    )
    .await;
    assert_eq!(r.status(), 201);
    let body: Value = r.json().await.unwrap();
    let raw_key = body["key"].as_str().unwrap().to_string();
    let key_id = body["id"].as_str().unwrap().to_string();
    assert!(raw_key.starts_with("sk-helexa-"));

    // account balance reflects the free grant
    let r = reqwest::Client::new()
        .get(format!("{base}/web/v1/account"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.json::<Value>().await.unwrap()["allocation_total"],
        1_000_000
    );

    // list keys shows the prefix, never the raw secret
    let r = reqwest::Client::new()
        .get(format!("{base}/web/v1/keys"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    let listed = r.json::<Value>().await.unwrap();
    let k = &listed["keys"][0];
    assert_eq!(k["id"], key_id);
    assert!(k.get("key").is_none(), "raw secret never listed");
    assert!(k["prefix"].as_str().unwrap().starts_with("sk-helexa-"));

    // the key authorizes at the authz surface
    let r = post(
        format!("{base}/authz/v1/resolve"),
        json!({"api_key": raw_key}),
        Some(CLIENT_TOKEN),
    )
    .await;
    assert_eq!(r.status(), 200);

    // archive → the key no longer resolves
    let r = post(
        format!("{base}/web/v1/keys/{key_id}/archive"),
        json!({}),
        Some(&token),
    )
    .await;
    assert_eq!(r.status(), 204);
    let r = post(
        format!("{base}/authz/v1/resolve"),
        json!({"api_key": raw_key}),
        Some(CLIENT_TOKEN),
    )
    .await;
    assert_eq!(r.status(), 401);
}

#[tokio::test]
async fn fingerprint_abuse_silently_deactivates_all_no_clue() {
    let Some((base, pool)) =
        spawn_or_skip("fingerprint_abuse_silently_deactivates_all_no_clue").await
    else {
        return;
    };
    let fp = format!("fp-{}", uuid::Uuid::new_v4());

    // 5 registrations sharing one fingerprint — every one returns a normal 202.
    let mut emails = Vec::new();
    for _ in 0..5 {
        let email = unique_email();
        let r = post(
            format!("{base}/web/v1/register"),
            json!({"email": email, "password": "password123", "fingerprint": fp}),
            None,
        )
        .await;
        assert_eq!(r.status(), 202, "registration always looks successful");
        emails.push(email);
    }

    // Silent effect: all 5 accounts are deactivated + flagged.
    let (deactivated, flagged): (i64, i64) = {
        let row = pool
            .fetch_one(
                sqlx::query(
                    "SELECT \
                   count(*) FILTER (WHERE a.status = 'deactivated') AS d, \
                   count(*) FILTER (WHERE a.fingerprint_flagged) AS f \
                 FROM accounts a JOIN users u ON u.id = a.owner_user_id \
                 WHERE u.registration_fingerprint = $1",
                )
                .bind(&fp),
            )
            .await
            .unwrap();
        (row.get("d"), row.get("f"))
    };
    assert_eq!(deactivated, 5, "all sharing accounts silently deactivated");
    assert_eq!(flagged, 5);

    // No clue at the authz surface: a key on a deactivated account resolves
    // as an ordinary 401, indistinguishable from an unknown key.
    let acct: uuid::Uuid = pool
        .fetch_one(
            sqlx::query(
                "SELECT a.id FROM accounts a JOIN users u ON u.id = a.owner_user_id \
                 WHERE u.registration_fingerprint = $1 LIMIT 1",
            )
            .bind(&fp),
        )
        .await
        .unwrap()
        .get("id");
    let raw = "sk-helexa-deactivated-probe";
    pool.execute(
        sqlx::query(
            "INSERT INTO api_keys (account_id, key_hash, key_prefix) VALUES ($1, $2, 'sk-helexa-')",
        )
        .bind(acct)
        .bind(sha256(raw)),
    )
    .await
    .unwrap();
    let r = post(
        format!("{base}/authz/v1/resolve"),
        json!({"api_key": raw}),
        Some(CLIENT_TOKEN),
    )
    .await;
    assert_eq!(
        r.status(),
        401,
        "deactivated account's key looks like any invalid key"
    );
}

#[tokio::test]
async fn topup_redeem_raises_allocation_single_use() {
    let Some((base, pool)) = spawn_or_skip("topup_redeem_raises_allocation_single_use").await
    else {
        return;
    };
    let email = unique_email();
    post(
        format!("{base}/web/v1/register"),
        json!({"email": email, "password": "password123"}),
        None,
    )
    .await;
    pool.execute(
        sqlx::query("UPDATE users SET email_verified = true WHERE email = $1").bind(&email),
    )
    .await
    .unwrap();
    let token = post(
        format!("{base}/web/v1/login"),
        json!({"email": email, "password": "password123"}),
        None,
    )
    .await
    .json::<Value>()
    .await
    .unwrap()["token"]
        .as_str()
        .unwrap()
        .to_string();

    // Mint a code worth 500_000 (mint path used by the CLI/faucet).
    let codes = helexa_upstream::topup::mint(&pool, 500_000, 1, Some("test"))
        .await
        .unwrap();
    let code = &codes[0];

    // Redeem → allocation_total rises from the 1_000_000 free grant.
    let r = post(
        format!("{base}/web/v1/redeem"),
        json!({"code": code}),
        Some(&token),
    )
    .await;
    assert_eq!(r.status(), 200);
    assert_eq!(
        r.json::<Value>().await.unwrap()["allocation_total"],
        1_500_000
    );

    // Single-use: a second redemption fails generically (no oracle).
    let r = post(
        format!("{base}/web/v1/redeem"),
        json!({"code": code}),
        Some(&token),
    )
    .await;
    assert_eq!(r.status(), 400);

    // Unknown code: same generic 400.
    let r = post(
        format!("{base}/web/v1/redeem"),
        json!({"code": "helexa-topup-does-not-exist"}),
        Some(&token),
    )
    .await;
    assert_eq!(r.status(), 400);
}

#[tokio::test]
async fn topup_concurrent_double_redeem_one_winner() {
    let Some((base, pool)) = spawn_or_skip("topup_concurrent_double_redeem_one_winner").await
    else {
        return;
    };
    // Two verified accounts.
    let mut tokens = Vec::new();
    for _ in 0..2 {
        let email = unique_email();
        post(
            format!("{base}/web/v1/register"),
            json!({"email": email, "password": "password123"}),
            None,
        )
        .await;
        pool.execute(
            sqlx::query("UPDATE users SET email_verified = true WHERE email = $1").bind(&email),
        )
        .await
        .unwrap();
        let t = post(
            format!("{base}/web/v1/login"),
            json!({"email": email, "password": "password123"}),
            None,
        )
        .await
        .json::<Value>()
        .await
        .unwrap()["token"]
            .as_str()
            .unwrap()
            .to_string();
        tokens.push(t);
    }
    let code = helexa_upstream::topup::mint(&pool, 100, 1, None)
        .await
        .unwrap()
        .remove(0);

    // Both accounts race to redeem the same code; exactly one wins.
    let (a, b) = tokio::join!(
        post(
            format!("{base}/web/v1/redeem"),
            json!({"code": code}),
            Some(&tokens[0])
        ),
        post(
            format!("{base}/web/v1/redeem"),
            json!({"code": code}),
            Some(&tokens[1])
        ),
    );
    let wins = [a.status(), b.status()]
        .iter()
        .filter(|s| s.as_u16() == 200)
        .count();
    assert_eq!(wins, 1, "exactly one redemption wins the single-use code");
}
