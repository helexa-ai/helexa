//! B3: the chained entitlement provider (local → upstream) and fail-closed
//! semantics, exercised against a mock helexa-upstream `/authz/v1`.

use axum::{Json, Router, routing::post};
use cortex_core::config::{ApiKeyConfig, EntitlementsConfig, UpstreamClientConfig};
use cortex_core::entitlements::{AuthError, EntitlementProvider};
use cortex_gateway::entitlements_chain::ChainedEntitlementProvider;
use cortex_gateway::entitlements_local::LocalEntitlementProvider;
use cortex_gateway::entitlements_upstream::UpstreamEntitlementProvider;
use serde_json::{Value, json};
use tokio::net::TcpListener;

/// Mock upstream: `mesh-key` resolves to a mesh account; anything else 401.
/// reserve always grants reservation 1.
async fn spawn_mock_upstream() -> String {
    async fn resolve(Json(body): Json<Value>) -> axum::response::Response {
        use axum::response::IntoResponse;
        if body["api_key"] == "mesh-key" {
            Json(json!({"principal": {"account_id": "mesh-acct", "key_id": "mesh-key-1"}}))
                .into_response()
        } else {
            (
                axum::http::StatusCode::UNAUTHORIZED,
                Json(json!({"error": {"code": "invalid_api_key"}})),
            )
                .into_response()
        }
    }
    async fn reserve() -> Json<Value> {
        Json(json!({ "reservation_id": 1 }))
    }
    let app = Router::new()
        .route("/authz/v1/resolve", post(resolve))
        .route("/authz/v1/reserve", post(reserve));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

fn local_with_key() -> LocalEntitlementProvider {
    let cfg = EntitlementsConfig {
        require_auth: false,
        keys: vec![ApiKeyConfig {
            key: "local-key".into(),
            account_id: "op".into(),
            key_id: None,
            hard_cap: None,
            window: Default::default(),
        }],
    };
    LocalEntitlementProvider::from_config(&cfg)
}

fn chain(local: LocalEntitlementProvider, url: &str) -> ChainedEntitlementProvider {
    let upstream = UpstreamEntitlementProvider::new(&UpstreamClientConfig {
        enabled: true,
        url: url.to_string(),
        bearer: "client-secret".into(),
        timeout_secs: 5,
        served_usage_report_interval_secs: 60,
    });
    ChainedEntitlementProvider::new(local, upstream)
}

#[tokio::test]
async fn local_key_resolves_locally() {
    let url = spawn_mock_upstream().await;
    let c = chain(local_with_key(), &url);
    let p = c.resolve("local-key").await.expect("local resolves");
    assert_eq!(p.account_id, "op");
}

#[tokio::test]
async fn unknown_key_falls_through_to_upstream() {
    let url = spawn_mock_upstream().await;
    let c = chain(local_with_key(), &url);
    let p = c.resolve("mesh-key").await.expect("upstream resolves");
    assert_eq!(p.account_id, "mesh-acct");
    assert_eq!(p.key_id, "mesh-key-1");
}

#[tokio::test]
async fn unknown_everywhere_is_invalid_key() {
    let url = spawn_mock_upstream().await;
    let c = chain(local_with_key(), &url);
    match c.resolve("nope").await {
        Err(AuthError::InvalidKey) => {}
        other => panic!("expected InvalidKey, got {other:?}"),
    }
}

#[tokio::test]
async fn upstream_unreachable_fails_closed_as_unavailable() {
    // No mock — point at a dead port. A locally-unknown key must surface
    // Unavailable (→ 503), never InvalidKey (→ 401).
    let c = chain(local_with_key(), "http://127.0.0.1:1");
    match c.resolve("some-mesh-key").await {
        Err(AuthError::Unavailable { retry_after_secs }) => assert!(retry_after_secs > 0),
        other => panic!("expected Unavailable, got {other:?}"),
    }
    // A local key still resolves without touching upstream.
    assert_eq!(c.resolve("local-key").await.unwrap().account_id, "op");
}
