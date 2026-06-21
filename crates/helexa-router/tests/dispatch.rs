//! Capacity-aware dispatch acceptance tests for #73.
//!
//! Covers: a request routes to a cortex serving the model; the client's
//! bearer reaches the cortex; cortex's #63 rejections pass through verbatim
//! and are NOT retried away; transport failure fails over to another
//! feasible cortex; unknown model → 404, no reachable capacity → 503; and
//! the selection ranking (warm/region/headroom).

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use helexa_router::config::{CortexEndpoint, RouterConfig};
use helexa_router::dispatch::{Selection, dispatch, select_cortexes};
use helexa_router::state::{CortexTopology, RouterModelStatus, RouterState};
use serde_json::{Value, json};
use std::collections::HashMap;
use tokio::net::TcpListener;

const MODEL: &str = "Qwen/Qwen3-Coder-30B";

// ── Mock cortex backend ──────────────────────────────────────────────

/// Behaviour of a mock cortex, carried in axum State.
#[derive(Clone)]
struct MockCortex {
    /// Identifies which cortex answered, echoed in the 200 body.
    name: &'static str,
    /// When true, return a genuine #63-shaped `429 rate_limit_exceeded`.
    rate_limited: bool,
}

async fn mock_handler(State(m): State<MockCortex>, headers: HeaderMap) -> Response {
    if m.rate_limited {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"error":{"type":"rate_limit_error","code":"rate_limit_exceeded","message":"slow down","param":null}})),
        )
            .into_response();
    }
    let auth = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    Json(json!({ "served_by": m.name, "auth_seen": auth })).into_response()
}

async fn spawn_cortex(mock: MockCortex) -> String {
    let app = Router::new()
        .route("/v1/chat/completions", post(mock_handler))
        .with_state(mock);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

fn ok_cortex(name: &'static str) -> MockCortex {
    MockCortex {
        name,
        rate_limited: false,
    }
}

// ── Helpers to build state with a hand-set topology ──────────────────

fn state_with(cortexes: Vec<CortexEndpoint>, region: Option<String>) -> RouterState {
    let cfg = RouterConfig {
        cortexes,
        ..Default::default()
    };
    let mut state = RouterState::from_config(&cfg);
    state.region = region;
    state
}

/// Overwrite the topology entry for `name` so tests control reachability and
/// model serveability directly (no live poll).
async fn set_topology(
    state: &RouterState,
    name: &str,
    reachable: bool,
    loaded: bool,
    feasible: bool,
    healthy_nodes: u32,
) {
    let mut topo = state.topology.write().await;
    let mut models = HashMap::new();
    models.insert(MODEL.to_string(), RouterModelStatus { loaded, feasible });
    topo.insert(
        name.to_string(),
        CortexTopology {
            reachable,
            consecutive_failures: 0,
            last_poll: None,
            healthy_nodes,
            total_nodes: healthy_nodes,
            models,
        },
    );
}

fn ep(name: &str, endpoint: &str, region: Option<&str>) -> CortexEndpoint {
    CortexEndpoint {
        name: name.into(),
        endpoint: endpoint.into(),
        region: region.map(str::to_string),
    }
}

fn chat_body() -> Bytes {
    Bytes::from(format!("{{\"model\":\"{MODEL}\",\"stream\":false}}"))
}

async fn body_json(resp: Response) -> (StatusCode, Value) {
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

// ── Tests ────────────────────────────────────────────────────────────

#[tokio::test]
async fn routes_to_serving_cortex_and_forwards_bearer() {
    let url = spawn_cortex(ok_cortex("c1")).await;
    let state = state_with(vec![ep("c1", &url, None)], None);
    set_topology(&state, "c1", true, true, true, 2).await;

    let mut headers = HeaderMap::new();
    headers.insert("authorization", "Bearer sk-test-123".parse().unwrap());

    let resp = dispatch(&state, "/v1/chat/completions", headers, chat_body()).await;
    let (status, body) = body_json(resp).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["served_by"], "c1");
    // Bearer reached the cortex unchanged.
    assert_eq!(body["auth_seen"], "Bearer sk-test-123");
}

#[tokio::test]
async fn cortex_429_passes_through_and_is_not_retried() {
    // c1 (ranked first: loaded) returns a genuine 429; c2 would return 200.
    let c1 = spawn_cortex(MockCortex {
        name: "c1",
        rate_limited: true,
    })
    .await;
    let c2 = spawn_cortex(ok_cortex("c2")).await;
    let state = state_with(vec![ep("c1", &c1, None), ep("c2", &c2, None)], None);
    // Both reachable + loaded; c1 has more headroom so it ranks first.
    set_topology(&state, "c1", true, true, true, 5).await;
    set_topology(&state, "c2", true, true, true, 1).await;

    let resp = dispatch(
        &state,
        "/v1/chat/completions",
        HeaderMap::new(),
        chat_body(),
    )
    .await;
    let (status, body) = body_json(resp).await;

    // The genuine 4xx is returned verbatim — NOT retried to c2.
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(body["error"]["code"], "rate_limit_exceeded");
    assert!(body.get("served_by").is_none(), "must not have hit c2");
}

#[tokio::test]
async fn fails_over_to_next_cortex_on_transport_error() {
    // c_dead ranks first (more headroom) but its endpoint is a closed port;
    // c_live is the fallback. The router must fail over and c_live serves.
    let live = spawn_cortex(ok_cortex("c_live")).await;
    let state = state_with(
        vec![
            ep("c_dead", "http://127.0.0.1:1", None),
            ep("c_live", &live, None),
        ],
        None,
    );
    set_topology(&state, "c_dead", true, true, true, 9).await;
    set_topology(&state, "c_live", true, true, true, 1).await;

    let resp = dispatch(
        &state,
        "/v1/chat/completions",
        HeaderMap::new(),
        chat_body(),
    )
    .await;
    let (status, body) = body_json(resp).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["served_by"], "c_live");
}

#[tokio::test]
async fn unknown_model_is_404() {
    let state = state_with(vec![ep("c1", "http://127.0.0.1:1", None)], None);
    // Topology has no entry for MODEL at all.
    let resp = dispatch(
        &state,
        "/v1/chat/completions",
        HeaderMap::new(),
        chat_body(),
    )
    .await;
    let (status, body) = body_json(resp).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "model_not_found");
}

#[tokio::test]
async fn known_but_all_unreachable_is_503() {
    let state = state_with(vec![ep("c1", "http://127.0.0.1:1", None)], None);
    // Cortex knows the model (from a prior good poll) but is now unreachable.
    set_topology(&state, "c1", false, true, true, 2).await;
    let resp = dispatch(
        &state,
        "/v1/chat/completions",
        HeaderMap::new(),
        chat_body(),
    )
    .await;
    let (status, body) = body_json(resp).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "service_unavailable");
}

#[tokio::test]
async fn missing_model_field_is_400() {
    let state = state_with(vec![ep("c1", "http://127.0.0.1:1", None)], None);
    let resp = dispatch(
        &state,
        "/v1/chat/completions",
        HeaderMap::new(),
        Bytes::from_static(b"{\"messages\":[]}"),
    )
    .await;
    let (status, body) = body_json(resp).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "missing_model_field");
}

#[tokio::test]
async fn ranking_prefers_loaded_then_region_then_headroom() {
    // Router is in eu-west. Candidates:
    //   warm-eu : loaded, region match, 1 node   → best
    //   warm-us : loaded, no region,    9 nodes
    //   cold-eu : feasible only, region match     → worst (cold)
    let state = state_with(
        vec![
            ep("warm-eu", "http://127.0.0.1:1", Some("eu-west")),
            ep("warm-us", "http://127.0.0.1:1", Some("us-east")),
            ep("cold-eu", "http://127.0.0.1:1", Some("eu-west")),
        ],
        Some("eu-west".into()),
    );
    set_topology(&state, "warm-eu", true, true, true, 1).await;
    set_topology(&state, "warm-us", true, true, true, 9).await;
    set_topology(&state, "cold-eu", true, false, true, 5).await;

    let Selection::Candidates(order) = select_cortexes(&state, MODEL).await else {
        panic!("expected candidates");
    };
    let names: Vec<&str> = order.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["warm-eu", "warm-us", "cold-eu"]);
}
