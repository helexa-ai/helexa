//! Topology-poller acceptance tests for #72: the router maintains a live
//! map of which cortexes serve which models, marks an unreachable/erroring
//! cortex unhealthy and excludes it from routing, and recovers it once
//! reachable again.

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::get;
use axum::{Json, Router};
use helexa_router::config::{CortexEndpoint, RouterConfig};
use helexa_router::poller::{POLL_FAILURE_THRESHOLD, poll_once};
use helexa_router::state::RouterState;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::net::TcpListener;

/// Shared "is this mock cortex up?" flag, toggled by tests to simulate
/// outage and recovery.
#[derive(Clone)]
struct MockState {
    up: Arc<AtomicBool>,
}

async fn mock_models(State(s): State<MockState>) -> Result<Json<Value>, StatusCode> {
    if !s.up.load(Ordering::SeqCst) {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    Ok(Json(json!({
        "object": "list",
        "data": [
            {
                "id": "Qwen/Qwen3-Coder-30B",
                "object": "model",
                "created": 0,
                "owned_by": "helexa",
                "loaded": true,
                "feasible_on": ["beast"],
                "locations": [{"node": "beast", "status": "loaded", "vram_estimate_mb": 19000}]
            },
            {
                "id": "Qwen/Qwen3-VL-8B",
                "object": "model",
                "created": 0,
                "owned_by": "helexa",
                "loaded": false,
                "feasible_on": ["beast"],
                "locations": []
            }
        ]
    })))
}

async fn mock_health(State(s): State<MockState>) -> Result<Json<Value>, StatusCode> {
    if !s.up.load(Ordering::SeqCst) {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    Ok(Json(json!({
        "status": "ok",
        "nodes": { "healthy": 2, "total": 3 }
    })))
}

/// Spawn a mock cortex; returns (base_url, up_flag).
async fn spawn_mock_cortex() -> (String, Arc<AtomicBool>) {
    let up = Arc::new(AtomicBool::new(true));
    let state = MockState { up: up.clone() };
    let app = Router::new()
        .route("/v1/models", get(mock_models))
        .route("/health", get(mock_health))
        .with_state(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}"), up)
}

fn state_for(name: &str, endpoint: &str) -> RouterState {
    let cfg = RouterConfig {
        cortexes: vec![CortexEndpoint {
            name: name.into(),
            endpoint: endpoint.into(),
            region: None,
        }],
        ..Default::default()
    };
    RouterState::from_config(&cfg)
}

#[tokio::test]
async fn poll_builds_live_topology() {
    let (base, _up) = spawn_mock_cortex().await;
    let state = state_for("c1", &base);

    poll_once(&state).await;

    let topo = state.topology.read().await;
    let c1 = topo.get("c1").expect("cortex present");
    assert!(c1.reachable, "should be reachable after a good poll");
    assert_eq!(c1.consecutive_failures, 0);
    assert!(c1.last_poll.is_some());
    assert_eq!((c1.healthy_nodes, c1.total_nodes), (2, 3));

    // Loaded model: loaded + feasible. Catalogue-only model: feasible only.
    let coder = c1.models.get("Qwen/Qwen3-Coder-30B").unwrap();
    assert!(coder.loaded && coder.feasible);
    let vl = c1.models.get("Qwen/Qwen3-VL-8B").unwrap();
    assert!(!vl.loaded && vl.feasible);
    drop(topo);

    // The routing helper sees both serveable models on the reachable cortex.
    assert_eq!(
        state.cortexes_serving("Qwen/Qwen3-VL-8B").await,
        vec!["c1".to_string()]
    );
}

#[tokio::test]
async fn unreachable_cortex_excluded_then_recovers() {
    let (base, up) = spawn_mock_cortex().await;
    let state = state_for("c1", &base);

    // Healthy first.
    poll_once(&state).await;
    assert!(state.topology.read().await["c1"].reachable);

    // Take it down. The first failures debounce (stay reachable) until the
    // threshold; only then is it excluded.
    up.store(false, Ordering::SeqCst);
    for i in 1..POLL_FAILURE_THRESHOLD {
        poll_once(&state).await;
        assert!(
            state.topology.read().await["c1"].reachable,
            "still reachable after {i} failure(s) (below threshold)"
        );
    }
    poll_once(&state).await; // crosses the threshold
    {
        let topo = state.topology.read().await;
        assert!(!topo["c1"].reachable, "excluded after threshold failures");
        assert!(topo["c1"].consecutive_failures >= POLL_FAILURE_THRESHOLD);
    }
    // Excluded from routing.
    assert!(
        state
            .cortexes_serving("Qwen/Qwen3-Coder-30B")
            .await
            .is_empty()
    );

    // Bring it back: the next successful poll restores it.
    up.store(true, Ordering::SeqCst);
    poll_once(&state).await;
    let topo = state.topology.read().await;
    assert!(topo["c1"].reachable, "recovered after a good poll");
    assert_eq!(topo["c1"].consecutive_failures, 0);
}

#[tokio::test]
async fn unconfigured_endpoint_is_unreachable() {
    // Nothing listening on this port → polls fail; below threshold it stays
    // at its initial unreachable state, and never panics.
    let state = state_for("dead", "http://127.0.0.1:1");
    poll_once(&state).await;
    let topo = state.topology.read().await;
    assert!(!topo["dead"].reachable);
    assert_eq!(topo["dead"].consecutive_failures, 1);
}
