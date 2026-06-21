//! End-to-end federation-catalogue test for #75: poll two mock cortexes
//! that overlap on a model, then `GET /v1/models` on the router and verify
//! the deduped union with merged availability and preserved limit/cost.

use axum::Router;
use axum::routing::get;
use helexa_router::config::{CortexEndpoint, RouterConfig};
use helexa_router::poller::poll_once;
use helexa_router::state::RouterState;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::net::TcpListener;

/// Spawn a mock cortex serving the given `/v1/models` `data` array.
async fn spawn_cortex(models: Value) -> String {
    let models = Arc::new(models);
    let app = Router::new()
        .route(
            "/v1/models",
            get({
                let models = Arc::clone(&models);
                move || {
                    let models = Arc::clone(&models);
                    async move { axum::Json(json!({ "object": "list", "data": &*models })) }
                }
            }),
        )
        .route(
            "/health",
            get(|| async { axum::Json(json!({"status":"ok","nodes":{"healthy":1,"total":1}})) }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// Spawn the router (with poller) wired to the given cortex endpoints, and
/// poll once synchronously so the topology is populated before we query.
async fn spawn_router(cortexes: Vec<CortexEndpoint>) -> String {
    let cfg = RouterConfig {
        cortexes,
        ..Default::default()
    };
    let state = Arc::new(RouterState::from_config(&cfg));
    poll_once(&state).await; // deterministic: fill topology now

    let app = helexa_router::build_app(Arc::clone(&state));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

fn model(id: &str, loaded: bool, feasible_on: &[&str], ctx: u64, input_cost: f64) -> Value {
    json!({
        "id": id,
        "object": "model",
        "created": 0,
        "owned_by": "helexa",
        "loaded": loaded,
        "feasible_on": feasible_on,
        "locations": [],
        "limit": { "context": ctx, "output": 4096 },
        "cost": { "input": input_cost, "output": input_cost * 3.0 }
    })
}

#[tokio::test]
async fn federation_catalogue_dedupes_and_preserves_limit_cost() {
    // cortex A: "shared" loaded (ctx 32768, $0.50) + "only-a" loaded.
    let a = spawn_cortex(json!([
        model("shared", true, &["beast"], 32_768, 0.50),
        model("only-a", true, &["beast"], 8_192, 1.00),
    ]))
    .await;
    // cortex B: "shared" cold-loadable, tighter ctx (16384), cheaper ($0.20).
    let b = spawn_cortex(json!([model("shared", false, &["benjy"], 16_384, 0.20)])).await;

    let router = spawn_router(vec![
        CortexEndpoint {
            name: "op-a".into(),
            endpoint: a,
            region: None,
            tls_ca: None,
        },
        CortexEndpoint {
            name: "op-b".into(),
            endpoint: b,
            region: None,
            tls_ca: None,
        },
    ])
    .await;

    let body: Value = reqwest::get(format!("{router}/v1/models"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().unwrap();
    // Deduped union: "shared" once + "only-a".
    assert_eq!(data.len(), 2);

    let shared = data.iter().find(|m| m["id"] == "shared").unwrap();
    // Loaded somewhere (op-a) → loaded.
    assert_eq!(shared["loaded"], true);
    // feasible_on re-tiered to operator names, both present, sorted.
    let feasible: Vec<&str> = shared["feasible_on"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(feasible, vec!["op-a", "op-b"]);
    // Tightest limit (16384) and cheapest cost ($0.20) win.
    assert_eq!(shared["limit"]["context"], 16_384);
    assert_eq!(shared["cost"]["input"], 0.20);
    // Loaded location named by operator, no neuron VRAM leaked.
    let locs = shared["locations"].as_array().unwrap();
    assert_eq!(locs.len(), 1);
    assert_eq!(locs[0]["node"], "op-a");

    assert!(data.iter().any(|m| m["id"] == "only-a"));
}
