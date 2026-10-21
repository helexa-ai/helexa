//! Hosts that leave the mesh on a schedule (#397).
//!
//! A solar-powered rig goes dark every evening. It announces the window
//! on `/health`, including when it stops taking new work so what it is
//! running can finish. cortex must stop routing there at that moment,
//! treat the silence that follows as planned, and pick the host up again
//! when it returns. Across the whole cycle a client sees only reduced
//! capacity, never a failed request.
//!
//! Also here: a host carrying several replicas of a model (#397) counts as
//! that much capacity, both for routing and for the audio job gate.

mod common;

use axum::Json;
use axum::Router;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use cortex_core::node::unix_now;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

const MODEL: &str = "test-model";

/// A neuron whose announced schedule, load and presence the test controls.
#[derive(Default)]
struct Rig {
    /// Answers 502 to everything, as a host whose power is off would fail
    /// to answer at all.
    gone: bool,
    next_dark: Option<Value>,
    in_flight: usize,
    max_in_flight: usize,
    /// Inference requests that reached this neuron.
    served: usize,
}

type Shared = Arc<Mutex<Rig>>;

async fn spawn_rig(rig: Shared) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let endpoint = base.clone();
    let app = Router::new()
        .route("/models", get(models))
        .route("/health", get(health))
        .route(
            "/models/{id}/endpoint",
            get(move |State(rig): State<Shared>, Path(_): Path<String>| {
                let url = endpoint.clone();
                async move {
                    if rig.lock().unwrap().gone {
                        return StatusCode::BAD_GATEWAY.into_response();
                    }
                    Json(json!({ "url": url })).into_response()
                }
            }),
        )
        .route("/v1/chat/completions", post(chat))
        .with_state(rig);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    base
}

async fn models(State(rig): State<Shared>) -> Response {
    if rig.lock().unwrap().gone {
        return StatusCode::BAD_GATEWAY.into_response();
    }
    Json(json!([{
        "id": MODEL, "harness": "candle", "status": "loaded", "devices": [0],
        "vram_used_mb": null
    }]))
    .into_response()
}

async fn health(State(rig): State<Shared>) -> Response {
    let r = rig.lock().unwrap();
    if r.gone {
        return StatusCode::BAD_GATEWAY.into_response();
    }
    let mut body = json!({
        "uptime_secs": 1,
        "devices": [],
        "models": [{
            "id": MODEL,
            "in_flight": r.in_flight,
            "queue_depth": 0,
            "max_in_flight": r.max_in_flight,
        }],
    });
    if let Some(d) = &r.next_dark {
        body["next_dark"] = d.clone();
    }
    Json(body).into_response()
}

async fn chat(State(rig): State<Shared>) -> Response {
    let mut r = rig.lock().unwrap();
    if r.gone {
        return StatusCode::BAD_GATEWAY.into_response();
    }
    r.served += 1;
    Json(json!({
        "id": "chatcmpl-rig", "object": "chat.completion", "created": 1, "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    }))
    .into_response()
}

fn dark(drain_at: u64, starts_at: u64, ends_at: u64) -> Option<Value> {
    Some(json!({ "drain_at": drain_at, "starts_at": starts_at, "ends_at": ends_at }))
}

/// Send `n` requests through the gateway; every one must succeed.
async fn all_succeed(gateway: &str, n: usize) {
    let client = reqwest::Client::new();
    for i in 0..n {
        let resp = client
            .post(format!("{gateway}/v1/chat/completions"))
            .json(&json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "request {i} failed");
    }
}

fn served(rig: &Shared) -> usize {
    rig.lock().unwrap().served
}

#[tokio::test]
async fn a_host_leaving_and_returning_on_schedule_fails_no_request() {
    let now = unix_now();
    // The rig: eight replicas, idle, dark from tonight.
    let rig: Shared = Arc::new(Mutex::new(Rig {
        max_in_flight: 8,
        next_dark: dark(now + 3600, now + 4500, now + 40_000),
        ..Default::default()
    }));
    // An always-on single card that is already busy.
    let card: Shared = Arc::new(Mutex::new(Rig {
        max_in_flight: 1,
        in_flight: 1,
        ..Default::default()
    }));
    let fleet = common::two_neuron_fleet(
        &spawn_rig(Arc::clone(&rig)).await,
        &spawn_rig(Arc::clone(&card)).await,
    )
    .await;
    let app = cortex_gateway::build_app(Arc::clone(&fleet));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let gateway = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // Daytime: the idle rig takes the work.
    cortex_gateway::poller::poll_once(&fleet).await;
    all_succeed(&gateway, 10).await;
    assert_eq!(served(&rig), 10, "the idle rig should take the work");

    // Draining: still up and polling fine, but its announced drain time has
    // passed. Nothing new goes there.
    rig.lock().unwrap().next_dark = dark(now - 1, now + 60, now + 40_000);
    cortex_gateway::poller::poll_once(&fleet).await;
    all_succeed(&gateway, 10).await;
    assert_eq!(served(&rig), 10, "a draining host must get no new work");
    assert_eq!(served(&card), 10);

    // Dark: the rig stops answering. Its absence is planned, so it leaves
    // routing at once and the gateway reports it as scheduled, not failed.
    rig.lock().unwrap().gone = true;
    cortex_gateway::poller::poll_once(&fleet).await;
    {
        let nodes = fleet.nodes.read().await;
        let a = &nodes["node-a"];
        assert!(!a.healthy, "a host away on schedule is out of routing");
        assert!(a.in_scheduled_absence);
    }
    all_succeed(&gateway, 10).await;
    let health: Value = reqwest::get(format!("{gateway}/health"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["nodes"]["scheduled_dark"], 1);

    // Morning: the rig is back with tomorrow's window, and takes work again.
    {
        let mut r = rig.lock().unwrap();
        r.gone = false;
        r.next_dark = dark(now + 86_400, now + 87_300, now + 126_000);
    }
    cortex_gateway::poller::poll_once(&fleet).await;
    {
        let nodes = fleet.nodes.read().await;
        assert!(nodes["node-a"].healthy);
        assert!(!nodes["node-a"].in_scheduled_absence);
    }
    all_succeed(&gateway, 10).await;
    assert_eq!(served(&rig), 20, "the returned rig takes work again");
    assert_eq!(served(&card), 20);
}

/// An unannounced outage is still a fault: debounced, not quietly accepted.
#[tokio::test]
async fn an_unannounced_absence_is_not_treated_as_scheduled() {
    let rig: Shared = Arc::new(Mutex::new(Rig {
        max_in_flight: 1,
        ..Default::default()
    }));
    let card: Shared = Arc::new(Mutex::new(Rig {
        max_in_flight: 1,
        ..Default::default()
    }));
    let fleet = common::two_neuron_fleet(
        &spawn_rig(Arc::clone(&rig)).await,
        &spawn_rig(Arc::clone(&card)).await,
    )
    .await;
    cortex_gateway::poller::poll_once(&fleet).await;
    rig.lock().unwrap().gone = true;
    cortex_gateway::poller::poll_once(&fleet).await;
    let nodes = fleet.nodes.read().await;
    let a = &nodes["node-a"];
    assert!(a.healthy, "one missed poll rides the debounce");
    assert!(!a.in_scheduled_absence);
}

/// Busy-ness is a share of capacity: a rig with eight replicas and two
/// renders running is less busy than a single card running one.
#[tokio::test]
async fn routing_weighs_load_against_replica_capacity() {
    let rig: Shared = Arc::new(Mutex::new(Rig {
        max_in_flight: 8,
        in_flight: 2,
        ..Default::default()
    }));
    let card: Shared = Arc::new(Mutex::new(Rig {
        max_in_flight: 1,
        in_flight: 1,
        ..Default::default()
    }));
    let fleet = common::two_neuron_fleet(
        &spawn_rig(Arc::clone(&rig)).await,
        &spawn_rig(Arc::clone(&card)).await,
    )
    .await;
    cortex_gateway::poller::poll_once(&fleet).await;
    let route = cortex_gateway::router::resolve(&fleet, MODEL)
        .await
        .expect("loaded on both");
    assert_eq!(route.node_name, "node-a");
}

/// The audio job gate follows the fleet: every replica of a loaded music
/// model is a render slot, and a draining host's are not.
#[tokio::test]
async fn render_capacity_counts_replicas_on_hosts_taking_work() {
    use cortex_core::discovery::{DarkWindow, ModelLoad};
    use cortex_core::node::{ModelEntry, ModelStatus};
    use cortex_gateway::audio_jobs::fleet_render_capacity;

    let fleet = common::two_neuron_fleet("http://127.0.0.1:1", "http://127.0.0.1:2").await;
    let now = unix_now();
    {
        let mut nodes = fleet.nodes.write().await;
        for (name, slots) in [("node-a", 8), ("node-b", 1)] {
            let n = nodes.get_mut(name).unwrap();
            n.healthy = true;
            n.models.insert(
                "ACE-Step/Ace-Step1.5".into(),
                ModelEntry {
                    id: "ACE-Step/Ace-Step1.5".into(),
                    status: ModelStatus::Loaded,
                    last_accessed: None,
                    vram_estimate_mb: None,
                    capabilities: vec![cortex_core::audio::CAPABILITY.into()],
                    tool_call: false,
                    reasoning: false,
                    limit: None,
                    servable: None,
                    reasoning_budget: Vec::new(),
                },
            );
            let load: ModelLoad = serde_json::from_value(json!({
                "id": "ACE-Step/Ace-Step1.5", "in_flight": 0, "queue_depth": 0,
                "max_in_flight": slots
            }))
            .unwrap();
            n.model_load.insert(load.id.clone(), load);
        }
    }
    let nodes = fleet.nodes.read().await;
    assert_eq!(fleet_render_capacity(nodes.values(), now), 9);
    drop(nodes);

    fleet
        .nodes
        .write()
        .await
        .get_mut("node-a")
        .unwrap()
        .next_dark = Some(DarkWindow {
        drain_at: now - 1,
        starts_at: now + 60,
        ends_at: now + 600,
    });
    let nodes = fleet.nodes.read().await;
    assert_eq!(fleet_render_capacity(nodes.values(), now), 1);
}
