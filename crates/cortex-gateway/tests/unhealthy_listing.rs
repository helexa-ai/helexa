//! `/v1/models` agrees with routing about unhealthy neurons (#376).
//!
//! When a neuron stops answering, the poller marks it unhealthy but keeps
//! its last `/models` and activation snapshot. The router then refuses to
//! send it anything, so a model that only that neuron holds cannot be
//! served without a cold-load elsewhere. The listing used to report it as
//! `loaded: true` anyway — and through tier aliases too — so a client that
//! trusted `loaded` picked a model that failed at once.

mod common;

use cortex_core::config::{
    EvictionSettings, EvictionStrategy, GatewayConfig, GatewaySettings, NeuronEndpoint,
};
use cortex_gateway::state::CortexState;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

fn write_catalogue() -> std::path::PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let toml = r#"
[[models]]
id = "Qwen/large"
harness = "candle"
min_devices = 1

[aliases]
"helexa/large" = "Qwen/large"
"#;
    let path = std::env::temp_dir().join(format!(
        "cortex_test_unhealthy_listing.{}.{}.toml",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, toml).unwrap();
    path
}

/// Two neurons: `big` holds `Qwen/large` and is pre-warming `Qwen/next`;
/// `small` holds `Qwen/small`. Both are polled healthy once.
async fn polled_fleet() -> Arc<CortexState> {
    let big = common::spawn_mock_neuron_with_models_and_health(
        json!([
            {"id": "Qwen/large", "harness": "candle", "status": "loaded", "devices": [0], "vram_used_mb": 20000}
        ]),
        json!({
            "uptime_secs": 30,
            "devices": [],
            "activation": {
                "state": "pre_warming",
                "pending": [],
                "in_progress": "Qwen/next",
                "completed": [],
                "failed": []
            }
        }),
    )
    .await;
    let small = common::spawn_mock_neuron_with_models(json!([
        {"id": "Qwen/small", "harness": "candle", "status": "loaded", "devices": [0], "vram_used_mb": 4000}
    ]))
    .await;
    let config = GatewayConfig {
        gateway: GatewaySettings {
            listen: "127.0.0.1:0".into(),
            metrics_listen: "127.0.0.1:0".into(),
        },
        eviction: EvictionSettings {
            strategy: EvictionStrategy::Lru,
            defrag_after_cycles: 0,
        },
        neurons: vec![
            NeuronEndpoint {
                name: "big".into(),
                endpoint: big,
            },
            NeuronEndpoint {
                name: "small".into(),
                endpoint: small,
            },
        ],
        models_config: write_catalogue().to_string_lossy().into_owned(),
        ..Default::default()
    };
    let fleet = Arc::new(CortexState::from_config(&config));
    cortex_gateway::poller::poll_once(&fleet).await;
    fleet
}

async fn listing(fleet: Arc<CortexState>) -> Vec<Value> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = cortex_gateway::build_app(fleet);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let body: Value = reqwest::get(format!("http://{addr}/v1/models"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    body["data"].as_array().unwrap().clone()
}

fn entry<'a>(data: &'a [Value], id: &str) -> &'a Value {
    data.iter()
        .find(|m| m["id"] == id)
        .unwrap_or_else(|| panic!("{id} should be listed"))
}

#[tokio::test]
async fn models_on_an_unhealthy_neuron_are_not_reported_loaded() {
    let fleet = polled_fleet().await;

    // Control: while both neurons answer, the cached state is live.
    let data = listing(fleet.clone()).await;
    assert_eq!(entry(&data, "Qwen/large")["loaded"], true);
    assert_eq!(entry(&data, "helexa/large")["loaded"], true);
    assert_eq!(
        entry(&data, "Qwen/next")["locations"][0]["status"],
        "loading"
    );

    // `big` goes dark. The poller flips the flag after three failed
    // polls (test_poller_marks_unreachable_node_unhealthy) and keeps the
    // snapshot, which is the state this sets up.
    fleet.nodes.write().await.get_mut("big").unwrap().healthy = false;

    let data = listing(fleet).await;
    let large = entry(&data, "Qwen/large");
    assert_eq!(
        large["loaded"], false,
        "a model held only by an unhealthy neuron cannot be served as loaded"
    );
    assert_eq!(
        large["locations"],
        json!([{"node": "big", "status": "unreachable", "vram_estimate_mb": large["locations"][0]["vram_estimate_mb"]}]),
        "the location stays visible for diagnostics, marked unreachable"
    );
    assert_eq!(
        entry(&data, "helexa/large")["loaded"],
        false,
        "the tier alias must not report its target as loaded either"
    );
    assert!(
        data.iter().all(|m| m["id"] != "Qwen/next"),
        "a stale activation snapshot must not advertise a model as loading"
    );

    // The healthy neuron is unaffected.
    assert_eq!(entry(&data, "Qwen/small")["loaded"], true);
}
