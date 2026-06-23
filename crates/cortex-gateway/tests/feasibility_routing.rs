//! Router: a catalogued model whose only topologically-feasible neuron is
//! currently unhealthy is a *transient* condition (retryable 503), not a
//! permanent 404. This is the exact shape of the beast incident: benjy/
//! quadbrat (1 GPU, healthy) can't host the 27B, and beast (2 GPU) — the
//! sole feasible node — briefly drops out → clients must back off and retry,
//! not hard-fail.

use cortex_core::config::{
    EvictionSettings, EvictionStrategy, GatewayConfig, GatewaySettings, NeuronEndpoint,
};
use cortex_core::discovery::{DeviceInfo, DiscoveryResponse};
use cortex_gateway::router::{self, RouteError};
use cortex_gateway::state::CortexState;
use std::sync::Arc;

fn devices(n: usize) -> Vec<DeviceInfo> {
    (0..n)
        .map(|i| DeviceInfo {
            index: i as u32,
            name: "RTX 5090".into(),
            vram_total_mb: 32_768,
            compute_capability: "9.0".into(),
        })
        .collect()
}

fn discovery(host: &str, n_devices: usize) -> DiscoveryResponse {
    DiscoveryResponse {
        hostname: host.into(),
        os: "Linux".into(),
        kernel: "7.0".into(),
        cuda_version: Some("13.0".into()),
        driver_version: Some("999".into()),
        devices: devices(n_devices),
        harnesses: vec!["candle".into()],
        cuda_unavailable_reason: None,
        max_prompt_tokens: 49_152,
    }
}

/// Catalogue with one model needing 2 devices. Returns a temp path.
fn write_catalogue() -> std::path::PathBuf {
    let toml = r#"
[[models]]
id = "big-model"
harness = "candle"
min_devices = 2
"#;
    let path = std::env::temp_dir().join("cortex_test_feasibility_models.toml");
    std::fs::write(&path, toml).unwrap();
    path
}

async fn fleet_with(big_healthy: bool, big_devices: usize) -> Arc<CortexState> {
    let cat = write_catalogue();
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
                name: "small".into(),
                endpoint: "http://127.0.0.1:1".into(),
            },
            NeuronEndpoint {
                name: "big".into(),
                endpoint: "http://127.0.0.1:2".into(),
            },
        ],
        models_config: cat.to_string_lossy().into_owned(),
        entitlements: Default::default(),
        upstream: Default::default(),
    };
    let fleet = Arc::new(CortexState::from_config(&config));
    {
        let mut nodes = fleet.nodes.write().await;
        // "small" is healthy but only has 1 GPU → not feasible for the model.
        let small = nodes.get_mut("small").unwrap();
        small.healthy = true;
        small.discovery = Some(discovery("small", 1));
        // "big" has enough GPUs but its health is the variable under test.
        let big = nodes.get_mut("big").unwrap();
        big.healthy = big_healthy;
        big.discovery = Some(discovery("big", big_devices));
    }
    fleet
}

#[tokio::test]
async fn feasible_node_unhealthy_is_transient_503() {
    // big (2 GPU, the only feasible node) is unhealthy; small (1 GPU) is
    // healthy but can't host the model → retryable, not a permanent 404.
    let fleet = fleet_with(false, 2).await;
    let err = router::resolve(&fleet, "big-model")
        .await
        .expect_err("model can't be served right now");
    assert!(
        matches!(err, RouteError::FeasibleNodeUnhealthy { .. }),
        "expected FeasibleNodeUnhealthy, got {err:?}"
    );
    assert_eq!(err.http_status(), 503);
    assert_eq!(err.retry_after_secs(), Some(3));
    assert_eq!(err.code(), "service_unavailable");
}

#[tokio::test]
async fn no_node_can_ever_satisfy_is_permanent_404() {
    // big is healthy but only has 1 GPU now (e.g. topology genuinely can't
    // satisfy min_devices=2 anywhere) → permanent, non-retryable 404.
    let fleet = fleet_with(true, 1).await;
    let err = router::resolve(&fleet, "big-model")
        .await
        .expect_err("no feasible topology");
    assert!(
        matches!(err, RouteError::NoFeasibleNeuron { .. }),
        "expected NoFeasibleNeuron, got {err:?}"
    );
    assert_eq!(err.http_status(), 404);
    assert_eq!(err.retry_after_secs(), None);
}
