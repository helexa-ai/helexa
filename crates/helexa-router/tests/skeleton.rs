//! Skeleton acceptance tests for #70: the router builds, serves `/health`
//! and `/v1/models` on a plaintext port, and loads its cortex-endpoint list
//! from TOML with env overrides.

use helexa_router::config::{CortexEndpoint, RouterConfig};
use helexa_router::state::RouterState;
use std::sync::Arc;
use tokio::net::TcpListener;

/// Bind the router app on an ephemeral port and return its base URL.
async fn spawn_router(cortexes: Vec<CortexEndpoint>) -> String {
    let cfg = RouterConfig {
        cortexes,
        ..Default::default()
    };
    let state = Arc::new(RouterState::from_config(&cfg));
    let app = helexa_router::build_app(state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    format!("http://{addr}")
}

#[tokio::test]
async fn health_reports_configured_cortex_count() {
    let base = spawn_router(vec![
        CortexEndpoint {
            name: "a".into(),
            endpoint: "https://a.example.com".into(),
        },
        CortexEndpoint {
            name: "b".into(),
            endpoint: "https://b.example.com".into(),
        },
    ])
    .await;

    let body: serde_json::Value = reqwest::get(format!("{base}/health"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(body["status"], "ok");
    assert_eq!(body["cortexes"]["configured"], 2);
}

#[tokio::test]
async fn models_returns_empty_openai_list() {
    let base = spawn_router(vec![]).await;

    let resp = reqwest::get(format!("{base}/v1/models")).await.unwrap();
    assert!(resp.status().is_success());

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"].as_array().unwrap().len(), 0);
}

#[test]
#[allow(clippy::result_large_err)]
fn config_loads_from_toml_with_env_override() {
    figment::Jail::expect_with(|jail| {
        jail.create_file(
            "helexa-router.toml",
            r#"
[router]
listen = "127.0.0.1:8088"

[[cortexes]]
name = "lair-cafe"
endpoint = "https://cortex.lair.cafe"
"#,
        )?;

        // Env override wins over the TOML value.
        jail.set_env("HELEXA_ROUTER_ROUTER__LISTEN", "0.0.0.0:9099");

        let cfg = RouterConfig::load("helexa-router.toml").expect("load config");

        assert_eq!(cfg.router.listen, "0.0.0.0:9099");
        assert_eq!(cfg.cortexes.len(), 1);
        assert_eq!(cfg.cortexes[0].name, "lair-cafe");
        assert_eq!(cfg.cortexes[0].endpoint, "https://cortex.lair.cafe");

        Ok(())
    });
}
