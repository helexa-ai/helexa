mod common;

use serde_json::json;

#[tokio::test]
async fn test_metrics_emitted_after_proxy() {
    let handle = cortex_gateway::metrics::install_test_recorder().expect("recorder should install");

    let mock_url = common::spawn_mock_neuron().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let before = handle.render();
    assert!(
        !before.contains("cortex_requests_total"),
        "no request metrics before any requests"
    );

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .expect("request should succeed");
    assert_eq!(resp.status(), 200);
    let _body: serde_json::Value = resp.json().await.unwrap();

    let after = handle.render();

    assert!(
        after.contains("cortex_requests_total"),
        "cortex_requests_total should be present after a request.\nMetrics:\n{after}"
    );
    assert!(
        after.contains("cortex_request_duration_seconds"),
        "cortex_request_duration_seconds should be present.\nMetrics:\n{after}"
    );
    assert!(
        !after.contains("cortex_request_errors_total"),
        "no errors expected for a successful request"
    );
}
