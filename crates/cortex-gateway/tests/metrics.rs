mod common;

use serde_json::json;
use std::sync::OnceLock;

/// The metrics recorder is a process-wide global; both tests in this
/// binary run against one shared install. Assertions must therefore be
/// order-independent (presence of names / monotonic counters, not
/// "empty before").
fn recorder() -> &'static metrics_exporter_prometheus::PrometheusHandle {
    static HANDLE: OnceLock<metrics_exporter_prometheus::PrometheusHandle> = OnceLock::new();
    HANDLE.get_or_init(|| {
        cortex_gateway::metrics::install_test_recorder().expect("recorder should install")
    })
}

#[tokio::test]
async fn test_metrics_emitted_after_proxy() {
    let handle = recorder();

    let mock_url = common::spawn_mock_neuron().await;
    let gw_url = common::spawn_gateway(&mock_url).await;

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

#[tokio::test]
async fn test_token_metrics_emitted_for_streamed_request() {
    // #21: a streamed chat completion with a final usage chunk must
    // produce TTFT + tok/s histograms and prompt/completion token
    // counters, labelled with model and node. The recorder is global
    // per-process, so this test runs in its own binary invocation —
    // cargo's per-file integration binaries give us that as long as
    // only one test in this file installs the recorder... it isn't:
    // test_metrics_emitted_after_proxy also installs. Whichever wins
    // the race, both render from the same recorder, so assert on
    // delta-able names rather than exact totals.
    let handle = recorder();

    let mock_url = common::spawn_streaming_mock_neuron_with_usage(
        5,
        std::time::Duration::from_millis(40),
        225,
        42,
    )
    .await;
    let gw_url = common::spawn_gateway(&mock_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .json(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true
        }))
        .send()
        .await
        .expect("request should succeed");
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.expect("stream should complete");
    assert!(body.contains("[DONE]"));

    let rendered = handle.render();
    for needle in [
        "cortex_time_to_first_token_seconds",
        "cortex_tokens_per_second",
    ] {
        assert!(
            rendered.contains(needle),
            "{needle} should be present.\nMetrics:\n{rendered}"
        );
    }
    // The recorder is shared with the sibling test (same model/node
    // labels), so counters are lower bounds, not exact values: this
    // request contributed prompt=225 / completion=42.
    let counter_value = |name: &str| -> u64 {
        rendered
            .lines()
            .find(|l| l.starts_with(name) && l.contains(r#"model="test-model""#))
            .and_then(|l| l.rsplit(' ').next())
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("{name} should be present.\nMetrics:\n{rendered}"))
    };
    assert!(
        counter_value("cortex_prompt_tokens_total") >= 225,
        "prompt token counter should include this request's 225.\nMetrics:\n{rendered}"
    );
    assert!(
        counter_value("cortex_completion_tokens_total") >= 42,
        "completion token counter should include this request's 42.\nMetrics:\n{rendered}"
    );
}
