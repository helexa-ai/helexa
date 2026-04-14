//! Prometheus metrics exporter.
//!
//! Runs on a separate port from the main API, exposing `/metrics`
//! in Prometheus text format.

use anyhow::Result;
use metrics_exporter_prometheus::PrometheusBuilder;
use std::net::SocketAddr;

/// Install the Prometheus metrics recorder and return a handle.
/// The `/metrics` endpoint is served by the exporter's built-in HTTP server.
pub fn install(listen: &str) -> Result<()> {
    let addr: SocketAddr = listen.parse()?;

    PrometheusBuilder::new()
        .with_http_listener(addr)
        .install()
        .map_err(|e| anyhow::anyhow!("failed to install Prometheus exporter: {e}"))?;

    tracing::info!("prometheus metrics exporter on {addr}");
    describe_metrics();
    Ok(())
}

/// Install a recorder for testing (no HTTP listener). Returns a handle
/// that can render the current metrics as Prometheus text.
pub fn install_test_recorder() -> Result<metrics_exporter_prometheus::PrometheusHandle> {
    let handle = PrometheusBuilder::new()
        .install_recorder()
        .map_err(|e| anyhow::anyhow!("failed to install test recorder: {e}"))?;
    describe_metrics();
    Ok(handle)
}

fn describe_metrics() {
    metrics::describe_histogram!(
        "cortex_request_duration_seconds",
        "Total request latency in seconds"
    );
    metrics::describe_histogram!(
        "cortex_time_to_first_token_seconds",
        "Time to first token in seconds"
    );
    metrics::describe_histogram!(
        "cortex_tokens_per_second",
        "Generation throughput in tokens per second"
    );
    metrics::describe_counter!("cortex_requests_total", "Total number of proxied requests");
    metrics::describe_counter!(
        "cortex_request_errors_total",
        "Total number of failed proxy requests"
    );
    metrics::describe_counter!("cortex_evictions_total", "Total number of model evictions");
    metrics::describe_counter!(
        "cortex_cold_starts_total",
        "Total number of cold-start model loads"
    );
}
