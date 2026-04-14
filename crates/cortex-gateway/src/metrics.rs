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

    // Register histograms and counters used by the proxy layer.
    // The `metrics` crate lazily creates metrics on first use, but
    // describing them up front gives Prometheus proper HELP/TYPE lines.
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

    Ok(())
}
