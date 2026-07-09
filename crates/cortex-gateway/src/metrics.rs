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
        "cortex_prompt_tokens_total",
        "Total prompt tokens reported by upstream usage objects"
    );
    metrics::describe_counter!(
        "cortex_completion_tokens_total",
        "Total completion tokens reported by upstream usage objects"
    );
    metrics::describe_counter!(
        "cortex_request_errors_total",
        "Total number of failed proxy requests"
    );
    metrics::describe_counter!("cortex_evictions_total", "Total number of model evictions");
    metrics::describe_counter!(
        "cortex_cold_starts_total",
        "Total number of cold-start model loads"
    );
    metrics::describe_counter!(
        "cortex_spend_tokens_total",
        "Total metered tokens (prompt + completion) per principal, labelled by account/key (#51)"
    );
    metrics::describe_counter!(
        "cortex_spend_prompt_tokens_total",
        "Metered prompt tokens per principal, labelled by account/key (#51)"
    );
    metrics::describe_counter!(
        "cortex_spend_completion_tokens_total",
        "Metered completion tokens per principal, labelled by account/key (#51)"
    );
    // Live capacity signals polled from neuron /health (#137), {node,model}.
    metrics::describe_gauge!(
        "cortex_model_in_flight",
        "Requests currently running on a neuron:model (#137)"
    );
    metrics::describe_gauge!(
        "cortex_model_queue_depth",
        "Requests queued in admission for a neuron:model (#137)"
    );
    metrics::describe_gauge!(
        "cortex_model_max_in_flight",
        "Configured concurrency ceiling; saturation = in_flight / max_in_flight (#137)"
    );
    metrics::describe_gauge!(
        "cortex_model_max_queue_depth",
        "Configured admission queue capacity before a neuron:model sheds load (#137)"
    );
    // Per-device GPU headroom polled from neuron /health (#137), {node,device}.
    metrics::describe_gauge!(
        "cortex_device_vram_used_mb",
        "Per-device VRAM used, MB (#137)"
    );
    metrics::describe_gauge!(
        "cortex_device_vram_free_mb",
        "Per-device VRAM free, MB (#137)"
    );
    metrics::describe_gauge!(
        "cortex_device_utilization_pct",
        "Per-device GPU utilization, percent (#137)"
    );
    metrics::describe_gauge!(
        "cortex_device_temp_c",
        "Per-device GPU temperature, Celsius (#137)"
    );
    metrics::describe_counter!(
        "cortex_model_rejections_total",
        "Admission rejections per neuron:model by reason: queue_full / wait_timeout / per_principal — the load-shedding signal (#137)"
    );
    metrics::describe_gauge!(
        "cortex_model_tok_s_decode",
        "Live decode throughput per neuron:model, tokens/sec EMA — the headline capacity number (#137)"
    );
    metrics::describe_gauge!(
        "cortex_model_tok_s_prefill",
        "Live prefill throughput per neuron:model, tokens/sec EMA (#137)"
    );
}
