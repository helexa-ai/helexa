//! Prometheus metrics exporter.
//!
//! Runs on a separate port from the main API, exposing `/metrics`
//! in Prometheus text format.

use anyhow::Result;
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder};
use std::net::SocketAddr;

/// Install the Prometheus metrics recorder and return a handle.
/// The `/metrics` endpoint is served by the exporter's built-in HTTP server.
pub fn install(listen: &str) -> Result<()> {
    let addr: SocketAddr = listen.parse()?;

    with_buckets(PrometheusBuilder::new().with_http_listener(addr))?
        .install()
        .map_err(|e| anyhow::anyhow!("failed to install Prometheus exporter: {e}"))?;

    tracing::info!("prometheus metrics exporter on {addr}");
    describe_metrics();
    Ok(())
}

/// Install a recorder for testing (no HTTP listener). Returns a handle
/// that can render the current metrics as Prometheus text.
pub fn install_test_recorder() -> Result<metrics_exporter_prometheus::PrometheusHandle> {
    let handle = with_buckets(PrometheusBuilder::new())?
        .install_recorder()
        .map_err(|e| anyhow::anyhow!("failed to install test recorder: {e}"))?;
    describe_metrics();
    Ok(handle)
}

/// Give every histogram explicit buckets, so it is exported as a
/// Prometheus **histogram** rather than a summary.
///
/// Without this, `metrics-exporter-prometheus` renders every
/// `histogram!` as a summary: `{quantile="0.95"}` series and no
/// `_bucket` series at all. Three things break as a result, and the
/// third is what made it visible.
///
/// 1. `histogram_quantile()` has nothing to read, so any dashboard
///    panel written the idiomatic way returns *No data* forever. The
///    fleet dashboard's TTFT panel did exactly that while cortex was
///    serving thousands of requests.
/// 2. Summary quantiles are computed per process over a rolling
///    window, so they **cannot be aggregated**. Averaging p95 across
///    two gateways is not p95 of anything.
/// 3. They decay: with no samples in the window every quantile reads
///    `0`, which is indistinguishable from "genuinely instant" on a
///    graph.
///
/// Buckets are per-metric because the quantities differ by orders of
/// magnitude — a request lasting minutes and a TTFT of milliseconds
/// have no useful shared scale. Ranges are chosen from observed fleet
/// behaviour rather than round numbers: decode runs for minutes on a
/// long turn, prefill is seconds on a long prompt, and decode
/// throughput sits in the tens of tokens/sec.
fn with_buckets(builder: PrometheusBuilder) -> Result<PrometheusBuilder> {
    let seconds_short = &[
        0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0,
    ];
    let seconds_long = &[
        0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0,
    ];
    let tokens_per_second = &[
        1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 200.0,
    ];
    builder
        // Whole-request latency: a long agentic turn legitimately runs
        // for many minutes, so the tail has to reach there.
        .set_buckets_for_metric(
            Matcher::Full("cortex_request_duration_seconds".into()),
            seconds_long,
        )?
        // Prefill: sub-second for a short prompt, seconds for a long
        // one. Anything past a minute is pathological and belongs in
        // the overflow bucket.
        .set_buckets_for_metric(
            Matcher::Full("cortex_time_to_first_token_seconds".into()),
            seconds_short,
        )?
        // Image generation is inherently slower than a text turn and
        // scales with resolution and step count.
        .set_buckets_for_metric(
            Matcher::Full("cortex_images_generation_seconds".into()),
            seconds_long,
        )?
        // Not a duration — decode throughput, tens of tokens/sec on
        // this fleet.
        .set_buckets_for_metric(
            Matcher::Full("cortex_tokens_per_second".into()),
            tokens_per_second,
        )
        .map_err(|e| anyhow::anyhow!("failed to configure histogram buckets: {e}"))
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
        "Admission rejections per neuron:model by reason: queue_full / wait_timeout / per_principal / anon_yield — the load-shedding signal (#137, #262)"
    );
    metrics::describe_gauge!(
        "cortex_model_anon_in_flight",
        "Anonymous (unattributable) requests holding a seat on a neuron:model (#262)"
    );
    metrics::describe_gauge!(
        "cortex_model_anon_max_in_flight",
        "Seats anonymous traffic may hold at once, so it cannot starve identified callers (#262)"
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

#[cfg(test)]
mod bucket_tests {
    /// Histograms must export as Prometheus **histograms**, with
    /// `_bucket` series — not as summaries.
    ///
    /// This asserts the exported *shape*, not the recording call,
    /// because the recording was never the problem: cortex measured
    /// TTFT correctly for months while the fleet dashboard's panel sat
    /// on "No data", since `histogram_quantile()` reads `_bucket`
    /// series and a summary has none. A metric that is collected but
    /// cannot be queried is indistinguishable from one that was never
    /// collected — from the graph, and from the operator's chair.
    #[test]
    fn histograms_export_buckets_not_summary_quantiles() {
        let handle = match super::install_test_recorder() {
            Ok(h) => h,
            // Another test in this binary owns the global recorder;
            // it installs the same buckets, so skipping is honest.
            Err(_) => return,
        };
        metrics::histogram!("cortex_time_to_first_token_seconds", "node" => "n", "model" => "m")
            .record(0.42);
        let rendered = handle.render();

        assert!(
            rendered.contains("cortex_time_to_first_token_seconds_bucket"),
            "no _bucket series — histogram_quantile() cannot read this:\n{rendered}"
        );
        assert!(
            rendered.contains("# TYPE cortex_time_to_first_token_seconds histogram"),
            "exported as the wrong metric type:\n{rendered}"
        );
        assert!(
            !rendered
                .contains(r#"cortex_time_to_first_token_seconds{node="n",model="m",quantile="#),
            "still exporting summary quantiles, which cannot be aggregated across gateways"
        );
    }
}
