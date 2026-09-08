//! Prometheus metrics for neuron.
//!
//! neuron had no `/metrics` surface at all: every signal it produced
//! reached Prometheus only by cortex polling `/health` every ~10 s and
//! republishing it as gauges. That is enough for routing decisions and
//! useless for asking where a request's time went — an EMA sampled at
//! 10 s cannot see inside a one-second forward, and the finest split
//! that existed anywhere was prefill versus decode.
//!
//! Served from the main API port rather than a second listener: neuron
//! already runs an axum server, and a separate port is one more thing
//! to open in the firewall for no benefit.

use anyhow::Result;
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use std::sync::OnceLock;

static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();

/// Install the process-wide recorder. Idempotent in effect: a second
/// call leaves the first handle in place.
pub fn install() -> Result<()> {
    let handle = with_buckets(PrometheusBuilder::new())?
        .install_recorder()
        .map_err(|e| anyhow::anyhow!("failed to install Prometheus recorder: {e}"))?;
    describe();
    let _ = HANDLE.set(handle);
    Ok(())
}

/// Render the current metrics, or `None` if the recorder was never
/// installed. A `None` here must surface as an error rather than an
/// empty body: an empty scrape is indistinguishable from a healthy
/// process with nothing to report.
pub fn render() -> Option<String> {
    HANDLE.get().map(|h| h.render())
}

/// Explicit buckets, or `metrics-exporter-prometheus` renders every
/// histogram as a **summary** — `{quantile=...}` series, no `_bucket`,
/// no aggregation across hosts, and every quantile decaying to `0`
/// when idle so "no traffic" looks like "instant". The same trap
/// cortex's exporter documents; the fix has to be repeated per metric.
fn with_buckets(builder: PrometheusBuilder) -> Result<PrometheusBuilder> {
    // Per-request phase sums: microseconds for an emit on a fast model,
    // minutes for the forward on a long turn.
    let phase_seconds = &[
        0.001, 0.005, 0.025, 0.1, 0.25, 1.0, 2.5, 5.0, 15.0, 30.0, 60.0, 300.0,
    ];
    let seconds_short = &[
        0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0,
    ];
    let seconds_long = &[
        0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0,
    ];
    builder
        .set_buckets_for_metric(
            Matcher::Full("neuron_decode_phase_seconds".into()),
            phase_seconds,
        )?
        .set_buckets_for_metric(
            Matcher::Full("neuron_prefill_seconds".into()),
            seconds_short,
        )?
        .set_buckets_for_metric(Matcher::Full("neuron_decode_seconds".into()), seconds_long)
        .map_err(|e| anyhow::anyhow!("failed to configure histogram buckets: {e}"))
}

fn describe() {
    metrics::describe_histogram!(
        "neuron_decode_phase_seconds",
        "Decode-loop time by phase, summed per request. The `_sum` \
         series is the point: ratio one phase against the total to see \
         where decode goes, without needing per-step resolution."
    );
    metrics::describe_histogram!("neuron_prefill_seconds", "Prefill wall-clock per request");
    metrics::describe_histogram!("neuron_decode_seconds", "Decode wall-clock per request");
    metrics::describe_counter!(
        "neuron_completion_tokens_total",
        "Tokens generated, the denominator for per-token phase cost"
    );
}

/// Record one finished inference.
///
/// `phases` is optional because not every serving path measures them,
/// and a path that does not must emit **nothing** rather than zeros —
/// a zero sample pulls the histogram's mean down and reads as "this
/// phase is free here", which is the opposite of "nobody looked".
///
/// The residual is emitted as its own phase. Whatever the brackets
/// failed to cover is then visible as a labelled series instead of
/// being silently divided among the phases that were measured.
pub fn record_finish(
    model: &str,
    prefill_ms: u32,
    decode_ms: u32,
    completion_tokens: u32,
    phases: Option<(u32, u32, u32)>,
) {
    let m = model.to_string();
    metrics::histogram!("neuron_prefill_seconds", "model" => m.clone())
        .record(prefill_ms as f64 / 1e3);
    metrics::histogram!("neuron_decode_seconds", "model" => m.clone())
        .record(decode_ms as f64 / 1e3);
    metrics::counter!("neuron_completion_tokens_total", "model" => m.clone())
        .increment(completion_tokens as u64);

    let Some((forward_ms, sample_ms, emit_ms)) = phases else {
        return;
    };
    for (phase, ms) in [
        ("forward", forward_ms),
        ("sample", sample_ms),
        ("emit", emit_ms),
    ] {
        metrics::histogram!("neuron_decode_phase_seconds", "model" => m.clone(), "phase" => phase)
            .record(ms as f64 / 1e3);
    }
    // Saturating: the phases are measured inside the decode window, so
    // they cannot legitimately exceed it, but clock granularity can
    // round them a millisecond over and an underflow here would report
    // an enormous residual.
    let residual =
        decode_ms.saturating_sub(forward_ms.saturating_add(sample_ms).saturating_add(emit_ms));
    metrics::histogram!("neuron_decode_phase_seconds", "model" => m, "phase" => "residual")
        .record(residual as f64 / 1e3);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An unmeasured path must contribute no phase samples at all.
    /// Zeros would drag the histogram's mean toward zero and read as
    /// "this phase costs nothing on this model".
    #[test]
    fn phases_absent_means_no_phase_series() {
        let handle = with_buckets(PrometheusBuilder::new())
            .expect("buckets")
            .install_recorder()
            .expect("recorder");
        describe();

        record_finish("measured", 100, 1000, 64, Some((900, 50, 10)));
        record_finish("unmeasured", 100, 1000, 64, None);
        let out = handle.render();

        assert!(
            out.contains("neuron_decode_phase_seconds") && out.contains(r#"model="measured""#),
            "measured model must emit phases:\n{out}"
        );
        for line in out
            .lines()
            .filter(|l| l.contains("neuron_decode_phase_seconds"))
        {
            assert!(
                !line.contains(r#"model="unmeasured""#),
                "unmeasured model must emit no phase series, got: {line}"
            );
        }
        // Buckets, not a summary: `histogram_quantile()` needs `_bucket`.
        assert!(
            out.contains("neuron_decode_phase_seconds_bucket"),
            "must export as a histogram, not a summary:\n{out}"
        );
        // 1000 - (900 + 50 + 10) = 40 ms of residual, as its own series.
        assert!(
            out.contains(r#"phase="residual""#),
            "residual must be its own series:\n{out}"
        );
    }
}
