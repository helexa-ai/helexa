//! The extensible test suite.
//!
//! A [`Scenario`] puts one warm model through one shaped request and
//! reports operator-felt metrics (TTFT, decode tok/s, total). Phase 1
//! ships the chat-latency family ported faithfully from `script/bench.py`;
//! the trait is the seam for future families (vision, concurrency,
//! long-generation, cold-start) selected per model via [`Scenario::applies_to`].

use crate::config::ScenarioConfig;
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use cortex_core::harness::ModelInfo;
use cortex_core::openai::ChatCompletionChunk;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde_json::json;
use std::time::{Duration, Instant};

/// A paragraph of filler re-used to synthesise prompts of a target
/// approximate token count (~4 chars/token heuristic — close enough for
/// bucketing; real token counts are read back from the usage object).
/// Mirrors `script/bench.py::FILLER`.
const FILLER: &str = "The quick brown fox jumps over the lazy dog while the band plays \
a slow waltz in the background and somebody counts the beats. ";

/// `/no_think`: Qwen3-family soft switch keeping thinking models from
/// burning the token budget invisibly. Harmless for non-thinking models.
const QUESTION: &str = "\n\nRetell the scene above as a vivid story of about 300 words. /no_think";

/// Build a synthetic prompt of approximately `approx_tokens` tokens.
/// Ported from `bench.py::build_prompt`.
pub fn build_prompt(approx_tokens: u32) -> String {
    let target_chars = (approx_tokens.max(16) as usize) * 4;
    let reps = target_chars / FILLER.len() + 1;
    let mut body = FILLER.repeat(reps);
    body.truncate(target_chars);
    body.push_str(QUESTION);
    body
}

/// Per-request inputs shared by every scenario.
pub struct RunCtx<'a> {
    pub client: &'a reqwest::Client,
    /// Fully-qualified chat-completions URL for the target.
    pub chat_url: String,
    pub model_id: String,
    pub max_tokens: u64,
    pub timeout: Duration,
}

/// Operator-felt metrics for a single measured request.
#[derive(Debug, Clone)]
pub struct ScenarioMetrics {
    /// Time to first content chunk (seconds).
    pub ttft_s: f64,
    /// Completion tokens / decode window. `None` when the window is too
    /// short to be honest (≤ 200 ms), matching bench.py.
    pub decode_tps: Option<f64>,
    /// Wall-clock for the whole request (seconds).
    pub total_s: f64,
    /// Prompt tokens from the final `usage` object, if the server sent one.
    pub prompt_tokens: Option<u64>,
    /// Completion tokens: from `usage` when present, else content-chunk count.
    pub completion_tokens: u64,
    /// Server-measured prefill duration (ms), from the `usage.helexa_timing`
    /// extension (#85). `None` when the server didn't emit it (external
    /// engines, non-instrumented paths). The honest prefill-phase number,
    /// distinct from client-observed `ttft_s` which also includes request
    /// setup + first-byte network latency.
    pub prefill_ms: Option<u64>,
    /// Server-measured decode duration (ms), from `usage.helexa_timing`.
    pub decode_ms: Option<u64>,
    /// Tokens submitted to prefill — the denominator for prefill tok/s.
    pub prefill_tokens: Option<u64>,
    // ── Concurrency / agentic-load fields (#89) ──────────────────────────
    // Set only by the concurrency scenario, which fans out N simultaneous
    // streams to characterize the real a0/hermes/opencode workload that
    // batch-1 single-request measurement can't see. `None` for single
    // requests. For a concurrency burst, the inherited fields carry the
    // aggregate: `ttft_s` = median TTFT across streams, `decode_tps` = node
    // throughput (total tokens / burst window), `total_s` = burst wall-clock,
    // `completion_tokens` = total across streams.
    /// Number of simultaneous streams in the burst (the cell dimension).
    pub concurrency: Option<u32>,
    /// p95 of per-stream TTFT within the burst — the tail under simultaneous
    /// load, where batch-1 serialization actually hurts.
    pub ttft_p95_s: Option<f64>,
    /// Median per-stream admission queue-wait (ms), approximated as
    /// `ttft − prefill_ms` (#85): on a batch-1 server, later streams wait for
    /// earlier ones, so TTFT inflates while server prefill stays constant —
    /// the gap is the wait. `None` if streams didn't report `helexa_timing`.
    pub queue_wait_ms_median: Option<f64>,
    /// Streams shed by admission control (HTTP 429/503) during the burst —
    /// honest backpressure, not silent failures.
    pub rejected: Option<u32>,
}

#[async_trait]
pub trait Scenario: Send + Sync {
    /// Stable id, e.g. `chat:128`. Used as the version-aware skip key
    /// dimension and recorded against every run.
    fn id(&self) -> &str;

    /// Approximate prompt size in tokens (the cell dimension), recorded
    /// for reporting.
    fn prompt_size(&self) -> u32;

    /// Whether this scenario should run against the given model. Default
    /// runs against everything; vision/audio scenarios will gate on
    /// [`ModelInfo::capabilities`].
    fn applies_to(&self, _model: &ModelInfo) -> bool {
        true
    }

    /// Issue one shaped request and measure it.
    async fn run(&self, ctx: &RunCtx) -> Result<ScenarioMetrics>;
}

/// Build the active scenario set from config: one chat-latency scenario per
/// prompt size, plus one concurrency scenario per configured level (#89).
/// Concurrency levels default to empty (opt-in), since a burst puts real
/// simultaneous load on a serving fleet — operators enable it deliberately.
pub fn build_scenarios(cfg: &ScenarioConfig) -> Vec<Box<dyn Scenario>> {
    let mut scenarios: Vec<Box<dyn Scenario>> = cfg
        .prompt_sizes
        .iter()
        .map(|&size| {
            Box::new(ChatLatencyScenario {
                id: format!("chat:{size}"),
                approx_prompt_tokens: size,
            }) as Box<dyn Scenario>
        })
        .collect();
    for &n in &cfg.concurrency_levels {
        scenarios.push(Box::new(ConcurrencyScenario {
            id: format!("concurrency:{n}"),
            concurrency: n,
            approx_prompt_tokens: cfg.concurrency_prompt_tokens,
        }) as Box<dyn Scenario>);
    }
    scenarios
}

/// The chat-completions request body shared by the latency and concurrency
/// scenarios — streamed, deterministic (temperature 0), usage included.
fn chat_payload(ctx: &RunCtx, prompt: &str) -> serde_json::Value {
    json!({
        "model": ctx.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": ctx.max_tokens,
        "temperature": 0,
        "stream": true,
        "stream_options": {"include_usage": true},
    })
}

/// Streamed single-request chat-completions latency probe — the batch-1
/// regime bench.py measures.
pub struct ChatLatencyScenario {
    id: String,
    approx_prompt_tokens: u32,
}

#[async_trait]
impl Scenario for ChatLatencyScenario {
    fn id(&self) -> &str {
        &self.id
    }

    fn prompt_size(&self) -> u32 {
        self.approx_prompt_tokens
    }

    async fn run(&self, ctx: &RunCtx) -> Result<ScenarioMetrics> {
        let prompt = build_prompt(self.approx_prompt_tokens);
        let payload = chat_payload(ctx, &prompt);
        let fut = stream_and_measure(ctx, &payload);
        tokio::time::timeout(ctx.timeout, fut)
            .await
            .map_err(|_| anyhow!("request timed out after {:?}", ctx.timeout))?
    }
}

/// Fan-out load probe: fire `concurrency` identical streams at once and
/// measure how the fleet behaves under simultaneous pressure (#89). This is
/// the only scenario that exercises the real a0/hermes/opencode pattern —
/// many agentic requests per user turn — which batch-1 single-request
/// timing cannot characterize. On a batch-1 serialized server, aggregate
/// throughput stays ~flat while TTFT/queue-wait inflate with `concurrency`;
/// that gap is the evidence for/against continuous batching.
pub struct ConcurrencyScenario {
    id: String,
    concurrency: u32,
    approx_prompt_tokens: u32,
}

#[async_trait]
impl Scenario for ConcurrencyScenario {
    fn id(&self) -> &str {
        &self.id
    }

    fn prompt_size(&self) -> u32 {
        self.approx_prompt_tokens
    }

    async fn run(&self, ctx: &RunCtx) -> Result<ScenarioMetrics> {
        let prompt = build_prompt(self.approx_prompt_tokens);
        let payload = chat_payload(ctx, &prompt);

        // Fire all streams at once; each is independently timed and capped by
        // the per-request timeout so one hung stream can't stall the burst.
        let burst_start = Instant::now();
        let futs = (0..self.concurrency).map(|_| async {
            tokio::time::timeout(ctx.timeout, stream_and_measure(ctx, &payload)).await
        });
        let results = futures::future::join_all(futs).await;
        let burst_window = burst_start.elapsed().as_secs_f64();

        let mut streams: Vec<ScenarioMetrics> = Vec::new();
        let mut rejected: u32 = 0;
        for r in results {
            match r {
                Ok(Ok(m)) => streams.push(m),
                // Admission backpressure (429/503) is shed load, counted
                // separately from genuine failures/timeouts.
                Ok(Err(e)) if is_admission_reject(&e) => rejected += 1,
                Ok(Err(_)) | Err(_) => {}
            }
        }
        if streams.is_empty() {
            return Err(anyhow!(
                "all {} concurrent streams failed ({rejected} shed by admission)",
                self.concurrency
            ));
        }

        let total_tokens: u64 = streams.iter().map(|m| m.completion_tokens).sum();
        let ttfts: Vec<f64> = streams.iter().map(|m| m.ttft_s).collect();
        // queue-wait ≈ TTFT − server prefill (#85); only for streams that
        // reported helexa_timing.
        let queue_waits: Vec<f64> = streams
            .iter()
            .filter_map(|m| {
                m.prefill_ms
                    .map(|p| (m.ttft_s * 1000.0 - p as f64).max(0.0))
            })
            .collect();
        // Aggregate decode throughput across the whole node for the burst.
        let aggregate_tps = if burst_window > 0.0 {
            Some(total_tokens as f64 / burst_window)
        } else {
            None
        };

        Ok(ScenarioMetrics {
            ttft_s: median(&ttfts).unwrap_or(0.0),
            decode_tps: aggregate_tps,
            total_s: burst_window,
            prompt_tokens: streams.iter().find_map(|m| m.prompt_tokens),
            completion_tokens: total_tokens,
            prefill_ms: None,
            decode_ms: None,
            prefill_tokens: None,
            concurrency: Some(self.concurrency),
            ttft_p95_s: percentile(&ttfts, 95.0),
            queue_wait_ms_median: median(&queue_waits),
            rejected: Some(rejected),
        })
    }
}

/// Whether a stream error was admission backpressure (HTTP 429/503) rather
/// than a genuine failure. `stream_and_measure` renders the upstream status
/// into the error string, so a substring check is sufficient.
fn is_admission_reject(e: &anyhow::Error) -> bool {
    let s = e.to_string();
    s.contains("429") || s.contains("503")
}

/// Median of a slice (sorted copy). `None` if empty.
fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = (v.len() - 1) / 2;
    let hi = v.len() / 2;
    Some((v[lo] + v[hi]) / 2.0)
}

/// Nearest-rank percentile of a slice (`p` in 0..=100). `None` if empty.
fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = (p / 100.0 * v.len() as f64).ceil() as usize;
    Some(v[rank.clamp(1, v.len()) - 1])
}

/// The SSE-timing core, ported from `bench.py::one_run`. Kept free of the
/// `Scenario` trait so it's unit-testable against a mock byte stream.
async fn stream_and_measure(
    ctx: &RunCtx<'_>,
    payload: &serde_json::Value,
) -> Result<ScenarioMetrics> {
    let start = Instant::now();
    let resp = ctx
        .client
        .post(&ctx.chat_url)
        .json(payload)
        .send()
        .await
        .context("sending chat request")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(anyhow!("upstream returned {status}: {}", body.trim()));
    }

    let mut stream = resp.bytes_stream().eventsource();
    let mut first: Option<Instant> = None;
    let mut last: Option<Instant> = None;
    let mut chunk_count: u64 = 0;
    let mut prompt_tokens: Option<u64> = None;
    let mut completion_tokens: Option<u64> = None;
    let mut prefill_ms: Option<u64> = None;
    let mut decode_ms: Option<u64> = None;
    let mut prefill_tokens: Option<u64> = None;

    while let Some(event) = stream.next().await {
        let event = event.context("reading SSE stream")?;
        let now = Instant::now();
        let data = event.data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let chunk: ChatCompletionChunk = match serde_json::from_str(data) {
            Ok(c) => c,
            Err(_) => continue, // tolerate non-JSON keepalive frames
        };
        if let Some(choice) = chunk.choices.first()
            && choice
                .delta
                .get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|s| !s.is_empty())
        {
            if first.is_none() {
                first = Some(now);
            }
            last = Some(now);
            chunk_count += 1;
        }
        if let Some(usage) = chunk.usage {
            prompt_tokens = Some(usage.prompt_tokens);
            completion_tokens = Some(usage.completion_tokens);
            if let Some(t) = usage.helexa_timing {
                prefill_ms = Some(t.prefill_ms);
                decode_ms = Some(t.decode_ms);
                prefill_tokens = Some(t.prefill_tokens);
            }
        }
    }
    let end = Instant::now();

    let first = first.ok_or_else(|| anyhow!("no content chunks received"))?;

    // neuron emits one SSE chunk per visible token, so chunk_count is an
    // engine-truth count when no usage frame is sent.
    let tokens = completion_tokens.filter(|&t| t > 0).unwrap_or(chunk_count);
    // decode rate is only meaningful over a real inter-chunk window.
    let window = last
        .filter(|&l| l > first)
        .map(|l| (l - first).as_secs_f64())
        .unwrap_or(0.0);
    Ok(ScenarioMetrics {
        ttft_s: (first - start).as_secs_f64(),
        decode_tps: if window > 0.2 {
            Some(tokens as f64 / window)
        } else {
            None
        },
        total_s: (end - start).as_secs_f64(),
        prompt_tokens,
        completion_tokens: tokens,
        prefill_ms,
        decode_ms,
        prefill_tokens,
        // Concurrency fields unset on the single-request path; the
        // concurrency scenario builds its own aggregate (#89).
        concurrency: None,
        ttft_p95_s: None,
        queue_wait_ms_median: None,
        rejected: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_grows_with_token_target() {
        let small = build_prompt(128);
        let big = build_prompt(4096);
        assert!(big.len() > small.len());
        // ~4 chars/token + the trailing question.
        assert!(small.len() >= 128 * 4);
        assert!(small.ends_with("/no_think"));
    }

    #[test]
    fn median_and_percentile_basics() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[]), None);
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&v, 50.0), Some(3.0));
        assert_eq!(percentile(&v, 95.0), Some(5.0)); // nearest-rank → max with n=5
        assert_eq!(percentile(&[], 95.0), None);
    }

    #[test]
    fn admission_rejects_detected_by_status() {
        assert!(is_admission_reject(&anyhow!(
            "upstream returned 429 Too Many Requests"
        )));
        assert!(is_admission_reject(&anyhow!(
            "upstream returned 503 Service Unavailable"
        )));
        assert!(!is_admission_reject(&anyhow!(
            "upstream returned 500 Internal"
        )));
        assert!(!is_admission_reject(&anyhow!("connection refused")));
    }

    #[test]
    fn concurrency_scenarios_built_from_config() {
        use crate::config::ScenarioConfig;
        let cfg = ScenarioConfig {
            prompt_sizes: vec![128],
            max_tokens: 64,
            concurrency_levels: vec![2, 8],
            concurrency_prompt_tokens: 512,
        };
        let ids: Vec<String> = build_scenarios(&cfg)
            .iter()
            .map(|s| s.id().to_string())
            .collect();
        assert!(ids.contains(&"chat:128".to_string()));
        assert!(ids.contains(&"concurrency:2".to_string()));
        assert!(ids.contains(&"concurrency:8".to_string()));
    }

    #[test]
    fn prompt_floor_for_tiny_targets() {
        // max(approx,16) floor means even 0 yields a non-trivial prompt.
        let p = build_prompt(0);
        assert!(p.len() >= 16 * 4);
    }
}
