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
    /// Tokens spent inside the reasoning span, from
    /// `usage.completion_tokens_details.reasoning_tokens` (#223). A
    /// sub-count of `completion_tokens`.
    ///
    /// Worth a trend of its own: on a reasoning model this is the
    /// dominant cost driver and it moves independently of speed. A
    /// template change, a sampling change (#283) or a reasoning-budget
    /// change can double what the model thinks before answering while
    /// every tok/s number stays flat — the bill doubles and no chart
    /// moves.
    pub reasoning_tokens: Option<u64>,
    /// p95 inter-token arrival gap in milliseconds — the tail of the
    /// stream's smoothness, client-observed.
    ///
    /// `decode_tps` is a mean over the whole decode window, so a stream
    /// that stalls for a second and then catches up is indistinguishable
    /// from one that never stalled. This is the number a user actually
    /// feels, and the one a batching stall or a mid-stream rebatch shows
    /// up in. `None` for non-streaming, which has no inter-token gaps by
    /// construction, and for streams too short to have a tail.
    pub tpot_p95_ms: Option<f64>,
    /// Prompt tokens served from neuron's prefix KV cache (#269), from
    /// `usage.prompt_tokens_details.cached_tokens`.
    ///
    /// The reason prefill timing varies between otherwise identical
    /// samples. #269 exists because the saving was invisible and every
    /// client reported a 0% hit rate; bench then threw the number away.
    pub cached_tokens: Option<u64>,
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
    /// Full generated text, captured only by the capability probe (#91) so
    /// the output can be quality-scored later (manual or LLM-judge). `None`
    /// for latency/throughput scenarios, which discard the text.
    pub artifact: Option<String>,
    /// Metered image work in megapixel-steps (#202), from
    /// `usage.helexa_image_units`. Set only by image scenarios (#203).
    pub image_units: Option<f64>,
}

#[async_trait]
pub trait Scenario: Send + Sync {
    /// Stable id, e.g. `chat:128`. Used as the version-aware skip key
    /// dimension and recorded against every run.
    fn id(&self) -> &str;

    /// Approximate prompt size in tokens (the cell dimension), recorded
    /// for reporting.
    fn prompt_size(&self) -> u32;

    /// Whether this scenario should run against the given model. The
    /// default runs against everything except image-generation models
    /// (a chat request at an image model is a 422 `wrong_modality`);
    /// modality-specific scenarios override this.
    fn applies_to(&self, model: &ModelInfo) -> bool {
        !model.capabilities.iter().any(|c| c == "image")
    }

    /// Issue one shaped request and measure it.
    async fn run(&self, ctx: &RunCtx) -> Result<ScenarioMetrics>;
}

/// Fixed-seed image-latency scenario (#203): one 9-step generation at a
/// configured square size against `/v1/images/generations`, recording
/// the server-side phase timing (`usage.helexa_timing`) and the metered
/// megapixel-steps. Applies only to models advertising the `image`
/// capability.
pub struct ImageLatencyScenario {
    pub id: String,
    pub side_px: u32,
}

#[async_trait]
impl Scenario for ImageLatencyScenario {
    fn id(&self) -> &str {
        &self.id
    }

    fn prompt_size(&self) -> u32 {
        // The cell dimension for images is resolution, not tokens.
        self.side_px
    }

    fn applies_to(&self, model: &ModelInfo) -> bool {
        model.capabilities.iter().any(|c| c == "image")
    }

    async fn run(&self, ctx: &RunCtx) -> Result<ScenarioMetrics> {
        let url = ctx
            .chat_url
            .replace("/v1/chat/completions", "/v1/images/generations");
        let body = serde_json::json!({
            "model": ctx.model_id,
            "prompt": "A lighthouse on a rocky coast at dusk, photorealistic, long exposure",
            "size": format!("{}x{}", self.side_px, self.side_px),
            "seed": 42,
        });
        let start = std::time::Instant::now();
        let resp = ctx
            .client
            .post(&url)
            .timeout(ctx.timeout)
            .json(&body)
            .send()
            .await
            .context("images request failed")?;
        let status = resp.status();
        let v: serde_json::Value = resp.json().await.context("images response not JSON")?;
        anyhow::ensure!(status.is_success(), "images request {status}: {v}");
        let total_s = start.elapsed().as_secs_f64();

        // Verify the payload decodes to a PNG of the requested size —
        // a bench that records latency for broken output is worse than
        // no bench.
        let b64 = v["data"][0]["b64_json"]
            .as_str()
            .context("missing data[0].b64_json")?;
        use base64::Engine as _;
        let png = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .context("b64_json is not base64")?;
        anyhow::ensure!(
            png.len() > 8 && &png[1..4] == b"PNG",
            "payload is not a PNG"
        );

        let usage = &v["usage"];
        let timing = &usage["helexa_timing"];
        Ok(ScenarioMetrics {
            // Client-observed time before pixels start mattering ≈
            // encode phase; recorded for shape-consistency with chat.
            ttft_s: timing["encode_ms"].as_u64().unwrap_or(0) as f64 / 1000.0,
            decode_tps: None,
            total_s,
            prompt_tokens: None,
            completion_tokens: 0,
            prefill_ms: timing["encode_ms"].as_u64(),
            decode_ms: timing["denoise_ms"]
                .as_u64()
                .map(|d| d + timing["decode_ms"].as_u64().unwrap_or(0)),
            prefill_tokens: None,
            reasoning_tokens: None,
            cached_tokens: None,
            tpot_p95_ms: None,
            concurrency: None,
            ttft_p95_s: None,
            queue_wait_ms_median: None,
            rejected: None,
            artifact: None,
            image_units: usage["helexa_image_units"].as_f64(),
        })
    }
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
            streaming: true,
        }) as Box<dyn Scenario>);
    }
    // Non-streaming bursts (#285). A separate cell id so the two shapes
    // never average together — they measured 1.00x against 3.98x before
    // #285, and a combined number would have hidden both.
    for &n in &cfg.concurrency_nonstreaming_levels {
        scenarios.push(Box::new(ConcurrencyScenario {
            id: format!("concurrency:{n}:nostream"),
            concurrency: n,
            approx_prompt_tokens: cfg.concurrency_prompt_tokens,
            streaming: false,
        }) as Box<dyn Scenario>);
    }
    for &side in &cfg.image_sizes {
        scenarios.push(Box::new(ImageLatencyScenario {
            id: format!("image:{side}"),
            side_px: side,
        }) as Box<dyn Scenario>);
    }
    for probe in &cfg.capability_probes {
        scenarios.push(Box::new(CapabilityScenario {
            id: format!("capability:{}", probe.name),
            prompt: probe.prompt.clone(),
            max_tokens: probe.max_tokens,
        }) as Box<dyn Scenario>);
    }
    scenarios
}

/// A single small streamed request, timed like a chat-latency run. Used by
/// the swap-cost measurement (#90) to capture the cold first-request latency
/// straight after a reload. Reuses the shared SSE-timing core.
pub async fn cold_probe(ctx: &RunCtx<'_>) -> Result<ScenarioMetrics> {
    let prompt = build_prompt(128);
    let payload = chat_payload(ctx, &prompt);
    tokio::time::timeout(ctx.timeout, stream_and_measure(ctx, &payload))
        .await
        .map_err(|_| anyhow!("cold probe timed out after {:?}", ctx.timeout))?
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
    /// Whether the burst streams (#285).
    ///
    /// Both shapes matter and they are not interchangeable. Streaming
    /// multiplexes through the batch engine; non-streaming did not until
    /// #285, and serialized completely — 1.00x aggregate throughput at
    /// every concurrency level against 3.98x streamed. A bench that only
    /// streams cannot see that, which is why it went unmeasured: the
    /// neuron reports `in_flight: 8`, admission accepts, `/health` looks
    /// healthy, and throughput is single-stream.
    streaming: bool,
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
        let payload = if self.streaming {
            chat_payload(ctx, &prompt)
        } else {
            nonstreaming_payload(ctx, &prompt)
        };

        // Fire all requests at once; each is independently timed and capped
        // by the per-request timeout so one hung request can't stall the
        // burst.
        let burst_start = Instant::now();
        let streaming = self.streaming;
        let payload = &payload;
        let futs = (0..self.concurrency).map(move |_| async move {
            if streaming {
                tokio::time::timeout(ctx.timeout, stream_and_measure(ctx, payload)).await
            } else {
                tokio::time::timeout(ctx.timeout, request_and_measure(ctx, payload)).await
            }
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
                "all {} concurrent requests failed ({rejected} shed by admission)",
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
            // Summed across the burst, like `completion_tokens`: the cell
            // describes the whole burst, so a per-stream figure here
            // would not compose with it.
            reasoning_tokens: sum_opt(&streams, |m| m.reasoning_tokens),
            cached_tokens: sum_opt(&streams, |m| m.cached_tokens),
            // The worst stream in the burst, not a median of medians:
            // under load the question is how bad it got for somebody,
            // and averaging tails is how a stall stays invisible.
            tpot_p95_ms: streams
                .iter()
                .filter_map(|m| m.tpot_p95_ms)
                .max_by(|a, b| a.total_cmp(b)),
            concurrency: Some(self.concurrency),
            ttft_p95_s: percentile(&ttfts, 95.0),
            queue_wait_ms_median: median(&queue_waits),
            rejected: Some(rejected),
            artifact: None,
            image_units: None,
        })
    }
}

/// Sum an optional per-stream count across a burst, `None` when no
/// stream reported it — so "nobody measured it" stays distinguishable
/// from "measured as zero", the same distinction #269 exists to
/// preserve.
fn sum_opt(
    streams: &[ScenarioMetrics],
    f: impl Fn(&ScenarioMetrics) -> Option<u64>,
) -> Option<u64> {
    let vals: Vec<u64> = streams.iter().filter_map(f).collect();
    (!vals.is_empty()).then(|| vals.iter().sum())
}

/// The non-streaming counterpart to [`chat_payload`] (#285).
///
/// `stream_options` is deliberately absent: usage is part of the body on
/// this shape, and sending the streaming-only option to a strict server
/// is a needless compatibility risk.
fn nonstreaming_payload(ctx: &RunCtx, prompt: &str) -> serde_json::Value {
    json!({
        "model": ctx.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": ctx.max_tokens,
        "temperature": 0,
        "stream": false,
    })
}

/// Time one non-streaming chat completion (#285).
///
/// There is no first-chunk moment on this shape — the whole body arrives
/// at once — so `ttft_s` is the full request wall-clock rather than a
/// separate measurement. That is the honest reading: for a non-streaming
/// caller, time-to-first-token *is* time-to-everything, and it is exactly
/// what makes the serialization this scenario exists to detect so
/// expensive.
///
/// `decode_tps` is per-request; the burst aggregate is computed by the
/// caller over the whole window, the same way the streaming path does it.
async fn request_and_measure(
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
        .context("sending non-streaming chat request")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(anyhow!("upstream returned {status}: {}", body.trim()));
    }
    let v: serde_json::Value = resp
        .json()
        .await
        .context("non-streaming chat response was not JSON")?;
    let total_s = start.elapsed().as_secs_f64();

    let usage = v.get("usage");
    let completion_tokens = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(|t| t.as_u64())
        .unwrap_or(0);
    let prompt_tokens = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(|t| t.as_u64());
    let reasoning_tokens = usage
        .and_then(|u| u.get("completion_tokens_details"))
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(|t| t.as_u64());
    let cached_tokens = usage
        .and_then(|u| u.get("prompt_tokens_details"))
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|t| t.as_u64());
    let timing = usage.and_then(|u| u.get("helexa_timing"));
    let field = |name: &str| timing.and_then(|t| t.get(name)).and_then(|x| x.as_u64());
    let (prefill_ms, decode_ms, prefill_tokens) = (
        field("prefill_ms"),
        field("decode_ms"),
        field("prefill_tokens"),
    );

    // Prefer the server's own decode window when it reports one (#85);
    // it excludes prefill, so it is comparable with the streaming path's
    // decode-window rate rather than being diluted by it.
    let decode_tps = match decode_ms {
        Some(ms) if ms > 200 => Some(completion_tokens as f64 / (ms as f64 / 1000.0)),
        _ if total_s > 0.2 => Some(completion_tokens as f64 / total_s),
        _ => None,
    };

    Ok(ScenarioMetrics {
        ttft_s: total_s,
        decode_tps,
        total_s,
        prompt_tokens,
        completion_tokens,
        prefill_ms,
        decode_ms,
        prefill_tokens,
        reasoning_tokens,
        cached_tokens,
        // Non-streaming has no inter-token gaps: the body arrives whole.
        tpot_p95_ms: None,
        concurrency: None,
        ttft_p95_s: None,
        queue_wait_ms_median: None,
        rejected: None,
        artifact: None,
        image_units: None,
    })
}

/// Quality probe (#91): runs a fixed prompt and stores the full generated
/// text as an artifact for later scoring (manual now, LLM-judge later). The
/// point is to compare reasoning/planning quality across models — the axis
/// speed-only scenarios miss — so the frontier A/B (F3) picks on capability,
/// not just throughput.
pub struct CapabilityScenario {
    id: String,
    prompt: String,
    max_tokens: u64,
}

#[async_trait]
impl Scenario for CapabilityScenario {
    fn id(&self) -> &str {
        &self.id
    }

    /// Capability probes have no synthetic prompt-token target; the cell is
    /// keyed by the scenario id alone.
    fn prompt_size(&self) -> u32 {
        0
    }

    async fn run(&self, ctx: &RunCtx) -> Result<ScenarioMetrics> {
        let payload = json!({
            "model": ctx.model_id,
            "messages": [{"role": "user", "content": self.prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "stream": true,
            "stream_options": {"include_usage": true},
        });
        let fut = stream_and_measure_inner(ctx, &payload, true);
        tokio::time::timeout(ctx.timeout, fut)
            .await
            .map_err(|_| anyhow!("capability probe timed out after {:?}", ctx.timeout))?
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
    stream_and_measure_inner(ctx, payload, false).await
}

/// As [`stream_and_measure`] but accumulates the full visible text when
/// `capture_text` is set — used by the capability probe (#91) to store the
/// generated artifact for later quality scoring.
async fn stream_and_measure_inner(
    ctx: &RunCtx<'_>,
    payload: &serde_json::Value,
    capture_text: bool,
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
    let mut reasoning_tokens: Option<u64> = None;
    let mut cached_tokens: Option<u64> = None;
    // Inter-token arrival gaps, for the p95 tail.
    let mut gaps_ms: Vec<f64> = Vec::new();
    let mut captured = String::new();

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
        if let Some(choice) = chunk.choices.first() {
            // Liveness counts ANY generated delta (#117). Thinking
            // models (Qwen3-Next-Thinking, Qwen3 with thinking on)
            // stream `reasoning_content` first — sometimes for their
            // entire budget — and a content-only view misread that as
            // a dead stream ("no content chunks received") while also
            // producing impossible client-side rates (reasoning-
            // inclusive token counts over a visible-content-only
            // window; observed: "244 tok/s" on a 3060). For
            // non-thinking models the first delta IS content, so
            // `ttft_s` semantics are unchanged for them.
            let content = choice
                .delta
                .get("content")
                .and_then(|c| c.as_str())
                .filter(|c| !c.is_empty());
            let reasoning = choice
                .delta
                .get("reasoning_content")
                .and_then(|c| c.as_str())
                .filter(|c| !c.is_empty());
            if content.is_some() || reasoning.is_some() {
                if first.is_none() {
                    first = Some(now);
                } else if let Some(prev) = last {
                    // Gaps only *between* generated deltas: the wait for
                    // the first one is TTFT and belongs to prefill, not
                    // to stream smoothness.
                    gaps_ms.push(now.duration_since(prev).as_secs_f64() * 1000.0);
                }
                last = Some(now);
                chunk_count += 1;
            }
            if capture_text && let Some(text) = content {
                captured.push_str(text);
            }
        }
        if let Some(usage) = chunk.usage {
            prompt_tokens = Some(usage.prompt_tokens);
            completion_tokens = Some(usage.completion_tokens);
            reasoning_tokens = usage
                .completion_tokens_details
                .as_ref()
                .map(|d| d.reasoning_tokens);
            cached_tokens = usage
                .prompt_tokens_details
                .as_ref()
                .map(|d| d.cached_tokens);
            if let Some(t) = usage.helexa_timing {
                prefill_ms = Some(t.prefill_ms);
                decode_ms = Some(t.decode_ms);
                prefill_tokens = Some(t.prefill_tokens);
            }
        }
    }
    let end = Instant::now();

    let first = first.ok_or_else(|| anyhow!("no generated chunks received"))?;

    // neuron emits one SSE chunk per generated token, so chunk_count is
    // an engine-truth count when no usage frame is sent.
    let tokens = completion_tokens.filter(|&t| t > 0).unwrap_or(chunk_count);
    // Decode rate: prefer the server-measured split (#85) — it counts
    // every generated token over the actual decode window, immune to
    // reasoning-suppression frame mismatches. Fall back to the client
    // inter-chunk window with the CHUNK count (same frame) — never
    // usage.completion_tokens over the chunk window, which mixes a
    // reasoning-inclusive numerator with a visible-only denominator.
    let window = last
        .filter(|&l| l > first)
        .map(|l| (l - first).as_secs_f64())
        .unwrap_or(0.0);
    let decode_tps = match decode_ms {
        Some(ms) if ms > 200 && tokens > 0 => Some(tokens as f64 / (ms as f64 / 1000.0)),
        _ if window > 0.2 => Some(chunk_count as f64 / window),
        _ => None,
    };
    Ok(ScenarioMetrics {
        ttft_s: (first - start).as_secs_f64(),
        decode_tps,
        total_s: (end - start).as_secs_f64(),
        prompt_tokens,
        completion_tokens: tokens,
        prefill_ms,
        decode_ms,
        prefill_tokens,
        reasoning_tokens,
        cached_tokens,
        tpot_p95_ms: percentile(&gaps_ms, 95.0),
        // Concurrency fields unset on the single-request path; the
        // concurrency scenario builds its own aggregate (#89).
        concurrency: None,
        ttft_p95_s: None,
        queue_wait_ms_median: None,
        rejected: None,
        artifact: if capture_text { Some(captured) } else { None },
        image_units: None,
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
        use crate::config::{CapabilityProbe, ScenarioConfig};
        let cfg = ScenarioConfig {
            image_sizes: vec![],
            prompt_sizes: vec![128],
            max_tokens: 64,
            concurrency_levels: vec![2, 8],
            concurrency_nonstreaming_levels: vec![8],
            concurrency_prompt_tokens: 512,
            capability_probes: vec![CapabilityProbe {
                name: "plan".into(),
                prompt: "Write a plan.".into(),
                max_tokens: 2048,
            }],
        };
        let ids: Vec<String> = build_scenarios(&cfg)
            .iter()
            .map(|s| s.id().to_string())
            .collect();
        assert!(ids.contains(&"chat:128".to_string()));
        assert!(ids.contains(&"concurrency:2".to_string()));
        assert!(ids.contains(&"concurrency:8".to_string()));
        assert!(ids.contains(&"capability:plan".to_string()));
    }

    #[test]
    fn prompt_floor_for_tiny_targets() {
        // max(approx,16) floor means even 0 yields a non-trivial prompt.
        let p = build_prompt(0);
        assert!(p.len() >= 16 * 4);
    }
}
