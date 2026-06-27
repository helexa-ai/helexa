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

/// Build the active scenario set from config. One chat-latency scenario
/// per configured prompt size.
pub fn build_scenarios(cfg: &ScenarioConfig) -> Vec<Box<dyn Scenario>> {
    cfg.prompt_sizes
        .iter()
        .map(|&size| {
            Box::new(ChatLatencyScenario {
                id: format!("chat:{size}"),
                approx_prompt_tokens: size,
            }) as Box<dyn Scenario>
        })
        .collect()
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
        let payload = json!({
            "model": ctx.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": ctx.max_tokens,
            "temperature": 0,
            "stream": true,
            "stream_options": {"include_usage": true},
        });

        let fut = stream_and_measure(ctx, &payload);
        tokio::time::timeout(ctx.timeout, fut)
            .await
            .map_err(|_| anyhow!("request timed out after {:?}", ctx.timeout))?
    }
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
    fn prompt_floor_for_tiny_targets() {
        // max(approx,16) floor means even 0 yields a non-trivial prompt.
        let p = build_prompt(0);
        assert!(p.len() >= 16 * 4);
    }
}
