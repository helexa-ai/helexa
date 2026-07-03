//! Lockstep batched decode engine (#98) — single-GPU worker path.
//!
//! One engine task per loaded model replaces the per-request
//! `inference_lock` serialization for **text** chat streams when the
//! operator raises `[admission] max_in_flight` above 1 (and the arch
//! supports cache snapshots — qwen3_5 only today). The engine owns the
//! model's forward exclusively and multiplexes up to `max_slots`
//! concurrent streams through one `(B, 1)` forward per decode step:
//!
//! - **Join** (new request): prefill runs alone at B=1 through the
//!   existing chunked-prefill + prefix-cache paths, then the fresh
//!   state is snapshotted and the batch re-assembled
//!   (`ExtractKvRows` survivors → `AssembleKvBatch` everyone). Decode
//!   for running slots stalls for the duration of the newcomer's
//!   prefill — the accepted v1 cost, bounded by chunked prefill.
//! - **Step**: one `ForwardLogitsBatch` job; per-slot CPU sampling
//!   (each slot has its own `LogitsProcessor` + repeat-penalty
//!   history); sampled tokens go to per-slot **router tasks** that own
//!   the incremental detokenizer and the reasoning/tool-call state
//!   machine and emit `InferenceEvent`s on the request's channel.
//! - **Leave** (EOS / length / consumer hangup): the slot's Finish is
//!   emitted and the batch compacts at the next rebatch point (which
//!   runs immediately after any step that finished a slot).
//!
//! Routers are separate tasks (not inline state) because
//! `tokenizers::DecodeStream` borrows the tokenizer and carries five
//! generic parameters — owning both inside one async block sidesteps
//! the self-referential-struct problem the same way the
//! `route_token!` macro does at its call sites, and decouples slow
//! consumers from the lockstep loop.
//!
//! A worker error mid-step is fatal for the whole engine: every
//! active slot's stream ends, the model is marked poisoned when the
//! error classifies as a device fault, and the engine exits (later
//! submits fail fast).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

use super::admission::AdmissionPermit;
use super::candle::{
    ModelPrefixCache, ToolCallMarker, ToolSchemas, chunked_prefill_via_worker, emit_delta,
    handle_reasoning_marker, handle_tool_call_marker, is_device_fault, logits_health_slice,
    parse_tool_call_body, prompt_opens_reasoning, restore_or_clear_via_worker, sample_with_penalty,
    stable_snapshot_cut, store_prefix_snapshot_via_worker,
};
use super::context_limit::PrefillRateEma;
use super::device_worker::{ArchHandle, DeviceWorkerHandle};
use crate::wire::event::{
    FinishReason, FinishTiming, InferenceEvent, ReasoningTokenPair, ToolCallTokenPair,
};

/// Runtime kill switch: `NEURON_BATCHING=0` (or `false`) keeps the
/// per-request `inference_lock` path even when `max_in_flight > 1`.
/// Read once.
pub fn batching_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let on = !std::env::var("NEURON_BATCHING").is_ok_and(|v| v == "0" || v == "false");
        tracing::info!(enabled = on, "batched decode engine (#98)");
        on
    })
}

/// Everything the engine needs that is per-model (not per-request).
/// Deliberately does NOT hold `Arc<LoadedModel>` — the engine task
/// must not keep the model alive (the task exits when the model drops
/// its `EngineHandle` and the channel closes).
pub struct EngineConfig {
    pub model_id: String,
    pub worker: Arc<DeviceWorkerHandle>,
    pub handle: ArchHandle,
    pub tokenizer: Tokenizer,
    pub prefix_cache: Option<Arc<ModelPrefixCache>>,
    pub prefill_rate: Arc<PrefillRateEma>,
    pub reasoning_tokens: Option<ReasoningTokenPair>,
    pub tool_call_tokens: Option<ToolCallTokenPair>,
    /// Shared with `LoadedModel.poisoned` so a device fault inside the
    /// engine fast-rejects subsequent requests at the harness boundary.
    pub poisoned: Arc<AtomicBool>,
    /// Shared with `LoadedModel.inference_lock`. Held for the whole
    /// active phase (first join → last slot finished) so the
    /// non-engine forward paths (vision, non-streaming chat) can never
    /// clobber the live batched cache state mid-decode. Released while
    /// idle.
    pub inference_lock: Arc<tokio::sync::Mutex<()>>,
    pub max_slots: usize,
}

/// One queued request. Admission has already been passed — the permit
/// rides along and is released when the slot finishes.
pub struct EngineRequest {
    pub prompt_tokens: Vec<u32>,
    pub max_new: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub seed: u64,
    pub eos_id: Option<u32>,
    pub tool_schemas: ToolSchemas,
    pub tx: mpsc::Sender<InferenceEvent>,
    pub admit: AdmissionPermit,
    pub span: tracing::Span,
}

/// Cheap handle held by `LoadedModel`. Submitting fails once the
/// engine task has exited (fatal worker error) — callers surface that
/// as an inference error; the model is typically poisoned by then.
#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<EngineRequest>,
}

impl EngineHandle {
    pub fn spawn(cfg: EngineConfig) -> Self {
        // Depth beyond max_slots only buffers between admission and the
        // engine momentarily; admission's queue is the real bound.
        let (tx, rx) = mpsc::channel::<EngineRequest>(cfg.max_slots.max(1) * 2);
        tokio::spawn(run_engine(cfg, rx));
        Self { tx }
    }

    pub async fn submit(&self, req: EngineRequest) -> Result<()> {
        self.tx
            .send(req)
            .await
            .map_err(|_| anyhow::anyhow!("batch engine is not running (model poisoned?)"))
    }
}

/// Messages from the engine loop to a slot's router task.
enum RouterMsg {
    Token(u32),
    Finish {
        reason: FinishReason,
        prompt_tokens: u32,
        completion_tokens: u32,
        timing: FinishTiming,
    },
}

struct Slot {
    /// Contiguous valid tokens this row held at the last rebatch
    /// (prompt + tokens decoded before that point).
    prefix_len: usize,
    prompt_len: usize,
    /// Completion tokens so far — the repeat-penalty history. EOS is
    /// never pushed (mirrors the B=1 paths).
    generated: Vec<u32>,
    next_token: u32,
    max_new: usize,
    eos_id: Option<u32>,
    lp: LogitsProcessor,
    router: mpsc::Sender<RouterMsg>,
    /// Set by the router when the consumer hangs up; the engine stops
    /// feeding the slot and compacts it out.
    hangup: Arc<AtomicBool>,
    finished: Option<FinishReason>,
    prefill_ms: u32,
    prefill_tokens: u32,
    decode_start: std::time::Instant,
    _admit: AdmissionPermit,
}

impl Slot {
    fn finish(&mut self, reason: FinishReason) {
        self.finished = Some(reason);
    }
}

async fn run_engine(cfg: EngineConfig, mut rx: mpsc::Receiver<EngineRequest>) {
    let mut slots: Vec<Slot> = Vec::new();
    // Uniform padded KV length of the current batch and steps decoded
    // since the last rebatch — the geometry `ForwardLogitsBatch` and
    // `ExtractKvRows` key off.
    let mut padded_len = 0usize;
    let mut step = 0usize;
    // Held while any slot is active — see `EngineConfig.inference_lock`.
    let mut lock_guard: Option<tokio::sync::OwnedMutexGuard<()>> = None;

    tracing::info!(
        model = %cfg.model_id,
        max_slots = cfg.max_slots,
        "batch engine started"
    );

    'main: loop {
        // Gather joins: block when idle, drain opportunistically when
        // busy. Slots that finished or hung up leave at the rebatch
        // below.
        let mut joins: Vec<EngineRequest> = Vec::new();
        if slots.is_empty() {
            match rx.recv().await {
                Some(r) => joins.push(r),
                None => break 'main, // model unloaded
            }
        }
        while slots.len() + joins.len() < cfg.max_slots {
            match rx.try_recv() {
                Ok(r) => joins.push(r),
                Err(_) => break,
            }
        }

        // Take the model's inference lock before touching cache state;
        // release it whenever the batch drains so vision/non-streaming
        // requests get their turn.
        if !joins.is_empty() && lock_guard.is_none() {
            lock_guard = Some(Arc::clone(&cfg.inference_lock).lock_owned().await);
        }

        let needs_compaction = slots
            .iter()
            .any(|s| s.finished.is_some() || s.hangup.load(Ordering::Acquire));
        if (!joins.is_empty() || needs_compaction)
            && let Err(e) = rebatch(&cfg, &mut slots, joins, &mut padded_len, &mut step).await
        {
            fail_engine(&cfg, &mut slots, &mut rx, &e);
            break 'main;
        }
        if slots.is_empty() {
            lock_guard = None; // every join finished during prefill
            continue;
        }

        // One lockstep decode step.
        let tokens: Vec<u32> = slots.iter().map(|s| s.next_token).collect();
        let prefix_lens: Vec<usize> = slots.iter().map(|s| s.prefix_len).collect();
        let rows = match cfg
            .worker
            .forward_logits_batch(cfg.handle, tokens, prefix_lens, padded_len, step)
            .await
        {
            Ok(rows) => rows,
            Err(e) => {
                let e = anyhow::anyhow!("batched decode step {step}: {e}");
                fail_engine(&cfg, &mut slots, &mut rx, &e);
                break 'main;
            }
        };
        step += 1;

        let mut fatal: Option<anyhow::Error> = None;
        for (slot, logits_vec) in slots.iter_mut().zip(rows) {
            if slot.finished.is_some() || slot.hangup.load(Ordering::Acquire) {
                // Compacted out at the next rebatch; discard its row.
                continue;
            }
            let logits = match Tensor::new(logits_vec.as_slice(), &Device::Cpu) {
                Ok(t) => t,
                Err(e) => {
                    fatal = Some(e.into());
                    break;
                }
            };
            let nt = match sample_with_penalty(&logits, &slot.generated, &mut slot.lp) {
                Ok(t) => t,
                Err(e) => {
                    let health = logits_health_slice(&logits_vec);
                    tracing::warn!(
                        ?health,
                        error = %e,
                        "batch engine: sample failed; logits unhealthy"
                    );
                    // Unhealthy logits are a device-level problem —
                    // fail the whole engine, mirroring the B=1 path's
                    // poison classification.
                    fatal = Some(e);
                    break;
                }
            };
            if Some(nt) == slot.eos_id {
                finish_slot(slot, FinishReason::Stop).await;
                continue;
            }
            slot.generated.push(nt);
            slot.next_token = nt;
            if slot.router.send(RouterMsg::Token(nt)).await.is_err() {
                // Router exited (consumer hung up mid-drain).
                slot.hangup.store(true, Ordering::Release);
                slot.finish(FinishReason::Stop);
                continue;
            }
            if slot.generated.len() >= slot.max_new {
                finish_slot(slot, FinishReason::Length).await;
            }
        }
        if let Some(e) = fatal {
            fail_engine(&cfg, &mut slots, &mut rx, &e);
            break 'main;
        }
    }

    tracing::info!(model = %cfg.model_id, "batch engine stopped");
}

/// Emit the slot's Finish through its router and mark it for
/// compaction.
async fn finish_slot(slot: &mut Slot, reason: FinishReason) {
    slot.finish(reason);
    let _ = slot
        .router
        .send(RouterMsg::Finish {
            reason,
            prompt_tokens: slot.prompt_len as u32,
            completion_tokens: slot.generated.len() as u32,
            timing: FinishTiming {
                prefill_ms: slot.prefill_ms,
                decode_ms: slot.decode_start.elapsed().as_millis() as u32,
                prefill_tokens: slot.prefill_tokens,
            },
        })
        .await;
}

/// Fatal-path teardown: classify + record the poison flag, end every
/// active stream (routers exit when their channel drops without a
/// Finish), and drain queued requests so their clients aren't left
/// hanging on a dead channel.
fn fail_engine(
    cfg: &EngineConfig,
    slots: &mut Vec<Slot>,
    rx: &mut mpsc::Receiver<EngineRequest>,
    error: &anyhow::Error,
) {
    let chain = format!("{error:#}");
    if is_device_fault(&chain) {
        cfg.poisoned.store(true, Ordering::Release);
        tracing::error!(
            model = %cfg.model_id,
            error = %chain,
            "batch engine: device fault, model marked poisoned"
        );
    } else {
        tracing::error!(
            model = %cfg.model_id,
            error = %chain,
            "batch engine: fatal error (non-device fault)"
        );
    }
    slots.clear();
    rx.close();
    while let Ok(req) = rx.try_recv() {
        drop(req); // dropping tx ends the client stream
    }
}

/// Rebatch point: drop finished/hung slots, extract survivors from the
/// live batched state, prefill every join at B=1, and assemble the new
/// batch. On return `step == 0` and `padded_len` describes the new
/// geometry.
async fn rebatch(
    cfg: &EngineConfig,
    slots: &mut Vec<Slot>,
    joins: Vec<EngineRequest>,
    padded_len: &mut usize,
    step: &mut usize,
) -> Result<()> {
    // 1. Extract survivors BEFORE any prefill clobbers the live state.
    let mut kept: Vec<Slot> = Vec::new();
    let mut extracted: Vec<(super::device_worker::jobs::KvSnapshotId, usize)> = Vec::new();
    let leavers_or_joiners = joins.len()
        + slots
            .iter()
            .filter(|s| s.finished.is_some() || s.hangup.load(Ordering::Acquire))
            .count();
    let survivors: Vec<usize> = slots
        .iter()
        .enumerate()
        .filter(|(_, s)| s.finished.is_none() && !s.hangup.load(Ordering::Acquire))
        .map(|(i, _)| i)
        .collect();
    if !survivors.is_empty() && leavers_or_joiners > 0 {
        let rows: Vec<(usize, usize)> = survivors
            .iter()
            .map(|&i| (i, slots[i].prefix_len))
            .collect();
        let ids = cfg
            .worker
            .extract_kv_rows(cfg.handle, rows, *padded_len, *step)
            .await
            .map_err(|e| anyhow::anyhow!("extract_kv_rows: {e}"))?;
        for (&i, (id, _bytes)) in survivors.iter().zip(ids) {
            let new_len = slots[i].prefix_len + *step;
            extracted.push((id, new_len));
        }
    }
    // Drain slots preserving survivor order; finished/hung slots drop
    // here (permits release, routers wind down).
    for (order, i) in survivors.iter().enumerate() {
        debug_assert!(*i >= order);
        let mut s = slots.remove(*i - order);
        s.prefix_len += *step;
        kept.push(s);
    }
    slots.clear();

    // 2. Prefill each join at B=1 (prefix cache + chunked prefill
    //    exactly as the per-request path).
    let mut assemble: Vec<(super::device_worker::jobs::KvSnapshotId, usize)> = extracted.clone();
    for req in joins {
        let req_span = req.span.clone();
        // `None` = finished during prefill (EOS / hangup / max_new 0).
        if let Some((slot, snap_id)) = prefill_join(cfg, req).instrument_in(req_span).await? {
            assemble.push((snap_id, slot.prompt_len));
            kept.push(slot);
        }
    }

    // 3. Assemble the new batch (or go idle).
    if kept.is_empty() {
        // Nothing active. Temp snapshots for extraction are dropped.
        for (id, _) in &assemble {
            let _ = cfg.worker.drop_kv_snapshot(cfg.handle, *id).await;
        }
        *padded_len = 0;
        *step = 0;
        return Ok(());
    }
    let seqs: Vec<(super::device_worker::jobs::KvSnapshotId, usize)> = assemble.clone();
    let new_padded = cfg
        .worker
        .assemble_kv_batch(cfg.handle, seqs)
        .await
        .map_err(|e| anyhow::anyhow!("assemble_kv_batch: {e}"))?;
    for (id, _) in &assemble {
        let _ = cfg.worker.drop_kv_snapshot(cfg.handle, *id).await;
    }
    *padded_len = new_padded;
    *step = 0;
    *slots = kept;
    Ok(())
}

/// Prefill one joining request at B=1 and snapshot its state for
/// assembly. Returns `None` when the request finished during prefill
/// (EOS as first token, `max_new == 0`, or the consumer already hung
/// up) — its Finish has been emitted and no slot joins the batch.
async fn prefill_join(
    cfg: &EngineConfig,
    req: EngineRequest,
) -> Result<Option<(Slot, super::device_worker::jobs::KvSnapshotId)>> {
    use candle_transformers::generation::Sampling;

    let EngineRequest {
        prompt_tokens,
        max_new,
        temperature,
        top_p,
        seed,
        eos_id,
        tool_schemas,
        tx,
        admit,
        span,
    } = req;

    let mut lp = {
        let sampling = if temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            match top_p {
                Some(p) => Sampling::TopP { p, temperature },
                None => Sampling::All { temperature },
            }
        };
        LogitsProcessor::from_sampling(seed, sampling)
    };

    let prefix_cache = cfg.prefix_cache.as_deref();
    let prompt_len = prompt_tokens.len();
    let prefill_start = std::time::Instant::now();
    let reused =
        restore_or_clear_via_worker(&cfg.worker, cfg.handle, prefix_cache, &prompt_tokens).await?;
    let cut = if prefix_cache.is_some() {
        stable_snapshot_cut(&prompt_tokens, cfg.tokenizer.token_to_id("<|im_start|>"))
            .filter(|&c| c > reused)
    } else {
        None
    };
    let logits_vec = match cut {
        Some(c) => {
            chunked_prefill_via_worker(&cfg.worker, cfg.handle, &prompt_tokens[..c], reused)
                .await?;
            store_prefix_snapshot_via_worker(
                &cfg.worker,
                cfg.handle,
                prefix_cache,
                prompt_tokens[..c].to_vec(),
            )
            .await;
            chunked_prefill_via_worker(&cfg.worker, cfg.handle, &prompt_tokens, c).await?
        }
        None => chunked_prefill_via_worker(&cfg.worker, cfg.handle, &prompt_tokens, reused).await?,
    };
    let prefill_elapsed = prefill_start.elapsed();
    cfg.prefill_rate.record(prompt_len, prefill_elapsed);

    // First token from the prefill logits.
    let generated: Vec<u32> = Vec::new();
    let logits = Tensor::new(logits_vec.as_slice(), &Device::Cpu)?;
    let first = match sample_with_penalty(&logits, &generated, &mut lp) {
        Ok(t) => t,
        Err(e) => {
            let health = logits_health_slice(&logits_vec);
            tracing::warn!(
                ?health,
                "batch engine: prefill sample failed; logits unhealthy"
            );
            return Err(e);
        }
    };

    // Router task for this slot.
    let hangup = Arc::new(AtomicBool::new(false));
    let (router_tx, router_rx) = mpsc::channel::<RouterMsg>(1024);
    let starts_in_reasoning = prompt_opens_reasoning(&prompt_tokens, cfg.reasoning_tokens.as_ref());
    tokio::spawn(
        run_router(
            cfg.tokenizer.clone(),
            cfg.reasoning_tokens.clone(),
            cfg.tool_call_tokens.clone(),
            tool_schemas,
            starts_in_reasoning,
            tx,
            Arc::clone(&hangup),
            router_rx,
        )
        .instrument_in(span.clone()),
    );

    let mut slot = Slot {
        prefix_len: prompt_len,
        prompt_len,
        generated,
        next_token: first,
        max_new,
        eos_id,
        lp,
        router: router_tx,
        hangup,
        finished: None,
        prefill_ms: prefill_elapsed.as_millis() as u32,
        prefill_tokens: prompt_len as u32,
        decode_start: std::time::Instant::now(),
        _admit: admit,
    };

    // First-token bookkeeping mirrors the B=1 path: EOS finishes
    // without routing; max_new bounds include the first token.
    if Some(first) == slot.eos_id || slot.max_new == 0 {
        let reason = if slot.max_new == 0 {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };
        finish_slot(&mut slot, reason).await;
        return Ok(None);
    }
    slot.generated.push(first);
    if slot.router.send(RouterMsg::Token(first)).await.is_err() {
        return Ok(None); // consumer already gone
    }
    if slot.generated.len() >= slot.max_new {
        finish_slot(&mut slot, FinishReason::Length).await;
        return Ok(None);
    }

    // Snapshot the freshly prefilled state for assembly.
    let (snap_id, _bytes) = cfg
        .worker
        .snapshot_kv(cfg.handle)
        .await
        .map_err(|e| anyhow::anyhow!("snapshot after prefill: {e}"))?;
    Ok(Some((slot, snap_id)))
}

/// Per-slot router: owns the incremental detokenizer and the
/// reasoning/tool-call state machine (the same logic as the
/// `route_token!` macro in the B=1 stream path) and emits
/// `InferenceEvent`s on the request's channel. Sets `hangup` and
/// drains silently once the consumer goes away.
#[allow(clippy::too_many_arguments)]
async fn run_router(
    tokenizer: Tokenizer,
    reasoning_tokens: Option<ReasoningTokenPair>,
    tool_call_tokens: Option<ToolCallTokenPair>,
    tool_schemas: ToolSchemas,
    starts_in_reasoning: bool,
    tx: mpsc::Sender<InferenceEvent>,
    hangup: Arc<AtomicBool>,
    mut rx: mpsc::Receiver<RouterMsg>,
) {
    let mut decode_stream = tokenizer.decode_stream(true);
    let mut in_reasoning = starts_in_reasoning;
    let mut reasoning_token_count: u32 = 0;
    let mut in_tool_call = false;
    let mut tool_call_buf = String::new();
    let mut tool_call_idx: usize = 0;
    let mut emitted_tool_call = false;
    let mut consumer_alive = true;

    while let Some(msg) = rx.recv().await {
        match msg {
            RouterMsg::Token(nt) => {
                if !consumer_alive {
                    continue; // drain
                }
                'route: {
                    match handle_tool_call_marker(
                        nt,
                        tool_call_tokens.as_ref(),
                        &mut in_tool_call,
                        &mut tool_call_buf,
                    ) {
                        ToolCallMarker::Enter => break 'route,
                        ToolCallMarker::Exit { buffer } => {
                            let idx = tool_call_idx;
                            tool_call_idx += 1;
                            match parse_tool_call_body(&buffer, idx, &tool_schemas) {
                                Some((id, name, arguments)) => {
                                    emitted_tool_call = true;
                                    if tx
                                        .send(InferenceEvent::ToolCall {
                                            index: idx,
                                            id,
                                            name,
                                            arguments,
                                        })
                                        .await
                                        .is_err()
                                    {
                                        consumer_alive = false;
                                    }
                                }
                                None => {
                                    let open = tool_call_tokens
                                        .as_ref()
                                        .map(|p| p.open_text.as_str())
                                        .unwrap_or("<tool_call>");
                                    let close = tool_call_tokens
                                        .as_ref()
                                        .map(|p| p.close_text.as_str())
                                        .unwrap_or("</tool_call>");
                                    let raw = format!("{open}{buffer}{close}");
                                    if !emit_delta(&raw, &tx, in_reasoning).await {
                                        consumer_alive = false;
                                    }
                                }
                            }
                            break 'route;
                        }
                        ToolCallMarker::None => {}
                    }
                    if in_tool_call {
                        match decode_stream.step(nt) {
                            Ok(Some(s)) => tool_call_buf.push_str(&s),
                            Ok(None) => {}
                            Err(e) => tracing::warn!(
                                error = %e,
                                "decode_stream step failed (in tool_call)"
                            ),
                        }
                        break 'route;
                    }
                    if handle_reasoning_marker(nt, reasoning_tokens.as_ref(), &mut in_reasoning) {
                        break 'route;
                    }
                    if in_reasoning {
                        reasoning_token_count += 1;
                    }
                    match decode_stream.step(nt) {
                        Ok(Some(delta)) => {
                            if !emit_delta(&delta, &tx, in_reasoning).await {
                                consumer_alive = false;
                            }
                        }
                        Ok(None) => {}
                        Err(e) => tracing::warn!(error = %e, "decode_stream step failed"),
                    }
                }
                if !consumer_alive {
                    hangup.store(true, Ordering::Release);
                }
            }
            RouterMsg::Finish {
                mut reason,
                prompt_tokens,
                completion_tokens,
                timing,
            } => {
                if emitted_tool_call && reason == FinishReason::Stop {
                    reason = FinishReason::ToolCalls;
                }
                let _ = tx
                    .send(InferenceEvent::Finish {
                        reason,
                        prompt_tokens,
                        completion_tokens,
                        reasoning_tokens: reasoning_token_count,
                        timing: Some(timing),
                    })
                    .await;
                break;
            }
        }
    }
}

/// `tracing::Instrument` without importing the trait at every use
/// site.
trait InstrumentExt: Sized + std::future::Future {
    fn instrument_in(self, span: tracing::Span) -> tracing::instrument::Instrumented<Self> {
        tracing::Instrument::instrument(self, span)
    }
}
impl<F: std::future::Future> InstrumentExt for F {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AdmissionConfig;
    use crate::harness::admission::AdmissionController;

    /// A WordLevel tokenizer whose vocab covers the whole fixture
    /// vocab (`w0`..`w511`), so every decoded token maps to a unique
    /// word and the emitted text uniquely encodes the token sequence.
    fn tiny_tokenizer(vocab_size: usize) -> Tokenizer {
        // WordLevel's builder vocab type (AHashMap) is private, so go
        // through the vocab-file loader instead.
        let vocab: std::collections::HashMap<String, u32> = (0..vocab_size as u32)
            .map(|i| (format!("w{i}"), i))
            .collect();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("vocab.json");
        std::fs::write(&path, serde_json::to_string(&vocab).expect("vocab json"))
            .expect("write vocab");
        let model = tokenizers::models::wordlevel::WordLevel::from_file(
            path.to_str().expect("utf8 path"),
            "w0".into(),
        )
        .expect("build WordLevel");
        Tokenizer::new(tokenizers::ModelWrapper::WordLevel(model))
    }

    async fn collect_run(
        engine: &EngineHandle,
        admission: &AdmissionController,
        prompt: Vec<u32>,
        max_new: usize,
    ) -> (String, u32, FinishReason) {
        let admit = admission.enter(None).await.expect("admitted");
        let (tx, mut rx) = mpsc::channel::<InferenceEvent>(32);
        engine
            .submit(EngineRequest {
                prompt_tokens: prompt,
                max_new,
                temperature: 0.0, // greedy — deterministic
                top_p: None,
                seed: 0,
                eos_id: None,
                tool_schemas: ToolSchemas::new(),
                tx,
                admit,
                span: tracing::Span::none(),
            })
            .await
            .expect("submit");
        let mut text = String::new();
        loop {
            match rx.recv().await {
                Some(InferenceEvent::TextDelta(d)) | Some(InferenceEvent::ReasoningDelta(d)) => {
                    text.push_str(&d)
                }
                Some(InferenceEvent::Finish {
                    reason,
                    completion_tokens,
                    ..
                }) => return (text, completion_tokens, reason),
                Some(_) => {}
                None => panic!("stream ended without Finish"),
            }
        }
    }

    /// The engine's gold test: three greedy requests submitted
    /// concurrently (batched lockstep decode, ragged prompts, joins
    /// mid-flight) must produce byte-identical output to the same
    /// requests run one-at-a-time through the same engine.
    #[tokio::test]
    async fn concurrent_engine_output_matches_sequential() {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/numerical/qwen3_next-tiny");
        if !fixture.join("model.safetensors").exists() {
            eprintln!("SKIP concurrent_engine_output_matches_sequential: fixture not generated");
            return;
        }
        let worker = DeviceWorkerHandle::spawn(0).expect("spawn worker");
        let handle = worker
            .load_dense(
                fixture.join("config.json"),
                vec![fixture.join("model.safetensors")],
                "qwen3_next-tiny".into(),
            )
            .await
            .expect("load fixture");

        let admission_cfg = AdmissionConfig {
            max_in_flight: 3,
            ..Default::default()
        };
        let admission = AdmissionController::new(&admission_cfg);
        let engine = EngineHandle::spawn(EngineConfig {
            model_id: "qwen3_next-tiny".into(),
            worker: Arc::clone(&worker),
            handle,
            tokenizer: tiny_tokenizer(512),
            prefix_cache: None,
            prefill_rate: Arc::new(PrefillRateEma::new()),
            reasoning_tokens: None,
            tool_call_tokens: None,
            poisoned: Arc::new(AtomicBool::new(false)),
            inference_lock: Arc::new(tokio::sync::Mutex::new(())),
            max_slots: 3,
        });

        let prompts: [&[u32]; 3] = [&[1, 2, 3], &[4, 5], &[7, 3, 2, 5, 6]];
        let max_new = 6;

        // Sequential reference: one at a time through the same engine.
        let mut expected = Vec::new();
        for p in prompts {
            expected.push(collect_run(&engine, &admission, p.to_vec(), max_new).await);
        }

        // Concurrent: all three at once — they batch.
        let futs: Vec<_> = prompts
            .iter()
            .map(|p| collect_run(&engine, &admission, p.to_vec(), max_new))
            .collect();
        let got = futures::future::join_all(futs).await;

        for (i, (want, got)) in expected.iter().zip(got.iter()).enumerate() {
            assert_eq!(want.2, got.2, "request {i} finish reason");
            assert_eq!(want.1, got.1, "request {i} completion tokens");
            assert_eq!(want.0, got.0, "request {i} text");
        }
        assert!(
            expected
                .iter()
                .all(|(t, n, _)| !t.is_empty() && *n as usize == max_new),
            "greedy runs should hit the length cap with non-empty text: {expected:?}"
        );
    }
}
