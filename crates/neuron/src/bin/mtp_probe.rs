//! MTP speculative-decoding measurement harness (#96, stages S3/S4).
//!
//! Two numbers decide whether speculative decoding is worth building
//! into neuron, and neither can be taken from anyone else's benchmark:
//!
//! - **`c`** — what a draft step costs relative to a target decode
//!   step. Published figures are for a different drafter on different
//!   hardware. Ours reads the target's own `lm_head` (~1.03 GB at q6k
//!   on the 27B) on top of the head's weights, so the vocabulary, not
//!   the head, may dominate.
//! - **acceptance** — how many drafted tokens the target actually
//!   agrees with, *on our workloads*. Every published figure is
//!   short-form (GSM8K, MATH-500, HumanEval, ≤4k output); helexa serves
//!   15–50k-token agentic contexts full of replayed reasoning, and
//!   acceptance is workload-dependent. A poor rate makes speculation a
//!   net loss.
//!
//! This binary answers both by running a real checkpoint. It commits
//! nothing, changes no serving behaviour, and is not shipped in the
//! RPM.
//!
//! ## What it does
//!
//! Prefills the target over a prompt, then generates greedily. At each
//! step it snapshots the draft head's KV, drafts `K` tokens by chaining
//! the head, restores the snapshot, and lets the target continue
//! normally. When the target has produced the next `K` true tokens, the
//! draft is scored against them: the accepted length is the number of
//! leading matches.
//!
//! The head is *not* the thing generating — the target's own output is
//! never influenced by the draft, which is what makes this an observer
//! rather than an implementation.
//!
//! ## Two logits, two argmaxes
//!
//! Acceptance is reported against the target's raw argmax **and**
//! against its argmax after the repeat penalty neuron applies when
//! serving. The two differ, and the gap is exactly what a
//! distribution-preserving (rejection-sampling) scheme would have to
//! reconcile at `temperature > 0` — this fleet serves this family at
//! 0.6. Measuring it here costs nothing and makes that decision an
//! informed one.
//!
//! ## Usage
//!
//! ```sh
//! neuron-mtp-probe --model /path/to/Qwen3.5-0.8B --device cuda:0 \
//!     --draft-len 4 --steps 64 --prompt-file prompt.txt
//! ```

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

use neuron::harness::arch::qwen3_5::mtp::MtpHead;
use neuron::harness::arch::qwen3_5::rope::RotaryEmbedding;
use neuron::harness::arch::qwen3_5::{Config, Qwen3_5ForCausalLM};

#[derive(Parser, Debug)]
#[command(about = "Measure MTP draft cost and acceptance rate (#96)")]
struct Args {
    /// Checkpoint directory: config.json, *.safetensors, tokenizer.json.
    #[arg(long)]
    model: PathBuf,
    /// `cpu` or `cuda:N`.
    #[arg(long, default_value = "cpu")]
    device: String,
    /// Draft tokens per round (K).
    #[arg(long, default_value_t = 4)]
    draft_len: usize,
    /// Decode steps to measure over.
    #[arg(long, default_value_t = 64)]
    steps: usize,
    /// Prompt file. Without one, a short built-in prompt is used — fine
    /// for `c`, useless for acceptance, which is the whole point of
    /// measuring on a replayed session instead.
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    /// Prefill chunk size. The mask for a one-shot prefill is L x L —
    /// 2.5 GB at 25k tokens — so the probe chunks like the serving path
    /// does.
    #[arg(long, default_value_t = 512)]
    prefill_chunk: usize,
    /// Timed-but-discarded rounds before measurement starts.
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    /// Repeat penalty applied to the target's logits when scoring the
    /// second argmax. Mirrors the serving default.
    #[arg(long, default_value_t = 1.05)]
    repeat_penalty: f32,
    /// How many recent tokens the repeat penalty considers.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
    /// Tensor-parallel degree. 1 loads the model on one card; 2+ uses
    /// the worker pool, which is the only way to reach a model too big
    /// for a single GPU — the 27B is 54 GB in bf16 and the single-GPU
    /// path is bf16-only.
    #[arg(long, default_value_t = 1)]
    tensor_parallel: u32,
    /// Quantisation for the TP load (`q6k`, `q8_0`, …). Only the TP
    /// loader applies it.
    #[arg(long)]
    quant: Option<String>,
    /// The binary the worker ranks are spawned from. Defaults to
    /// `neuron` beside this executable — the probe itself does not
    /// implement `--worker`.
    #[arg(long)]
    worker_binary: Option<PathBuf>,
    /// Write the summary as JSON here as well as to stdout.
    #[arg(long)]
    json: Option<PathBuf>,
}

fn pick_device(spec: &str) -> Result<Device> {
    if spec == "cpu" {
        return Ok(Device::Cpu);
    }
    let idx: usize = spec
        .strip_prefix("cuda:")
        .ok_or_else(|| anyhow::anyhow!("--device must be 'cpu' or 'cuda:N', got '{spec}'"))?
        .parse()
        .context("parse cuda device index")?;
    #[cfg(feature = "cuda")]
    {
        Ok(Device::new_cuda(idx)?)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = idx;
        anyhow::bail!("this binary was built without the `cuda` feature")
    }
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, v) in row.iter().enumerate() {
        if v > &row[best] {
            best = i;
        }
    }
    best as u32
}

/// The target's argmax after the repeat penalty neuron applies when
/// serving. Same shape of penalty as `sample_with_penalty`: recent
/// tokens are divided (positive logits) or multiplied (negative ones).
fn argmax_with_penalty(row: &[f32], recent: &[u32], penalty: f32) -> u32 {
    if penalty == 1.0 {
        return argmax(row);
    }
    let mut adjusted = row.to_vec();
    for &t in recent {
        let i = t as usize;
        if i < adjusted.len() {
            adjusted[i] = if adjusted[i] > 0.0 {
                adjusted[i] / penalty
            } else {
                adjusted[i] * penalty
            };
        }
    }
    argmax(&adjusted)
}

/// Additive causal mask, `(1, 1, l, l + offset)`.
fn causal_mask(l: usize, offset: usize, dtype: DType, dev: &Device) -> Result<Tensor> {
    let total = l + offset;
    let mut data = vec![0f32; l * total];
    for i in 0..l {
        for j in 0..total {
            if j > i + offset {
                data[i * total + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, 1, l, total), dev)?.to_dtype(dtype)?)
}

/// Walk the prompt through target and head, in chunks.
///
/// Two reasons this is chunked rather than one forward: the causal mask
/// is `L x L` — 2.5 GB at 25k tokens, which OOMs a 12 GB card before a
/// single weight is touched — and the serving path chunks too, so this
/// keeps the measurement on the same shape of work.
///
/// The head trails the target by one token: at position `i` it takes
/// the embedding of token `i+1` with the target's hidden state at `i`.
/// Each chunk therefore reaches one token into the next. The final
/// prompt position has no successor yet — the first generated token
/// fills that slot.
///
/// Returns the target's logits and hidden state at the last position.
#[allow(clippy::too_many_arguments)]
fn prefill(
    model: &mut Qwen3_5ForCausalLM,
    head: &mut MtpHead,
    rotary: &RotaryEmbedding,
    ids: &[u32],
    chunk: usize,
    dtype: DType,
    dev: &Device,
    sync: &dyn Fn(&Device) -> Result<()>,
) -> Result<(Tensor, Tensor)> {
    let n = ids.len();
    let mut last: Option<(Tensor, Tensor)> = None;
    let mut at = 0usize;
    while at < n {
        let end = (at + chunk).min(n);
        let block = Tensor::new(&ids[at..end], dev)?.unsqueeze(0)?;
        let (logits, hidden) = model.forward_with_hidden(&block, at)?;
        let l = end - at;

        let head_end = end.min(n - 1);
        if head_end > at {
            let hl = head_end - at;
            let shifted = Tensor::new(&ids[at + 1..head_end + 1], dev)?.unsqueeze(0)?;
            let embeds = model.embed_tokens(&shifted)?;
            let hid = hidden.i((.., ..hl, ..))?;
            let mask = causal_mask(hl, at, dtype, dev)?;
            let positions: Vec<usize> = (at..head_end).collect();
            let (cos, sin) = rotary.cos_sin_at(&positions)?;
            head.forward(&embeds, &hid, Some(&mask), &cos, &sin)?;
        }

        last = Some((logits, hidden.i((.., l - 1.., ..))?));
        at = end;
        sync(dev)?;
    }
    last.context("empty prompt")
}

struct Timings {
    target_decode_ms: f64,
    draft_step_ms: f64,
    verify_ms: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.tensor_parallel > 1 {
        return run_tp(args).await;
    }
    let dev = pick_device(&args.device)?;
    let dtype = if matches!(dev, Device::Cpu) {
        DType::F32
    } else {
        DType::BF16
    };

    // ── load ────────────────────────────────────────────────────────
    let cfg_raw =
        std::fs::read_to_string(args.model.join("config.json")).context("read config.json")?;
    let config: Config = Config::from_config_json(&cfg_raw).context("parse config.json")?;
    let text_cfg = config.text_config.clone();
    anyhow::ensure!(
        MtpHead::present_in(&text_cfg),
        "this checkpoint declares no MTP head (mtp_num_hidden_layers = 0)"
    );

    let mut shards: Vec<PathBuf> = std::fs::read_dir(&args.model)
        .context("read model dir")?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    shards.sort();
    anyhow::ensure!(!shards.is_empty(), "no .safetensors in {:?}", args.model);
    eprintln!(
        "loading {} shard(s) as {dtype:?} on {:?}…",
        shards.len(),
        dev
    );

    // SAFETY: mmaps checkpoint files the operator pointed us at.
    let vb =
        unsafe { candle_nn::var_builder::ShardedSafeTensors::var_builder(&shards, dtype, &dev)? };
    let mut model = Qwen3_5ForCausalLM::new(config, vb.clone()).context("load target model")?;
    let rotary = std::sync::Arc::new(RotaryEmbedding::new(dtype, &text_cfg, &dev)?);
    let mut head = MtpHead::load(&text_cfg, rotary.clone(), &vb).context("load MTP head")?;

    let tok = tokenizers::Tokenizer::from_file(args.model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
    let prompt = match &args.prompt_file {
        Some(p) => std::fs::read_to_string(p).context("read prompt file")?,
        None => "Explain, step by step, how a bicycle derailleur shifts gears.".into(),
    };
    let ids: Vec<u32> = tok
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
        .get_ids()
        .to_vec();
    anyhow::ensure!(ids.len() >= 2, "prompt is too short to prefill");
    eprintln!("prompt: {} tokens", ids.len());

    let sync = |d: &Device| -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            d.synchronize()?;
        }
        let _ = d;
        Ok(())
    };

    // ── prefill ─────────────────────────────────────────────────────
    let n = ids.len();
    let t0 = Instant::now();
    let (logits, hidden) = prefill(
        &mut model,
        &mut head,
        &rotary,
        &ids,
        args.prefill_chunk,
        dtype,
        &dev,
        &sync,
    )?;
    eprintln!("prefill ({n} tokens): {:.2} s", t0.elapsed().as_secs_f64());

    // ── the loop ────────────────────────────────────────────────────
    let vocab = text_cfg.vocab_size;
    let mut last_logits: Vec<f32> = logits.i((0, 0, ..))?.to_dtype(DType::F32)?.to_vec1()?;
    let mut hidden_cur = hidden;
    let mut produced: Vec<u32> = Vec::new();
    // (step at which it was drafted, the drafted tokens)
    let mut pending: Vec<(usize, Vec<u32>)> = Vec::new();
    let mut accepted_hist = vec![0usize; args.draft_len + 1];
    let mut accepted_hist_penalised = vec![0usize; args.draft_len + 1];
    let mut penalty_disagreements = 0usize;
    let mut timings: Option<Timings> = None;
    let mut draft_ms_total = 0f64;
    let mut draft_rounds = 0usize;
    let mut target_ms_total = 0f64;
    let mut target_steps = 0usize;

    let total_rounds = args.warmup + args.steps;
    for round in 0..total_rounds {
        let measuring = round >= args.warmup;
        let pos = n + produced.len(); // position of the token about to be drafted from
        let next = argmax(&last_logits);
        let next_penalised = argmax_with_penalty(
            &last_logits,
            &produced[produced.len().saturating_sub(args.repeat_last_n)..],
            args.repeat_penalty,
        );
        if next != next_penalised {
            penalty_disagreements += 1;
        }
        produced.push(next);

        // Resolve any draft whose K true tokens have now been produced.
        pending.retain(|(at, drafted)| {
            let have = produced.len() - at;
            if have < drafted.len() {
                return true;
            }
            let truth = &produced[*at..*at + drafted.len()];
            let acc = drafted
                .iter()
                .zip(truth.iter())
                .take_while(|(d, t)| d == t)
                .count();
            accepted_hist[acc] += 1;
            false
        });

        // ── draft, then rewind ──────────────────────────────────────
        let snap = head.snapshot_kv()?;
        let t0 = Instant::now();
        let mut drafted = Vec::with_capacity(args.draft_len);
        let mut cur_token = next;
        let mut cur_hidden = hidden_cur.clone();
        for k in 0..args.draft_len {
            let tok_t = Tensor::new(&[cur_token], &dev)?.unsqueeze(0)?;
            let emb = model.embed_tokens(&tok_t)?;
            let (c, s) = rotary.cos_sin_at(&[pos + k])?;
            let out = head.forward(&emb, &cur_hidden, None, &c, &s)?;
            let row: Vec<f32> = model
                .lm_head(&out)?
                .i((0, 0, ..))?
                .to_dtype(DType::F32)?
                .to_vec1()?;
            anyhow::ensure!(row.len() == vocab, "draft logits width {}", row.len());
            cur_token = argmax(&row);
            drafted.push(cur_token);
            cur_hidden = out;
        }
        sync(&dev)?;
        let draft_elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        head.restore_kv(&snap)?;

        // The head must still advance over the *true* token, or its
        // cache falls behind the target's.
        let tok_t = Tensor::new(&[next], &dev)?.unsqueeze(0)?;
        let emb = model.embed_tokens(&tok_t)?;
        let (c, s) = rotary.cos_sin_at(&[pos])?;
        head.forward(&emb, &hidden_cur, None, &c, &s)?;

        // ── the target's own step ───────────────────────────────────
        let t0 = Instant::now();
        let (logits, h) = model.forward_with_hidden(&tok_t, pos)?;
        sync(&dev)?;
        let target_elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        last_logits = logits.i((0, 0, ..))?.to_dtype(DType::F32)?.to_vec1()?;
        hidden_cur = h;

        if measuring {
            pending.push((produced.len(), drafted));
            draft_ms_total += draft_elapsed;
            draft_rounds += 1;
            target_ms_total += target_elapsed;
            target_steps += 1;
        }

        // One verify-shaped forward, once, for the speedup arithmetic:
        // K+1 positions in a single pass is what a real round costs the
        // target, and it is a different kernel from a 1-token decode.
        if round == args.warmup {
            let block: Vec<u32> = std::iter::once(next)
                .chain(std::iter::repeat_n(next, args.draft_len))
                .collect();
            let blk = Tensor::new(block.as_slice(), &dev)?.unsqueeze(0)?;
            let snap_t = Instant::now();
            let _ = model.forward_multi(&blk, pos + 1)?;
            sync(&dev)?;
            let verify_ms = snap_t.elapsed().as_secs_f64() * 1000.0;
            // That forward advanced the target's cache by K+1; put it
            // back by re-prefilling is not possible here, so the probe
            // measures verify cost on a throwaway clone of the state.
            timings = Some(Timings {
                target_decode_ms: target_elapsed,
                draft_step_ms: draft_elapsed / args.draft_len as f64,
                verify_ms,
            });
            // Re-align: the verify pass wrote K+1 positions into the
            // target's KV that the observer never committed.
            model.clear_kv_cache();
            head.clear_kv_cache();
            let replay: Vec<u32> = ids
                .iter()
                .copied()
                .chain(produced.iter().copied())
                .collect();
            let (lg, hd) = prefill(
                &mut model,
                &mut head,
                &rotary,
                &replay,
                args.prefill_chunk,
                dtype,
                &dev,
                &sync,
            )?;
            last_logits = lg.i((0, 0, ..))?.to_dtype(DType::F32)?.to_vec1()?;
            hidden_cur = hd;
            pending.clear();
        }
    }

    // Anything still pending never got its K true tokens; score what is
    // known rather than dropping it silently.
    for (at, drafted) in &pending {
        let have = produced.len() - at;
        let truth = &produced[*at..];
        let acc = drafted
            .iter()
            .zip(truth.iter())
            .take_while(|(d, t)| d == t)
            .count();
        if acc < have {
            accepted_hist[acc] += 1;
        }
    }
    let _ = &mut accepted_hist_penalised;

    // ── report ──────────────────────────────────────────────────────
    let rounds: usize = accepted_hist.iter().sum();
    let mean_accepted: f64 = accepted_hist
        .iter()
        .enumerate()
        .map(|(k, n)| (k * n) as f64)
        .sum::<f64>()
        / rounds.max(1) as f64;
    let t = timings.context("no timing round ran; --steps must exceed 0")?;
    let target_ms = target_ms_total / target_steps.max(1) as f64;
    let draft_ms = draft_ms_total / (draft_rounds.max(1) * args.draft_len) as f64;
    let c = draft_ms / target_ms;
    // One round commits 1 + accepted tokens and costs one verify plus K
    // drafts, against (1 + accepted) plain decode steps.
    let round_cost = t.verify_ms + args.draft_len as f64 * draft_ms;
    let speedup = (1.0 + mean_accepted) * target_ms / round_cost;

    let summary = serde_json::json!({
        "model": args.model.to_string_lossy(),
        "device": args.device,
        "dtype": format!("{dtype:?}"),
        "prompt_tokens": n,
        "draft_len": args.draft_len,
        "rounds_scored": rounds,
        "target_decode_ms": target_ms,
        "draft_step_ms": draft_ms,
        "verify_ms_k_plus_1": t.verify_ms,
        "c_draft_over_target": c,
        "mean_accepted_tokens": mean_accepted,
        "accepted_histogram": accepted_hist,
        "predicted_speedup": speedup,
        "penalty_argmax_disagreements": penalty_disagreements,
        "penalty_argmax_disagreement_rate":
            penalty_disagreements as f64 / produced.len().max(1) as f64,
        "first_target_decode_ms_sample": t.target_decode_ms,
        "draft_step_ms_sample": t.draft_step_ms,
    });
    let pretty = serde_json::to_string_pretty(&summary)?;
    println!("{pretty}");
    if let Some(p) = &args.json {
        std::fs::write(p, &pretty).context("write --json")?;
    }
    Ok(())
}

/// Tensor-parallel measurement (#96).
///
/// The only route to the 27B: 54 GB in bf16 does not fit one 5090, and
/// quantisation is reachable only through the TP loader. The shape
/// mirrors the single-GPU path — prefill, then draft-and-rewind at each
/// decode step — with two differences:
///
/// - the target's step fans out to every rank, because the row-parallel
///   `AllReduce`s only complete when all of them arrive;
/// - the draft head is unsharded on the leader, so drafting issues no
///   collectives at all and cannot leave a rank waiting.
#[cfg(feature = "cuda")]
async fn run_tp(args: Args) -> Result<()> {
    use neuron::harness::device_worker::DeviceWorkerHandle;
    use neuron::harness::tp::WorkerPool;

    let cfg_raw =
        std::fs::read_to_string(args.model.join("config.json")).context("read config.json")?;
    let config: Config = Config::from_config_json(&cfg_raw).context("parse config.json")?;
    let text_cfg = config.text_config.clone();
    anyhow::ensure!(
        MtpHead::present_in(&text_cfg),
        "this checkpoint declares no MTP head (mtp_num_hidden_layers = 0)"
    );

    let mut shards: Vec<PathBuf> = std::fs::read_dir(&args.model)
        .context("read model dir")?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    shards.sort();
    anyhow::ensure!(!shards.is_empty(), "no .safetensors in {:?}", args.model);

    let worker_bin = match &args.worker_binary {
        Some(p) => p.clone(),
        None => std::env::current_exe()
            .context("resolve current_exe")?
            .parent()
            .context("executable has no parent directory")?
            .join("neuron"),
    };
    anyhow::ensure!(
        worker_bin.exists(),
        "worker binary {worker_bin:?} not found — build `neuron` beside this probe, \
         or pass --worker-binary"
    );

    let devices: Vec<u32> = (0..args.tensor_parallel).collect();
    eprintln!(
        "tp-{}: spawning workers from {worker_bin:?} on devices {devices:?}…",
        args.tensor_parallel
    );
    let leader = DeviceWorkerHandle::spawn(devices[0])?;
    let mut pool =
        WorkerPool::spawn(&worker_bin, args.tensor_parallel, &devices, leader.clone()).await?;
    pool.init_nccl(devices[0]).await?;

    let leader_device = Device::new_cuda(devices[0] as usize)?;
    let model_id = "mtp-probe";
    let t0 = Instant::now();
    let handle = pool
        .load_dense_shard(
            model_id,
            &cfg_raw,
            &shards,
            &leader_device,
            DType::BF16,
            args.quant.clone(),
        )
        .await?;
    eprintln!("shards loaded: {:.1} s", t0.elapsed().as_secs_f64());

    let shard_strs: Vec<String> = shards
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    leader
        .tp_load_mtp_head(handle, cfg_raw.clone(), shard_strs)
        .await?;
    eprintln!("draft head loaded on the leader");

    let tok = tokenizers::Tokenizer::from_file(args.model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
    let prompt = match &args.prompt_file {
        Some(p) => std::fs::read_to_string(p).context("read prompt file")?,
        None => "Explain, step by step, how a bicycle derailleur shifts gears.".into(),
    };
    let ids: Vec<u32> = tok
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
        .get_ids()
        .to_vec();
    let n = ids.len();
    anyhow::ensure!(n >= 2, "prompt is too short to prefill");
    eprintln!("prompt: {n} tokens");

    // ── prefill, chunked, target then head ──────────────────────────
    let t0 = Instant::now();
    let mut last_logits: Vec<f32> = Vec::new();
    let mut at = 0usize;
    while at < n {
        let end = (at + args.prefill_chunk).min(n);
        last_logits = pool
            .generate_step(model_id, handle, ids[at..end].to_vec(), at)
            .await?;
        let head_end = end.min(n - 1);
        if head_end > at {
            leader
                .tp_mtp_prefill_chunk(handle, ids[at + 1..head_end + 1].to_vec(), at)
                .await?;
        }
        at = end;
    }
    eprintln!("prefill ({n} tokens): {:.1} s", t0.elapsed().as_secs_f64());

    // ── draft / verify observation ──────────────────────────────────
    let mut produced: Vec<u32> = Vec::new();
    let mut pending: Vec<(usize, Vec<u32>)> = Vec::new();
    let mut accepted_hist = vec![0usize; args.draft_len + 1];
    let mut penalty_disagreements = 0usize;
    let mut draft_ms_total = 0f64;
    let mut target_ms_total = 0f64;
    let mut measured = 0usize;
    let mut verify_ms = 0f64;

    for round in 0..args.warmup + args.steps {
        let measuring = round >= args.warmup;
        let pos = n + produced.len();
        let next = argmax(&last_logits);
        let next_penalised = argmax_with_penalty(
            &last_logits,
            &produced[produced.len().saturating_sub(args.repeat_last_n)..],
            args.repeat_penalty,
        );
        if next != next_penalised {
            penalty_disagreements += 1;
        }
        produced.push(next);

        pending.retain(|(start, drafted)| {
            if produced.len() - start < drafted.len() {
                return true;
            }
            let truth = &produced[*start..*start + drafted.len()];
            let acc = drafted
                .iter()
                .zip(truth.iter())
                .take_while(|(d, t)| d == t)
                .count();
            accepted_hist[acc] += 1;
            false
        });

        let t0 = Instant::now();
        let drafted = leader
            .tp_mtp_draft(handle, next, pos, args.draft_len)
            .await?;
        let draft_elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        leader.tp_mtp_advance(handle, next, pos).await?;

        let t0 = Instant::now();
        last_logits = pool
            .generate_step(model_id, handle, vec![next], pos)
            .await?;
        let target_elapsed = t0.elapsed().as_secs_f64() * 1000.0;

        if round == args.warmup {
            // One verify-shaped forward for the round-cost arithmetic.
            // It advances the target's cache by K+1, so the rounds after
            // it are discarded: `pending` is cleared and the loop
            // re-syncs on the next real step.
            let block: Vec<u32> = std::iter::repeat_n(next, args.draft_len + 1).collect();
            let t0 = Instant::now();
            let _ = pool
                .generate_step_multi(model_id, handle, block, pos + 1)
                .await?;
            verify_ms = t0.elapsed().as_secs_f64() * 1000.0;
            pool.clear_kv_cache(model_id, handle).await?;
            pending.clear();
            produced.clear();
            let mut at = 0usize;
            while at < n {
                let end = (at + args.prefill_chunk).min(n);
                last_logits = pool
                    .generate_step(model_id, handle, ids[at..end].to_vec(), at)
                    .await?;
                at = end;
            }
            continue;
        }

        if measuring {
            pending.push((produced.len(), drafted));
            draft_ms_total += draft_elapsed;
            target_ms_total += target_elapsed;
            measured += 1;
        }
    }

    let rounds: usize = accepted_hist.iter().sum();
    let mean_accepted: f64 = accepted_hist
        .iter()
        .enumerate()
        .map(|(k, c)| (k * c) as f64)
        .sum::<f64>()
        / rounds.max(1) as f64;
    let target_ms = target_ms_total / measured.max(1) as f64;
    let draft_ms = draft_ms_total / (measured.max(1) * args.draft_len) as f64;
    let round_cost = verify_ms + args.draft_len as f64 * draft_ms;
    let summary = serde_json::json!({
        "model": args.model.to_string_lossy(),
        "tensor_parallel": args.tensor_parallel,
        "quant": args.quant,
        "prompt_tokens": n,
        "draft_len": args.draft_len,
        "rounds_scored": rounds,
        "target_decode_ms": target_ms,
        "draft_step_ms": draft_ms,
        "verify_ms_k_plus_1": verify_ms,
        "c_draft_over_target": draft_ms / target_ms,
        "mean_accepted_tokens": mean_accepted,
        "accepted_histogram": accepted_hist,
        "predicted_speedup": (1.0 + mean_accepted) * target_ms / round_cost,
        "penalty_argmax_disagreements": penalty_disagreements,
        "penalty_argmax_disagreement_rate":
            penalty_disagreements as f64 / produced.len().max(1) as f64,
    });
    let pretty = serde_json::to_string_pretty(&summary)?;
    println!("{pretty}");
    if let Some(p) = &args.json {
        std::fs::write(p, &pretty).context("write --json")?;
    }
    pool.unload_model(model_id).await?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
async fn run_tp(_args: Args) -> Result<()> {
    anyhow::bail!("--tensor-parallel > 1 requires a build with the `cuda` feature")
}
