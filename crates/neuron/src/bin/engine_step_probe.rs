//! Checks the native decode step (#427) end to end against candle's
//! model, on the real checkpoint and GGUF.
//!
//! ```sh
//! NEURON_EXPERT_DEVICE_ROUTING=1 neuron-engine-step-probe GGUF_FIRST_SHARD HF_SNAPSHOT_DIR [STEPS]
//! ```
//!
//! Decodes a fixed token sequence from position 0 through both, one token
//! at a time, and compares each step's next-token distribution: KL from
//! candle's to the engine's, and whether the top token agrees. Both share
//! one expert source (no tier, so both read every expert alike) and the
//! GGUF's Q8_0 dense projections; candle runs in BF16 as served. The
//! engine takes PLE's contribution from candle's own PLE block, run on the
//! engine's residual, and candle's state is restored around that so its
//! own forward sees each token once.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{CudaStorage, DType, Device, Storage, Tensor};
use candle_nn::var_builder::ShardedSafeTensors;

use neuron::harness::arch::qwen3_5::proj::LinearSource;
use neuron::harness::arch::qwen4_exp::config::Config;
use neuron::harness::arch::qwen4_exp::engine::decoder::NativeDecoder;
use neuron::harness::arch::qwen4_exp::engine::weights::Gguf;
use neuron::harness::arch::qwen4_exp::gguf_dense::{GgufDenseSource, Heads};
use neuron::harness::arch::qwen4_exp::model::Qwen4ExpForCausalLM;
use neuron::harness::arch::qwen4_exp::native_experts::{GgufExpertSource, NativeExpertFactory};

fn softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = x.iter().map(|&v| f64::from(v - m).exp()).collect();
    let z: f64 = e.iter().sum();
    e.into_iter().map(|v| v / z).collect()
}

fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!(
            "usage: {} GGUF_FIRST_SHARD HF_SNAPSHOT_DIR [STEPS]",
            args[0]
        );
    }
    if std::env::var("NEURON_EXPERT_DEVICE_ROUTING").as_deref() != Ok("1") {
        bail!("needs NEURON_EXPERT_DEVICE_ROUTING=1");
    }
    let (gguf_path, hf) = (Path::new(&args[1]), Path::new(&args[2]));
    let steps: usize = args.get(3).map_or(Ok(16), |v| v.parse())?;

    let device = Device::new_cuda(0)?;
    let cuda = device.as_cuda_device()?.clone();
    let cfg = Config::from_config_json(
        &std::fs::read_to_string(hf.join("config.json")).context("read config.json")?,
    )?;
    let text = &cfg.text_config;

    let experts = Arc::new(GgufExpertSource::open(
        gguf_path,
        &device,
        text.num_hidden_layers,
        text.num_experts,
        text.hidden_size,
        text.moe_intermediate_size,
        None,
    )?);
    let dense: Arc<dyn LinearSource> = Arc::new(GgufDenseSource::open(
        gguf_path,
        &device,
        Heads {
            key_heads: text.linear_num_key_heads,
            value_heads: text.linear_num_value_heads,
            head_dim: text.linear_value_head_dim,
        },
        text.linear_num_key_heads * text.linear_key_head_dim,
    )?);
    let mut paths: Vec<_> = std::fs::read_dir(hf)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    paths.sort();
    // SAFETY: the checkpoint is not modified while mapped.
    let vb = unsafe { ShardedSafeTensors::var_builder(&paths, DType::BF16, &device)? };
    let mut model = Qwen4ExpForCausalLM::load(
        &cfg,
        DType::BF16,
        &device,
        &vb,
        &neuron::harness::arch::qwen4_exp::LoadOptions {
            quant: None,
            safetensors_paths: &paths,
            isq_cache: None,
            experts_onto: &Device::Cpu,
            experts_vb: None,
            expert_tier: None,
            external_experts: Some(Arc::new(NativeExpertFactory(experts.clone()))),
            dense: Some(dense),
        },
    )?;

    let gguf = Gguf::open(gguf_path, &cuda)?;
    let mut engine = NativeDecoder::load(&gguf, text, experts.clone(), steps.max(16))?;
    let wide = text.hidden_size * text.hc_count;

    // PROBE_TOKENS: a comma-separated id file (a real prompt); else a
    // fixed synthetic sequence. PROBE_DUMP_DIR: each step's logits as
    // e<pos>.f32 (engine) and c<pos>.f32 (candle), for comparing both
    // against another engine on the same prefix.
    let tokens: Vec<u32> = match std::env::var_os("PROBE_TOKENS") {
        Some(path) => std::fs::read_to_string(path)?
            .trim()
            .split(',')
            .take(steps)
            .map(|t| t.trim().parse())
            .collect::<Result<_, _>>()?,
        None => (0..steps)
            .map(|i| (1000 + 7919 * i) as u32 % 150_000)
            .collect(),
    };
    let dump = std::env::var_os("PROBE_DUMP_DIR").map(std::path::PathBuf::from);
    let write = |name: String, values: &[f32]| -> Result<()> {
        if let Some(dir) = &dump {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(dir.join(name), bytes)?;
        }
        Ok(())
    };
    let (mut kl_sum, mut agree, mut worst, mut kl_head_sum) = (0f64, 0usize, 0f64, 0f64);
    let lm_head = vb
        .pp("lm_head")
        .get((text.vocab_size, text.hidden_size), "weight")?;
    for (pos, &token) in tokens.iter().enumerate() {
        let ids = Tensor::new(&[[token]], &device)?;
        let snapshot = model.snapshot_kv_cache()?;

        engine.set_step(token, pos)?;
        engine.segment_head()?;
        // SAFETY: written in full by the copy below.
        let mut copy = unsafe { cuda.alloc::<f32>(wide)? };
        cuda.memcpy_dtod(engine.residual(), &mut copy)?;
        let residual = Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(copy, cuda.clone())),
            (1, 1, wide),
        ));
        if let Some(ple) = model.ple_contribution(&ids, &residual)? {
            let ple = ple.to_dtype(DType::F32)?.contiguous()?;
            let (storage, layout) = ple.storage_and_layout();
            let Storage::Cuda(storage) = &*storage else {
                bail!("PLE ran off the GPU");
            };
            let slice = storage.as_cuda_slice::<f32>()?;
            engine.add_to_residual(
                slice.device_ptr(slice.stream()).0 + (layout.start_offset() * 4) as u64,
            )?;
        }
        engine.segment_tail()?;
        let got = engine.logits()?;
        // The engine's final hidden state through candle's own BF16 head:
        // separates the Q6_K head from everything before it.
        // SAFETY: written in full by the copy below.
        let mut mixed = unsafe { cuda.alloc::<f32>(text.hidden_size)? };
        cuda.memcpy_dtod(engine.mixed(), &mut mixed)?;
        let mixed = Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(mixed, cuda.clone())),
            (1, text.hidden_size),
        ));
        let via_head: Vec<f32> = mixed
            .to_dtype(DType::BF16)?
            .matmul(&lm_head.t()?)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;

        model.restore_kv_cache(&snapshot)?;
        let want: Vec<f32> = model
            .forward(&ids, pos)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;

        let kl_of = |p: &[f64], q: &[f64]| -> f64 {
            p.iter()
                .zip(q)
                .filter(|(a, _)| **a > 0.0)
                .map(|(a, b)| a * (a / b.max(1e-30)).ln())
                .sum()
        };
        write(format!("e{pos}.f32"), &got)?;
        write(format!("c{pos}.f32"), &want)?;
        let p = softmax(&want);
        let kl = kl_of(&p, &softmax(&got));
        let kl_head = kl_of(&p, &softmax(&via_head));
        kl_head_sum += kl_head;
        let same = argmax(&want) == argmax(&got);
        kl_sum += kl;
        agree += usize::from(same);
        worst = worst.max(kl);
        println!(
            "pos {pos} token {token}: KL {kl:.4e} (BF16 head {kl_head:.4e}) top1 candle {} engine {} {}",
            argmax(&want),
            argmax(&got),
            if same { "same" } else { "DIFFERENT" }
        );
    }
    println!(
        "{steps} steps: mean KL {:.4e} (through candle's BF16 head {:.4e}), worst {worst:.4e}, top-1 agree {agree}/{steps}",
        kl_sum / steps as f64,
        kl_head_sum / steps as f64
    );
    Ok(())
}
