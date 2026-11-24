//! Checks a native engine layer (#427) against candle's own, on the real
//! checkpoint and GGUF.
//!
//! ```sh
//! neuron-engine-layer-probe GGUF_FIRST_SHARD HF_SNAPSHOT_DIR [LAYER] [STEPS]
//! ```
//!
//! Runs the mixer half of linear-attention layer `LAYER` (default 0) for
//! `STEPS` decode tokens (default 8), carrying state, through both paths
//! from the same input: each step's residual is a real token embedding
//! repeated over the streams. Candle runs in BF16 (as served; its conv
//! kernel takes nothing wider) with the same GGUF Q8_0
//! projections the engine reads, so what differs is the engine's own
//! arithmetic and its BF16 hyper-connection and gate weights. Prints, per
//! step, the relative error of the layer's contribution (`out - in`).

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Device, IndexOp, Storage, Tensor};
use candle_nn::var_builder::ShardedSafeTensors;

use neuron::harness::arch::qwen3_5::proj::LinearSource;
use neuron::harness::arch::qwen4_exp::config::Config;
use neuron::harness::arch::qwen4_exp::engine::gdn;
use neuron::harness::arch::qwen4_exp::engine::weights::Gguf;
use neuron::harness::arch::qwen4_exp::gguf_dense::{GgufDenseSource, Heads};
use neuron::harness::arch::qwen4_exp::hyper::HyperConnection;
use neuron::harness::arch::qwen4_exp::linear_attn;

fn address(t: &Tensor) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        bail!("expected a CUDA tensor");
    };
    let slice = storage.as_cuda_slice::<f32>()?;
    Ok(slice.device_ptr(slice.stream()).0 + (layout.start_offset() * 4) as u64)
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        num += f64::from(x - y).powi(2);
        den += f64::from(*y).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!(
            "usage: {} GGUF_FIRST_SHARD HF_SNAPSHOT_DIR [LAYER] [STEPS]",
            args[0]
        );
    }
    let (gguf_path, hf) = (Path::new(&args[1]), Path::new(&args[2]));
    let layer: usize = args.get(3).map_or(Ok(0), |v| v.parse())?;
    let steps: usize = args.get(4).map_or(Ok(8), |v| v.parse())?;

    let device = Device::new_cuda(0)?;
    let cuda = device.as_cuda_device()?.clone();
    let cfg = Config::from_config_json(
        &std::fs::read_to_string(hf.join("config.json")).context("read config.json")?,
    )?;
    let text = &cfg.text_config;
    let shapes = gdn::shapes(text)?;

    // ---- candle, in f32, with the GGUF's Q8_0 projections.
    let mut paths: Vec<_> = std::fs::read_dir(hf)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    paths.sort();
    // SAFETY: the checkpoint is not modified while mapped.
    let vb = unsafe { ShardedSafeTensors::var_builder(&paths, DType::BF16, &device)? };
    let dense = GgufDenseSource::open(
        gguf_path,
        &device,
        Heads {
            key_heads: text.linear_num_key_heads,
            value_heads: text.linear_num_value_heads,
            head_dim: text.linear_value_head_dim,
        },
        text.linear_num_key_heads * text.linear_key_head_dim,
    )?;
    let source: Option<&dyn LinearSource> = Some(&dense);
    let lvb = vb.pp(format!("model.language_model.layers.{layer}"));
    let hc = HyperConnection::load(
        &lvb.pp("attn_hyper_connection"),
        text.hidden_size,
        text.hc_count,
        text.hc_lowrank,
        text.rms_norm_eps,
        true,
        source,
    )?;
    let mut net = linear_attn::load(text, &lvb.pp("linear_attn"), source)?;
    let embed = vb
        .pp("model.language_model.embed_tokens")
        .get((text.vocab_size, text.hidden_size), "weight")?;

    // ---- the engine.
    gdn::configure();
    let gguf = Gguf::open(gguf_path, &cuda)?;
    let weights = gdn::GdnWeights::load(&gguf, layer, &shapes)?;
    let mut state = gdn::GdnState::zeroed(&cuda, &shapes)?;
    let mut scratch = gdn::GdnScratch::new(&cuda, &shapes)?;
    let out = cuda.alloc_zeros::<f32>(shapes.hc * shapes.n_embd)?;
    let stream = cuda.cuda_stream().cu_stream() as *mut std::ffi::c_void;

    // A deliberate fault, to show the probe can see one: the engine's
    // state is reset every step while candle's carries on.
    let fault = std::env::var_os("PROBE_FAULT_RESET_STATE").is_some();
    let mut worst = 0f64;
    for step in 0..steps {
        if fault {
            state = gdn::GdnState::zeroed(&cuda, &shapes)?;
        }
        let token = (1000 + 7919 * step) % text.vocab_size;
        let h = embed
            .i(token)?
            .reshape((1, 1, text.hidden_size))?
            .repeat((1, 1, text.hc_count))?;
        let h = h.contiguous()?;

        let split = hc.split(&h)?;
        let inject = split
            .inject
            .as_ref()
            .context("a decoder hyper-connection injects")?;
        let mixed = net.forward(&split.input)?;
        let want = hc.combine(&split.residual, &mixed, inject)?;

        let h32 = h.to_dtype(DType::F32)?.contiguous()?;
        gdn::forward(
            &weights,
            &mut state,
            &mut scratch,
            &shapes,
            address(&h32)?,
            out.device_ptr(out.stream()).0,
            stream,
        )?;
        cuda.cuda_stream().synchronize()?;

        let input: Vec<f32> = h32.flatten_all()?.to_vec1()?;
        let want: Vec<f32> = want.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let got = cuda.clone_dtoh(&out)?;
        let delta = |r: &[f32]| -> Vec<f32> { r.iter().zip(&input).map(|(a, b)| a - b).collect() };
        let err = rel(&delta(&got), &delta(&want));
        worst = worst.max(err);
        println!(
            "layer {layer} step {step} token {token}: contribution rel err {err:.3e}, |out-in| {:.3e}",
            delta(&want)
                .iter()
                .map(|v| f64::from(*v).powi(2))
                .sum::<f64>()
                .sqrt()
        );
    }
    println!("layer {layer}: worst {worst:.3e} over {steps} steps");
    Ok(())
}
