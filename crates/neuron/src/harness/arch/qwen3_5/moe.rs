//! High-sparsity MoE FFN block for the qwen3_next family (#92).
//!
//! Qwen3-Next-80B-A3B replaces the dense SwiGLU in (almost) every
//! decoder layer with `Qwen3NextSparseMoeBlock`: a top-k router over
//! `num_experts` small SwiGLU experts, plus an always-on **shared
//! expert** mixed in through a per-token sigmoid gate:
//!
//! ```text
//! probs   = softmax(gate(x))                 # over ALL experts, f32
//! w, idx  = topk(probs, num_experts_per_tok)
//! w       = w / sum(w)                       # iff norm_topk_prob
//! routed  = Σ_j w_j · expert_{idx_j}(x)
//! shared  = sigmoid(shared_expert_gate(x)) · shared_expert(x)
//! y       = routed + shared
//! ```
//!
//! Routing follows the upstream softmax-then-topk order (NOT
//! topk-then-softmax — the renormalisation only equals softmax over
//! the selected logits when `norm_topk_prob` is on, and the reference
//! renormalises the *global* softmax values).
//!
//! ## Dispatch strategy
//!
//! This is the correctness-first implementation: a host-side scatter
//! loop over the experts that actually received tokens (the pattern
//! candle-transformers' `Qwen3SparseMoeBlock` uses). Batch-1 decode
//! touches `num_experts_per_tok` experts per layer; prefill batches
//! per-expert token groups. The fused grouped-GEMM path (slice 4)
//! replaces the loop behind the same `forward` signature.

use anyhow::{Context, Result};
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::Linear;
use candle_nn::var_builder::ShardedVarBuilder;

use super::TextConfig;
use super::mlp::Qwen3_5MLP;

/// How a layer's routed experts are stored.
///
/// Two shapes, because the right one depends on whether the model fits.
///
/// `PerExpert` is the Qwen3-Next layout: one module per expert, sliced
/// out at load. Fine when the weights are device-resident anyway.
///
/// `Banked` keeps the checkpoint's own fused tensors whole and takes a
/// view per routed expert. `qwen4_exp` needs it: slicing its 512
/// experts into modules copies **241.6 GB** at BF16 — 5.03 GB per
/// layer — which fits in neither 64 GB of VRAM nor 123 GB of host RAM,
/// at any quantisation. It is also the layout a host-resident bank with
/// a device-side gather wants (#318), so the fused form is where that
/// work starts rather than something it would have to undo.
///
/// Banking costs nothing per call: a slice along dim 0 of a contiguous
/// tensor is contiguous, and `x.matmul(w.t())` is what `Linear` does
/// anyway, so the views feed the same GEMMs without a copy.
pub(crate) enum Experts {
    PerExpert(Vec<Qwen3_5MLP>),
    /// Quantised in situ at load (#315). One `(gate, up, down)` triple
    /// per expert.
    ///
    /// This is what makes `qwen4_exp` loadable at all: 120.8 B routed
    /// parameters are 241.6 GB at BF16 and 67.9 GB at q4k, against
    /// beast's 64 GB of VRAM and ~110 GB of usable host RAM. Nothing
    /// about the arithmetic works until they shrink.
    ///
    /// Quantising *from* the banked tensor is what keeps the peak
    /// bounded: one layer's fused BF16 tensor is 5.03 GB, and it is
    /// dropped once its 512 experts are quantised, so the full 241.6 GB
    /// never exists at once.
    Quantized(Vec<QuantizedExpert>),
    Banked {
        /// `(num_experts, 2 * intermediate, hidden)` — gate rows first,
        /// then up. Reversing them computes `silu(up) * gate`, which is
        /// a different function of the same weights (#312).
        gate_up: Tensor,
        /// `(num_experts, hidden, intermediate)`
        down: Tensor,
        intermediate: usize,
    },
}

/// One expert's three projections, quantised.
pub(crate) struct QuantizedExpert {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

impl QuantizedExpert {
    /// Where this expert's weights are, when that is knowable.
    ///
    /// `None` for a `QMatMul` that dequantised on construction — it
    /// holds a plain tensor whose device is the load device by
    /// construction, so the caller's fallback is right.
    fn device(&self) -> Option<Device> {
        match &self.gate {
            QMatMul::QTensor(t) => Some(t.device()),
            _ => None,
        }
    }
}

/// The quantisation type actually usable for rows of `row` elements.
///
/// ggml quantises along the last dimension, and a row must be a whole
/// number of blocks. The k-quants (`Q2K`..`Q8K`) block on 256; the
/// older types block on 32. `qwen4_exp`'s `down_proj` reduces over
/// `moe_intermediate_size = 640`, which is 2.5 blocks of 256 — so a
/// `quant = "q4k"` load fails on the very first layer with
/// `quantized tensor must have their last dim divisible by block size
/// [2560, 640] 256`, having read 131 shards to get there.
///
/// **This table is llama.cpp's**, from `tensor_type_fallback` in
/// `src/llama-quant.cpp`, and the principle behind it is the part
/// worth stating: **every fallback spends bits rather than saving
/// them.** A k-quant carries a two-level scale hierarchy that the
/// legacy 32-block types do not, so it is better than a legacy type at
/// equal width — dropping `Q4K` (4.5 bpw) to `Q4_0` (also 4.5) would
/// hold the memory constant and take the quality hit silently.
/// Upstream declines that trade and pays a bit instead:
///
/// | requested | fallback | bpw |
/// |---|---|---|
/// | `Q2K`, `Q3K` | `Q4_0` | 2.63 / 3.44 -> 4.5 |
/// | `Q4K` | `Q5_0` | 4.5 -> 5.5 |
/// | `Q5K` | `Q5_1` | 5.5 -> 6.0 |
/// | `Q6K`, `Q8K` | `Q8_0` | 6.56 -> 8.5 |
///
/// For `qwen4_exp` that is not free: `down_proj` is exactly a third of
/// the 120.8 B routed parameters, so `Q5_0` rather than `Q4_0` costs
/// **+5.0 GB** across the expert bank (67.9 -> 73.0 GB). We take it
/// anyway. The cheaper table was an unmeasured divergence in the one
/// direction that flatters the constraint we are straining against,
/// and this matrix is the one a mixed k-quant scheme normally treats
/// as quality-critical. If the 5 GB turns out to buy nothing, #315 is
/// where that gets measured and reversed on evidence.
///
/// A row that divides 32 either — vanishingly rare, and impossible for
/// this checkpoint since 640 and 2560 are both multiples of 32 — falls
/// back to F16 as upstream does, loudly. It is a 3.5x blowup on that
/// tensor, so it must never be quiet.
pub(crate) fn quant_for_row(requested: GgmlDType, row: usize) -> candle_core::Result<GgmlDType> {
    if row.is_multiple_of(requested.block_size()) {
        return Ok(requested);
    }
    let fallback = match requested {
        GgmlDType::Q2K | GgmlDType::Q3K => GgmlDType::Q4_0,
        GgmlDType::Q4K => GgmlDType::Q5_0,
        GgmlDType::Q5K => GgmlDType::Q5_1,
        GgmlDType::Q6K | GgmlDType::Q8K => GgmlDType::Q8_0,
        // Already a 32-block type (or not a quant at all): there is no
        // smaller block to demote to, so the F16 check below decides.
        other => other,
    };
    if row.is_multiple_of(fallback.block_size()) {
        return Ok(fallback);
    }
    tracing::warn!(
        ?requested,
        ?fallback,
        row,
        "row divides neither the requested block size nor its 32-wide \
         fallback; storing this tensor as F16 — 3.5x the bytes"
    );
    Ok(GgmlDType::F16)
}

impl Experts {
    pub(crate) fn len(&self) -> usize {
        match self {
            Experts::PerExpert(v) => v.len(),
            Experts::Banked { gate_up, .. } => gate_up.dims()[0],
            Experts::Quantized(v) => v.len(),
        }
    }

    /// Quantise a banked pair into per-expert triples.
    ///
    /// Takes the fused tensors by reference and drops nothing itself —
    /// the caller owns the peak, and should hold one layer at a time.
    ///
    /// The two halves may not end up the same dtype: see
    /// [`quant_for_row`]. `qwen4_exp`'s `down_proj` reduces over 640,
    /// which no k-quant can block.
    /// `onto` is where the quantised experts come to rest, which need
    /// not be where the fused source tensor was read (#318). Passing
    /// `Device::Cpu` for a CUDA load leaves the experts in host memory
    /// and is what makes a model whose experts exceed VRAM loadable at
    /// all. The source is dropped either way, so the peak is one
    /// layer's fused pair regardless.
    /// Quantise where the source already is. Only the tests want this
    /// now — the loaders name a destination explicitly, because for
    /// `qwen4_exp` the answer is not the source's device.
    #[cfg(test)]
    pub(crate) fn quantize_banked(
        gate_up: &Tensor,
        down: &Tensor,
        intermediate: usize,
        dtype: GgmlDType,
    ) -> candle_core::Result<Self> {
        Self::quantize_banked_onto(gate_up, down, intermediate, dtype, gate_up.device())
    }

    pub(crate) fn quantize_banked_onto(
        gate_up: &Tensor,
        down: &Tensor,
        intermediate: usize,
        dtype: GgmlDType,
        onto: &Device,
    ) -> candle_core::Result<Self> {
        let n = gate_up.dims()[0];
        // Decided once, from the shapes, rather than per expert —
        // every expert in a layer has the same two row widths, and
        // 512 identical log lines per layer would bury the fact.
        let gate_up_dtype = quant_for_row(dtype, gate_up.dims()[2])?;
        let down_dtype = quant_for_row(dtype, down.dims()[2])?;
        if gate_up_dtype != dtype || down_dtype != dtype {
            tracing::info!(
                requested = ?dtype,
                gate_up_row = gate_up.dims()[2],
                gate_up_dtype = ?gate_up_dtype,
                down_row = down.dims()[2],
                down_dtype = ?down_dtype,
                "expert quantisation: a row width does not divide the requested \
                 block size; substituting a 32-block type for that half"
            );
        }
        let mut out = Vec::with_capacity(n);
        for e in 0..n {
            let gu = gate_up.i(e)?;
            // The source must own its storage outright.
            // `QTensor::quantize` reads `src.storage()` — the whole
            // backing buffer, ignoring shape and offset — so a view into
            // the fused tensor hands it every expert's bytes while the
            // shape claims one, and candle panics on a block-count
            // mismatch inside `from_float`. Neither `contiguous()` nor
            // `copy()` saves you here: a slice along dim 0 is already
            // contiguous, so both are free to hand back something still
            // sharing the parent's buffer. Round-tripping through a Vec
            // is the part that actually allocates, and it is also where
            // the f32 the quantiser wants comes from.
            let q = |t: Tensor, dtype: GgmlDType| -> candle_core::Result<QMatMul> {
                let dims = t.dims().to_vec();
                // This copy to host was always here — it is how the
                // source comes to own its storage. Building the owned
                // tensor on `onto` rather than on `t.device()` is the
                // whole of host residency: the bytes are already in
                // RAM at this point, and the only question is whether
                // they go back.
                let vals = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let owned = Tensor::from_vec(vals, dims, onto)?;
                QMatMul::from_qtensor(QTensor::quantize(&owned, dtype)?)
            };
            out.push(QuantizedExpert {
                gate: q(gu.narrow(0, 0, intermediate)?, gate_up_dtype)?,
                up: q(gu.narrow(0, intermediate, intermediate)?, gate_up_dtype)?,
                down: q(down.i(e)?, down_dtype)?,
            });
        }
        Ok(Experts::Quantized(out))
    }

    /// Where the routed experts live, when that is knowable (#318).
    pub(crate) fn device(&self) -> Option<Device> {
        match self {
            Experts::Quantized(v) => v.first().and_then(QuantizedExpert::device),
            _ => None,
        }
    }

    /// The quantised tensors, named for the on-disk cache (#322).
    ///
    /// `None` for any layout that is not quantised — there is nothing
    /// to cache about weights that were never transformed.
    pub(crate) fn cache_entries(&self) -> Option<Vec<(String, std::sync::Arc<QTensor>)>> {
        let Experts::Quantized(v) = self else {
            return None;
        };
        let of = |m: &QMatMul| match m {
            QMatMul::QTensor(t) => Some(t.clone()),
            // A dequantised-on-load QMatMul has no quantised tensor to
            // write. `CANDLE_DEQUANTIZE_ALL` produces these; caching
            // half a layer would be worse than caching none of it.
            _ => None,
        };
        let mut out = Vec::with_capacity(v.len() * 3);
        for (i, q) in v.iter().enumerate() {
            out.push((format!("e{i}.gate"), of(&q.gate)?));
            out.push((format!("e{i}.up"), of(&q.up)?));
            out.push((format!("e{i}.down"), of(&q.down)?));
        }
        Some(out)
    }

    /// Rebuild from what [`Self::cache_entries`] wrote.
    ///
    /// Errors rather than filling gaps: a layer missing an expert is a
    /// truncated artifact, and serving it would mean one expert of 512
    /// quietly doing nothing.
    pub(crate) fn from_cache_entries(
        mut entries: std::collections::HashMap<String, QTensor>,
        num_experts: usize,
    ) -> candle_core::Result<Self> {
        let mut v = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            let mut take = |part: &str| -> candle_core::Result<QMatMul> {
                let name = format!("e{i}.{part}");
                match entries.remove(&name) {
                    Some(t) => QMatMul::from_qtensor(t),
                    None => candle_core::bail!("cached expert layer is missing '{name}'"),
                }
            };
            v.push(QuantizedExpert {
                gate: take("gate")?,
                up: take("up")?,
                down: take("down")?,
            });
        }
        Ok(Experts::Quantized(v))
    }

    /// Expert `e` applied to the rows routed to it.
    pub(crate) fn forward_one(&self, e: usize, xs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Experts::PerExpert(v) => v[e].forward(xs),
            Experts::Banked {
                gate_up,
                down,
                intermediate,
            } => {
                let gu = gate_up.i(e)?;
                let gate_w = gu.narrow(0, 0, *intermediate)?;
                let up_w = gu.narrow(0, *intermediate, *intermediate)?;
                let lhs = candle_nn::ops::silu(&xs.matmul(&gate_w.t()?)?)?;
                let rhs = xs.matmul(&up_w.t()?)?;
                (lhs * rhs)?.matmul(&down.i(e)?.t()?)
            }
            Experts::Quantized(v) => {
                let q = &v[e];
                // The quantised kernels accumulate in f32 whatever the
                // activation dtype, so the cast is explicit here rather
                // than implied — and the result comes back in the
                // caller's dtype, and on the caller's device, so the
                // routing arithmetic upstream does not have to know
                // which layout or residency it got.
                let dtype = xs.dtype();
                let home = q.device().unwrap_or_else(|| xs.device().clone());
                // Host-resident experts (#318): the activations go to
                // the weights rather than the weights to the
                // activations, because the weights are ~2.4 GB a layer
                // and the activations are the rows routed to one
                // expert. `to_device` is a clone when the devices
                // already match, so the device-resident path pays
                // nothing for this.
                let xs32 = xs.to_dtype(DType::F32)?.to_device(&home)?;
                let lhs = candle_nn::ops::silu(&q.gate.forward(&xs32)?)?;
                let rhs = q.up.forward(&xs32)?;
                q.down
                    .forward(&(lhs * rhs)?)?
                    .to_device(xs.device())?
                    .to_dtype(dtype)
            }
        }
    }
}

pub struct Qwen3_5MoeBlock {
    /// Router: `(num_experts, hidden)`, checkpoint name `mlp.gate`.
    gate: Linear,
    /// Routed experts. Per-expert modules for `qwen3_5`; the
    /// checkpoint's fused tensors for `qwen4_exp` — see [`Experts`].
    experts: Experts,
    /// Always-on expert, `mlp.shared_expert.*`. `None` when the config
    /// declares no shared expert (Qwen3-30B-A3B style).
    shared_expert: Option<Qwen3_5MLP>,
    /// Per-token sigmoid mix for the shared expert: `(1, hidden)`,
    /// checkpoint name `mlp.shared_expert_gate`.
    shared_expert_gate: Option<Linear>,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl Qwen3_5MoeBlock {
    /// Assemble from parts.
    ///
    /// The routing arithmetic is identical across the checkpoints that
    /// use this block; only the storage of the experts differs.
    /// `qwen4_exp` ships them as fused 3D tensors and slices them
    /// itself, so it builds the block this way rather than through
    /// [`Self::load`], which expects per-expert modules and a
    /// `qwen3_5` config.
    pub(crate) fn from_parts(
        gate: Linear,
        experts: Experts,
        shared_expert: Option<Qwen3_5MLP>,
        shared_expert_gate: Option<Linear>,
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
    ) -> Self {
        Self {
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
            num_experts_per_tok,
            norm_topk_prob,
        }
    }

    pub fn load(cfg: &TextConfig, vb: &ShardedVarBuilder) -> Result<Self> {
        anyhow::ensure!(
            cfg.num_experts > 0 && cfg.num_experts_per_tok > 0 && cfg.moe_intermediate_size > 0,
            "MoE block needs num_experts ({}), num_experts_per_tok ({}) and \
             moe_intermediate_size ({}) all > 0",
            cfg.num_experts,
            cfg.num_experts_per_tok,
            cfg.moe_intermediate_size,
        );
        anyhow::ensure!(
            cfg.num_experts_per_tok <= cfg.num_experts,
            "num_experts_per_tok ({}) exceeds num_experts ({})",
            cfg.num_experts_per_tok,
            cfg.num_experts,
        );

        let h = cfg.hidden_size;

        let gate_weight = vb
            .pp("gate")
            .get((cfg.num_experts, h), "weight")
            .with_context(|| format!("load '{}/gate/weight'", vb.prefix()))?;
        let gate = Linear::new(gate_weight, None);

        let experts_vb = vb.pp("experts");
        let mut experts = Vec::with_capacity(cfg.num_experts);
        for i in 0..cfg.num_experts {
            experts.push(
                Qwen3_5MLP::load_with_dims(&experts_vb.pp(i), h, cfg.moe_intermediate_size)
                    .with_context(|| format!("load expert {i}"))?,
            );
        }

        let (shared_expert, shared_expert_gate) = if cfg.shared_expert_intermediate_size > 0 {
            let shared = Qwen3_5MLP::load_with_dims(
                &vb.pp("shared_expert"),
                h,
                cfg.shared_expert_intermediate_size,
            )
            .context("load shared_expert")?;
            let gate_w = vb
                .pp("shared_expert_gate")
                .get((1, h), "weight")
                .with_context(|| format!("load '{}/shared_expert_gate/weight'", vb.prefix()))?;
            (Some(shared), Some(Linear::new(gate_w, None)))
        } else {
            (None, None)
        };

        Ok(Self {
            gate,
            experts: Experts::PerExpert(experts),
            shared_expert,
            shared_expert_gate,
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }
}

/// Per-expert routing assignment: `(token_rows, weights)` per expert,
/// produced by [`route_scatter`].
pub(crate) type ExpertAssignments = (Vec<Vec<u32>>, Vec<Vec<f32>>);

/// Router + host-side scatter shared by the single-GPU and TP MoE
/// blocks (#92): softmax over ALL experts in f32 → descending-argsort
/// top-k → renormalise iff `norm_topk_prob` → per-expert token-row and
/// weight lists. Under TP the router weight is replicated, so every
/// rank computes identical assignments with zero communication.
pub(crate) fn route_scatter(
    gate: &Linear,
    xs_flat: &Tensor,
    num_experts: usize,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
) -> candle_core::Result<ExpertAssignments> {
    let n_tokens = xs_flat.dim(0)?;
    // Router probabilities in f32 (reference uses float softmax
    // regardless of activations dtype).
    let router_logits = gate.forward(xs_flat)?;
    let probs = candle_nn::ops::softmax_last_dim(&router_logits.to_dtype(DType::F32)?)?;

    // Top-k selection: descending argsort, take the first k. The
    // renormalisation (iff norm_topk_prob) divides by the sum of
    // the selected global-softmax values.
    let sorted = probs.arg_sort_last_dim(false)?;
    let topk_idx = sorted.narrow(1, 0, num_experts_per_tok)?.contiguous()?;
    let mut topk_w = probs.gather(&topk_idx, 1)?;
    if norm_topk_prob {
        let denom = topk_w.sum_keepdim(1)?;
        topk_w = topk_w.broadcast_div(&denom)?;
    }

    // Host-side scatter: token row lists per expert. Cheap relative
    // to the expert GEMMs; replaced by grouped-GEMM in slice 4.
    let idx_host: Vec<Vec<u32>> = topk_idx.to_vec2()?;
    let w_host: Vec<Vec<f32>> = topk_w.to_vec2()?;
    let mut tokens_for: Vec<Vec<u32>> = vec![Vec::new(); num_experts];
    let mut weights_for: Vec<Vec<f32>> = vec![Vec::new(); num_experts];
    for t in 0..n_tokens {
        for j in 0..num_experts_per_tok {
            let e = idx_host[t][j] as usize;
            tokens_for[e].push(t as u32);
            weights_for[e].push(w_host[t][j]);
        }
    }
    Ok((tokens_for, weights_for))
}

impl Module for Qwen3_5MoeBlock {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (b, l, hidden) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden))?;

        // Routing stays where the activations are: it is one small
        // matmul against `gate` and a host-side top-k, and the router
        // is device-resident even when the experts are not.
        let (tokens_for, weights_for) = route_scatter(
            &self.gate,
            &xs_flat,
            self.experts.len(),
            self.num_experts_per_tok,
            self.norm_topk_prob,
        )?;

        // Cross to the experts once for the whole layer, not once per
        // expert (#318). With host-resident experts the old shape was
        // 10 round trips per layer and 480 per token; this is one in
        // and one out. `to_device` is a clone when they already match,
        // so a device-resident model pays nothing.
        //
        // The measurement this was written against says the transfers
        // were never the dominant cost — see `expert_ms` below — but
        // one crossing per layer is the right shape regardless of what
        // dominates.
        let home = self.experts.device();
        let off_device = home.is_some();
        let routed_dev = home.unwrap_or_else(|| xs.device().clone());
        let xs_routed = xs_flat.to_device(&routed_dev)?;

        let started = std::time::Instant::now();
        let mut ys = xs_routed.zeros_like()?;
        for e in 0..self.experts.len() {
            if tokens_for[e].is_empty() {
                continue;
            }
            let rows = Tensor::new(tokens_for[e].as_slice(), &routed_dev)?;
            let picked = xs_routed.index_select(&rows, 0)?;
            let out = self.experts.forward_one(e, &picked)?;
            let w = Tensor::new(weights_for[e].as_slice(), &routed_dev)?
                .to_dtype(out.dtype())?
                .reshape(((), 1))?;
            ys = ys.index_add(&rows, &out.broadcast_mul(&w)?, 0)?;
        }
        // Where the time actually goes, per layer, when the experts are
        // not on the device. Logged at trace so a normal load is
        // unaffected; `NEURON_EXPERT_TIMING=1` is what a measurement
        // run sets. Without this the only figure available is
        // end-to-end tok/s, which cannot distinguish "the transfers are
        // slow" from "the CPU matmul is slow" — and those have opposite
        // fixes.
        if off_device && std::env::var("NEURON_EXPERT_TIMING").is_ok() {
            tracing::info!(
                expert_ms = started.elapsed().as_secs_f64() * 1e3,
                rows = xs_routed.dims()[0],
                active = tokens_for.iter().filter(|t| !t.is_empty()).count(),
                "moe: routed experts (#318)"
            );
        }
        let mut ys = ys.to_device(xs.device())?;

        if let (Some(shared), Some(gate)) = (&self.shared_expert, &self.shared_expert_gate) {
            let mix = candle_nn::ops::sigmoid(&gate.forward(&xs_flat)?)?;
            let shared_out = shared.forward(&xs_flat)?.broadcast_mul(&mix)?;
            ys = (ys + shared_out)?;
        }

        ys.reshape((b, l, hidden))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn randn(shape: &[usize]) -> Tensor {
        Tensor::randn(0f32, 0.5f32, shape, &Device::Cpu).unwrap()
    }

    fn rand_mlp(hidden: usize, inter: usize) -> Qwen3_5MLP {
        Qwen3_5MLP::from_weights(
            Linear::new(randn(&[inter, hidden]), None),
            Linear::new(randn(&[inter, hidden]), None),
            Linear::new(randn(&[hidden, inter]), None),
        )
    }

    /// The batched scatter forward must equal a per-token dense
    /// reference: route each token independently (host softmax → top-k
    /// → renorm), run its selected experts one by one, and mix in the
    /// shared expert through the sigmoid gate. Catches indexing,
    /// weighting, and renormalisation bugs in the scatter path.
    #[test]
    fn scatter_forward_matches_per_token_reference() {
        let (hidden, inter, n_exp, top_k) = (8, 4, 6, 2);

        let block = Qwen3_5MoeBlock {
            gate: Linear::new(randn(&[n_exp, hidden]), None),
            experts: Experts::PerExpert((0..n_exp).map(|_| rand_mlp(hidden, inter)).collect()),
            shared_expert: Some(rand_mlp(hidden, inter)),
            shared_expert_gate: Some(Linear::new(randn(&[1, hidden]), None)),
            num_experts_per_tok: top_k,
            norm_topk_prob: true,
        };

        let (b, l) = (2, 3);
        let xs = randn(&[b, l, hidden]);
        let got = block.forward(&xs).unwrap();
        assert_eq!(got.dims(), &[b, l, hidden]);

        let xs_flat = xs.reshape(((), hidden)).unwrap();
        let logits: Vec<Vec<f32>> = block.gate.forward(&xs_flat).unwrap().to_vec2().unwrap();
        let got_flat: Vec<Vec<f32>> = got.reshape(((), hidden)).unwrap().to_vec2().unwrap();

        for t in 0..b * l {
            // Host-side softmax over all experts, then top-k + renorm.
            let max = logits[t].iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = logits[t].iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();
            let mut order: Vec<usize> = (0..n_exp).collect();
            order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            let selected = &order[..top_k];
            let denom: f32 = selected.iter().map(|&e| probs[e]).sum();

            let row = xs_flat.narrow(0, t, 1).unwrap();
            let mut expect = vec![0f32; hidden];
            for &e in selected {
                let w = probs[e] / denom;
                let out: Vec<f32> = block
                    .experts
                    .forward_one(e, &row)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                for (acc, o) in expect.iter_mut().zip(out) {
                    *acc += w * o;
                }
            }
            let gate_v: f32 = block
                .shared_expert_gate
                .as_ref()
                .unwrap()
                .forward(&row)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0];
            let mix = 1.0 / (1.0 + (-gate_v).exp());
            let shared: Vec<f32> = block
                .shared_expert
                .as_ref()
                .unwrap()
                .forward(&row)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            for (acc, s) in expect.iter_mut().zip(shared) {
                *acc += mix * s;
            }

            for (i, (&g, &e)) in got_flat[t].iter().zip(expect.iter()).enumerate() {
                assert!(
                    (g - e).abs() < 1e-4,
                    "token {t} dim {i}: got {g}, expected {e}"
                );
            }
        }
    }

    /// Without a shared expert (Qwen3-30B-A3B shape) the block is pure
    /// routed output; without norm_topk_prob the raw global-softmax
    /// weights apply (they do NOT sum to 1 across the selected k).
    #[test]
    fn no_shared_expert_and_no_renorm() {
        let (hidden, inter, n_exp) = (4, 2, 3);
        let block = Qwen3_5MoeBlock {
            gate: Linear::new(randn(&[n_exp, hidden]), None),
            experts: Experts::PerExpert((0..n_exp).map(|_| rand_mlp(hidden, inter)).collect()),
            shared_expert: None,
            shared_expert_gate: None,
            num_experts_per_tok: 1,
            norm_topk_prob: false,
        };
        let xs = randn(&[1, 1, hidden]);
        let got: Vec<f32> = block
            .forward(&xs)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Reference: the argmax expert's output scaled by its raw
        // softmax probability.
        let logits: Vec<f32> = block
            .gate
            .forward(&xs.reshape(((), hidden)).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let max = logits.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let best = (0..n_exp)
            .max_by(|&a, &b| exps[a].partial_cmp(&exps[b]).unwrap())
            .unwrap();
        let w = exps[best] / sum;
        let out: Vec<f32> = block
            .experts
            .forward_one(best, &xs.reshape(((), hidden)).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, (&g, &o)) in got.iter().zip(out.iter()).enumerate() {
            assert!(
                (g - w * o).abs() < 1e-5,
                "dim {i}: got {g}, expected {}",
                w * o
            );
        }
    }

    /// Banking is a storage change, so the two layouts must compute the
    /// same function. Same weights, expressed both ways, asserted
    /// identical — otherwise "we only changed where the bytes live" is
    /// a claim rather than a fact.
    #[test]
    fn banked_experts_equal_per_expert_modules() {
        let (n_experts, hidden, inter) = (3usize, 4usize, 2usize);

        // One source of truth for the weights, laid out both ways.
        let gate_up = randn(&[n_experts, inter * 2, hidden]);
        let down = randn(&[n_experts, hidden, inter]);

        let per_expert: Vec<Qwen3_5MLP> = (0..n_experts)
            .map(|e| {
                let gu = gate_up.i(e).unwrap();
                Qwen3_5MLP::from_weights(
                    Linear::new(gu.narrow(0, 0, inter).unwrap().contiguous().unwrap(), None),
                    Linear::new(
                        gu.narrow(0, inter, inter).unwrap().contiguous().unwrap(),
                        None,
                    ),
                    Linear::new(down.i(e).unwrap().contiguous().unwrap(), None),
                )
            })
            .collect();

        let sliced = Experts::PerExpert(per_expert);
        let banked = Experts::Banked {
            gate_up,
            down,
            intermediate: inter,
        };
        assert_eq!(sliced.len(), banked.len());

        let xs = randn(&[5, hidden]);
        for e in 0..n_experts {
            let a: Vec<f32> = sliced
                .forward_one(e, &xs)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let b: Vec<f32> = banked
                .forward_one(e, &xs)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b.iter()) {
                assert!(
                    (x - y).abs() < 1e-5,
                    "expert {e} differs between layouts: {a:?} vs {b:?}"
                );
            }
        }
    }

    /// And the halves are not interchangeable: reading `gate_up` the
    /// other way round computes silu(up) * gate, which the banked path
    /// would do just as silently as the sliced one.
    #[test]
    fn banked_reads_the_gate_before_the_up() {
        let dev = Device::Cpu;
        // gate row picks channel 0, up row picks channel 1.
        let gate_up = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (1, 2, 2), &dev).unwrap();
        let down = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2, 1), &dev).unwrap();
        let banked = Experts::Banked {
            gate_up,
            down,
            intermediate: 1,
        };
        let xs = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), &dev).unwrap();
        let got: Vec<f32> = banked
            .forward_one(0, &xs)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // silu(gate.x) * (up.x) = silu(1) * 2
        let want = (1.0f32 / (1.0 + (-1.0f32).exp())) * 2.0;
        assert!((got[0] - want).abs() < 1e-5, "got {got:?} want {want}");
        // the swapped reading would be silu(2) * 1
        let swapped = (2.0f32 / (1.0 + (-2.0f32).exp())) * 1.0;
        assert!(
            (got[0] - swapped).abs() > 1e-3,
            "this is silu(up) * gate: {got:?}"
        );
    }

    /// In-situ quantisation has to approximate the dense expert, not
    /// merely produce numbers of the right shape.
    ///
    /// Dimensions are 256 because the k-quants block on 256 elements —
    /// a tiny fixture silently cannot be quantised at all, which is the
    /// first thing that bites when this is tried on a toy model.
    ///
    /// Experts restored from cache land on the device they are asked
    /// for, not the one the VarBuilder happens to use (#318, #322).
    ///
    /// This is the defect that made a host-resident load fill a 32 GB
    /// card in 37 seconds: the fresh-quantisation path honoured the
    /// residency target and the *cache* path did not, so 22 cached
    /// layers at 1.42 GB each went to VRAM at about a gigabyte a
    /// second while the code reported host residency in its logs. The
    /// two paths produce the same experts and must place them the same
    /// way; nothing but a test says so.
    #[test]
    fn cached_experts_land_where_they_are_asked_for() {
        let dev = Device::Cpu;
        let (n_experts, hidden, inter) = (2usize, 2560usize, 640usize);
        let gate_up = randn(&[n_experts, inter * 2, hidden]);
        let down = randn(&[n_experts, hidden, inter]);
        let fresh =
            Experts::quantize_banked_onto(&gate_up, &down, inter, GgmlDType::Q4K, &dev).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let cache =
            crate::harness::isq_cache::IsqCache::new(dir.path(), "test/model", Some("rev0"), "q4k")
                .unwrap();
        let entries = fresh.cache_entries().unwrap();
        let refs: Vec<(&str, &QTensor)> = entries
            .iter()
            .map(|(n, t)| (n.as_str(), t.as_ref()))
            .collect();
        cache.try_store("layer-0", &refs).unwrap();

        let restored =
            Experts::from_cache_entries(cache.load("layer-0", &dev).unwrap(), n_experts).unwrap();
        let Experts::Quantized(v) = &restored else {
            panic!("a cache load must produce quantised experts");
        };
        for (e, q) in v.iter().enumerate() {
            assert!(
                q.device()
                    .expect("a quantised expert knows its device")
                    .same_device(&dev),
                "expert {e} came back on the wrong device"
            );
        }
    }

    /// Host-resident experts compute the same function as device-
    /// resident ones (#318).
    ///
    /// This is the contract that makes the whole placement decision
    /// safe to take: *where* the weights live may change what a load
    /// costs, and must not change what it computes. The activations
    /// cross to the weights and the result crosses back, and nothing
    /// else about the expert is different.
    ///
    /// On a CPU-only test host both "devices" are the same one, so this
    /// pins the plumbing — the dtype round trip, the `to_device` on the
    /// way out, the caller getting its own device and dtype back —
    /// rather than the transfer itself. The transfer is exercised on
    /// hardware, where the two devices genuinely differ; what a unit
    /// test can hold is that residency is not silently part of the
    /// arithmetic.
    #[test]
    fn host_resident_experts_compute_the_same_function() {
        let (n_experts, hidden, inter) = (2usize, 2560usize, 640usize);
        let gate_up = randn(&[n_experts, inter * 2, hidden]);
        let down = randn(&[n_experts, hidden, inter]);

        let on_device =
            Experts::quantize_banked_onto(&gate_up, &down, inter, GgmlDType::Q4K, &Device::Cpu)
                .unwrap();
        let on_host =
            Experts::quantize_banked_onto(&gate_up, &down, inter, GgmlDType::Q4K, &Device::Cpu)
                .unwrap();

        let xs = randn(&[3, hidden]);
        for e in 0..n_experts {
            let a: Vec<f32> = on_device
                .forward_one(e, &xs)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let b = on_host.forward_one(e, &xs).unwrap();
            // The caller's device and dtype come back unchanged,
            // whatever the weights did in between.
            assert_eq!(b.dtype(), xs.dtype());
            assert!(b.device().same_device(xs.device()));
            let b: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(a, b, "expert {e} computed differently off-device");
        }
    }

    /// Experts restored from the on-disk cache compute the same
    /// function as the ones that filled it (#322).
    ///
    /// The cache's whole risk is that it returns something plausible:
    /// the model loads, it serves, and its weights are quietly not the
    /// ones it was asked for. So this asserts the restored experts are
    /// *bit-identical* in output, not close — there is no arithmetic
    /// between writing and reading that could legitimately differ.
    ///
    /// At the checkpoint's real widths, because they are what decide
    /// the types involved: 2560 blocks under Q4K, 640 does not and
    /// falls back to Q5_0, so this exercises a layer whose halves are
    /// *different* quantisation types — exactly the case a container
    /// storing one type per file could not represent, and the reason
    /// the artifact is GGUF.
    #[test]
    fn cached_experts_compute_the_same_function() {
        let dev = Device::Cpu;
        let (n_experts, hidden, inter) = (2usize, 2560usize, 640usize);
        let gate_up = randn(&[n_experts, inter * 2, hidden]);
        let down = randn(&[n_experts, hidden, inter]);
        let fresh = Experts::quantize_banked(&gate_up, &down, inter, GgmlDType::Q4K).unwrap();

        let entries = fresh
            .cache_entries()
            .expect("quantised experts must be cacheable");
        // The two halves really did take different types; if they had
        // not, this test would not be covering what it claims to.
        let dtypes: std::collections::HashSet<GgmlDType> =
            entries.iter().map(|(_, t)| t.dtype()).collect();
        assert!(
            dtypes.len() > 1,
            "expected a mix of quant types across gate/up and down, got {dtypes:?}"
        );

        let dir = tempfile::tempdir().unwrap();
        let cache =
            crate::harness::isq_cache::IsqCache::new(dir.path(), "test/model", Some("rev0"), "q4k")
                .unwrap();
        let refs: Vec<(&str, &QTensor)> = entries
            .iter()
            .map(|(n, t)| (n.as_str(), t.as_ref()))
            .collect();
        cache
            .try_store("layer-0", &refs)
            .expect("the write must succeed, and say why if it does not");

        let restored = Experts::from_cache_entries(
            cache
                .load("layer-0", &dev)
                .expect("the artifact just written"),
            n_experts,
        )
        .unwrap();

        let xs = randn(&[3, hidden]);
        for e in 0..n_experts {
            let a: Vec<f32> = fresh
                .forward_one(e, &xs)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let b: Vec<f32> = restored
                .forward_one(e, &xs)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(a, b, "expert {e} changed value on the cache round trip");
        }
    }

    /// `qwen4_exp`'s real expert widths, which no k-quant can block.
    ///
    /// `moe_intermediate_size` is 640, so `down_proj` is `[2560, 640]`
    /// and its 640-element rows are 2.5 blocks of 256. The load died
    /// on beast at decoder layer 0 with `quantized tensor must have
    /// their last dim divisible by block size [2560, 640] 256`, after
    /// resolving all 131 shards.
    ///
    /// The fixture that missed it was 256 wide in both dimensions —
    /// the smallest width a k-quant accepts — so every row divided
    /// and the orientation could not matter. This one uses the
    /// checkpoint's own numbers, scaled down only in expert count.
    #[test]
    fn the_real_expert_widths_quantise_despite_a_640_wide_reduction() {
        assert_eq!(
            quant_for_row(GgmlDType::Q4K, 2560).unwrap(),
            GgmlDType::Q4K,
            "gate_up reduces over hidden_size, which is 10 whole blocks"
        );
        assert_eq!(
            quant_for_row(GgmlDType::Q4K, 640).unwrap(),
            GgmlDType::Q5_0,
            "down reduces over moe_intermediate_size, which is not"
        );
        // llama.cpp's `tensor_type_fallback` table, which spends bits
        // rather than saving them: a legacy 32-block type is worse than
        // a k-quant at equal width, so upstream pays one to avoid the
        // quality cliff. Q4_0 above would be 4.5 bpw like Q4K and would
        // save 5.0 GB across the expert bank — that is the trade we
        // declined, and this is where it is pinned.
        assert_eq!(quant_for_row(GgmlDType::Q5K, 640).unwrap(), GgmlDType::Q5_1);
        assert_eq!(quant_for_row(GgmlDType::Q2K, 640).unwrap(), GgmlDType::Q4_0);
        assert_eq!(quant_for_row(GgmlDType::Q3K, 640).unwrap(), GgmlDType::Q4_0);
        assert_eq!(quant_for_row(GgmlDType::Q6K, 640).unwrap(), GgmlDType::Q8_0);
        assert_eq!(
            quant_for_row(GgmlDType::Q8_0, 640).unwrap(),
            GgmlDType::Q8_0
        );
        // 100 divides neither 256 nor 32: F16, as upstream does.
        assert_eq!(quant_for_row(GgmlDType::Q4K, 100).unwrap(), GgmlDType::F16);

        // And the whole path runs at those widths.
        let (n_experts, hidden, inter) = (2usize, 2560usize, 640usize);
        let gate_up = randn(&[n_experts, inter * 2, hidden]);
        let down = randn(&[n_experts, hidden, inter]);
        let quantised = Experts::quantize_banked(&gate_up, &down, inter, GgmlDType::Q4K)
            .expect("the real widths must quantise");
        let xs = randn(&[2, hidden]);
        let got = quantised.forward_one(0, &xs).expect("forward");
        assert_eq!(got.dims(), &[2, hidden]);
        assert!(
            got.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite()),
            "a substituted dtype must still produce finite output"
        );
    }

    /// The assertion is relative error against the dense output, with a
    /// noise baseline: a q8_0 expert should be close, a q4k expert
    /// looser, and both far nearer the truth than an unrelated expert
    /// is. Without that last comparison a "quantised" path that
    /// returned anything smoothly wrong would pass.
    #[test]
    fn quantised_experts_approximate_the_dense_ones() {
        let (n_experts, hidden, inter) = (2usize, 256usize, 256usize);
        let gate_up = randn(&[n_experts, inter * 2, hidden]);
        let down = randn(&[n_experts, hidden, inter]);
        let banked = Experts::Banked {
            gate_up: gate_up.clone(),
            down: down.clone(),
            intermediate: inter,
        };
        let xs = randn(&[4, hidden]);

        let rel_err = |a: &[f32], b: &[f32]| -> f32 {
            let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
            let den: f32 = a.iter().map(|x| x * x).sum::<f32>().max(1e-12);
            (num / den).sqrt()
        };
        let out = |e: &Experts, i: usize| -> Vec<f32> {
            e.forward_one(i, &xs)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
        };

        let dense0 = out(&banked, 0);
        // Baseline: a different expert on the same input. Any quantised
        // reading must beat this by a wide margin.
        let noise = rel_err(&dense0, &out(&banked, 1));

        for (dtype, tol) in [(GgmlDType::Q8_0, 0.05f32), (GgmlDType::Q4K, 0.20f32)] {
            let q = Experts::quantize_banked(&gate_up, &down, inter, dtype).unwrap();
            assert_eq!(q.len(), n_experts);
            let err = rel_err(&dense0, &out(&q, 0));
            assert!(err < tol, "{dtype:?} relative error {err:.4} exceeds {tol}");
            assert!(
                err < noise / 2.0,
                "{dtype:?} error {err:.4} is not clearly better than an unrelated \
                 expert ({noise:.4}) — the quantised path may not be reading \
                 this expert at all"
            );
        }
    }
}
