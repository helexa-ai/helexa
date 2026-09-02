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
use candle_core::{DType, IndexOp, Module, Tensor};
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
    pub(crate) fn quantize_banked(
        gate_up: &Tensor,
        down: &Tensor,
        intermediate: usize,
        dtype: GgmlDType,
    ) -> candle_core::Result<Self> {
        let n = gate_up.dims()[0];
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
            let q = |t: Tensor| -> candle_core::Result<QMatMul> {
                let dims = t.dims().to_vec();
                let vals = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let owned = Tensor::from_vec(vals, dims, t.device())?;
                QMatMul::from_qtensor(QTensor::quantize(&owned, dtype)?)
            };
            out.push(QuantizedExpert {
                gate: q(gu.narrow(0, 0, intermediate)?)?,
                up: q(gu.narrow(0, intermediate, intermediate)?)?,
                down: q(down.i(e)?)?,
            });
        }
        Ok(Experts::Quantized(out))
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
                // caller's dtype so the routing arithmetic upstream
                // does not have to know which layout it got.
                let dtype = xs.dtype();
                let xs32 = xs.to_dtype(DType::F32)?;
                let lhs = candle_nn::ops::silu(&q.gate.forward(&xs32)?)?;
                let rhs = q.up.forward(&xs32)?;
                q.down.forward(&(lhs * rhs)?)?.to_dtype(dtype)
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

        let (tokens_for, weights_for) = route_scatter(
            &self.gate,
            &xs_flat,
            self.experts.len(),
            self.num_experts_per_tok,
            self.norm_topk_prob,
        )?;

        let mut ys = xs_flat.zeros_like()?;
        for e in 0..self.experts.len() {
            if tokens_for[e].is_empty() {
                continue;
            }
            let rows = Tensor::new(tokens_for[e].as_slice(), xs.device())?;
            let picked = xs_flat.index_select(&rows, 0)?;
            let out = self.experts.forward_one(e, &picked)?;
            let w = Tensor::new(weights_for[e].as_slice(), xs.device())?
                .to_dtype(out.dtype())?
                .reshape(((), 1))?;
            ys = ys.index_add(&rows, &out.broadcast_mul(&w)?, 0)?;
        }

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
