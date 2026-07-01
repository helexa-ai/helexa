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
use candle_core::{DType, Module, Tensor};
use candle_nn::Linear;
use candle_nn::var_builder::ShardedVarBuilder;

use super::TextConfig;
use super::mlp::Qwen3_5MLP;

pub struct Qwen3_5MoeBlock {
    /// Router: `(num_experts, hidden)`, checkpoint name `mlp.gate`.
    gate: Linear,
    /// Routed experts, checkpoint names `mlp.experts.{i}.{gate,up,down}_proj`.
    experts: Vec<Qwen3_5MLP>,
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
            experts,
            shared_expert,
            shared_expert_gate,
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }
}

impl Module for Qwen3_5MoeBlock {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (b, l, hidden) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden))?;
        let n_tokens = b * l;

        // Router probabilities in f32 (reference uses float softmax
        // regardless of activations dtype).
        let router_logits = self.gate.forward(&xs_flat)?;
        let probs = candle_nn::ops::softmax_last_dim(&router_logits.to_dtype(DType::F32)?)?;

        // Top-k selection: descending argsort, take the first k. The
        // renormalisation (iff norm_topk_prob) divides by the sum of
        // the selected global-softmax values.
        let sorted = probs.arg_sort_last_dim(false)?;
        let topk_idx = sorted
            .narrow(1, 0, self.num_experts_per_tok)?
            .contiguous()?;
        let mut topk_w = probs.gather(&topk_idx, 1)?;
        if self.norm_topk_prob {
            let denom = topk_w.sum_keepdim(1)?;
            topk_w = topk_w.broadcast_div(&denom)?;
        }

        // Host-side scatter: token row lists per expert. Cheap relative
        // to the expert GEMMs; replaced by grouped-GEMM in slice 4.
        let idx_host: Vec<Vec<u32>> = topk_idx.to_vec2()?;
        let w_host: Vec<Vec<f32>> = topk_w.to_vec2()?;
        let mut tokens_for: Vec<Vec<u32>> = vec![Vec::new(); self.experts.len()];
        let mut weights_for: Vec<Vec<f32>> = vec![Vec::new(); self.experts.len()];
        for t in 0..n_tokens {
            for j in 0..self.num_experts_per_tok {
                let e = idx_host[t][j] as usize;
                tokens_for[e].push(t as u32);
                weights_for[e].push(w_host[t][j]);
            }
        }

        let mut ys = xs_flat.zeros_like()?;
        for (e, expert) in self.experts.iter().enumerate() {
            if tokens_for[e].is_empty() {
                continue;
            }
            let rows = Tensor::new(tokens_for[e].as_slice(), xs.device())?;
            let picked = xs_flat.index_select(&rows, 0)?;
            let out = expert.forward(&picked)?;
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
            experts: (0..n_exp).map(|_| rand_mlp(hidden, inter)).collect(),
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
                let out: Vec<f32> = block.experts[e]
                    .forward(&row)
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
            experts: (0..n_exp).map(|_| rand_mlp(hidden, inter)).collect(),
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
        let out: Vec<f32> = block.experts[best]
            .forward(&xs.reshape(((), hidden)).unwrap())
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
}
