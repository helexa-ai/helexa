//! `qwen4_exp`'s full-attention layer — one in every four.
//!
//! Mechanically this is `qwen3_5`'s: GQA 24:2, per-head `(1 + w)`
//! RMSNorm on q and k, a sigmoid output gate carried in a widened
//! `q_proj`, and `head_dim^-0.5` scaling. It is written here rather
//! than reused because of the one thing that differs — the [`Indexer`]
//! that decides which of the past this layer may look at — and because
//! the two architectures' configs are different types.
//!
//! Two details are worth stating where the code can be read against
//! them:
//!
//! 1. **The gate split is per head, not a halving.** `q_proj` is
//!    `num_heads * head_dim * 2` wide; reshape to
//!    `(.., num_heads, 2 * head_dim)` and take the halves *within* each
//!    head. Splitting the flat 12288 vector down the middle gives a
//!    different permutation — the first twelve heads' queries as the
//!    query and the last twelve as the gate — which trains nothing and
//!    reads as a weak model.
//! 2. **QSA replaces the causal mask; it does not join it.** The
//!    selection only ever offers positions at or behind the query, so
//!    the returned mask is already causal. Below the budget the indexer
//!    declines to produce one at all and this layer keeps the plain
//!    causal mask — which is also what keeps the flash path available
//!    for short prompts, since an additive mask forecloses it.
//!
//! See `doc/qwen4_exp-port-spec.md` §3 and §4.

use anyhow::{Context, Result};
use candle_core::{Module, Tensor};
use candle_nn::Linear;
use candle_nn::kv_cache::ConcatKvCache;
use candle_nn::var_builder::ShardedVarBuilder;
use std::sync::Arc;

use crate::harness::arch::qwen3_5::full_attn::{AttnMask, attention_context};
use crate::harness::arch::qwen3_5::rmsnorm::Qwen3_5RmsNorm;
use crate::harness::arch::qwen3_5::rope::RotaryEmbedding;

use super::config::TextConfig;
use super::qsa::Indexer;

pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Qwen3_5RmsNorm,
    k_norm: Qwen3_5RmsNorm,
    /// The sparse-attention side channel. Owns its own KV cache, which
    /// must be cleared with this layer's.
    indexer: Indexer,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    /// `num_heads * head_dim` — the attention output width, which is
    /// not `hidden_size` here (6144 against 2560).
    attn_width: usize,
    rotary: Arc<RotaryEmbedding>,
    kv_cache: ConcatKvCache,
}

impl Attention {
    pub fn load(
        cfg: &TextConfig,
        rotary: Arc<RotaryEmbedding>,
        vb: &ShardedVarBuilder,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        anyhow::ensure!(
            num_kv_heads > 0 && num_heads.is_multiple_of(num_kv_heads),
            "num_attention_heads ({num_heads}) must be a positive multiple of \
             num_key_value_heads ({num_kv_heads})"
        );
        let attn_width = num_heads * head_dim;
        Ok(Self {
            // Doubled output: query and gate, interleaved per head.
            q_proj: linear(vb, "q_proj", cfg.hidden_size, attn_width * 2)?,
            k_proj: linear(vb, "k_proj", cfg.hidden_size, num_kv_heads * head_dim)?,
            v_proj: linear(vb, "v_proj", cfg.hidden_size, num_kv_heads * head_dim)?,
            o_proj: linear(vb, "o_proj", attn_width, cfg.hidden_size)?,
            q_norm: Qwen3_5RmsNorm::load(&vb.pp("q_norm"), head_dim, cfg.rms_norm_eps)?,
            k_norm: Qwen3_5RmsNorm::load(&vb.pp("k_norm"), head_dim, cfg.rms_norm_eps)?,
            // The indexer is a submodule of the attention block, not a
            // sibling: `self_attn.indexer.index_qk_proj`, per the
            // checkpoint's own index.
            indexer: Indexer::load(&vb.pp("indexer"), cfg)?,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            attn_width,
            rotary,
            kv_cache: ConcatKvCache::new(2),
        })
    }

    /// `x` is the hyper-connection's mixed `hidden_size` input, not the
    /// four-stream residual. `causal_mask` is the layer's usual mask,
    /// used whenever the indexer declines to narrow it. `past_len` is
    /// the sequence offset, and must match what the KV cache holds.
    pub fn forward(
        &mut self,
        x: &Tensor,
        causal_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        past_len: usize,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        // The indexer reads the same hidden states this layer does, and
        // advances its own cache whether or not it produces a mask.
        let selected = self
            .indexer
            .binding_mask(x, &self.rotary, past_len, x.dtype())?;

        // q_proj is doubled: (query, gate) *within* each head.
        let q_raw = self
            .q_proj
            .forward(x)?
            .reshape((b, l, self.num_heads, self.head_dim * 2))?;
        let q = q_raw.narrow(3, 0, self.head_dim)?;
        let gate = q_raw
            .narrow(3, self.head_dim, self.head_dim)?
            .contiguous()?
            .reshape((b, l, self.attn_width))?;

        let q = self
            .q_norm
            .forward(&q.contiguous()?)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, l, self.num_kv_heads, self.head_dim))?;
        let k = self
            .k_norm
            .forward(&k.contiguous()?)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = self.rotary.apply_cos_sin(&q, &k, cos, sin)?;
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // A QSA mask is additive and arbitrary, so it must not reach the
        // flash kernel — the type says so, rather than a comment.
        let mask = match &selected {
            Some(m) => AttnMask::Additive(m),
            None => AttnMask::causal(causal_mask),
        };
        let scale = 1.0_f64 / (self.head_dim as f64).sqrt();
        let ctx = attention_context(&q, &k, &v, mask, self.num_kv_groups, scale)?;

        let ctx = ctx
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, l, self.attn_width))?;
        let gated = (ctx * candle_nn::ops::sigmoid(&gate)?)?;
        Ok(self.o_proj.forward(&gated)?)
    }

    /// Capture both caches at one token boundary.
    ///
    /// The same reason `clear_kv_cache` drops them together: a
    /// snapshot holding only the attention K/V restores a layer whose
    /// indexer still holds the previous request's keys, and the
    /// selection it computes from them is wrong without being invalid.
    pub fn snapshot_kv(&self) -> (Option<(Tensor, Tensor)>, Option<Tensor>) {
        let kv = match (self.kv_cache.k(), self.kv_cache.v()) {
            (Some(k), Some(v)) => Some((k.clone(), v.clone())),
            _ => None,
        };
        (kv, self.indexer.snapshot_keys())
    }

    /// Replace both caches from a snapshot.
    pub fn restore_kv(
        &mut self,
        kv: Option<&(Tensor, Tensor)>,
        indexer_keys: Option<&Tensor>,
    ) -> candle_core::Result<()> {
        self.kv_cache.reset();
        if let Some((k, v)) = kv {
            self.kv_cache.append(k, v)?;
        }
        self.indexer.restore_keys(indexer_keys);
        Ok(())
    }

    /// Both caches, together. Dropping only one leaves the indexer
    /// selecting blocks of a previous request's text.
    pub fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
        self.indexer.clear_cache();
    }

    pub fn indexer(&self) -> &Indexer {
        &self.indexer
    }
}

fn linear(vb: &ShardedVarBuilder, name: &str, in_dim: usize, out_dim: usize) -> Result<Linear> {
    let weight = vb
        .pp(name)
        .get((out_dim, in_dim), "weight")
        .with_context(|| format!("load '{}/{name}/weight'", vb.prefix()))?;
    Ok(Linear::new(weight, None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::arch::qwen3_5::TextConfig as Qwen3_5TextConfig;
    use crate::harness::arch::qwen4_exp::qsa::BlockSelector;
    use candle_core::{DType, Device};

    const HIDDEN: usize = 4;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 2;

    fn rope() -> Arc<RotaryEmbedding> {
        let json = format!(
            r#"{{"vocab_size": 8, "hidden_size": {HIDDEN}, "intermediate_size": 8,
                 "num_hidden_layers": 1, "num_attention_heads": {HEADS},
                 "num_key_value_heads": 1, "head_dim": {HEAD_DIM},
                 "max_position_embeddings": 64, "rms_norm_eps": 1e-6,
                 "rope_parameters": {{"rope_theta": 10000.0,
                                      "partial_rotary_factor": 1.0}}}}"#
        );
        let cfg: Qwen3_5TextConfig = serde_json::from_str(&json).unwrap();
        Arc::new(RotaryEmbedding::new(DType::F32, &cfg, &Device::Cpu).unwrap())
    }

    /// Columns of a weight matrix, given as rows of `[out]` values for
    /// input channel 0 and zeros elsewhere — so `W x` for `x = e0` is
    /// exactly the values given.
    fn from_first_column(values: &[f32], in_dim: usize) -> Linear {
        let out = values.len();
        let mut w = vec![0.0f32; out * in_dim];
        for (row, v) in values.iter().enumerate() {
            w[row * in_dim] = *v;
        }
        Linear::new(
            Tensor::from_vec(w, (out, in_dim), &Device::Cpu).unwrap(),
            None,
        )
    }

    fn eye(n: usize) -> Linear {
        let mut w = vec![0.0f32; n * n];
        for i in 0..n {
            w[i * n + i] = 1.0;
        }
        Linear::new(Tensor::from_vec(w, (n, n), &Device::Cpu).unwrap(), None)
    }

    /// A layer whose `q_proj` produces `[1,1, 2,2, 3,3, 4,4]` for the
    /// unit input, so the per-head and the halved reading of the gate
    /// disagree: per head the gate is `[2,2, 4,4]`, halved it would be
    /// `[3,3, 4,4]`.
    fn layer(budget: usize) -> Attention {
        let attn_width = HEADS * HEAD_DIM;
        Attention {
            q_proj: from_first_column(&[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0], HIDDEN),
            k_proj: from_first_column(&[0.0; HEAD_DIM], HIDDEN),
            v_proj: from_first_column(&[1.0; HEAD_DIM], HIDDEN),
            o_proj: eye(attn_width),
            q_norm: Qwen3_5RmsNorm::from_weight(
                Tensor::zeros(HEAD_DIM, DType::F32, &Device::Cpu).unwrap(),
                1e-6,
                None,
            ),
            k_norm: Qwen3_5RmsNorm::from_weight(
                Tensor::zeros(HEAD_DIM, DType::F32, &Device::Cpu).unwrap(),
                1e-6,
                None,
            ),
            indexer: Indexer::from_parts(
                from_first_column(&[0.5; 2 * HEAD_DIM], HIDDEN),
                BlockSelector::new(1, HEAD_DIM, 4, budget).unwrap(),
                1,
                1,
                HEAD_DIM,
            ),
            num_heads: HEADS,
            num_kv_heads: 1,
            num_kv_groups: HEADS,
            head_dim: HEAD_DIM,
            attn_width,
            rotary: rope(),
            kv_cache: ConcatKvCache::new(2),
        }
    }

    fn unit_input(seq: usize) -> Tensor {
        let mut v = vec![0.0f32; seq * HIDDEN];
        for i in 0..seq {
            v[i * HIDDEN] = 1.0;
        }
        Tensor::from_vec(v, (1, seq, HIDDEN), &Device::Cpu).unwrap()
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// The gate is the second half of each head, not the second half of
    /// the vector. With one position the attention is the identity on
    /// `v`, so the output is the gate alone — and the halved reading
    /// gives sigmoid(3) where the correct one gives sigmoid(2).
    #[test]
    fn the_output_gate_splits_within_each_head() {
        let mut attn = layer(2048);
        let (cos, sin) = attn.rotary.plain_cos_sin(0, 1).unwrap();
        let out: Vec<f32> = attn
            .forward(&unit_input(1), None, &cos, &sin, 0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let want = [sigmoid(2.0), sigmoid(2.0), sigmoid(4.0), sigmoid(4.0)];
        for (g, w) in out.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "got {out:?} want {want:?}");
        }
        // The halved reading would put sigmoid(3) in the first two.
        assert!(
            (out[0] - sigmoid(3.0)).abs() > 1e-3,
            "this is the halved split, not the per-head one: {out:?}"
        );
    }

    /// The indexer's cache and the main KV cache must advance together
    /// across a prefill and a decode step, and be dropped together — a
    /// surviving indexer cache selects blocks of the previous request.
    #[test]
    fn both_caches_advance_and_clear_together() {
        let mut attn = layer(2048);
        let (cos, sin) = attn.rotary.plain_cos_sin(0, 5).unwrap();
        attn.forward(&unit_input(5), None, &cos, &sin, 0).unwrap();
        assert_eq!(attn.indexer().cached_len(), 5);

        let (cos, sin) = attn.rotary.plain_cos_sin(5, 1).unwrap();
        attn.forward(&unit_input(1), None, &cos, &sin, 5).unwrap();
        assert_eq!(attn.indexer().cached_len(), 6);
        // Six positions is one block of four plus a tail — nowhere near
        // a 2048-token budget, so the layer stayed on the causal path.
        assert!(attn.indexer().selector().is_dense(6));

        attn.clear_kv_cache();
        assert_eq!(attn.indexer().cached_len(), 0);
    }

    /// A layer whose budget binds still produces a well-formed output —
    /// the sparse path is exercised end to end, not just the mask.
    #[test]
    fn the_sparse_path_runs() {
        let mut attn = layer(4);
        let (cos, sin) = attn.rotary.plain_cos_sin(0, 12).unwrap();
        let out = attn.forward(&unit_input(12), None, &cos, &sin, 0).unwrap();
        assert_eq!(out.dims(), &[1, 12, HIDDEN]);
        assert!(!attn.indexer().selector().is_dense(12));
        let values: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "a masked-out row would softmax to NaN: {values:?}"
        );
    }
}
