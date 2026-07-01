//! Qwen3-Next's `linear_attention` layer: Gated DeltaNet.
//!
//! The recurrent linear-attention block that occupies 3 out of every 4
//! decoder layers in Qwen3.6 (`layer_types[i] == "linear_attention"`).
//! Implemented against the reference Python in
//! `huggingface/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py`
//! (class `Qwen3_5GatedDeltaNet`).
//!
//! ## Block structure
//!
//! ```text
//! x  ── in_proj_qkv ── transpose ─► (B, conv_dim, L)
//!                                    │
//!     ┌──────────────── conv_state ──┤  prepend cached state (decode)
//!     ▼
//!   depthwise causal Conv1d (k=4) → SiLU
//!     │
//!     └─ split → q (k_dim), k (k_dim), v (v_dim)  ─►  per-head reshape
//!
//! x  ── in_proj_z ────────────────►  z (gate for the output RMSNorm)
//! x  ── in_proj_b ── sigmoid ─────►  beta (per-head per-token update rate)
//! x  ── in_proj_a ── softplus ────►  g (decay; see eqn below)
//!
//! g = -exp(A_log) * softplus(a + dt_bias)        # discretisation
//! beta = sigmoid(b)
//!
//! (q, k) ─── L2norm ─── delta rule loop ──── core_attn_out
//!                          (per-token, per-head):
//!                          state *= exp(g_t)
//!                          mem    = state^T · k_t
//!                          delta  = (v_t - mem) * beta_t
//!                          state += outer(k_t, delta)
//!                          out_t  = state^T · q_t
//!
//! core_attn_out ── RMSNormGated(z) ── reshape ── out_proj ── y
//! ```
//!
//! ## State
//!
//! Two tensors persist across decode steps:
//! - `conv_state`: `(B, conv_dim, conv_kernel_size)` — left-padded
//!   tail of the input to the depthwise conv, so the next causal
//!   window has the right left-context.
//! - `recurrent_state`: `(B, num_v_heads, head_k_dim, head_v_dim)` —
//!   the delta-rule outer-product memory.
//!
//! Both are cleared via [`GatedDeltaNet::clear_kv_cache`] at the start
//! of every new request.
//!
//! ## Performance note
//!
//! Prefill (seq_len ≥ 64) runs the **chunked** delta rule (#23) — the
//! algorithm in `torch_chunk_gated_delta_rule`, reorganised into
//! per-chunk batched matmuls; see [`run_chunk_gated_delta_rule`].
//! Decode steps and short prompts keep the **recurrent** per-token
//! rule (`torch_recurrent_gated_delta_rule`): a CUDA kernel on
//! device, a pure-Rust loop on CPU. Both produce identical results
//! (pinned by the `chunked_matches_recurrent_*` parity tests);
//! `NEURON_GDN_CHUNKED=0` forces the recurrent paths for A/B
//! measurement.

use anyhow::{Context, Result};
use candle_core::{Module, Tensor};
use candle_nn::Linear;
use candle_nn::var_builder::ShardedVarBuilder;

#[cfg(test)]
use super::RopeParameters;
use super::TextConfig;
use super::rmsnorm::{Qwen3_5RmsNormGated, l2norm};

/// Per-rank, per-layer state for the linear-attention block.
///
/// `conv_state` is left-padded with zeros on first use; `recurrent_state`
/// is initialised lazily to zeros once we know the batch size.
#[derive(Default)]
pub struct GatedDeltaNetState {
    pub conv_state: Option<Tensor>,
    pub recurrent_state: Option<Tensor>,
}

pub struct GatedDeltaNet {
    // Projections.
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    out_proj: Linear,

    // Depthwise causal Conv1d weight; shape (conv_dim, 1, kernel_size).
    // No bias (Python sets bias=False).
    conv1d_weight: Tensor,

    // Per-head discretisation params.
    dt_bias: Tensor,
    a_log: Tensor,

    // Output norm + gate.
    norm: Qwen3_5RmsNormGated,

    // Shape hyperparams (cached for forward).
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    conv_kernel_size: usize,

    // Recurrent state held inline. Each request resets via
    // `clear_kv_cache`; otherwise the state persists across forwards
    // and the per-token offset advances naturally.
    state: GatedDeltaNetState,
}

impl GatedDeltaNet {
    pub fn load(cfg: &TextConfig, vb: &ShardedVarBuilder) -> Result<Self> {
        let num_v_heads = cfg.linear_num_value_heads;
        let num_k_heads = cfg.linear_num_key_heads;
        let head_k_dim = cfg.linear_key_head_dim;
        let head_v_dim = cfg.linear_value_head_dim;
        let conv_kernel_size = cfg.linear_conv_kernel_dim;

        if num_v_heads == 0 || num_k_heads == 0 {
            anyhow::bail!(
                "Qwen3-Next linear_num_*_heads must be set; got v={num_v_heads}, k={num_k_heads}"
            );
        }
        if !num_v_heads.is_multiple_of(num_k_heads) {
            anyhow::bail!(
                "linear_num_value_heads ({num_v_heads}) must be a multiple of \
                 linear_num_key_heads ({num_k_heads}) for GQA-style head expansion"
            );
        }

        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;

        // ----- Linear projections (all `bias=False` in the reference). -----
        // Two checkpoint layouts exist for the input projections:
        // - Qwen3.6 (qwen3_5): separate `in_proj_qkv` / `in_proj_z` /
        //   `in_proj_b` / `in_proj_a`, with qkv stored as contiguous
        //   [Q | K | V] blocks — loads directly.
        // - Qwen3-Next 80B-A3B (qwen3_next, #92): fused `in_proj_qkvz`
        //   + `in_proj_ba`, **interleaved per key-head group** (see
        //   `split_fused_qkvz`/`split_fused_ba`) — de-interleaved once
        //   at load into the same contiguous layout, so the forward
        //   path (incl. the conv over [Q|K|V] channels) is unchanged.
        let (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a) =
            if vb.contains_tensor("in_proj_qkvz.weight") {
                let qkvz = vb
                    .pp("in_proj_qkvz")
                    .get((2 * key_dim + 2 * value_dim, cfg.hidden_size), "weight")
                    .with_context(|| format!("load '{}/in_proj_qkvz/weight'", vb.prefix()))?;
                let ba = vb
                    .pp("in_proj_ba")
                    .get((2 * num_v_heads, cfg.hidden_size), "weight")
                    .with_context(|| format!("load '{}/in_proj_ba/weight'", vb.prefix()))?;
                let (qkv_w, z_w) =
                    split_fused_qkvz(&qkvz, num_k_heads, num_v_heads, head_k_dim, head_v_dim)?;
                let (b_w, a_w) = split_fused_ba(&ba, num_k_heads, num_v_heads)?;
                (
                    Linear::new(qkv_w, None),
                    Linear::new(z_w, None),
                    Linear::new(b_w, None),
                    Linear::new(a_w, None),
                )
            } else {
                (
                    load_linear_no_bias(vb, "in_proj_qkv", cfg.hidden_size, conv_dim)?,
                    load_linear_no_bias(vb, "in_proj_z", cfg.hidden_size, value_dim)?,
                    load_linear_no_bias(vb, "in_proj_b", cfg.hidden_size, num_v_heads)?,
                    load_linear_no_bias(vb, "in_proj_a", cfg.hidden_size, num_v_heads)?,
                )
            };
        let out_proj = load_linear_no_bias(vb, "out_proj", value_dim, cfg.hidden_size)?;

        // ----- Conv1d weight (depthwise, bias=False). -----
        let conv1d_weight = vb
            .pp("conv1d")
            .get((conv_dim, 1, conv_kernel_size), "weight")
            .with_context(|| format!("load '{}/conv1d/weight'", vb.prefix()))?;

        // ----- dt_bias + A_log: per-head 1D params. -----
        let dt_bias = vb
            .get(num_v_heads, "dt_bias")
            .with_context(|| format!("load '{}/dt_bias'", vb.prefix()))?;
        let a_log = vb
            .get(num_v_heads, "A_log")
            .with_context(|| format!("load '{}/A_log'", vb.prefix()))?;

        // ----- Output gated RMSNorm (per-head_v_dim). -----
        let norm = Qwen3_5RmsNormGated::load(&vb.pp("norm"), head_v_dim, cfg.rms_norm_eps)?;

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            out_proj,
            conv1d_weight,
            dt_bias,
            a_log,
            norm,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size,
            state: GatedDeltaNetState::default(),
        })
    }

    pub fn clear_kv_cache(&mut self) {
        self.state = GatedDeltaNetState::default();
    }

    /// Deep-copy the recurrent state for a prefix snapshot. Must be a
    /// real copy (`Tensor::copy`), not a refcount clone: the CUDA
    /// delta-rule kernels write the state buffer in place, so a
    /// shared-storage snapshot would be corrupted by the next forward.
    pub fn snapshot_state(&self) -> candle_core::Result<(Option<Tensor>, Option<Tensor>)> {
        let conv = self
            .state
            .conv_state
            .as_ref()
            .map(Tensor::copy)
            .transpose()?;
        let rec = self
            .state
            .recurrent_state
            .as_ref()
            .map(Tensor::copy)
            .transpose()?;
        Ok((conv, rec))
    }

    /// Replace the live recurrent state with a deep copy of a
    /// previously captured snapshot. Deep copy for the same in-place
    /// kernel reason as [`Self::snapshot_state`] — the snapshot must
    /// survive being restored more than once.
    pub fn restore_state(
        &mut self,
        conv_state: Option<&Tensor>,
        recurrent_state: Option<&Tensor>,
    ) -> candle_core::Result<()> {
        self.state = GatedDeltaNetState {
            conv_state: conv_state.map(Tensor::copy).transpose()?,
            recurrent_state: recurrent_state.map(Tensor::copy).transpose()?,
        };
        Ok(())
    }

    /// `x` shape: `(B, L, hidden_size)`. Returns the same shape.
    pub fn forward(&mut self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let dtype = x.dtype();
        let device = x.device().clone();

        // ----- Projections. -----
        // mixed_qkv: (B, L, conv_dim)
        let mixed_qkv = self.in_proj_qkv.forward(x)?;
        // (B, conv_dim, L) for the conv1d.
        let mixed_qkv_chw = mixed_qkv.transpose(1, 2)?.contiguous()?;

        // z: (B, L, value_dim) → (B, L, num_v_heads, head_v_dim)
        let z = self.in_proj_z.forward(x)?.reshape((
            batch_size,
            seq_len,
            self.num_v_heads,
            self.head_v_dim,
        ))?;

        // b, a: (B, L, num_v_heads)
        let b = self.in_proj_b.forward(x)?;
        let a = self.in_proj_a.forward(x)?;

        // ----- Depthwise causal Conv1d + SiLU (with state continuation). -----
        // Dispatches to a cuda kernel that fuses conv1d + silu when
        // available; falls back to candle's `conv1d` + `silu` on cpu.
        let (conv_out, new_state) = run_causal_conv1d(
            &mixed_qkv_chw,
            &self.conv1d_weight,
            self.state.conv_state.take(),
            batch_size,
            self.conv_dim,
            seq_len,
            self.conv_kernel_size,
        )?;
        self.state.conv_state = Some(new_state);
        // Back to (B, L, conv_dim).
        let mixed_qkv = conv_out.transpose(1, 2)?.contiguous()?;

        // ----- Split into q, k, v. -----
        let q = mixed_qkv.narrow(2, 0, self.key_dim)?;
        let k = mixed_qkv.narrow(2, self.key_dim, self.key_dim)?;
        let v = mixed_qkv.narrow(2, 2 * self.key_dim, self.value_dim)?;

        let q = q.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        // ----- beta + g (per-head, per-token gates). -----
        // Fused on cuda; per-op Rust on cpu. Both paths produce:
        //   beta = sigmoid(b)
        //   g    = -exp(A_log) * softplus(a + dt_bias)
        let (beta, g) = run_fused_gating(&b, &a, &self.a_log, &self.dt_bias)?;

        // ----- GQA-style key expansion if num_v_heads > num_k_heads. -----
        let (q, k) = if self.num_v_heads > self.num_k_heads {
            let rep = self.num_v_heads / self.num_k_heads;
            (
                repeat_interleave(&q, rep, 2)?,
                repeat_interleave(&k, rep, 2)?,
            )
        } else {
            (q, k)
        };

        // ----- L2-norm on q, k (use_qk_l2norm_in_kernel=True in ref). -----
        let q = l2norm(&q, 1e-6)?;
        let k = l2norm(&k, 1e-6)?;

        // ----- Recurrent delta rule. -----
        // Inputs: q, k (B, L, H, D_k); v (B, L, H, D_v); g (B, L, H); beta (B, L, H).
        // The reference transposes to (B, H, L, D) before the loop. We
        // do the same — it makes per-token indexing trivial.
        let q = q.transpose(1, 2)?.contiguous()?; // (B, H, L, D_k)
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?; // (B, H, L, D_v)
        let g = g.transpose(1, 2)?.contiguous()?; // (B, H, L)
        let beta = beta.transpose(1, 2)?.contiguous()?; // (B, H, L)

        // Pre-scale q by 1/sqrt(D_k) once. Everything goes to f32 here
        // since the delta rule mixes broadcast_mul ops that candle won't
        // accept across mixed dtypes. On the cuda gating path both beta
        // and g come back in model dtype; on the cpu path g is already
        // f32 — both casts are cheap idempotent ops.
        let scale = 1.0_f64 / (self.head_k_dim as f64).sqrt();
        let q = (q.to_dtype(candle_core::DType::F32)? * scale)?;
        let k = k.to_dtype(candle_core::DType::F32)?;
        let v = v.to_dtype(candle_core::DType::F32)?;
        let g = g.to_dtype(candle_core::DType::F32)?;
        let beta = beta.to_dtype(candle_core::DType::F32)?;

        // Initialise the recurrent state from cache or zeros.
        let state_init = match self.state.recurrent_state.take() {
            Some(s) => s.to_dtype(candle_core::DType::F32)?,
            None => Tensor::zeros(
                (
                    batch_size,
                    self.num_v_heads,
                    self.head_k_dim,
                    self.head_v_dim,
                ),
                candle_core::DType::F32,
                &device,
            )?,
        };

        // The delta-rule body: cuda-accelerated `gated_delta_rule_recurrence`
        // kernel when we have a cuda device + the kernels are linked in,
        // pure-Rust per-token fallback otherwise.
        let (core_attn_out, new_state) = run_delta_rule(
            &q,
            &k,
            &v,
            &g,
            &beta,
            state_init,
            batch_size,
            self.num_v_heads,
            seq_len,
            self.head_k_dim,
            self.head_v_dim,
        )?;
        // Stash the updated recurrent state for the next call.
        self.state.recurrent_state = Some(new_state.to_dtype(dtype)?);

        // core_attn_out: (B, H, L, D_v) → (B, L, H, D_v) → (B*L*H, D_v).
        let core_attn_out = core_attn_out.transpose(1, 2)?.contiguous()?; // (B, L, H, D_v)
        let core_attn_out = core_attn_out.to_dtype(dtype)?;
        let core_attn_flat =
            core_attn_out.reshape((batch_size * seq_len * self.num_v_heads, self.head_v_dim))?;
        let z_flat = z.reshape((batch_size * seq_len * self.num_v_heads, self.head_v_dim))?;

        // RMSNormGated: (out * silu(z) * weight) with the norm.
        let normed = self.norm.forward(&core_attn_flat, &z_flat)?;
        let normed = normed.reshape((batch_size, seq_len, self.num_v_heads * self.head_v_dim))?;

        // Output projection: (B, L, value_dim) → (B, L, hidden_size).
        self.out_proj.forward(&normed)
    }
}

/// Run the per-token delta-rule recurrence.
///
/// `q`, `k`: `(B, H, L, D_k)` (F32). `v`: `(B, H, L, D_v)`. `g`,
/// `beta`: `(B, H, L)`. `state`: `(B, H, D_k, D_v)`.
///
/// Returns `(core_attn_out: (B, H, L, D_v), state: (B, H, D_k, D_v))`,
/// both F32. Caller is responsible for cast back to model dtype.
///
/// Cuda path: dispatches to the `gated_delta_rule_recurrence` kernel
/// ported from `EricLBuehler/mistral.rs::mistralrs-core/src/cuda/gdn.cu`.
/// All five inputs must be cuda f32 tensors. The kernel is V-tiled
/// with compile-time BK; one block per (V-tile, batch*head) and one
/// thread per V-column. Each thread holds BK state floats in
/// registers — eliminates the launch-overhead floor we hit with
/// candle's per-op dispatch (was ~12s/token on Qwen3.6-27B).
///
/// CPU path: pure-Rust per-token loop. Correct, slow.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Tensor,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> candle_core::Result<(Tensor, Tensor)> {
    // Prefill takes the chunk-parallel algorithm (#23): identical
    // delta-rule math reorganised into per-chunk matmuls (cuBLAS /
    // tensor cores on CUDA, gemm on CPU) instead of an O(L)-sequential
    // per-token recurrence. Decode steps (seq_len 1) and short
    // prompts stay on the recurrent paths below. The env kill switch
    // exists for A/B measurement on the fleet.
    const CHUNK_ALGO_THRESHOLD: usize = 64;
    if seq_len >= CHUNK_ALGO_THRESHOLD && chunked_prefill_enabled() {
        return run_chunk_gated_delta_rule(q, k, v, g, beta, state);
    }
    #[cfg(feature = "cuda")]
    {
        // Only dispatch to the kernel if the inputs are on a CUDA
        // device — CPU tests fall back to the Rust loop below.
        if q.device().is_cuda() {
            return run_delta_rule_cuda(
                q, k, v, g, beta, state, batch_size, num_heads, seq_len, head_k_dim, head_v_dim,
            );
        }
    }
    let _ = (batch_size, num_heads, head_k_dim, head_v_dim);
    run_delta_rule_rust(q, k, v, g, beta, state, seq_len)
}

/// `NEURON_GDN_CHUNKED=0` falls back to the per-token recurrent
/// paths for prefill — kept for A/B measurement on live hosts.
fn chunked_prefill_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NEURON_GDN_CHUNKED")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true)
    })
}

/// Chunk-parallel gated delta rule — a faithful port of the HF
/// reference `torch_chunk_gated_delta_rule` (chunk_size = 64) in
/// `transformers/models/qwen3_5/modeling_qwen3_5.py`, minus the steps
/// our caller has already done (q/k L2-norm, q pre-scaled by
/// `1/sqrt(D_k)`, inputs already `(B, H, L, D)` f32).
///
/// Same inputs/outputs as [`run_delta_rule`]'s recurrent paths:
/// `q`/`k`: `(B, H, L, D_k)`, `v`: `(B, H, L, D_v)`, `g`/`beta`:
/// `(B, H, L)`, `state`: `(B, H, D_k, D_v)` (zeros or a restored
/// prefix snapshot's recurrent state). Returns
/// `(out: (B, H, L, D_v), state: (B, H, D_k, D_v))`, all f32.
///
/// The reference's in-place UT-transform row loop is kept as-is
/// (with rows accumulating into a fresh tensor — candle tensors are
/// immutable); see the numerical-caution note at the loop for why the
/// tempting nilpotent-squaring shortcut is wrong. The parity tests
/// pin this against the recurrent path.
pub(crate) fn run_chunk_gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Tensor,
) -> candle_core::Result<(Tensor, Tensor)> {
    const C: usize = 64;
    let (b, h, l, dk) = q.dims4()?;
    let dv = v.dim(3)?;
    let device = q.device().clone();

    // Pad L up to a multiple of the chunk size. Padded positions
    // carry beta = 0 (no state update) and g = 0 (no decay), so they
    // are inert in the recurrence; their outputs are sliced off at
    // the end.
    let pad = (C - l % C) % C;
    let (q, k, v, g, beta) = if pad > 0 {
        (
            q.pad_with_zeros(2, 0, pad)?,
            k.pad_with_zeros(2, 0, pad)?,
            v.pad_with_zeros(2, 0, pad)?,
            g.pad_with_zeros(2, 0, pad)?,
            beta.pad_with_zeros(2, 0, pad)?,
        )
    } else {
        (q.clone(), k.clone(), v.clone(), g.clone(), beta.clone())
    };
    let lt = l + pad;
    let n = lt / C;

    let beta_e = beta.unsqueeze(3)?; // (B, H, Lt, 1)
    let v_beta = v.broadcast_mul(&beta_e)?;
    let k_beta = k.broadcast_mul(&beta_e)?;

    // Chunk reshape, flattening (B, H, N) into one batch dim — candle's
    // matmul supports at most two batch dims, so the chunk-local math
    // runs rank-3 over B·H·N and reshapes back to rank-5 for the
    // inter-chunk loop's per-chunk narrows.
    let bhn = b * h * n;
    let q3 = q.reshape((bhn, C, dk))?;
    let k3 = k.reshape((bhn, C, dk))?;
    let k_beta3 = k_beta.reshape((bhn, C, dk))?;
    let v_beta3 = v_beta.reshape((bhn, C, dv))?;

    // Within-chunk cumulative log-decay.
    let g3 = g.reshape((bhn, C))?.cumsum(1)?;

    // Lower-triangular masks, broadcast over the batch dim.
    let tril_incl = {
        let mut m = vec![0f32; C * C];
        for i in 0..C {
            for j in 0..=i {
                m[i * C + j] = 1.0;
            }
        }
        Tensor::from_vec(m, (C, C), &device)?
    };
    let tril_strict = {
        let mut m = vec![0f32; C * C];
        for i in 0..C {
            for j in 0..i {
                m[i * C + j] = 1.0;
            }
        }
        Tensor::from_vec(m, (C, C), &device)?
    };

    // decay_mask[i][j] = exp(g_i - g_j) on the lower triangle
    // (diagonal = 1), zero above. Mask-multiply replaces the
    // reference's tril/exp/tril dance: upper entries become
    // exp(0) = 1 mid-way and are re-zeroed.
    let g_col = g3.unsqueeze(2)?; // (BHN, C, 1)
    let g_row = g3.unsqueeze(1)?; // (BHN, 1, C)
    let decay_mask3 = g_col
        .broadcast_sub(&g_row)?
        .broadcast_mul(&tril_incl)?
        .exp()?
        .broadcast_mul(&tril_incl)?
        .contiguous()?;

    // T = strict lower of -((k_beta k^T) ⊙ decay), then
    // M = (I - T)^{-1} by forward substitution over rows — the
    // reference's in-place UT-transform loop, with processed rows
    // accumulating in `done` instead of mutating in place.
    //
    // Numerical caution: T is nilpotent (T^64 = 0), so the inverse
    // also equals Π (I + T^(2^j)) — six matmuls — but that form is
    // numerically unsafe: raw powers of T grow combinatorially
    // (path counts up to C(62,31) ≈ 4.6e17) before nilpotency
    // collapses them, destroying f32 precision on real prompts with
    // correlated keys. The forward substitution's intermediates are
    // the convergent M entries themselves, matching the reference's
    // behaviour exactly. Pinned by `chunked_ut_transform_survives_
    // correlated_keys`.
    let kkt = k_beta3.matmul(&k3.transpose(1, 2)?.contiguous()?)?;
    let t = kkt
        .broadcast_mul(&decay_mask3)?
        .broadcast_mul(&tril_strict)?
        .neg()?
        .contiguous()?;
    let eye = Tensor::eye(C, candle_core::DType::F32, &device)?;
    // Row 0 of the strict-lower T is all zeros and passes through
    // unchanged, seeding the processed-rows accumulator.
    let mut done = t.narrow(1, 0, 1)?.contiguous()?;
    for i in 1..C {
        let row = t.narrow(1, i, 1)?; // (BHN, 1, C)
        let coeffs = row.narrow(2, 0, i)?.contiguous()?; // (BHN, 1, i)
        let updated = (&row + coeffs.matmul(&done)?)?; // (BHN, 1, C)
        done = Tensor::cat(&[&done, &updated], 1)?;
    }
    let m = done.broadcast_add(&eye)?.contiguous()?;

    // value' = M v_beta ; k_cumdecay = M (k_beta ⊙ exp(g)).
    let value_c3 = m.matmul(&v_beta3.contiguous()?)?;
    let g_exp3 = g3.exp()?.unsqueeze(2)?; // (BHN, C, 1)
    let k_cumdecay3 = m.matmul(&k_beta3.broadcast_mul(&g_exp3)?.contiguous()?)?;

    // Rank-5 views for the per-chunk narrows below.
    let q = q3.reshape((b, h, n, C, dk))?;
    let k = k3.reshape((b, h, n, C, dk))?;
    let value_c = value_c3.reshape((b, h, n, C, dv))?;
    let k_cumdecay = k_cumdecay3.reshape((b, h, n, C, dk))?;
    let decay_mask = decay_mask3.reshape((b, h, n, C, C))?;
    let g = g3.reshape((b, h, n, C))?;

    // Inter-chunk recurrence: a handful of matmuls per 64 tokens.
    let mut state = state.to_dtype(candle_core::DType::F32)?;
    let mut outs: Vec<Tensor> = Vec::with_capacity(n);
    for i in 0..n {
        let q_i = q.narrow(2, i, 1)?.squeeze(2)?.contiguous()?; // (B, H, C, Dk)
        let k_i = k.narrow(2, i, 1)?.squeeze(2)?.contiguous()?;
        let v_i = value_c.narrow(2, i, 1)?.squeeze(2)?.contiguous()?; // (B, H, C, Dv)
        let dm_i = decay_mask.narrow(2, i, 1)?.squeeze(2)?; // (B, H, C, C)
        let g_i = g.narrow(2, i, 1)?.squeeze(2)?; // (B, H, C)
        let kcd_i = k_cumdecay.narrow(2, i, 1)?.squeeze(2)?.contiguous()?;

        let attn = q_i
            .matmul(&k_i.transpose(2, 3)?.contiguous()?)?
            .broadcast_mul(&dm_i)?
            .contiguous()?;
        let v_prime = kcd_i.matmul(&state)?;
        let v_new = (v_i - v_prime)?.contiguous()?;
        let g_i_exp = g_i.exp()?.unsqueeze(3)?; // (B, H, C, 1)
        let attn_inter = q_i.broadcast_mul(&g_i_exp)?.contiguous()?.matmul(&state)?;
        let out_i = (attn_inter + attn.matmul(&v_new)?)?;
        outs.push(out_i.unsqueeze(2)?);

        // state ← state · exp(g_last) + (k_i ⊙ exp(g_last - g_i))^T v_new
        let g_last = g_i.narrow(2, C - 1, 1)?; // (B, H, 1)
        let carry = g_last.exp()?.unsqueeze(3)?; // (B, H, 1, 1)
        let w = k_i.broadcast_mul(&g_last.broadcast_sub(&g_i)?.exp()?.unsqueeze(3)?)?;
        state =
            (state.broadcast_mul(&carry)? + w.transpose(2, 3)?.contiguous()?.matmul(&v_new)?)?;
    }

    let out = Tensor::cat(&outs, 2)?
        .reshape((b, h, lt, dv))?
        .narrow(2, 0, l)?
        .contiguous()?;
    Ok((out, state))
}

/// CUDA path. Flattens (B, H, ...) → (BH, ...) at the kernel boundary
/// (the kernel uses BH = batch*heads as its outer batch axis) and
/// reshapes the kernel's outputs back to (B, H, ...) for the caller.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_delta_rule_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Tensor,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> candle_core::Result<(Tensor, Tensor)> {
    let q_bh = q.flatten(0, 1)?.contiguous()?;
    let k_bh = k.flatten(0, 1)?.contiguous()?;
    let v_bh = v.flatten(0, 1)?.contiguous()?;
    let g_bh = g.flatten(0, 1)?.contiguous()?;
    let beta_bh = beta.flatten(0, 1)?.contiguous()?;
    let mut state_bh = state.flatten(0, 1)?.contiguous()?;
    // For long prefills, the chunked kernel (BT=64) processes a chunk
    // of tokens at a time instead of one-by-one — same delta-rule math,
    // far fewer block launches. Threshold matches mistralrs.
    const CHUNK_THRESHOLD: usize = 64;
    let output_bh = if seq_len >= CHUNK_THRESHOLD {
        crate::cuda::gdn::chunked_gated_delta_rule_recurrence_cuda(
            &q_bh,
            &k_bh,
            &v_bh,
            &g_bh,
            &beta_bh,
            &mut state_bh,
        )?
    } else {
        crate::cuda::gdn::gated_delta_rule_recurrence_cuda(
            &q_bh,
            &k_bh,
            &v_bh,
            &g_bh,
            &beta_bh,
            &mut state_bh,
        )?
    };
    let core_attn_out = output_bh.reshape((batch_size, num_heads, seq_len, head_v_dim))?;
    let new_state = state_bh.reshape((batch_size, num_heads, head_k_dim, head_v_dim))?;
    Ok((core_attn_out, new_state))
}

#[allow(clippy::too_many_arguments)]
fn run_delta_rule_rust(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    mut state: Tensor,
    seq_len: usize,
) -> candle_core::Result<(Tensor, Tensor)> {
    use candle_core::IndexOp;
    let mut outputs: Vec<Tensor> = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let q_t = q.i((.., .., t, ..))?;
        let k_t = k.i((.., .., t, ..))?;
        let v_t = v.i((.., .., t, ..))?;
        let g_t = g.i((.., .., t))?;
        let beta_t = beta.i((.., .., t))?;
        let decay = g_t
            .exp()?
            .unsqueeze(candle_core::D::Minus1)?
            .unsqueeze(candle_core::D::Minus1)?;
        state = state.broadcast_mul(&decay)?;
        let k_col = k_t.unsqueeze(candle_core::D::Minus1)?;
        let kv_mem = state.broadcast_mul(&k_col)?.sum(2)?;
        let beta_col = beta_t.unsqueeze(candle_core::D::Minus1)?;
        let delta = (v_t - kv_mem)?.broadcast_mul(&beta_col)?;
        let delta_row = delta.unsqueeze(2)?;
        let outer = k_col.broadcast_mul(&delta_row)?;
        state = (state + outer)?;
        let q_col = q_t.unsqueeze(candle_core::D::Minus1)?;
        let out_t = state.broadcast_mul(&q_col)?.sum(2)?;
        outputs.push(out_t.unsqueeze(2)?);
    }
    let core_attn_out = Tensor::cat(&outputs, 2)?; // (B, H, L, D_v)
    Ok((core_attn_out, state))
}

/// Depthwise causal conv1d + SiLU, with rolling `conv_state`.
///
/// `x`: `(B, conv_dim, L)` model dtype (f16/bf16 on cuda, anything on cpu).
/// `weight`: `(conv_dim, 1, kernel_size)` model dtype.
/// `conv_state`: `Some((B, conv_dim, kernel_size))` for decode continuation,
/// or `None` for fresh prefill.
///
/// Returns `(conv_out: (B, conv_dim, L), new_conv_state: (B, conv_dim, kernel_size))`.
/// SiLU is baked in.
///
/// Cuda path: dispatches to `causal_conv1d_update` (decode, seq_len=1 with
/// existing state) or `causal_conv1d_full` (prefill / first call), both
/// ported from mistralrs `gdn.cu`. Each kernel fuses the depthwise conv
/// and SiLU activation in one launch — that's ~4× fewer cuda launches per
/// linear-attention layer than the candle `conv1d` + `silu` combo.
///
/// CPU path: the original prepend-narrow-conv1d-silu sequence.
pub(crate) fn run_causal_conv1d(
    x: &Tensor,
    weight: &Tensor,
    conv_state: Option<Tensor>,
    batch_size: usize,
    conv_dim: usize,
    seq_len: usize,
    conv_kernel_size: usize,
) -> candle_core::Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    {
        if x.device().is_cuda() {
            return run_causal_conv1d_cuda(
                x,
                weight,
                conv_state,
                batch_size,
                conv_dim,
                seq_len,
                conv_kernel_size,
            );
        }
    }
    run_causal_conv1d_rust(
        x,
        weight,
        conv_state,
        batch_size,
        conv_dim,
        seq_len,
        conv_kernel_size,
    )
}

#[cfg(feature = "cuda")]
fn run_causal_conv1d_cuda(
    x: &Tensor,
    weight: &Tensor,
    conv_state: Option<Tensor>,
    batch_size: usize,
    conv_dim: usize,
    seq_len: usize,
    conv_kernel_size: usize,
) -> candle_core::Result<(Tensor, Tensor)> {
    // Kernel expects weight as (conv_dim, kernel_size) — squeeze the
    // depthwise channel-multiplier dim.
    let w = weight.squeeze(1)?.to_dtype(x.dtype())?.contiguous()?;

    // Decode path: seq_len == 1 AND we have an existing conv_state.
    // Otherwise (prefill or fresh-start decode), use the full path which
    // zero-pads on the left internally.
    if let Some(cs) = conv_state
        && seq_len == 1
    {
        let cs = cs.contiguous()?;
        let (output, new_conv_state) =
            crate::cuda::gdn::causal_conv1d_cuda(x, &w, &cs, conv_kernel_size, true)?;
        return Ok((output, new_conv_state));
    }

    // Prefill / fresh-start: the kernel ignores any prior conv_state and
    // zero-pads. If we had a non-zero prior state and >1 input tokens
    // (multi-turn continuation), we'd need to fall back to Rust. Match
    // mistralrs's behaviour: fresh prefill always.
    let device = x.device().clone();
    let zeros_cs = Tensor::zeros((batch_size, conv_dim, conv_kernel_size), x.dtype(), &device)?;
    let (output, new_conv_state) =
        crate::cuda::gdn::causal_conv1d_cuda(x, &w, &zeros_cs, conv_kernel_size, false)?;
    Ok((output, new_conv_state))
}

/// Fused GDN gating: computes `beta = sigmoid(b)` and
/// `g = -exp(a_log) * softplus(a + dt_bias)` together.
///
/// `b`, `a`: `(B, L, num_heads)` model dtype.
/// `a_log`, `dt_bias`: `(num_heads,)` model dtype (cast to f32 internally).
///
/// Returns `(beta, g)` both in model dtype on the cuda path, both in f32
/// on the cpu fallback. The caller casts to f32 before the delta rule.
///
/// Cuda path: dispatches to `fused_gdn_gating_cuda` — one kernel
/// replaces sigmoid + neg(exp) + softplus + broadcast_mul (≈10 candle
/// launches per layer).
pub(crate) fn run_fused_gating(
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> candle_core::Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    {
        if b.device().is_cuda() {
            let a_log_f32 = a_log.to_dtype(candle_core::DType::F32)?.contiguous()?;
            let dt_bias_f32 = dt_bias.to_dtype(candle_core::DType::F32)?.contiguous()?;
            return crate::cuda::gdn::fused_gdn_gating_cuda(b, a, &a_log_f32, &dt_bias_f32);
        }
    }
    run_fused_gating_rust(b, a, a_log, dt_bias)
}

fn run_fused_gating_rust(
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> candle_core::Result<(Tensor, Tensor)> {
    let beta = candle_nn::ops::sigmoid(b)?;
    let a_log_f32 = a_log.to_dtype(candle_core::DType::F32)?;
    let neg_a_exp = a_log_f32.exp()?.neg()?;
    let dt_b_f32 = dt_bias.to_dtype(candle_core::DType::F32)?;
    let a_f32 = a.to_dtype(candle_core::DType::F32)?;
    let a_plus_dt = a_f32.broadcast_add(&dt_b_f32)?;
    let softplus_val = softplus(&a_plus_dt)?;
    let neg_a_exp_b = neg_a_exp.unsqueeze(0)?.unsqueeze(0)?;
    let g = neg_a_exp_b.broadcast_mul(&softplus_val)?;
    Ok((beta, g))
}

fn run_causal_conv1d_rust(
    x: &Tensor,
    weight: &Tensor,
    conv_state: Option<Tensor>,
    batch_size: usize,
    conv_dim: usize,
    seq_len: usize,
    conv_kernel_size: usize,
) -> candle_core::Result<(Tensor, Tensor)> {
    let dtype = x.dtype();
    let device = x.device().clone();

    let prepended = match &conv_state {
        Some(prev) => Tensor::cat(&[prev, x], 2)?,
        None => x.clone(),
    };
    let prep_len = prepended.dims()[2];

    let new_state = if prep_len >= conv_kernel_size {
        prepended.narrow(2, prep_len - conv_kernel_size, conv_kernel_size)?
    } else {
        let pad = Tensor::zeros(
            (batch_size, conv_dim, conv_kernel_size - prep_len),
            dtype,
            &device,
        )?;
        Tensor::cat(&[&pad, &prepended], 2)?
    };

    let conv_out = prepended.conv1d(weight, conv_kernel_size - 1, 1, 1, conv_dim)?;
    let conv_out = conv_out.narrow(2, 0, prep_len)?;
    let conv_out = candle_nn::ops::silu(&conv_out)?;
    let conv_out = conv_out.narrow(2, prep_len - seq_len, seq_len)?;
    Ok((conv_out, new_state))
}

/// Load a no-bias linear from the ShardedVarBuilder. Weight shape is
/// the standard `[out, in]` order.
fn load_linear_no_bias(
    vb: &ShardedVarBuilder,
    name: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<Linear> {
    let weight = vb
        .pp(name)
        .get((out_dim, in_dim), "weight")
        .with_context(|| format!("load '{}/{name}/weight'", vb.prefix()))?;
    Ok(Linear::new(weight, None))
}

/// De-interleave a fused `in_proj_qkvz.weight` (qwen3_next layout, #92)
/// into a contiguous `[Q | K | V]` qkv weight plus a `Z` weight.
///
/// The fused rows are grouped **per key head**: for each of the
/// `num_k_heads` groups (`r = num_v_heads / num_k_heads`, group stride
/// `s = 2*head_k + 2*head_v*r`), the group holds
/// `[q (head_k) | k (head_k) | v (head_v*r) | z (head_v*r)]` — the
/// reshape in upstream `fix_query_key_value_ordering`
/// `(num_k_heads, 2*head_k + 2*head_v*num_v/num_k)`. Concatenating the
/// per-group regions restores the global-contiguous layout the rest of
/// this module (incl. the conv over `[Q|K|V]` channels) expects.
fn split_fused_qkvz(
    qkvz: &Tensor,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> Result<(Tensor, Tensor)> {
    let r = num_v_heads / num_k_heads;
    let stride = 2 * head_k_dim + 2 * head_v_dim * r;
    let (mut qs, mut ks, mut vs, mut zs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for g in 0..num_k_heads {
        let base = g * stride;
        qs.push(qkvz.narrow(0, base, head_k_dim)?);
        ks.push(qkvz.narrow(0, base + head_k_dim, head_k_dim)?);
        vs.push(qkvz.narrow(0, base + 2 * head_k_dim, head_v_dim * r)?);
        zs.push(qkvz.narrow(0, base + 2 * head_k_dim + head_v_dim * r, head_v_dim * r)?);
    }
    let parts: Vec<Tensor> = qs.into_iter().chain(ks).chain(vs).collect();
    let qkv = Tensor::cat(&parts, 0)?.contiguous()?;
    let z = Tensor::cat(&zs, 0)?.contiguous()?;
    Ok((qkv, z))
}

/// De-interleave a fused `in_proj_ba.weight` (qwen3_next layout, #92)
/// into per-v-head `b` (beta) and `a` (decay) weights. Same per-key-head
/// grouping as [`split_fused_qkvz`]: each group holds `[b (r) | a (r)]`
/// rows, `r = num_v_heads / num_k_heads`.
fn split_fused_ba(ba: &Tensor, num_k_heads: usize, num_v_heads: usize) -> Result<(Tensor, Tensor)> {
    let r = num_v_heads / num_k_heads;
    let (mut bs, mut r#as) = (Vec::new(), Vec::new());
    for g in 0..num_k_heads {
        let base = g * 2 * r;
        bs.push(ba.narrow(0, base, r)?);
        r#as.push(ba.narrow(0, base + r, r)?);
    }
    let b = Tensor::cat(&bs, 0)?.contiguous()?;
    let a = Tensor::cat(&r#as, 0)?.contiguous()?;
    Ok((b, a))
}

/// Numerically-stable `softplus(x) = ln(1 + exp(x))`. Matches PyTorch's
/// `F.softplus` default (beta=1, threshold=20: for large positive x,
/// returns x as-is to avoid overflow in the exp).
pub(crate) fn softplus(x: &Tensor) -> candle_core::Result<Tensor> {
    let threshold = 20.0_f64;
    let big = x.ge(threshold)?; // Tensor<u8> mask
    let safe = x.minimum(&x.affine(0.0, 0.0)?.affine(0.0, threshold)?)?; // min(x, threshold)
    let small = ((safe.exp()? + 1.0_f64)?).log()?;
    // Select x where big, else small.
    big.where_cond(x, &small)
}

/// `repeat_interleave` along a single dim. Candle has no built-in for
/// this; emulate with unsqueeze + expand + reshape.
pub(crate) fn repeat_interleave(
    x: &Tensor,
    repeats: usize,
    dim: usize,
) -> candle_core::Result<Tensor> {
    if repeats == 1 {
        return Ok(x.clone());
    }
    let mut shape = x.dims().to_vec();
    let orig = shape[dim];
    shape.insert(dim + 1, repeats);
    let mut expanded_shape = shape.clone();
    expanded_shape[dim + 1] = repeats;
    let x = x.unsqueeze(dim + 1)?;
    let x = x.expand(expanded_shape)?;
    let mut out_shape = x.dims().to_vec();
    out_shape.remove(dim + 1);
    out_shape[dim] = orig * repeats;
    x.contiguous()?.reshape(out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Plausible delta-rule inputs matching `run_delta_rule`'s
    /// contract: q/k L2-normed (q pre-scaled by 1/sqrt(D_k)), g a
    /// negative log-decay, beta in (0, 1). All f32 on CPU.
    fn delta_rule_inputs(
        b: usize,
        h: usize,
        l: usize,
        dk: usize,
        dv: usize,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let dev = Device::Cpu;
        let scale = 1.0 / (dk as f64).sqrt();
        let q = Tensor::randn(0f32, 1.0, (b, h, l, dk), &dev).unwrap();
        let q = (l2norm(&q, 1e-6).unwrap() * scale).unwrap();
        let k = Tensor::randn(0f32, 1.0, (b, h, l, dk), &dev).unwrap();
        let k = l2norm(&k, 1e-6).unwrap();
        let v = (Tensor::randn(0f32, 1.0, (b, h, l, dv), &dev).unwrap() * 0.5).unwrap();
        // g in (-1, 0): a realistic per-token log-decay.
        let g = (Tensor::rand(0f32, 1f32, (b, h, l), &dev).unwrap() * -1.0).unwrap();
        let beta = Tensor::rand(0.05f32, 0.95f32, (b, h, l), &dev).unwrap();
        (q, k, v, g, beta)
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// The #23 parity gate: the chunk-parallel algorithm must produce
    /// the same outputs and final state as the per-token recurrence.
    /// L = 130 exercises the pad-to-chunk-multiple path (130 = 2×64 + 2).
    #[test]
    fn chunked_matches_recurrent_with_padding() {
        let (b, h, l, dk, dv) = (1, 2, 130, 16, 16);
        let (q, k, v, g, beta) = delta_rule_inputs(b, h, l, dk, dv);
        let zeros = || Tensor::zeros((b, h, dk, dv), DType::F32, &Device::Cpu).unwrap();

        let (out_rec, state_rec) = run_delta_rule_rust(&q, &k, &v, &g, &beta, zeros(), l).unwrap();
        let (out_chk, state_chk) =
            run_chunk_gated_delta_rule(&q, &k, &v, &g, &beta, zeros()).unwrap();

        assert_eq!(out_chk.dims(), out_rec.dims());
        let d_out = max_abs_diff(&out_rec, &out_chk);
        let d_state = max_abs_diff(&state_rec, &state_chk);
        assert!(d_out < 2e-4, "output diverged: {d_out}");
        assert!(d_state < 2e-4, "final state diverged: {d_state}");
    }

    /// Exact chunk multiple (no padding) continuing from a non-zero
    /// initial state — the prefix-cache-restore (#11) interaction.
    #[test]
    fn chunked_matches_recurrent_with_initial_state() {
        let (b, h, dk, dv) = (1, 2, 16, 16);
        let dev = Device::Cpu;
        // Build a non-trivial initial state by running the recurrent
        // path over a 50-token "restored prefix".
        let (pq, pk, pv, pg, pbeta) = delta_rule_inputs(b, h, 50, dk, dv);
        let zeros = Tensor::zeros((b, h, dk, dv), DType::F32, &dev).unwrap();
        let (_, state0) = run_delta_rule_rust(&pq, &pk, &pv, &pg, &pbeta, zeros, 50).unwrap();

        let l = 128;
        let (q, k, v, g, beta) = delta_rule_inputs(b, h, l, dk, dv);
        let (out_rec, state_rec) =
            run_delta_rule_rust(&q, &k, &v, &g, &beta, state0.clone(), l).unwrap();
        let (out_chk, state_chk) =
            run_chunk_gated_delta_rule(&q, &k, &v, &g, &beta, state0).unwrap();

        let d_out = max_abs_diff(&out_rec, &out_chk);
        let d_state = max_abs_diff(&state_rec, &state_chk);
        assert!(d_out < 2e-4, "output diverged: {d_out}");
        assert!(d_state < 2e-4, "final state diverged: {d_state}");
    }

    /// Adversarially correlated inputs: near-identical keys with
    /// beta ≈ 1 and negligible decay make the UT-transform matrix T
    /// maximally coherent — raw powers of T grow combinatorially
    /// (≈ C(62,31) paths), which destroyed f32 precision in the
    /// nilpotent-squaring formulation this test exists to forbid.
    /// Real prompts hit this through repetitive text (observed live
    /// on beast: NaN logits → "!!!" replies). Forward substitution
    /// must stay finite and match the recurrent path.
    #[test]
    fn chunked_ut_transform_survives_correlated_keys() {
        let (b, h, l, dk, dv) = (1, 1, 192, 16, 16);
        let dev = Device::Cpu;
        let scale = 1.0 / (dk as f64).sqrt();
        // One base direction plus a whisper of noise: every key is
        // nearly the same unit vector.
        let base = Tensor::randn(0f32, 1.0, (1, 1, 1, dk), &dev).unwrap();
        let noise = (Tensor::randn(0f32, 1.0, (b, h, l, dk), &dev).unwrap() * 0.01).unwrap();
        let k = l2norm(&base.broadcast_add(&noise).unwrap(), 1e-6).unwrap();
        let q = (l2norm(&base.broadcast_add(&noise).unwrap(), 1e-6).unwrap() * scale).unwrap();
        let v = (Tensor::randn(0f32, 1.0, (b, h, l, dv), &dev).unwrap() * 0.5).unwrap();
        // Almost no decay, near-unit update rate — worst case for T.
        let g = (Tensor::rand(0f32, 1f32, (b, h, l), &dev).unwrap() * -1e-3).unwrap();
        let beta = Tensor::rand(0.98f32, 0.999f32, (b, h, l), &dev).unwrap();
        let zeros = || Tensor::zeros((b, h, dk, dv), DType::F32, &dev).unwrap();

        let (out_rec, state_rec) = run_delta_rule_rust(&q, &k, &v, &g, &beta, zeros(), l).unwrap();
        let (out_chk, state_chk) =
            run_chunk_gated_delta_rule(&q, &k, &v, &g, &beta, zeros()).unwrap();

        let finite: Vec<f32> = out_chk.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            finite.iter().all(|x| x.is_finite()),
            "chunked output not finite on correlated inputs"
        );
        let d_out = max_abs_diff(&out_rec, &out_chk);
        let d_state = max_abs_diff(&state_rec, &state_chk);
        assert!(
            d_out < 5e-3,
            "output diverged on correlated inputs: {d_out}"
        );
        assert!(
            d_state < 5e-3,
            "final state diverged on correlated inputs: {d_state}"
        );
    }

    /// A single exact chunk — the smallest input the dispatch sends to
    /// the chunked path.
    #[test]
    fn chunked_matches_recurrent_single_chunk() {
        let (b, h, l, dk, dv) = (2, 3, 64, 8, 8);
        let (q, k, v, g, beta) = delta_rule_inputs(b, h, l, dk, dv);
        let zeros = || Tensor::zeros((b, h, dk, dv), DType::F32, &Device::Cpu).unwrap();

        let (out_rec, state_rec) = run_delta_rule_rust(&q, &k, &v, &g, &beta, zeros(), l).unwrap();
        let (out_chk, state_chk) =
            run_chunk_gated_delta_rule(&q, &k, &v, &g, &beta, zeros()).unwrap();

        let d_out = max_abs_diff(&out_rec, &out_chk);
        let d_state = max_abs_diff(&state_rec, &state_chk);
        assert!(d_out < 2e-4, "output diverged: {d_out}");
        assert!(d_state < 2e-4, "final state diverged: {d_state}");
    }

    #[test]
    fn softplus_small_x() {
        // softplus(0) = ln(2) ≈ 0.6931
        let x = Tensor::new(&[0.0_f32], &Device::Cpu).unwrap();
        let out: Vec<f32> = softplus(&x).unwrap().to_vec1().unwrap();
        assert!((out[0] - 2.0_f32.ln()).abs() < 1e-4);
    }

    #[test]
    fn softplus_large_x_returns_x() {
        // For x = 30, softplus(x) ≈ x (the threshold branch).
        let x = Tensor::new(&[30.0_f32], &Device::Cpu).unwrap();
        let out: Vec<f32> = softplus(&x).unwrap().to_vec1().unwrap();
        assert!((out[0] - 30.0).abs() < 1e-4);
    }

    #[test]
    fn repeat_interleave_doubles_dim() {
        let x = Tensor::new(&[[1.0_f32, 2.0], [3.0, 4.0]], &Device::Cpu).unwrap(); // shape (2, 2)
        let out = repeat_interleave(&x, 2, 1).unwrap(); // each col duplicated
        let v: Vec<Vec<f32>> = out.to_vec2().unwrap();
        // Row 0: 1, 1, 2, 2
        // Row 1: 3, 3, 4, 4
        assert_eq!(v[0], vec![1.0, 1.0, 2.0, 2.0]);
        assert_eq!(v[1], vec![3.0, 3.0, 4.0, 4.0]);
    }

    /// Sanity: the recurrent path produces a finite tensor of the right
    /// shape on tiny dimensions. Doesn't validate numerical correctness
    /// against the Python reference — that would need a fixed-weight
    /// fixture to compare against. Catches structural mistakes
    /// (broadcasting shapes, off-by-one slices) early.
    #[test]
    fn forward_smoke_with_tiny_dimensions() {
        let dev = Device::Cpu;
        let dtype = DType::F32;
        let (b, l) = (1, 3);
        let cfg = TextConfig {
            vocab_size: 100,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 32,
            rope_parameters: RopeParameters {
                rope_theta: 10000.0,
                partial_rotary_factor: 1.0,
                rope_type: None,
                mrope_section: Vec::new(),
                mrope_interleaved: false,
            },
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            attn_output_gate: true,
            layer_types: vec!["linear_attention".into()],
            full_attention_interval: Some(4),
            hidden_act: "silu".into(),
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            decoder_sparse_step: 1,
            mlp_only_layers: Vec::new(),
            norm_topk_prob: false,
        };

        // Build a synthetic VarBuilder with all-zeros weights.
        // Easier path: skip the load and construct GatedDeltaNet
        // manually by hand-rolling the Linear/Tensor inputs.
        let zeros = |shape: &[usize]| Tensor::zeros(shape, dtype, &dev).unwrap();
        let key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads;
        let value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let mut net = GatedDeltaNet {
            in_proj_qkv: Linear::new(zeros(&[conv_dim, cfg.hidden_size]), None),
            in_proj_z: Linear::new(zeros(&[value_dim, cfg.hidden_size]), None),
            in_proj_b: Linear::new(zeros(&[cfg.linear_num_value_heads, cfg.hidden_size]), None),
            in_proj_a: Linear::new(zeros(&[cfg.linear_num_value_heads, cfg.hidden_size]), None),
            out_proj: Linear::new(zeros(&[cfg.hidden_size, value_dim]), None),
            conv1d_weight: zeros(&[conv_dim, 1, cfg.linear_conv_kernel_dim]),
            dt_bias: zeros(&[cfg.linear_num_value_heads]),
            a_log: zeros(&[cfg.linear_num_value_heads]),
            norm: {
                let weight = Tensor::ones(&[cfg.linear_value_head_dim], dtype, &dev).unwrap();
                Qwen3_5RmsNormGated::from_weight(weight, cfg.rms_norm_eps)
            },
            num_v_heads: cfg.linear_num_value_heads,
            num_k_heads: cfg.linear_num_key_heads,
            head_k_dim: cfg.linear_key_head_dim,
            head_v_dim: cfg.linear_value_head_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size: cfg.linear_conv_kernel_dim,
            state: GatedDeltaNetState::default(),
        };

        let x = Tensor::ones(&[b, l, cfg.hidden_size], dtype, &dev).unwrap();
        let y = net.forward(&x).unwrap();
        assert_eq!(y.dims(), &[b, l, cfg.hidden_size]);
        // All zero weights → output should be zero. Confirms no NaN/Inf
        // poisoning from the f32 promotions.
        let v: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    /// Interleave known per-head Q/K/V/Z (and B/A) rows into the fused
    /// qwen3_next layout, split, and expect the original contiguous
    /// blocks back. Layout under test: per key-head group g,
    /// `[q_g | k_g | v_g | z_g]` with r = num_v/num_k value heads per
    /// group (upstream `fix_query_key_value_ordering`).
    #[test]
    fn split_fused_qkvz_and_ba_roundtrip() {
        let dev = Device::Cpu;
        let (num_k, num_v, head_k, head_v, hidden) = (2usize, 4usize, 3usize, 2usize, 5usize);
        let r = num_v / num_k;

        // Distinct constant per logical row so any mis-slicing shows.
        let row = |tag: f32| Tensor::full(tag, (1, hidden), &dev).unwrap();
        let mut fused_rows: Vec<Tensor> = Vec::new();
        let (mut q_rows, mut k_rows, mut v_rows, mut z_rows) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for g in 0..num_k {
            let base = 1000.0 * (g as f32 + 1.0);
            for i in 0..head_k {
                let t = row(base + i as f32);
                fused_rows.push(t.clone());
                q_rows.push(t);
            }
            for i in 0..head_k {
                let t = row(base + 100.0 + i as f32);
                fused_rows.push(t.clone());
                k_rows.push(t);
            }
            for i in 0..head_v * r {
                let t = row(base + 200.0 + i as f32);
                fused_rows.push(t.clone());
                v_rows.push(t);
            }
            for i in 0..head_v * r {
                let t = row(base + 300.0 + i as f32);
                fused_rows.push(t.clone());
                z_rows.push(t);
            }
        }
        let fused = Tensor::cat(&fused_rows, 0).unwrap();
        let expected_qkv = Tensor::cat(
            &q_rows
                .iter()
                .chain(k_rows.iter())
                .chain(v_rows.iter())
                .cloned()
                .collect::<Vec<_>>(),
            0,
        )
        .unwrap();
        let expected_z = Tensor::cat(&z_rows, 0).unwrap();

        let (qkv, z) = split_fused_qkvz(&fused, num_k, num_v, head_k, head_v).unwrap();
        assert_eq!(qkv.dims(), &[2 * num_k * head_k + num_v * head_v, hidden]);
        let diff_qkv: f32 = (qkv - expected_qkv)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        let diff_z: f32 = (z - expected_z)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert_eq!(diff_qkv, 0.0);
        assert_eq!(diff_z, 0.0);

        // ba: per group, [b (r rows) | a (r rows)].
        let mut ba_rows = Vec::new();
        let (mut b_rows, mut a_rows) = (Vec::new(), Vec::new());
        for g in 0..num_k {
            let base = 10.0 * (g as f32 + 1.0);
            for i in 0..r {
                let t = row(base + i as f32);
                ba_rows.push(t.clone());
                b_rows.push(t);
            }
            for i in 0..r {
                let t = row(base + 5.0 + i as f32);
                ba_rows.push(t.clone());
                a_rows.push(t);
            }
        }
        let ba = Tensor::cat(&ba_rows, 0).unwrap();
        let (b, a) = split_fused_ba(&ba, num_k, num_v).unwrap();
        let diff_b: f32 = (b - Tensor::cat(&b_rows, 0).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        let diff_a: f32 = (a - Tensor::cat(&a_rows, 0).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert_eq!(diff_b, 0.0);
        assert_eq!(diff_a, 0.0);
    }
}
