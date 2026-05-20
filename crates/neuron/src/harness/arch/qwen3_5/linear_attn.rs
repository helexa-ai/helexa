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
//! This impl is the **recurrent** delta-rule for both prefill and
//! decode — i.e. the algorithm in `torch_recurrent_gated_delta_rule`.
//! Correctness-first. The chunked algorithm (chunk_size=64) in
//! `torch_chunk_gated_delta_rule` is a perf optimisation for long
//! prefill; can be added later without changing the surface.

use anyhow::{Context, Result};
use candle_core::{IndexOp, Module, Tensor};
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
        let in_proj_qkv = load_linear_no_bias(vb, "in_proj_qkv", cfg.hidden_size, conv_dim)?;
        let in_proj_z = load_linear_no_bias(vb, "in_proj_z", cfg.hidden_size, value_dim)?;
        let in_proj_b = load_linear_no_bias(vb, "in_proj_b", cfg.hidden_size, num_v_heads)?;
        let in_proj_a = load_linear_no_bias(vb, "in_proj_a", cfg.hidden_size, num_v_heads)?;
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
        // If the previous step left a `conv_state`, prepend it so the
        // causal kernel window sees the correct left-context.
        let prepended = match &self.state.conv_state {
            Some(prev) => Tensor::cat(&[prev, &mixed_qkv_chw], 2)?,
            None => mixed_qkv_chw.clone(),
        };
        let prep_len = prepended.dims()[2];

        // Update conv_state: keep the last `conv_kernel_size` columns
        // of the (possibly prepended) sequence. If the sequence is
        // shorter than `conv_kernel_size` (very-short prefill or first
        // decode step before warmup), left-pad with zeros.
        let new_state = if prep_len >= self.conv_kernel_size {
            prepended.narrow(2, prep_len - self.conv_kernel_size, self.conv_kernel_size)?
        } else {
            let pad = Tensor::zeros(
                (batch_size, self.conv_dim, self.conv_kernel_size - prep_len),
                dtype,
                &device,
            )?;
            Tensor::cat(&[&pad, &prepended], 2)?
        };
        self.state.conv_state = Some(new_state);

        // Apply the depthwise conv with padding=kernel-1 (so output
        // length = input + kernel - 1), then trim back to `prep_len`.
        // Matches the reference Python which calls the same nn.Conv1d
        // with its baked-in padding and slices `[..., :input_len]`.
        let conv_out = prepended.conv1d(
            &self.conv1d_weight,
            self.conv_kernel_size - 1,
            1,
            1,
            self.conv_dim,
        )?;
        let conv_out = conv_out.narrow(2, 0, prep_len)?;
        let conv_out = candle_nn::ops::silu(&conv_out)?;
        // Keep only the last L outputs (drop the prepended-state contribution).
        let conv_out = conv_out.narrow(2, prep_len - seq_len, seq_len)?;
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
        // beta = sigmoid(b)
        let beta = candle_nn::ops::sigmoid(&b)?;
        // g = -exp(A_log) * softplus(a + dt_bias)
        // Promote everything to f32 — the Python does the same to
        // avoid underflow on the -exp path.
        let a_log_f32 = self.a_log.to_dtype(candle_core::DType::F32)?;
        let neg_a_exp = a_log_f32.exp()?.neg()?; // (num_v_heads,)
        let dt_b_f32 = self.dt_bias.to_dtype(candle_core::DType::F32)?;
        let a_f32 = a.to_dtype(candle_core::DType::F32)?;
        // a is (B, L, num_v_heads); broadcast-add dt_bias.
        let a_plus_dt = a_f32.broadcast_add(&dt_b_f32)?;
        let softplus = softplus(&a_plus_dt)?;
        // (1, 1, num_v_heads) × (B, L, num_v_heads).
        let neg_a_exp_b = neg_a_exp.unsqueeze(0)?.unsqueeze(0)?;
        let g = neg_a_exp_b.broadcast_mul(&softplus)?;

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

        // Pre-scale q by 1/sqrt(D_k) once.
        let scale = 1.0_f64 / (self.head_k_dim as f64).sqrt();
        let q = (q.to_dtype(candle_core::DType::F32)? * scale)?;
        let k = k.to_dtype(candle_core::DType::F32)?;
        let v = v.to_dtype(candle_core::DType::F32)?;

        // Initialise the recurrent state from cache or zeros.
        let mut state = match self.state.recurrent_state.take() {
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

        // Per-token delta-rule loop. Slow-but-correct path; chunked
        // optimisation is for later.
        let mut outputs: Vec<Tensor> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            // (B, H, D_k) and (B, H, D_v) for token t.
            let q_t = q.i((.., .., t, ..))?; // (B, H, D_k)
            let k_t = k.i((.., .., t, ..))?;
            let v_t = v.i((.., .., t, ..))?;
            let g_t = g.i((.., .., t))?; // (B, H)
            let beta_t = beta.i((.., .., t))?; // (B, H)

            // Decay: state *= exp(g_t).  exp(g_t) shape (B, H) → broadcast to (B, H, 1, 1).
            let decay = g_t
                .exp()?
                .unsqueeze(candle_core::D::Minus1)?
                .unsqueeze(candle_core::D::Minus1)?; // (B, H, 1, 1)
            state = state.broadcast_mul(&decay)?;

            // Memory readout: sum_{d_k} state[d_k, d_v] * k_t[d_k]  → (B, H, D_v).
            // state: (B, H, D_k, D_v); k_t.unsqueeze(-1): (B, H, D_k, 1).
            let k_col = k_t.unsqueeze(candle_core::D::Minus1)?; // (B, H, D_k, 1)
            let kv_mem = state.broadcast_mul(&k_col)?.sum(2)?; // sum over D_k → (B, H, D_v)

            // delta = (v_t - kv_mem) * beta_t  (broadcast beta on last dim).
            let beta_col = beta_t.unsqueeze(candle_core::D::Minus1)?; // (B, H, 1)
            let delta = (v_t - kv_mem)?.broadcast_mul(&beta_col)?; // (B, H, D_v)

            // state += outer(k_t, delta) = k_col * delta_row, broadcast to (B, H, D_k, D_v).
            let delta_row = delta.unsqueeze(2)?; // (B, H, 1, D_v)
            let outer = k_col.broadcast_mul(&delta_row)?; // (B, H, D_k, D_v)
            state = (state + outer)?;

            // out_t = sum_{d_k} state[d_k, d_v] * q_t[d_k]   → (B, H, D_v).
            let q_col = q_t.unsqueeze(candle_core::D::Minus1)?; // (B, H, D_k, 1)
            let out_t = state.broadcast_mul(&q_col)?.sum(2)?; // (B, H, D_v)
            outputs.push(out_t.unsqueeze(2)?); // (B, H, 1, D_v)
        }
        // Stash the updated recurrent state for the next call.
        self.state.recurrent_state = Some(state.to_dtype(dtype)?);

        // core_attn_out: (B, H, L, D_v) → (B, L, H, D_v) → (B*L*H, D_v).
        let core_attn_out = Tensor::cat(&outputs, 2)?; // (B, H, L, D_v)
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

/// Numerically-stable `softplus(x) = ln(1 + exp(x))`. Matches PyTorch's
/// `F.softplus` default (beta=1, threshold=20: for large positive x,
/// returns x as-is to avoid overflow in the exp).
fn softplus(x: &Tensor) -> candle_core::Result<Tensor> {
    let threshold = 20.0_f64;
    let big = x.ge(threshold)?; // Tensor<u8> mask
    let safe = x.minimum(&x.affine(0.0, 0.0)?.affine(0.0, threshold)?)?; // min(x, threshold)
    let small = ((safe.exp()? + 1.0_f64)?).log()?;
    // Select x where big, else small.
    big.where_cond(x, &small)
}

/// `repeat_interleave` along a single dim. Candle has no built-in for
/// this; emulate with unsqueeze + expand + reshape.
fn repeat_interleave(x: &Tensor, repeats: usize, dim: usize) -> candle_core::Result<Tensor> {
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
}
