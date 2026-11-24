//! A linear-attention (gated delta-net) layer's mixer half, native.
//!
//! One decode token: the attention hyper-connection reads the four
//! residual streams into the mixer's input, the gated delta-net runs on
//! it, and the hyper-connection writes the result back. The MoE half of
//! the layer is separate. Ten kernel launches, against ~180 candle ops:
//!
//! 1. `gr_read`: stream norm, down/up bottleneck, gated mean, injection.
//! 2. quantize the mixer input to q8_1 for the Q8_0 projections.
//! 3. `qkv = attn_qkv @ x` (mmvq).
//! 4. causal conv + SiLU over all channels, L2 norm of the q and k heads.
//! 5. alpha/beta projections (BF16) with their softplus/sigmoid gates.
//! 6. `z = attn_gate @ x` (mmvq).
//! 7. the delta-rule step and the gated RMS norm of its readout.
//! 8. quantize that readout; 9. `out = ssm_out @ y` (mmvq).
//! 10. `gr_write`: the output added back to every stream, gated.
//!
//! Everything is in the GGUF's own layout (value heads tiled by key
//! head) and the state in Strata's: recurrent `[row][head][col]` f32,
//! conv history `[channel][3]` f32.

use std::ffi::{CStr, c_void};

use anyhow::{Result, bail, ensure};
use candle_core::CudaDevice;
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr};

use super::weights::{DeviceTensor, Gguf};
use crate::cuda::strata as ffi;

const Q8_1_BLOCK_VALUES: usize = 32;
const Q8_1_BLOCK_BYTES: usize = 36;
const STATE_SIZE: usize = 128;
const CONV_HISTORY: usize = 3;

/// Turns a kernel status into an error naming the step.
pub(crate) fn check(status: i32, what: &str) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    // SAFETY: returns a NUL-terminated string owned by the shim.
    let message = unsafe { CStr::from_ptr(ffi::neuron_engine_last_error()) }.to_string_lossy();
    bail!("{what} failed (status {status}): {message}")
}

/// Selects the kernel contracts Strata's decode uses. Idempotent.
pub fn configure() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: sets process-wide kernel selections before any launch.
    ONCE.call_once(|| unsafe { ffi::neuron_engine_configure() });
}

fn address<T>(slice: &CudaSlice<T>) -> u64 {
    slice.device_ptr(slice.stream()).0
}

/// Shapes of the hyper-connection and the gated delta-net.
#[derive(Debug, Clone, Copy)]
pub struct GdnShapes {
    pub n_embd: usize,
    pub hc: usize,
    pub hc_lr: usize,
    pub k_heads: usize,
    pub v_heads: usize,
    pub eps: f32,
}

impl GdnShapes {
    fn qk(&self) -> usize {
        self.k_heads * STATE_SIZE
    }
    fn value(&self) -> usize {
        self.v_heads * STATE_SIZE
    }
    fn channels(&self) -> usize {
        2 * self.qk() + self.value()
    }
}

/// One layer's weights for the mixer half.
pub struct GdnWeights {
    hc_norm: DeviceTensor,
    hc_down: DeviceTensor,
    hc_up: DeviceTensor,
    hc_inject: DeviceTensor,
    qkv: DeviceTensor,
    gate: DeviceTensor,
    out: DeviceTensor,
    alpha: DeviceTensor,
    beta: DeviceTensor,
    conv: DeviceTensor,
    norm: DeviceTensor,
    dt: DeviceTensor,
    ssm_a: DeviceTensor,
}

fn row_bytes(ggml_type: u32, n: u64) -> usize {
    // SAFETY: a pure function of its arguments.
    unsafe { ffi::neuron_strata_row_bytes(ggml_type as i32, n as i64) }
}

impl GdnWeights {
    pub fn load(gguf: &Gguf, layer: usize, s: &GdnShapes) -> Result<Self> {
        let name = |t: &str| format!("blk.{layer}.{t}");
        let projection = |t: &str, n_in: usize, n_out: usize| -> Result<DeviceTensor> {
            let w = gguf.upload_raw(&name(t), row_bytes)?;
            ensure!(
                w.dims == [n_in as u64, n_out as u64],
                "{} is {:?}, expected [{n_in}, {n_out}]",
                name(t),
                w.dims
            );
            // SAFETY: a pure function of its argument.
            ensure!(
                unsafe { ffi::neuron_strata_mmvq_supported(w.ggml_type as i32) } != 0,
                "{}: no mmvq for type {}",
                name(t),
                w.ggml_type
            );
            Ok(w)
        };
        let wide = s.hc * s.n_embd;
        let weights = Self {
            hc_norm: gguf.upload_f32(&name("hc_attn_norm.weight"))?,
            hc_down: gguf.upload_bf16(&name("hc_attn_down.weight"))?,
            hc_up: gguf.upload_bf16(&name("hc_attn_up.weight"))?,
            hc_inject: gguf.upload_bf16(&name("hc_attn_inject.weight"))?,
            qkv: projection("attn_qkv.weight", s.n_embd, s.channels())?,
            gate: projection("attn_gate.weight", s.n_embd, s.value())?,
            out: projection("ssm_out.weight", s.value(), s.n_embd)?,
            alpha: gguf.upload_bf16(&name("ssm_alpha.weight"))?,
            beta: gguf.upload_bf16(&name("ssm_beta.weight"))?,
            conv: gguf.upload_f32(&name("ssm_conv1d.weight"))?,
            norm: gguf.upload_f32(&name("ssm_norm.weight"))?,
            dt: gguf.upload_f32(&name("ssm_dt.bias"))?,
            ssm_a: gguf.upload_f32(&name("ssm_a"))?,
        };
        for (t, want) in [
            (&weights.hc_norm, wide),
            (&weights.hc_down, s.hc_lr * wide),
            (&weights.hc_up, wide * s.hc_lr),
            (&weights.hc_inject, s.hc * wide),
            (&weights.alpha, s.v_heads * s.n_embd),
            (&weights.beta, s.v_heads * s.n_embd),
            (&weights.conv, s.channels() * (CONV_HISTORY + 1)),
            (&weights.norm, STATE_SIZE),
            (&weights.dt, s.v_heads),
            (&weights.ssm_a, s.v_heads),
        ] {
            ensure!(
                t.values() == want,
                "layer {layer}: a tensor has {} values, expected {want}",
                t.values()
            );
        }
        Ok(weights)
    }
}

/// A sequence's recurrent and conv state for one layer.
pub struct GdnState {
    pub recurrent: CudaSlice<f32>,
    pub conv: CudaSlice<f32>,
}

impl GdnState {
    /// Zeroes the state in place: a captured step keeps its addresses.
    pub fn clear(&mut self, device: &CudaDevice) -> Result<()> {
        let stream = device.cuda_stream();
        stream.memset_zeros(&mut self.recurrent)?;
        stream.memset_zeros(&mut self.conv)?;
        Ok(())
    }

    pub fn zeroed(device: &CudaDevice, s: &GdnShapes) -> Result<Self> {
        Ok(Self {
            recurrent: device.alloc_zeros::<f32>(STATE_SIZE * s.v_heads * STATE_SIZE)?,
            conv: device.alloc_zeros::<f32>(s.channels() * CONV_HISTORY)?,
        })
    }
}

/// Intermediate buffers, shared by every layer of a step.
pub struct GdnScratch {
    gr: CudaSlice<u8>,
    mixed: CudaSlice<f32>,
    inject: CudaSlice<f32>,
    x_q8: CudaSlice<u8>,
    qkv: CudaSlice<f32>,
    h: CudaSlice<f32>,
    z: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    beta: CudaSlice<f32>,
    y: CudaSlice<f32>,
    y_q8: CudaSlice<u8>,
    out: CudaSlice<f32>,
}

impl GdnScratch {
    pub fn new(device: &CudaDevice, s: &GdnShapes) -> Result<Self> {
        // SAFETY: a pure function of its arguments.
        let gr_bytes =
            unsafe { ffi::neuron_gr_workspace_bytes(s.n_embd as i64, s.hc as i64, s.hc_lr as i64) };
        let q8 = |n: usize| n / Q8_1_BLOCK_VALUES * Q8_1_BLOCK_BYTES;
        Ok(Self {
            gr: device.alloc_zeros::<u8>(gr_bytes)?,
            mixed: device.alloc_zeros::<f32>(s.n_embd)?,
            inject: device.alloc_zeros::<f32>(s.hc)?,
            x_q8: device.alloc_zeros::<u8>(q8(s.n_embd))?,
            qkv: device.alloc_zeros::<f32>(s.channels())?,
            h: device.alloc_zeros::<f32>(s.channels())?,
            z: device.alloc_zeros::<f32>(s.value())?,
            gate: device.alloc_zeros::<f32>(s.v_heads)?,
            beta: device.alloc_zeros::<f32>(s.v_heads)?,
            y: device.alloc_zeros::<f32>(s.value())?,
            y_q8: device.alloc_zeros::<u8>(q8(s.value()))?,
            out: device.alloc_zeros::<f32>(s.n_embd)?,
        })
    }
}

/// `y = W x` for a quantized `W` and an f32 `x` already in q8_1.
fn mmvq(w: &DeviceTensor, x_q8: u64, y: u64, stream: *mut c_void) {
    // SAFETY: `w` is `dims[1]` rows of `dims[0]`, `x_q8` holds `dims[0]`
    // values in q8_1 and `y` has room for `dims[1]` floats.
    unsafe {
        ffi::neuron_strata_mmvq(
            w.ggml_type as i32,
            w.address() as *const c_void,
            x_q8 as *const c_void,
            y as *mut f32,
            w.dims[0] as i32,
            w.dims[1] as i32,
            1,
            stream,
        )
    }
}

fn quantize(x: u64, n: usize, q8: u64, stream: *mut c_void) {
    // SAFETY: `x` holds `n` floats and `q8` room for their q8_1 blocks.
    unsafe {
        ffi::neuron_strata_quantize_q8_1(x as *const f32, 1, n as i64, q8 as *mut c_void, stream)
    }
}

/// The mixer half of one linear-attention layer, for one token:
/// `residual_out = residual + gated(gdn(gr_read(residual)))`.
///
/// `residual` and `residual_out` are `hc * n_embd` f32 device buffers and
/// may not alias. Nothing allocates or waits, so this can be captured.
pub fn forward(
    w: &GdnWeights,
    state: &mut GdnState,
    sc: &mut GdnScratch,
    s: &GdnShapes,
    residual: u64,
    residual_out: u64,
    stream: *mut c_void,
) -> Result<()> {
    let (n_embd, hc, hc_lr) = (s.n_embd as i64, s.hc as i64, s.hc_lr as i64);
    let qk = s.qk() as u64 * 4;
    // SAFETY (all calls below): every pointer is a live device buffer of
    // the size its kernel reads or writes, as allocated in `GdnScratch`,
    // `GdnState` and `GdnWeights`, and every launch is on `stream`.
    unsafe {
        check(
            ffi::neuron_gr_read(
                residual as *const f32,
                w.hc_norm.address() as *const f32,
                w.hc_down.address() as *const u16,
                w.hc_up.address() as *const u16,
                w.hc_inject.address() as *const u16,
                s.eps,
                n_embd,
                hc,
                hc_lr,
                address(&sc.gr) as *mut c_void,
                address(&sc.mixed) as *mut f32,
                address(&sc.inject) as *mut f32,
                stream,
            ),
            "gr_read",
        )?;
        quantize(address(&sc.mixed), s.n_embd, address(&sc.x_q8), stream);
        mmvq(&w.qkv, address(&sc.x_q8), address(&sc.qkv), stream);
        check(
            ffi::neuron_gdn_conv_l2(
                address(&state.conv) as *mut f32,
                address(&sc.qkv) as *const f32,
                w.conv.address() as *const f32,
                address(&sc.h) as *mut f32,
                s.channels() as i32,
                (2 * s.k_heads) as i32,
                s.eps,
                stream,
            ),
            "gdn conv + l2",
        )?;
        check(
            ffi::neuron_gdn_ab(
                address(&sc.mixed) as *const f32,
                w.alpha.address() as *const u16,
                w.beta.address() as *const u16,
                w.dt.address() as *const f32,
                w.ssm_a.address() as *const f32,
                address(&sc.gate) as *mut f32,
                address(&sc.beta) as *mut f32,
                s.n_embd as i32,
                s.v_heads as i32,
                stream,
            ),
            "gdn alpha/beta",
        )?;
        mmvq(&w.gate, address(&sc.x_q8), address(&sc.z), stream);
        let h = address(&sc.h);
        check(
            ffi::neuron_gdn_step_norm(
                address(&state.recurrent) as *mut f32,
                h as *const f32,
                (h + qk) as *const f32,
                (h + 2 * qk) as *const f32,
                address(&sc.gate) as *const f32,
                address(&sc.beta) as *const f32,
                address(&sc.z) as *const f32,
                w.norm.address() as *const f32,
                s.eps,
                address(&sc.y) as *mut f32,
                s.k_heads as i32,
                s.v_heads as i32,
                stream,
            ),
            "gdn step + norm",
        )?;
        quantize(address(&sc.y), s.value(), address(&sc.y_q8), stream);
        mmvq(&w.out, address(&sc.y_q8), address(&sc.out), stream);
        check(
            ffi::neuron_gr_write(
                residual as *const f32,
                address(&sc.out) as *const f32,
                address(&sc.inject) as *const f32,
                n_embd,
                hc,
                hc_lr,
                residual_out as *mut f32,
                stream,
            ),
            "gr_write",
        )?;
    }
    Ok(())
}

/// The shapes of a qwen4_exp config.
pub fn shapes(cfg: &super::super::config::TextConfig) -> Result<GdnShapes> {
    let s = GdnShapes {
        n_embd: cfg.hidden_size,
        hc: cfg.hc_count,
        hc_lr: cfg.hc_lowrank,
        k_heads: cfg.linear_num_key_heads,
        v_heads: cfg.linear_num_value_heads,
        eps: cfg.rms_norm_eps as f32,
    };
    ensure!(
        cfg.linear_key_head_dim == STATE_SIZE && cfg.linear_value_head_dim == STATE_SIZE,
        "the native gated delta-net is built for 128-wide heads"
    );
    ensure!(
        s.v_heads.is_multiple_of(s.k_heads),
        "value heads must be a multiple of key heads"
    );
    Ok(s)
}
