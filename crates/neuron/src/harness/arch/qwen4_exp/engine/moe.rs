//! A decoder layer's MoE half, native.
//!
//! One decode token: the FFN hyper-connection reads the streams into the
//! block's input; the router picks ten experts on the device; the routed
//! experts run on the grouped kernel through the device-routed expert
//! path (#425), misses gathered into the arena; the shared expert runs
//! beside them; the weighted combination is written back to every
//! stream through the hyper-connection. Every layer, linear or full
//! attention, has this half.

use std::ffi::c_void;

use anyhow::{Result, ensure};
use candle_core::CudaDevice;
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr};

use super::gdn::{GdnShapes, check};
use super::weights::{DeviceTensor, Gguf};
use crate::cuda::strata as ffi;
use crate::harness::arch::qwen4_exp::native_experts::GgufExpertSource;

const Q8_1_BLOCK_VALUES: usize = 32;
const Q8_1_BLOCK_BYTES: usize = 36;

fn address<T>(slice: &CudaSlice<T>) -> u64 {
    slice.device_ptr(slice.stream()).0
}

/// Routing and expert widths.
#[derive(Debug, Clone, Copy)]
pub struct MoeShapes {
    pub experts: usize,
    pub k: usize,
    pub n_ff: usize,
}

pub struct MoeWeights {
    hc_norm: DeviceTensor,
    hc_down: DeviceTensor,
    hc_up: DeviceTensor,
    hc_inject: DeviceTensor,
    router: DeviceTensor,
    shared_gate: DeviceTensor,
    shared_up: DeviceTensor,
    shared_down: DeviceTensor,
    shared_scale: DeviceTensor,
}

fn row_bytes(ggml_type: u32, n: u64) -> usize {
    // SAFETY: a pure function of its arguments.
    unsafe { ffi::neuron_strata_row_bytes(ggml_type as i32, n as i64) }
}

impl MoeWeights {
    pub fn load(gguf: &Gguf, layer: usize, s: &GdnShapes, m: &MoeShapes) -> Result<Self> {
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
        let w = Self {
            hc_norm: gguf.upload_f32(&name("hc_ffn_norm.weight"))?,
            hc_down: gguf.upload_bf16(&name("hc_ffn_down.weight"))?,
            hc_up: gguf.upload_bf16(&name("hc_ffn_up.weight"))?,
            hc_inject: gguf.upload_bf16(&name("hc_ffn_inject.weight"))?,
            router: gguf.upload_bf16(&name("ffn_gate_inp.weight"))?,
            shared_gate: projection("ffn_gate_shexp.weight", s.n_embd, m.n_ff)?,
            shared_up: projection("ffn_up_shexp.weight", s.n_embd, m.n_ff)?,
            shared_down: projection("ffn_down_shexp.weight", m.n_ff, s.n_embd)?,
            shared_scale: gguf.upload_bf16(&name("ffn_gate_inp_shexp.weight"))?,
        };
        for (t, want) in [
            (&w.hc_norm, wide),
            (&w.hc_down, s.hc_lr * wide),
            (&w.hc_up, wide * s.hc_lr),
            (&w.hc_inject, s.hc * wide),
            (&w.router, m.experts * s.n_embd),
            (&w.shared_scale, s.n_embd),
        ] {
            ensure!(
                t.values() == want,
                "layer {layer}: a tensor has {} values, expected {want}",
                t.values()
            );
        }
        Ok(w)
    }
}

pub struct MoeScratch {
    gr: CudaSlice<u8>,
    mixed: CudaSlice<f32>,
    inject: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    ids: CudaSlice<i32>,
    weights: CudaSlice<f32>,
    parts: CudaSlice<f32>,
    w_route: CudaSlice<f32>,
    x_q8: CudaSlice<u8>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    h: CudaSlice<f32>,
    h_q8: CudaSlice<u8>,
    shared: CudaSlice<f32>,
    scale: CudaSlice<f32>,
    out: CudaSlice<f32>,
}

impl MoeScratch {
    /// The experts the last step routed to, for inspection.
    pub fn routed(&self, device: &CudaDevice) -> Result<(Vec<i32>, Vec<f32>)> {
        Ok((
            device.clone_dtoh(&self.ids)?,
            device.clone_dtoh(&self.logits)?,
        ))
    }

    pub fn new(device: &CudaDevice, s: &GdnShapes, m: &MoeShapes) -> Result<Self> {
        // SAFETY: a pure function of its arguments.
        let gr_bytes =
            unsafe { ffi::neuron_gr_workspace_bytes(s.n_embd as i64, s.hc as i64, s.hc_lr as i64) };
        let q8 = |n: usize| n / Q8_1_BLOCK_VALUES * Q8_1_BLOCK_BYTES;
        Ok(Self {
            gr: device.alloc_zeros::<u8>(gr_bytes)?,
            mixed: device.alloc_zeros::<f32>(s.n_embd)?,
            inject: device.alloc_zeros::<f32>(s.hc)?,
            logits: device.alloc_zeros::<f32>(m.experts)?,
            ids: device.alloc_zeros::<i32>(m.k)?,
            weights: device.alloc_zeros::<f32>(m.k)?,
            parts: device.alloc_zeros::<f32>(m.k * s.n_embd)?,
            w_route: device.alloc_zeros::<f32>(m.k)?,
            x_q8: device.alloc_zeros::<u8>(q8(s.n_embd))?,
            gate: device.alloc_zeros::<f32>(m.n_ff)?,
            up: device.alloc_zeros::<f32>(m.n_ff)?,
            h: device.alloc_zeros::<f32>(m.n_ff)?,
            h_q8: device.alloc_zeros::<u8>(q8(m.n_ff))?,
            shared: device.alloc_zeros::<f32>(s.n_embd)?,
            scale: device.alloc_zeros::<f32>(1)?,
            out: device.alloc_zeros::<f32>(s.n_embd)?,
        })
    }
}

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

/// The MoE half of layer `layer`, for one token:
/// `residual_out = residual + gated(moe(gr_read(residual)))`.
///
/// The experts come from `experts`, which must route on the device. It
/// does not adapt the expert tier; the caller does that between steps.
#[allow(clippy::too_many_arguments)]
pub fn forward(
    w: &MoeWeights,
    experts: &GgufExpertSource,
    layer: usize,
    sc: &mut MoeScratch,
    s: &GdnShapes,
    m: &MoeShapes,
    residual: u64,
    residual_out: u64,
    stream: *mut c_void,
) -> Result<()> {
    let (n_embd, hc, hc_lr) = (s.n_embd as i64, s.hc as i64, s.hc_lr as i64);
    // SAFETY (all calls below): every pointer is a live device buffer of
    // the size its kernel reads or writes, as allocated in `MoeScratch`
    // and `MoeWeights`, and every launch is on `stream`.
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
            "ffn gr_read",
        )?;
        check(
            ffi::neuron_bf16_gemv(
                address(&sc.mixed) as *const f32,
                w.router.address() as *const u16,
                address(&sc.logits) as *mut f32,
                n_embd,
                m.experts as i64,
                stream,
            ),
            "router",
        )?;
        check(
            ffi::neuron_router_top10(
                address(&sc.logits) as *const f32,
                address(&sc.ids) as *mut i32,
                address(&sc.weights) as *mut f32,
                stream,
            ),
            "router top-10",
        )?;
    }
    experts.launch_routed(
        layer,
        address(&sc.mixed),
        address(&sc.ids),
        address(&sc.weights),
        1,
        m.k,
        address(&sc.parts),
        address(&sc.w_route),
        stream,
    )?;
    experts.record_usage(layer, address(&sc.ids), m.k)?;
    quantize(address(&sc.mixed), s.n_embd, address(&sc.x_q8), stream);
    mmvq(&w.shared_gate, address(&sc.x_q8), address(&sc.gate), stream);
    mmvq(&w.shared_up, address(&sc.x_q8), address(&sc.up), stream);
    // SAFETY: as above.
    unsafe {
        check(
            ffi::neuron_swiglu(
                address(&sc.gate) as *const f32,
                address(&sc.up) as *const f32,
                address(&sc.h) as *mut f32,
                m.n_ff as i32,
                stream,
            ),
            "shared swiglu",
        )?;
    }
    quantize(address(&sc.h), m.n_ff, address(&sc.h_q8), stream);
    mmvq(
        &w.shared_down,
        address(&sc.h_q8),
        address(&sc.shared),
        stream,
    );
    // SAFETY: as above.
    unsafe {
        check(
            ffi::neuron_bf16_gemv(
                address(&sc.mixed) as *const f32,
                w.shared_scale.address() as *const u16,
                address(&sc.scale) as *mut f32,
                n_embd,
                1,
                stream,
            ),
            "shared gate",
        )?;
        check(
            ffi::neuron_scale_by_sigmoid(
                address(&sc.shared) as *mut f32,
                address(&sc.scale) as *const f32,
                s.n_embd as i32,
                stream,
            ),
            "shared gate apply",
        )?;
        check(
            ffi::neuron_moe_combine(
                address(&sc.parts) as *const f32,
                address(&sc.w_route) as *const f32,
                address(&sc.shared) as *const f32,
                address(&sc.out) as *mut f32,
                n_embd,
                m.k as i64,
                stream,
            ),
            "moe combine",
        )?;
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
            "ffn gr_write",
        )?;
    }
    Ok(())
}

/// The MoE shapes of a qwen4_exp config.
pub fn shapes(cfg: &super::super::config::TextConfig) -> Result<MoeShapes> {
    let m = MoeShapes {
        experts: cfg.num_experts,
        k: cfg.num_experts_per_tok,
        n_ff: cfg.moe_intermediate_size,
    };
    ensure!(
        m.experts == 512 && m.k == 10,
        "the native router is built for 512 experts, top 10"
    );
    ensure!(
        cfg.shared_expert_intermediate_size == m.n_ff,
        "the shared expert is assumed as wide as a routed one"
    );
    Ok(m)
}
