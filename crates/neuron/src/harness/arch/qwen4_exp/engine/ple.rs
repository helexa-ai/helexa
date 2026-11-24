//! The PLE block on the device (#427), for one decode token.
//!
//! The n-gram rows are hashed and read on the host (they live in a 27 GB
//! table), into pinned memory the step copies from, so the block itself
//! is fixed work over fixed addresses and joins the step's graph:
//!
//! ```text
//! rows (pinned) ─ copy ─ emb ─ q8_1 ─┬─ ple_key   (Q8_0) ─ key   [hc * n_embd]
//!                                    └─ ple_value (Q8_0) ─ value [n_embd]
//! residual += gated + silu(conv(normalized))   Strata's native_ple_postops
//! history  <- history[1..] ++ normalized       9 rows per channel
//! ```

use std::ffi::c_void;

use anyhow::{Result, ensure};
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr};
use candle_core::{CudaDevice, DType, Tensor};

use super::attn::{mmvq, quantize};
use super::gdn::{GdnShapes, check};
use super::weights::{DeviceTensor, GGML_Q8_0, Gguf};
use crate::cuda::strata as ffi;

/// Rows of normalized history the dilated conv reaches back over.
pub const HISTORY: usize = 9;

const Q8_1_BLOCK_VALUES: usize = 32;
const Q8_1_BLOCK_BYTES: usize = 36;

fn address<T>(slice: &CudaSlice<T>) -> u64 {
    slice.device_ptr(slice.stream()).0
}

/// Pinned host memory the n-gram rows are written to.
struct Pinned(*mut f32);

// SAFETY: plain host memory, written by the device worker between steps.
unsafe impl Send for Pinned {}

impl Drop for Pinned {
    fn drop(&mut self) {
        // SAFETY: allocated by `neuron_strata_pinned_alloc`, freed once.
        unsafe { ffi::neuron_strata_pinned_free(self.0.cast()) };
    }
}

pub struct PleNative {
    key_w: DeviceTensor,
    value_w: DeviceTensor,
    norm_key: DeviceTensor,
    norm_query: DeviceTensor,
    norm_conv: DeviceTensor,
    conv_w: DeviceTensor,
    /// `[channel][row]`, the oldest row first.
    history: CudaSlice<f32>,
    emb: CudaSlice<f32>,
    emb_q8: CudaSlice<u8>,
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
    key_norm: CudaSlice<f32>,
    normalized: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    gated: CudaSlice<f32>,
    conv: CudaSlice<f32>,
    host: Pinned,
    n_embd: usize,
    wide: usize,
}

impl PleNative {
    /// The block of GGUF layer `layer` (`blk.{layer}.ple_*`).
    pub fn load(gguf: &Gguf, layer: usize, s: &GdnShapes) -> Result<Self> {
        let device = gguf.device().clone();
        let (n_embd, wide) = (s.n_embd, s.n_embd * s.hc);
        let name = |t: &str| format!("blk.{layer}.ple_{t}.weight");
        let row_bytes = |ty: u32, n: u64| -> usize {
            // SAFETY: a pure function of its arguments.
            unsafe { ffi::neuron_strata_row_bytes(ty as i32, n as i64) }
        };
        let key_w = gguf.upload_raw(&name("key"), row_bytes)?;
        let value_w = gguf.upload_raw(&name("value"), row_bytes)?;
        for (w, out) in [(&key_w, wide), (&value_w, n_embd)] {
            ensure!(
                w.ggml_type == GGML_Q8_0 && w.dims == [n_embd as u64, out as u64],
                "PLE projection is type {} {:?}, expected Q8_0 [{n_embd}, {out}]",
                w.ggml_type,
                w.dims
            );
        }
        // The GGUF stores the three norms already as 1 + w, which is the
        // gamma the kernel multiplies by.
        let norm_key = gguf.upload_f32(&name("norm_key"))?;
        let norm_query = gguf.upload_f32(&name("norm_query"))?;
        let norm_conv = gguf.upload_f32(&name("norm_conv"))?;
        // ggml's [4, channels]: tap k of channel c at k + 4c, as the
        // kernel reads it.
        let conv_w = gguf.upload_f16(&name("conv1d"))?;
        ensure!(
            conv_w.dims == [4, wide as u64],
            "ple_conv1d is {:?}",
            conv_w.dims
        );
        // SAFETY: returns null or `n_embd` f32 of pinned host memory.
        let host = unsafe { ffi::neuron_strata_pinned_alloc(n_embd * 4) }.cast::<f32>();
        ensure!(!host.is_null(), "cannot pin the PLE input");
        Ok(Self {
            key_w,
            value_w,
            norm_key,
            norm_query,
            norm_conv,
            conv_w,
            history: device.alloc_zeros::<f32>(HISTORY * wide)?,
            emb: device.alloc_zeros::<f32>(n_embd)?,
            emb_q8: device.alloc_zeros::<u8>(n_embd / Q8_1_BLOCK_VALUES * Q8_1_BLOCK_BYTES)?,
            key: device.alloc_zeros::<f32>(wide)?,
            value: device.alloc_zeros::<f32>(n_embd)?,
            key_norm: device.alloc_zeros::<f32>(wide)?,
            normalized: device.alloc_zeros::<f32>(wide)?,
            gate: device.alloc_zeros::<f32>(s.hc)?,
            gated: device.alloc_zeros::<f32>(wide)?,
            conv: device.alloc_zeros::<f32>(wide)?,
            host: Pinned(host),
            n_embd,
            wide,
        })
    }

    /// Where the step's n-gram rows go (`n_embd` f32, head-major). Write
    /// only between steps: the step's first copy reads it.
    pub fn input(&mut self) -> &mut [f32] {
        // SAFETY: the buffer holds `n_embd` f32 and lives as long as
        // `self`; `&mut self` keeps any other view out.
        unsafe { std::slice::from_raw_parts_mut(self.host.0, self.n_embd) }
    }

    /// The block for one token, in place on `residual` (`hc * n_embd`).
    pub fn forward(&self, residual: u64, stream: *mut c_void) -> Result<()> {
        // SAFETY: every buffer is sized in `load` for one token; the
        // pinned source outlives the copy.
        unsafe {
            check(
                ffi::neuron_strata_copy_to_device(
                    address(&self.emb) as *mut c_void,
                    self.host.0.cast(),
                    self.n_embd * 4,
                    stream,
                ),
                "PLE input copy",
            )?;
        }
        quantize(
            address(&self.emb),
            self.n_embd,
            address(&self.emb_q8),
            stream,
        );
        mmvq(
            &self.key_w,
            address(&self.emb_q8),
            address(&self.key),
            stream,
        );
        mmvq(
            &self.value_w,
            address(&self.emb_q8),
            address(&self.value),
            stream,
        );
        // SAFETY: as above; `result` is `hidden`, which the kernel allows.
        unsafe {
            check(
                ffi::neuron_ple_postops(
                    address(&self.key) as *const f32,
                    residual as *const f32,
                    address(&self.value) as *const f32,
                    address(&self.history) as *const f32,
                    self.norm_key.address() as *const f32,
                    self.norm_query.address() as *const f32,
                    self.norm_conv.address() as *const f32,
                    self.conv_w.address() as *const u16,
                    address(&self.key_norm) as *mut f32,
                    address(&self.normalized) as *mut f32,
                    address(&self.gate) as *mut f32,
                    address(&self.gated) as *mut f32,
                    address(&self.conv) as *mut f32,
                    residual as *mut f32,
                    stream,
                ),
                "PLE",
            )?;
            check(
                ffi::neuron_ple_history_advance(
                    address(&self.history) as *mut f32,
                    address(&self.normalized) as *const f32,
                    self.wide as i32,
                    stream,
                ),
                "PLE history",
            )
        }
    }

    /// Takes candle's rolling conv context: `(1, channels, 9)`, the same
    /// row-fastest layout, or `None` for a sequence that has none yet.
    pub fn import(&mut self, device: &CudaDevice, state: Option<&Tensor>) -> Result<()> {
        match state {
            None => self.clear(device),
            Some(t) => {
                ensure!(
                    t.dims() == [1, self.wide, HISTORY],
                    "PLE conv state is {:?}",
                    t.dims()
                );
                let t = t.to_dtype(DType::F32)?.contiguous()?;
                super::decoder::copy_into(device, &t, &mut self.history)
            }
        }
    }

    /// The conv context in candle's form, `(1, channels, 9)` in `dtype`.
    pub fn export(&self, device: &CudaDevice, dtype: DType) -> Result<Tensor> {
        Ok(
            super::decoder::tensor_of(device, &self.history, (1, self.wide, HISTORY))?
                .to_dtype(dtype)?,
        )
    }

    pub fn clear(&mut self, device: &CudaDevice) -> Result<()> {
        device.cuda_stream().memset_zeros(&mut self.history)?;
        Ok(())
    }
}
