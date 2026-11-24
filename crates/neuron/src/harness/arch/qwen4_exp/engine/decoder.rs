//! The native decode step (#427): every layer of the model for one token,
//! over buffers allocated once.
//!
//! A step is two segments around the PLE layer: `[embedding, layers
//! before PLE]`, then (after the caller adds PLE's contribution to the
//! residual) `[layers from PLE on, the final mixer, the LM head]`. Each
//! segment is fixed work over fixed addresses, with the token and
//! position read from a device buffer filled from pinned host memory at
//! the start of the step, so each can be captured as a CUDA graph and
//! replayed. PLE's n-gram gather stays on the host for now.

use std::ffi::c_void;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use candle_core::cuda_backend::cudarc::driver::{CudaGraph, CudaSlice, DevicePtr, sys};
use candle_core::{CudaDevice, Storage, Tensor};

use super::attn::{self, AttnCache, AttnScratch, AttnShapes, AttnWeights};
use super::gdn::{self, GdnScratch, GdnShapes, GdnState, GdnWeights, check};
use super::moe::{self, MoeScratch, MoeShapes, MoeWeights};
use super::weights::{DeviceTensor, Gguf};
use crate::cuda::strata as ffi;
use crate::harness::arch::qwen4_exp::config::TextConfig;
use crate::harness::arch::qwen4_exp::native_experts::GgufExpertSource;

/// Positions are written for every query-head row the rotary reads.
const POSITION_SLOTS: usize = 32;
/// The step buffer: `POSITION_SLOTS` positions, then the token id.
const STEP_SLOTS: usize = POSITION_SLOTS + 1;

fn address<T>(slice: &CudaSlice<T>) -> u64 {
    slice.device_ptr(slice.stream()).0
}

enum Mixer {
    Linear(GdnWeights, GdnState),
    Full(AttnWeights, AttnCache),
}

struct Layer {
    mixer: Mixer,
    moe: MoeWeights,
}

/// Pinned host memory the step inputs are copied from.
struct HostStep(*mut i32);

// SAFETY: plain host memory, written by the device worker between steps.
unsafe impl Send for HostStep {}

impl Drop for HostStep {
    fn drop(&mut self) {
        // SAFETY: allocated by `neuron_strata_pinned_alloc`, freed once.
        unsafe { ffi::neuron_strata_pinned_free(self.0.cast()) };
    }
}

/// A captured segment.
struct Graph(CudaGraph);

// SAFETY: a graph is launched only from the device worker that owns the
// decoder, on the stream it was captured on.
unsafe impl Send for Graph {}

pub struct NativeDecoder {
    /// Captured segments, once a warm step has run.
    head_graph: Option<Graph>,
    tail_graph: Option<Graph>,
    warmed: bool,
    device: CudaDevice,
    s: GdnShapes,
    a: AttnShapes,
    m: MoeShapes,
    layers: Vec<Layer>,
    ple_layer: usize,
    experts: Arc<GgufExpertSource>,
    gdn_sc: GdnScratch,
    attn_sc: AttnScratch,
    moe_sc: MoeScratch,
    embed: DeviceTensor,
    embed_row: CudaSlice<f32>,
    mixer_norm: DeviceTensor,
    mixer_down: DeviceTensor,
    mixer_up: DeviceTensor,
    mixer_ws: CudaSlice<u8>,
    mixed: CudaSlice<f32>,
    unused_inject: CudaSlice<f32>,
    head: DeviceTensor,
    logits: CudaSlice<f32>,
    /// The residual and its other half: every half-layer reads one and
    /// writes the other, so each layer starts and ends in `residual[0]`.
    residual: [CudaSlice<f32>; 2],
    step: CudaSlice<i32>,
    host_step: HostStep,
    vocab: usize,
}

fn row_bytes(ggml_type: u32, n: u64) -> usize {
    // SAFETY: a pure function of its arguments.
    unsafe { ffi::neuron_strata_row_bytes(ggml_type as i32, n as i64) }
}

impl NativeDecoder {
    /// Loads every layer from the GGUF. `experts` must route on the
    /// device; `max_len` sizes the attention caches.
    pub fn load(
        gguf: &Gguf,
        cfg: &TextConfig,
        experts: Arc<GgufExpertSource>,
        max_len: usize,
    ) -> Result<Self> {
        gdn::configure();
        let device = gguf.device().clone();
        let s = gdn::shapes(cfg)?;
        let a = attn::shapes(cfg, max_len)?;
        let m = moe::shapes(cfg)?;
        ensure!(
            a.heads <= POSITION_SLOTS,
            "more query heads than position slots"
        );
        let ple_layers = cfg.ple_layers();
        ensure!(
            ple_layers.len() == 1,
            "the engine expects exactly one PLE layer"
        );
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (i, kind) in cfg
            .layer_types
            .iter()
            .enumerate()
            .take(cfg.num_hidden_layers)
        {
            let mixer = match kind.as_str() {
                "linear_attention" => Mixer::Linear(
                    GdnWeights::load(gguf, i, &s)?,
                    GdnState::zeroed(&device, &s)?,
                ),
                "full_attention" => Mixer::Full(
                    AttnWeights::load(gguf, i, &s, &a)?,
                    AttnCache::new(&device, &a)?,
                ),
                other => bail!("layer {i}: unknown layer type {other}"),
            };
            layers.push(Layer {
                mixer,
                moe: MoeWeights::load(gguf, i, &s, &m)?,
            });
        }
        let embed = gguf.upload_raw("token_embd.weight", row_bytes)?;
        ensure!(
            embed.dims == [s.n_embd as u64, cfg.vocab_size as u64],
            "token_embd is {:?}",
            embed.dims
        );
        let head = gguf.upload_bf16("output.weight")?;
        ensure!(
            head.dims == [s.n_embd as u64, cfg.vocab_size as u64],
            "output is {:?}",
            head.dims
        );
        // SAFETY: a pure function of its arguments.
        let gr_bytes =
            unsafe { ffi::neuron_gr_workspace_bytes(s.n_embd as i64, s.hc as i64, s.hc_lr as i64) };
        // SAFETY: returns null or STEP_SLOTS i32 of pinned host memory.
        let host = unsafe { ffi::neuron_strata_pinned_alloc(STEP_SLOTS * 4) }.cast::<i32>();
        ensure!(!host.is_null(), "cannot pin the step buffer");
        let wide = s.hc * s.n_embd;
        Ok(Self {
            head_graph: None,
            tail_graph: None,
            warmed: false,
            gdn_sc: GdnScratch::new(&device, &s)?,
            attn_sc: AttnScratch::new(&device, &s, &a)?,
            moe_sc: MoeScratch::new(&device, &s, &m)?,
            embed,
            embed_row: device.alloc_zeros::<f32>(s.n_embd)?,
            mixer_norm: gguf.upload_f32("output_hc_norm.weight")?,
            mixer_down: gguf.upload_bf16("output_hc_down.weight")?,
            mixer_up: gguf.upload_bf16("output_hc_up.weight")?,
            mixer_ws: device.alloc_zeros::<u8>(gr_bytes)?,
            mixed: device.alloc_zeros::<f32>(s.n_embd)?,
            unused_inject: device.alloc_zeros::<f32>(s.hc)?,
            head,
            logits: device.alloc_zeros::<f32>(cfg.vocab_size)?,
            residual: [
                device.alloc_zeros::<f32>(wide)?,
                device.alloc_zeros::<f32>(wide)?,
            ],
            step: device.alloc_zeros::<i32>(STEP_SLOTS)?,
            host_step: HostStep(host),
            vocab: cfg.vocab_size,
            ple_layer: ple_layers[0],
            device,
            s,
            a,
            m,
            layers,
            experts,
        })
    }

    pub fn stream(&self) -> *mut c_void {
        self.device.cuda_stream().cu_stream() as *mut c_void
    }

    /// The residual between the segments, `hc * n_embd` f32.
    pub fn residual(&self) -> &CudaSlice<f32> {
        &self.residual[0]
    }

    pub fn shapes(&self) -> (GdnShapes, AttnShapes, MoeShapes) {
        (self.s, self.a, self.m)
    }

    /// Writes the step's inputs to the pinned buffer. Must come before
    /// `segment_head` and after the previous step's logits were read.
    pub fn set_step(&mut self, token: u32, pos: usize) -> Result<()> {
        ensure!(
            pos < self.a.max_len,
            "position {pos} is past the {}-position cache",
            self.a.max_len
        );
        // SAFETY: the buffer holds STEP_SLOTS i32, and no copy from it is
        // in flight (the previous step's logits were read).
        let host = unsafe { std::slice::from_raw_parts_mut(self.host_step.0, STEP_SLOTS) };
        host[..POSITION_SLOTS].fill(pos as i32);
        host[POSITION_SLOTS] = token as i32;
        Ok(())
    }

    /// The host-side tier work between steps: what the experts learned
    /// from the last step, and admissions.
    pub fn adapt(&self) -> Result<()> {
        for l in 0..self.layers.len() {
            self.experts.adapt(l)?;
        }
        Ok(())
    }

    /// After a replay: the usage copies it made are this step's.
    pub fn usage_replayed(&self) {
        for l in 0..self.layers.len() {
            self.experts.usage_replayed(l);
        }
    }

    fn run_layer(&mut self, i: usize) -> Result<()> {
        let stream = self.stream();
        let (r0, r1) = (address(&self.residual[0]), address(&self.residual[1]));
        let positions = address(&self.step);
        let layer = &mut self.layers[i];
        match &mut layer.mixer {
            Mixer::Linear(w, state) => {
                gdn::forward(w, state, &mut self.gdn_sc, &self.s, r0, r1, stream)?
            }
            Mixer::Full(w, cache) => attn::forward(
                w,
                cache,
                &mut self.attn_sc,
                &self.s,
                &self.a,
                positions,
                r0,
                r1,
                stream,
            )?,
        }
        moe::forward(
            &layer.moe,
            &self.experts,
            i,
            &mut self.moe_sc,
            &self.s,
            &self.m,
            r1,
            r0,
            stream,
        )
    }

    /// Segment one: the step inputs, the embedding into every stream,
    /// and the layers before the PLE layer. Leaves the residual for the
    /// caller to add PLE's contribution to.
    pub fn segment_head(&mut self) -> Result<()> {
        let stream = self.stream();
        // SAFETY: the pinned source holds STEP_SLOTS i32 and the device
        // buffer as many; both live as long as `self`.
        let status = unsafe {
            ffi::neuron_strata_copy_to_device(
                address(&self.step) as *mut c_void,
                self.host_step.0.cast(),
                STEP_SLOTS * 4,
                stream,
            )
        };
        ensure!(
            status == 0,
            "step input copy failed with CUDA status {status}"
        );
        let (n_embd, hc) = (self.s.n_embd, self.s.hc);
        let token = address(&self.step) + (POSITION_SLOTS * 4) as u64;
        // SAFETY: the table holds `vocab` rows; `token` is one device id;
        // the outputs are sized for one row and `hc` copies of it.
        unsafe {
            check(
                ffi::neuron_embed_rows(
                    self.embed.ggml_type as i32,
                    self.embed.address() as *const c_void,
                    row_bytes(self.embed.ggml_type, n_embd as u64),
                    token as *const i32,
                    1,
                    n_embd as i64,
                    address(&self.embed_row) as *mut f32,
                    stream,
                ),
                "embedding",
            )?;
            check(
                ffi::neuron_repeat_row(
                    address(&self.embed_row) as *const f32,
                    address(&self.residual[0]) as *mut f32,
                    n_embd as i32,
                    hc as i32,
                    stream,
                ),
                "embedding into the streams",
            )?;
        }
        for i in 0..self.ple_layer {
            self.run_layer(i).with_context(|| format!("layer {i}"))?;
        }
        Ok(())
    }

    /// Segment one, from its graph: captured on first use after a warm
    /// eager step (which sizes every buffer the segment touches), then
    /// replayed. The first call of all runs it eagerly.
    pub fn run_head(&mut self) -> Result<()> {
        self.run_captured(true)
    }

    /// Segment two, as [`Self::run_head`].
    pub fn run_tail(&mut self) -> Result<()> {
        self.run_captured(false)
    }

    fn run_captured(&mut self, head: bool) -> Result<()> {
        if !self.warmed {
            if head {
                self.segment_head()?;
            } else {
                self.segment_tail()?;
                self.warmed = true;
            }
            return Ok(());
        }
        let has = if head {
            self.head_graph.is_some()
        } else {
            self.tail_graph.is_some()
        };
        if !has {
            let stream = self.device.cuda_stream();
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let ran = if head {
                self.segment_head()
            } else {
                self.segment_tail()
            };
            let graph = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            ran.context("while capturing")?;
            let graph = graph?.context("capture produced no graph")?;
            if head {
                self.head_graph = Some(Graph(graph));
            } else {
                self.tail_graph = Some(Graph(graph));
            }
        }
        let graph = if head {
            &self.head_graph
        } else {
            &self.tail_graph
        };
        graph.as_ref().expect("captured above").0.launch()?;
        if !head {
            self.usage_replayed();
        }
        Ok(())
    }

    /// Adds `contribution` (`hc * n_embd` f32 on the device) to the
    /// residual: the PLE layer's addition before its attention half.
    pub fn add_to_residual(&mut self, contribution: u64) -> Result<()> {
        let n = (self.s.hc * self.s.n_embd) as i32;
        // SAFETY: both buffers hold `n` floats.
        unsafe {
            check(
                ffi::neuron_add_inplace(
                    address(&self.residual[0]) as *mut f32,
                    contribution as *const f32,
                    n,
                    self.stream(),
                ),
                "ple add",
            )
        }
    }

    /// Segment two: the layers from the PLE layer on, the final mixer and
    /// the LM head, into the logits buffer.
    pub fn segment_tail(&mut self) -> Result<()> {
        for i in self.ple_layer..self.layers.len() {
            self.run_layer(i).with_context(|| format!("layer {i}"))?;
        }
        let stream = self.stream();
        let s = self.s;
        // SAFETY: as in the layers; a null injection is the final
        // mixer's, which collapses the streams without a write-back.
        unsafe {
            check(
                ffi::neuron_gr_read(
                    address(&self.residual[0]) as *const f32,
                    self.mixer_norm.address() as *const f32,
                    self.mixer_down.address() as *const u16,
                    self.mixer_up.address() as *const u16,
                    std::ptr::null(),
                    s.eps,
                    s.n_embd as i64,
                    s.hc as i64,
                    s.hc_lr as i64,
                    address(&self.mixer_ws) as *mut c_void,
                    address(&self.mixed) as *mut f32,
                    address(&self.unused_inject) as *mut f32,
                    stream,
                ),
                "final mixer",
            )?;
            check(
                ffi::neuron_bf16_gemv(
                    address(&self.mixed) as *const f32,
                    self.head.address() as *const u16,
                    address(&self.logits) as *mut f32,
                    s.n_embd as i64,
                    self.vocab as i64,
                    stream,
                ),
                "lm head",
            )?;
        }
        Ok(())
    }

    /// The final mixer's output of the last step, `n_embd` f32: what the
    /// LM head reads.
    pub fn mixed(&self) -> &CudaSlice<f32> {
        &self.mixed
    }

    /// The logits of the last step (waits for it).
    pub fn logits(&self) -> Result<Vec<f32>> {
        Ok(self.device.clone_dtoh(&self.logits)?)
    }

    /// How many layers there are, and whether layer `i` is linear.
    pub fn layer_is_linear(&self, i: usize) -> Option<bool> {
        self.layers
            .get(i)
            .map(|l| matches!(l.mixer, Mixer::Linear(..)))
    }

    /// Takes layer `i`'s linear-attention state from candle after a
    /// prefill (#427): `conv` is candle's `(1, channels, 4)` window and
    /// `recurrent` its `(1, v_heads, 128, 128)` state, both in the
    /// checkpoint's head order. The engine keeps the GGUF's order (value
    /// heads tiled by key head) and Strata's layouts: a `[channel][3]`
    /// history of the last three inputs, and `[row][head][col]` state.
    pub fn import_linear(&mut self, i: usize, conv: &Tensor, recurrent: &Tensor) -> Result<()> {
        let s = self.s;
        let (hk, hv) = (s.k_heads, s.v_heads);
        let r = hv / hk;
        // GGUF value head p = g * hk + k holds checkpoint head r * k + g.
        let head_src: Vec<u32> = (0..hv).map(|p| ((p % hk) * r + p / hk) as u32).collect();
        let qk = 2 * hk * 128;
        let mut channel_src: Vec<u32> = (0..qk as u32).collect();
        for &h in &head_src {
            channel_src.extend((0..128u32).map(|d| qk as u32 + h * 128 + d));
        }
        let device = candle_core::Device::Cuda(self.device.clone());
        let channels = Tensor::new(channel_src.as_slice(), &device)?;
        let heads = Tensor::new(head_src.as_slice(), &device)?;
        let width = conv.dim(2)?;
        ensure!(width >= 3, "conv state is {width} wide");
        let conv = conv
            .squeeze(0)?
            .narrow(1, width - 3, 3)?
            .contiguous()?
            .index_select(&channels, 0)?
            .to_dtype(candle_core::DType::F32)?
            .contiguous()?;
        let rec = recurrent
            .squeeze(0)?
            .to_dtype(candle_core::DType::F32)?
            .contiguous()?
            .index_select(&heads, 0)?
            .permute((1, 0, 2))?
            .contiguous()?;
        let Mixer::Linear(_, state) = &mut self.layers[i].mixer else {
            bail!("layer {i} is not linear attention");
        };
        copy_into(&self.device, &conv, &mut state.conv)?;
        copy_into(&self.device, &rec, &mut state.recurrent)
    }

    /// The engine's linear-attention state of layer `i` back in candle's
    /// form (the inverse of [`Self::import_linear`]): the conv window
    /// `(1, channels, 4)` in `dtype` (its first column, which candle's
    /// next step shifts out, zero) and the recurrent state
    /// `(1, v_heads, 128, 128)` f32, both in the checkpoint's head order.
    pub fn export_linear(&self, i: usize, dtype: candle_core::DType) -> Result<(Tensor, Tensor)> {
        let s = self.s;
        let (hk, hv) = (s.k_heads, s.v_heads);
        let r = hv / hk;
        // Checkpoint head h = r * k + g sits at GGUF position g * hk + k.
        let head_at: Vec<u32> = (0..hv).map(|h| ((h % r) * hk + h / r) as u32).collect();
        let qk = 2 * hk * 128;
        let mut channel_at: Vec<u32> = (0..qk as u32).collect();
        for &p in &head_at {
            channel_at.extend((0..128u32).map(|d| qk as u32 + p * 128 + d));
        }
        let device = candle_core::Device::Cuda(self.device.clone());
        let Mixer::Linear(_, state) = &self.layers[i].mixer else {
            bail!("layer {i} is not linear attention");
        };
        let channels = s.channels_total();
        let hist = tensor_of(&self.device, &state.conv, (channels, 3))?;
        let conv = Tensor::cat(
            &[
                &Tensor::zeros((channels, 1), candle_core::DType::F32, &device)?,
                &hist,
            ],
            1,
        )?
        .index_select(&Tensor::new(channel_at.as_slice(), &device)?, 0)?
        .to_dtype(dtype)?
        .unsqueeze(0)?;
        let rec = tensor_of(&self.device, &state.recurrent, (128, hv, 128))?
            .permute((1, 0, 2))?
            .contiguous()?
            .index_select(&Tensor::new(head_at.as_slice(), &device)?, 0)?
            .unsqueeze(0)?;
        Ok((conv, rec))
    }

    /// The engine's attention caches of layer `i` for the first `len`
    /// positions, in candle's form: K and V `(1, kv_heads, len, head_dim)`
    /// and the indexer keys `(1, len, index_dim)`, in `dtype`.
    pub fn export_full(
        &self,
        i: usize,
        len: usize,
        dtype: candle_core::DType,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let a = self.a;
        let Mixer::Full(_, cache) = &self.layers[i].mixer else {
            bail!("layer {i} is not full attention");
        };
        let (k, v, keys) = cache.slices();
        let kv = |c: &CudaSlice<f32>| -> Result<Tensor> {
            Ok(
                tensor_of(&self.device, c, (a.max_len, a.kv_heads, a.head_dim))?
                    .narrow(0, 0, len)?
                    .transpose(0, 1)?
                    .to_dtype(dtype)?
                    .contiguous()?
                    .unsqueeze(0)?,
            )
        };
        let keys = tensor_of(&self.device, keys, (a.max_len, a.index_dim))?
            .narrow(0, 0, len)?
            .to_dtype(dtype)?
            .contiguous()?
            .unsqueeze(0)?;
        Ok((kv(k)?, kv(v)?, keys))
    }

    /// Takes layer `i`'s attention caches from candle after a prefill:
    /// K and V `(1, kv_heads, len, head_dim)` (normed and rotated) and the
    /// indexer's raw keys `(1, len, index_dim)`.
    pub fn import_full(
        &mut self,
        i: usize,
        k: &Tensor,
        v: &Tensor,
        index_keys: &Tensor,
    ) -> Result<()> {
        let len = k.dim(2)?;
        ensure!(
            len <= self.a.max_len,
            "{len} cached positions, the engine holds {}",
            self.a.max_len
        );
        let rows = |t: &Tensor| -> Result<Tensor> {
            Ok(t.squeeze(0)?
                .transpose(0, 1)?
                .to_dtype(candle_core::DType::F32)?
                .contiguous()?)
        };
        let (k, v) = (rows(k)?, rows(v)?);
        let index_keys = index_keys
            .squeeze(0)?
            .to_dtype(candle_core::DType::F32)?
            .contiguous()?;
        let Mixer::Full(_, cache) = &mut self.layers[i].mixer else {
            bail!("layer {i} is not full attention");
        };
        cache.import(&self.device, &k, &v, &index_keys)
    }

    /// Clears every sequence state, in place, for a new request.
    pub fn reset(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            match &mut layer.mixer {
                Mixer::Linear(_, state) => state.clear(&self.device)?,
                Mixer::Full(_, cache) => cache.clear(&self.device)?,
            }
        }
        Ok(())
    }
}

/// Copies a contiguous f32 tensor's values into the front of `dst`.
pub(crate) fn copy_into(device: &CudaDevice, src: &Tensor, dst: &mut CudaSlice<f32>) -> Result<()> {
    let n = src.elem_count();
    ensure!(n <= dst.len(), "{n} values into a buffer of {}", dst.len());
    let (storage, layout) = src.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        bail!("expected a CUDA tensor");
    };
    let slice = storage.as_cuda_slice::<f32>()?;
    let start = layout.start_offset();
    device
        .cuda_stream()
        .memcpy_dtod(&slice.slice(start..start + n), &mut dst.slice_mut(0..n))?;
    Ok(())
}

/// A copy of a device buffer's front as a candle tensor of `shape`.
fn tensor_of<S: Into<candle_core::Shape>>(
    device: &CudaDevice,
    src: &CudaSlice<f32>,
    shape: S,
) -> Result<Tensor> {
    let shape = shape.into();
    let n = shape.elem_count();
    ensure!(n <= src.len(), "{n} values from a buffer of {}", src.len());
    // SAFETY: written in full by the copy below.
    let mut copy = unsafe { device.alloc::<f32>(n)? };
    device
        .cuda_stream()
        .memcpy_dtod(&src.slice(0..n), &mut copy)?;
    Ok(Tensor::from((
        Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
            copy,
            device.clone(),
        )),
        shape,
    )))
}
