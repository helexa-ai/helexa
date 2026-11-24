//! Routed experts on Strata's grouped kernels, from a GGUF's own blocks.
//!
//! Neuron keeps routing. For each layer this module groups the routed
//! tokens by expert, makes each active expert's `[gate | up | down]`
//! blob addressable on the device, and runs them all in one
//! `native_expert_grouped` launch.
//!
//! An expert comes from the VRAM tier when it is held. Decode-sized
//! forwards admit what they miss. A prompt chunk must not (see
//! [`ExpertTier::peek`]), so its misses are copied into a reusable arena
//! for the duration of that layer.

use std::ffi::c_void;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use candle_core::cuda_backend::cudarc::driver::result::mem_pool;
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, DeviceSlice, sys};
use candle_core::{CudaDevice, CudaStorage, DType, Device, Storage, Tensor};
use memmap2::{Advice, Mmap};
use rayon::prelude::*;

use super::gguf_index::{GgufIndex, GgufTensor, read_expert_profile};
use crate::cuda::strata as ffi;
use crate::harness::arch::qwen3_5::expert_tier::ExpertTier;
use crate::harness::arch::qwen3_5::moe::{
    ExternalExpertSource, ExternalExperts, TIER_ADMIT_MAX_ROWS,
};

const Q8_1_BLOCK_BYTES: usize = 36;
const Q8_1_BLOCK_VALUES: usize = 32;
/// Pinned host memory expert blobs are staged through. Pinned because
/// an unpinned source makes the driver stage it again, at a fraction of
/// the link's speed.
const STAGING_BYTES: usize = 2 << 30;

/// Pinned host memory. Freed on the device worker with the model.
struct PinnedBuffer {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the buffer is plain host memory, and every access is made
// while holding the workspace lock.
unsafe impl Send for PinnedBuffer {}

impl PinnedBuffer {
    fn new(len: usize) -> candle_core::Result<Self> {
        // SAFETY: returns null or an allocation of `len` bytes.
        let ptr = unsafe { ffi::neuron_strata_pinned_alloc(len) }.cast::<u8>();
        if ptr.is_null() {
            candle_core::bail!("cannot pin {len} bytes of staging memory");
        }
        Ok(Self { ptr, len })
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        // SAFETY: allocated by `neuron_strata_pinned_alloc`, freed once.
        unsafe { ffi::neuron_strata_pinned_free(self.ptr.cast()) };
    }
}

/// An expert's blob held in VRAM.
pub struct ExpertSlot {
    blob: CudaSlice<u8>,
}

/// A CUDA memory pool for the tier's experts alone.
///
/// Thousands of long-lived expert blobs allocated from the default pool
/// sit among the forward pass's transient tensors and pin the blocks
/// they share, so the pool cannot hand those blocks back: measured on
/// one 5090, 29 GB reserved against a 21 GB peak of live data. With
/// their own pool the experts only ever share blocks with each other,
/// and the default pool's trims return the transients' memory.
struct TierPool(sys::CUmemoryPool);

// SAFETY: a pool handle may be used from any thread of its context, and
// the source is only used from its device worker.
unsafe impl Send for TierPool {}
// SAFETY: as above; allocation from a pool is thread-safe in the driver.
unsafe impl Sync for TierPool {}

impl TierPool {
    fn new(ordinal: usize) -> candle_core::Result<Self> {
        // SAFETY: an all-zero props struct is valid; the fields that
        // matter are set below, and zero handle types means none.
        let mut props: sys::CUmemPoolProps = unsafe { std::mem::zeroed() };
        props.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        props.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id = ordinal as i32;
        // SAFETY: `props` describes this device.
        let pool = unsafe { mem_pool::create(&props) }.map_err(candle_err)?;
        // Keep what eviction frees: the next admission reuses it.
        let mut keep = u64::MAX;
        // SAFETY: the release threshold is a cuuint64_t.
        unsafe {
            mem_pool::set_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&raw mut keep).cast(),
            )
        }
        .map_err(candle_err)?;
        Ok(Self(pool))
    }

    fn alloc(&self, device: &CudaDevice, bytes: usize) -> candle_core::Result<CudaSlice<u8>> {
        let stream = device.cuda_stream();
        let mut ptr: sys::CUdeviceptr = 0;
        // SAFETY: the pool and stream belong to this device's context.
        unsafe { sys::cuMemAllocFromPoolAsync(&mut ptr, bytes, self.0, stream.cu_stream()) }
            .result()
            .map_err(candle_err)?;
        // SAFETY: `ptr` is a fresh allocation of `bytes` bytes. Its drop
        // frees it stream-ordered, which returns it to this pool.
        Ok(unsafe { stream.upgrade_device_ptr(ptr, bytes) })
    }
}

impl Drop for TierPool {
    fn drop(&mut self) {
        // SAFETY: created by `new`; the driver defers the destruction
        // until every allocation from it has been freed.
        let _ = unsafe { mem_pool::destroy(self.0) };
    }
}

fn device_address<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr + 'static>(
    slice: &CudaSlice<T>,
) -> u64 {
    slice.device_ptr(slice.stream()).0
}

/// Per-layer timing, from `NEURON_EXPERT_TIMING`.
fn timing_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("NEURON_EXPERT_TIMING").is_some())
}

fn candle_err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::msg(e.to_string())
}

/// Where one projection of a layer's experts lies.
#[derive(Clone, Copy)]
struct Role {
    shard: usize,
    offset: usize,
    bytes: usize,
}

struct Layer {
    gu_type: i32,
    d_type: i32,
    gate: Role,
    up: Role,
    down: Role,
    blob_bytes: usize,
}

/// One expert blob to copy to the device.
#[derive(Clone, Copy)]
struct Upload {
    expert: usize,
    address: u64,
    in_arena: bool,
}

/// Device buffers reused from layer to layer.
///
/// Reuse is safe because each layer synchronizes the stream before
/// returning, so no queued kernel still reads them.
#[derive(Default)]
struct Workspace {
    q8: Option<CudaSlice<u8>>,
    scratch: Option<CudaSlice<u8>>,
    grp_ptr: Option<CudaSlice<u64>>,
    grp_start: Option<CudaSlice<i32>>,
    n_groups: Option<CudaSlice<i32>>,
    ent_dst: Option<CudaSlice<i32>>,
    ent_tok: Option<CudaSlice<i32>>,
    arena: Option<CudaSlice<u8>>,
    staging: Option<PinnedBuffer>,
}

fn ensure_capacity<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr + 'static>(
    device: &CudaDevice,
    slot: &mut Option<CudaSlice<T>>,
    len: usize,
) -> candle_core::Result<()> {
    if slot.as_ref().is_none_or(|s| s.len() < len) {
        *slot = None;
        // SAFETY: every user writes the range it later reads.
        *slot = Some(unsafe { device.alloc::<T>(len.max(1))? });
    }
    Ok(())
}

fn upload<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr + 'static>(
    device: &CudaDevice,
    slot: &mut Option<CudaSlice<T>>,
    data: &[T],
) -> candle_core::Result<u64> {
    ensure_capacity(device, slot, data.len())?;
    let buffer = slot.as_mut().expect("capacity was just ensured");
    device.memcpy_htod(data, &mut buffer.slice_mut(..data.len()))?;
    Ok(device_address(buffer))
}

/// The model's GGUF experts, shared by its MoE blocks.
pub struct GgufExpertSource {
    device: CudaDevice,
    n_embd: usize,
    n_ff: usize,
    maps: Vec<Mmap>,
    layers: Vec<Layer>,
    tier: Option<Arc<ExpertTier<ExpertSlot>>>,
    workspace: Mutex<Workspace>,
    /// Declared after `tier`, so the experts are freed first.
    tier_pool: TierPool,
}

fn role(
    index: &GgufIndex,
    layer: usize,
    name: &str,
    want_dims: [u64; 3],
) -> Result<(GgufTensor, usize)> {
    let key = format!("blk.{layer}.ffn_{name}_exps.weight");
    let tensor = index
        .tensors
        .get(&key)
        .with_context(|| format!("GGUF has no {key}"))?;
    ensure!(
        tensor.dims == want_dims,
        "{key} is {:?}, expected {want_dims:?}",
        tensor.dims
    );
    // SAFETY: a pure function of its integer arguments.
    let row = unsafe { ffi::neuron_strata_row_bytes(tensor.ggml_type as i32, want_dims[0] as i64) };
    ensure!(row > 0, "{key}: unsupported type {}", tensor.ggml_type);
    Ok((tensor.clone(), row * want_dims[1] as usize))
}

impl GgufExpertSource {
    /// Indexes the GGUF whose first shard is `first` and maps its shards.
    pub fn open(
        first: &Path,
        device: &Device,
        layers: usize,
        experts: usize,
        n_embd: usize,
        n_ff: usize,
        tier_budget_mb: Option<u64>,
    ) -> Result<Self> {
        let device = device.as_cuda_device()?.clone();
        let index = GgufIndex::open(first)?;
        let mut maps = Vec::with_capacity(index.shards.len());
        for path in &index.shards {
            let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
            // SAFETY: model files are not modified while serving.
            let map =
                unsafe { Mmap::map(&file) }.with_context(|| format!("map {}", path.display()))?;
            map.advise(Advice::Random).context("advise expert map")?;
            maps.push(map);
        }
        let (e, h, f) = (experts as u64, n_embd as u64, n_ff as u64);
        let mut parsed = Vec::with_capacity(layers);
        for layer in 0..layers {
            let (gate, gate_bytes) = role(&index, layer, "gate", [h, f, e])?;
            let (up, up_bytes) = role(&index, layer, "up", [h, f, e])?;
            let (down, down_bytes) = role(&index, layer, "down", [f, h, e])?;
            ensure!(
                gate.ggml_type == up.ggml_type,
                "layer {layer}: gate and up differ in type"
            );
            let (gu_type, d_type) = (gate.ggml_type as i32, down.ggml_type as i32);
            // SAFETY: pure functions of their integer arguments.
            let (supported, blob_bytes) = unsafe {
                (
                    ffi::neuron_strata_expert_supported(
                        gu_type,
                        d_type,
                        n_embd as i64,
                        n_ff as i64,
                    ),
                    ffi::neuron_strata_expert_bytes(gu_type, d_type, n_embd as i64, n_ff as i64),
                )
            };
            ensure!(
                supported != 0,
                "layer {layer}: no kernels for types {gu_type}/{d_type}"
            );
            ensure!(
                blob_bytes == gate_bytes + up_bytes + down_bytes,
                "layer {layer}: blob is {blob_bytes} bytes, roles sum to {}",
                gate_bytes + up_bytes + down_bytes
            );
            for (tensor, bytes) in [(&gate, gate_bytes), (&up, up_bytes), (&down, down_bytes)] {
                let end = tensor.offset + (bytes * experts) as u64;
                ensure!(
                    end <= maps[tensor.shard].len() as u64,
                    "layer {layer}: expert tensor runs past the end of its shard"
                );
            }
            let as_role = |t: &GgufTensor, bytes| Role {
                shard: t.shard,
                offset: t.offset as usize,
                bytes,
            };
            parsed.push(Layer {
                gu_type,
                d_type,
                gate: as_role(&gate, gate_bytes),
                up: as_role(&up, up_bytes),
                down: as_role(&down, down_bytes),
                blob_bytes,
            });
        }
        let tier = tier_budget_mb.map(|mb| ExpertTier::new(mb << 20));
        let tier_pool = TierPool::new(device.cuda_stream().context().ordinal())?;
        let profile = std::env::var_os("NEURON_EXPERT_PROFILE")
            .map(|path| read_expert_profile(Path::new(&path), layers, experts))
            .transpose()?;
        tracing::info!(
            shards = maps.len(),
            layers,
            tier_budget_mb,
            "native expert kernels enabled, reading experts from GGUF"
        );
        let source = Self {
            device,
            n_embd,
            n_ff,
            maps,
            layers: parsed,
            tier,
            workspace: Mutex::new(Workspace::default()),
            tier_pool,
        };
        if let Some(ranked) = profile {
            let started = std::time::Instant::now();
            let filled = source.warm_tier(&ranked)?;
            tracing::info!(
                filled,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "VRAM expert tier filled from profile"
            );
        }
        Ok(source)
    }

    fn copy_blob(&self, layer: &Layer, expert: usize, dst: &mut [u8]) {
        let (gate, rest) = dst.split_at_mut(layer.gate.bytes);
        let (up, down) = rest.split_at_mut(layer.up.bytes);
        for (out, role) in [(gate, layer.gate), (up, layer.up), (down, layer.down)] {
            let start = role.offset + expert * role.bytes;
            out.copy_from_slice(&self.maps[role.shard][start..start + role.bytes]);
        }
    }

    /// Queues blob copies on the stream.
    ///
    /// The last batch's staging memory stays in use until the caller
    /// synchronizes. Neighbouring arena blobs are copied as one; slots
    /// are separate allocations even when their addresses adjoin, and a
    /// copy may not span two allocations.
    fn stage_blobs(
        &self,
        staging: &mut Option<PinnedBuffer>,
        layer: &Layer,
        items: &[Upload],
    ) -> candle_core::Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        if staging.is_none() {
            *staging = Some(PinnedBuffer::new(STAGING_BYTES)?);
        }
        let pinned = staging.as_ref().expect("allocated above");
        let stream = self.device.cuda_stream();
        let per_batch = pinned.len / layer.blob_bytes;
        for (b, batch) in items.chunks(per_batch).enumerate() {
            if b > 0 {
                stream.synchronize().map_err(candle_err)?;
            }
            // SAFETY: the stream has passed any earlier use of this
            // range, and the lock gives this call the only reference.
            let host = unsafe {
                std::slice::from_raw_parts_mut(pinned.ptr, batch.len() * layer.blob_bytes)
            };
            host.par_chunks_mut(layer.blob_bytes)
                .zip(batch.par_iter())
                .for_each(|(dst, item)| self.copy_blob(layer, item.expert, dst));
            let mut i = 0;
            while i < batch.len() {
                let mut run = 1;
                while batch[i].in_arena
                    && i + run < batch.len()
                    && batch[i + run].in_arena
                    && batch[i + run].address == batch[i].address + (run * layer.blob_bytes) as u64
                {
                    run += 1;
                }
                // SAFETY: both ranges are live for `run` blobs, and the
                // staging range is not rewritten before a synchronize.
                let status = unsafe {
                    ffi::neuron_strata_copy_to_device(
                        batch[i].address as *mut c_void,
                        host.as_ptr().add(i * layer.blob_bytes).cast(),
                        run * layer.blob_bytes,
                        stream.cu_stream() as *mut c_void,
                    )
                };
                if status != 0 {
                    candle_core::bail!("expert upload failed with CUDA status {status}");
                }
                i += run;
            }
        }
        Ok(())
    }

    /// Fills the tier with the profile's hottest experts, so the first
    /// requests do not pay for its warm-up.
    fn warm_tier(&self, ranked: &[(u16, u16)]) -> candle_core::Result<usize> {
        let Some(tier) = &self.tier else {
            return Ok(0);
        };
        let mut workspace = self.workspace.lock().map_err(candle_err)?;
        let capacity = tier.capacity_bytes();
        let mut used = 0_u64;
        let mut by_layer: Vec<Vec<Upload>> = vec![Vec::new(); self.layers.len()];
        let mut filled = 0;
        for &(l, e) in ranked {
            let (l, e) = (l as usize, e as usize);
            let bytes = self.layers[l].blob_bytes as u64;
            if used + bytes > capacity {
                break;
            }
            used += bytes;
            let pending = &mut by_layer[l];
            tier.get_or_load(l as u32, e as u32, bytes, || -> candle_core::Result<_> {
                // Filled by `stage_blobs` before first use.
                let blob = self.tier_pool.alloc(&self.device, bytes as usize)?;
                pending.push(Upload {
                    expert: e,
                    address: device_address(&blob),
                    in_arena: false,
                });
                Ok(ExpertSlot { blob })
            })?;
            filled += 1;
        }
        for (l, pending) in by_layer.iter().enumerate() {
            self.stage_blobs(&mut workspace.staging, &self.layers[l], pending)?;
            self.device
                .cuda_stream()
                .synchronize()
                .map_err(candle_err)?;
        }
        Ok(filled)
    }

    fn forward_layer(
        &self,
        layer_index: usize,
        xs: &Tensor,
        tokens_for: &[Vec<u32>],
        weights_for: &[Vec<f32>],
        k: usize,
    ) -> candle_core::Result<Tensor> {
        let started = std::time::Instant::now();
        let layer = &self.layers[layer_index];
        let (rows, hidden) = xs.dims2()?;
        if hidden != self.n_embd {
            candle_core::bail!(
                "activations are {hidden} wide, experts take {}",
                self.n_embd
            );
        }
        let entries = rows * k;
        let active: Vec<usize> = (0..tokens_for.len())
            .filter(|&e| !tokens_for[e].is_empty())
            .collect();
        let admit = rows <= TIER_ADMIT_MAX_ROWS;

        let mut workspace = self.workspace.lock().map_err(candle_err)?;
        let ws = &mut *workspace;

        // Resolve every active expert to a device address.
        let mut held = Vec::with_capacity(active.len());
        let mut addresses = vec![0_u64; active.len()];
        let mut misses = Vec::new();
        let mut uploads: Vec<Upload> = Vec::new();
        for (g, &e) in active.iter().enumerate() {
            let slot = match &self.tier {
                Some(tier) if admit => Some(tier.get_or_load(
                    layer_index as u32,
                    e as u32,
                    layer.blob_bytes as u64,
                    || -> candle_core::Result<_> {
                        // Filled by `stage_blobs` before the launch.
                        let blob = self.tier_pool.alloc(&self.device, layer.blob_bytes)?;
                        uploads.push(Upload {
                            expert: e,
                            address: device_address(&blob),
                            in_arena: false,
                        });
                        Ok(ExpertSlot { blob })
                    },
                )?),
                Some(tier) => tier.peek(layer_index as u32, e as u32),
                None => None,
            };
            match slot {
                Some(slot) => {
                    addresses[g] = device_address(&slot.blob);
                    held.push(slot);
                }
                None => misses.push((g, e)),
            }
        }
        let resolved = started.elapsed();
        if !misses.is_empty() {
            ensure_capacity(&self.device, &mut ws.arena, misses.len() * layer.blob_bytes)?;
            let arena_address = device_address(ws.arena.as_ref().expect("allocated above"));
            for (i, &(g, e)) in misses.iter().enumerate() {
                let address = arena_address + (i * layer.blob_bytes) as u64;
                addresses[g] = address;
                uploads.push(Upload {
                    expert: e,
                    address,
                    in_arena: true,
                });
            }
        }
        self.stage_blobs(&mut ws.staging, layer, &uploads)?;
        let streamed = started.elapsed();

        // Each token writes its k outputs side by side, so the mix below
        // is one multiply and one sum rather than a scatter.
        let mut grp_start = Vec::with_capacity(active.len() + 1);
        let mut ent_dst = Vec::with_capacity(entries);
        let mut ent_tok = Vec::with_capacity(entries);
        let mut weights = vec![0_f32; entries];
        let mut rank = vec![0_usize; rows];
        for &e in &active {
            grp_start.push(ent_dst.len() as i32);
            for (&token, &weight) in tokens_for[e].iter().zip(&weights_for[e]) {
                let token = token as usize;
                let dst = token * k + rank[token];
                rank[token] += 1;
                ent_dst.push(dst as i32);
                ent_tok.push(token as i32);
                weights[dst] = weight;
            }
        }
        grp_start.push(ent_dst.len() as i32);
        if ent_dst.len() != entries || rank.iter().any(|&r| r != k) {
            candle_core::bail!("routing did not give every token {k} experts");
        }

        let grp_ptr = upload(&self.device, &mut ws.grp_ptr, &addresses)?;
        let grp_start_ptr = upload(&self.device, &mut ws.grp_start, &grp_start)?;
        let n_groups = upload(&self.device, &mut ws.n_groups, &[active.len() as i32])?;
        let ent_dst_ptr = upload(&self.device, &mut ws.ent_dst, &ent_dst)?;
        let ent_tok_ptr = upload(&self.device, &mut ws.ent_tok, &ent_tok)?;
        ensure_capacity(
            &self.device,
            &mut ws.q8,
            rows * (hidden / Q8_1_BLOCK_VALUES) * Q8_1_BLOCK_BYTES,
        )?;
        // SAFETY: a pure function of its integer arguments.
        let scratch_bytes =
            unsafe { ffi::neuron_strata_expert_scratch_bytes(entries as i64, self.n_ff as i64) };
        ensure_capacity(&self.device, &mut ws.scratch, scratch_bytes)?;

        let prepared = started.elapsed();
        let x = xs.to_dtype(DType::F32)?.contiguous()?;
        let (storage, x_layout) = x.storage_and_layout();
        let Storage::Cuda(storage) = &*storage else {
            candle_core::bail!("native experts need activations on the GPU");
        };
        let x_slice = storage.as_cuda_slice::<f32>()?;
        let x_ptr = x_slice
            .slice(x_layout.start_offset()..)
            .device_ptr(x_slice.stream())
            .0 as *const f32;

        // SAFETY: the kernel writes `entries * hidden` values.
        let out = unsafe { self.device.alloc::<f32>(entries * hidden)? };
        let stream = self.device.cuda_stream().cu_stream() as *mut c_void;
        // SAFETY: every pointer is a live device allocation sized above
        // and owned for the call. `held` keeps tier slots alive until
        // the synchronize below, including any evicted while resolving.
        unsafe {
            let q8 = device_address(ws.q8.as_ref().expect("allocated")) as *mut c_void;
            ffi::neuron_strata_quantize_q8_1(x_ptr, rows as i64, hidden as i64, q8, stream);
            ffi::neuron_strata_expert_grouped(
                layer.gu_type,
                layer.d_type,
                self.n_embd as i64,
                self.n_ff as i64,
                grp_ptr as *const u64,
                grp_start_ptr as *const i32,
                n_groups as *const i32,
                ent_dst_ptr as *const i32,
                ent_tok_ptr as *const i32,
                active.len() as i64,
                entries as i64,
                q8 as *const c_void,
                device_address(ws.scratch.as_ref().expect("allocated")) as *mut c_void,
                device_address(&out) as *mut f32,
                stream,
            );
        }
        self.device
            .cuda_stream()
            .synchronize()
            .map_err(candle_err)?;
        drop(held);
        if timing_enabled() {
            let finished = started.elapsed();
            tracing::info!(
                layer = layer_index,
                rows,
                active = active.len(),
                misses = misses.len(),
                resolve_us = resolved.as_micros() as u64,
                arena_us = (streamed - resolved).as_micros() as u64,
                arrays_us = (prepared - streamed).as_micros() as u64,
                kernel_us = (finished - prepared).as_micros() as u64,
                "native experts"
            );
        }

        let out = Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, self.device.clone())),
            (rows, k, hidden),
        ));
        let weights = Tensor::from_vec(weights, (rows, k, 1), xs.device())?;
        out.broadcast_mul(&weights)?.sum(1)
    }
}

struct NativeBlock {
    source: Arc<GgufExpertSource>,
    layer: usize,
}

impl ExternalExperts for NativeBlock {
    fn forward(
        &self,
        xs: &Tensor,
        tokens_for: &[Vec<u32>],
        weights_for: &[Vec<f32>],
        k: usize,
    ) -> candle_core::Result<Tensor> {
        self.source
            .forward_layer(self.layer, xs, tokens_for, weights_for, k)
    }
}

/// Hands each MoE block its view of one [`GgufExpertSource`].
pub struct NativeExpertFactory(pub Arc<GgufExpertSource>);

impl ExternalExpertSource for NativeExpertFactory {
    fn block(&self, layer: usize) -> Result<Box<dyn ExternalExperts>> {
        ensure!(
            layer < self.0.layers.len(),
            "no GGUF experts for layer {layer}"
        );
        Ok(Box::new(NativeBlock {
            source: self.0.clone(),
            layer,
        }))
    }
}
