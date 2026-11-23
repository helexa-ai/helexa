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
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, DeviceSlice};
use candle_core::{CudaDevice, CudaStorage, DType, Device, Storage, Tensor};
use memmap2::{Advice, Mmap};
use rayon::prelude::*;

use super::gguf_index::{GgufIndex, GgufTensor};
use crate::cuda::strata as ffi;
use crate::harness::arch::qwen3_5::expert_tier::ExpertTier;
use crate::harness::arch::qwen3_5::moe::{
    ExternalExpertSource, ExternalExperts, TIER_ADMIT_MAX_ROWS,
};

const Q8_1_BLOCK_BYTES: usize = 36;
const Q8_1_BLOCK_VALUES: usize = 32;

/// An expert's blob held in VRAM.
pub struct ExpertSlot {
    blob: CudaSlice<u8>,
}

fn device_address<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr + 'static>(
    slice: &CudaSlice<T>,
) -> u64 {
    slice.device_ptr(slice.stream()).0
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
    host_arena: Vec<u8>,
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
        tracing::info!(
            shards = maps.len(),
            layers,
            tier_budget_mb,
            "native expert kernels enabled, reading experts from GGUF"
        );
        Ok(Self {
            device,
            n_embd,
            n_ff,
            maps,
            layers: parsed,
            tier,
            workspace: Mutex::new(Workspace::default()),
        })
    }

    fn copy_blob(&self, layer: &Layer, expert: usize, dst: &mut [u8]) {
        let (gate, rest) = dst.split_at_mut(layer.gate.bytes);
        let (up, down) = rest.split_at_mut(layer.up.bytes);
        for (out, role) in [(gate, layer.gate), (up, layer.up), (down, layer.down)] {
            let start = role.offset + expert * role.bytes;
            out.copy_from_slice(&self.maps[role.shard][start..start + role.bytes]);
        }
    }

    fn load_slot(&self, layer: &Layer, expert: usize) -> candle_core::Result<ExpertSlot> {
        let mut host = vec![0_u8; layer.blob_bytes];
        self.copy_blob(layer, expert, &mut host);
        // SAFETY: fully written by the copy below.
        let mut blob = unsafe { self.device.alloc::<u8>(host.len())? };
        self.device.memcpy_htod(&host, &mut blob)?;
        Ok(ExpertSlot { blob })
    }

    fn forward_layer(
        &self,
        layer_index: usize,
        xs: &Tensor,
        tokens_for: &[Vec<u32>],
        weights_for: &[Vec<f32>],
        k: usize,
    ) -> candle_core::Result<Tensor> {
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
        for (g, &e) in active.iter().enumerate() {
            let slot = match &self.tier {
                Some(tier) if admit => Some(tier.get_or_load(
                    layer_index as u32,
                    e as u32,
                    layer.blob_bytes as u64,
                    || self.load_slot(layer, e),
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
        if !misses.is_empty() {
            let bytes = misses.len() * layer.blob_bytes;
            if ws.host_arena.len() < bytes {
                ws.host_arena.resize(bytes, 0);
            }
            ws.host_arena[..bytes]
                .par_chunks_mut(layer.blob_bytes)
                .zip(misses.par_iter())
                .for_each(|(dst, &(_, e))| self.copy_blob(layer, e, dst));
            let arena_address = upload(&self.device, &mut ws.arena, &ws.host_arena[..bytes])?;
            for (i, &(g, _)) in misses.iter().enumerate() {
                addresses[g] = arena_address + (i * layer.blob_bytes) as u64;
            }
        }

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
