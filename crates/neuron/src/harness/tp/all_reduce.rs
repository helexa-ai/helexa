//! `AllReduce` as a candle `CustomOp1` — the bridge between candle's
//! `Tensor` graph and `cudarc::nccl::Comm::all_reduce`.
//!
//! Ported from the canonical
//! `candle-examples/examples/llama_multiprocess/model.rs` pattern.
//! Row-parallel layers apply this op after their local matmul to sum
//! partial outputs across NCCL ranks.
//!
//! Available only under `--features cuda`; on CPU builds this module
//! is empty and row-parallel layers degenerate to local matmul only
//! (useful for compile-checking the model code; correctness requires
//! cuda).
//!
//! Thread-safety caveat: NCCL communicators are technically only
//! safe to use from a single thread at a time
//! (https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html).
//! We hold the `AllReduce` behind an `Arc<Comm>` and only issue ops
//! against it from the dedicated `spawn_blocking` thread the inference
//! pipeline already uses for candle's forward passes.

#![cfg(feature = "cuda")]

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CudaStorage, CustomOp1, DType, Layout, Result, Shape};
use cudarc::nccl::{Comm, ReduceOp};
use half::{bf16, f16};
use std::sync::Arc;

/// Wraps an NCCL `Comm` so it can be plugged into a candle forward
/// graph as a custom op. Each row-parallel layer holds one of these.
pub struct AllReduce {
    comm: Arc<Comm>,
}

// SAFETY: `Comm` contains a raw `ncclComm_t` pointer; NCCL's docs note
// that issuing ops against one comm from multiple threads concurrently
// is unsafe. We serialise via the single spawn_blocking thread that
// drives the model's forward pass. The Send/Sync impl is necessary
// because candle's CustomOp1 trait bounds require it; the correctness
// invariant is enforced at the call site, not the type level.
unsafe impl Send for AllReduce {}
unsafe impl Sync for AllReduce {}

impl AllReduce {
    pub fn new(comm: Arc<Comm>) -> Self {
        Self { comm }
    }

    pub fn comm(&self) -> &Arc<Comm> {
        &self.comm
    }
}

impl CustomOp1 for AllReduce {
    fn name(&self) -> &'static str {
        "neuron.tp.all_reduce"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("AllReduce custom-op invoked on CPU storage; TP requires CUDA")
    }

    fn cuda_fwd(&self, s: &CudaStorage, l: &Layout) -> Result<(CudaStorage, Shape)> {
        // Reject non-contiguous inputs explicitly — copying them
        // server-side would mask shape bugs (a TP layer feeding a
        // strided activation into all_reduce is almost certainly a
        // model construction error).
        fn require_contiguous<T: cudarc::driver::DeviceRepr>(
            slice: &cudarc::driver::CudaSlice<T>,
            l: &Layout,
        ) -> Result<()> {
            match l.contiguous_offsets() {
                Some((0, n)) if n == slice.len() => Ok(()),
                _ => candle_core::bail!(
                    "AllReduce input is non-contiguous: layout={:?}, slice_len={}",
                    l,
                    slice.len()
                ),
            }
        }

        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();

        let out = match s.dtype() {
            DType::BF16 => {
                let src = s.as_cuda_slice::<bf16>()?;
                require_contiguous(src, l)?;
                let mut dst = unsafe { dev.alloc::<bf16>(elem_count) }?;
                self.comm
                    .all_reduce(src, &mut dst, &ReduceOp::Sum)
                    .map_err(|e| candle_core::Error::Msg(format!("nccl all_reduce bf16: {e:?}")))?;
                CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F16 => {
                let src = s.as_cuda_slice::<f16>()?;
                require_contiguous(src, l)?;
                let mut dst = unsafe { dev.alloc::<f16>(elem_count) }?;
                self.comm
                    .all_reduce(src, &mut dst, &ReduceOp::Sum)
                    .map_err(|e| candle_core::Error::Msg(format!("nccl all_reduce f16: {e:?}")))?;
                CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F32 => {
                let src = s.as_cuda_slice::<f32>()?;
                require_contiguous(src, l)?;
                let mut dst = unsafe { dev.alloc::<f32>(elem_count) }?;
                self.comm
                    .all_reduce(src, &mut dst, &ReduceOp::Sum)
                    .map_err(|e| candle_core::Error::Msg(format!("nccl all_reduce f32: {e:?}")))?;
                CudaStorage::wrap_cuda_slice(dst, dev)
            }
            dtype => candle_core::bail!(
                "AllReduce: unsupported dtype {dtype:?}; TP path expects bf16/f16/f32"
            ),
        };
        Ok((out, l.shape().clone()))
    }
}
