//! CUDA kernels and their Rust wrappers.
//!
//! Currently scoped to what we need for Qwen3-Next (`qwen3_5`)
//! inference performance — the Gated DeltaNet kernels ported from
//! `EricLBuehler/mistral.rs` (MIT). Each kernel lives in a `.cu`
//! file alongside this module; `build.rs` compiles them all into a
//! static lib via `cudaforge` and links it under the `cuda` feature.
//!
//! When we absorb more upstream kernels (MoE GEMM, top-k, Mamba SSM,
//! etc.) they land here in their own `.cu` + `.rs` pairs.

#[cfg(feature = "cuda")]
pub mod ffi;
#[cfg(feature = "cuda")]
pub mod gdn;
