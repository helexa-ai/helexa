//! FFI declarations for the CUDA kernels in `gdn.cu`.
//!
//! Subset of `EricLBuehler/mistral.rs::mistralrs-core/src/cuda/ffi.rs`
//! covering only the Gated DeltaNet kernels we currently use. Other
//! kernels in the upstream file (MoE GEMM, top-k, Mamba selective
//! scan, etc.) would land here too as we absorb them.
//!
//! All function declarations are MIT-licensed from upstream and
//! unchanged apart from this header.

use std::ffi::c_void;

#[allow(dead_code)]
unsafe extern "C" {
    // GDN (Gated Delta Net) kernels for qwen3_5 / Qwen3-Next.
    pub(crate) fn gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        stream: i64,
    );

    /// Chunked GDN recurrence for prefill (processes tokens in BT=64 chunks).
    pub(crate) fn chunked_gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        stream: i64,
    );

    pub(crate) fn causal_conv1d_update(
        x: *const c_void,
        weight: *const c_void,
        conv_state: *mut c_void,
        output: *mut c_void,
        batch_size: i32,
        conv_dim: i32,
        kernel_size: i32,
        dtype: i32,
        stream: i64,
    );

    pub(crate) fn causal_conv1d_full(
        x: *const c_void,
        weight: *const c_void,
        conv_state_out: *mut c_void,
        output: *mut c_void,
        batch_size: i32,
        conv_dim: i32,
        seq_len: i32,
        kernel_size: i32,
        dtype: i32,
        stream: i64,
    );

    pub(crate) fn fused_gdn_gating(
        b: *const c_void,
        a: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        beta_out: *mut c_void,
        g_out: *mut c_void,
        total_elements: i32,
        num_heads: i32,
        dtype: i32,
        stream: i64,
    );
}
