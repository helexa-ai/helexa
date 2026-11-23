//! FFI to the expert kernels vendored from Strata (`vendor/strata`).
//!
//! Pointers are device addresses unless a parameter says otherwise, and
//! `stream` is a CUDA stream handle (null for the default stream).

use std::ffi::{c_char, c_void};

unsafe extern "C" {
    /// Parity of two real experts against a CPU recomputation. Returns 0
    /// on success.
    #[allow(clippy::too_many_arguments)]
    pub fn neuron_strata_expert_probe(
        path: *const c_char,
        gu: i32,
        down: i32,
        gate_off: u64,
        up_off: u64,
        down_off: u64,
        first_expert: i32,
        tokens: i32,
        relative_error: *mut f64,
        kernel_ms: *mut f64,
    ) -> i32;
    pub fn neuron_strata_pinned_alloc(bytes: usize) -> *mut c_void;
    pub fn neuron_strata_pinned_free(p: *mut c_void);
    /// Queues a copy on `stream`. `pinned_src` must stay unchanged until
    /// the stream has passed it. Returns the CUDA status.
    pub fn neuron_strata_copy_to_device(
        dst: *mut c_void,
        pinned_src: *const c_void,
        bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn neuron_strata_expert_supported(gu_type: i32, d_type: i32, n_embd: i64, n_ff: i64)
    -> i32;
    pub fn neuron_strata_row_bytes(ggml_type: i32, n: i64) -> usize;
    pub fn neuron_strata_expert_bytes(gu_type: i32, d_type: i32, n_embd: i64, n_ff: i64) -> usize;
    pub fn neuron_strata_expert_scratch_bytes(cap_entries: i64, n_ff: i64) -> usize;
    pub fn neuron_strata_quantize_q8_1(
        x: *const f32,
        n_rows: i64,
        n_cols: i64,
        y: *mut c_void,
        stream: *mut c_void,
    );
    pub fn neuron_strata_dequant_f32(
        ggml_type: i32,
        src: *const c_void,
        n: i64,
        dst: *mut f32,
        stream: *mut c_void,
    );
    #[allow(clippy::too_many_arguments)]
    pub fn neuron_strata_expert_grouped(
        gu_type: i32,
        d_type: i32,
        n_embd: i64,
        n_ff: i64,
        grp_ptr: *const u64,
        grp_start: *const i32,
        n_groups: *const i32,
        ent_dst: *const i32,
        ent_tok: *const i32,
        cap_groups: i64,
        cap_entries: i64,
        x_q8_1: *const c_void,
        scratch: *mut c_void,
        out: *mut f32,
        stream: *mut c_void,
    );
}
