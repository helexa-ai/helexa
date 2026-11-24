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
    /// Page-locked host memory mapped into the device's address space,
    /// or null. Freed with `neuron_strata_pinned_free`.
    pub fn neuron_strata_host_alloc_mapped(bytes: usize) -> *mut c_void;
    /// The device address of mapped host memory, or 0.
    pub fn neuron_strata_host_device_ptr(host: *mut c_void) -> u64;
    /// Queues `*entry = value` on `stream`. Returns the CUDA status.
    pub fn neuron_strata_table_set(entry: *mut u64, value: u64, stream: *mut c_void) -> i32;
    /// Builds the grouped expert kernel's arrays from one layer's
    /// routing (`ids`, `w`: rows x k) and its expert address table.
    /// Experts whose entry equals `host_table`'s are given arena slots
    /// and listed in `miss_src`; a null `host_table` disables that.
    /// Returns the CUDA status, or -1 for too many experts.
    #[allow(clippy::too_many_arguments)]
    pub fn neuron_strata_route_groups(
        ids: *const u32,
        w: *const f32,
        rows: i32,
        k: i32,
        table: *const u64,
        host_table: *const u64,
        arena: u64,
        blob_bytes: u64,
        n_experts: i32,
        grp_ptr: *mut u64,
        grp_start: *mut i32,
        n_groups: *mut i32,
        ent_dst: *mut i32,
        ent_tok: *mut i32,
        w_out: *mut f32,
        miss_src: *mut u64,
        n_miss: *mut i32,
        stream: *mut c_void,
    ) -> i32;
    /// Copies the first `*n_miss` of `src`'s blobs (`bytes` each) to
    /// consecutive slots at `dst`; `cap` bounds the launch.
    pub fn neuron_strata_gather_blobs(
        src: *const u64,
        n_miss: *const i32,
        cap: i32,
        dst: u64,
        bytes: u64,
        stream: *mut c_void,
    ) -> i32;
    /// Queues a device-to-host copy into pinned memory.
    pub fn neuron_strata_copy_to_host(
        pinned_dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn neuron_strata_event_new() -> *mut c_void;
    pub fn neuron_strata_event_record(event: *mut c_void, stream: *mut c_void) -> i32;
    pub fn neuron_strata_event_sync(event: *mut c_void) -> i32;
    pub fn neuron_strata_event_free(event: *mut c_void);
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
    /// Whether `neuron_strata_mmvq` takes weights of `ggml_type`.
    pub fn neuron_strata_mmvq_supported(ggml_type: i32) -> i32;
    /// `y[c][r] = W[r] . x[c]` for `ncols` q8_1 activation rows.
    #[allow(clippy::too_many_arguments)]
    pub fn neuron_strata_mmvq(
        ggml_type: i32,
        w: *const c_void,
        x_q8_1: *const c_void,
        y: *mut f32,
        n_in: i32,
        n_out: i32,
        ncols: i32,
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
