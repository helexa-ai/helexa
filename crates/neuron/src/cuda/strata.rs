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
    /// `cudaGetLastError`: the error of the most recent failing call,
    /// cleared by reading.
    pub fn neuron_strata_last_error() -> i32;
    /// Queues a kernel copying `n` 32-bit words (`dst` may be the device
    /// address of mapped pinned host memory).
    pub fn neuron_strata_copy_words(
        dst: *mut u32,
        src: *const u32,
        n: i32,
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

    // ---- the native decode engine's kernels (#427): see engine_shim.cu.
    // Each returns 0, a CUDA error code, or -1 with a message from
    // `neuron_engine_last_error`.
    pub fn neuron_engine_last_error() -> *const c_char;
    pub fn neuron_engine_configure();
    pub fn neuron_gr_workspace_bytes(n_embd: i64, hc: i64, hc_lr: i64) -> usize;
    #[allow(clippy::too_many_arguments)]
    pub fn neuron_gr_read(
        residual: *const f32,
        w_norm: *const f32,
        w_down: *const u16,
        w_up: *const u16,
        w_inject: *const u16,
        eps: f32,
        n_embd: i64,
        hc: i64,
        hc_lr: i64,
        workspace: *mut c_void,
        mixed: *mut f32,
        inject: *mut f32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn neuron_gr_write(
        residual: *const f32,
        block_out: *const f32,
        inject: *const f32,
        n_embd: i64,
        hc: i64,
        hc_lr: i64,
        residual_out: *mut f32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn neuron_gdn_conv_l2(
        history: *mut f32,
        qkv: *const f32,
        conv_w: *const f32,
        h: *mut f32,
        channels: i32,
        qk_heads: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn neuron_gdn_ab(
        x: *const f32,
        w_alpha: *const u16,
        w_beta: *const u16,
        dt: *const f32,
        ssm_a: *const f32,
        gate: *mut f32,
        beta: *mut f32,
        n_embd: i32,
        h_v: i32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn neuron_gdn_step_norm(
        state: *mut f32,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        gate: *const f32,
        beta: *const f32,
        z: *const f32,
        gamma: *const f32,
        eps: f32,
        y: *mut f32,
        h_k: i32,
        h_v: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn neuron_bf16_gemv(
        x: *const f32,
        w: *const u16,
        y: *mut f32,
        n_in: i64,
        n_out: i64,
        stream: *mut c_void,
    ) -> i32;
    pub fn neuron_router_top10(
        logits: *const f32,
        ids: *mut i32,
        weights: *mut f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn neuron_moe_combine(
        parts: *const f32,
        weights: *const f32,
        shared: *const f32,
        output: *mut f32,
        n_embd: i64,
        k: i64,
        stream: *mut c_void,
    ) -> i32;
    pub fn neuron_swiglu(
        gate: *const f32,
        up: *const f32,
        out: *mut f32,
        n: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn neuron_scale_by_sigmoid(
        x: *mut f32,
        gate: *const f32,
        n: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn neuron_append_rows(
        src_a: *const f32,
        src_b: *const f32,
        dst_a: *mut f32,
        dst_b: *mut f32,
        pos: *const i32,
        width: i32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn neuron_decode_attn(
        q: *const f32,
        kc: *const f32,
        vc: *const f32,
        pos: *const i32,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        scale: f32,
        gate: *const f32,
        gate_stride: i32,
        out: *mut f32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn neuron_norm_rope(
        x: *const f32,
        in_stride: i32,
        gamma: *const f32,
        out: *mut f32,
        rows: i32,
        head_dim: i32,
        n_rot: i32,
        eps: f32,
        positions: *const i32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn neuron_embed_rows(
        ggml_type: i32,
        table: *const c_void,
        row_bytes: usize,
        tokens: *const i32,
        n_tok: i64,
        n_embd: i64,
        out: *mut f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn neuron_repeat_row(
        src: *const f32,
        dst: *mut f32,
        n: i32,
        times: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn neuron_add_inplace(x: *mut f32, y: *const f32, n: i32, stream: *mut c_void) -> i32;
    /// Strata's PLE block after its two projections, for one token:
    /// `result = hidden + gated + silu(conv(normalized))`. `result` may be
    /// `hidden`; `key`, `normalized`, `gate`, `gated`, `conv` are scratch
    /// (`normalized` is left holding this token's conv input).
    #[allow(clippy::too_many_arguments)]
    pub fn neuron_ple_postops(
        projected_key: *const f32,
        hidden: *const f32,
        value: *const f32,
        history: *const f32,
        norm_key: *const f32,
        norm_query: *const f32,
        norm_conv: *const f32,
        conv1d_f16: *const u16,
        key: *mut f32,
        normalized: *mut f32,
        gate: *mut f32,
        gated: *mut f32,
        conv: *mut f32,
        result: *mut f32,
        stream: *mut c_void,
    ) -> i32;
    /// Shifts PLE's row-fastest conv history by one and appends `normalized`.
    pub fn neuron_ple_history_advance(
        hist: *mut f32,
        normalized: *const f32,
        channels: i32,
        stream: *mut c_void,
    ) -> i32;
}
