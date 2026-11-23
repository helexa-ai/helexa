// C ABI over the vendored Strata expert kernels (`vendor/strata`).
//
// Rust cannot name the C++ layout struct or its default arguments, so
// every call here takes the expert's formats and rebuilds the layout.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "strata/kernels/iq_kernels.hpp"

namespace k = strata::kernels;

extern "C" {

int neuron_strata_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    return k::native_expert_supported(gu_type, d_type, n_embd, n_ff) ? 1 : 0;
}

size_t neuron_strata_row_bytes(int ggml_type, int64_t n) { return k::iq_row_bytes(ggml_type, n); }

size_t neuron_strata_expert_bytes(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    return k::native_expert_layout(gu_type, d_type, n_embd, n_ff).bytes;
}

size_t neuron_strata_expert_scratch_bytes(int64_t cap_entries, int64_t n_ff) {
    return k::native_expert_scratch_bytes(cap_entries, n_ff);
}

void neuron_strata_quantize_q8_1(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    k::quantize_q8_1_rows(x, n_rows, n_cols, y, stream);
}

void neuron_strata_dequant_f32(int ggml_type, const void* src, int64_t n, float* dst, void* stream) {
    k::iq_dequant_f32(ggml_type, src, n, dst, stream);
}

void neuron_strata_expert_grouped(int gu_type, int d_type, int64_t n_embd, int64_t n_ff,
                                  const unsigned long long* grp_ptr, const int32_t* grp_start,
                                  const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok,
                                  int64_t cap_groups, int64_t cap_entries, const void* x_q8_1, void* scratch,
                                  float* out, void* stream) {
    const k::NativeExpertLayout layout = k::native_expert_layout(gu_type, d_type, n_embd, n_ff);
    k::native_expert_grouped(layout, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, cap_groups, cap_entries, x_q8_1,
                             scratch, out, stream);
}

void* neuron_strata_pinned_alloc(size_t bytes) {
    void* p = nullptr;
    return cudaHostAlloc(&p, bytes, cudaHostAllocDefault) == cudaSuccess ? p : nullptr;
}

void neuron_strata_pinned_free(void* p) { cudaFreeHost(p); }

int neuron_strata_copy_to_device(void* dst, const void* pinned_src, size_t bytes, void* stream) {
    return cudaMemcpyAsync(dst, pinned_src, bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream));
}

}  // extern "C"
