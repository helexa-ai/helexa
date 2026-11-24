// C ABI over the vendored Strata decode kernels the native qwen4_exp
// engine drives (#427): linear attention, the hyper-connection, the
// router, the MoE combine and the BF16 projections.
//
// Several of these throw on invalid geometry. An exception must not
// unwind into Rust, so every entry point catches and returns a status:
// 0 on success, otherwise a CUDA error code or -1 for a thrown error,
// whose message `neuron_engine_last_error` returns.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>

#include <cuda_runtime.h>

#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/pdl.hpp"

namespace k = strata::kernels;

// Programmatic dependent launch is defined upstream in verify_kernels.cu,
// with the verify-window machinery the engine does not use. With
// `pdl_scope()` off, `launch_pdl` takes its ordinary launch, which is all
// the decode kernels need; PDL comes in with the verify windows (#313).
namespace strata::kernels {
bool& pdl_scope() {
    static thread_local bool on = false;
    return on;
}
bool pdl_supported() { return false; }
bool pdl_launch_ok(const void*, cudaStream_t) { return false; }
}  // namespace strata::kernels

namespace {

thread_local std::string g_last_error;

template <typename F>
int guarded(F&& f) {
    try {
        f();
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return -1;
    } catch (...) {
        g_last_error = "unknown exception";
        return -1;
    }
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        g_last_error = cudaGetErrorString(status);
        return (int) status;
    }
    return 0;
}

k::GrShapes gr_shapes(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    k::GrShapes s;
    s.n_embd = n_embd;
    s.hc = hc;
    s.hc_lr = hc_lr;
    return s;
}

}  // namespace

extern "C" {

const char* neuron_engine_last_error() { return g_last_error.c_str(); }

/// Selects the pinned single-token FP32 contract for the hyper-connection
/// and GDN kernels, as Strata's decode does. Call before any capture.
void neuron_engine_configure() {
    k::gr_set_native_mmvf(true);
    k::native_gdn_set_enabled(true);
    k::native_router_set_enabled(true);
    k::native_moe_combine_set_enabled(true);
}

size_t neuron_gr_workspace_bytes(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return k::gr_workspace_bytes(gr_shapes(n_embd, hc, hc_lr));
}

int neuron_gr_read(const float* residual, const float* w_norm, const uint16_t* w_down, const uint16_t* w_up,
                   const uint16_t* w_inject, float eps, int64_t n_embd, int64_t hc, int64_t hc_lr, void* workspace,
                   float* mixed, float* inject, void* stream) {
    return guarded([&] {
        const k::GrShapes s = gr_shapes(n_embd, hc, hc_lr);
        k::GrWorkspace ws;
        k::gr_workspace_init(s, workspace, ws);
        k::gr_read(residual, w_norm, w_down, w_up, w_inject, eps, s, ws, mixed, inject, stream);
    });
}

int neuron_gr_write(const float* residual, const float* block_out, const float* inject, int64_t n_embd, int64_t hc,
                    int64_t hc_lr, float* residual_out, void* stream) {
    return guarded([&] {
        k::gr_write(residual, block_out, inject, gr_shapes(n_embd, hc, hc_lr), residual_out, stream);
    });
}

int neuron_gdn_conv_l2(float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, void* stream) {
    return guarded([&] { k::fused_gdn_conv_l2(history, qkv, conv_w, h, channels, qk_heads, eps, stream); });
}

int neuron_gdn_ab(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt,
                  const float* ssm_a, float* gate, float* beta, int n_embd, int h_v, void* stream) {
    return guarded([&] { k::fused_gdn_ab(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, stream); });
}

int neuron_gdn_step_norm(float* state, const float* q, const float* k_, const float* v, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                         void* stream) {
    return guarded(
        [&] { k::fused_gdn_step_norm(state, q, k_, v, gate, beta, z, gamma, eps, y, h_k, h_v, stream); });
}

int neuron_bf16_gemv(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    return guarded([&] { k::bf16_gemv_fp32_mmvf(x, w, y, n_in, n_out, stream); });
}

int neuron_router_top10(const float* logits, int32_t* ids, float* weights, void* stream) {
    return guarded([&] { k::native_router_top10(logits, ids, weights, stream); });
}

int neuron_moe_combine(const float* parts, const float* weights, const float* shared, float* output, int64_t n_embd,
                       int64_t k_, void* stream) {
    return guarded([&] { k::native_moe_combine(parts, weights, shared, output, n_embd, k_, stream); });
}

}  // extern "C"
