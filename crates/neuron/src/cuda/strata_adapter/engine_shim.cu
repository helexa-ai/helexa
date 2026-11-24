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
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_rope.hpp"
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

// The shared expert's elementwise steps, which upstream fuses into code
// the engine does not vendor: SiLU(gate) * up, and scaling the expert's
// output by the sigmoid of its one-scalar gate.
__global__ void swiglu_kernel(const float* gate, const float* up, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        out[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

__global__ void scale_by_sigmoid_kernel(float* x, const float* gate, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 1.0f / (1.0f + expf(-gate[0]));
}

// ---- dense decode attention over a cache allocated once (#427).
//
// The cache is `[position][kv_head][head_dim]` f32. The current position
// lives in device memory (`pos[0]`), so nothing here takes a per-token
// argument and the step can be captured.

// Writes one token's rows (`width` floats each, `src_b` optional) into
// `dst_a[pos * width]` and `dst_b[pos * width]`.
__global__ void append_rows_kernel(const float* src_a, const float* src_b, float* dst_a, float* dst_b,
                                   const int* pos, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const size_t at = (size_t) pos[0] * width + i;
    dst_a[at] = src_a[i];
    if (src_b != nullptr) dst_b[at] = src_b[i];
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// One block of 8 warps per query head, D = 256 (8 values per lane). Each
// warp takes positions w, w + 8, ... up to pos[0], keeping an online
// softmax; the warps' partial results are merged at the end. The output
// is multiplied by sigmoid(gate) (Qwen's per-head output gate), the gate
// row of head h at `gate[h * gate_stride]`.
template <int D>
__global__ void __launch_bounds__(256) decode_attn_kernel(const float* q, const float* kc, const float* vc,
                                                          const int* pos, int kv_heads, int group, float scale,
                                                          const float* gate, int gate_stride, float* out) {
    constexpr int PER = D / 32;
    constexpr int WARPS = 8;
    const int h = blockIdx.x, kvh = h / group;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n = pos[0] + 1;
    float qv[PER], acc[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        qv[i] = q[(size_t) h * D + lane * PER + i];
        acc[i] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    for (int p = warp; p < n; p += WARPS) {
        const float* kr = kc + ((size_t) p * kv_heads + kvh) * D + lane * PER;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < PER; ++i) dot += qv[i] * kr[i];
        const float sc = warp_sum(dot) * scale;
        const float m_new = fmaxf(m, sc);
        const float corr = expf(m - m_new), e = expf(sc - m_new);
        l = l * corr + e;
        const float* vr = vc + ((size_t) p * kv_heads + kvh) * D + lane * PER;
#pragma unroll
        for (int i = 0; i < PER; ++i) acc[i] = acc[i] * corr + e * vr[i];
        m = m_new;
    }
    __shared__ float s_m[WARPS], s_l[WARPS];
    __shared__ float s_acc[WARPS][D];
    if (lane == 0) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
#pragma unroll
    for (int i = 0; i < PER; ++i) s_acc[warp][lane * PER + i] = acc[i];
    __syncthreads();
    float mx = -INFINITY;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) mx = fmaxf(mx, s_m[w]);
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) total += s_m[w] == -INFINITY ? 0.0f : s_l[w] * expf(s_m[w] - mx);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w)
            if (s_m[w] != -INFINITY) v += s_acc[w][d] * expf(s_m[w] - mx);
        const float g = gate[(size_t) h * gate_stride + d];
        out[(size_t) h * D + d] = v / total * (1.0f / (1.0f + expf(-g)));
    }
}

// `dst` = `times` copies of the `n`-float row `src` (the embedding into
// every residual stream).
__global__ void repeat_row_kernel(const float* src, float* dst, int n, int times) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * times) dst[i] = src[i % n];
}

__global__ void add_inplace_kernel(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// PLE's conv history, row-fastest (`hist[row + 9 * channel]`): drop the
// oldest row and append this token's normalized value. One thread owns a
// channel, so the in-place shift is well defined.
__global__ void ple_history_kernel(float* hist, const float* normalized, int channels) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float* row = hist + (size_t) c * 9;
#pragma unroll
    for (int r = 0; r < 8; ++r) row[r] = row[r + 1];
    row[8] = normalized[c];
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

int neuron_append_rows(const float* src_a, const float* src_b, float* dst_a, float* dst_b, const int* pos, int width,
                       void* stream) {
    return guarded([&] {
        append_rows_kernel<<<(width + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(src_a, src_b, dst_a,
                                                                                              dst_b, pos, width);
    });
}

int neuron_decode_attn(const float* q, const float* kc, const float* vc, const int* pos, int heads, int kv_heads,
                       int head_dim, float scale, const float* gate, int gate_stride, float* out, void* stream) {
    return guarded([&] {
        if (head_dim != 256 || heads % kv_heads != 0) throw std::invalid_argument("decode attention: head_dim 256 only");
        decode_attn_kernel<256><<<heads, 256, 0, static_cast<cudaStream_t>(stream)>>>(
            q, kc, vc, pos, kv_heads, heads / kv_heads, scale, gate, gate_stride, out);
    });
}

int neuron_norm_rope(const float* x, int in_stride, const float* gamma, float* out, int rows, int head_dim, int n_rot,
                     float eps, const int* positions, void* stream) {
    return guarded([&] {
        const k::RopeScaling scaling{};
        k::native_qsa_rms_norm_rope(x, in_stride, gamma, out, rows, head_dim, n_rot, eps, scaling, positions, stream);
    });
}

int neuron_embed_rows(int ggml_type, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok,
                      int64_t n_embd, float* out, void* stream) {
    return guarded([&] { k::iq_embed_rows(ggml_type, table, row_bytes, tokens, n_tok, n_embd, out, stream); });
}

int neuron_repeat_row(const float* src, float* dst, int n, int times, void* stream) {
    return guarded([&] {
        repeat_row_kernel<<<(n * times + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(src, dst, n, times);
    });
}

int neuron_ple_postops(const float* projected_key, const float* hidden, const float* value,
                       const float* history, const float* norm_key, const float* norm_query,
                       const float* norm_conv, const uint16_t* conv1d_f16, float* key, float* normalized,
                       float* gate, float* gated, float* conv, float* result, void* stream) {
    return guarded([&] {
        k::PleWeights w;
        w.norm_key = norm_key;
        w.norm_query = norm_query;
        w.norm_conv = norm_conv;
        w.conv1d_f16 = conv1d_f16;
        // The query's scratch is reused for the normalized gated values,
        // the one alias the kernel allows besides result == hidden.
        k::NativePlePostopsBuffers b{key, normalized, gate, gated, normalized, conv, result};
        k::native_ple_postops(projected_key, hidden, value, history, w, b, stream);
    });
}

int neuron_ple_history_advance(float* hist, const float* normalized, int channels, void* stream) {
    return guarded([&] {
        ple_history_kernel<<<(channels + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
            hist, normalized, channels);
    });
}

int neuron_add_inplace(float* x, const float* y, int n, void* stream) {
    return guarded([&] {
        add_inplace_kernel<<<(n + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(x, y, n);
    });
}

int neuron_swiglu(const float* gate, const float* up, float* out, int n, void* stream) {
    return guarded([&] {
        swiglu_kernel<<<(n + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(gate, up, out, n);
    });
}

int neuron_scale_by_sigmoid(float* x, const float* gate, int n, void* stream) {
    return guarded([&] {
        scale_by_sigmoid_kernel<<<(n + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(x, gate, n);
    });
}

}  // extern "C"
