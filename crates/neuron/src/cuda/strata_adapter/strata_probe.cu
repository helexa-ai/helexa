// Parity probe for the vendored expert kernels on real GGUF weights.
//
// Runs two experts over random activations through the grouped kernels,
// then recomputes the SwiGLU on the CPU from independently dequantised
// weights. The kernels read 8-bit activations, so a small error is
// expected; a wrong layout or block format produces a large one.

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "strata/kernels/iq_kernels.hpp"

namespace k = strata::kernels;

namespace {

constexpr int64_t kEmbd = 2560;
constexpr int64_t kFf = 640;
constexpr size_t kQ81Block = 36;  // fp16 scale, fp16 sum, 32 int8 values

template <typename T>
T* device_alloc(size_t count) {
    T* p = nullptr;
    if (cudaMalloc(&p, count * sizeof(T)) != cudaSuccess) return nullptr;
    return p;
}

bool read_at(int fd, uint8_t* dst, size_t bytes, uint64_t offset) {
    size_t done = 0;
    while (done < bytes) {
        const ssize_t n = pread(fd, dst + done, bytes - done, offset + done);
        if (n <= 0) return false;
        done += static_cast<size_t>(n);
    }
    return true;
}

std::vector<float> dequant(int type, const uint8_t* device_src, int64_t n) {
    float* d = device_alloc<float>(n);
    std::vector<float> host(n);
    k::iq_dequant_f32(type, device_src, n, d, nullptr);
    cudaMemcpy(host.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);
    return host;
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

}  // namespace

extern "C" int neuron_strata_expert_probe(const char* path, int gu, int dn, uint64_t gate_off, uint64_t up_off,
                                          uint64_t down_off, int first_expert, int tokens, double* relative_error,
                                          double* kernel_ms) {
    if (!k::native_expert_supported(gu, dn, kEmbd, kFf) || tokens < 1) return 2;
    const k::NativeExpertLayout L = k::native_expert_layout(gu, dn, kEmbd, kFf);
    const size_t gate_bytes = kFf * L.gu_row, down_bytes = kEmbd * L.d_row;
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return 3;
    constexpr int groups = 2;
    std::vector<uint8_t> blobs(groups * L.bytes);
    for (int g = 0; g < groups; ++g) {
        const uint64_t e = first_expert + g;
        uint8_t* blob = blobs.data() + g * L.bytes;
        if (!read_at(fd, blob, gate_bytes, gate_off + e * gate_bytes) ||
            !read_at(fd, blob + L.up_off, gate_bytes, up_off + e * gate_bytes) ||
            !read_at(fd, blob + L.down_off, down_bytes, down_off + e * down_bytes)) {
            close(fd);
            return 4;
        }
    }
    close(fd);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> x(tokens * kEmbd);
    for (float& v : x) v = dist(rng);

    const int64_t entries = groups * tokens;
    auto* d_blobs = device_alloc<uint8_t>(blobs.size());
    auto* d_x = device_alloc<float>(x.size());
    auto* d_q = device_alloc<uint8_t>(tokens * (kEmbd / 32) * kQ81Block);
    auto* d_ptr = device_alloc<unsigned long long>(groups);
    auto* d_start = device_alloc<int32_t>(groups + 1);
    auto* d_n = device_alloc<int32_t>(1);
    auto* d_dst = device_alloc<int32_t>(entries);
    auto* d_tok = device_alloc<int32_t>(entries);
    auto* d_out = device_alloc<float>(entries * kEmbd);
    auto* d_scratch = device_alloc<uint8_t>(k::native_expert_scratch_bytes(entries, kFf));
    if (!d_blobs || !d_x || !d_q || !d_ptr || !d_start || !d_n || !d_dst || !d_tok || !d_out || !d_scratch) return 5;

    std::vector<unsigned long long> ptrs(groups);
    std::vector<int32_t> start(groups + 1), dst(entries), tok(entries);
    for (int g = 0; g < groups; ++g) {
        ptrs[g] = reinterpret_cast<unsigned long long>(d_blobs + g * L.bytes);
        start[g] = g * tokens;
        for (int t = 0; t < tokens; ++t) {
            dst[g * tokens + t] = g * tokens + t;
            tok[g * tokens + t] = t;
        }
    }
    start[groups] = entries;
    const int32_t n_groups = groups;
    cudaMemcpy(d_blobs, blobs.data(), blobs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr, ptrs.data(), groups * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start, start.data(), start.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, &n_groups, sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, dst.data(), dst.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tok, tok.data(), tok.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    k::quantize_q8_1_rows(d_x, tokens, kEmbd, d_q, nullptr);
    cudaDeviceSynchronize();

    const auto t0 = std::chrono::steady_clock::now();
    k::native_expert_grouped(L, d_ptr, d_start, d_n, d_dst, d_tok, groups, entries, d_q, d_scratch, d_out, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) return 6;
    *kernel_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    std::vector<float> out(entries * kEmbd);
    cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost);

    double err = 0.0, norm = 0.0;
    std::vector<float> h(kFf);
    for (int g = 0; g < groups; ++g) {
        const uint8_t* blob = d_blobs + g * L.bytes;
        const auto gate = dequant(L.gu_type, blob, kFf * kEmbd);
        const auto up = dequant(L.gu_type, blob + L.up_off, kFf * kEmbd);
        const auto down = dequant(L.d_type, blob + L.down_off, kEmbd * kFf);
        for (int t = 0; t < tokens; ++t) {
            const float* xt = x.data() + t * kEmbd;
            for (int64_t r = 0; r < kFf; ++r) {
                float a = 0.0f, b = 0.0f;
                for (int64_t c = 0; c < kEmbd; ++c) {
                    a += gate[r * kEmbd + c] * xt[c];
                    b += up[r * kEmbd + c] * xt[c];
                }
                h[r] = silu(a) * b;
            }
            const float* got = out.data() + (g * tokens + t) * kEmbd;
            for (int64_t r = 0; r < kEmbd; ++r) {
                float y = 0.0f;
                for (int64_t c = 0; c < kFf; ++c) y += down[r * kFf + c] * h[c];
                err += (double(got[r]) - y) * (double(got[r]) - y);
                norm += double(y) * y;
            }
        }
    }
    *relative_error = std::sqrt(err / norm);

    for (void* p : {static_cast<void*>(d_blobs), static_cast<void*>(d_x), static_cast<void*>(d_q),
                    static_cast<void*>(d_ptr), static_cast<void*>(d_start), static_cast<void*>(d_n),
                    static_cast<void*>(d_dst), static_cast<void*>(d_tok), static_cast<void*>(d_out),
                    static_cast<void*>(d_scratch)}) {
        cudaFree(p);
    }
    return 0;
}
