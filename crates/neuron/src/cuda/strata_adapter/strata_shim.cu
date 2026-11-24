// C ABI over the vendored Strata expert kernels (`vendor/strata`).
//
// Rust cannot name the C++ layout struct or its default arguments, so
// every call here takes the expert's formats and rebuilds the layout.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "strata/kernels/iq_kernels.hpp"

namespace k = strata::kernels;

namespace {

__global__ void table_set_kernel(unsigned long long* entry, unsigned long long value) { *entry = value; }

// Turns one layer's routing, `ids`/`w` (rows x k, row-major), into the
// grouped kernel's arrays. Groups are the routed experts in ascending id,
// entries within a group in ascending token, and each token's k outputs
// are ranked by ascending expert id: the order the host-side builder
// uses, so the arithmetic downstream is unchanged.
constexpr int kMaxExperts = 1024;

// An expert whose table entry is its host copy is a miss: it gets the
// next arena slot, and `miss_src` records where the gather copies it
// from. With `host_table` null every expert is read where the table says.
__global__ void route_groups_kernel(const uint32_t* ids, const float* w, int rows, int k,
                                    const unsigned long long* table, const unsigned long long* host_table,
                                    unsigned long long arena, unsigned long long blob_bytes, int n_experts,
                                    unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups,
                                    int32_t* ent_dst, int32_t* ent_tok, float* w_out, unsigned long long* miss_src,
                                    int32_t* n_miss) {
    __shared__ int counts[kMaxExperts];
    __shared__ int base[kMaxExperts];
    const int entries = rows * k;
    for (int i = threadIdx.x; i < n_experts; i += blockDim.x) counts[i] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < entries; e += blockDim.x) {
        const uint32_t x = ids[e];
        if (x < (uint32_t) n_experts) atomicAdd(&counts[x], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, off = 0, m = 0;
        for (int x = 0; x < n_experts; ++x) {
            const int c = counts[x];
            base[x] = off;
            if (c > 0) {
                const unsigned long long at = table[x];
                if (host_table != nullptr && at == host_table[x]) {
                    grp_ptr[g] = arena + (unsigned long long) m * blob_bytes;
                    miss_src[m] = at;
                    ++m;
                } else {
                    grp_ptr[g] = at;
                }
                grp_start[g] = off;
                ++g;
                off += c;
            }
        }
        grp_start[g] = off;
        *n_groups = g;
        if (n_miss != nullptr) *n_miss = m;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < entries; e += blockDim.x) {
        const int t = e / k;
        const uint32_t x = ids[e];
        if (x >= (uint32_t) n_experts) continue;
        int pos = base[x];
        for (int tt = 0; tt < t; ++tt)
            for (int jj = 0; jj < k; ++jj) pos += ids[tt * k + jj] == x;
        int r = 0;
        for (int jj = 0; jj < k; ++jj) r += ids[t * k + jj] < x;
        const int dst = t * k + r;
        ent_dst[pos] = dst;
        ent_tok[pos] = t;
        w_out[dst] = w[e];
    }
}

// Copies blob `blockIdx.x` of the miss list into the arena, the grid's
// y dimension splitting each blob. Blocks past `*n_miss` do nothing, so
// the launch has a fixed shape whatever the routing was.
constexpr int kGatherSplit = 16;

__global__ void gather_blobs_kernel(const unsigned long long* src, const int32_t* n_miss, unsigned long long dst,
                                    unsigned long long bytes) {
    const int m = blockIdx.x;
    if (m >= *n_miss) return;
    const unsigned long long chunk = ((bytes + kGatherSplit - 1) / kGatherSplit + 15) & ~15ull;
    const unsigned long long lo = (unsigned long long) blockIdx.y * chunk;
    if (lo >= bytes) return;
    const unsigned long long hi = lo + chunk < bytes ? lo + chunk : bytes;
    const uint8_t* from = (const uint8_t*) src[m];
    uint8_t* to = (uint8_t*) (dst + (unsigned long long) m * bytes);
    const bool aligned = ((((unsigned long long) from) | ((unsigned long long) to) | lo) & 15) == 0;
    unsigned long long tail = lo;
    if (aligned) {
        const unsigned long long words = (hi - lo) / 16;
        for (unsigned long long i = threadIdx.x; i < words; i += blockDim.x)
            *(uint4*) (to + lo + i * 16) = *(const uint4*) (from + lo + i * 16);
        tail = lo + words * 16;
    }
    for (unsigned long long t = tail + threadIdx.x; t < hi; t += blockDim.x) to[t] = from[t];
}

// Copies `n` 32-bit words, e.g. into mapped pinned host memory: a kernel
// rather than a memcpy so the copy captures like any other launch.
__global__ void copy_words_kernel(uint32_t* dst, const uint32_t* src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

}  // namespace

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

int neuron_strata_mmvq_supported(int ggml_type) {
    switch (ggml_type) {
        // iq_kernels.cu's STRATA_MMVQ_FMTS.
        case 16: case 17: case 18: case 20: case 21: case 22: case 23: case 29: case 42:
        case 10: case 11: case 12: case 13: case 7: case 6: case 2: case 3: case 8: case 40:
        case 144: case 145: case 146: case 147: case 148: case 149:
            return 1;
        default:
            return 0;
    }
}

void neuron_strata_mmvq(int ggml_type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                        void* stream) {
    k::iq_mmvq(ggml_type, w, x_q8_1, y, n_in, n_out, ncols, stream);
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

void* neuron_strata_host_alloc_mapped(size_t bytes) {
    void* p = nullptr;
    return cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess ? p : nullptr;
}

unsigned long long neuron_strata_host_device_ptr(void* host) {
    void* d = nullptr;
    return cudaHostGetDevicePointer(&d, host, 0) == cudaSuccess ? (unsigned long long) d : 0ull;
}

int neuron_strata_table_set(unsigned long long* entry, unsigned long long value, void* stream) {
    table_set_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(entry, value);
    return cudaGetLastError();
}

int neuron_strata_route_groups(const uint32_t* ids, const float* w, int rows, int k, const unsigned long long* table,
                               const unsigned long long* host_table, unsigned long long arena,
                               unsigned long long blob_bytes, int n_experts, unsigned long long* grp_ptr,
                               int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok, float* w_out,
                               unsigned long long* miss_src, int32_t* n_miss, void* stream) {
    if (n_experts > kMaxExperts) return -1;
    route_groups_kernel<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        ids, w, rows, k, table, host_table, arena, blob_bytes, n_experts, grp_ptr, grp_start, n_groups, ent_dst,
        ent_tok, w_out, miss_src, n_miss);
    return cudaGetLastError();
}

int neuron_strata_gather_blobs(const unsigned long long* src, const int32_t* n_miss, int cap, unsigned long long dst,
                               unsigned long long bytes, void* stream) {
    if (cap <= 0) return 0;
    gather_blobs_kernel<<<dim3((unsigned) cap, kGatherSplit), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        src, n_miss, dst, bytes);
    return cudaGetLastError();
}

int neuron_strata_last_error() { return cudaGetLastError(); }

int neuron_strata_copy_words(uint32_t* dst, const uint32_t* src, int n, void* stream) {
    copy_words_kernel<<<(n + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(dst, src, n);
    return cudaGetLastError();
}

int neuron_strata_copy_to_host(void* pinned_dst, const void* src, size_t bytes, void* stream) {
    return cudaMemcpyAsync(pinned_dst, src, bytes, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream));
}

void* neuron_strata_event_new() {
    cudaEvent_t e = nullptr;
    return cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess ? (void*) e : nullptr;
}

int neuron_strata_event_record(void* event, void* stream) {
    return cudaEventRecord(static_cast<cudaEvent_t>(event), static_cast<cudaStream_t>(stream));
}

int neuron_strata_event_sync(void* event) { return cudaEventSynchronize(static_cast<cudaEvent_t>(event)); }

void neuron_strata_event_free(void* event) { cudaEventDestroy(static_cast<cudaEvent_t>(event)); }

}  // extern "C"
