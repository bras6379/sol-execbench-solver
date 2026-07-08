// Problem 012 — Fused cos/sin embedding generation (RoPE), CUDA.
//
// Shape-specialized B200 implementation, starting from the 0.900 frontier
// and adding one untried lever: a persistent grid-stride launch for the
// memory-bandwidth-bound 4-freq/thread path.
//
// Paths:
//   * small/latency-bound shapes (B*S < 4096): proven 2-freq/thread float2
//     load + packed __nv_bfloat162 stores, natural grid.
//   * medium/large shapes (B*S >= 4096): 4-freq/thread float4 load + uint2
//     packed stores, launched as a persistent grid-stride kernel.  The grid
//     is clamped to at most 148 blocks so each block iterates over the work
//     with stride grid*block, cutting block-scheduling jitter while keeping
//     enough warps in flight to hide SFU latency.
//
// All paths use one fp32 Cody-Waite range-reduction step before __sincosf,
// then scale and round-to-nearest-even into bf16.  The cat(freqs,freqs)
// duplication is done at store time: each unique frequency is written to
// column i and i+64 of the output row.
//
// Reference:
//   emb = cat(freqs, freqs)
//   cos = emb.cos() * attention_scaling -> bf16
//   sin = emb.sin() * attention_scaling -> bf16
//
// freqs is [B,S,64] fp32; outputs are [B,S,128] bf16.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tuple>

#define HALF 64
#define FULL 128
#define MAX_PERSISTENT_BLOCKS 148

// Cody-Waite range reduction for 2*pi, using fp32 only.  Avoids the slow
// FP64 divide path while preserving enough accuracy for the bf16 tolerance.
__device__ __forceinline__ float cw_reduce(float x) {
    const float INV_2PI  = 0.15915494f;
    const float TWOPI_HI = 6.28318548f;
    const float TWOPI_LO = -1.7484556e-7f;
    float k = nearbyintf(x * INV_2PI);
    float r = fmaf(-k, TWOPI_HI, x);
    return r - k * TWOPI_LO;
}

__device__ __forceinline__ unsigned int bf162_as_u32(__nv_bfloat162 v) {
    union {
        __nv_bfloat162 b;
        unsigned int u;
    } cvt;
    cvt.b = v;
    return cvt.u;
}

__device__ __forceinline__ uint2 pack_bf16x4(float v0, float v1,
                                             float v2, float v3) {
    __nv_bfloat162 lo = __floats2bfloat162_rn(v0, v1);
    __nv_bfloat162 hi = __floats2bfloat162_rn(v2, v3);
    return make_uint2(bf162_as_u32(lo), bf162_as_u32(hi));
}

// 2-freq/thread path for small shapes (and any non-64 half dim fallback).
__global__ void fused_rope2_kernel(const float* __restrict__ freqs,
                                   __nv_bfloat16* __restrict__ cos_out,
                                   __nv_bfloat16* __restrict__ sin_out,
                                   long n_pairs, float scale) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_pairs) return;

    const float2 f = reinterpret_cast<const float2*>(freqs)[t];

    float r0 = cw_reduce(f.x);
    float r1 = cw_reduce(f.y);

    float s0, c0, s1, c1;
    __sincosf(r0, &s0, &c0);
    __sincosf(r1, &s1, &c1);

    __nv_bfloat162 cpair = __floats2bfloat162_rn(c0 * scale, c1 * scale);
    __nv_bfloat162 spair = __floats2bfloat162_rn(s0 * scale, s1 * scale);

    long two_t = t << 1;
    long row = two_t >> 6;
    long col = two_t & (HALF - 1);
    long lo = (row << 7) + col;
    long hi = lo + HALF;

    __nv_bfloat162* cptr = reinterpret_cast<__nv_bfloat162*>(cos_out);
    __nv_bfloat162* sptr = reinterpret_cast<__nv_bfloat162*>(sin_out);
    cptr[lo >> 1] = cpair;
    cptr[hi >> 1] = cpair;
    sptr[lo >> 1] = spair;
    sptr[hi >> 1] = spair;
}

// 4-freq/thread path with a persistent grid-stride loop.  Each thread
// processes multiple quads, stepping by gridDim.x * blockDim.x, which lets
// a fixed small grid saturate the 148-SM part without launching thousands
// of independent blocks.
__global__ void fused_rope4_persistent_kernel(const float* __restrict__ freqs,
                                              uint2* __restrict__ cos_out4,
                                              uint2* __restrict__ sin_out4,
                                              long n_quads, float scale) {
    long stride = (long)gridDim.x * blockDim.x;
    for (long q = (long)blockIdx.x * blockDim.x + threadIdx.x;
         q < n_quads; q += stride) {
        const float4 f = reinterpret_cast<const float4*>(freqs)[q];

        float r0 = cw_reduce(f.x);
        float r1 = cw_reduce(f.y);
        float r2 = cw_reduce(f.z);
        float r3 = cw_reduce(f.w);

        float s0, c0, s1, c1, s2, c2, s3, c3;
        __sincosf(r0, &s0, &c0);
        __sincosf(r1, &s1, &c1);
        __sincosf(r2, &s2, &c2);
        __sincosf(r3, &s3, &c3);

        uint2 cpack = pack_bf16x4(c0 * scale, c1 * scale,
                                  c2 * scale, c3 * scale);
        uint2 spack = pack_bf16x4(s0 * scale, s1 * scale,
                                  s2 * scale, s3 * scale);

        // Each row has 16 float4 groups in the input and 32 uint2 groups in
        // each bf16 output row.  The high half starts 16 uint2 groups after
        // the low half.
        long row = q >> 4;
        long lane = q & 15;
        long base = (row << 5) + lane;

        cos_out4[base]       = cpack;
        cos_out4[base + 16]  = cpack;
        sin_out4[base]       = spack;
        sin_out4[base + 16]  = spack;
    }
}

std::tuple<torch::Tensor, torch::Tensor> run(torch::Tensor freqs,
                                             double attention_scaling) {
    freqs = freqs.contiguous();

    const int64_t B = freqs.size(0);
    const int64_t S = freqs.size(1);
    const int64_t half = freqs.size(2);
    const int64_t full = half * 2;
    const int64_t n_freqs = B * S * half;
    const int64_t n_pairs = n_freqs >> 1;
    const int64_t n_quads = n_freqs >> 2;
    const int64_t rows = B * S;

    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(freqs.device());
    auto cos_out = torch::empty({B, S, full}, opts);
    auto sin_out = torch::empty({B, S, full}, opts);

    if (n_freqs > 0) {
        const float scale = (float)attention_scaling;
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        if (rows >= 4096 && half == HALF) {
            // Persistent 4-freq path: clamp grid to MAX_PERSISTENT_BLOCKS so
            // large shapes amortize launch/scheduling overhead via the stride
            // loop while smaller-but-still-large shapes fall back to a natural
            // grid when it is already smaller than the clamp.
            const int block = 256;
            int grid = (int)((n_quads + block - 1) / block);
            if (grid > MAX_PERSISTENT_BLOCKS) grid = MAX_PERSISTENT_BLOCKS;
            fused_rope4_persistent_kernel<<<grid, block, 0, stream>>>(
                freqs.data_ptr<float>(),
                reinterpret_cast<uint2*>(cos_out.data_ptr<at::BFloat16>()),
                reinterpret_cast<uint2*>(sin_out.data_ptr<at::BFloat16>()),
                n_quads, scale);
        } else {
            // Latency-bound shapes stay on the leaner 2-freq path with a
            // natural grid: just enough threads to cover the work.
            const int block = 256;
            const int grid = (int)((n_pairs + block - 1) / block);
            fused_rope2_kernel<<<grid, block, 0, stream>>>(
                freqs.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(cos_out.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(sin_out.data_ptr<at::BFloat16>()),
                n_pairs, scale);
        }
    }

    return std::make_tuple(cos_out, sin_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "fused cos/sin RoPE embedding generation (CUDA)");
}
