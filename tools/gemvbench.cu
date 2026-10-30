// Microbenchmark for Q5_K GEMV variants on synthetic weights.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define CK(x) do { cudaError_t e = (x); if (e) { printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// v0: baseline, 4B loads, shared scalar x loads
template <int WARPS>
__global__ void g0(float* __restrict__ y, const uint8_t* __restrict__ W,
                   const float* __restrict__ x, int K, int N) {
    extern __shared__ float sx[];
    for (int i = threadIdx.x; i < K; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 3, o = (lane & 7) * 4;
    const int nsb = K >> 8;
    const int stride = gridDim.x * WARPS;
    for (int row = blockIdx.x * WARPS + warp; row < N; row += stride) {
        const uint8_t* w = W + (size_t)row * (size_t)nsb * 176;
        float acc = 0.f;
        for (int sb = 0; sb < nsb; ++sb) {
            const uint8_t* b = w + sb * 176;
            const float d    = __half2float(*(const __half*)(b));
            const float dmin = __half2float(*(const __half*)(b + 2));
            const uint32_t w0 = *(const uint32_t*)(b + 4);
            const uint32_t w1 = *(const uint32_t*)(b + 8);
            const uint32_t w2 = *(const uint32_t*)(b + 12);
            const uint32_t qh4 = *(const uint32_t*)(b + 16 + o);
            const uint32_t qs4 = *(const uint32_t*)(b + 48 + g * 32 + o);
            const float* xs = sx + sb * 256;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int js = 2 * g + j;
                uint8_t s, m;
                if (js < 4) {
                    s = (w0 >> (8 * js)) & 63;
                    m = (w1 >> (8 * js)) & 63;
                } else {
                    uint32_t lo = (w2 >> (8 * (js - 4))) & 0xFF;
                    s = (uint8_t)((lo & 0xF) | ((((w0 >> (8 * (js - 4))) & 0xFF) >> 6) << 4));
                    m = (uint8_t)((lo >> 4)    | ((((w1 >> (8 * (js - 4))) & 0xFF) >> 6) << 4));
                }
                float ds = d * (float)s, dm = dmin * (float)m;
                const float* xb = xs + 64 * g + 32 * j + o;
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    int q = ((qs4 >> (8 * k + 4 * j)) & 0xF) | (((qh4 >> (8 * k + js)) & 1) << 4);
                    acc = fmaf(xb[k], ds * (float)q - dm, acc);
                }
            }
        }
        acc = warp_sum(acc);
        if (lane == 0) y[row] = acc;
    }
}

// v2: float4 shared x, 2 rows per warp
template <int WARPS>
__global__ void g2(float* __restrict__ y, const uint8_t* __restrict__ W,
                   const float* __restrict__ x, int K, int N) {
    extern __shared__ float sx[];
    for (int i = threadIdx.x; i < K; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 3, o = (lane & 7) * 4;
    const int nsb = K >> 8;
    const int stride = gridDim.x * WARPS * 2;
    for (int row0 = blockIdx.x * WARPS * 2 + warp * 2; row0 < N; row0 += stride) {
        const int row1 = min(row0 + 1, N - 1);
        const uint8_t* w0p = W + (size_t)row0 * (size_t)nsb * 176;
        const uint8_t* w1p = W + (size_t)row1 * (size_t)nsb * 176;
        float acc0 = 0.f, acc1 = 0.f;
        for (int sb = 0; sb < nsb; ++sb) {
            const float4* xs = (const float4*)(sx + sb * 256 + 64 * g + o);
            float4 xa = xs[0], xb4 = xs[8];
            const uint8_t* b0 = w0p + sb * 176;
            const uint8_t* b1 = w1p + sb * 176;
            const float d0    = __half2float(*(const __half*)(b0));
            const float dm0   = __half2float(*(const __half*)(b0 + 2));
            const float d1    = __half2float(*(const __half*)(b1));
            const float dm1   = __half2float(*(const __half*)(b1 + 2));
            const uint32_t s00 = *(const uint32_t*)(b0 + 4), s01 = *(const uint32_t*)(b0 + 8), s02 = *(const uint32_t*)(b0 + 12);
            const uint32_t s10 = *(const uint32_t*)(b1 + 4), s11 = *(const uint32_t*)(b1 + 8), s12 = *(const uint32_t*)(b1 + 12);
            const uint32_t qh0 = *(const uint32_t*)(b0 + 16 + o);
            const uint32_t qh1 = *(const uint32_t*)(b1 + 16 + o);
            const uint32_t qs0 = *(const uint32_t*)(b0 + 48 + g * 32 + o);
            const uint32_t qs1 = *(const uint32_t*)(b1 + 48 + g * 32 + o);
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int js = 2 * g + j;
                const float* xf = j ? (const float*)&xb4 : (const float*)&xa;
                #pragma unroll
                for (int r = 0; r < 2; ++r) {
                    uint32_t qh = r ? qh1 : qh0, qs = r ? qs1 : qs0;
                    uint32_t sw0 = r ? s10 : s00, sw1 = r ? s11 : s01, sw2 = r ? s12 : s02;
                    float d = r ? d1 : d0, dm = r ? dm1 : dm0;
                    uint8_t s, m;
                    if (js < 4) { s = (sw0 >> (8 * js)) & 63; m = (sw1 >> (8 * js)) & 63; }
                    else {
                        uint32_t lo = (sw2 >> (8 * (js - 4))) & 0xFF;
                        s = (uint8_t)((lo & 0xF) | ((((sw0 >> (8 * (js - 4))) & 0xFF) >> 6) << 4));
                        m = (uint8_t)((lo >> 4)    | ((((sw1 >> (8 * (js - 4))) & 0xFF) >> 6) << 4));
                    }
                    float ds = d * (float)s, dmn = dm * (float)m;
                    float acc = 0.f;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        int q = ((qs >> (8 * k + 4 * j)) & 0xF) | (((qh >> (8 * k + js)) & 1) << 4);
                        acc = fmaf(xf[k], ds * (float)q - dmn, acc);
                    }
                    if (r == 0) acc0 += acc; else acc1 += acc;
                }
            }
        }
        acc0 = warp_sum(acc0); acc1 = warp_sum(acc1);
        if (lane == 0) { y[row0] = acc0; if (row1 != row0) y[row1] = acc1; }
    }
}

__device__ __forceinline__ void unpack_sc(uint32_t w0, uint32_t w1, uint32_t w2,
                                          int js, float* ds_out, float* dm_out,
                                          float d, float dmin) {
    uint32_t sv, mv;
    if (js < 4) {
        sv = (w0 >> (8 * js)) & 63;
        mv = (w1 >> (8 * js)) & 63;
    } else {
        const uint32_t lo = (w2 >> (8 * (js - 4))) & 0xFF;
        sv = (lo & 0xF) | ((((w0 >> (8 * (js - 4))) & 0xFF) >> 6) << 4);
        mv = (lo >> 4)   | ((((w1 >> (8 * (js - 4))) & 0xFF) >> 6) << 4);
    }
    *ds_out = d * (float)sv;
    *dm_out = dmin * (float)mv;
}

// v3: int8 x + dp4a, 2 dp4a dot + 2 dp4a ones-sum per superblock/lane
template <int WARPS>
__global__ void g3(float* __restrict__ y, const uint8_t* __restrict__ W,
                   const int8_t* __restrict__ xq, const float* __restrict__ xd,
                   int K, int N) {
    extern __shared__ int8_t sxq[];
    for (int i = threadIdx.x; i < K; i += blockDim.x) sxq[i] = xq[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 3, o = (lane & 7) * 4;
    const int nsb = K >> 8;
    const int stride = gridDim.x * WARPS;
    for (int row = blockIdx.x * WARPS + warp; row < N; row += stride) {
        const uint8_t* w = W + (size_t)row * (size_t)nsb * 176;
        float acc = 0.f;
        for (int sb = 0; sb < nsb; ++sb) {
            const uint8_t* b = w + sb * 176;
            const float d    = __half2float(*(const __half*)(b));
            const float dmin = __half2float(*(const __half*)(b + 2));
            const uint32_t w0 = *(const uint32_t*)(b + 4);
            const uint32_t w1 = *(const uint32_t*)(b + 8);
            const uint32_t w2 = *(const uint32_t*)(b + 12);
            const uint32_t qh4 = *(const uint32_t*)(b + 16 + o);
            const uint32_t qs4 = *(const uint32_t*)(b + 48 + g * 32 + o);
            const int js0 = 2 * g, js1 = js0 + 1;
            const int lo = (int)(qs4 & 0x0F0F0F0Fu);
            const int hi = (int)((qs4 >> 4) & 0x0F0F0F0Fu);
            const int qv0 = lo | (int)(((qh4 >> js0) & 0x01010101u) << 4);
            const int qv1 = hi | (int)(((qh4 >> js1) & 0x01010101u) << 4);
            const int8_t* xp = sxq + sb * 256 + 64 * g + o;
            const int xw0 = *(const int*)(xp);
            const int xw1 = *(const int*)(xp + 32);
            float ds, dm;
            unpack_sc(w0, w1, w2, js0, &ds, &dm, d, dmin);
            acc += xd[sb * 8 + js0] * (ds * (float)__dp4a(xw0, qv0, 0) - dm * (float)__dp4a(xw0, 0x01010101, 0));
            unpack_sc(w0, w1, w2, js1, &ds, &dm, d, dmin);
            acc += xd[sb * 8 + js1] * (ds * (float)__dp4a(xw1, qv1, 0) - dm * (float)__dp4a(xw1, 0x01010101, 0));
        }
        acc = warp_sum(acc);
        if (lane == 0) y[row] = acc;
    }
}

// v4: int8 x + dp4a, precomputed per-subblock x sums (2 dp4a per superblock/lane)
template <int WARPS>
__global__ void g4(float* __restrict__ y, const uint8_t* __restrict__ W,
                   const int8_t* __restrict__ xq, const float* __restrict__ xd,
                   const int* __restrict__ xsum, int K, int N) {
    extern __shared__ int8_t sxq[];
    for (int i = threadIdx.x; i < K; i += blockDim.x) sxq[i] = xq[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 3, o = (lane & 7) * 4;
    const int nsb = K >> 8;
    const int stride = gridDim.x * WARPS;
    for (int row = blockIdx.x * WARPS + warp; row < N; row += stride) {
        const uint8_t* w = W + (size_t)row * (size_t)nsb * 176;
        float acc = 0.f;
        for (int sb = 0; sb < nsb; ++sb) {
            const uint8_t* b = w + sb * 176;
            const float d    = __half2float(*(const __half*)(b));
            const float dmin = __half2float(*(const __half*)(b + 2));
            const uint32_t w0 = *(const uint32_t*)(b + 4);
            const uint32_t w1 = *(const uint32_t*)(b + 8);
            const uint32_t w2 = *(const uint32_t*)(b + 12);
            const uint32_t qh4 = *(const uint32_t*)(b + 16 + o);
            const uint32_t qs4 = *(const uint32_t*)(b + 48 + g * 32 + o);
            const int js0 = 2 * g, js1 = js0 + 1;
            const int qv0 = (int)(qs4 & 0x0F0F0F0Fu) | (int)(((qh4 >> js0) & 0x01010101u) << 4);
            const int qv1 = (int)((qs4 >> 4) & 0x0F0F0F0Fu) | (int)(((qh4 >> js1) & 0x01010101u) << 4);
            const int8_t* xp = sxq + sb * 256 + 64 * g + o;
            const int xw0 = *(const int*)(xp);
            const int xw1 = *(const int*)(xp + 32);
            float ds0, dm0, ds1, dm1;
            unpack_sc(w0, w1, w2, js0, &ds0, &dm0, d, dmin);
            unpack_sc(w0, w1, w2, js1, &ds1, &dm1, d, dmin);
            acc += xd[sb * 8 + js0] * (ds0 * (float)__dp4a(xw0, qv0, 0) - dm0 * (float)xsum[sb * 8 + js0]);
            acc += xd[sb * 8 + js1] * (ds1 * (float)__dp4a(xw1, qv1, 0) - dm1 * (float)xsum[sb * 8 + js1]);
        }
        acc = warp_sum(acc);
        if (lane == 0) y[row] = acc;
    }
}

// v6: original layout, 1 sub-block (32 values) per lane, ROWS rows per warp, dp4a chain.
// 8 lanes per row, lane js covers values [32*js, 32*js+32).
// shared xq: word index sb*64 + q*8 + js holds xq[sb*256 + js*32 + q*4 .. +3]
template <int WARPS, int ROWS = 4, bool SCALES = true>
__global__ void g6(float* __restrict__ y, const uint8_t* __restrict__ W,
                   const int8_t* __restrict__ xq, const float* __restrict__ xd,
                   const int* __restrict__ xsum, int K, int N) {
    extern __shared__ int sx6[];
    const int nw = K >> 2;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) {
        const int sb = i >> 6, rem = i & 63, q = rem >> 3, js = rem & 7;
        sx6[i] = *(const int*)(xq + sb * 256 + js * 32 + q * 4);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int js = lane & 7, r = lane >> 3;
    const int sbh = js >> 1, nibl = js & 1;
    const int nsb = K >> 8;
    const int stride = gridDim.x * WARPS * ROWS;
    for (int row0 = blockIdx.x * WARPS * ROWS + warp * ROWS; row0 < N; row0 += stride) {
        float acc4[4] = {0.f, 0.f, 0.f, 0.f};
        const uint8_t* w0 = W + (size_t)(row0 + 0) * (size_t)nsb * 176;
        const uint8_t* w1 = W + (size_t)(row0 + 1) * (size_t)nsb * 176;
        const uint8_t* w2 = W + (size_t)(row0 + 2) * (size_t)nsb * 176;
        const uint8_t* w3 = W + (size_t)(row0 + 3) * (size_t)nsb * 176;
        for (int sb = 0; sb < nsb; ++sb) {
            int xw[8];
            #pragma unroll
            for (int q = 0; q < 8; ++q) xw[q] = sx6[sb * 64 + q * 8 + js];
            #pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                const uint8_t* b = (rr == 0 ? w0 : rr == 1 ? w1 : rr == 2 ? w2 : w3) + sb * 176;
                const uint32_t w0s = *(const uint32_t*)(b + 4);
                const uint32_t w1s = *(const uint32_t*)(b + 8);
                const uint32_t w2s = *(const uint32_t*)(b + 12);
                const int4 qa = *(const int4*)(b + 48 + 32 * sbh);
                const int4 qb = *(const int4*)(b + 64 + 32 * sbh);
                const int4 ha = *(const int4*)(b + 16);
                const int4 hb = *(const int4*)(b + 32);
                const uint32_t qsw[8] = {(uint32_t)qa.x, (uint32_t)qa.y, (uint32_t)qa.z, (uint32_t)qa.w,
                                         (uint32_t)qb.x, (uint32_t)qb.y, (uint32_t)qb.z, (uint32_t)qb.w};
                const uint32_t qhw[8] = {(uint32_t)ha.x, (uint32_t)ha.y, (uint32_t)ha.z, (uint32_t)ha.w,
                                         (uint32_t)hb.x, (uint32_t)hb.y, (uint32_t)hb.z, (uint32_t)hb.w};
                int dot = 0;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t nib = nibl ? ((qsw[i] >> 4) & 0x0F0F0F0Fu) : (qsw[i] & 0x0F0F0F0Fu);
                    const uint32_t hi = ((qhw[i] >> js) & 0x01010101u) << 4;
                    dot = __dp4a(xw[i], (int)(nib | hi), dot);
                }
                float ds, dm;
                unpack_sc(w0s, w1s, w2s, js, &ds, &dm, __half2float(*(const __half*)(b)),
                          __half2float(*(const __half*)(b + 2)));
                if (SCALES) acc4[rr] += xd[sb * 8 + js] * (ds * (float)dot - dm * (float)xsum[sb * 8 + js]);
                else acc4[rr] += (float)dot;
            }
        }
        float acc = acc4[r];
        acc += __shfl_down_sync(0xffffffffu, acc, 4, 8);
        acc += __shfl_down_sync(0xffffffffu, acc, 2, 8);
        acc += __shfl_down_sync(0xffffffffu, acc, 1, 8);
        if (js == 0 && row0 + r < N) y[row0 + r] = acc;
    }
}

// plain bandwidth test
__global__ void bwt(float* out, const float4* in, size_t n4) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float4 acc = {0,0,0,0};
    for (; i < n4; i += stride) { float4 v = in[i]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
    float s = acc.x + acc.y + acc.z + acc.w;
    if (s == 12345.f) out[0] = s;
}

// read-only: same load pattern as the gemv inner loop, no dequant math
template <int WARPS>
__global__ void r_pat(float* __restrict__ y, const uint8_t* __restrict__ W, int K, int N) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 3, o = (lane & 7) * 4;
    const int nsb = K >> 8;
    const int stride = gridDim.x * WARPS;
    for (int row = blockIdx.x * WARPS + warp; row < N; row += stride) {
        const uint8_t* w = W + (size_t)row * (size_t)nsb * 176;
        int acc = 0;
        for (int sb = 0; sb < nsb; ++sb) {
            const uint8_t* b = w + sb * 176;
            acc += *(const int*)(b + 4) + *(const int*)(b + 8) + *(const int*)(b + 12);
            acc += *(const int*)(b + 16 + o) + *(const int*)(b + 48 + g * 32 + o) + *(const int*)(b);
        }
        if (acc == 12345) y[row] = (float)acc;
    }
}

// read-only: 16B per lane, fully contiguous per row
template <int WARPS>
__global__ void r_wide(float* __restrict__ y, const uint8_t* __restrict__ W, int K, int N) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n4 = (K >> 8) * 11;  // 176 B = 11 float4 per superblock
    const int stride = gridDim.x * WARPS;
    for (int row = blockIdx.x * WARPS + warp; row < N; row += stride) {
        const float4* w = (const float4*)(W + (size_t)row * (size_t)n4 * 16);
        int acc = 0;
        for (int i = lane; i < n4; i += 32) {
            float4 v = w[i];
            acc += (int)v.x + (int)v.y + (int)v.z + (int)v.w;
        }
        if (acc == 12345) y[row] = (float)acc;
    }
}

// read-only: 16B per lane, 8 lanes per row (4 rows per warp)
template <int WARPS>
__global__ void r_wide4(float* __restrict__ y, const uint8_t* __restrict__ W, int K, int N) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r = lane >> 3, q = lane & 7;
    const int n4 = (K >> 8) * 11;
    const int stride = gridDim.x * WARPS * 4;
    for (int row0 = blockIdx.x * WARPS * 4 + warp * 4; row0 < N; row0 += stride) {
        const int row = min(row0 + r, N - 1);
        const float4* w = (const float4*)(W + (size_t)row * (size_t)n4 * 16);
        int acc = 0;
        for (int i = q; i < n4; i += 8) {
            float4 v = w[i];
            acc += (int)v.x + (int)v.y + (int)v.z + (int)v.w;
        }
        if (acc == 12345) y[row] = (float)acc;
    }
}

template <typename F>
static float bench(F f, int iters) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    f(); cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) f();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}

// reference dequant of one q5_K row into f32
static void deq_row(const uint8_t* row, int K, float* out) {
    const int nsb = K / 256;
    for (int i = 0; i < nsb; ++i) {
        const uint8_t* b = row + (size_t)i * 176;
        const float d = __half2float(*(const __half*)(b));
        const float dmin = __half2float(*(const __half*)(b + 2));
        const uint8_t* sc = b + 4;
        const uint8_t* qh = b + 16;
        const uint8_t* ql = b + 48;
        for (int g = 0; g < 4; ++g) {
            for (int j = 0; j < 2; ++j) {
                const int js = 2 * g + j;
                auto gs = [&](int idx, int* s, int* m) {
                    if (idx < 4) { *s = sc[idx] & 63; *m = sc[4 + idx] & 63; }
                    else {
                        const int lo = sc[8 + idx - 4];
                        *s = (lo & 0xF) | ((sc[idx - 4] >> 6) << 4);
                        *m = (lo >> 4)   | ((sc[4 + idx - 4] >> 6) << 4);
                    }
                };
                int s, m; gs(js, &s, &m);
                const float d1 = d * s, m1 = dmin * m;
                for (int l = 0; l < 32; ++l) {
                    const int q = j ? (ql[g * 32 + l] >> 4) : (ql[g * 32 + l] & 0xF);
                    const int hi = (qh[l] >> js) & 1;
                    out[i * 256 + g * 64 + j * 32 + l] = d1 * (float)(q | (hi << 4)) - m1;
                }
            }
        }
    }
}

// per 32-value sub-block stats of one q5_K row: integer dot with int8 x, int sum of x, and scales
static void sub_stats(const uint8_t* row, int sub, const int8_t* xq,
                      long long* dot, long long* sumq, int* sc, int* m, float* d, float* dmin) {
    const int sb = sub / 8, js = sub % 8;
    const uint8_t* b = row + (size_t)sb * 176;
    *d = __half2float(*(const __half*)(b));
    *dmin = __half2float(*(const __half*)(b + 2));
    const uint8_t* s = b + 4;
    if (js < 4) { *sc = s[js] & 63; *m = s[4 + js] & 63; }
    else {
        const int lo = s[8 + js - 4];
        *sc = (lo & 0xF) | ((s[js - 4] >> 6) << 4);
        *m = (lo >> 4) | ((s[js] >> 6) << 4);
    }
    const uint8_t* qh = b + 16;
    const uint8_t* ql = b + 48 + 32 * (js >> 1);
    long long dd = 0, ss = 0;
    for (int pos = 0; pos < 32; ++pos) {
        int q = (js & 1) ? (ql[pos] >> 4) : (ql[pos] & 0xF);
        q |= ((qh[pos] >> js) & 1) << 4;
        const int xv = xq[sb * 256 + js * 32 + pos];
        dd += (long long)xv * q; ss += xv;
    }
    *dot = dd; *sumq = ss;
}

// debug: dump per-sub-block integer dots for row 0
__global__ void g6dbg(float* __restrict__ y, const uint8_t* __restrict__ W,
                      const int8_t* __restrict__ xq, int K, int N) {
    extern __shared__ int sx6[];
    const int nw = K >> 2;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) {
        const int sb = i >> 6, rem = i & 63, q = rem >> 3, js = rem & 7;
        sx6[i] = *(const int*)(xq + sb * 256 + js * 32 + q * 4);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int js = lane & 7, r = lane >> 3;
    const int sbh = js >> 1, nibl = js & 1;
    const int nsb = K >> 8;
    const uint8_t* w = W + (size_t)r * (size_t)nsb * 176;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* b = w + sb * 176;
        const int4 qa = *(const int4*)(b + 48 + 32 * sbh);
        const int4 qb = *(const int4*)(b + 64 + 32 * sbh);
        const int4 ha = *(const int4*)(b + 16);
        const int4 hb = *(const int4*)(b + 32);
        const uint32_t qsw[8] = {(uint32_t)qa.x, (uint32_t)qa.y, (uint32_t)qa.z, (uint32_t)qa.w,
                                 (uint32_t)qb.x, (uint32_t)qb.y, (uint32_t)qb.z, (uint32_t)qb.w};
        const uint32_t qhw[8] = {(uint32_t)ha.x, (uint32_t)ha.y, (uint32_t)ha.z, (uint32_t)ha.w,
                                 (uint32_t)hb.x, (uint32_t)hb.y, (uint32_t)hb.z, (uint32_t)hb.w};
        int dot = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t nib = nibl ? ((qsw[i] >> 4) & 0x0F0F0F0Fu) : (qsw[i] & 0x0F0F0F0Fu);
            const uint32_t hi = ((qhw[i] >> js) & 0x01010101u) << 4;
            dot = __dp4a(sx6[sb * 64 + i * 8 + js], (int)(nib | hi), dot);
        }
        if (r == 0) y[sb * 8 + js] = (float)dot;
    }
}

// same loads as g6, integer adds only (no dp4a, no scale unpack)
template <int WARPS, int ROWS = 4>
__global__ void g6ld(float* __restrict__ y, const uint8_t* __restrict__ W,
                     const int8_t* __restrict__ xq, const float* __restrict__ xd,
                     const int* __restrict__ xsum, int K, int N) {
    extern __shared__ int sx6[];
    const int nw = K >> 2;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) {
        const int sb = i >> 6, rem = i & 63, q = rem >> 3, js = rem & 7;
        sx6[i] = *(const int*)(xq + sb * 256 + js * 32 + q * 4);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int js = lane & 7, r = lane >> 3;
    const int sbh = js >> 1;
    const int nsb = K >> 8;
    const int stride = gridDim.x * WARPS * ROWS;
    for (int row0 = blockIdx.x * WARPS * ROWS + warp * ROWS; row0 < N; row0 += stride) {
        int acc4[4] = {0, 0, 0, 0};
        const uint8_t* wp = W + (size_t)row0 * (size_t)nsb * 176;
        for (int sb = 0; sb < nsb; ++sb) {
            int xw[8];
            #pragma unroll
            for (int q = 0; q < 8; ++q) xw[q] = sx6[sb * 64 + q * 8 + js];
            #pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                const uint8_t* b = wp + (size_t)rr * (size_t)nsb * 176 + sb * 176;
                const int4 qa = *(const int4*)(b + 48 + 32 * sbh);
                const int4 qb = *(const int4*)(b + 64 + 32 * sbh);
                const int4 ha = *(const int4*)(b + 16);
                const int4 hb = *(const int4*)(b + 32);
                const uint32_t w0s = *(const uint32_t*)(b + 4);
                const uint32_t w1s = *(const uint32_t*)(b + 8);
                const uint32_t w2s = *(const uint32_t*)(b + 12);
                acc4[rr] += qa.x + qa.y + qa.z + qa.w + qb.x + qb.y + qb.z + qb.w
                          + ha.x + ha.y + ha.z + ha.w + hb.x + hb.y + hb.z + hb.w
                          + (int)w0s + (int)w1s + (int)w2s + xw[rr] + xw[rr + 4];
            }
        }
        int acc = acc4[r];
        acc += __shfl_down_sync(0xffffffffu, acc, 4, 8);
        acc += __shfl_down_sync(0xffffffffu, acc, 2, 8);
        acc += __shfl_down_sync(0xffffffffu, acc, 1, 8);
        if (js == 0 && row0 + r < N) y[row0 + r] = (float)acc;
    }
}

// v9: 4 lanes per row, one 64-value group per lane, 8 rows per warp, dp4a
template <int WARPS, int RU = 8>
__global__ void g9(float* __restrict__ y, const uint8_t* __restrict__ W,
                   const int8_t* __restrict__ xq, const float* __restrict__ xd,
                   const int* __restrict__ xsum, int K, int N) {
    extern __shared__ int sx9[];
    const int nw = K >> 2;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) {
        const int sb = i >> 6, u = i & 63;
        const int js = 2 * ((u >> 3) & 3) + (u >> 5), q = u & 7;
        sx9[i] = *(const int*)(xq + sb * 256 + js * 32 + q * 4);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane & 3, r = lane >> 2;
    const int js0 = 2 * g, js1 = js0 + 1;
    const int nsb = K >> 8;
    const size_t rstride = (size_t)nsb * 176;
    const int stride = gridDim.x * WARPS * 8;
    for (int row0 = blockIdx.x * WARPS * 8 + warp * 8; row0 < N; row0 += stride) {
        float acc8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const uint8_t* wbase = W + (size_t)row0 * rstride;
        for (int sb = 0; sb < nsb; ++sb) {
            int xw[16];
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                xw[t] = sx9[sb * 64 + g * 8 + t];
                xw[8 + t] = sx9[sb * 64 + 32 + g * 8 + t];
            }
            #pragma unroll RU
            for (int rr = 0; rr < 8; ++rr) {
                const uint8_t* b = wbase + (size_t)rr * rstride + sb * 176;
                const int4 qa = *(const int4*)(b + 48 + 32 * g);
                const int4 qb = *(const int4*)(b + 64 + 32 * g);
                const int4 ha = *(const int4*)(b + 16);
                const int4 hb = *(const int4*)(b + 32);
                const uint32_t qsl[8] = {(uint32_t)qa.x, (uint32_t)qa.y, (uint32_t)qa.z, (uint32_t)qa.w,
                                         (uint32_t)qb.x, (uint32_t)qb.y, (uint32_t)qb.z, (uint32_t)qb.w};
                const uint32_t qhl[8] = {(uint32_t)ha.x, (uint32_t)ha.y, (uint32_t)ha.z, (uint32_t)ha.w,
                                         (uint32_t)hb.x, (uint32_t)hb.y, (uint32_t)hb.z, (uint32_t)hb.w};
                int d0 = 0, d1 = 0;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t lo = qsl[i] & 0x0F0F0F0Fu;
                    const uint32_t hi = (qsl[i] >> 4) & 0x0F0F0F0Fu;
                    const uint32_t h0 = ((qhl[i] >> js0) & 0x01010101u) << 4;
                    const uint32_t h1 = ((qhl[i] >> js1) & 0x01010101u) << 4;
                    d0 = __dp4a(xw[i], (int)(lo | h0), d0);
                    d1 = __dp4a(xw[8 + i], (int)(hi | h1), d1);
                }
                const uint32_t w0s = *(const uint32_t*)(b + 4);
                const uint32_t w1s = *(const uint32_t*)(b + 8);
                const uint32_t w2s = *(const uint32_t*)(b + 12);
                const float d = __half2float(*(const __half*)(b));
                const float dmin = __half2float(*(const __half*)(b + 2));
                float ds0, dm0, ds1, dm1;
                unpack_sc(w0s, w1s, w2s, js0, &ds0, &dm0, d, dmin);
                unpack_sc(w0s, w1s, w2s, js1, &ds1, &dm1, d, dmin);
                acc8[rr] += xd[sb * 8 + js0] * (ds0 * (float)d0 - dm0 * (float)xsum[sb * 8 + js0])
                          + xd[sb * 8 + js1] * (ds1 * (float)d1 - dm1 * (float)xsum[sb * 8 + js1]);
            }
        }
        float acc = acc8[r];
        acc += __shfl_down_sync(0xffffffffu, acc, 2, 4);
        acc += __shfl_down_sync(0xffffffffu, acc, 1, 4);
        if (g == 0 && row0 + r < N) y[row0 + r] = acc;
    }
}

// v11: one row per warp, 32 lanes cover 4 superblocks (one sub-block per lane)
template <int WARPS, int CH = 4>
__global__ void g11(float* __restrict__ y, const uint8_t* __restrict__ W,
                    const int8_t* __restrict__ xq, const float* __restrict__ xd,
                    const int* __restrict__ xsum, int K, int N) {
    extern __shared__ int sx11[];
    const int nw = K >> 2;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) {
        const int sb = i >> 6, u = i & 63;
        const int js = 2 * ((u >> 3) & 3) + (u >> 5), q = u & 7;
        sx11[i] = *(const int*)(xq + sb * 256 + js * 32 + q * 4);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int js = lane & 7, sl = lane >> 3;
    const int sbh = js >> 1;
    const bool nibl = js & 1;
    const int xsbase = (js >> 1) * 8 + (js & 1) * 32;
    const int nsb = K >> 8;
    const int stride = gridDim.x * WARPS;
    for (int row = blockIdx.x * WARPS + warp; row < N; row += stride) {
        const uint8_t* rp = W + (size_t)row * (size_t)nsb * 176;
        float acc = 0.f;
        for (int sb0 = 0; sb0 < nsb; sb0 += CH) {
            const int nch = min(CH, nsb - sb0);
            if (sl < nch) {
                const int sb = sb0 + sl;
                const uint8_t* b = rp + sb * 176;
                const int4 qa = *(const int4*)(b + 48 + 32 * sbh);
                const int4 qb = *(const int4*)(b + 64 + 32 * sbh);
                const int4 ha = *(const int4*)(b + 16);
                const int4 hb = *(const int4*)(b + 32);
                const uint32_t qsl[8] = {(uint32_t)qa.x, (uint32_t)qa.y, (uint32_t)qa.z, (uint32_t)qa.w,
                                         (uint32_t)qb.x, (uint32_t)qb.y, (uint32_t)qb.z, (uint32_t)qb.w};
                const uint32_t qhl[8] = {(uint32_t)ha.x, (uint32_t)ha.y, (uint32_t)ha.z, (uint32_t)ha.w,
                                         (uint32_t)hb.x, (uint32_t)hb.y, (uint32_t)hb.z, (uint32_t)hb.w};
                int dot = 0;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t nib = nibl ? ((qsl[i] >> 4) & 0x0F0F0F0Fu) : (qsl[i] & 0x0F0F0F0Fu);
                    const uint32_t hi = ((qhl[i] >> js) & 0x01010101u) << 4;
                    dot = __dp4a(sx11[sb * 64 + xsbase + i], (int)(nib | hi), dot);
                }
                const uint32_t w0s = *(const uint32_t*)(b + 4);
                const uint32_t w1s = *(const uint32_t*)(b + 8);
                const uint32_t w2s = *(const uint32_t*)(b + 12);
                float ds, dm;
                unpack_sc(w0s, w1s, w2s, js, &ds, &dm, __half2float(*(const __half*)(b)),
                          __half2float(*(const __half*)(b + 2)));
                acc += xd[sb * 8 + js] * (ds * (float)dot - dm * (float)xsum[sb * 8 + js]);
            }
        }
        acc = warp_sum(acc);
        if (lane == 0) y[row] = acc;
    }
}

int main() {
    const int K = 2560;
    const int N = 7168;
    const size_t wbytes = (size_t)N * (K / 256) * 176;
    uint8_t* dW; float *dx, *dy; int8_t* dxq; float* dxd; int* dxsum;
    CK(cudaMalloc(&dW, wbytes));
    CK(cudaMalloc(&dx, K * 4));
    CK(cudaMalloc(&dy, N * 4));
    CK(cudaMalloc(&dxq, K));
    CK(cudaMalloc(&dxd, K / 32 * 4));
    CK(cudaMalloc(&dxsum, K / 32 * 4));
    // synthetic but plausible q5_k data
    uint8_t* hW = (uint8_t*)malloc(wbytes);
    for (size_t i = 0; i < wbytes; ++i) hW[i] = (uint8_t)(i * 1103515245u >> 13);
    for (size_t i = 0; i + 3 < wbytes; i += 176) { *(uint16_t*)(hW + i) = 0x2c00; *(uint16_t*)(hW + i + 2) = 0x2c00; }
    CK(cudaMemcpy(dW, hW, wbytes, cudaMemcpyHostToDevice));
    float* hx = (float*)malloc(K * 4);
    for (int i = 0; i < K; ++i) hx[i] = (float)((i * 37 % 101) - 50) / 50.f;
    CK(cudaMemcpy(dx, hx, K * 4, cudaMemcpyHostToDevice));

    // per-32 int8 quantization of x (llama.cpp q8_0 style, d = amax/127)
    int8_t hxq[2560]; float hxd[80]; int hxsum[80];
    for (int s = 0; s < K / 32; ++s) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(hx[s * 32 + i]));
        const float d = amax / 127.f;
        const float id = d ? 1.f / d : 0.f;
        int sum = 0;
        for (int i = 0; i < 32; ++i) {
            int v = (int)lrintf(hx[s * 32 + i] * id);
            if (v > 127) v = 127; if (v < -127) v = -127;
            hxq[s * 32 + i] = (int8_t)v;
            sum += v;
        }
        hxd[s] = d; hxsum[s] = sum;
    }
    CK(cudaMemcpy(dxq, hxq, K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxd, hxd, sizeof(hxd), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxsum, hxsum, sizeof(hxsum), cudaMemcpyHostToDevice));

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("GPU: %s, SMs %d, bus %.0f bit\n", prop.name, prop.multiProcessorCount,
           prop.memoryBusWidth);
    double bytes = (double)wbytes;

    // bandwidth ceiling
    {
        size_t n4 = (size_t)256 * 1024 * 1024 / 16;
        float4 *a; float *b; CK(cudaMalloc(&a, n4 * 16)); CK(cudaMalloc(&b, 4));
        float ms = bench([&]{ bwt<<<1024, 256>>>(b, a, n4); bwt<<<1024, 256>>>(b, a, n4); }, 5);
        printf("bw test read 2x%zu MB: %.2f ms -> %.1f GB/s\n", n4 * 16 / 1024 / 1024 / 2, ms,
               (double)n4 * 16 * 2 / (ms * 1e6));
        cudaFree(a); cudaFree(b);
    }

    // correctness: compare f32-x vs quantized-x paths on identical weights
    {
        float* got = (float*)malloc(N * 4);
        float hxdq[2560];
        for (int i = 0; i < K; ++i) hxdq[i] = hxd[i / 32] * (float)hxq[i];

        CK(cudaMemset(dy, 0, N * 4));
        g0<8><<<(N + 15) / 16, 256, K * 4>>>(dy, dW, dx, K, N);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got, dy, N * 4, cudaMemcpyDeviceToHost));
        const float r0_f32 = got[0], rl_f32 = got[N - 1];

        float* dxdq; CK(cudaMalloc(&dxdq, K * 4));
        CK(cudaMemcpy(dxdq, hxdq, K * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(dy, 0, N * 4));
        g0<8><<<(N + 15) / 16, 256, K * 4>>>(dy, dW, dxdq, K, N);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got, dy, N * 4, cudaMemcpyDeviceToHost));
        const float r0_q32 = got[0], rl_q32 = got[N - 1];

        CK(cudaMemset(dy, 0, N * 4));
        g4<8><<<(N + 7) / 8, 256, K>>>(dy, dW, dxq, dxd, dxsum, K, N);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got, dy, N * 4, cudaMemcpyDeviceToHost));
        const float r0_q8 = got[0], rl_q8 = got[N - 1];

        CK(cudaMemset(dy, 0, N * 4));
        g6<8><<<(N + 31) / 32, 256, K>>>(dy, dW, dxq, dxd, dxsum, K, N);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got, dy, N * 4, cudaMemcpyDeviceToHost));
        const float r0_g6 = got[0], rl_g6 = got[N - 1];

        CK(cudaMemset(dy, 0, N * 4));
        g9<8><<<(N + 63) / 64, 256, K>>>(dy, dW, dxq, dxd, dxsum, K, N);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got, dy, N * 4, cudaMemcpyDeviceToHost));
        const float r0_g9 = got[0], rl_g9 = got[N - 1];

        CK(cudaMemset(dy, 0, N * 4));
        g11<8><<<(N + 7) / 8, 256, K>>>(dy, dW, dxq, dxd, dxsum, K, N);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got, dy, N * 4, cudaMemcpyDeviceToHost));
        const float r0_g11 = got[0], rl_g11 = got[N - 1];

        float* ref = (float*)malloc(K * 4);
        double h0 = 0, hl = 0;
        long long raw0 = 0, rawl = 0;
        double sc0 = 0, scl = 0;
        for (int row : {0, N - 1}) {
            deq_row(hW + (size_t)row * (K / 256) * 176, K, ref);
            double s = 0;
            for (int i = 0; i < K; ++i) s += (double)ref[i] * hxdq[i];
            long long rd = 0; double sd = 0;
            for (int sub = 0; sub < K / 32; ++sub) {
                long long dot, sumq; int sc, m; float d, dmin;
                sub_stats(hW + (size_t)row * (K / 256) * 176, sub, hxq, &dot, &sumq, &sc, &m, &d, &dmin);
                rd += dot;
                sd += (double)hxd[sub] * ((double)d * sc * (double)dot - (double)dmin * m * (double)sumq);
            }
            if (row == 0) { h0 = s; raw0 = rd; sc0 = sd; }
            else { hl = s; rawl = rd; scl = sd; }
        }
        CK(cudaMemset(dy, 0, N * 4));
        g6<8, 4, false><<<(N + 31) / 32, 256, K>>>(dy, dW, dxq, dxd, dxsum, K, N);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got, dy, N * 4, cudaMemcpyDeviceToHost));
        const float r0_raw = got[0], rl_raw = got[N - 1];

        printf("row 0    : g0 %.4f  g6 %.4f  g9 %.4f  g11 %.4f  hostdeq %.4f\n",
               r0_f32, r0_g6, r0_g9, r0_g11, h0);
        printf("row %4d : g0 %.4f  g6 %.4f  g9 %.4f  g11 %.4f  hostdeq %.4f\n",
               N - 1, rl_f32, rl_g6, rl_g9, rl_g11, hl);
        printf("rawdot   : g6 %.0f  host %lld | scdot: g6 %.4f host %.4f\n", r0_raw, raw0, sc0, sc0);
        printf("rawdot   : g6 %.0f  host %lld\n", rl_raw, rawl);
        printf("scale ref: row0 host %.4f  rowlast host %.4f\n", sc0, scl);
        CK(cudaMemset(dy, 0, N * 4));
        g6dbg<<<1, 32, K>>>(dy, dW, dxq, K, N);
        CK(cudaDeviceSynchronize());
        float hd[8];
        CK(cudaMemcpy(hd, dy, 8 * 4, cudaMemcpyDeviceToHost));
        printf("sub0..7 host:");
        for (int sub = 0; sub < 8; ++sub) {
            long long dot, sumq; int sc, m; float d, dmin;
            sub_stats(hW, sub, hxq, &dot, &sumq, &sc, &m, &d, &dmin);
            printf(" %lld", dot);
        }
        printf("\nsub0..7 gpu :");
        for (int sub = 0; sub < 8; ++sub) printf(" %.0f", (double)hd[sub]);
        printf("\n");
        cudaFree(dxdq); free(ref); free(got);
    }

    const int iters = 100;
#define RUN(name, kern, warps, blocks) do { \
    float ms = bench([&]{ kern<warps><<<blocks, warps * 32, K * 4>>>(dy, dW, dx, K, N); }, iters); \
    printf("%-28s %3d blocks x %2dw: %7.4f ms  %6.1f GB/s\n", name, blocks, warps, ms, bytes / (ms * 1e6)); \
} while (0)
#define RUNQ(name, kern, warps, blocks) do { \
    float ms = bench([&]{ kern<warps><<<blocks, warps * 32, K>>>(dy, dW, dxq, dxd, dxsum, K, N); }, iters); \
    printf("%-28s %3d blocks x %2dw: %7.4f ms  %6.1f GB/s\n", name, blocks, warps, ms, bytes / (ms * 1e6)); \
} while (0)
#define RUNQ3(name, kern, warps, blocks) do { \
    float ms = bench([&]{ kern<warps><<<blocks, warps * 32, K>>>(dy, dW, dxq, dxd, K, N); }, iters); \
    printf("%-28s %3d blocks x %2dw: %7.4f ms  %6.1f GB/s\n", name, blocks, warps, ms, bytes / (ms * 1e6)); \
} while (0)
#define RUNR(name, kern, warps, blocks) do { \
    float ms = bench([&]{ kern<warps><<<blocks, warps * 32>>>(dy, dW, K, N); }, iters); \
    printf("%-28s %3d blocks x %2dw: %7.4f ms  %6.1f GB/s\n", name, blocks, warps, ms, bytes / (ms * 1e6)); \
} while (0)

    RUNR("r_pat  read-only w8",  r_pat,   8, (N + 15) / 8);
    RUNR("r_wide read-only w8",  r_wide,  8, (N + 15) / 8);
    RUNR("r_wide read-only w4",  r_wide,  4, (N + 7) / 4);
    RUNR("r_wide4 read-only w8", r_wide4, 8, (N + 31) / 32);
    RUNR("r_wide4 read-only w4", r_wide4, 4, (N + 15) / 16);
    RUN ("g0 current w8",        g0, 8, (N + 15) / 16);
    RUN ("g2 2rows/warp w8",     g2, 8, (N + 15) / 32);
    RUNQ3("g3 dp4a w8",          g3, 8, (N + 15) / 8);
    RUNQ ("g4 dp4a+xsum w8",     g4, 8, (N + 15) / 8);
    RUNQ ("g4 dp4a+xsum w4",     g4, 4, (N + 7) / 4);
    RUNQ ("g4 dp4a+xsum w16",    g4, 16, (N + 31) / 16);
    RUNQ ("g4 dp4a+xsum w8 b112", g4, 8, 112);
    RUNQ ("g4 dp4a+xsum w8 b56",  g4, 8, 56);
    RUNQ ("g4 dp4a+xsum w4 b112", g4, 4, 112);
    RUNQ ("g4 dp4a+xsum w4 b224", g4, 4, 224);
    RUNQ ("g6 sub/lane w8",      g6, 8, (N + 31) / 32);
    RUNQ ("g6 sub/lane w16",     g6, 16, (N + 63) / 64);
    RUNQ ("g6 sub/lane w8 b448", g6, 8, 448);
    RUNQ ("g6 sub/lane w8 b112", g6, 8, 112);
    RUNQ ("g6 sub/lane w4 b448", g6, 4, 448);
    RUNQ ("g6 sub/lane w4 b224", g6, 4, 224);
    RUNQ ("g6ld loads-only w8",  g6ld, 8, (N + 31) / 32);
    RUNQ ("g6ld loads-only w16", g6ld, 16, (N + 63) / 64);
    RUNQ ("g9 64val/lane w8",    g9, 8, (N + 63) / 64);
    {
        float ms = bench([&]{ g9<8, 2><<<(N + 63) / 64, 256, K>>>(dy, dW, dxq, dxd, dxsum, K, N); }, iters);
        printf("%-28s %3d blocks x %2dw: %7.4f ms  %6.1f GB/s\n", "g9 ru2", (N + 63) / 64, 8, ms, bytes / (ms * 1e6));
        ms = bench([&]{ g9<8, 1><<<(N + 63) / 64, 256, K>>>(dy, dW, dxq, dxd, dxsum, K, N); }, iters);
        printf("%-28s %3d blocks x %2dw: %7.4f ms  %6.1f GB/s\n", "g9 ru1", (N + 63) / 64, 8, ms, bytes / (ms * 1e6));
        ms = bench([&]{ g9<8, 4><<<(N + 63) / 64, 256, K>>>(dy, dW, dxq, dxd, dxsum, K, N); }, iters);
        printf("%-28s %3d blocks x %2dw: %7.4f ms  %6.1f GB/s\n", "g9 ru4", (N + 63) / 64, 8, ms, bytes / (ms * 1e6));
    }
    RUNQ ("g9 64val/lane w4",    g9, 4, (N + 31) / 32);
    RUNQ ("g9 64val/lane w16",   g9, 16, (N + 127) / 128);
    RUNQ ("g9 64val/lane w8 b224", g9, 8, 224);
    RUNQ ("g9 64val/lane w8 b448", g9, 8, 448);
    RUNQ ("g9 64val/lane w8 b56",  g9, 8, 56);
    RUNQ ("g11 1row/warp w8",    g11, 8, (N + 7) / 8);
    RUNQ ("g11 1row/warp w4",    g11, 4, (N + 3) / 4);
    RUNQ ("g11 1row/warp w16",   g11, 16, (N + 15) / 16);
    RUNQ ("g11 1row/warp w8 b448", g11, 8, 448);
    RUNQ ("g11 1row/warp w8 b112", g11, 8, 112);
    RUNQ ("g11 1row/warp w4 b448", g11, 4, 448);
    return 0;
}
