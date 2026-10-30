// Batched (multi-row) quantized GEMM microbenchmark.
// Variant A: activations staged in shared (padded 9-word layout, conflict-free LDS.32).
// Variant B: activations read straight from global memory (L1-resident rows).
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e) { printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

static const int KMAX_WORDS = 4096;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ void q5k_unpack_sc(uint32_t w0, uint32_t w1, uint32_t w2, int j,
                                              float d, float dmin, float* ds, float* dm) {
    const uint32_t kmask1 = 0x3f3f3f3fu, kmask2 = 0x0f0f0f0fu, kmask3 = 0x03030303u;
    uint32_t utmp[2];
    utmp[0] = w1; utmp[1] = w2;
    uint32_t sc, m;
    if (j < 4) {
        sc = w0 >> (8 * j);
        m  = w0 >> (8 * j + 16) | 16;
        utmp[0] >>= (8 * j);
        utmp[1] >>= (8 * j);
    } else {
        sc = w2 >> (8 * (j - 4));
        m  = w1 >> (8 * (j - 4)) | 16;
    }
    sc &= 63; m &= 63;
    uint32_t mm = (utmp[1] & kmask3) << 4 | (utmp[0] & kmask2);
    mm = (mm & kmask1) | ((m & kmask1) << 16);
    utmp[1] = utmp[1] >> 6 | utmp[0] << 10;
    utmp[0] = utmp[0] >> 6;
    mm |= (utmp[1] & kmask3) << 6 | (utmp[0] & kmask2) << 2;
    *ds = d * (float)(sc & 0xFF);
    *dm = dmin * (float)(mm & 0xFF);
}

// ---- variant A: shared staging
template <int M, int WARPS>
__global__ void k_gemmA(float* __restrict__ y, const uint8_t* __restrict__ W,
                        const int8_t* __restrict__ xq, const float* __restrict__ xd,
                        const int* __restrict__ xsum, int K, int N) {
    extern __shared__ int sxi[];
    const int nw = K >> 2;
    const int nsb = K >> 8;
    const int nwp = nsb * 72;
    const int total = nw * M;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        const int m = i / nw, wi = i - m * nw;
        const int sb = wi >> 6, u = wi & 63;
        const int js = 2 * ((u >> 3) & 3) + (u >> 5), q = u & 7;
        sxi[m * nwp + (sb >> 2) * 288 + ((sb & 3) * 8 + js) * 9 + q] =
            *(const int*)(xq + (size_t)m * K + sb * 256 + js * 32 + q * 4);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int js = lane & 7, sl = lane >> 3;
    const int sbh = js >> 1;
    const bool nibl = js & 1;
    const int xgrp = (sl * 8 + js) * 9;
    const int stride = gridDim.x * WARPS;
    for (int row = blockIdx.x * WARPS + warp; row < N; row += stride) {
        const uint8_t* rp = W + (size_t)row * (size_t)nsb * 176;
        float acc[M];
        #pragma unroll
        for (int m = 0; m < M; ++m) acc[m] = 0.f;
        for (int sb0 = 0; sb0 < nsb; sb0 += 4) {
            const int nch = min(4, nsb - sb0);
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
                int dot[M];
                #pragma unroll
                for (int m = 0; m < M; ++m) dot[m] = 0;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t nib = nibl ? ((qsl[i] >> 4) & 0x0F0F0F0Fu) : (qsl[i] & 0x0F0F0F0Fu);
                    const int qw = (int)(nib | (((qhl[i] >> js) & 0x01010101u) << 4));
                    #pragma unroll
                    for (int m = 0; m < M; ++m)
                        dot[m] = __dp4a(sxi[m * nwp + (sb0 >> 2) * 288 + xgrp + i], qw, dot[m]);
                }
                const uint32_t w0s = *(const uint32_t*)(b + 4);
                const uint32_t w1s = *(const uint32_t*)(b + 8);
                const uint32_t w2s = *(const uint32_t*)(b + 12);
                float ds, dm;
                q5k_unpack_sc(w0s, w1s, w2s, js, __half2float(*(const __half*)(b)),
                              __half2float(*(const __half*)(b + 2)), &ds, &dm);
                const int isb = sb * 8 + js;
                const int nb32 = K >> 5;
                #pragma unroll
                for (int m = 0; m < M; ++m)
                    acc[m] += xd[m * nb32 + isb] * (ds * (float)dot[m] - dm * (float)xsum[m * nb32 + isb]);
            }
        }
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            float s = warp_sum(acc[m]);
            if (lane == 0) y[(size_t)m * N + row] = s;
        }
    }
}

// ---- variant B: activations straight from global
template <int M, int WARPS>
__global__ void k_gemmB(float* __restrict__ y, const uint8_t* __restrict__ W,
                        const int8_t* __restrict__ xq, const float* __restrict__ xd,
                        const int* __restrict__ xsum, int K, int N) {
    const int nsb = K >> 8;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int js = lane & 7, sl = lane >> 3;
    const int sbh = js >> 1;
    const bool nibl = js & 1;
    const int xw = (js >> 1) * 8 + (js & 1) * 32;    // word offset inside the 256-value block
    const int stride = gridDim.x * WARPS;
    for (int row = blockIdx.x * WARPS + warp; row < N; row += stride) {
        const uint8_t* rp = W + (size_t)row * (size_t)nsb * 176;
        float acc[M];
        #pragma unroll
        for (int m = 0; m < M; ++m) acc[m] = 0.f;
        for (int sb0 = 0; sb0 < nsb; sb0 += 4) {
            const int nch = min(4, nsb - sb0);
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
                int dot[M];
                #pragma unroll
                for (int m = 0; m < M; ++m) dot[m] = 0;
                const int8_t* xr = xq + (size_t)sb * 256 + xw * 4;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t nib = nibl ? ((qsl[i] >> 4) & 0x0F0F0F0Fu) : (qsl[i] & 0x0F0F0F0Fu);
                    const int qw = (int)(nib | (((qhl[i] >> js) & 0x01010101u) << 4));
                    #pragma unroll
                    for (int m = 0; m < M; ++m)
                        dot[m] = __dp4a(*(const int*)(xr + (size_t)m * K + 4 * i), qw, dot[m]);
                }
                const uint32_t w0s = *(const uint32_t*)(b + 4);
                const uint32_t w1s = *(const uint32_t*)(b + 8);
                const uint32_t w2s = *(const uint32_t*)(b + 12);
                float ds, dm;
                q5k_unpack_sc(w0s, w1s, w2s, js, __half2float(*(const __half*)(b)),
                              __half2float(*(const __half*)(b + 2)), &ds, &dm);
                const int isb = sb * 8 + js;
                const int nb32 = K >> 5;
                #pragma unroll
                for (int m = 0; m < M; ++m)
                    acc[m] += xd[m * nb32 + isb] * (ds * (float)dot[m] - dm * (float)xsum[m * nb32 + isb]);
            }
        }
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            float s = warp_sum(acc[m]);
            if (lane == 0) y[(size_t)m * N + row] = s;
        }
    }
}

// ---- variant C: like A but rows outermost, 8 x-loads batched before 8 dp4a
template <int M, int WARPS>
__global__ void k_gemmC(float* __restrict__ y, const uint8_t* __restrict__ W,
                        const int8_t* __restrict__ xq, const float* __restrict__ xd,
                        const int* __restrict__ xsum, int K, int N) {
    extern __shared__ int sxi[];
    const int nw = K >> 2;
    const int nsb = K >> 8;
    const int nwp = nsb * 72;
    const int total = nw * M;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        const int m = i / nw, wi = i - m * nw;
        const int sb = wi >> 6, u = wi & 63;
        const int js = 2 * ((u >> 3) & 3) + (u >> 5), q = u & 7;
        sxi[m * nwp + (sb >> 2) * 288 + ((sb & 3) * 8 + js) * 9 + q] =
            *(const int*)(xq + (size_t)m * K + sb * 256 + js * 32 + q * 4);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int js = lane & 7, sl = lane >> 3;
    const int sbh = js >> 1;
    const bool nibl = js & 1;
    const int xgrp = (sl * 8 + js) * 9;
    const int stride = gridDim.x * WARPS;
    for (int row = blockIdx.x * WARPS + warp; row < N; row += stride) {
        const uint8_t* rp = W + (size_t)row * (size_t)nsb * 176;
        float acc[M];
        #pragma unroll
        for (int m = 0; m < M; ++m) acc[m] = 0.f;
        for (int sb0 = 0; sb0 < nsb; sb0 += 4) {
            const int nch = min(4, nsb - sb0);
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
                int qw[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t nib = nibl ? ((qsl[i] >> 4) & 0x0F0F0F0Fu) : (qsl[i] & 0x0F0F0F0Fu);
                    qw[i] = (int)(nib | (((qhl[i] >> js) & 0x01010101u) << 4));
                }
                const uint32_t w0s = *(const uint32_t*)(b + 4);
                const uint32_t w1s = *(const uint32_t*)(b + 8);
                const uint32_t w2s = *(const uint32_t*)(b + 12);
                float ds, dm;
                q5k_unpack_sc(w0s, w1s, w2s, js, __half2float(*(const __half*)(b)),
                              __half2float(*(const __half*)(b + 2)), &ds, &dm);
                const int isb = sb * 8 + js;
                const int nb32 = K >> 5;
                const int* xb = sxi + (sb0 >> 2) * 288 + xgrp;
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    float part = 0.f;
                    int dot = 0;
                    const int* xm = xb + m * nwp;
                    int xw[8];
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) xw[i] = xm[i];
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) dot = __dp4a(xw[i], qw[i], dot);
                    part = xd[m * nb32 + isb] * (ds * (float)dot - dm * (float)xsum[m * nb32 + isb]);
                    acc[m] += part;
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            float s = warp_sum(acc[m]);
            if (lane == 0) y[(size_t)m * N + row] = s;
        }
    }
}

// ---- variant E: R rows unrolled so the staged activations are loaded once per R rows
template <int M, int W, int NR>
__global__ void k_gemmE(float* __restrict__ y, const uint8_t* __restrict__ Wt, const int8_t* __restrict__ xq, const float* __restrict__ xd, const int* __restrict__ xsum, int K, int N) {
    extern __shared__ int sxi[];
    const int nw = K >> 2;
    const int nsb = K >> 8;
    const int nwp = nsb * 72;
    const int total = nw * M;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        const int m = i / nw, wi = i - m * nw;
        const int sb = wi >> 6, u = wi & 63;
        const int js = 2 * ((u >> 3) & 3) + (u >> 5), q = u & 7;
        sxi[m * nwp + (sb >> 2) * 288 + ((sb & 3) * 8 + js) * 9 + q] =
            *(const int*)(xq + (size_t)m * K + sb * 256 + js * 32 + q * 4);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int js = lane & 7, sl = lane >> 3;
    const int sbh = js >> 1;
    const bool nibl = js & 1;
    const int xgrp = (sl * 8 + js) * 9;
    const int stride = gridDim.x * W * NR;
    for (int row0 = (blockIdx.x * W + warp) * NR; row0 < N; row0 += stride) {
        float acc[NR][M];
        #pragma unroll
        for (int r = 0; r < NR; ++r)
            #pragma unroll
            for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
        for (int sb0 = 0; sb0 < nsb; sb0 += 4) {
            const int nch = min(4, nsb - sb0);
            if (sl < nch) {
                const int sb = sb0 + sl;
                // staged activations for this superblock group, once per R rows
                int xw[M][8];
                const int* xb = sxi + (sb0 >> 2) * 288 + xgrp;
                #pragma unroll
                for (int m = 0; m < M; ++m)
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) xw[m][i] = xb[m * nwp + i];
                const int isb = sb * 8 + js;
                const int nb32 = K >> 5;
                #pragma unroll
                for (int r = 0; r < NR; ++r) {
                    const int row = row0 + r;
                    if (row >= N) break;
                    const uint8_t* b = Wt + (size_t)row * (size_t)nsb * 176 + sb * 176;
                    const int4 qa = *(const int4*)(b + 48 + 32 * sbh);
                    const int4 qb = *(const int4*)(b + 64 + 32 * sbh);
                    const int4 ha = *(const int4*)(b + 16);
                    const int4 hb = *(const int4*)(b + 32);
                    const uint32_t qsl[8] = {(uint32_t)qa.x, (uint32_t)qa.y, (uint32_t)qa.z, (uint32_t)qa.w,
                                             (uint32_t)qb.x, (uint32_t)qb.y, (uint32_t)qb.z, (uint32_t)qb.w};
                    const uint32_t qhl[8] = {(uint32_t)ha.x, (uint32_t)ha.y, (uint32_t)ha.z, (uint32_t)ha.w,
                                             (uint32_t)hb.x, (uint32_t)hb.y, (uint32_t)hb.z, (uint32_t)hb.w};
                    int qw[8];
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const uint32_t nib = nibl ? ((qsl[i] >> 4) & 0x0F0F0F0Fu) : (qsl[i] & 0x0F0F0F0Fu);
                        qw[i] = (int)(nib | (((qhl[i] >> js) & 0x01010101u) << 4));
                    }
                    const uint32_t w0s = *(const uint32_t*)(b + 4);
                    const uint32_t w1s = *(const uint32_t*)(b + 8);
                    const uint32_t w2s = *(const uint32_t*)(b + 12);
                    float ds, dm;
                    q5k_unpack_sc(w0s, w1s, w2s, js, __half2float(*(const __half*)(b)),
                                  __half2float(*(const __half*)(b + 2)), &ds, &dm);
                    #pragma unroll
                    for (int m = 0; m < M; ++m) {
                        int dot = 0;
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) dot = __dp4a(xw[m][i], qw[i], dot);
                        acc[r][m] += xd[m * nb32 + isb] * (ds * (float)dot - dm * (float)xsum[m * nb32 + isb]);
                    }
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < NR; ++r) {
            const int row = row0 + r;
            if (row >= N) break;
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                float s = warp_sum(acc[r][m]);
                if (lane == 0) y[(size_t)m * N + row] = s;
            }
        }
    }
}

static float bench(void (*launch)(float*, const uint8_t*, const int8_t*, const float*, const int*, int, int,
                                  int, int, cudaStream_t),
                   float* y, const uint8_t* W, const int8_t* xq, const float* xd, const int* xsum,
                   int K, int N, int M, int blocks, int warps, int iters) {
    launch(y, W, xq, xd, xsum, K, N, M, blocks, 0);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch(y, W, xq, xd, xsum, K, N, M, blocks, (cudaStream_t)0);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}

static int g_R = 4;

template <int M, int W>
static void la(char* variant, float* y, const uint8_t* Wt, const int8_t* xq, const float* xd,
               const int* xsum, int K, int N, int, int blocks, cudaStream_t) {
    int nsb = K >> 8, nw = K >> 2;
    size_t sm = (size_t)nsb * 72 * M * 4;
    int want = (int)(sm > 1024 ? sm + 1024 : 48 * 1024);
    if (variant[0] == 'A') {
        CK(cudaFuncSetAttribute((const void*)&k_gemmA<M, W>, cudaFuncAttributeMaxDynamicSharedMemorySize, want));
        k_gemmA<M, W><<<blocks, 32 * W, sm>>>(y, Wt, xq, xd, xsum, K, N);
        CK(cudaGetLastError());
    } else if (variant[0] == 'C') {
        CK(cudaFuncSetAttribute((const void*)&k_gemmC<M, W>, cudaFuncAttributeMaxDynamicSharedMemorySize, want));
        k_gemmC<M, W><<<blocks, 32 * W, sm>>>(y, Wt, xq, xd, xsum, K, N);
        CK(cudaGetLastError());
    } else if (variant[0] == 'E') {
        #define LE(R) do { \
            CK(cudaFuncSetAttribute((const void*)&k_gemmE<M, W, R>, cudaFuncAttributeMaxDynamicSharedMemorySize, want)); \
            k_gemmE<M, W, R><<<blocks, 32 * W, sm>>>(y, Wt, xq, xd, xsum, K, N); \
            CK(cudaGetLastError()); } while (0)
        if (g_R == 1) LE(1); else if (g_R == 2) LE(2); else if (g_R == 4) LE(4); else LE(8);
        #undef LE
    } else {
        k_gemmB<M, W><<<blocks, 32 * W, 0>>>(y, Wt, xq, xd, xsum, K, N);
    }
    (void)nw;
}

template <int M>
static void dispatch(char variant, int warps, float* y, const uint8_t* Wt, const int8_t* xq,
                     const float* xd, const int* xsum, int K, int N, int blocks) {
    char v[2] = {variant, 0};
    if (warps == 4) la<M, 4>(v, y, Wt, xq, xd, xsum, K, N, 0, blocks, 0);
    else if (warps == 8) la<M, 8>(v, y, Wt, xq, xd, xsum, K, N, 0, blocks, 0);
    else if (warps == 16) la<M, 16>(v, y, Wt, xq, xd, xsum, K, N, 0, blocks, 0);
    else if (warps == 32) la<M, 32>(v, y, Wt, xq, xd, xsum, K, N, 0, blocks, 0);
}

static double gbytes(int K, int N) { return (double)(K >> 8) * 176.0 * N; }

int main(int argc, char** argv) {
    int K = argc > 1 ? atoi(argv[1]) : 7168;
    int N = argc > 2 ? atoi(argv[2]) : 2560;
    int M = argc > 3 ? atoi(argv[3]) : 8;
    int blocks = argc > 4 ? atoi(argv[4]) : 28;
    int warps = argc > 5 ? atoi(argv[5]) : 8;
    char variant = argc > 6 ? argv[6][0] : 'A';
    int iters = argc > 7 ? atoi(argv[7]) : 20;
    if (argc > 8) g_R = atoi(argv[8]);

    const int nsb = K >> 8;
    size_t wb = (size_t)nsb * 176 * N;
    uint8_t* dW; CK(cudaMalloc(&dW, wb));
    int8_t* dxq; CK(cudaMalloc(&dxq, (size_t)M * K));
    float* dxd; CK(cudaMalloc(&dxd, (size_t)M * (K / 32) * 4));
    int* dxsum; CK(cudaMalloc(&dxsum, (size_t)M * (K / 32) * 4));
    float* dy; CK(cudaMalloc(&dy, (size_t)M * N * 4));
    uint8_t* hW = (uint8_t*)malloc(wb);
    // structured pseudo-random q5_K blocks: d in [0.002,0.02], scales small, q random, min random
    srand(1234);
    for (size_t blk = 0; blk < wb / 176; ++blk) {
        uint8_t* b = hW + blk * 176;
        for (int i = 0; i < 176; ++i) b[i] = (uint8_t)(rand() & 0xFF);
        float d = 0.002f + 0.018f * (rand() / (float)RAND_MAX);
        float dmin = 0.001f + 0.004f * (rand() / (float)RAND_MAX);
        __half hd = __float2half(d), hm = __float2half(dmin);
        memcpy(b, &hd, 2); memcpy(b + 2, &hm, 2);
    }
    CK(cudaMemcpy(dW, hW, wb, cudaMemcpyHostToDevice));
    int8_t* hx = (int8_t*)malloc((size_t)M * K);
    float* hxd = (float*)malloc((size_t)M * (K / 32) * 4);
    int* hxs = (int*)malloc((size_t)M * (K / 32) * 4);
    for (size_t i = 0; i < (size_t)M * K; ++i) hx[i] = (int8_t)((rand() % 255) - 127);
    for (int m = 0; m < M; ++m)
        for (int i = 0; i < K / 32; ++i) {
            hxd[m * (K / 32) + i] = 0.0005f + 0.0005f * (rand() / (float)RAND_MAX);
            hxs[m * (K / 32) + i] = rand() % 4000 - 2000;
        }
    CK(cudaMemcpy(dxq, hx, (size_t)M * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxd, hxd, (size_t)M * (K / 32) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxsum, hxs, (size_t)M * (K / 32) * 4, cudaMemcpyHostToDevice));

    // timing via template dispatch
    float ms = 0;
    #define RUN(MM) do { \
        if (M == MM) { \
            auto fn = [&](int it) { dispatch<MM>(variant, warps, dy, dW, dxq, dxd, dxsum, K, N, blocks); return 0; }; \
            fn(1); CK(cudaDeviceSynchronize()); \
            cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); \
            CK(cudaEventRecord(a)); for (int i = 0; i < iters; ++i) fn(1); CK(cudaEventRecord(b)); \
            CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms, a, b)); ms /= iters; \
            cudaEventDestroy(a); cudaEventDestroy(b); \
        } } while (0)
    RUN(1); RUN(2); RUN(4); RUN(8);
    #undef RUN
    double gb = gbytes(K, N);
    printf("K=%5d N=%5d M=%d var=%c blocks=%3d warps=%2d R=%d: %7.3f ms  %6.1f GB/s\n",
           K, N, M, variant, blocks, warps, g_R, ms, gb / ms / 1e6);

    // correctness vs variant A (validated against the engine)
    if (variant != 'A') {
        float* dyref; CK(cudaMalloc(&dyref, (size_t)M * N * 4));
        char va[2] = {'A', 0};
        #define RUNREF(MV) if (M == MV) dispatch<MV>(va[0], warps, dyref, dW, dxq, dxd, dxsum, K, N, blocks)
        RUNREF(1); RUNREF(2); RUNREF(4); RUNREF(8);
        #undef RUNREF
        CK(cudaDeviceSynchronize());
        float* h1 = (float*)malloc((size_t)M * N * 4);
        float* h2 = (float*)malloc((size_t)M * N * 4);
        CK(cudaMemcpy(h1, dy, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h2, dyref, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
        double mx = 0, rms = 0;
        for (size_t i = 0; i < (size_t)M * N; ++i) {
            double d = fabs((double)h1[i] - h2[i]);
            if (d > mx) mx = d;
            rms += d * d;
        }
        rms = sqrt(rms / ((double)M * N));
        printf("   check vs A: maxabs %.4e  rms %.4e\n", mx, rms);
        free(h1); free(h2); cudaFree(dyref);
    }
    return 0;
}
