// Tiled s8 tensor-core GEMM for Q5_K weights.  BM=64 activation rows, BN=64 weight
// rows, one 256-wide Q5_K superblock per K step.  A (int8 activations) and B (weights
// unpacked to int8) are staged in shared memory; the per-32-block scales are applied
// on the int32 MMA result, so the arithmetic is the same int8 dot + fp32 scaled sum
// as the dp4a path (only the summation order differs).
//
// Standalone dev harness: correctness vs a CPU reference, then T-MAC/s.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

#define BKP 272          // padded row stride (bytes) of the staged int8 tiles

__host__ __device__ __forceinline__ void q5k_scale(uint32_t w0, uint32_t w1, uint32_t w2, int js,
                                          float d, float dmin, float* ds, float* dm) {
    uint32_t sv, mv;
    if (js < 4) {
        sv = (w0 >> (8 * js)) & 63;
        mv = (w1 >> (8 * js)) & 63;
    } else {
        const uint32_t lo = (w2 >> (8 * (js - 4))) & 0xFF;
        sv = (lo & 0xF) | ((((w0 >> (8 * (js - 4))) & 0xFF) >> 6) << 4);
        mv = (lo >> 4)   | ((((w1 >> (8 * (js - 4))) & 0xFF) >> 6) << 4);
    }
    *ds = d * (float)sv;
    *dm = dmin * (float)mv;
}

__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// y[m*N + n] = sum_kb xd[m][kb]*(ds[n][kb]*dot - dm[n][kb]*xsum[m][kb])
template <int MODE>
__global__ void __launch_bounds__(256, 2)
k_gemm_mma_q5k(float* __restrict__ y, const uint8_t* __restrict__ W,
               const int8_t* __restrict__ xq, const float* __restrict__ xd,
               const int* __restrict__ xsum, int K, int N, int rows) {
    extern __shared__ __align__(16) int8_t sm[];
    int8_t* sA  = sm;                       // [64][BKP] activations
    int8_t* sB  = (MODE == 6) ? sm : sA + 64 * BKP;   // [64][BKP] unpacked weights
    float*  sAd = (float*)(sB + 64 * BKP);  // [64][8] activation scale
    float*  sAx = sAd + 512;                // [64][8] xd*xsum
    float*  sBd = sAx + 512;                // [64][8] weight scale
    float*  sBm = sBd + 512;                // [64][8] weight min

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int grp = lane >> 2, tg = lane & 3;
    const int wm = warp >> 2;               // 0..1 -> 32 activation rows
    const int wn = warp & 3;                // 0..3 -> 16 weight rows
    const int nsb = K >> 8;
    const int nbase = blockIdx.x * 64;      // weight row tile
    const int row0 = blockIdx.y * 64;       // activation row tile

    float acc[2][2][4];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.f;

    for (int sb = 0; sb < nsb; ++sb) {
        if (MODE < 3 || MODE == 4 || MODE == 6) {
            __syncthreads();
            // ---- stage 64 x 256 int8 activations (int4 copies) + scales
            if (MODE != 6)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int t = tid + i * 256;         // 1024 int4 = 64 rows * 16
                const int m = t >> 4, q4 = t & 15;
                const int gm = row0 + m;
                int4 v = make_int4(0, 0, 0, 0);
                if (gm < rows) v = *(const int4*)(xq + (size_t)gm * K + (sb << 8) + (q4 << 4));
                *(int4*)(sA + m * BKP + (q4 << 4)) = v;
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int t = tid + i * 256;
                const int m = t >> 3, b = t & 7;
                const int gm = row0 + m;
                const int kb = (sb << 3) + b;
                float dv = 0.f; int sv = 0;
                if (gm < rows) { dv = xd[(size_t)gm * (K >> 5) + kb]; sv = xsum[(size_t)gm * (K >> 5) + kb]; }
                sAd[m * 8 + b] = dv;
                sAx[m * 8 + b] = dv * (float)sv;
            }
            // ---- stage + unpack 64 weight rows x 256 values, coalesced: one thread per
            // (row, 16-byte chunk of the nibble stream); consecutive lanes cover one row.
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int t = tid + i * 256;
                const int n = t >> 3, ci = t & 7;          // ci: 8 chunks of 16 B over the 128-B qs
                const int pair = ci >> 1, half = ci & 1;
                const uint8_t* wb = W + (size_t)(nbase + n) * (size_t)nsb * 176 + (size_t)sb * 176;
                const int4 qsv = *(const int4*)(wb + 48 + 32 * pair + 16 * half);
                const int4 qhv = *(const int4*)(wb + 16 + 16 * half);
                const uint32_t* qs = (const uint32_t*)&qsv;
                const uint32_t* qh = (const uint32_t*)&qhv;
                uint32_t v0[4], v1[4];
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    v0[u] = (qs[u] & 0x0F0F0F0Fu) | (((qh[u] >> (2 * pair)) & 0x01010101u) << 4);
                    v1[u] = ((qs[u] >> 4) & 0x0F0F0F0Fu) | (((qh[u] >> (2 * pair + 1)) & 0x01010101u) << 4);
                }
                int8_t* db = sB + (size_t)n * BKP + (size_t)(2 * pair) * 32 + 16 * half;
                *(int4*)(db) = *(const int4*)v0;
                *(int4*)(db + 32) = *(const int4*)v1;
            }
            // ---- weight scales: 2 sub-blocks per thread
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int t = tid + i * 256;
                const int n = t >> 3, b = t & 7;
                const uint8_t* wb = W + (size_t)(nbase + n) * (size_t)nsb * 176 + (size_t)sb * 176;
                float ds, dm;
                q5k_scale(*(const uint32_t*)(wb + 4), *(const uint32_t*)(wb + 8), *(const uint32_t*)(wb + 12),
                          b, __half2float(*(const __half*)(wb)), __half2float(*(const __half*)(wb + 2)), &ds, &dm);
                sBd[n * 8 + b] = ds;
                sBm[n * 8 + b] = dm;
            }
            __syncthreads();
        }

        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            const int kb = (sb << 3) + b;
            uint32_t a[2][4], bf[2][2];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                if (MODE == 6) {
                    const int8_t* ap = xq + (size_t)(row0 + wm * 32 + mt * 16 + grp) * K + (sb << 8) + (b << 5) + tg * 4;
                    a[mt][0] = *(const uint32_t*)(ap);
                    a[mt][1] = *(const uint32_t*)(ap + (size_t)8 * K);
                    a[mt][2] = *(const uint32_t*)(ap + 16);
                    a[mt][3] = *(const uint32_t*)(ap + (size_t)8 * K + 16);
                } else {
                    const int8_t* ap = sA + (wm * 32 + mt * 16 + grp) * BKP + (b << 5) + tg * 4;
                    a[mt][0] = *(const uint32_t*)(ap);
                    a[mt][1] = *(const uint32_t*)(ap + 8 * BKP);
                    a[mt][2] = *(const uint32_t*)(ap + 16);
                    a[mt][3] = *(const uint32_t*)(ap + 8 * BKP + 16);
                }
            }
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int8_t* bp = sB + (wn * 16 + nt * 8 + grp) * BKP + (b << 5) + tg * 4;
                bf[nt][0] = *(const uint32_t*)(bp);
                bf[nt][1] = *(const uint32_t*)(bp + 16);
            }
            // per-lane scale operands (2 rows x 2 m-tiles, 2 cols x 2 n-tiles)
            const int r0 = wm * 32 + grp, r1 = r0 + 8;          // mt = 0
            const int r2 = r0 + 16, r3 = r0 + 24;               // mt = 1
            const int c0 = wn * 16 + tg * 2;                     // nt = 0 -> cols c0, c0+1
            const int c1 = c0 + 8;                               // nt = 1
            float xd0 = sAd[r0 * 8 + b], xd1 = sAd[r1 * 8 + b], xd2 = sAd[r2 * 8 + b], xd3 = sAd[r3 * 8 + b];
            float xs0 = sAx[r0 * 8 + b], xs1 = sAx[r1 * 8 + b], xs2 = sAx[r2 * 8 + b], xs3 = sAx[r3 * 8 + b];
            float bd00 = sBd[c0 * 8 + b], bd01 = sBd[(c0 + 1) * 8 + b];
            float bm00 = sBm[c0 * 8 + b], bm01 = sBm[(c0 + 1) * 8 + b];
            float bd10 = sBd[c1 * 8 + b], bd11 = sBd[(c1 + 1) * 8 + b];
            float bm10 = sBm[c1 * 8 + b], bm11 = sBm[(c1 + 1) * 8 + b];
            const float xdv[2][2] = {{xd0, xd1}, {xd2, xd3}};
            const float xsv[2][2] = {{xs0, xs1}, {xs2, xs3}};
            const float bdv[2][2] = {{bd00, bd01}, {bd10, bd11}};
            const float bmv[2][2] = {{bm00, bm01}, {bm10, bm11}};
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt)
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    int c[4] = {0, 0, 0, 0};
                    mma_s8(c, a[mt], bf[nt]);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int rr = i >> 1, cc = i & 1;      // c0,c1 -> row rA; c2,c3 -> row rB
                        if (MODE == 0 || MODE == 5)
                            acc[mt][nt][i] += (float)c[i] * xdv[mt][rr] * bdv[nt][cc]
                                              - bmv[nt][cc] * xsv[mt][rr];
                        else if (MODE == 1)
                            acc[mt][nt][i] += (float)c[i] * xdv[mt][rr] * bdv[nt][cc];
                        else
                            acc[mt][nt][i] += (float)c[i];
                    }
                }
        }
    }

    // ---- store: 32 rows x 16 cols per warp
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int ra = row0 + wm * 32 + mt * 16 + grp;
            const int cb = nbase + wn * 16 + nt * 8 + tg * 2;
            if (ra < rows)     *(float2*)(y + (size_t)ra * N + cb) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
            if (ra + 8 < rows) *(float2*)(y + (size_t)(ra + 8) * N + cb) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
        }
}

// ---------------------------------------------------------------------------
static void fill_q5k(uint8_t* p, int N, int K, unsigned seed) {
    const int nsb = K >> 8;
    srand(seed);
    for (int n = 0; n < N; ++n)
        for (int s = 0; s < nsb; ++s) {
            uint8_t* b = p + ((size_t)n * nsb + s) * 176;
            *(uint16_t*)(b) = 0x1c00;            // ~2^-14 half
            *(uint16_t*)(b + 2) = 0x1400;        // ~2^-15 half
            for (int i = 4; i < 176; ++i) b[i] = (uint8_t)(rand() & 0xFF);
        }
}

static void ref_gemm(const uint8_t* W, const int8_t* xq, const float* xd, const int* xsum,
                     int K, int N, int M, float* y) {
    const int nsb = K >> 8, nb = K >> 5;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int sb = 0; sb < nsb; ++sb) {
                const uint8_t* wb = W + ((size_t)n * nsb + sb) * 176;
                const float d = __half2float(*(const __half*)wb), dmin = __half2float(*(const __half*)(wb + 2));
                for (int j = 0; j < 8; ++j) {
                    float ds, dm;
                    q5k_scale(*(const uint32_t*)(wb + 4), *(const uint32_t*)(wb + 8), *(const uint32_t*)(wb + 12),
                              j, d, dmin, &ds, &dm);
                    const int kb = sb * 8 + j;
                    int dot = 0;
                    for (int o = 0; o < 32; ++o) {
                        const int u = ((wb[48 + 32 * (j >> 1) + o] >> (4 * (j & 1))) & 0xF)
                                    | (((wb[16 + o] >> j) & 1) << 4);
                        dot += u * (int)xq[(size_t)m * K + kb * 32 + o];
                    }
                    acc += xd[(size_t)m * nb + kb] * (ds * (float)dot - dm * (float)xsum[(size_t)m * nb + kb]);
                }
            }
            y[(size_t)m * N + n] = acc;
        }
}

int main(int argc, char** argv) {
    const int K = argc > 1 ? atoi(argv[1]) : 7168;
    const int N = argc > 2 ? atoi(argv[2]) : 7168;
    const int M = 64;
    const int nsb = K >> 8, nb = K >> 5;

    // ---- correctness on a small case
    {
        const int kK = 512, kN = 64, kM = 64, knsb = kK >> 8, knb = kK >> 5;
        uint8_t* hW = (uint8_t*)malloc((size_t)kN * knsb * 176);
        int8_t* hx = (int8_t*)malloc((size_t)kM * kK);
        float* hxd = (float*)malloc((size_t)kM * knb * 4);
        int* hxs = (int*)malloc((size_t)kM * knb * 4);
        float* hy = (float*)malloc((size_t)kM * kN * 4);
        float* hr = (float*)malloc((size_t)kM * kN * 4);
        fill_q5k(hW, kN, kK, 7);
        srand(11);
        for (int i = 0; i < kM * kK; ++i) hx[i] = (int8_t)(rand() % 255 - 127);
        for (int i = 0; i < kM * knb; ++i) {
            hxd[i] = 0.001f + 0.0005f * (rand() % 100) / 100.f;
            int s = 0;
            for (int o = 0; o < 32; ++o) s += hx[(size_t)(i / knb) * kK + i % knb * 32 + o];
            hxs[i] = s;
        }
        uint8_t *dW; int8_t* dx; float* dxd; int* dxs; float* dy;
        CK(cudaMalloc(&dW, (size_t)kN * knsb * 176)); CK(cudaMalloc(&dx, (size_t)kM * kK));
        CK(cudaMalloc(&dxd, (size_t)kM * knb * 4)); CK(cudaMalloc(&dxs, (size_t)kM * knb * 4));
        CK(cudaMalloc(&dy, (size_t)kM * kN * 4));
        CK(cudaMemcpy(dW, hW, (size_t)kN * knsb * 176, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, hx, (size_t)kM * kK, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dxd, hxd, (size_t)kM * knb * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dxs, hxs, (size_t)kM * knb * 4, cudaMemcpyHostToDevice));
        const size_t sm = 2 * 64 * BKP + 4 * 512 * 4;
        CK(cudaFuncSetAttribute((const void*)&k_gemm_mma_q5k<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
        ref_gemm(hW, hx, hxd, hxs, kK, kN, kM, hr);
        cudaMemset(dy, 0, (size_t)kM * kN * 4);
        dim3 g(kN / 64, (kM + 63) / 64);
        k_gemm_mma_q5k<0><<<g, 256, sm>>>(dy, dW, dx, dxd, dxs, kK, kN, kM);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hy, dy, (size_t)kM * kN * 4, cudaMemcpyDeviceToHost));
        double maxrel = 0, maxabs = 0, mag = 0;
        for (int i = 0; i < kM * kN; ++i) {
            maxabs = fmax(maxabs, fabs(hy[i] - hr[i]));
            mag = fmax(mag, fabs(hr[i]));
            maxrel = fmax(maxrel, fabs(hy[i] - hr[i]) / (1e-6 + fabs(hr[i])));
        }
        printf("correctness: max |diff| %.3e  (values up to %.3e)  max rel %.3e  %s\n",
               maxabs, mag, maxrel, maxabs < 1e-3 * mag + 1e-6 ? "OK" : "FAIL");
        if (getenv("DBG")) {
            for (int m = 0; m < 2; ++m) {
                printf("m=%d gpu:", m);
                for (int n = 0; n < 8; ++n) printf(" %8.4f", hy[m * kN + n]);
                printf("\n     cpu:");
                for (int n = 0; n < 8; ++n) printf(" %8.4f", hr[m * kN + n]);
                printf("\n");
            }
        }
        cudaFree(dW); cudaFree(dx); cudaFree(dxd); cudaFree(dxs); cudaFree(dy);
        free(hW); free(hx); free(hxd); free(hxs); free(hy); free(hr);
    }

    // ---- benchmark
    {
        uint8_t* hW = (uint8_t*)malloc((size_t)N * nsb * 176);
        int8_t* hx = (int8_t*)malloc((size_t)M * K);
        float* hxd = (float*)malloc((size_t)M * nb * 4);
        int* hxs = (int*)malloc((size_t)M * nb * 4);
        fill_q5k(hW, N, K, 3);
        srand(5);
        for (int i = 0; i < M * K; ++i) hx[i] = (int8_t)(rand() % 255 - 127);
        for (int i = 0; i < M * nb; ++i) { hxd[i] = 0.001f; hxs[i] = 13 * (i % 7) - 40; }
        uint8_t *dW; int8_t* dx; float* dxd; int* dxs; float* dy;
        CK(cudaMalloc(&dW, (size_t)N * nsb * 176)); CK(cudaMalloc(&dx, (size_t)M * K));
        CK(cudaMalloc(&dxd, (size_t)M * nb * 4)); CK(cudaMalloc(&dxs, (size_t)M * nb * 4));
        CK(cudaMalloc(&dy, (size_t)M * N * 4));
        CK(cudaMemcpy(dW, hW, (size_t)N * nsb * 176, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, hx, (size_t)M * K, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dxd, hxd, (size_t)M * nb * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dxs, hxs, (size_t)M * nb * 4, cudaMemcpyHostToDevice));
        const size_t sm = 2 * 64 * BKP + 4 * 512 * 4;
        const size_t sm6 = 64 * BKP + 4 * 512 * 4;
        CK(cudaFuncSetAttribute((const void*)&k_gemm_mma_q5k<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm6));
        dim3 g(N / 64, (M + 63) / 64);
        for (int mode = 0; mode < 7; ++mode) {
            if (mode == 1 || mode == 2 || mode == 3 || mode == 5) continue;
            const size_t use_sm = (mode == 6) ? sm6 : sm;
            float best = 1e9f;
            for (int rep = 0; rep < 3; ++rep) {
                cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
                CK(cudaEventRecord(e0));
                for (int i = 0; i < 20; ++i) {
                    switch (mode) {
                        case 0: k_gemm_mma_q5k<0><<<g, 256, use_sm>>>(dy, dW, dx, dxd, dxs, K, N, M); break;
                        case 4: k_gemm_mma_q5k<4><<<g, 256, use_sm>>>(dy, dW, dx, dxd, dxs, K, N, M); break;
                        default: k_gemm_mma_q5k<6><<<g, 256, use_sm>>>(dy, dW, dx, dxd, dxs, K, N, M); break;
                    }
                }
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); ms /= 20;
                best = fminf(best, ms);
                cudaEventDestroy(e0); cudaEventDestroy(e1);
            }
            const char* names[6] = {"full", "no-min", "no-scale", "no-stage", "coalesced", "stage-free"};
            double macs = (double)M * N * K;
            double wb = (double)N * nsb * 176;
            printf("bench %-9s K=%d N=%d M=%d: %.3f ms, %.1f T-MAC/s, %.0f GB/s, grid %dx%d\n",
                   names[mode], K, N, M, best, macs / (best * 1e9), wb / (best * 1e6), g.x, g.y);
        }
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        const int reps = 50;
        CK(cudaEventRecord(e0));
        for (int i = 0; i < reps; ++i)
            k_gemm_mma_q5k<0><<<g, 256, sm>>>(dy, dW, dx, dxd, dxs, K, N, M);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); ms /= reps;
        double macs = (double)M * N * K;
        double wb = (double)N * nsb * 176;
        printf("bench K=%d N=%d M=%d: %.3f ms, %.1f T-MAC/s, %.0f GB/s (q5k weight bytes), grid %dx%d\n",
               K, N, M, ms, macs / (ms * 1e9), wb / (ms * 1e6), g.x, g.y);
        cudaFree(dW); cudaFree(dx); cudaFree(dxd); cudaFree(dxs); cudaFree(dy);
        free(hW); free(hx); free(hxd); free(hxs);
    }
    return 0;
}
