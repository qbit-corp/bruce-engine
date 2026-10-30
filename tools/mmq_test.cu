// Standalone correctness harness for the 128x128 Q5_K MMQ kernel.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

#define MMQ_TN 128
#define MMQ_TM 128
#define MMQ_BKP 144
#define MMQ_SMEM (2 * 128 * MMQ_BKP * 2 + 2 * 128 * 32 + 4 * 2 * 128 * 8 * 2)

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
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait()   { asm volatile("cp.async.wait_group 0;\n"); }

__device__ __forceinline__ void ldmatrix2(uint32_t (&r)[2], const void* p) {
    unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}
__device__ __forceinline__ void ldmatrix4(uint32_t (&r)[4], const void* p) {
    unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ float2 lds64(const void* p) {
    float2 v;
    unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];\n" : "=f"(v.x), "=f"(v.y) : "r"(s));
    return v;
}

__global__ void __launch_bounds__(256, 1)
k_mmq(float* __restrict__ y, const uint8_t* __restrict__ W,
      const int8_t* __restrict__ xq, const float* __restrict__ xd,
      const int* __restrict__ xsum, int K, int N, int rows, int8_t* dbg = nullptr) {
    extern __shared__ __align__(16) int8_t mmq_sh[];
    int8_t* sW   = mmq_sh;
    int8_t* sX   = sW + 2 * 128 * MMQ_BKP;
    int8_t* sQh  = sX + 2 * 128 * MMQ_BKP;
    float2* sWdm = (float2*)(sQh + 2 * 128 * 32);
    float2* sXds = sWdm + 2 * 128 * 4;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int grp = lane >> 2, tg = lane & 3;
    const int wm = warp >> 1, wn = warp & 1;
    const int nsb = K >> 8, nstage = K >> 7;
    const int row0 = blockIdx.x * MMQ_TM;
    const int nbase = blockIdx.y * MMQ_TN;
    const int srow = tid >> 2, sci = tid & 3;

    auto stage_async = [&](int s, int buf) {
        const int hh = s & 1, h2 = s >> 1, gk = s * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = tid + i * 256;
            const int r = c >> 3, o = (c & 7) << 4;
            int8_t* d = sX + ((size_t)buf * 128 + r) * MMQ_BKP + o;
            const int gm = row0 + r;
            if (gm < rows) cp16(d, xq + (size_t)gm * K + gk + o);
            else           *(int4*)d = make_int4(0, 0, 0, 0);
        }
        if (tid < 128) {
            const int gm = row0 + tid;
            int8_t* d = sW + ((size_t)buf * 128 + tid) * MMQ_BKP + 128;
            if (gm < rows) cp16(d, xd + (size_t)gm * (K >> 5) + s * 4);
            else           *(int4*)d = make_int4(0, 0, 0, 0);
        } else {
            const int r = tid - 128, gm = row0 + r;
            int8_t* d = sX + ((size_t)buf * 128 + r) * MMQ_BKP + 128;
            if (gm < rows) cp16(d, xsum + (size_t)gm * (K >> 5) + s * 4);
            else           *(int4*)d = make_int4(0, 0, 0, 0);
        }
        if (tid < 128) {
            const uint8_t* wb = W + (size_t)(nbase + tid) * (size_t)nsb * 176 + (size_t)h2 * 176;
            cp16(sW + ((size_t)buf * 128 + tid) * MMQ_BKP, wb);
        }
        if (hh == 0) {
            const int r = tid >> 1, o = (tid & 1) << 4;
            const uint8_t* wb = W + (size_t)(nbase + r) * (size_t)nsb * 176 + (size_t)h2 * 176;
            cp16(sQh + ((size_t)(h2 & 1) * 128 + r) * 32 + o, wb + 16 + o);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int c = tid + i * 256;
            const int r = c >> 2, k = c & 3;
            const uint8_t* wb = W + (size_t)(nbase + r) * (size_t)nsb * 176 + (size_t)h2 * 176;
            cp16(sW + ((size_t)buf * 128 + r) * MMQ_BKP + 16 + (k << 4),
                 wb + 48 + 64 * hh + (k << 4));
        }
        cp_commit();
    };

    auto unpack = [&](int s, int buf) {
        const int hh = s & 1, h2 = s >> 1;
        #pragma unroll
        for (int rep = 0; rep < 2; ++rep) {
            const int r = srow + rep * 64;
            const int8_t* rowp = sW + ((size_t)buf * 128 + r) * MMQ_BKP;
            const int4 hdr = *(const int4*)(rowp);
            const int4 qh  = *(const int4*)(sQh + ((size_t)(h2 & 1) * 128 + r) * 32 + 16 * (sci & 1));
            const int4 qs  = *(const int4*)(rowp + 16 + 16 * sci);
            const int j0 = 4 * hh + 2 * (sci >> 1);
            const uint32_t* qsv = (const uint32_t*)&qs;
            const uint32_t* qhv = (const uint32_t*)&qh;
            uint32_t v0[4], v1[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                v0[u] = (qsv[u] & 0x0F0F0F0Fu) | (((qhv[u] >> j0) & 0x01010101u) << 4);
                v1[u] = ((qsv[u] >> 4) & 0x0F0F0F0Fu) | (((qhv[u] >> (j0 + 1)) & 0x01010101u) << 4);
            }
            __syncwarp();
            int8_t* db = sW + ((size_t)buf * 128 + r) * MMQ_BKP + 64 * (sci >> 1) + 16 * (sci & 1);
            *(int4*)(db) = *(const int4*)v0;
            *(int4*)(db + 32) = *(const int4*)v1;
            const float d = __half2float(*(const __half*)&hdr.x);
            const float dmin = __half2float(*(const __half*)((const char*)&hdr.x + 2));
            float ds, dm;
            q5k_scale(hdr.y, hdr.z, hdr.w, 4 * hh + sci, d, dmin, &ds, &dm);
            sWdm[buf * 512 + r * 4 + sci] = make_float2(ds, -dm);
        }
    };

    auto scale_x = [&](int s, int buf) {
        (void)s;
        for (int i = tid; i < 512; i += 256) {
            const int r = i >> 2, c = i & 3;
            const float d = *(const float*)(sW + ((size_t)buf * 128 + r) * MMQ_BKP + 128 + c * 4);
            const int sx = *(const int*)(sX + ((size_t)buf * 128 + r) * MMQ_BKP + 128 + c * 4);
            sXds[buf * 512 + i] = make_float2(d, d * (float)sx);
        }
    };

    float acc[2][8][4];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.f;

    stage_async(0, 0);
    cp_wait();
    __syncthreads();
    unpack(0, 0);
    scale_x(0, 0);
    __syncthreads();
    if (dbg) {
        for (int i = tid; i < 128 * 128; i += 256) {
            const int r = i >> 7, k = i & 127;
            dbg[i] = sW[r * MMQ_BKP + k];
        }
        __syncthreads();
    }


    for (int s = 0; s < nstage; ++s) {
        const int buf = s & 1;
        if (s + 1 < nstage) stage_async(s + 1, buf ^ 1);
        const int8_t* Wb = sW + (size_t)buf * 128 * MMQ_BKP;
        const int8_t* Xb = sX + (size_t)buf * 128 * MMQ_BKP;
        const float2* Wdm = sWdm + buf * 512;
        const float2* Xds = sXds + buf * 512;
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            uint32_t a[2][4], bf[8][2];
            float2 wdm[2][2], xds[8][2];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int8_t* ap = Wb + (size_t)(wm * 32 + mt * 16 + grp) * MMQ_BKP + (b << 5) + tg * 4;
                a[mt][0] = *(const uint32_t*)(ap);
                a[mt][1] = *(const uint32_t*)(ap + 8 * MMQ_BKP);
                a[mt][2] = *(const uint32_t*)(ap + 16);
                a[mt][3] = *(const uint32_t*)(ap + 8 * MMQ_BKP + 16);
                const int r0 = wm * 32 + mt * 16 + grp;
                wdm[mt][0] = lds64(&Wdm[r0 * 4 + b]);
                wdm[mt][1] = lds64(&Wdm[(r0 + 8) * 4 + b]);
            }
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const int8_t* bp = Xb + (size_t)(wn * 64 + nt * 8 + grp) * MMQ_BKP + (b << 5) + tg * 4;
                bf[nt][0] = *(const uint32_t*)(bp);
                bf[nt][1] = *(const uint32_t*)(bp + 16);
                const int j0 = wn * 64 + nt * 8 + 2 * tg;
                xds[nt][0] = lds64(&Xds[j0 * 4 + b]);
                xds[nt][1] = lds64(&Xds[(j0 + 1) * 4 + b]);
            }
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt)
                #pragma unroll
                for (int nt = 0; nt < 8; ++nt) {
                    int c[4] = {0, 0, 0, 0};
                    mma_s8(c, a[mt], bf[nt]);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int rr = i >> 1, cc = i & 1;
                        acc[mt][nt][i] = fmaf(wdm[mt][rr].x * xds[nt][cc].x, (float)c[i],
                                              fmaf(wdm[mt][rr].y, xds[nt][cc].y, acc[mt][nt][i]));
                    }

                }
        }
        if (s + 1 < nstage) {
            cp_wait();
            __syncthreads();
            unpack(s + 1, buf ^ 1);
            scale_x(s + 1, buf ^ 1);
            __syncthreads();
        }
    }

    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int ra = nbase + wm * 32 + mt * 16 + grp;
            const int ca = row0 + wn * 64 + nt * 8 + 2 * tg;
            if (ca < rows) {
                if (ra < N)     y[(size_t)ca * N + ra] = acc[mt][nt][0];
                if (ra + 8 < N) y[(size_t)ca * N + ra + 8] = acc[mt][nt][2];
            }
            if (ca + 1 < rows) {
                if (ra < N)     y[(size_t)(ca + 1) * N + ra] = acc[mt][nt][1];
                if (ra + 8 < N) y[(size_t)(ca + 1) * N + ra + 8] = acc[mt][nt][3];
            }
        }
}

static void fill_q5k(uint8_t* p, int N, int K, unsigned seed) {
    const int nsb = K >> 8;
    srand(seed);
    for (int n = 0; n < N; ++n)
        for (int s = 0; s < nsb; ++s) {
            uint8_t* b = p + ((size_t)n * nsb + s) * 176;
            *(uint16_t*)(b) = 0x1c00;
            *(uint16_t*)(b + 2) = 0x1400;
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
    const int K = argc > 1 ? atoi(argv[1]) : 512;
    const int N = argc > 2 ? atoi(argv[2]) : 128;
    const int M = argc > 3 ? atoi(argv[3]) : 128;
    const int nsb = K >> 8, nb = K >> 5;
    CK(cudaFuncSetAttribute((const void*)&k_mmq, cudaFuncAttributeMaxDynamicSharedMemorySize, MMQ_SMEM));

    uint8_t* hW = (uint8_t*)malloc((size_t)N * nsb * 176);
    int8_t* hx = (int8_t*)malloc((size_t)M * K);
    float* hxd = (float*)malloc((size_t)M * nb * 4);
    int* hxs = (int*)malloc((size_t)M * nb * 4);
    float* hy = (float*)malloc((size_t)M * N * 4);
    float* hr = (float*)malloc((size_t)M * N * 4);
    fill_q5k(hW, N, K, 7);
    const int simple = argc > 4 ? atoi(argv[4]) : 0;
    if (simple >= 2) {
        const int nsb2 = K >> 8;
        for (int n = 0; n < N; ++n)
            for (int s = 0; s < nsb2; ++s) {
                uint8_t* b = hW + ((size_t)n * nsb2 + s) * 176;
                *(uint16_t*)(b) = 0x3c00;      // d = 1
                *(uint16_t*)(b + 2) = 0x0000;  // dmin = 0
                *(uint32_t*)(b + 4) = 0x01010101u;
                *(uint32_t*)(b + 8) = 0u;
                *(uint32_t*)(b + 12) = 0x11111111u;
                if (simple == 3) {
                    for (int i = 16; i < 48; ++i) b[i] = 0;      // qh
                    for (int i = 48; i < 176; ++i) b[i] = 0x11;  // all codes 1
                }
            }
    }
    srand(11);
    for (int i = 0; i < M * K; ++i) hx[i] = (simple == 3) ? 1 : (int8_t)(rand() % 255 - 127);
    for (int i = 0; i < M * nb; ++i) {
        hxd[i] = simple ? 1.0f : (0.001f + 0.0005f * (rand() % 100) / 100.f);
        int s = 0;
        for (int o = 0; o < 32; ++o) s += hx[(size_t)(i / nb) * K + i % nb * 32 + o];
        hxs[i] = simple ? 0 : s;
    }
    uint8_t *dW; int8_t* dx; float* dxd; int* dxs; float* dy;
    CK(cudaMalloc(&dW, (size_t)N * nsb * 176)); CK(cudaMalloc(&dx, (size_t)M * K));
    CK(cudaMalloc(&dxd, (size_t)M * nb * 4)); CK(cudaMalloc(&dxs, (size_t)M * nb * 4));
    CK(cudaMalloc(&dy, (size_t)M * N * 4));
    CK(cudaMemcpy(dW, hW, (size_t)N * nsb * 176, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, hx, (size_t)M * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxd, hxd, (size_t)M * nb * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxs, hxs, (size_t)M * nb * 4, cudaMemcpyHostToDevice));
    const bool bench = N > 256;
    if (!bench) ref_gemm(hW, hx, hxd, hxs, K, N, M, hr);
    cudaMemset(dy, 0, (size_t)M * N * 4);
    if (bench) {
        for (int i = 0; i < 5; ++i)
            k_mmq<<<dim3((M + 127) / 128, N / 128), 256, MMQ_SMEM>>>(dy, dW, dx, dxd, dxs, K, N, M);
        CK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        CK(cudaEventRecord(e0));
        for (int i = 0; i < 20; ++i)
            k_mmq<<<dim3((M + 127) / 128, N / 128), 256, MMQ_SMEM>>>(dy, dW, dx, dxd, dxs, K, N, M);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); ms /= 20;
        printf("bench K=%d N=%d M=%d: %.3f ms, %.2f T-MAC/s, grid %dx%d, %.0f GB/s weights\n",
               K, N, M, ms, (double)M * N * K / (ms * 1e9),
               (M + 127) / 128, N / 128, (double)N * nsb * 176 / (ms * 1e6));
        cudaFree(dW); cudaFree(dx); cudaFree(dxd); cudaFree(dxs); cudaFree(dy);
        free(hW); free(hx); free(hxd); free(hxs); free(hy); free(hr);
        return 0;
    }
    int8_t* ddbg = nullptr;
    CK(cudaMalloc(&ddbg, 128 * 128));
    k_mmq<<<dim3((M + 127) / 128, N / 128), 256, MMQ_SMEM>>>(dy, dW, dx, dxd, dxs, K, N, M, ddbg);
    CK(cudaDeviceSynchronize());
    {
        int8_t* hd = (int8_t*)malloc(128 * 128);
        CK(cudaMemcpy(hd, ddbg, 128 * 128, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int r = 0; r < 128 && bad < 8; ++r)
            for (int k = 0; k < 128; ++k) {
                const int j = k >> 5, o = k & 31;
                const uint8_t* wb = hW + (size_t)r * nsb * 176;
                const int u = ((wb[48 + 32 * (j >> 1) + o] >> (4 * (j & 1))) & 0xF)
                            | (((wb[16 + o] >> j) & 1) << 4);
                if (hd[r * 128 + k] != (int8_t)u) {
                    if (bad < 8) printf("unpack mismatch r=%d k=%d got %d want %d\n", r, k, hd[r * 128 + k], u);
                    ++bad;
                }
            }
        printf("unpack mismatches: %d\n", bad);
        free(hd); cudaFree(ddbg);
    }
    CK(cudaMemcpy(hy, dy, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
    double maxabs = 0, mag = 0, maxt = 0; int bad = 0, nan = 0;
    for (int i = 0; i < M * N; ++i) {
        if (!isfinite(hy[i])) ++nan;
        maxabs = fmax(maxabs, fabs(hy[i] - hr[i]));
        mag = fmax(mag, fabs(hr[i]));
        if (fabs(hy[i] - hr[i]) > 1e-3f * fabs(hr[i]) + 1e-4f) ++bad;
    }
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            maxt = fmax(maxt, fabs(hy[(size_t)m * N + n] - hr[(size_t)n * N + m]));
    printf("K=%d N=%d M=%d: max |diff| %.4g (max |y| %.4g), bad %d/%d, nonfinite %d, transposed %.4g\n",
           K, N, M, maxabs, mag, bad, M * N, nan, maxt);
    for (int i = 0; i < 8; ++i) printf("  y[%d]=%.6g ref=%.6g\n", i, hy[i], hr[i]);
    cudaFree(dW); cudaFree(dx); cudaFree(dxd); cudaFree(dxs); cudaFree(dy);
    free(hW); free(hx); free(hxd); free(hxs); free(hy); free(hr);
    return 0;
}
