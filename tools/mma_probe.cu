// Probe: peak dp4a vs int8-tensor-core (mma.m16n8k32.s8) vs fp16 mma on this GPU,
// plus a layout validation of the m16n8k32 s8 fragment mapping.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

__global__ void k_dp4a(int iters, int* out) {
    int a = threadIdx.x * 3 + 1, b = threadIdx.x * 5 + 2;
    int c[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = 0;
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) c[j] = __dp4a(a + j + (i & 1), b + j - (i & 1), c[j]);
    }
    int s = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) s += c[i];
    if (s == 12345678) out[threadIdx.x] = s;
}

__global__ void k_mma_s8(int iters, int* out) {
    int a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3;
    int b0 = threadIdx.x + 5, b1 = b0 + 1;
    int c[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) c[i][j] = 0;
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0 + j), "r"(b1 + j));
        }
    }
    int s = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) s += c[i][j];
    if (s == 12345678) out[threadIdx.x] = s;
}

__global__ void k_mma_f16(int iters, int* out) {
    unsigned a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3;
    unsigned b0 = threadIdx.x + 5, b1 = b0 + 1;
    float c[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) c[i][j] = 0.f;
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0 + j), "r"(b1 + j));
        }
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) s += c[i][j];
    if (s == 12345678.f) out[threadIdx.x] = (int)s;
}

// ---------------- layout check: C[m][n] = sum_k A[m][k] * B[n][k]
__global__ void k_check(const signed char* A, const signed char* B, int* C) {
    const int lane = threadIdx.x & 31;
    const int grp = lane >> 2, tg = (lane & 3) * 4;
    auto pack4 = [](const signed char* p) {
        unsigned v = 0;
        v |= (unsigned)(unsigned char)p[0];
        v |= (unsigned)(unsigned char)p[1] << 8;
        v |= (unsigned)(unsigned char)p[2] << 16;
        v |= (unsigned)(unsigned char)p[3] << 24;
        return v;
    };
    const unsigned a0 = pack4(A + grp * 32 + tg);
    const unsigned a1 = pack4(A + (grp + 8) * 32 + tg);
    const unsigned a2 = pack4(A + grp * 32 + tg + 16);
    const unsigned a3 = pack4(A + (grp + 8) * 32 + tg + 16);
    const unsigned b0 = pack4(B + grp * 32 + tg);
    const unsigned b1 = pack4(B + grp * 32 + tg + 16);
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    const int col = (lane & 3) * 2;
    C[grp * 8 + col] = c0;
    C[grp * 8 + col + 1] = c1;
    C[(grp + 8) * 8 + col] = c2;
    C[(grp + 8) * 8 + col + 1] = c3;
}

int main() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s  SMs %d\n", p.name, p.multiProcessorCount);

    const int blocks = p.multiProcessorCount, thr = 512, reps = 5;
    const int it_d = 20000, it_m = 2000;
    int* d; CK(cudaMalloc(&d, 4096));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    float ms;

    // dp4a
    k_dp4a<<<blocks, thr>>>(it_d, d); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) k_dp4a<<<blocks, thr>>>(it_d, d);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms, a, b));
    {
        double macs = 32.0 * it_d * reps * (double)thr * blocks;
        printf("dp4a            : %7.2f T-MAC/s   (ms %.2f, macs %.3e)\n", macs / ms / 1e9, ms, macs);
    }
    // int8 mma
    k_mma_s8<<<blocks, thr>>>(it_m, d); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) k_mma_s8<<<blocks, thr>>>(it_m, d);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms, a, b));
    {
        double macs = 512.0 * it_m * reps * (double)thr * blocks;
        printf("mma.m16n8k32.s8 : %7.2f T-MAC/s   (ms %.2f, macs %.3e)\n", macs / ms / 1e9, ms, macs);
    }
    // fp16 mma
    k_mma_f16<<<blocks, thr>>>(it_m, d); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) k_mma_f16<<<blocks, thr>>>(it_m, d);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms, a, b));
    {
        double macs = 256.0 * it_m * reps * (double)thr * blocks;
        printf("mma.m16n8k16.f16: %7.2f T-MAC/s   (ms %.2f, macs %.3e)\n", macs / ms / 1e9, ms, macs);
    }
    cudaFree(d);

    signed char hA[16 * 32], hB[8 * 32];
    int hC[16 * 8], hR[16 * 8];
    srand(1234);
    for (int i = 0; i < 16 * 32; ++i) hA[i] = (signed char)(rand() % 17 - 8);
    for (int i = 0; i < 8 * 32; ++i) hB[i] = (signed char)(rand() % 17 - 8);
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n) {
            int s = 0;
            for (int k = 0; k < 32; ++k) s += (int)hA[m * 32 + k] * (int)hB[n * 32 + k];
            hR[m * 8 + n] = s;
        }
    signed char *dA, *dB; int* dC;
    CK(cudaMalloc(&dA, sizeof(hA))); CK(cudaMalloc(&dB, sizeof(hB))); CK(cudaMalloc(&dC, sizeof(hC)));
    CK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    k_check<<<1, 32>>>(dA, dB, dC);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < 16 * 8; ++i) if (hC[i] != hR[i]) bad++;
    printf("layout check    : %s (%d/%d mismatched)\n", bad ? "FAIL" : "OK", bad, 16 * 8);
    return 0;
}
