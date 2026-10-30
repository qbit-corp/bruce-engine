// Bruce inference engine for Dory 1.0 4B on RTX 3060 (SM86).
// Looped normalized-GPT (nGPT), interleaved SWA/full attention, last-loop KV.
// Math mirrors the llama.cpp PR #30306 reference graph.

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#if defined(_MSC_VER)
#pragma comment(lib, "ws2_32.lib")
#endif
#include <io.h>
#else
#include <unistd.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdarg>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>

#include "gguf.h"
#include "tokenizer.h"

#define N_EMBD 2560
#define N_HEAD 16
#define N_KV   4
#define HD     256
#define N_FF   7168
#define VOCAB  131072
#define SWA    1024
// the sliding window holds SWA positions; the ring keeps 2x slots so a speculative
// verify chunk (written before it is known which drafts are accepted) can never
// alias a position that is still inside the live window
#define SWA_SLOTS (2 * SWA)
// q6_K repacked at load: 128 B int8 + 32 B fp32 scales per (row, 128 values)
#define Q6K_SROW 160
#define KV_POS_BYTES 1088   // N_KV * HD values in q8_0 blocks
// Per kv head, per position: 8 fp16 scales (16 B) followed by 8 x 32 int8 values (256 B).
// Putting the scales first keeps the quantised data 8-byte aligned, so a lane can fetch
// its eight weights with a single 64-bit load instead of eight byte loads.
#define KV_KVH_BYTES   272
#define KV_SCALES_BYTES 16
// KV caches are head-major ([kvh][slot][272]) inside each layer buffer, so a block,
// which owns one KV head, streams one contiguous 272 B record per slot instead of
// 272 B out of every 1088 B.  g_kv_nctx is the slot count of full-attention layers.
__device__ int g_kv_nctx = 0;
#define N_INPUT 16
#define N_REC   40
#define N_OUT   16

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while(0)

static constexpr float EPS_L2     = 1e-6f;
static constexpr float INIT_STD   = 0.01769999973475933f;
static constexpr float ALPHA_INIT = 0.05000000074505806f;
static constexpr float GATE_INIT  = 50.59644317626953f;

struct TensorW { const void* d = nullptr; const void* dp = nullptr; const void* dm = nullptr; uint32_t type = 0; int K = 0, N = 0; };

struct BlockW {
    int is_ffn = 0, full = 0, theta_sel = 1;
    TensorW wq, wk, wv, wo, wg, wfg, wfu, wfd;
    const float* sqk = nullptr;
    const float* sgate = nullptr;
    const float* suv_gate = nullptr;
    const float* suv_up = nullptr;
    const float* alpha = nullptr;
};

// Serving slots: each slot owns a contiguous slice of the KV pool and behaves
// exactly like the single-sequence engine (positions 0..slot_ctx-1).  A decode
// step batches one row per active slot; prefill stays per-slot.
#define MAX_SLOTS 8
#define MAX_ROWS  8

struct Model {
    const __nv_bfloat16* emb = nullptr;
    const __nv_bfloat16* out_w = nullptr;
    const uint8_t* out_q8 = nullptr;   // head-aligned Q8_0 logits matrix (356.5 MB)
    const float* out_scale = nullptr;
    BlockW blocks[72];
    int order[512];
    int n_virtual = 0;
    int n_loops = 2;
    int n_ctx = 0;                     // total KV pool (all slots)
    int n_slots = 1;
    int slot_ctx = 0;                  // positions per slot
    int full_splits = 64, swa_splits = 32;

    uint8_t* kvf_k = nullptr;
    uint8_t* kvf_v = nullptr;
    uint8_t* kvs_k = nullptr;
    uint8_t* kvs_v = nullptr;
    uint8_t* kvk_ptr[72];              // current slot (existing code paths)
    uint8_t* kvv_ptr[72];
    uint8_t* kvk_slot[MAX_SLOTS][72];
    uint8_t* kvv_slot[MAX_SLOTS][72];
    uint8_t* kvk_layer[72];            // slot-0 base == layer base (multi-slot kernels add the slot offset)
    uint8_t* kvv_layer[72];

    float *x, *input_h, *q, *k, *v, *gate, *aout, *attn_in;
    float *ffn_g, *ffn_u, *ffn_t, *proj, *logits, *rope_cos, *rope_sin;
    int8_t* xq = nullptr; float* xd = nullptr; int* xsum = nullptr;
    // batched path (MBUF rows)
    float *bm_x, *bm_input_h, *bm_q, *bm_k, *bm_v, *bm_gate, *bm_aout, *bm_attn_in;
    float *bm_ffn_g, *bm_ffn_u, *bm_ffn_t, *bm_proj, *bm_logits, *bm_cos, *bm_sin;
    int8_t* bm_xq; float* bm_xd; int* bm_xsum; __half* bm_xdh;
    int* bm_tok;
    // multi-slot decode: per-row local position, KV partition and active flag
    int* d_row_pos = nullptr;
    int* d_row_slot = nullptr;
    int* d_row_act = nullptr;
    float* slot_logits[MAX_SLOTS] = {nullptr};
    unsigned long long* slot_rng[MAX_SLOTS] = {nullptr};
    float *probs, *sort_key, *samp_max, *part;
    int *sort_idx, *idx_in, *d_pos, *d_tok;
    void* sort_tmp = nullptr; size_t sort_tmp_sz = 0;
    unsigned long long* d_rng = nullptr;
    float* dbg_x = nullptr;
    float* dbg_taps = nullptr;

    cudaGraphExec_t ex_fwd = nullptr, ex_fwd_adv = nullptr;
    cudaGraphExec_t ex_fwd1 = nullptr;      // loop-1 skipped (adaptive decoding)
    cudaGraphExec_t ex_fwd_slot[MAX_SLOTS] = {nullptr};   // per-slot M=1 decode/prefill graph
    cudaGraphExec_t ex_verify[MAX_SLOTS][9] = {{nullptr}};
    cudaGraphExec_t ex_verify1[MAX_SLOTS][9] = {{nullptr}};
    cudaGraphExec_t ex_ms[MAX_ROWS + 1] = {nullptr};      // 2..8 concurrent decode rows
    int* d_amax = nullptr;
    float* x_prev = nullptr;
    float* d_delta = nullptr;
    // loop-1 attention reuse: loop-0 attention output of the 5 recurrent
    // full-attention blocks, 8 rows per slot (decode/verify only; prefill never
    // touches it).  Extending this to the SWA blocks was measured: +0.3 ms/token
    // but it breaks the 128/128 code-prompt token match, so it stays full-only.
    float* aout_l0 = nullptr;
    int l1_slot[72];
};

static Model M;
static cudaStream_t S;
static bool g_verbose = false;
static bool g_debug = false;
static int g_debug_block = -1;

// debug tap offsets inside M.dbg_taps
#define TAP_Q    0
#define TAP_K    4096
#define TAP_V    5120
#define TAP_GATE 6144
#define TAP_AOUT 10240
#define TAP_AIN  14336
#define TAP_PROJ 18432
#define TAP_XIN  20992
#define TAP_IH   23552
#define TAP_X0   26112
#define TAP_Q2   28672
#define TAP_KVK  31232   // 2 slots * KV_POS_BYTES = 2176 B = 544 floats
#define TAP_KVV  31776
#define TAP_TOK  32320
#define TAP_POS  32321
#define TAP_N    32322

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- live status / logging
static bool g_tty = false;
static bool g_no_live = false;
static std::mutex g_log_mu;
static double g_status_last = 0.0;
static bool g_status_open = false;
static int g_status_len = 0;

static bool stream_is_tty(FILE* f) {
#ifdef _WIN32
    return _isatty(_fileno(f)) != 0;
#else
    return isatty(fileno(f)) != 0;
#endif
}
static void init_live_log() { g_tty = stream_is_tty(stderr); }

static void log_line(const char* fmt, ...) {
    std::lock_guard<std::mutex> lk(g_log_mu);
    if (g_tty && g_status_open) { fputc('\n', stderr); g_status_open = false; g_status_len = 0; }
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
}

// One self-overwriting status line on a terminal, or a throttled line when redirected.
static void status_line(const char* fmt, ...) {
    if (g_no_live) return;
    std::lock_guard<std::mutex> lk(g_log_mu);
    const double now = now_ms();
    if (!g_tty && now - g_status_last < 500.0) return;
    g_status_last = now;

    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    if (g_tty) {
        int n = (int)strlen(buf);
        fputc('\r', stderr);
        fputs(buf, stderr);
        for (int i = n; i < g_status_len; ++i) fputc(' ', stderr);
        fflush(stderr);
        g_status_len = n;
        g_status_open = true;
    } else {
        fprintf(stderr, "%s\n", buf);
        fflush(stderr);
        g_status_open = false;
        g_status_len = 0;
    }
}

static void status_end() {
    std::lock_guard<std::mutex> lk(g_log_mu);
    if (g_tty && g_status_open) {
        fputc('\n', stderr);
        fflush(stderr);
    }
    g_status_open = false;
    g_status_len = 0;
}

static void* dev_alloc(size_t bytes) { void* p = nullptr; CK(cudaMalloc(&p, bytes)); return p; }

// load-phase profiling (BRUCE_LOAD_PROF=1)
static double g_lp_open = 0, g_lp_tok = 0, g_lp_h2d = 0;
static double g_lp_q8 = 0, g_lp_scale = 0, g_lp_prepack = 0, g_lp_alloc = 0, g_lp_drain = 0;

// ---------------------------------------------------------------- device helpers

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_sum_bcast(float v) {
    v = warp_sum(v);
    return __shfl_sync(0xffffffffu, v, 0);
}
__device__ __forceinline__ float block_sum(float v, float* red) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    if (w == 0) {
        float t = (lane < nw) ? red[lane] : 0.f;
        t = warp_sum(t);
        if (lane == 0) red[0] = t;
    }
    __syncthreads();
    return red[0];
}
__device__ __forceinline__ float block_max(float v, float* red) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    if (w == 0) {
        float t = (lane < nw) ? red[lane] : -INFINITY;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_down_sync(0xffffffffu, t, o));
        if (lane == 0) red[0] = t;
    }
    __syncthreads();
    return red[0];
}
__device__ __forceinline__ float l2_scale(float ss, int n) {
    return (1.0f / sqrtf(ss / (float)n + EPS_L2 / (float)n)) * (1.0f / sqrtf((float)n));
}
__device__ __forceinline__ uint32_t ld32u(const uint8_t* p) {
    return (uint32_t)(*(const uint16_t*)p) | ((uint32_t)(*(const uint16_t*)(p + 2)) << 16);
}
__device__ __forceinline__ unsigned long long splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// ---------------------------------------------------------------- kernels

__global__ void k_embed(float* __restrict__ x, const __nv_bfloat16* __restrict__ emb,
                        const int* __restrict__ tok) {
    int t = *tok;
    const __nv_bfloat16* src = emb + (size_t)t * N_EMBD;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N_EMBD; i += gridDim.x * blockDim.x)
        x[i] = __bfloat162float(src[i]);
}

__global__ void k_rope_tables(float* __restrict__ cs, float* __restrict__ sn,
                              const int* __restrict__ pos) {
    int i = threadIdx.x;
    int p = *pos;
    float th = (float)p * powf(10000000.0f, -2.0f * (float)i / (float)HD);
    cs[i] = cosf(th); sn[i] = sinf(th);
    float th2 = (float)p * powf(10000.0f, -2.0f * (float)i / (float)HD);
    cs[128 + i] = cosf(th2); sn[128 + i] = sinf(th2);
}

// emit f32 dst plus int8 quantized copy (per-32 scale + sum) for the dp4a gemv
static __device__ __forceinline__ void q8_emit(float* __restrict__ dst, const float* __restrict__ buf,
                                               int8_t* __restrict__ xq, float* __restrict__ xd,
                                               int* __restrict__ xsum, int n) {
    if (dst) for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = buf[i];
    // One warp per 32-element group, one lane per element: the int8 write is a single
    // coalesced 32-byte store per group instead of 32 scattered byte stores per thread.
    // amax / sum reductions are order-exact, so the produced q8 data is bit-identical
    // to the previous per-thread implementation.
    const int nb = n >> 5;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int nw = (blockDim.x + 31) >> 5;
    for (int g = w; g < nb; g += nw) {
        const int i = g * 32 + lane;
        const float v = buf[i];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = amax * (1.f / 127.f);
        const float id = d > 0.f ? 1.f / d : 0.f;
        int q = __float2int_rn(v * id);
        q = max(-127, min(127, q));
        xq[i] = (int8_t)q;
        int s = q;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) { xd[g] = d; xsum[g] = s; }
    }
}

__device__ __forceinline__ void q5k_unpack_sc(uint32_t w0, uint32_t w1, uint32_t w2, int js,
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

// one row per warp; 32 lanes cover 4 superblocks (one 32-value sub-block per lane)
template <int WARPS>
__global__ void k_gemv_q5k(float* __restrict__ y, const uint8_t* __restrict__ W,
                           const int8_t* __restrict__ xq, const float* __restrict__ xd,
                           const int* __restrict__ xsum, int K, int N) {
    // shared copy of xq, permuted so each sub-block is 8 consecutive words in bank-disjoint slots
    extern __shared__ int sxi[];
    const int nw = K >> 2;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) {
        const int sb = i >> 6, u = i & 63;
        const int js = 2 * ((u >> 3) & 3) + (u >> 5), q = u & 7;
        sxi[i] = *(const int*)(xq + sb * 256 + js * 32 + q * 4);
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
                int dot = 0;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t nib = nibl ? ((qsl[i] >> 4) & 0x0F0F0F0Fu) : (qsl[i] & 0x0F0F0F0Fu);
                    const uint32_t hi = ((qhl[i] >> js) & 0x01010101u) << 4;
                    dot = __dp4a(sxi[sb * 64 + xsbase + i], (int)(nib | hi), dot);
                }
                const uint32_t w0s = *(const uint32_t*)(b + 4);
                const uint32_t w1s = *(const uint32_t*)(b + 8);
                const uint32_t w2s = *(const uint32_t*)(b + 12);
                float ds, dm;
                q5k_unpack_sc(w0s, w1s, w2s, js, __half2float(*(const __half*)(b)),
                              __half2float(*(const __half*)(b + 2)), &ds, &dm);
                const int isb = sb * 8 + js;
                acc += xd[isb] * (ds * (float)dot - dm * (float)xsum[isb]);
            }
        }
        acc = warp_sum(acc);
        if (lane == 0) y[row] = acc;
    }
}

template <int WARPS>
__global__ void k_gemv_q6k(float* __restrict__ y, const uint8_t* __restrict__ W,
                           const float* __restrict__ x, int K, int N) {
    extern __shared__ float sx[];
    for (int i = threadIdx.x; i < K; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int half = lane >> 3;
    const int o = (lane & 7) * 4;
    const int is = o >> 4;
    const int nsb = K >> 8;
    const int stride = gridDim.x * WARPS;
    for (int row = blockIdx.x * WARPS + warp; row < N; row += stride) {
        const uint8_t* w = W + (size_t)row * (size_t)nsb * 210;
        float acc = 0.f;
        for (int s2 = 0; s2 < nsb; s2 += 2) {
            const int sb = half >> 1;
            const int h  = half & 1;
            const uint8_t* b = w + (s2 + sb) * 210;
            const uint8_t* base = b + h * 64;
            uint32_t qla = ld32u(base + o);
            uint32_t qlb = ld32u(base + 32 + o);
            uint32_t qh4 = ld32u(b + 128 + h * 32 + o);
            const int8_t* sc = (const int8_t*)(b + 192 + h * 8);
            uint32_t s0 = ld32u((const uint8_t*)sc);
            uint32_t s1 = ld32u((const uint8_t*)sc + 4);
            float d = __half2float(*(const __half*)(b + 208));
            float scl0 = d * (float)(int8_t)((s0 >> (8 * is)) & 0xFF);
            float scl1 = d * (float)(int8_t)((s0 >> (8 * is + 16)) & 0xFF);
            float scl2 = d * (float)(int8_t)((s1 >> (8 * is)) & 0xFF);
            float scl3 = d * (float)(int8_t)((s1 >> (8 * is + 16)) & 0xFF);
            const float* xs = sx + (s2 + sb) * 256 + h * 128 + o;
            float p1 = 0, p2 = 0, p3 = 0, p4 = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int qa = ((qla >> (8 * k)) & 0xF) | (((qh4 >> (8 * k)) & 3) << 4);
                int qb = ((qlb >> (8 * k)) & 0xF) | (((qh4 >> (8 * k + 2)) & 3) << 4);
                int qc = ((qla >> (8 * k + 4)) & 0xF) | (((qh4 >> (8 * k + 4)) & 3) << 4);
                int qd = ((qlb >> (8 * k + 4)) & 0xF) | (((qh4 >> (8 * k + 6)) & 3) << 4);
                p1 = fmaf(xs[k],      scl0 * (float)(qa - 32), p1);
                p2 = fmaf(xs[32 + k], scl1 * (float)(qb - 32), p2);
                p3 = fmaf(xs[64 + k], scl2 * (float)(qc - 32), p3);
                p4 = fmaf(xs[96 + k], scl3 * (float)(qd - 32), p4);
            }
            acc += (p1 + p2) + (p3 + p4);
        }
        acc = warp_sum(acc);
        if (lane == 0) y[row] = acc;
    }
}

template <int WARPS, bool SCALED>
__global__ void k_gemv_bf16(float* __restrict__ y, const __nv_bfloat16* __restrict__ W,
                            const float* __restrict__ x, int K, int N,
                            const float* __restrict__ rs) {
    extern __shared__ float sx[];
    for (int i = threadIdx.x; i < K; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nchunk = K / 256;
    const int stride = gridDim.x * WARPS;
    for (int row = blockIdx.x * WARPS + warp; row < N; row += stride) {
        const __nv_bfloat16* w = W + (size_t)row * K;
        float acc = 0.f;
        for (int c = 0; c < nchunk; ++c) {
            const int base = c * 256 + lane * 8;
            uint4 pk = *(const uint4*)(w + base);
            const __nv_bfloat16* h = (const __nv_bfloat16*)&pk;
            #pragma unroll
            for (int k = 0; k < 8; ++k) acc = fmaf(__bfloat162float(h[k]), sx[base + k], acc);
        }
        acc = warp_sum(acc);
        if (lane == 0) y[row] = SCALED ? acc * rs[row] : acc;
    }
}

__global__ void k_q8_quant(int8_t* __restrict__ xq, float* __restrict__ xd,
                           int* __restrict__ xsum, const float* __restrict__ x, int n) {
    const int nb = n >> 5;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        const float* p = x + b * 32;
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(p[i]));
        const float d = amax * (1.f / 127.f);
        const float id = d > 0.f ? 1.f / d : 0.f;
        int s = 0;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            int v = __float2int_rn(p[i] * id);
            v = max(-127, min(127, v));
            xq[b * 32 + i] = (int8_t)v;
            s += v;
        }
        xd[b] = d;
        xsum[b] = s;
    }
}

__global__ void k_norm(float* __restrict__ dst, const float* __restrict__ src, int n,
                       int8_t* __restrict__ xq, float* __restrict__ xd, int* __restrict__ xs) {
    extern __shared__ float sm[];
    float* buf = sm;
    float* red = sm + n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float t = src[i]; buf[i] = t; ss += t * t; }
    __syncthreads();
    ss = block_sum(ss, red);
    float s = l2_scale(ss, n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) buf[i] *= s;
    __syncthreads();
    q8_emit(dst, buf, xq, xd, xs, n);
}

__global__ void k_add_norm(float* __restrict__ x, const float* __restrict__ h, int n,
                           int8_t* __restrict__ xq, float* __restrict__ xd, int* __restrict__ xs) {
    extern __shared__ float sm[];
    float* a = sm;
    float* red = sm + n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float t = x[i] + h[i]; a[i] = t; ss += t * t; }
    __syncthreads();
    ss = block_sum(ss, red);
    float k = l2_scale(ss, n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) a[i] *= k;
    __syncthreads();
    q8_emit(x, a, xq, xd, xs, n);
}

__global__ void k_resid(float* __restrict__ x, const float* __restrict__ upd,
                        const float* __restrict__ alpha, int n,
                        int8_t* __restrict__ xq, float* __restrict__ xd, int* __restrict__ xs) {
    extern __shared__ float sm[];
    float* a = sm;
    float* b = sm + n;
    float* red = sm + 2 * n;
    float sa = 0.f, sb = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float xv = x[i], uv = upd[i];
        a[i] = xv; b[i] = uv; sa += xv * xv; sb += uv * uv;
    }
    __syncthreads();
    sa = block_sum(sa, red);
    __syncthreads();
    sb = block_sum(sb, red);
    float ka = l2_scale(sa, n), kb = l2_scale(sb, n);
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float an = a[i] * ka, bn = b[i] * kb;
        float t = an + (bn - an) * alpha[i];
        a[i] = t; ss += t * t;
    }
    __syncthreads();
    ss = block_sum(ss, red);
    float k = l2_scale(ss, n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) a[i] *= k;
    __syncthreads();
    q8_emit(x, a, xq, xd, xs, n);
}

__global__ void k_gate_norm(float* __restrict__ dst, const float* __restrict__ aout,
                            const float* __restrict__ gate, const float* __restrict__ sg, int n,
                            int8_t* __restrict__ xq, float* __restrict__ xd, int* __restrict__ xs) {
    extern __shared__ float sm[];
    float* buf = sm;
    float* red = sm + n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float gt = gate[i] * sg[i];
        float t = aout[i] / (1.f + __expf(-gt));
        buf[i] = t; ss += t * t;
    }
    __syncthreads();
    ss = block_sum(ss, red);
    float s = l2_scale(ss, n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) buf[i] *= s;
    __syncthreads();
    q8_emit(dst, buf, xq, xd, xs, n);
}

__global__ void k_silu_mul_norm(float* __restrict__ dst, const float* __restrict__ g,
                                const float* __restrict__ u, const float* __restrict__ sg,
                                const float* __restrict__ su, int n,
                                int8_t* __restrict__ xq, float* __restrict__ xd, int* __restrict__ xs) {
    extern __shared__ float sm[];
    float* buf = sm;
    float* red = sm + n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float gv = g[i] * sg[i];
        float t = (gv / (1.f + __expf(-gv))) * u[i] * su[i];
        buf[i] = t; ss += t * t;
    }
    __syncthreads();
    ss = block_sum(ss, red);
    float s = l2_scale(ss, n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) buf[i] *= s;
    __syncthreads();
    q8_emit(dst, buf, xq, xd, xs, n);
}

// q rope+norm+scale (16 blocks), k rope+norm + k/v q8_0 store (first 4 blocks)
__global__ void k_qkv_prep(float* __restrict__ q, const float* __restrict__ k,
                           const float* __restrict__ v, const float* __restrict__ sqk,
                           uint8_t* __restrict__ kvk, uint8_t* __restrict__ kvv,
                           const int* __restrict__ pos, int is_swa, int theta_sel,
                           const float* __restrict__ cs, const float* __restrict__ sn) {
    __shared__ float sm[HD];
    __shared__ float sk[HD];
    __shared__ float red[32];
    const int h = blockIdx.x;
    const int p = *pos;
    const float* csp = cs + (theta_sel == 2 ? 128 : 0);
    const float* snp = sn + (theta_sel == 2 ? 128 : 0);

    {
        float* qh = q + h * HD;
        float ss = 0.f;
        for (int i = threadIdx.x; i < HD; i += blockDim.x) { float t = qh[i]; sm[i] = t; ss += t * t; }
        __syncthreads();
        ss = block_sum(ss, red);
        float norm = l2_scale(ss, HD);
        for (int i = threadIdx.x; i < HD / 2; i += blockDim.x) {
            float x0 = sm[i], x1 = sm[i + HD / 2];
            float c = csp[i], s = snp[i];
            qh[i]          = (x0 * c - x1 * s) * norm * sqk[h * HD + i];
            qh[i + HD / 2] = (x0 * s + x1 * c) * norm * sqk[h * HD + i + HD / 2];
        }
    }
    __syncthreads();
    if (h >= N_KV) return;

    const float* vh = v + h * HD;
    {
        float ss = 0.f;
        for (int i = threadIdx.x; i < HD; i += blockDim.x) { float t = k[h * HD + i]; sk[i] = t; ss += t * t; }
        __syncthreads();
        ss = block_sum(ss, red);
        float norm = l2_scale(ss, HD);
        for (int i = threadIdx.x; i < HD / 2; i += blockDim.x) {
            float x0 = sk[i], x1 = sk[i + HD / 2];
            float c = csp[i], s = snp[i];
            sk[i]          = (x0 * c - x1 * s) * norm;
            sk[i + HD / 2] = (x0 * s + x1 * c) * norm;
        }
    }
    __syncthreads();

    const int pstride = is_swa ? SWA_SLOTS : g_kv_nctx;
    const int slot = is_swa ? (p & (SWA_SLOTS - 1)) : p;
    uint8_t* dstk = kvk + (size_t)h * pstride * KV_KVH_BYTES + (size_t)slot * KV_KVH_BYTES;
    uint8_t* dstv = kvv + (size_t)h * pstride * KV_KVH_BYTES + (size_t)slot * KV_KVH_BYTES;
    const int blk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    float kf = sk[blk * 32 + lane];
    float vf = vh[blk * 32 + lane];
    float amax = fabsf(kf), vmax = fabsf(vf);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
    }
    float d = amax / 127.f, dv = vmax / 127.f;
    float id = d != 0.f ? 1.f / d : 0.f, idv = dv != 0.f ? 1.f / dv : 0.f;
    uint8_t* dk = dstk + KV_SCALES_BYTES + blk * 32;
    uint8_t* dvv = dstv + KV_SCALES_BYTES + blk * 32;
    if (lane == 0) { ((__half*)dstk)[blk] = __float2half(d); ((__half*)dstv)[blk] = __float2half(dv); }
    dk[lane]  = (uint8_t)(int8_t)lrintf(kf * id);
    dvv[lane] = (uint8_t)(int8_t)lrintf(vf * idv);
}

__global__ void k_attn(float* __restrict__ out, const float* __restrict__ q,
                       const uint8_t* __restrict__ kvk, const uint8_t* __restrict__ kvv,
                       const int* __restrict__ pos, int is_swa, float* __restrict__ part) {
    __shared__ float sq[4][HD];
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int nsplit = gridDim.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < 4 * HD; i += blockDim.x) (&sq[0][0])[i] = q[kvh * 4 * HD + i];
    __syncthreads();

    const int p = *pos;
    const int kend = p + 1;
    const int kstart = is_swa ? max(0, kend - SWA) : 0;
    const int nkeys = kend - kstart;
    const int per = (nkeys + nsplit - 1) / nsplit;
    int a0 = kstart + split * per, a1 = min(kend, a0 + per);
    const float* qv = sq[warp];
    const int bi = lane >> 2, li = (lane & 3) * 8;
    const int pstride = is_swa ? SWA_SLOTS : g_kv_nctx;
    const uint8_t* kbase = kvk + (size_t)kvh * pstride * KV_KVH_BYTES;
    const uint8_t* vbase = kvv + (size_t)kvh * pstride * KV_KVH_BYTES;
    const int off = KV_SCALES_BYTES + bi * 32 + li;   // 8-byte aligned, this lane's weights

    float m = -1e30f, l = 0.f;
    float o[8] = {0,0,0,0,0,0,0,0};
    for (int key = a0; key < a1; ++key) {
        const int slot = is_swa ? (key & (SWA_SLOTS - 1)) : key;
        const uint8_t* kk = kbase + (size_t)slot * KV_KVH_BYTES;
        float d = __half2float(((const __half*)kk)[bi]);
        const uint2 w = *(const uint2*)(kk + off);
        const int lo = (int)w.x, hi = (int)w.y;
        float dot = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = (lo >> (8 * j)) & 0xff; v = (v ^ 0x80) - 0x80;
            dot = fmaf(qv[bi * 32 + li + j], d * (float)v, dot);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = (hi >> (8 * j)) & 0xff; v = (v ^ 0x80) - 0x80;
            dot = fmaf(qv[bi * 32 + li + 4 + j], d * (float)v, dot);
        }
        dot = warp_sum_bcast(dot) * 0.0625f;
        float mnew = fmaxf(m, dot);
        float p_ = __expf(dot - mnew);
        float corr = __expf(m - mnew);
        l = l * corr + p_;
        m = mnew;
        const uint8_t* vv = vbase + (size_t)slot * KV_KVH_BYTES;
        float dv = __half2float(((const __half*)vv)[bi]);
        const uint2 wv = *(const uint2*)(vv + off);
        const int vlo = (int)wv.x, vhi = (int)wv.y;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = (vlo >> (8 * j)) & 0xff; v = (v ^ 0x80) - 0x80;
            o[j] = fmaf(o[j], corr, p_ * dv * (float)v);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = (vhi >> (8 * j)) & 0xff; v = (v ^ 0x80) - 0x80;
            o[4 + j] = fmaf(o[4 + j], corr, p_ * dv * (float)v);
        }
    }
    if (nsplit == 1) {
        float il = l > 0.f ? 1.f / l : 0.f;
        for (int j = 0; j < 8; ++j) out[(size_t)(kvh * 4 + warp) * HD + lane * 8 + j] = o[j] * il;
    } else {
        float* base = part + (((size_t)kvh * nsplit + split) * 4 + warp) * (HD + 2);
        for (int j = 0; j < 8; ++j) base[lane * 8 + j] = o[j];
        if (lane == 0) { base[HD] = m; base[HD + 1] = l; }
    }
}

// Four-key software pipeline for single-token decode.  The four K/V loads, the
// four QK dots and the four V accumulations are independent, so one warp keeps
// four DRAM round trips in flight instead of one; the online-softmax state is
// updated once per group of four keys.  Algebraically identical to k_attn.
__global__ void __launch_bounds__(128) k_attn_u4(
        float* __restrict__ out, const float* __restrict__ q,
        const uint8_t* __restrict__ kvk, const uint8_t* __restrict__ kvv,
        const int* __restrict__ pos, int is_swa, float* __restrict__ part) {
    __shared__ float sq[4][HD];
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int nsplit = gridDim.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < 4 * HD; i += blockDim.x) (&sq[0][0])[i] = q[kvh * 4 * HD + i];
    __syncthreads();

    const int p = *pos;
    const int kend = p + 1;
    const int kstart = is_swa ? max(0, kend - SWA) : 0;
    const int nkeys = kend - kstart;
    const int per = (nkeys + nsplit - 1) / nsplit;
    const int a0 = kstart + split * per, a1 = min(kend, a0 + per);
    const float* qv = sq[warp];
    const int bi = lane >> 2, li = (lane & 3) * 8;
    const int pstride = is_swa ? SWA_SLOTS : g_kv_nctx;
    const uint8_t* kbase = kvk + (size_t)kvh * pstride * KV_KVH_BYTES;
    const uint8_t* vbase = kvv + (size_t)kvh * pstride * KV_KVH_BYTES;
    const int off = KV_SCALES_BYTES + bi * 32 + li;
    const int smask = is_swa ? (SWA_SLOTS - 1) : 0x7fffffff;
    const float* ql = qv + bi * 32 + li;

    float m = -1e30f, l = 0.f;
    float o[8] = {0,0,0,0,0,0,0,0};
    int key = a0;
    for (; key + 3 < a1; key += 4) {
        uint2 wk[4], wv[4];
        float dk[4], dv[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int slot = (key + u) & smask;
            const uint8_t* kk = kbase + (size_t)slot * KV_KVH_BYTES;
            const uint8_t* vv = vbase + (size_t)slot * KV_KVH_BYTES;
            wk[u] = *(const uint2*)(kk + off);
            wv[u] = *(const uint2*)(vv + off);
            dk[u] = __half2float(((const __half*)kk)[bi]);
            dv[u] = __half2float(((const __half*)vv)[bi]);
        }
        float dot[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int lo = (int)wk[u].x, hi = (int)wk[u].y;
            float s = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int w = (lo >> (8 * j)) & 0xff; w = (w ^ 0x80) - 0x80;
                s = fmaf(ql[j], dk[u] * (float)w, s);
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int w = (hi >> (8 * j)) & 0xff; w = (w ^ 0x80) - 0x80;
                s = fmaf(ql[4 + j], dk[u] * (float)w, s);
            }
            dot[u] = warp_sum_bcast(s) * 0.0625f;
        }
        const float mx = fmaxf(fmaxf(dot[0], dot[1]), fmaxf(dot[2], dot[3]));
        const float mn = fmaxf(m, mx);
        const float corr = __expf(m - mn);
        m = mn;
        l *= corr;
        float pv[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) { pv[u] = __expf(dot[u] - mn); l += pv[u]; }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int lo = (int)wv[u].x, hi = (int)wv[u].y;
            const float pu = pv[u] * dv[u];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int w = (lo >> (8 * j)) & 0xff; w = (w ^ 0x80) - 0x80;
                o[j] = fmaf(o[j], corr, pu * (float)w);
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int w = (hi >> (8 * j)) & 0xff; w = (w ^ 0x80) - 0x80;
                o[4 + j] = fmaf(o[4 + j], corr, pu * (float)w);
            }
        }
    }
    for (; key < a1; ++key) {
        const int slot = key & smask;
        const uint8_t* kk = kbase + (size_t)slot * KV_KVH_BYTES;
        const uint8_t* vv = vbase + (size_t)slot * KV_KVH_BYTES;
        const float d = __half2float(((const __half*)kk)[bi]);
        const uint2 w = *(const uint2*)(kk + off);
        const int lo = (int)w.x, hi = (int)w.y;
        float dot = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = (lo >> (8 * j)) & 0xff; v = (v ^ 0x80) - 0x80;
            dot = fmaf(qv[bi * 32 + li + j], d * (float)v, dot);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = (hi >> (8 * j)) & 0xff; v = (v ^ 0x80) - 0x80;
            dot = fmaf(qv[bi * 32 + li + 4 + j], d * (float)v, dot);
        }
        dot = warp_sum_bcast(dot) * 0.0625f;
        const float mnew = fmaxf(m, dot);
        const float p_ = __expf(dot - mnew);
        const float corr = __expf(m - mnew);
        l = l * corr + p_;
        m = mnew;
        const float dv = __half2float(((const __half*)vv)[bi]);
        const uint2 wv = *(const uint2*)(vv + off);
        const int vlo = (int)wv.x, vhi = (int)wv.y;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = (vlo >> (8 * j)) & 0xff; v = (v ^ 0x80) - 0x80;
            o[j] = fmaf(o[j], corr, p_ * dv * (float)v);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = (vhi >> (8 * j)) & 0xff; v = (v ^ 0x80) - 0x80;
            o[4 + j] = fmaf(o[4 + j], corr, p_ * dv * (float)v);
        }
    }
    if (nsplit == 1) {
        const float il = l > 0.f ? 1.f / l : 0.f;
        for (int j = 0; j < 8; ++j) out[(size_t)(kvh * 4 + warp) * HD + lane * 8 + j] = o[j] * il;
    } else {
        float* base = part + (((size_t)kvh * nsplit + split) * 4 + warp) * (HD + 2);
        for (int j = 0; j < 8; ++j) base[lane * 8 + j] = o[j];
        if (lane == 0) { base[HD] = m; base[HD + 1] = l; }
    }
}

// Split-K merge for single-token decode.  One block per query head, one thread
// per output dimension, and the per-split softmax weights are computed by the
// first `nsplit` threads and broadcast through shared memory.  The old form used
// one warp per head and walked the split list serially, which is latency-bound
// (a 21-way L2 chain) rather than bandwidth-bound.
__global__ void __launch_bounds__(256) k_attn_combine_fast(
        float* __restrict__ out, const float* __restrict__ part, int nsplit) {
#define COMBINE_MAX_SPLITS 128
    __shared__ float sw[COMBINE_MAX_SPLITS + 1];
    __shared__ float sred[8];
    const int h = blockIdx.x, tid = threadIdx.x;
    const int kvh = h >> 2, warp = h & 3;
    const size_t pbase = (size_t)kvh * nsplit * 4 + warp;
    const size_t pstep = 4;
    // phase 1: split maxima
    float mx = -1e30f;
    for (int s = tid; s < nsplit; s += blockDim.x) {
        const float* b = part + (pbase + (size_t)s * pstep) * (HD + 2);
        mx = fmaxf(mx, b[HD]);
    }
    // block max
    {
        const int lane = tid & 31, w = tid >> 5;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        if (lane == 0) sred[w] = mx;
        __syncthreads();
        if (w == 0) {
            const int nw = blockDim.x >> 5;
            mx = (lane < nw) ? sred[lane] : -1e30f;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            if (lane == 0) sred[0] = mx;
        }
        __syncthreads();
        mx = sred[0];
    }
    // phase 2: per-split weight, then the shared denominator
    __syncthreads();
    float ls = 0.f;
    for (int s = tid; s < nsplit; s += blockDim.x) {
        const float* b = part + (pbase + (size_t)s * pstep) * (HD + 2);
        const float w = __expf(b[HD] - mx);
        sw[s] = w;
        ls += w * b[HD + 1];
    }
    {
        const int lane = tid & 31, w = tid >> 5;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ls += __shfl_xor_sync(0xffffffffu, ls, o);
        if (lane == 0) sred[w] = ls;
        __syncthreads();
        if (tid == 0) {
            float t = 0.f;
            const int nw = blockDim.x >> 5;
            for (int i = 0; i < nw; ++i) t += sred[i];
            sw[COMBINE_MAX_SPLITS] = 1.f / t;   // scratch slot for 1/l
        }
        __syncthreads();
        ls = sw[COMBINE_MAX_SPLITS];
    }
    float acc = 0.f;
    for (int s = 0; s < nsplit; ++s) {
        const float* b = part + (pbase + (size_t)s * pstep) * (HD + 2);
        acc = fmaf(b[tid], sw[s], acc);
    }
    out[(size_t)h * HD + tid] = acc * ls;
}

__global__ void k_attn_combine(float* __restrict__ out, const float* __restrict__ part, int nsplit) {
    const int kvh = blockIdx.x / 4, warp = blockIdx.x % 4;
    const int lane = threadIdx.x & 31;
    float m = -1e30f;
    for (int s = 0; s < nsplit; ++s) m = fmaxf(m, part[(((size_t)kvh * nsplit + s) * 4 + warp) * (HD + 2) + HD]);
    float l = 0.f, acc[8] = {0,0,0,0,0,0,0,0};
    for (int s = 0; s < nsplit; ++s) {
        const float* base = part + (((size_t)kvh * nsplit + s) * 4 + warp) * (HD + 2);
        float w = __expf(base[HD] - m);
        l += base[HD + 1] * w;
        for (int j = 0; j < 8; ++j) acc[j] = fmaf(base[lane * 8 + j], w, acc[j]);
    }
    float il = 1.f / l;
    const int qh = kvh * 4 + warp;
    for (int j = 0; j < 8; ++j) out[(size_t)qh * HD + lane * 8 + j] = acc[j] * il;
}

__global__ void k_copy(float* __restrict__ d, const float* __restrict__ s, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) d[i] = s[i];
}

// fixed-point diagnostics: out[0] = 1 - cos(x, x_prev), out[1] = ||x-xp||/||x||
__global__ void k_l2_delta(float* __restrict__ out, const float* __restrict__ x,
                           const float* __restrict__ xp, int n) {
    __shared__ float red[64];
    float sd = 0.f, sa = 0.f, sb = 0.f, dot = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float a = x[i], b = xp[i];
        sd += (a - b) * (a - b);
        sa += a * a; sb += b * b; dot += a * b;
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sd += __shfl_down_sync(0xffffffffu, sd, o);
        sa += __shfl_down_sync(0xffffffffu, sa, o);
        sb += __shfl_down_sync(0xffffffffu, sb, o);
        dot += __shfl_down_sync(0xffffffffu, dot, o);
    }
    __syncthreads();
    if (lane == 0) { red[w] = sd; red[16 + w] = sa; red[32 + w] = sb; red[48 + w] = dot; }
    __syncthreads();
    if (w == 0) {
        sd = (lane < 8) ? red[lane] : 0.f;
        sa = (lane < 8) ? red[16 + lane] : 0.f;
        sb = (lane < 8) ? red[32 + lane] : 0.f;
        dot = (lane < 8) ? red[48 + lane] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sd += __shfl_down_sync(0xffffffffu, sd, o);
            sa += __shfl_down_sync(0xffffffffu, sa, o);
            sb += __shfl_down_sync(0xffffffffu, sb, o);
            dot += __shfl_down_sync(0xffffffffu, dot, o);
        }
        if (lane == 0) {
            const float den = sqrtf(sa * sb) + 1e-30f;
            out[0] = 1.f - dot / den;
            out[1] = sqrtf(sd / (sa + 1e-30f));
        }
    }
}

__global__ void k_advance(int* __restrict__ pos) { *pos += 1; }

// ---------------------------------------------------------------- batched (multi-token) path

#define MBUF 512
#define PART_MAX_SPLITS 128
// `part` holds rows * nsplit split-records.  The split count is divided by the
// row-chunk count, so rows * nsplit never exceeds 16 * PART_MAX_SPLITS (the
// attention launch keeps at most ~16 row-chunks per split pass).  Sizing the
// buffer for that product instead of MBUF * PART_MAX_SPLITS saves >1 GB at
// MBUF=512 while keeping the row stride exact for the active split count.
#define PART_ROWS_MAX (32 * PART_MAX_SPLITS)
#define PART_ROW_FLOATS (N_KV * 4 * (HD + 2))
static_assert(COMBINE_MAX_SPLITS == PART_MAX_SPLITS, "combine kernel split cap");
__device__ __forceinline__ int part_row_stride(int nsplit) {
    return N_KV * nsplit * 4 * (HD + 2);
}

__global__ void k_embed_rows(float* __restrict__ x, const __nv_bfloat16* __restrict__ emb,
                             const int* __restrict__ toks, int rows) {
    const int m = blockIdx.x;
    if (m >= rows) return;
    const __nv_bfloat16* src = emb + (size_t)toks[m] * N_EMBD;
    float* dst = x + (size_t)m * N_EMBD;
    for (int i = threadIdx.x; i < N_EMBD; i += blockDim.x) dst[i] = __bfloat162float(src[i]);
}

__global__ void k_q8_rows(int8_t* __restrict__ xq, float* __restrict__ xd, int* __restrict__ xsum,
                          const float* __restrict__ x, int n, int rows) {
    const int m = blockIdx.x;
    if (m >= rows) return;
    const int nb = n >> 5;
    const float* p0 = x + (size_t)m * n;
    int8_t* q0 = xq + (size_t)m * n;
    float* d0 = xd + (size_t)m * (n >> 5);
    int* s0 = xsum + (size_t)m * (n >> 5);
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        const float* p = p0 + b * 32;
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(p[i]));
        const float d = amax * (1.f / 127.f);
        const float id = d > 0.f ? 1.f / d : 0.f;
        int s = 0;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            int v = __float2int_rn(p[i] * id);
            v = max(-127, min(127, v));
            q0[b * 32 + i] = (int8_t)v;
            s += v;
        }
        d0[b] = d;
        s0[b] = s;
    }
}

__global__ void k_norm_rows(float* __restrict__ dst, const float* __restrict__ src, int n,
                            int8_t* __restrict__ xq, float* __restrict__ xd, int* __restrict__ xs) {
    extern __shared__ float sm[];
    const int m = blockIdx.x;
    float* buf = sm;
    float* red = sm + n;
    const float* s = src + (size_t)m * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float t = s[i]; buf[i] = t; ss += t * t; }
    __syncthreads();
    ss = block_sum(ss, red);
    float k = l2_scale(ss, n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) buf[i] *= k;
    __syncthreads();
    q8_emit(dst + (size_t)m * n, buf, xq + (size_t)m * n, xd + (size_t)m * (n >> 5),
            xs + (size_t)m * (n >> 5), n);
}

__global__ void k_add_norm_rows(float* __restrict__ x, const float* __restrict__ h, int n,
                                int8_t* __restrict__ xq, float* __restrict__ xd, int* __restrict__ xs) {
    extern __shared__ float sm[];
    const int m = blockIdx.x;
    float* a = sm;
    float* red = sm + n;
    const float* xr = x + (size_t)m * n;
    const float* hr = h + (size_t)m * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float t = xr[i] + hr[i]; a[i] = t; ss += t * t; }
    __syncthreads();
    ss = block_sum(ss, red);
    float k = l2_scale(ss, n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) a[i] *= k;
    __syncthreads();
    q8_emit(x + (size_t)m * n, a, xq + (size_t)m * n, xd + (size_t)m * (n >> 5),
            xs + (size_t)m * (n >> 5), n);
}

__global__ void k_resid_rows(float* __restrict__ x, const float* __restrict__ upd,
                             const float* __restrict__ alpha, int n,
                             int8_t* __restrict__ xq, float* __restrict__ xd, int* __restrict__ xs) {
    extern __shared__ float sm[];
    const int m = blockIdx.x;
    float* a = sm;
    float* b = sm + n;
    float* red = sm + 2 * n;
    float* xr = x + (size_t)m * n;
    const float* ur = upd + (size_t)m * n;
    float sa = 0.f, sb = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float xv = xr[i], uv = ur[i];
        a[i] = xv; b[i] = uv; sa += xv * xv; sb += uv * uv;
    }
    __syncthreads();
    sa = block_sum(sa, red);
    __syncthreads();
    sb = block_sum(sb, red);
    float ka = l2_scale(sa, n), kb = l2_scale(sb, n);
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float an = a[i] * ka, bn = b[i] * kb;
        float t = an + (bn - an) * alpha[i];
        a[i] = t; ss += t * t;
    }
    __syncthreads();
    ss = block_sum(ss, red);
    float k = l2_scale(ss, n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) a[i] *= k;
    __syncthreads();
    q8_emit(xr, a, xq + (size_t)m * n, xd + (size_t)m * (n >> 5), xs + (size_t)m * (n >> 5), n);
}

__global__ void k_gate_norm_rows(float* __restrict__ dst, const float* __restrict__ aout,
                                 const float* __restrict__ gate, const float* __restrict__ sg, int n,
                                 int8_t* __restrict__ xq, float* __restrict__ xd, int* __restrict__ xs) {
    extern __shared__ float sm[];
    const int m = blockIdx.x;
    float* buf = sm;
    float* red = sm + n;
    const float* ar = aout + (size_t)m * n;
    const float* gr = gate + (size_t)m * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float gt = gr[i] * sg[i];
        float t = ar[i] / (1.f + __expf(-gt));
        buf[i] = t; ss += t * t;
    }
    __syncthreads();
    ss = block_sum(ss, red);
    float k = l2_scale(ss, n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) buf[i] *= k;
    __syncthreads();
    q8_emit(dst ? dst + (size_t)m * n : nullptr, buf, xq + (size_t)m * n, xd + (size_t)m * (n >> 5),
            xs + (size_t)m * (n >> 5), n);
}

__global__ void k_silu_mul_norm_rows(float* __restrict__ dst, const float* __restrict__ g,
                                     const float* __restrict__ u, const float* __restrict__ sg,
                                     const float* __restrict__ su, int n,
                                     int8_t* __restrict__ xq, float* __restrict__ xd, int* __restrict__ xs) {
    extern __shared__ float sm[];
    const int m = blockIdx.x;
    float* buf = sm;
    float* red = sm + n;
    const float* gr = g + (size_t)m * n;
    const float* ur = u + (size_t)m * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float gv = gr[i] * sg[i];
        float t = (gv / (1.f + __expf(-gv))) * ur[i] * su[i];
        buf[i] = t; ss += t * t;
    }
    __syncthreads();
    ss = block_sum(ss, red);
    float k = l2_scale(ss, n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) buf[i] *= k;
    __syncthreads();
    q8_emit(dst ? dst + (size_t)m * n : nullptr, buf, xq + (size_t)m * n, xd + (size_t)m * (n >> 5),
            xs + (size_t)m * (n >> 5), n);
}

// ---------------------------------------------------------------------------
// Register-resident batched norms.  256 threads (8 warps); warp w owns the
// contiguous 32-element q8_0 groups w, w+8, ... so every lane holds one element
// of each of its groups in a register array.  The per-group int8 quantisation is
// then a warp reduction over the register values (same amax and integer sum as
// q8_emit, hence bit-identical q8 output), with no shared-memory staging and
// only two block barriers per norm instead of 8-9.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_red(float v, float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) red[w] = v;
    __syncthreads();
    if (w == 0) {
        float t = (lane < 8) ? red[lane] : 0.f;
        t = warp_sum(t);
        if (lane == 0) red[0] = t;
    }
    __syncthreads();
    return red[0];
}

template <int NWARPS>
__device__ __forceinline__ float block_red_n(float v, float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) red[w] = v;
    __syncthreads();
    if (w == 0) {
        float t = (lane < NWARPS) ? red[lane] : 0.f;
        t = warp_sum(t);
        if (lane == 0) red[0] = t;
    }
    __syncthreads();
    return red[0];
}

template <int NWARPS>
__device__ __forceinline__ float2 block_red2_n(float2 v, float2* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v.x = warp_sum(v.x); v.y = warp_sum(v.y);
    if (lane == 0) red[w] = v;
    __syncthreads();
    if (w == 0) {
        float2 t = (lane < NWARPS) ? red[lane] : make_float2(0.f, 0.f);
        t.x = warp_sum(t.x); t.y = warp_sum(t.y);
        if (lane == 0) red[0] = t;
    }
    __syncthreads();
    return red[0];
}

__device__ __forceinline__ float2 block_red2(float2 v, float2* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v.x = warp_sum(v.x); v.y = warp_sum(v.y);
    if (lane == 0) red[w] = v;
    __syncthreads();
    if (w == 0) {
        float2 t = (lane < 8) ? red[lane] : make_float2(0.f, 0.f);
        t.x = warp_sum(t.x); t.y = warp_sum(t.y);
        if (lane == 0) red[0] = t;
    }
    __syncthreads();
    return red[0];
}

// one 32-element q8_0 group, one element per lane
__device__ __forceinline__ void q8_group(int8_t* __restrict__ xq, float* __restrict__ xd,
                                         int* __restrict__ xsum, int g, float v, int lane) {
    float amax = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax * (1.f / 127.f);
    const float id = d > 0.f ? 1.f / d : 0.f;
    int q = __float2int_rn(v * id);
    q = max(-127, min(127, q));
    xq[g * 32 + lane] = (int8_t)q;
    int s = q;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) { xd[g] = d; xsum[g] = s; }
}

// N_EMBD residual: x = norm(x + (norm(upd) - norm(x)) * alpha), then q8.
template <int G, int NWARPS = 8>
__global__ void __launch_bounds__(32 * NWARPS) k_resid_rows_opt(
        float* __restrict__ x, const float* __restrict__ upd, const float* __restrict__ alpha, int n,
        int8_t* __restrict__ xq, float* __restrict__ xd, int* __restrict__ xs) {
    __shared__ float2 red2[NWARPS];
    __shared__ float red[NWARPS];
    const int m = blockIdx.x;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int ng = n >> 5;
    float* xr = x + (size_t)m * n;
    const float* ur = upd + (size_t)m * n;
    float rx[G], ru[G];
    float2 ss2 = make_float2(0.f, 0.f);
    #pragma unroll
    for (int b = 0; b < G; ++b) {
        const int g = w + b * NWARPS;
        if (g >= ng) break;
        const int i = g * 32 + lane;
        rx[b] = xr[i]; ru[b] = ur[i];
        ss2.x += rx[b] * rx[b];
        ss2.y += ru[b] * ru[b];
    }
    ss2 = block_red2_n<NWARPS>(ss2, red2);
    const float ka = l2_scale(ss2.x, n), kb = l2_scale(ss2.y, n);
    float ss = 0.f;
    #pragma unroll
    for (int b = 0; b < G; ++b) {
        const int g = w + b * NWARPS;
        if (g >= ng) break;
        const int i = g * 32 + lane;
        const float an = rx[b] * ka, bn = ru[b] * kb;
        const float t = an + (bn - an) * alpha[i];
        rx[b] = t; ss += t * t;
    }
    ss = block_red_n<NWARPS>(ss, red);
    const float k = l2_scale(ss, n);
    #pragma unroll
    for (int b = 0; b < G; ++b) {
        const int g = w + b * NWARPS;
        if (g >= ng) break;
        const int i = g * 32 + lane;
        const float v = rx[b] * k;
        xr[i] = v;
        q8_group(xq + (size_t)m * n, xd + (size_t)m * (n >> 5), xs + (size_t)m * (n >> 5), g, v, lane);
    }
}

// N_FF silu(gate)*up, L2 norm, then q8.  dst may be null (bm_ffn_t is never read).
template <int G, int NWARPS = 8>
__global__ void __launch_bounds__(32 * NWARPS) k_silu_mul_norm_rows_opt(
        float* __restrict__ dst, const float* __restrict__ g, const float* __restrict__ u,
        const float* __restrict__ sg, const float* __restrict__ su, int n,
        int8_t* __restrict__ xq, float* __restrict__ xd, int* __restrict__ xs) {
    __shared__ float red[NWARPS];
    const int m = blockIdx.x;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int ng = n >> 5;
    const float* gr = g + (size_t)m * n;
    const float* ur = u + (size_t)m * n;
    float* dr = dst ? dst + (size_t)m * n : nullptr;
    float r[G];
    float ss = 0.f;
    #pragma unroll
    for (int b = 0; b < G; ++b) {
        const int g_ = w + b * NWARPS;
        if (g_ >= ng) break;
        const int i = g_ * 32 + lane;
        const float gv = gr[i] * sg[i];
        const float t = (gv / (1.f + __expf(-gv))) * ur[i] * su[i];
        r[b] = t; ss += t * t;
    }
    ss = block_red_n<NWARPS>(ss, red);
    const float k = l2_scale(ss, n);
    #pragma unroll
    for (int b = 0; b < G; ++b) {
        const int g_ = w + b * NWARPS;
        if (g_ >= ng) break;
        const int i = g_ * 32 + lane;
        const float v = r[b] * k;
        if (dr) dr[i] = v;
        q8_group(xq + (size_t)m * n, xd + (size_t)m * (n >> 5), xs + (size_t)m * (n >> 5), g_, v, lane);
    }
}

// ---------------------------------------------------------------------------
// float4 register-resident variants of the row norms.  A quarter warp (8 lanes)
// owns one 32-element q8_0 group, so the group's quantisation costs three
// shuffles instead of five and one 4-byte coalesced store instead of 32 byte
// stores.  The elementwise chain (including d, id, __float2int_rn and the
// integer sum) is identical to q8_group.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void q8_group_vec4(int8_t* __restrict__ xq, float* __restrict__ xd,
                                              int* __restrict__ xsum, int g, float4 v, int sub8) {
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    const float d = amax * (1.f / 127.f);
    const float id = d > 0.f ? 1.f / d : 0.f;
    int q0 = max(-127, min(127, __float2int_rn(v.x * id)));
    int q1 = max(-127, min(127, __float2int_rn(v.y * id)));
    int q2 = max(-127, min(127, __float2int_rn(v.z * id)));
    int q3 = max(-127, min(127, __float2int_rn(v.w * id)));
    const uint32_t pack = ((uint32_t)(uint8_t)q0) | (((uint32_t)(uint8_t)q1) << 8)
                        | (((uint32_t)(uint8_t)q2) << 16) | (((uint32_t)(uint8_t)q3) << 24);
    *(uint32_t*)(xq + g * 32 + sub8 * 4) = pack;
    int s = (q0 + q1) + (q2 + q3);
    s += __shfl_xor_sync(0xffffffffu, s, 4);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    if (sub8 == 0) { xd[g] = d; xsum[g] = s; }
}

// N_FF silu(gate)*up, L2 norm, then q8.  dst may be null (bm_ffn_t is never read).
__global__ void __launch_bounds__(256, 2) k_silu_mul_norm_vec4(
        float* __restrict__ dst, const float* __restrict__ g, const float* __restrict__ u,
        const float* __restrict__ sg, const float* __restrict__ su, int n,
        int8_t* __restrict__ xq, float* __restrict__ xd, int* __restrict__ xs) {
    __shared__ float red[8];
    const int m = blockIdx.x;
    const int tid = threadIdx.x, qwarp = tid >> 3, sub8 = tid & 7;
    const int ng4 = n >> 5;                 // q8 groups (32 values each)
    const int nq = (blockDim.x + 7) >> 3;   // quarter-warps per block
    const float4* gr4 = (const float4*)(g + (size_t)m * n);
    const float4* ur4 = (const float4*)(u + (size_t)m * n);
    const float4* sg4 = (const float4*)sg;
    const float4* su4 = (const float4*)su;
    float4 r[7];
    float ss = 0.f;
    #pragma unroll
    for (int b = 0; b < 7; ++b) {
        const int g_ = qwarp + b * nq;
        if (g_ >= ng4) break;
        const int idx4 = g_ * 8 + sub8;
        const float4 gv = gr4[idx4], uv = ur4[idx4], sgv = sg4[idx4], suv = su4[idx4];
        float4 t;
        float gx = gv.x * sgv.x; t.x = (gx / (1.f + __expf(-gx))) * uv.x * suv.x;
        float gy = gv.y * sgv.y; t.y = (gy / (1.f + __expf(-gy))) * uv.y * suv.y;
        float gz = gv.z * sgv.z; t.z = (gz / (1.f + __expf(-gz))) * uv.z * suv.z;
        float gw = gv.w * sgv.w; t.w = (gw / (1.f + __expf(-gw))) * uv.w * suv.w;
        r[b] = t;
        ss += (t.x * t.x + t.y * t.y) + (t.z * t.z + t.w * t.w);
    }
    ss = block_red_n<8>(ss, red);
    const float k = l2_scale(ss, n);
    float* dr = dst ? dst + (size_t)m * n : nullptr;
    int8_t* xqr = xq + (size_t)m * n;
    float* xdr = xd + (size_t)m * (n >> 5);
    int* xsr = xs + (size_t)m * (n >> 5);
    #pragma unroll
    for (int b = 0; b < 7; ++b) {
        const int g_ = qwarp + b * nq;
        if (g_ >= ng4) break;
        const float4 v = make_float4(r[b].x * k, r[b].y * k, r[b].z * k, r[b].w * k);
        if (dr) ((float4*)(dr))[g_ * 8 + sub8] = v;
        q8_group_vec4(xqr, xdr, xsr, g_, v, sub8);
    }
}

// N_EMBD residual: x = norm(x + (norm(upd) - norm(x)) * alpha), then q8.
__global__ void __launch_bounds__(160, 4) k_resid_vec4(
        float* __restrict__ x, const float* __restrict__ upd, const float* __restrict__ alpha, int n,
        int8_t* __restrict__ xq, float* __restrict__ xd, int* __restrict__ xs) {
    __shared__ float2 red2[5];
    __shared__ float red[5];
    const int m = blockIdx.x;
    const int tid = threadIdx.x, qwarp = tid >> 3, sub8 = tid & 7;
    float4* xr4 = (float4*)(x + (size_t)m * n);
    const float4* ur4 = (const float4*)(upd + (size_t)m * n);
    const float4* al4 = (const float4*)alpha;
    float4 rx[4], ru[4];
    float2 ss2 = make_float2(0.f, 0.f);
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int g = qwarp + b * 20;
        const int idx4 = g * 8 + sub8;
        const float4 xv = xr4[idx4], uv = ur4[idx4];
        rx[b] = xv; ru[b] = uv;
        ss2.x += (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
        ss2.y += (uv.x * uv.x + uv.y * uv.y) + (uv.z * uv.z + uv.w * uv.w);
    }
    ss2 = block_red2_n<5>(ss2, red2);
    const float ka = l2_scale(ss2.x, n), kb = l2_scale(ss2.y, n);
    float ss = 0.f;
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int idx4 = (qwarp + b * 20) * 8 + sub8;
        const float4 a = al4[idx4];
        float4 t;
        float anx = rx[b].x * ka, bnx = ru[b].x * kb; t.x = anx + (bnx - anx) * a.x;
        float any = rx[b].y * ka, bny = ru[b].y * kb; t.y = any + (bny - any) * a.y;
        float anz = rx[b].z * ka, bnz = ru[b].z * kb; t.z = anz + (bnz - anz) * a.z;
        float anw = rx[b].w * ka, bnw = ru[b].w * kb; t.w = anw + (bnw - anw) * a.w;
        rx[b] = t;
        ss += (t.x * t.x + t.y * t.y) + (t.z * t.z + t.w * t.w);
    }
    ss = block_red_n<5>(ss, red);
    const float k = l2_scale(ss, n);
    int8_t* xqr = xq + (size_t)m * n;
    float* xdr = xd + (size_t)m * (n >> 5);
    int* xsr = xs + (size_t)m * (n >> 5);
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int g = qwarp + b * 20;
        const int idx4 = g * 8 + sub8;
        const float4 v = make_float4(rx[b].x * k, rx[b].y * k, rx[b].z * k, rx[b].w * k);
        xr4[idx4] = v;
        q8_group_vec4(xqr, xdr, xsr, g, v, sub8);
    }
}

// N_HEAD*HD gate norm (attention output * sigmoid(gate)); dst may be null.
__global__ void __launch_bounds__(256, 2) k_gate_norm_vec4(
        float* __restrict__ dst, const float* __restrict__ aout, const float* __restrict__ gate,
        const float* __restrict__ sg, int n,
        int8_t* __restrict__ xq, float* __restrict__ xd, int* __restrict__ xs) {
    __shared__ float red[8];
    const int m = blockIdx.x;
    const int tid = threadIdx.x, qwarp = tid >> 3, sub8 = tid & 7;
    const int ng4 = n >> 5;                 // 128 groups for 16*256
    const int nq = (blockDim.x + 7) >> 3;   // 32 quarter-warps
    const float4* ar4 = (const float4*)(aout + (size_t)m * n);
    const float4* gr4 = (const float4*)(gate + (size_t)m * n);
    const float4* sg4 = (const float4*)sg;
    float4 r[4];
    float ss = 0.f;
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int g_ = qwarp + b * nq;
        if (g_ >= ng4) break;
        const int idx4 = g_ * 8 + sub8;
        const float4 av = ar4[idx4], gv = gr4[idx4], sv = sg4[idx4];
        float4 t;
        float gx = gv.x * sv.x; t.x = av.x / (1.f + __expf(-gx));
        float gy = gv.y * sv.y; t.y = av.y / (1.f + __expf(-gy));
        float gz = gv.z * sv.z; t.z = av.z / (1.f + __expf(-gz));
        float gw = gv.w * sv.w; t.w = av.w / (1.f + __expf(-gw));
        r[b] = t;
        ss += (t.x * t.x + t.y * t.y) + (t.z * t.z + t.w * t.w);
    }
    ss = block_red_n<8>(ss, red);
    const float k = l2_scale(ss, n);
    float* dr = dst ? dst + (size_t)m * n : nullptr;
    int8_t* xqr = xq + (size_t)m * n;
    float* xdr = xd + (size_t)m * (n >> 5);
    int* xsr = xs + (size_t)m * (n >> 5);
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int g_ = qwarp + b * nq;
        if (g_ >= ng4) break;
        const float4 v = make_float4(r[b].x * k, r[b].y * k, r[b].z * k, r[b].w * k);
        if (dr) ((float4*)(dr))[g_ * 8 + sub8] = v;
        q8_group_vec4(xqr, xdr, xsr, g_, v, sub8);
    }
}


// Fused split-combine + gate norm: reads the attention partials directly in
// registers, so bm_aout is never materialised (and bm_attn_in never written).
// The per-split softmax weight only depends on the head, so it is computed once
// per head in shared memory (exactly the same sequential order as before) instead
// of 256 x G times per row.  With 32..128 splits that is a >100x cut in exp work.
template <int G>
__global__ void __launch_bounds__(256) k_combine_gate_norm_rows_opt(
        float* __restrict__ dst, const float* __restrict__ part, const float* __restrict__ gate,
        const float* __restrict__ sg, int nsplit, int n,
        int8_t* __restrict__ xq, float* __restrict__ xd, int* __restrict__ xs) {
    __shared__ float red[8];
    __shared__ float s_wgt[16][PART_MAX_SPLITS];
    __shared__ float s_l[16];
    const int m = blockIdx.x;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int ps = part_row_stride(nsplit);
    const float* prow = part + (size_t)m * ps;
    const float* gr = gate + (size_t)m * n;
    float* dr = dst ? dst + (size_t)m * n : nullptr;

    for (int head = w; head < 16; head += 8) {
        const int kvh = head >> 2, hh = head & 3;
        const float* hbase = prow + ((size_t)kvh * nsplit * 4 + hh) * (HD + 2);
        float mx = -1e30f;
        for (int s = lane; s < nsplit; s += 32)
            mx = fmaxf(mx, hbase[(size_t)s * 4 * (HD + 2) + HD]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        float lp = 0.f;
        for (int s = lane; s < nsplit; s += 32) {
            const float* base = hbase + (size_t)s * 4 * (HD + 2);
            const float wg = __expf(base[HD] - mx);
            lp += base[HD + 1] * wg;
            s_wgt[head][s] = wg;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) lp += __shfl_xor_sync(0xffffffffu, lp, o);
        if (lane == 0) s_l[head] = lp;
    }
    __syncthreads();

    float r[G];
    float ss = 0.f;
    #pragma unroll
    for (int b = 0; b < G; ++b) {
        const int g_ = w + b * 8;
        const int head = g_ >> 3;
        const int dim = ((g_ & 7) << 5) + lane;
        const float* bbase = prow + ((size_t)(head >> 2) * nsplit * 4 + (head & 3)) * (HD + 2);
        float acc = 0.f;
        for (int s = 0; s < nsplit; ++s) {
            const float* base = bbase + (size_t)s * 4 * (HD + 2);
            acc = fmaf(base[dim], s_wgt[head][s], acc);
        }
        const int i = g_ * 32 + lane;
        const float gt = gr[i] * sg[i];
        const float v = (acc / s_l[head]) / (1.f + __expf(-gt));
        r[b] = v; ss += v * v;
    }
    ss = block_red(ss, red);
    const float k = l2_scale(ss, n);
    #pragma unroll
    for (int b = 0; b < G; ++b) {
        const int g_ = w + b * 8, i = g_ * 32 + lane;
        const float v = r[b] * k;
        if (dr) dr[i] = v;
        q8_group(xq + (size_t)m * n, xd + (size_t)m * (n >> 5), xs + (size_t)m * (n >> 5), g_, v, lane);
    }
}

__global__ void k_rope_tables_rows(float* __restrict__ cs, float* __restrict__ sn,
                                   const int* __restrict__ pos, int rows) {
    const int m = blockIdx.x;
    if (m >= rows) return;
    const int p = *pos + m;
    const int i = threadIdx.x;
    float* c = cs + (size_t)m * 256;
    float* s = sn + (size_t)m * 256;
    float th = (float)p * powf(10000000.0f, -2.0f * (float)i / (float)HD);
    c[i] = cosf(th); s[i] = sinf(th);
    float th2 = (float)p * powf(10000.0f, -2.0f * (float)i / (float)HD);
    c[128 + i] = cosf(th2); s[128 + i] = sinf(th2);
}

// q/k/v prep for one row per blockIdx.y; qkv in [row][head][HD]
__global__ void k_qkv_prep_rows(float* __restrict__ q, const float* __restrict__ k,
                                const float* __restrict__ v, const float* __restrict__ sqk,
                                uint8_t* __restrict__ kvk, uint8_t* __restrict__ kvv,
                                const int* __restrict__ pos, int is_swa, int theta_sel,
                                const float* __restrict__ cs, const float* __restrict__ sn) {
    __shared__ float sm[HD];
    __shared__ float sk[HD];
    __shared__ float red[32];
    const int h = blockIdx.x;
    const int m = blockIdx.y;
    const int p = *pos + m;
    const float* csp = cs + (size_t)m * 256 + (theta_sel == 2 ? 128 : 0);
    const float* snp = sn + (size_t)m * 256 + (theta_sel == 2 ? 128 : 0);
    float* qrow = q + (size_t)m * N_HEAD * HD;
    {
        float* qh = qrow + h * HD;
        float ss = 0.f;
        for (int i = threadIdx.x; i < HD; i += blockDim.x) { float t = qh[i]; sm[i] = t; ss += t * t; }
        __syncthreads();
        ss = block_sum(ss, red);
        float norm = l2_scale(ss, HD);
        for (int i = threadIdx.x; i < HD / 2; i += blockDim.x) {
            float x0 = sm[i], x1 = sm[i + HD / 2];
            float c = csp[i], s = snp[i];
            qh[i]          = (x0 * c - x1 * s) * norm * sqk[h * HD + i];
            qh[i + HD / 2] = (x0 * s + x1 * c) * norm * sqk[h * HD + i + HD / 2];
        }
    }
    __syncthreads();
    if (h >= N_KV) return;

    const float* krow = k + (size_t)m * N_KV * HD;
    const float* vrow = v + (size_t)m * N_KV * HD;
    const float* vh = vrow + h * HD;
    {
        float ss = 0.f;
        for (int i = threadIdx.x; i < HD; i += blockDim.x) { float t = krow[h * HD + i]; sk[i] = t; ss += t * t; }
        __syncthreads();
        ss = block_sum(ss, red);
        float norm = l2_scale(ss, HD);
        for (int i = threadIdx.x; i < HD / 2; i += blockDim.x) {
            float x0 = sk[i], x1 = sk[i + HD / 2];
            float c = csp[i], s = snp[i];
            sk[i]          = (x0 * c - x1 * s) * norm;
            sk[i + HD / 2] = (x0 * s + x1 * c) * norm;
        }
    }
    __syncthreads();

    const int pstride = is_swa ? SWA_SLOTS : g_kv_nctx;
    const int slot = is_swa ? (p & (SWA_SLOTS - 1)) : p;
    uint8_t* dstk = kvk + (size_t)h * pstride * KV_KVH_BYTES + (size_t)slot * KV_KVH_BYTES;
    uint8_t* dstv = kvv + (size_t)h * pstride * KV_KVH_BYTES + (size_t)slot * KV_KVH_BYTES;
    const int blk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    float kf = sk[blk * 32 + lane];
    float vf = vh[blk * 32 + lane];
    float amax = fabsf(kf), vmax = fabsf(vf);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
    }
    float d = amax / 127.f, dv = vmax / 127.f;
    float id = d != 0.f ? 1.f / d : 0.f, idv = dv != 0.f ? 1.f / dv : 0.f;
    uint8_t* dk = dstk + KV_SCALES_BYTES + blk * 32;
    uint8_t* dvv = dstv + KV_SCALES_BYTES + blk * 32;
    if (lane == 0) { ((__half*)dstk)[blk] = __float2half(d); ((__half*)dstv)[blk] = __float2half(dv); }
    dk[lane]  = (uint8_t)(int8_t)lrintf(kf * id);
    dvv[lane] = (uint8_t)(int8_t)lrintf(vf * idv);
}

// One warp per head: the whole 256-float head stays in a warp's registers, so
// there is no shared staging and no __syncthreads().  Warps 0..7 normalise/RoPE
// two Q heads each, then warps 0..3 quantise one KV head each (quarter-warp per
// 32-value q8_0 sub-block, one 4-byte store per lane).
__global__ void __launch_bounds__(256, 4) k_qkv_prep_fast(
        float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
        const float* __restrict__ sqk, uint8_t* __restrict__ kvk, uint8_t* __restrict__ kvv,
        const int* __restrict__ pos, int is_swa, int theta_sel,
        const float* __restrict__ cs, const float* __restrict__ sn) {
    const int m = blockIdx.x, p = *pos + m;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float4 c = ((const float4*)(cs + (size_t)m * 256 + (theta_sel == 2 ? 128 : 0)))[lane];
    const float4 s = ((const float4*)(sn + (size_t)m * 256 + (theta_sel == 2 ? 128 : 0)))[lane];

    float* qrow = q + (size_t)m * N_HEAD * HD;
    #pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
        const int h = warp + rep * 8;
        float4* qh4 = (float4*)(qrow + h * HD);
        const float4* sqk4 = (const float4*)(sqk + h * HD);
        const float4 x0 = qh4[lane], x1 = qh4[lane + 32];
        const float4 s0 = sqk4[lane], s1 = sqk4[lane + 32];
        float ss = (x0.x * x0.x + x0.y * x0.y + x0.z * x0.z + x0.w * x0.w)
                 + (x1.x * x1.x + x1.y * x1.y + x1.z * x1.z + x1.w * x1.w);
        ss = warp_sum_bcast(ss);
        const float norm = l2_scale(ss, HD);
        qh4[lane] = make_float4((x0.x * c.x - x1.x * s.x) * norm * s0.x,
                                (x0.y * c.y - x1.y * s.y) * norm * s0.y,
                                (x0.z * c.z - x1.z * s.z) * norm * s0.z,
                                (x0.w * c.w - x1.w * s.w) * norm * s0.w);
        qh4[lane + 32] = make_float4((x0.x * s.x + x1.x * c.x) * norm * s1.x,
                                     (x0.y * s.y + x1.y * c.y) * norm * s1.y,
                                     (x0.z * s.z + x1.z * c.z) * norm * s1.z,
                                     (x0.w * s.w + x1.w * c.w) * norm * s1.w);
    }

    if (warp >= N_KV) return;
    const int h = warp;
    const float4* kh4 = (const float4*)(k + (size_t)m * N_KV * HD + h * HD);
    const float4* vh4 = (const float4*)(v + (size_t)m * N_KV * HD + h * HD);
    const float4 kx0 = kh4[lane], kx1 = kh4[lane + 32];
    float kss = (kx0.x * kx0.x + kx0.y * kx0.y + kx0.z * kx0.z + kx0.w * kx0.w)
              + (kx1.x * kx1.x + kx1.y * kx1.y + kx1.z * kx1.z + kx1.w * kx1.w);
    kss = warp_sum_bcast(kss);
    const float knorm = l2_scale(kss, HD);
    const float4 kr0 = make_float4((kx0.x * c.x - kx1.x * s.x) * knorm,
                                   (kx0.y * c.y - kx1.y * s.y) * knorm,
                                   (kx0.z * c.z - kx1.z * s.z) * knorm,
                                   (kx0.w * c.w - kx1.w * s.w) * knorm);
    const float4 kr1 = make_float4((kx0.x * s.x + kx1.x * c.x) * knorm,
                                   (kx0.y * s.y + kx1.y * c.y) * knorm,
                                   (kx0.z * s.z + kx1.z * c.z) * knorm,
                                   (kx0.w * s.w + kx1.w * c.w) * knorm);
    const float4 vr0 = vh4[lane], vr1 = vh4[lane + 32];

    const int pstride = is_swa ? SWA_SLOTS : g_kv_nctx;
    const int slot = is_swa ? (p & (SWA_SLOTS - 1)) : p;
    uint8_t* dstk = kvk + (size_t)h * pstride * KV_KVH_BYTES + (size_t)slot * KV_KVH_BYTES;
    uint8_t* dstv = kvv + (size_t)h * pstride * KV_KVH_BYTES + (size_t)slot * KV_KVH_BYTES;

    const int blk_lo = lane >> 3, sub8 = lane & 7;
    auto quant_kv_half = [&](int blk, float4 kv, float4 vv) {
        float kamax = fmaxf(fmaxf(fabsf(kv.x), fabsf(kv.y)), fmaxf(fabsf(kv.z), fabsf(kv.w)));
        float vamax = fmaxf(fmaxf(fabsf(vv.x), fabsf(vv.y)), fmaxf(fabsf(vv.z), fabsf(vv.w)));
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            kamax = fmaxf(kamax, __shfl_xor_sync(0xffffffffu, kamax, o));
            vamax = fmaxf(vamax, __shfl_xor_sync(0xffffffffu, vamax, o));
        }
        const float d = kamax / 127.f, dv = vamax / 127.f;
        const float id = d != 0.f ? 1.f / d : 0.f, idv = dv != 0.f ? 1.f / dv : 0.f;
        if (sub8 == 0) { ((__half*)dstk)[blk] = __float2half(d); ((__half*)dstv)[blk] = __float2half(dv); }
        const uint32_t pk = ((uint32_t)(uint8_t)(int8_t)lrintf(kv.x * id))
                          | (((uint32_t)(uint8_t)(int8_t)lrintf(kv.y * id)) << 8)
                          | (((uint32_t)(uint8_t)(int8_t)lrintf(kv.z * id)) << 16)
                          | (((uint32_t)(uint8_t)(int8_t)lrintf(kv.w * id)) << 24);
        const uint32_t pv = ((uint32_t)(uint8_t)(int8_t)lrintf(vv.x * idv))
                          | (((uint32_t)(uint8_t)(int8_t)lrintf(vv.y * idv)) << 8)
                          | (((uint32_t)(uint8_t)(int8_t)lrintf(vv.z * idv)) << 16)
                          | (((uint32_t)(uint8_t)(int8_t)lrintf(vv.w * idv)) << 24);
        *(uint32_t*)(dstk + KV_SCALES_BYTES + blk * 32 + sub8 * 4) = pk;
        *(uint32_t*)(dstv + KV_SCALES_BYTES + blk * 32 + sub8 * 4) = pv;
    };
    quant_kv_half(blk_lo,     kr0, vr0);
    quant_kv_half(blk_lo + 4, kr1, vr1);
}

// ---------------------------------------------------------------------------
// Multi-slot decode: same math as k_rope_tables_rows / k_qkv_prep_fast, but every
// row is an independent sequence, so the position, KV partition and "writes KV"
// flag come from per-row device arrays.  Slot partitions are contiguous, and each
// slot uses local positions 0..slot_ctx-1 exactly like the single-sequence engine.
// ---------------------------------------------------------------------------
__global__ void k_rope_tables_rows_ms(float* __restrict__ cs, float* __restrict__ sn,
                                      const int* __restrict__ pos, int rows) {
    const int m = blockIdx.x;
    if (m >= rows) return;
    const int p = pos[m];
    const int i = threadIdx.x;
    float* c = cs + (size_t)m * 256;
    float* s = sn + (size_t)m * 256;
    float th = (float)p * powf(10000000.0f, -2.0f * (float)i / (float)HD);
    c[i] = cosf(th); s[i] = sinf(th);
    float th2 = (float)p * powf(10000.0f, -2.0f * (float)i / (float)HD);
    c[128 + i] = cosf(th2); s[128 + i] = sinf(th2);
}

__global__ void __launch_bounds__(256, 4) k_qkv_prep_fast_ms(
        float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
        const float* __restrict__ sqk, uint8_t* __restrict__ kvk, uint8_t* __restrict__ kvv,
        const int* __restrict__ pos, const int* __restrict__ kslot, const int* __restrict__ act,
        int is_swa, int theta_sel, const float* __restrict__ cs, const float* __restrict__ sn) {
    const int m = blockIdx.x, p = pos[m];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float4 c = ((const float4*)(cs + (size_t)m * 256 + (theta_sel == 2 ? 128 : 0)))[lane];
    const float4 s = ((const float4*)(sn + (size_t)m * 256 + (theta_sel == 2 ? 128 : 0)))[lane];

    float* qrow = q + (size_t)m * N_HEAD * HD;
    #pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
        const int h = warp + rep * 8;
        float4* qh4 = (float4*)(qrow + h * HD);
        const float4* sqk4 = (const float4*)(sqk + h * HD);
        const float4 x0 = qh4[lane], x1 = qh4[lane + 32];
        const float4 s0 = sqk4[lane], s1 = sqk4[lane + 32];
        float ss = (x0.x * x0.x + x0.y * x0.y + x0.z * x0.z + x0.w * x0.w)
                 + (x1.x * x1.x + x1.y * x1.y + x1.z * x1.z + x1.w * x1.w);
        ss = warp_sum_bcast(ss);
        const float norm = l2_scale(ss, HD);
        qh4[lane] = make_float4((x0.x * c.x - x1.x * s.x) * norm * s0.x,
                                (x0.y * c.y - x1.y * s.y) * norm * s0.y,
                                (x0.z * c.z - x1.z * s.z) * norm * s0.z,
                                (x0.w * c.w - x1.w * s.w) * norm * s0.w);
        qh4[lane + 32] = make_float4((x0.x * s.x + x1.x * c.x) * norm * s1.x,
                                     (x0.y * s.y + x1.y * c.y) * norm * s1.y,
                                     (x0.z * s.z + x1.z * c.z) * norm * s1.z,
                                     (x0.w * s.w + x1.w * c.w) * norm * s1.w);
    }

    if (warp >= N_KV) return;
    // idle rows still normalise q (harmless) but must not touch any live KV slot
    if (!act[m]) return;
    const int h = warp;
    const float4* kh4 = (const float4*)(k + (size_t)m * N_KV * HD + h * HD);
    const float4* vh4 = (const float4*)(v + (size_t)m * N_KV * HD + h * HD);
    const float4 kx0 = kh4[lane], kx1 = kh4[lane + 32];
    float kss = (kx0.x * kx0.x + kx0.y * kx0.y + kx0.z * kx0.z + kx0.w * kx0.w)
              + (kx1.x * kx1.x + kx1.y * kx1.y + kx1.z * kx1.z + kx1.w * kx1.w);
    kss = warp_sum_bcast(kss);
    const float knorm = l2_scale(kss, HD);
    const float4 kr0 = make_float4((kx0.x * c.x - kx1.x * s.x) * knorm,
                                   (kx0.y * c.y - kx1.y * s.y) * knorm,
                                   (kx0.z * c.z - kx1.z * s.z) * knorm,
                                   (kx0.w * c.w - kx1.w * s.w) * knorm);
    const float4 kr1 = make_float4((kx0.x * s.x + kx1.x * c.x) * knorm,
                                   (kx0.y * s.y + kx1.y * c.y) * knorm,
                                   (kx0.z * s.z + kx1.z * c.z) * knorm,
                                   (kx0.w * s.w + kx1.w * c.w) * knorm);
    const float4 vr0 = vh4[lane], vr1 = vh4[lane + 32];

    const int pstride = is_swa ? SWA_SLOTS : g_kv_nctx;
    const int slot = is_swa ? (p & (SWA_SLOTS - 1)) : p;
    const size_t sbase = (size_t)kslot[m] * pstride * KV_POS_BYTES;
    uint8_t* dstk = kvk + sbase + (size_t)h * pstride * KV_KVH_BYTES + (size_t)slot * KV_KVH_BYTES;
    uint8_t* dstv = kvv + sbase + (size_t)h * pstride * KV_KVH_BYTES + (size_t)slot * KV_KVH_BYTES;

    const int blk_lo = lane >> 3, sub8 = lane & 7;
    auto quant_kv_half = [&](int blk, float4 kv, float4 vv) {
        float kamax = fmaxf(fmaxf(fabsf(kv.x), fabsf(kv.y)), fmaxf(fabsf(kv.z), fabsf(kv.w)));
        float vamax = fmaxf(fmaxf(fabsf(vv.x), fabsf(vv.y)), fmaxf(fabsf(vv.z), fabsf(vv.w)));
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            kamax = fmaxf(kamax, __shfl_xor_sync(0xffffffffu, kamax, o));
            vamax = fmaxf(vamax, __shfl_xor_sync(0xffffffffu, vamax, o));
        }
        const float d = kamax / 127.f, dv = vamax / 127.f;
        const float id = d != 0.f ? 1.f / d : 0.f, idv = dv != 0.f ? 1.f / dv : 0.f;
        if (sub8 == 0) { ((__half*)dstk)[blk] = __float2half(d); ((__half*)dstv)[blk] = __float2half(dv); }
        const uint32_t pk = ((uint32_t)(uint8_t)(int8_t)lrintf(kv.x * id))
                          | (((uint32_t)(uint8_t)(int8_t)lrintf(kv.y * id)) << 8)
                          | (((uint32_t)(uint8_t)(int8_t)lrintf(kv.z * id)) << 16)
                          | (((uint32_t)(uint8_t)(int8_t)lrintf(kv.w * id)) << 24);
        const uint32_t pv = ((uint32_t)(uint8_t)(int8_t)lrintf(vv.x * idv))
                          | (((uint32_t)(uint8_t)(int8_t)lrintf(vv.y * idv)) << 8)
                          | (((uint32_t)(uint8_t)(int8_t)lrintf(vv.z * idv)) << 16)
                          | (((uint32_t)(uint8_t)(int8_t)lrintf(vv.w * idv)) << 24);
        *(uint32_t*)(dstk + KV_SCALES_BYTES + blk * 32 + sub8 * 4) = pk;
        *(uint32_t*)(dstv + KV_SCALES_BYTES + blk * 32 + sub8 * 4) = pv;
    };
    quant_kv_half(blk_lo,     kr0, vr0);
    quant_kv_half(blk_lo + 4, kr1, vr1);
}

__global__ void k_attn_rows(float* __restrict__ out, const float* __restrict__ q,
                            const uint8_t* __restrict__ kvk, const uint8_t* __restrict__ kvv,
                            const int* __restrict__ pos, int is_swa, float* __restrict__ part) {
    __shared__ float sq[4][HD];
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int row = blockIdx.z;
    const int nsplit = gridDim.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* qrow = q + (size_t)row * N_HEAD * HD;
    for (int i = threadIdx.x; i < 4 * HD; i += blockDim.x) (&sq[0][0])[i] = qrow[kvh * 4 * HD + i];
    __syncthreads();

    const int p = *pos + row;
    const int kend = p + 1;
    const int kstart = is_swa ? max(0, kend - SWA) : 0;
    const int nkeys = kend - kstart;
    const int per = (nkeys + nsplit - 1) / nsplit;
    int a0 = kstart + split * per, a1 = min(kend, a0 + per);
    const float* qv = sq[warp];
    const int bi = lane >> 2, li = (lane & 3) * 8;
    const int pstride = is_swa ? SWA_SLOTS : g_kv_nctx;
    const uint8_t* kbase = kvk + (size_t)kvh * pstride * KV_KVH_BYTES;
    const uint8_t* vbase = kvv + (size_t)kvh * pstride * KV_KVH_BYTES;
    const int off = KV_SCALES_BYTES + bi * 32 + li;   // 8-byte aligned, this lane's weights

    float* prow = part + (size_t)row * part_row_stride(nsplit);
    float m = -1e30f, l = 0.f;
    float o[8] = {0,0,0,0,0,0,0,0};
    for (int key = a0; key < a1; ++key) {
        const int slot = is_swa ? (key & (SWA_SLOTS - 1)) : key;
        const uint8_t* kk = kbase + (size_t)slot * KV_KVH_BYTES;
        float d = __half2float(((const __half*)kk)[bi]);
        const uint2 w = *(const uint2*)(kk + off);
        const int lo = (int)w.x, hi = (int)w.y;
        float dot = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = (lo >> (8 * j)) & 0xff; v = (v ^ 0x80) - 0x80;
            dot = fmaf(qv[bi * 32 + li + j], d * (float)v, dot);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = (hi >> (8 * j)) & 0xff; v = (v ^ 0x80) - 0x80;
            dot = fmaf(qv[bi * 32 + li + 4 + j], d * (float)v, dot);
        }
        dot = warp_sum_bcast(dot) * 0.0625f;
        float mnew = fmaxf(m, dot);
        float p_ = __expf(dot - mnew);
        float corr = __expf(m - mnew);
        l = l * corr + p_;
        m = mnew;
        const uint8_t* vv = vbase + (size_t)slot * KV_KVH_BYTES;
        float dv = __half2float(((const __half*)vv)[bi]);
        const uint2 wv = *(const uint2*)(vv + off);
        const int vlo = (int)wv.x, vhi = (int)wv.y;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = (vlo >> (8 * j)) & 0xff; v = (v ^ 0x80) - 0x80;
            o[j] = fmaf(o[j], corr, p_ * dv * (float)v);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = (vhi >> (8 * j)) & 0xff; v = (v ^ 0x80) - 0x80;
            o[4 + j] = fmaf(o[4 + j], corr, p_ * dv * (float)v);
        }
    }
    if (nsplit == 1) {
        float il = l > 0.f ? 1.f / l : 0.f;
        float* orow = out + (size_t)row * N_HEAD * HD;
        for (int j = 0; j < 8; ++j) orow[(size_t)(kvh * 4 + warp) * HD + lane * 8 + j] = o[j] * il;
    } else {
        float* base = prow + (((size_t)kvh * nsplit + split) * 4 + warp) * (HD + 2);
        for (int j = 0; j < 8; ++j) base[lane * 8 + j] = o[j];
        if (lane == 0) { base[HD] = m; base[HD + 1] = l; }
    }
}

__global__ void k_attn_combine_rows(float* __restrict__ out, const float* __restrict__ part, int nsplit) {
    const int kvh = blockIdx.x / 4, warp = blockIdx.x % 4, row = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const float* prow = part + (size_t)row * part_row_stride(nsplit);
    float m = -1e30f;
    for (int s = 0; s < nsplit; ++s) m = fmaxf(m, prow[(((size_t)kvh * nsplit + s) * 4 + warp) * (HD + 2) + HD]);
    float l = 0.f, acc[8] = {0,0,0,0,0,0,0,0};
    for (int s = 0; s < nsplit; ++s) {
        const float* base = prow + (((size_t)kvh * nsplit + s) * 4 + warp) * (HD + 2);
        float w = __expf(base[HD] - m);
        l += base[HD + 1] * w;
        for (int j = 0; j < 8; ++j) acc[j] = fmaf(base[lane * 8 + j], w, acc[j]);
    }
    float il = 1.f / l;
    const int qh = kvh * 4 + warp;
    float* orow = out + (size_t)row * N_HEAD * HD;
    for (int j = 0; j < 8; ++j) orow[(size_t)qh * HD + lane * 8 + j] = acc[j] * il;
}

// ---------------------------------------------------------------------------
// Multi-row attention for full (non-windowed) layers.  One block still owns one
// (kv head, split) pair, but now every key's K/V is read once and used by all M rows
// of the chunk instead of once per row.  At long context the KV stream dominates, so
// this turns an M-fold KV read into a single one (which is what makes speculative
// verification affordable at 128k context).
// ---------------------------------------------------------------------------
template <int M, bool MASK>
__device__ __forceinline__ void attn_m_step(const uint8_t* __restrict__ kbase,
                                            const uint8_t* __restrict__ vbase, int key, int p0,
                                            int bi, int li, int off,
                                            const float* __restrict__ qv, int qstride,
                                            float (&m)[M], float (&l)[M], float (&o)[M][8]) {
    const uint8_t* kk = kbase + (size_t)key * KV_KVH_BYTES;
    const float d = __half2float(((const __half*)kk)[bi]);
    const uint2 w = *(const uint2*)(kk + off);
    const int lo = (int)w.x, hi = (int)w.y;
    float dot[M];
    #pragma unroll
    for (int r = 0; r < M; ++r) dot[r] = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int v = (lo >> (8 * j)) & 0xff; v = (v ^ 0x80) - 0x80;
        const float dw = d * (float)v;
        #pragma unroll
        for (int r = 0; r < M; ++r) dot[r] = fmaf(qv[r * qstride + bi * 32 + li + j], dw, dot[r]);
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int v = (hi >> (8 * j)) & 0xff; v = (v ^ 0x80) - 0x80;
        const float dw = d * (float)v;
        #pragma unroll
        for (int r = 0; r < M; ++r) dot[r] = fmaf(qv[r * qstride + bi * 32 + li + 4 + j], dw, dot[r]);
    }
    float pv[M], cr[M];
    #pragma unroll
    for (int r = 0; r < M; ++r) {
        dot[r] = warp_sum_bcast(dot[r]) * 0.0625f;
        const bool on = !MASK || (key <= p0 + r);      // causal mask inside the chunk
        const float dt = on ? dot[r] : m[r];
        const float mn = fmaxf(m[r], dt);
        const float e = __expf(dt - mn);
        const float c = __expf(m[r] - mn);
        pv[r] = on ? e : 0.f;
        cr[r] = on ? c : 1.f;
        if (on) m[r] = mn;
        l[r] = l[r] * cr[r] + pv[r];
    }
    const uint8_t* vv = vbase + (size_t)key * KV_KVH_BYTES;
    const float dv = __half2float(((const __half*)vv)[bi]);
    const uint2 wv = *(const uint2*)(vv + off);
    const int vlo = (int)wv.x, vhi = (int)wv.y;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int v = (vlo >> (8 * j)) & 0xff; v = (v ^ 0x80) - 0x80;
        const float vw = dv * (float)v;
        #pragma unroll
        for (int r = 0; r < M; ++r) o[r][j] = fmaf(o[r][j], cr[r], pv[r] * vw);
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int v = (vhi >> (8 * j)) & 0xff; v = (v ^ 0x80) - 0x80;
        const float vw = dv * (float)v;
        #pragma unroll
        for (int r = 0; r < M; ++r) o[r][4 + j] = fmaf(o[r][4 + j], cr[r], pv[r] * vw);
    }
}

template <int M>
__global__ void __launch_bounds__(128) k_attn_m(float* __restrict__ out, const float* __restrict__ q,
                                                const uint8_t* __restrict__ kvk,
                                                const uint8_t* __restrict__ kvv,
                                                const int* __restrict__ pos, float* __restrict__ part) {
    __shared__ float sq[M * 4 * HD];
    const int kvh = blockIdx.x, split = blockIdx.y, nsplit = gridDim.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < M * 4 * HD; i += blockDim.x) {
        const int r = i / (4 * HD), j = i - r * (4 * HD);
        sq[i] = q[(size_t)r * N_HEAD * HD + kvh * 4 * HD + j];
    }
    __syncthreads();

    const int p0 = *pos;
    const int kend = p0 + M;                    // last key index + 1 (chunk tail)
    const int per = (kend + nsplit - 1) / nsplit;
    const int a0 = split * per, a1 = min(kend, a0 + per);

    const uint8_t* kbase = kvk + (size_t)kvh * g_kv_nctx * KV_KVH_BYTES;
    const uint8_t* vbase = kvv + (size_t)kvh * g_kv_nctx * KV_KVH_BYTES;
    const int bi = lane >> 2, li = (lane & 3) * 8;
    const int off = KV_SCALES_BYTES + bi * 32 + li;
    const float* qv = sq + warp * HD;            // row stride 4*HD

    float m[M], l[M], o[M][8];
    #pragma unroll
    for (int r = 0; r < M; ++r) {
        m[r] = -1e30f; l[r] = 0.f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o[r][j] = 0.f;
    }

    const int bulk = min(a1, p0);               // keys visible to every row
    for (int key = a0; key < bulk; ++key)
        attn_m_step<M, false>(kbase, vbase, key, p0, bi, li, off, qv, 4 * HD, m, l, o);
    for (int key = max(a0, p0); key < a1; ++key)
        attn_m_step<M, true>(kbase, vbase, key, p0, bi, li, off, qv, 4 * HD, m, l, o);

    if (nsplit == 1) {
        #pragma unroll
        for (int r = 0; r < M; ++r) {
            const float il = l[r] > 0.f ? 1.f / l[r] : 0.f;
            float* orow = out + ((size_t)r * N_HEAD + kvh * 4 + warp) * HD;
            #pragma unroll
            for (int j = 0; j < 8; ++j) orow[lane * 8 + j] = o[r][j] * il;
        }
    } else {
        #pragma unroll
        for (int r = 0; r < M; ++r) {
            float* base = part + (size_t)r * part_row_stride(nsplit) +
                          (((size_t)kvh * nsplit + split) * 4 + warp) * (HD + 2);
            #pragma unroll
            for (int j = 0; j < 8; ++j) base[lane * 8 + j] = o[r][j];
            if (lane == 0) { base[HD] = m[r]; base[HD + 1] = l[r]; }
        }
    }
}

static int g_attn_tc = 1;   // 0 disables the tensor-core verify kernel (A/B)
static int g_attn_u4 = -1;  // 1 = 4-key pipelined single-row decode attention
static inline bool attn_u4() {
    if (g_attn_u4 < 0) {
        const char* e = getenv("BRUCE_ATTN_U4");
        g_attn_u4 = (e && atoi(e)) ? 1 : 0;
    }
    return g_attn_u4 != 0;
}
static int g_attn_m1 = -1;  // 0 scalar, 1 tensor-core dim-split, 2 row-split (default)
static int g_attn_comb = -1; // 1 = 256-thread split merge (default)
static int g_lazy_v = -1;    // 1 = skip cold V tiles in 1..8-row decode attention
static inline bool lazy_v() {
    if (g_lazy_v < 0) {
        const char* e = getenv("BRUCE_LAZYV");
        g_lazy_v = (e && atoi(e) == 0) ? 0 : 1;
    }
    return g_lazy_v != 0;
}
static inline bool attn_comb_fast() {
    if (g_attn_comb < 0) {
        const char* e = getenv("BRUCE_COMB_FAST");
        g_attn_comb = (e && atoi(e) == 0) ? 0 : 1;
    }
    return g_attn_comb != 0;
}
static inline int attn_m1() {
    if (g_attn_m1 < 0) {
        const char* e = getenv("BRUCE_ATTN_M1PATH");
        g_attn_m1 = e ? std::max(0, std::min(3, atoi(e))) : 3;
    }
    return g_attn_m1;
}
static int g_prefill_tc = 1; // prompt rows share one KV pass on full layers (0 = per-row, bit-exact)
static int g_adaptive_loops = 0;      // skip loop 1 when the loop-0 state has converged
static float g_loop_tau = 0.04f;      // relative L2 step threshold
static float g_last_delta = 1e9f;
static float g_last_rel = 1e9f;
static bool g_use_1loop = false;
static bool g_last_ran_1 = false;
static int g_skip_used = 0;
static int g_loop_probe = 8;          // force a 2-loop token every N skipped tokens

// ---------------------------------------------------------------------------
// Tensor-core verify attention.  Same contract as k_attn_m (one block per
// (kv head, split), writes unnormalized O + (m,l) per row when nsplit > 1), but
// the 4 heads and up to 4 rows of a kv head are packed into one 16-row MMA tile
// (pair = row*4 + head), so the K/V stream is dequantized once per 16 keys and
// every row reuses it.  Each of the 4 warps owns 64 of the 256 output dims:
// O lives in 32 accumulators/lane instead of being reducible across warps, and
// QK^T is recomputed per warp (4x) because that is still cheaper than a
// cross-warp reduction at this arithmetic intensity.  K/V dequantization is
// spread over all 128 threads (8 threads per key, 32 dims each).
//
// Score numerics follow the fp16 flash-attention convention (fp16 Q*K into an
// fp32 accumulator, hence the fp16 rounding of the q8 dequant); everything else
// (softmax, PV accumulate, split merge) stays fp32.
// ---------------------------------------------------------------------------
#define TC_STRIDE 264           // half stride of a 256-dim row (528 B, conflict free)
#define TC_T      16            // keys per tile

// defined further down; used by the pipelined KV staging in k_attn_tc_rs_opt
__device__ __forceinline__ void mma_cp16(void* dst, const void* src);
__device__ __forceinline__ void mma_cp_commit();
__device__ __forceinline__ void mma_cp_wait();

__device__ __forceinline__ void mma_16816_acc(float (&d)[4], const uint32_t (&a)[4],
                                              const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t pack2(float x, float y) {
    __half2 h = __floats2half2_rn(x, y);
    return *(uint32_t*)&h;
}

// Exact int8x4 -> two __half2 dequant (d * (float)int8) without any XU-pipe
// I2F/F2FP traffic.  0x6400 is fp16 1024.0; (byte ^ 0x80) | 0x6400 decodes to
// 1024 + byte with the sign restored, and the low mantissa bits are exactly the
// integer because 2^25 has unit spacing at that magnitude.  HSUB2/HMUL2 run on
// the half2 pipe at full rate, PRMT is a single-cycle ALU op.
__device__ __forceinline__ void dequant_i8x4_to_h2(uint32_t w, __half2 d2, __half2 h1152,
                                                   __half2& out0, __half2& out1) {
    const uint32_t u   = w ^ 0x80808080u;
    const uint32_t h01 = __byte_perm(u, 0x64006400u, 0x7170u);
    const uint32_t h23 = __byte_perm(u, 0x64006400u, 0x7372u);
    out0 = __hmul2(d2, __hsub2(*(__half2*)&h01, h1152));
    out1 = __hmul2(d2, __hsub2(*(__half2*)&h23, h1152));
}

// full-rate fp16 accumulate (SM86 throttles f32 accumulate to half rate)
__device__ __forceinline__ void mma_16816_f16(uint32_t (&d)[2], const uint32_t (&a)[4],
                                              const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void ldmatrix2(uint32_t (&r)[2], const void* p) {
    unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}

__device__ __forceinline__ void ldmatrix2_trans(uint32_t (&r)[2], const void* p) {
    unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}

__device__ __forceinline__ float2 lds64(const void* p) {
    float2 v;
    unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];\n" : "=f"(v.x), "=f"(v.y) : "r"(s));
    return v;
}

template <int NG, int NWARPS>
__global__ void __launch_bounds__(32 * NWARPS) k_attn_tc(float* __restrict__ out, const float* __restrict__ q,
                                                 const uint8_t* __restrict__ kvk,
                                                 const uint8_t* __restrict__ kvv,
                                                 const int* __restrict__ pos, float* __restrict__ part,
                                                 int rows, int is_swa) {
    extern __shared__ __align__(16) __half tc_sh[];
    __half* qs  = tc_sh;                              // [NG*16][TC_STRIDE] packed Q
    __half* ksh = qs + NG * 16 * TC_STRIDE;           // [TC_T][TC_STRIDE]
    __half* vsh = ksh + TC_T * TC_STRIDE;             // [TC_T][TC_STRIDE]


    const int rchunk = blockIdx.x / N_KV, kvh = blockIdx.x % N_KV;
    const int split = blockIdx.y, nsplit = gridDim.y;
    const int RPB = 4 * NG;                        // prompt rows per block
    const int rbase = rchunk * RPB, rows_eff = min(RPB, rows - rbase);
    const int DIMW = HD / NWARPS;                  // output dims per warp
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, t4 = lane & 3;      // MMA row / row-pair index

    // stage Q: pair p of group g is (row g*4 + p/4, head p%4); rows past `rows` are 0
    for (int i = tid; i < NG * 16 * HD; i += 32 * NWARPS) {
        const int g = i / (16 * HD), j = i - g * (16 * HD);
        const int pair = j >> 8, d = j & 255;
        const int r = g * 4 + (pair >> 2), h = pair & 3;
        float v = 0.f;
        if (r < rows_eff) v = q[(size_t)(rbase + r) * N_HEAD * HD + kvh * 4 * HD + h * HD + d] * 0.0625f;
        qs[g * 16 * TC_STRIDE + pair * TC_STRIDE + d] = __float2half(v);
    }

    const int p0 = *pos + rbase;
    const int kend = p0 + rows_eff;                 // chunk tail (exclusive)
    // windowed layers: every row sees only the last SWA keys, so the block range
    // starts at the earliest window start of the chunk (per-row mask handles the rest)
    const int kstart = is_swa ? max(0, p0 - SWA + 1) : 0;
    const int nkeys = kend - kstart;
    const int per = (nkeys + nsplit - 1) / nsplit;
    const int a0 = kstart + split * per, a1 = min(kend, a0 + per);
    const int pstride = is_swa ? SWA_SLOTS : g_kv_nctx;

    float m[NG][2], l[NG][2], o[NG][HD / 8 / NWARPS][4];
    #pragma unroll
    for (int g = 0; g < NG; ++g) {
        m[g][0] = m[g][1] = -1e30f;
        l[g][0] = l[g][1] = 0.f;
        #pragma unroll
        for (int j = 0; j < HD / 8 / NWARPS; ++j) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) o[g][j][c] = 0.f;
        }
    }
    if (a0 >= a1) {                                 // empty split: contribute nothing
        #pragma unroll
        for (int g = 0; g < NG; ++g) { m[g][0] = m[g][1] = -1e30f; l[g][0] = l[g][1] = 0.f; }
    }

    const int dk0 = tid >> 3;                       // key slot inside the tile (0..15)
    const int sub = tid & 7;                        // 32-dim block inside the head
    for (int t0 = a0; t0 < a1; t0 += TC_T) {
        __syncthreads();
        if (tid < 128) {
            int kk = t0 + dk0;
            if (kk > a1 - 1) kk = a1 - 1;
            const int slot = is_swa ? (kk & (SWA_SLOTS - 1)) : kk;
            const size_t kb0 = (size_t)kvh * pstride * KV_KVH_BYTES + (size_t)slot * KV_KVH_BYTES;
            const size_t koff = kb0 + KV_SCALES_BYTES + sub * 32;
            const __half2 dk2 = __half2half2(((const __half*)(kvk + kb0))[sub]);
            const __half2 dv2 = __half2half2(((const __half*)(kvv + kb0))[sub]);
            const __half2 h1152 = __float2half2_rn(1152.0f);
            // 16 B per lane per load (was 4 x 8 B at 32 B stride): one instruction now
            // covers each 32 B chunk instead of four, halving the sectors touched per
            // instruction.  Same values land in the same shared slots.
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                const uint4 wk = *(const uint4*)(kvk + koff + u * 16);
                const uint4 wv = *(const uint4*)(kvv + koff + u * 16);
                __half2 hk[8], hv[8];
                dequant_i8x4_to_h2(wk.x, dk2, h1152, hk[0], hk[1]);
                dequant_i8x4_to_h2(wk.y, dk2, h1152, hk[2], hk[3]);
                dequant_i8x4_to_h2(wk.z, dk2, h1152, hk[4], hk[5]);
                dequant_i8x4_to_h2(wk.w, dk2, h1152, hk[6], hk[7]);
                dequant_i8x4_to_h2(wv.x, dv2, h1152, hv[0], hv[1]);
                dequant_i8x4_to_h2(wv.y, dv2, h1152, hv[2], hv[3]);
                dequant_i8x4_to_h2(wv.z, dv2, h1152, hv[4], hv[5]);
                dequant_i8x4_to_h2(wv.w, dv2, h1152, hv[6], hv[7]);
                *(uint4*)(&ksh[dk0 * TC_STRIDE + sub * 32 + u * 16])     = *(uint4*)&hk[0];
                *(uint4*)(&ksh[dk0 * TC_STRIDE + sub * 32 + u * 16 + 8]) = *(uint4*)&hk[4];
                *(uint4*)(&vsh[dk0 * TC_STRIDE + sub * 32 + u * 16])     = *(uint4*)&hv[0];
                *(uint4*)(&vsh[dk0 * TC_STRIDE + sub * 32 + u * 16 + 8]) = *(uint4*)&hv[4];
            }
        }
        __syncthreads();

        float s[NG][2][4];
        #pragma unroll
        for (int g = 0; g < NG; ++g)
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt)
                #pragma unroll
                for (int c = 0; c < 4; ++c) s[g][nt][c] = 0.f;

        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            uint32_t a[NG][4];
            #pragma unroll
            for (int g = 0; g < NG; ++g) {
                const __half* qb = qs + (g * 16 + gid) * TC_STRIDE + kk * 16 + 2 * t4;
                a[g][0] = *(const uint32_t*)(qb);
                a[g][1] = *(const uint32_t*)(qb + 8 * TC_STRIDE);
                a[g][2] = *(const uint32_t*)(qb + 8);
                a[g][3] = *(const uint32_t*)(qb + 8 * TC_STRIDE + 8);
            }
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const __half* kb = ksh + (nt * 8 + gid) * TC_STRIDE + kk * 16 + 2 * t4;
                uint32_t b[2] = { *(const uint32_t*)(kb), *(const uint32_t*)(kb + 8) };
                #pragma unroll
                for (int g = 0; g < NG; ++g) mma_16816_acc(s[g][nt], a[g], b);
            }
        }

        const bool full = (t0 + TC_T - 1 < a1) && (t0 + TC_T - 1 <= p0) &&
                          (!is_swa || t0 >= p0 + 7 - SWA + 1);
        uint32_t pa[NG][4];
        #pragma unroll
        for (int g = 0; g < NG; ++g) {
            const int r0 = g * 4 + (gid >> 2), r1 = r0 + 2;
            const int lim0 = p0 + r0, lim1 = p0 + r1;
            const int lo0 = is_swa ? max(0, lim0 - SWA + 1) : 0;
            const int lo1 = is_swa ? max(0, lim1 - SWA + 1) : 0;
            float sv[8];
            sv[0] = s[g][0][0]; sv[1] = s[g][0][1]; sv[2] = s[g][0][2]; sv[3] = s[g][0][3];
            sv[4] = s[g][1][0]; sv[5] = s[g][1][1]; sv[6] = s[g][1][2]; sv[7] = s[g][1][3];
            if (!full) {
                const int kA = t0 + 2 * t4, kB = kA + 8;
                if (kA > lim0 || kA >= a1 || kA < lo0) sv[0] = -1e30f;
                if (kA + 1 > lim0 || kA + 1 >= a1 || kA + 1 < lo0) sv[1] = -1e30f;
                if (kB > lim0 || kB >= a1 || kB < lo0) sv[4] = -1e30f;
                if (kB + 1 > lim0 || kB + 1 >= a1 || kB + 1 < lo0) sv[5] = -1e30f;
                if (kA > lim1 || kA >= a1 || kA < lo1) sv[2] = -1e30f;
                if (kA + 1 > lim1 || kA + 1 >= a1 || kA + 1 < lo1) sv[3] = -1e30f;
                if (kB > lim1 || kB >= a1 || kB < lo1) sv[6] = -1e30f;
                if (kB + 1 > lim1 || kB + 1 >= a1 || kB + 1 < lo1) sv[7] = -1e30f;
            }
            float mx0 = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[4], sv[5]));
            float mx1 = fmaxf(fmaxf(sv[2], sv[3]), fmaxf(sv[6], sv[7]));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
            const float mn0 = fmaxf(m[g][0], mx0), mn1 = fmaxf(m[g][1], mx1);
            const bool ok0 = mn0 > -1e29f, ok1 = mn1 > -1e29f;
            const float al0 = ok0 ? __expf(m[g][0] - mn0) : 1.f;
            const float al1 = ok1 ? __expf(m[g][1] - mn1) : 1.f;
            m[g][0] = mn0; m[g][1] = mn1;
            const float z0 = ok0 ? 1.f : 0.f, z1 = ok1 ? 1.f : 0.f;
            float e0 = __expf(sv[0] - mn0) * z0, e1 = __expf(sv[1] - mn0) * z0;
            float e2 = __expf(sv[2] - mn1) * z1, e3 = __expf(sv[3] - mn1) * z1;
            float e4 = __expf(sv[4] - mn0) * z0, e5 = __expf(sv[5] - mn0) * z0;
            float e6 = __expf(sv[6] - mn1) * z1, e7 = __expf(sv[7] - mn1) * z1;
            float sum0 = e0 + e1 + e4 + e5;
            float sum1 = e2 + e3 + e6 + e7;
            sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
            sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
            sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
            sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
            l[g][0] = l[g][0] * al0 + sum0;
            l[g][1] = l[g][1] * al1 + sum1;
            pa[g][0] = pack2(e0, e1);
            pa[g][1] = pack2(e2, e3);
            pa[g][2] = pack2(e4, e5);
            pa[g][3] = pack2(e6, e7);
            if (al0 != 1.f || al1 != 1.f) {          // running max moved: rescale accumulators
                #pragma unroll
                for (int j = 0; j < DIMW / 8; ++j) {
                    o[g][j][0] *= al0; o[g][j][1] *= al0;
                    o[g][j][2] *= al1; o[g][j][3] *= al1;
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < DIMW / 8; ++j) {
            const int dim = warp * DIMW + j * 8 + gid;
            const __half* vb = vsh + (2 * t4) * TC_STRIDE + dim;
            uint32_t b[2];
            b[0] = *(uint32_t*)&__halves2half2(vb[0], vb[TC_STRIDE]);
            b[1] = *(uint32_t*)&__halves2half2(vb[8 * TC_STRIDE], vb[9 * TC_STRIDE]);
            #pragma unroll
            for (int g = 0; g < NG; ++g) mma_16816_acc(o[g][j], pa[g], b);
        }
    }

    #pragma unroll
    for (int g = 0; g < NG; ++g) {
        #pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const int pair = gid + hf * 8;
            const int r = g * 4 + (pair >> 2), h = pair & 3;
            if (r >= rows_eff) continue;
            const float il = l[g][hf] > 0.f ? 1.f / l[g][hf] : 0.f;
            float* base = 0;
            if (nsplit == 1) {
                base = out + ((size_t)(rbase + r) * N_HEAD + kvh * 4 + h) * HD;
            } else {
                base = part + (size_t)(rbase + r) * part_row_stride(nsplit) +
                       (((size_t)kvh * nsplit + split) * 4 + h) * (HD + 2);
            }
            const int d0 = warp * DIMW + 2 * t4;
            #pragma unroll
            for (int j = 0; j < DIMW / 8; ++j)
                if (nsplit == 1) {
                    base[d0 + j * 8] = o[g][j][2 * hf] * il;
                    base[d0 + j * 8 + 1] = o[g][j][2 * hf + 1] * il;
                } else {
                    base[d0 + j * 8] = o[g][j][2 * hf];
                    base[d0 + j * 8 + 1] = o[g][j][2 * hf + 1];
                }
            if (nsplit != 1 && t4 == 0) {
                base[HD] = m[g][hf];
                base[HD + 1] = l[g][hf];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Row-split attention for prompt chunks (llama.cpp's fattn layout, adapted).
//
// k_attn_tc splits the 256 head dims across four warps so that 64 output floats
// per lane fit the register file.  Every warp then needs the full QK^T, so the
// score matrix is computed four times.  Here each warp instead owns FOUR prompt
// rows (16 mma m-slots = 4 rows x 4 q-heads) and walks ALL 256 dims, so:
//   * scores are computed once (attention mma per row: 40 -> 16),
//   * a block covers 32 rows per KV pass (KV traffic 4x lower than 8-row blocks),
//   * every issued mma is useful work.
// Cost: 128 accumulator floats per lane (255-register launch bounds, 1 block/SM).
// The <= 8-row path (verify/decode) keeps the dim-split kernel untouched.
template <int NWARPS>
__global__ void __launch_bounds__(32 * NWARPS) k_attn_tc_rs(
        float* __restrict__ out, const float* __restrict__ q,
        const uint8_t* __restrict__ kvk, const uint8_t* __restrict__ kvv,
        const int* __restrict__ pos, float* __restrict__ part, int rows, int is_swa) {
    extern __shared__ __align__(16) __half rs_sh[];
    __half* qs  = rs_sh;                                  // [NWARPS*16][TC_STRIDE]
    __half* ksh = qs + NWARPS * 16 * TC_STRIDE;           // [TC_T][TC_STRIDE]
    __half* vsh = ksh + TC_T * TC_STRIDE;                 // [TC_T][TC_STRIDE]

    const int rchunk = blockIdx.x / N_KV, kvh = blockIdx.x % N_KV;
    const int split = blockIdx.y, nsplit = gridDim.y;
    const int RPB = 4 * NWARPS;                            // prompt rows per block
    const int rbase = rchunk * RPB, rows_eff = min(RPB, rows - rbase);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, t4 = lane & 3;
    const int wrow = warp * 4;                            // first row of this warp

    for (int i = tid; i < NWARPS * 16 * HD; i += 32 * NWARPS) {
        const int w = i / (16 * HD), j = i - w * (16 * HD);
        const int pair = j >> 8, d = j & 255;
        const int r = w * 4 + (pair >> 2), h = pair & 3;
        float v = 0.f;
        if (r < rows_eff) v = q[(size_t)(rbase + r) * N_HEAD * HD + kvh * 4 * HD + h * HD + d] * 0.0625f;
        qs[(w * 16 + pair) * TC_STRIDE + d] = __float2half(v);
    }

    const int p0 = *pos + rbase;
    const int kend = p0 + rows_eff;
    const int kstart = is_swa ? max(0, p0 - SWA + 1) : 0;
    const int nkeys = kend - kstart;
    const int per = (nkeys + nsplit - 1) / nsplit;
    const int a0 = kstart + split * per, a1 = min(kend, a0 + per);
    const int pstride = is_swa ? SWA_SLOTS : g_kv_nctx;

    float m[2], l[2], o[HD / 8][4];
    #pragma unroll
    for (int j = 0; j < HD / 8; ++j) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) o[j][c] = 0.f;
    }
    m[0] = m[1] = -1e30f;
    l[0] = l[1] = 0.f;

    const int dk0 = tid >> 3;                       // key slot in the tile (0..15)
    const int sub = tid & 7;                        // 32-dim block inside the head
    for (int t0 = a0; t0 < a1; t0 += TC_T) {
        __syncthreads();
        {
            int kk = t0 + dk0;
            if (kk > a1 - 1) kk = a1 - 1;
            const int slot = is_swa ? (kk & (SWA_SLOTS - 1)) : kk;
            const size_t kb0 = (size_t)kvh * pstride * KV_KVH_BYTES + (size_t)slot * KV_KVH_BYTES;
            const size_t koff = kb0 + KV_SCALES_BYTES + sub * 32;
            const float dk = __half2float(((const __half*)(kvk + kb0))[sub]);
            const float dv = __half2float(((const __half*)(kvv + kb0))[sub]);
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                const uint4 wk = *(const uint4*)(kvk + koff + u * 16);
                const uint4 wv = *(const uint4*)(kvv + koff + u * 16);
                int bk[16], bv[16];
                bk[0]  = (int)(wk.x << 24) >> 24; bk[1]  = (int)(wk.x << 16) >> 24;
                bk[2]  = (int)(wk.x << 8) >> 24;  bk[3]  = (int)wk.x >> 24;
                bk[4]  = (int)(wk.y << 24) >> 24; bk[5]  = (int)(wk.y << 16) >> 24;
                bk[6]  = (int)(wk.y << 8) >> 24;  bk[7]  = (int)wk.y >> 24;
                bk[8]  = (int)(wk.z << 24) >> 24; bk[9]  = (int)(wk.z << 16) >> 24;
                bk[10] = (int)(wk.z << 8) >> 24;  bk[11] = (int)wk.z >> 24;
                bk[12] = (int)(wk.w << 24) >> 24; bk[13] = (int)(wk.w << 16) >> 24;
                bk[14] = (int)(wk.w << 8) >> 24;  bk[15] = (int)wk.w >> 24;
                bv[0]  = (int)(wv.x << 24) >> 24; bv[1]  = (int)(wv.x << 16) >> 24;
                bv[2]  = (int)(wv.x << 8) >> 24;  bv[3]  = (int)wv.x >> 24;
                bv[4]  = (int)(wv.y << 24) >> 24; bv[5]  = (int)(wv.y << 16) >> 24;
                bv[6]  = (int)(wv.y << 8) >> 24;  bv[7]  = (int)wv.y >> 24;
                bv[8]  = (int)(wv.z << 24) >> 24; bv[9]  = (int)(wv.z << 16) >> 24;
                bv[10] = (int)(wv.z << 8) >> 24;  bv[11] = (int)wv.z >> 24;
                bv[12] = (int)(wv.w << 24) >> 24; bv[13] = (int)(wv.w << 16) >> 24;
                bv[14] = (int)(wv.w << 8) >> 24;  bv[15] = (int)wv.w >> 24;
                __half2 hk[8], hv[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    hk[j] = __floats2half2_rn(dk * bk[2 * j], dk * bk[2 * j + 1]);
                    hv[j] = __floats2half2_rn(dv * bv[2 * j], dv * bv[2 * j + 1]);
                }
                uint4 pk0, pk1, pv0, pv1;
                pk0.x = *(uint32_t*)&hk[0]; pk0.y = *(uint32_t*)&hk[1];
                pk0.z = *(uint32_t*)&hk[2]; pk0.w = *(uint32_t*)&hk[3];
                pk1.x = *(uint32_t*)&hk[4]; pk1.y = *(uint32_t*)&hk[5];
                pk1.z = *(uint32_t*)&hk[6]; pk1.w = *(uint32_t*)&hk[7];
                pv0.x = *(uint32_t*)&hv[0]; pv0.y = *(uint32_t*)&hv[1];
                pv0.z = *(uint32_t*)&hv[2]; pv0.w = *(uint32_t*)&hv[3];
                pv1.x = *(uint32_t*)&hv[4]; pv1.y = *(uint32_t*)&hv[5];
                pv1.z = *(uint32_t*)&hv[6]; pv1.w = *(uint32_t*)&hv[7];
                *(uint4*)(&ksh[dk0 * TC_STRIDE + sub * 32 + u * 16]) = pk0;
                *(uint4*)(&ksh[dk0 * TC_STRIDE + sub * 32 + u * 16 + 8]) = pk1;
                *(uint4*)(&vsh[dk0 * TC_STRIDE + sub * 32 + u * 16]) = pv0;
                *(uint4*)(&vsh[dk0 * TC_STRIDE + sub * 32 + u * 16 + 8]) = pv1;
            }
        }
        __syncthreads();

        float s[2][4];
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int c = 0; c < 4; ++c) s[nt][c] = 0.f;

        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            const __half* qb = qs + (warp * 16 + gid) * TC_STRIDE + kk * 16 + 2 * t4;
            uint32_t a[4];
            a[0] = *(const uint32_t*)(qb);
            a[1] = *(const uint32_t*)(qb + 8 * TC_STRIDE);
            a[2] = *(const uint32_t*)(qb + 8);
            a[3] = *(const uint32_t*)(qb + 8 * TC_STRIDE + 8);
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const __half* kb = ksh + (nt * 8 + gid) * TC_STRIDE + kk * 16 + 2 * t4;
                uint32_t b[2] = { *(const uint32_t*)(kb), *(const uint32_t*)(kb + 8) };
                mma_16816_acc(s[nt], a, b);
            }
        }

        const bool full = (t0 + TC_T - 1 < a1) && (t0 + TC_T - 1 <= p0) &&
                          (!is_swa || t0 >= p0 + RPB - 1 - SWA + 1);
        const int r0 = wrow + (gid >> 2), r1 = r0 + 2;   // row offset inside the block
        const int lim0 = p0 + r0, lim1 = p0 + r1;
        const int lo0 = is_swa ? max(0, lim0 - SWA + 1) : 0;
        const int lo1 = is_swa ? max(0, lim1 - SWA + 1) : 0;
        float sv[8];
        sv[0] = s[0][0]; sv[1] = s[0][1]; sv[2] = s[0][2]; sv[3] = s[0][3];
        sv[4] = s[1][0]; sv[5] = s[1][1]; sv[6] = s[1][2]; sv[7] = s[1][3];
        if (!full) {
            const int kA = t0 + 2 * t4, kB = kA + 8;
            if (kA > lim0 || kA >= a1 || kA < lo0) sv[0] = -1e30f;
            if (kA + 1 > lim0 || kA + 1 >= a1 || kA + 1 < lo0) sv[1] = -1e30f;
            if (kB > lim0 || kB >= a1 || kB < lo0) sv[4] = -1e30f;
            if (kB + 1 > lim0 || kB + 1 >= a1 || kB + 1 < lo0) sv[5] = -1e30f;
            if (kA > lim1 || kA >= a1 || kA < lo1) sv[2] = -1e30f;
            if (kA + 1 > lim1 || kA + 1 >= a1 || kA + 1 < lo1) sv[3] = -1e30f;
            if (kB > lim1 || kB >= a1 || kB < lo1) sv[6] = -1e30f;
            if (kB + 1 > lim1 || kB + 1 >= a1 || kB + 1 < lo1) sv[7] = -1e30f;
        }
        float mx0 = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[4], sv[5]));
        float mx1 = fmaxf(fmaxf(sv[2], sv[3]), fmaxf(sv[6], sv[7]));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m[0], mx0), mn1 = fmaxf(m[1], mx1);
        const bool ok0 = mn0 > -1e29f, ok1 = mn1 > -1e29f;
        const float al0 = ok0 ? __expf(m[0] - mn0) : 1.f;
        const float al1 = ok1 ? __expf(m[1] - mn1) : 1.f;
        m[0] = mn0; m[1] = mn1;
        const float z0 = ok0 ? 1.f : 0.f, z1 = ok1 ? 1.f : 0.f;
        float e0 = __expf(sv[0] - mn0) * z0, e1 = __expf(sv[1] - mn0) * z0;
        float e2 = __expf(sv[2] - mn1) * z1, e3 = __expf(sv[3] - mn1) * z1;
        float e4 = __expf(sv[4] - mn0) * z0, e5 = __expf(sv[5] - mn0) * z0;
        float e6 = __expf(sv[6] - mn1) * z1, e7 = __expf(sv[7] - mn1) * z1;
        float sum0 = e0 + e1 + e4 + e5;
        float sum1 = e2 + e3 + e6 + e7;
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
        l[0] = l[0] * al0 + sum0;
        l[1] = l[1] * al1 + sum1;
        uint32_t pa[4];
        pa[0] = pack2(e0, e1);
        pa[1] = pack2(e2, e3);
        pa[2] = pack2(e4, e5);
        pa[3] = pack2(e6, e7);
        if (al0 != 1.f || al1 != 1.f) {
            #pragma unroll
            for (int j = 0; j < HD / 8; ++j) {
                o[j][0] *= al0; o[j][1] *= al0;
                o[j][2] *= al1; o[j][3] *= al1;
            }
        }

        #pragma unroll
        for (int j = 0; j < HD / 8; ++j) {
            const int dim = j * 8 + gid;
            const __half* vb = vsh + (2 * t4) * TC_STRIDE + dim;
            uint32_t b[2];
            b[0] = *(uint32_t*)&__halves2half2(vb[0], vb[TC_STRIDE]);
            b[1] = *(uint32_t*)&__halves2half2(vb[8 * TC_STRIDE], vb[9 * TC_STRIDE]);
            mma_16816_acc(o[j], pa, b);
        }
    }

    #pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
        const int pair = gid + hf * 8;
        const int r = wrow + (pair >> 2), h = pair & 3;
        if (r >= rows_eff) continue;
        const float il = l[hf] > 0.f ? 1.f / l[hf] : 0.f;
        float* base = 0;
        if (nsplit == 1) {
            base = out + ((size_t)(rbase + r) * N_HEAD + kvh * 4 + h) * HD;
        } else {
            base = part + (size_t)(rbase + r) * part_row_stride(nsplit) +
                   (((size_t)kvh * nsplit + split) * 4 + h) * (HD + 2);
        }
        const int d0 = 2 * t4;
        #pragma unroll
        for (int j = 0; j < HD / 8; ++j)
            if (nsplit == 1) {
                base[d0 + j * 8] = o[j][2 * hf] * il;
                base[d0 + j * 8 + 1] = o[j][2 * hf + 1] * il;
            } else {
                base[d0 + j * 8] = o[j][2 * hf];
                base[d0 + j * 8 + 1] = o[j][2 * hf + 1];
            }
        if (nsplit != 1 && t4 == 0) {
            base[HD] = m[hf];
            base[HD + 1] = l[hf];
        }
    }
}

// ---------------------------------------------------------------------------
// Optimised row-split prompt attention.
//
// Differences from k_attn_tc_rs:
//   * Q lives in registers (q_reg, 64 regs/lane) instead of a [16][264] fp16
//     shared tile, so dynamic shared memory drops from 50,688 B to 16,896 B and
//     two blocks fit per SM.
//   * O accumulates in half2 through mma.m16n8k16.f16.f16.f16.f16 (full rate on
//     GA10x) instead of fp32 (half rate), halving both the accumulator footprint
//     and the online-softmax rescale work.
//   * K and V fragments come from ldmatrix / ldmatrix.trans instead of 128
//     scalar LDS.U16 + PRMT + 64 LDS.U32 per key tile.
//   * blockIdx.x is kvh-major so all row chunks of a KV head share its L2 lines.
//
// Same contract as k_attn_tc_rs: unnormalized O + (m,l) in `part` when
// nsplit > 1, normalized O in `out` otherwise.
//
// LAZY adds the lazy-V traffic cut for the 1..8 row decode path: after the scores
// are known, a 16-key tile whose best score is more than 14 nats below the running
// max contributes exp(-14) < 8e-7 to the numerator, which is below fp16's
// resolution at the accumulated magnitude, so its V is never fetched, dequantised
// or multiplied.  The denominator l still includes the tile, so nothing but
// sub-ulp terms is dropped.  Prompt chunks keep the dense form (LAZY = false), so
// prefill is untouched.
template <int NWARPS, int MINBLK, bool LAZY = false, bool PIPE = false, bool DBUF = false>
__global__ void __launch_bounds__(32 * NWARPS, MINBLK) k_attn_tc_rs_opt(
        float* __restrict__ out, const float* __restrict__ q,
        const uint8_t* __restrict__ kvk, const uint8_t* __restrict__ kvv,
        const int* __restrict__ pos, float* __restrict__ part, int rows, int is_swa) {
    extern __shared__ __align__(16) __half rs_sh[];
    // DBUF: two fp16 K/V tiles and two raw q8 tiles, so the dequant of tile
    // t+1 writes a buffer nobody reads and the tile loop needs one barrier
    // instead of two (and the copy can be issued two tiles ahead).
    constexpr int NB = DBUF ? 2 : 1;
    __half* ksh_all = rs_sh;                                   // [NB][TC_T][TC_STRIDE]
    __half* vsh_all = ksh_all + NB * TC_T * TC_STRIDE;         // [NB][TC_T][TC_STRIDE]
    uint8_t* raw_all = (uint8_t*)(vsh_all + NB * TC_T * TC_STRIDE);  // [NB][TC_T][272]
    uint8_t* rawv_all = raw_all + NB * TC_T * KV_KVH_BYTES;
    __half* ksh = ksh_all;   // non-DBUF alias (also the lazy-V staging path)
    __half* vsh = vsh_all;
    constexpr int RAW_PER = KV_KVH_BYTES / 16;
    const int nthr = 32 * NWARPS;

    const int n_rchunks = gridDim.x / N_KV;
    const int kvh = blockIdx.x / n_rchunks;
    // Longest causal ranges first: the final partial wave is the cheapest one.
    const int rchunk = n_rchunks - 1 - (blockIdx.x - kvh * n_rchunks);
    const int split = blockIdx.y, nsplit = gridDim.y;
    const int RPB = 4 * NWARPS;
    const int rbase = rchunk * RPB, rows_eff = min(RPB, rows - rbase);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, t4 = lane & 3;
    const int wrow = warp * 4;
    const int ldm_r8 = lane & 7, ldm_half = ((lane >> 3) & 1) * 8;

    // Q fragment, pre-scaled by 1/sqrt(HD).  a0/a1 = cols 2*t4, a2/a3 = +8;
    // pairs gid and gid+8 are (row gid>>2, head gid&3) and row+2.
    uint32_t q_reg[16][4];
    #pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
        #pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const int pair = gid + hf * 8;
            const int r = wrow + (pair >> 2), h = pair & 3;
            const int rr = r < rows_eff ? r : 0;
            const float* qp = q + (size_t)(rbase + rr) * N_HEAD * HD
                                + (size_t)(kvh * 4 + h) * HD + kk * 16 + 2 * t4;
            const float sc = (r < rows_eff) ? 0.0625f : 0.f;
            const float2 v0 = *(const float2*)(qp), v1 = *(const float2*)(qp + 8);
            q_reg[kk][hf]     = pack2(v0.x * sc, v0.y * sc);
            q_reg[kk][hf + 2] = pack2(v1.x * sc, v1.y * sc);
        }
    }

    const int p0 = *pos + rbase;
    const int kend = p0 + rows_eff;
    const int kstart = is_swa ? max(0, p0 - SWA + 1) : 0;
    const int nkeys = kend - kstart;
    const int per = (nkeys + nsplit - 1) / nsplit;
    const int a0 = kstart + split * per, a1 = min(kend, a0 + per);
    const int pstride = is_swa ? SWA_SLOTS : g_kv_nctx;

    float m[2] = {-1e30f, -1e30f}, l[2] = {0.f, 0.f};
    uint32_t o[HD / 8][2];
    #pragma unroll
    for (int j = 0; j < HD / 8; ++j) { o[j][0] = 0u; o[j][1] = 0u; }

    const int dk0 = tid >> 3, sub = tid & 7;
    const __half2 h1152 = __float2half2_rn(1152.0f);
    const uint8_t* kbase_h = kvk + (size_t)kvh * pstride * KV_KVH_BYTES;
    const uint8_t* vbase_h = kvv + (size_t)kvh * pstride * KV_KVH_BYTES;
    // staged by the first 128 threads; both the dense and the lazy path use it
    auto stage_v = [&](int t0) {
        if (tid < 128) {
            const int kk = min(t0 + dk0, a1 - 1);
            const int slot = is_swa ? (kk & (SWA_SLOTS - 1)) : kk;
            const size_t kb0 = (size_t)slot * KV_KVH_BYTES;
            const size_t koff = kb0 + KV_SCALES_BYTES + sub * 32;
            const uint4 wv0 = *(const uint4*)(vbase_h + koff);
            const uint4 wv1 = *(const uint4*)(vbase_h + koff + 16);
            const __half2 dv2 = __half2half2(((const __half*)(vbase_h + kb0))[sub]);
            __half2 hv[8];
            dequant_i8x4_to_h2(wv0.x, dv2, h1152, hv[0], hv[1]);
            dequant_i8x4_to_h2(wv0.y, dv2, h1152, hv[2], hv[3]);
            dequant_i8x4_to_h2(wv0.z, dv2, h1152, hv[4], hv[5]);
            dequant_i8x4_to_h2(wv0.w, dv2, h1152, hv[6], hv[7]);
            *(uint4*)(&vsh[dk0 * TC_STRIDE + sub * 32])     = *(uint4*)&hv[0];
            *(uint4*)(&vsh[dk0 * TC_STRIDE + sub * 32 + 8]) = *(uint4*)&hv[4];
            dequant_i8x4_to_h2(wv1.x, dv2, h1152, hv[0], hv[1]);
            dequant_i8x4_to_h2(wv1.y, dv2, h1152, hv[2], hv[3]);
            dequant_i8x4_to_h2(wv1.z, dv2, h1152, hv[4], hv[5]);
            dequant_i8x4_to_h2(wv1.w, dv2, h1152, hv[6], hv[7]);
            *(uint4*)(&vsh[dk0 * TC_STRIDE + sub * 32 + 16]) = *(uint4*)&hv[0];
            *(uint4*)(&vsh[dk0 * TC_STRIDE + sub * 32 + 24]) = *(uint4*)&hv[4];
        }
    };
    // PIPE: raw q8 tile -> cp.async, issued one tile ahead of the compute.
    auto stage_raw = [&](int t0, int rb) {
        uint8_t* rk = raw_all + (size_t)rb * TC_T * KV_KVH_BYTES;
        uint8_t* rv = rawv_all + (size_t)rb * TC_T * KV_KVH_BYTES;
        for (int c = tid; c < TC_T * RAW_PER; c += nthr) {
            const int key = c / RAW_PER, chunk = c - key * RAW_PER;
            const int kk = min(t0 + key, a1 - 1);
            const int slot = is_swa ? (kk & (SWA_SLOTS - 1)) : kk;
            mma_cp16(rk + key * KV_KVH_BYTES + chunk * 16,
                     kbase_h + (size_t)slot * KV_KVH_BYTES + chunk * 16);
            mma_cp16(rv + key * KV_KVH_BYTES + chunk * 16,
                     vbase_h + (size_t)slot * KV_KVH_BYTES + chunk * 16);
        }
        mma_cp_commit();
    };
    // raw q8 tile -> fp16 K/V tile.  16 threads per key on wide CTAs: the
    // dequant sits on the tile's critical path, so halving the per-thread
    // chain (16 B of K and V instead of 32 B) halves that serial part.
    auto dequant_tile = [&](const uint8_t* rk, const uint8_t* rv, __half* kd, __half* vd) {
        if constexpr (NWARPS >= 8) {
            if (tid < 16 * TC_T) {
                const int dk = tid >> 4, sp = tid & 15;        // sp = 16-byte piece
                const int g = sp >> 1;                         // 32-value scale group
                const uint4 wk = *(const uint4*)(rk + dk * KV_KVH_BYTES + KV_SCALES_BYTES + sp * 16);
                const uint4 wv = *(const uint4*)(rv + dk * KV_KVH_BYTES + KV_SCALES_BYTES + sp * 16);
                const __half2 dk2 = __half2half2(*(const __half*)(rk + dk * KV_KVH_BYTES + g * 2));
                const __half2 dv2 = __half2half2(*(const __half*)(rv + dk * KV_KVH_BYTES + g * 2));
                __half2 hk[8], hv[8];
                dequant_i8x4_to_h2(wk.x, dk2, h1152, hk[0], hk[1]);
                dequant_i8x4_to_h2(wk.y, dk2, h1152, hk[2], hk[3]);
                dequant_i8x4_to_h2(wk.z, dk2, h1152, hk[4], hk[5]);
                dequant_i8x4_to_h2(wk.w, dk2, h1152, hk[6], hk[7]);
                *(uint4*)(&kd[dk * TC_STRIDE + sp * 16])     = *(uint4*)&hk[0];
                *(uint4*)(&kd[dk * TC_STRIDE + sp * 16 + 8]) = *(uint4*)&hk[4];
                dequant_i8x4_to_h2(wv.x, dv2, h1152, hv[0], hv[1]);
                dequant_i8x4_to_h2(wv.y, dv2, h1152, hv[2], hv[3]);
                dequant_i8x4_to_h2(wv.z, dv2, h1152, hv[4], hv[5]);
                dequant_i8x4_to_h2(wv.w, dv2, h1152, hv[6], hv[7]);
                *(uint4*)(&vd[dk * TC_STRIDE + sp * 16])     = *(uint4*)&hv[0];
                *(uint4*)(&vd[dk * TC_STRIDE + sp * 16 + 8]) = *(uint4*)&hv[4];
            }
        } else if (tid < 128) {
            const uint4 wk0 = *(const uint4*)(rk + dk0 * KV_KVH_BYTES + KV_SCALES_BYTES + sub * 32);
            const uint4 wk1 = *(const uint4*)(rk + dk0 * KV_KVH_BYTES + KV_SCALES_BYTES + sub * 32 + 16);
            const uint4 wv0 = *(const uint4*)(rv + dk0 * KV_KVH_BYTES + KV_SCALES_BYTES + sub * 32);
            const uint4 wv1 = *(const uint4*)(rv + dk0 * KV_KVH_BYTES + KV_SCALES_BYTES + sub * 32 + 16);
            const __half2 dk2 = __half2half2(*(const __half*)(rk + dk0 * KV_KVH_BYTES + sub * 2));
            const __half2 dv2 = __half2half2(*(const __half*)(rv + dk0 * KV_KVH_BYTES + sub * 2));
            __half2 hk[8], hv[8];
            dequant_i8x4_to_h2(wk0.x, dk2, h1152, hk[0], hk[1]);
            dequant_i8x4_to_h2(wk0.y, dk2, h1152, hk[2], hk[3]);
            dequant_i8x4_to_h2(wk0.z, dk2, h1152, hk[4], hk[5]);
            dequant_i8x4_to_h2(wk0.w, dk2, h1152, hk[6], hk[7]);
            *(uint4*)(&kd[dk0 * TC_STRIDE + sub * 32])     = *(uint4*)&hk[0];
            *(uint4*)(&kd[dk0 * TC_STRIDE + sub * 32 + 8]) = *(uint4*)&hk[4];
            dequant_i8x4_to_h2(wk1.x, dk2, h1152, hk[0], hk[1]);
            dequant_i8x4_to_h2(wk1.y, dk2, h1152, hk[2], hk[3]);
            dequant_i8x4_to_h2(wk1.z, dk2, h1152, hk[4], hk[5]);
            dequant_i8x4_to_h2(wk1.w, dk2, h1152, hk[6], hk[7]);
            *(uint4*)(&kd[dk0 * TC_STRIDE + sub * 32 + 16]) = *(uint4*)&hk[0];
            *(uint4*)(&kd[dk0 * TC_STRIDE + sub * 32 + 24]) = *(uint4*)&hk[4];
            dequant_i8x4_to_h2(wv0.x, dv2, h1152, hv[0], hv[1]);
            dequant_i8x4_to_h2(wv0.y, dv2, h1152, hv[2], hv[3]);
            dequant_i8x4_to_h2(wv0.z, dv2, h1152, hv[4], hv[5]);
            dequant_i8x4_to_h2(wv0.w, dv2, h1152, hv[6], hv[7]);
            *(uint4*)(&vd[dk0 * TC_STRIDE + sub * 32])     = *(uint4*)&hv[0];
            *(uint4*)(&vd[dk0 * TC_STRIDE + sub * 32 + 8]) = *(uint4*)&hv[4];
            dequant_i8x4_to_h2(wv1.x, dv2, h1152, hv[0], hv[1]);
            dequant_i8x4_to_h2(wv1.y, dv2, h1152, hv[2], hv[3]);
            dequant_i8x4_to_h2(wv1.z, dv2, h1152, hv[4], hv[5]);
            dequant_i8x4_to_h2(wv1.w, dv2, h1152, hv[6], hv[7]);
            *(uint4*)(&vd[dk0 * TC_STRIDE + sub * 32 + 16]) = *(uint4*)&hv[0];
            *(uint4*)(&vd[dk0 * TC_STRIDE + sub * 32 + 24]) = *(uint4*)&hv[4];
        }
    };
    if constexpr (PIPE) {
        stage_raw(a0, 0);
        mma_cp_wait();
        if constexpr (DBUF) {
            dequant_tile(raw_all, rawv_all, ksh_all, vsh_all);
            __syncthreads();
            if (a0 + TC_T < a1) stage_raw(a0 + TC_T, 1);
        }
    }
    int tile_i = 0;
    for (int t0 = a0; t0 < a1; t0 += TC_T, ++tile_i) {
        const bool full = (t0 + TC_T - 1 < a1) && (t0 + TC_T - 1 <= p0) &&
                          (!is_swa || t0 >= p0 + RPB - 1 - SWA + 1);
        __half* ksh_c = ksh_all;
        __half* vsh_c = vsh_all;
        if constexpr (DBUF) {
            // fp16 double buffer: dequantise tile t+1 into the other buffer
            // while this warp still works on tile t; one barrier per tile.
            const int buf = tile_i & 1;
            ksh_c = ksh_all + (size_t)buf * TC_T * TC_STRIDE;
            vsh_c = vsh_all + (size_t)buf * TC_T * TC_STRIDE;
            if (t0 + TC_T < a1) {
                const int nb = buf ^ 1;
                mma_cp_wait();
                dequant_tile(raw_all + (size_t)nb * TC_T * KV_KVH_BYTES,
                             rawv_all + (size_t)nb * TC_T * KV_KVH_BYTES,
                             ksh_all + (size_t)nb * TC_T * TC_STRIDE,
                             vsh_all + (size_t)nb * TC_T * TC_STRIDE);
                if (t0 + 2 * TC_T < a1) stage_raw(t0 + 2 * TC_T, buf);
            }
        } else {
        __syncthreads();
        if constexpr (PIPE) {
            dequant_tile(raw_all, rawv_all, ksh, vsh);
            __syncthreads();
            if (t0 + TC_T < a1) stage_raw(t0 + TC_T, 0);
        } else
        if (tid < 128) {
            const int kk = min(t0 + dk0, a1 - 1);
            const int slot = is_swa ? (kk & (SWA_SLOTS - 1)) : kk;
            const size_t kb0 = (size_t)slot * KV_KVH_BYTES;
            const size_t koff = kb0 + KV_SCALES_BYTES + sub * 32;
            // Issue all four 128-bit KV loads before the dequant chain (4-deep MLP).
            const uint4 wk0 = *(const uint4*)(kbase_h + koff);
            const uint4 wk1 = *(const uint4*)(kbase_h + koff + 16);
            const uint4 wv0 = LAZY ? make_uint4(0, 0, 0, 0) : *(const uint4*)(vbase_h + koff);
            const uint4 wv1 = LAZY ? make_uint4(0, 0, 0, 0) : *(const uint4*)(vbase_h + koff + 16);
            const __half2 dk2 = __half2half2(((const __half*)(kbase_h + kb0))[sub]);
            const __half2 dv2 = LAZY ? __half2half2(__float2half(0.f))
                                     : __half2half2(((const __half*)(vbase_h + kb0))[sub]);
            __half2 hk[8], hv[8];
            dequant_i8x4_to_h2(wk0.x, dk2, h1152, hk[0], hk[1]);
            dequant_i8x4_to_h2(wk0.y, dk2, h1152, hk[2], hk[3]);
            dequant_i8x4_to_h2(wk0.z, dk2, h1152, hk[4], hk[5]);
            dequant_i8x4_to_h2(wk0.w, dk2, h1152, hk[6], hk[7]);
            *(uint4*)(&ksh[dk0 * TC_STRIDE + sub * 32])     = *(uint4*)&hk[0];
            *(uint4*)(&ksh[dk0 * TC_STRIDE + sub * 32 + 8]) = *(uint4*)&hk[4];
            dequant_i8x4_to_h2(wk1.x, dk2, h1152, hk[0], hk[1]);
            dequant_i8x4_to_h2(wk1.y, dk2, h1152, hk[2], hk[3]);
            dequant_i8x4_to_h2(wk1.z, dk2, h1152, hk[4], hk[5]);
            dequant_i8x4_to_h2(wk1.w, dk2, h1152, hk[6], hk[7]);
            *(uint4*)(&ksh[dk0 * TC_STRIDE + sub * 32 + 16]) = *(uint4*)&hk[0];
            *(uint4*)(&ksh[dk0 * TC_STRIDE + sub * 32 + 24]) = *(uint4*)&hk[4];

            if (!LAZY) {
                dequant_i8x4_to_h2(wv0.x, dv2, h1152, hv[0], hv[1]);
                dequant_i8x4_to_h2(wv0.y, dv2, h1152, hv[2], hv[3]);
                dequant_i8x4_to_h2(wv0.z, dv2, h1152, hv[4], hv[5]);
                dequant_i8x4_to_h2(wv0.w, dv2, h1152, hv[6], hv[7]);
                *(uint4*)(&vsh[dk0 * TC_STRIDE + sub * 32])     = *(uint4*)&hv[0];
                *(uint4*)(&vsh[dk0 * TC_STRIDE + sub * 32 + 8]) = *(uint4*)&hv[4];
                dequant_i8x4_to_h2(wv1.x, dv2, h1152, hv[0], hv[1]);
                dequant_i8x4_to_h2(wv1.y, dv2, h1152, hv[2], hv[3]);
                dequant_i8x4_to_h2(wv1.z, dv2, h1152, hv[4], hv[5]);
                dequant_i8x4_to_h2(wv1.w, dv2, h1152, hv[6], hv[7]);
                *(uint4*)(&vsh[dk0 * TC_STRIDE + sub * 32 + 16]) = *(uint4*)&hv[0];
                *(uint4*)(&vsh[dk0 * TC_STRIDE + sub * 32 + 24]) = *(uint4*)&hv[4];
            }
        }
        if constexpr (!PIPE) __syncthreads();
        }

        uint32_t sh16[2][2] = {{0u, 0u}, {0u, 0u}};
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            uint32_t b0[2], b1[2];
            ldmatrix2(b0, &ksh_c[(0 + ldm_r8) * TC_STRIDE + kk * 16 + ldm_half]);
            ldmatrix2(b1, &ksh_c[(8 + ldm_r8) * TC_STRIDE + kk * 16 + ldm_half]);
            mma_16816_f16(sh16[0], q_reg[kk], b0);
            mma_16816_f16(sh16[1], q_reg[kk], b1);
        }
        const float2 s0l = __half22float2(*(const __half2*)&sh16[0][0]);
        const float2 s0h = __half22float2(*(const __half2*)&sh16[0][1]);
        const float2 s1l = __half22float2(*(const __half2*)&sh16[1][0]);
        const float2 s1h = __half22float2(*(const __half2*)&sh16[1][1]);

        const int r0 = wrow + (gid >> 2), r1 = r0 + 2;
        const int lim0 = p0 + r0, lim1 = p0 + r1;
        const int lo0 = is_swa ? max(0, lim0 - SWA + 1) : 0;
        const int lo1 = is_swa ? max(0, lim1 - SWA + 1) : 0;
        float sv[8];
        sv[0] = s0l.x; sv[1] = s0l.y; sv[2] = s0h.x; sv[3] = s0h.y;
        sv[4] = s1l.x; sv[5] = s1l.y; sv[6] = s1h.x; sv[7] = s1h.y;
        if (!full) {
            const int kA = t0 + 2 * t4, kB = kA + 8;
            if (kA > lim0 || kA >= a1 || kA < lo0) sv[0] = -1e30f;
            if (kA + 1 > lim0 || kA + 1 >= a1 || kA + 1 < lo0) sv[1] = -1e30f;
            if (kB > lim0 || kB >= a1 || kB < lo0) sv[4] = -1e30f;
            if (kB + 1 > lim0 || kB + 1 >= a1 || kB + 1 < lo0) sv[5] = -1e30f;
            if (kA > lim1 || kA >= a1 || kA < lo1) sv[2] = -1e30f;
            if (kA + 1 > lim1 || kA + 1 >= a1 || kA + 1 < lo1) sv[3] = -1e30f;
            if (kB > lim1 || kB >= a1 || kB < lo1) sv[6] = -1e30f;
            if (kB + 1 > lim1 || kB + 1 >= a1 || kB + 1 < lo1) sv[7] = -1e30f;
        }
        float mx0 = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[4], sv[5]));
        float mx1 = fmaxf(fmaxf(sv[2], sv[3]), fmaxf(sv[6], sv[7]));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m[0], mx0), mn1 = fmaxf(m[1], mx1);
        float al0, al1, e0, e1, e2, e3, e4, e5, e6, e7;
        if (full) {
            al0 = __expf(m[0] - mn0); al1 = __expf(m[1] - mn1);
            m[0] = mn0; m[1] = mn1;
            e0 = __expf(sv[0] - mn0); e1 = __expf(sv[1] - mn0);
            e2 = __expf(sv[2] - mn1); e3 = __expf(sv[3] - mn1);
            e4 = __expf(sv[4] - mn0); e5 = __expf(sv[5] - mn0);
            e6 = __expf(sv[6] - mn1); e7 = __expf(sv[7] - mn1);
        } else {
            const bool ok0 = mn0 > -1e29f, ok1 = mn1 > -1e29f;
            al0 = ok0 ? __expf(m[0] - mn0) : 1.f;
            al1 = ok1 ? __expf(m[1] - mn1) : 1.f;
            m[0] = mn0; m[1] = mn1;
            const float z0 = ok0 ? 1.f : 0.f, z1 = ok1 ? 1.f : 0.f;
            e0 = __expf(sv[0] - mn0) * z0; e1 = __expf(sv[1] - mn0) * z0;
            e2 = __expf(sv[2] - mn1) * z1; e3 = __expf(sv[3] - mn1) * z1;
            e4 = __expf(sv[4] - mn0) * z0; e5 = __expf(sv[5] - mn0) * z0;
            e6 = __expf(sv[6] - mn1) * z1; e7 = __expf(sv[7] - mn1) * z1;
        }
        float sum0 = e0 + e1 + e4 + e5;
        float sum1 = e2 + e3 + e6 + e7;
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
        l[0] = l[0] * al0 + sum0;
        l[1] = l[1] * al1 + sum1;
        uint32_t pa[4];
        pa[0] = pack2(e0, e1);
        pa[1] = pack2(e2, e3);
        pa[2] = pack2(e4, e5);
        pa[3] = pack2(e6, e7);
        if (al0 != 1.f || al1 != 1.f) {
            const __half2 a0h = __floats2half2_rn(al0, al0);
            const __half2 a1h = __floats2half2_rn(al1, al1);
            #pragma unroll
            for (int j = 0; j < HD / 8; ++j) {
                *(__half2*)&o[j][0] = __hmul2(*(__half2*)&o[j][0], a0h);
                *(__half2*)&o[j][1] = __hmul2(*(__half2*)&o[j][1], a1h);
            }
        }
        if (LAZY) {
            // Skip V entirely when this tile cannot move the numerator: exp(-14) is
            // below fp16 resolution at the accumulated magnitude.  The vote is
            // block-wide because V is staged cooperatively.
            const bool hot = (mx0 > mn0 - 14.f) || (mx1 > mn1 - 14.f);
            if (__syncthreads_or((int)hot)) {
                stage_v(t0);
                __syncthreads();
            } else {
                continue;
            }
        }
        #pragma unroll
        for (int j = 0; j < HD / 8; ++j) {
            uint32_t b[2];
            ldmatrix2_trans(b, &vsh_c[(ldm_half + ldm_r8) * TC_STRIDE + j * 8]);
            mma_16816_f16(o[j], pa, b);
        }
        if constexpr (DBUF) {
            __syncthreads();   // next tile's fp16 buffer is ready for every warp
        } else if constexpr (PIPE) { if (t0 + TC_T < a1) mma_cp_wait(); }
    }

    #pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
        const int pair = gid + hf * 8;
        const int r = wrow + (pair >> 2), h = pair & 3;
        if (r >= rows_eff) continue;
        const float il = l[hf] > 0.f ? 1.f / l[hf] : 0.f;
        float* base = 0;
        if (nsplit == 1) {
            base = out + ((size_t)(rbase + r) * N_HEAD + kvh * 4 + h) * HD;
        } else {
            base = part + (size_t)(rbase + r) * part_row_stride(nsplit) +
                   (((size_t)kvh * nsplit + split) * 4 + h) * (HD + 2);
        }
        const int d0 = 2 * t4;
        #pragma unroll
        for (int j = 0; j < HD / 8; ++j) {
            const float2 v = __half22float2(*(const __half2*)&o[j][hf]);
            if (nsplit == 1) {
                base[d0 + j * 8]     = v.x * il;
                base[d0 + j * 8 + 1] = v.y * il;
            } else {
                base[d0 + j * 8]     = v.x;
                base[d0 + j * 8 + 1] = v.y;
            }
        }
        if (nsplit != 1 && t4 == 0) {
            base[HD] = m[hf];
            base[HD + 1] = l[hf];
        }
    }
}

// ---------------------------------------------------------------------------
// Warp-independent decode/verify attention (1..4 rows).
//
// Each warp owns ALL 256 head dims of Q and O for one independent split, so it
// never shares keys with another warp: the whole kernel needs only __syncwarp()
// and 33.8 KB per 4-warp block (2 blocks/SM).  Q lives in registers (q_reg) and
// O in half2 (full-rate mma.m16n8k16.f16), K/V fragments come from ldmatrix /
// ldmatrix.trans, and the shared tile is reused for V after QK^T.
//
// Lazy V: V is fetched from DRAM only when the 16-key tile's score maximum is
// within 14 nats of the running max.  Beyond the sink and the local window that
// is false for >95% of tiles, which halves long-context KV traffic without
// changing the result (exp(-14) is below fp16 subnormal weight).
// ---------------------------------------------------------------------------
template <int M>
__global__ void __launch_bounds__(128, 2) k_attn_tc_wi(
        float* __restrict__ out, const float* __restrict__ q,
        const uint8_t* __restrict__ kvk, const uint8_t* __restrict__ kvv,
        const int* __restrict__ pos, float* __restrict__ part, int is_swa, int ns) {
    extern __shared__ __align__(16) __half wi_sh[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    __half* ws = wi_sh + (size_t)warp * 16 * TC_STRIDE;   // this warp's K tile, then V
    const int kvh = blockIdx.x;
    const int split = blockIdx.y * 4 + warp;
    if (split >= ns) return;

    const int gid = lane >> 2, t4 = lane & 3;
    const int ldm_r8 = lane & 7, ldm_half = ((lane >> 3) & 1) * 8;

    const int p0 = *pos;
    const int kend = p0 + M;
    const int kstart = is_swa ? max(0, p0 - SWA + 1) : 0;
    const int nkeys = kend - kstart;
    const int per = (nkeys + ns - 1) / ns;
    const int a0 = kstart + split * per, a1 = min(kend, a0 + per);
    const int pstride = is_swa ? SWA_SLOTS : g_kv_nctx;
    const uint8_t* kbase = kvk + (size_t)kvh * pstride * KV_KVH_BYTES;
    const uint8_t* vbase = kvv + (size_t)kvh * pstride * KV_KVH_BYTES;

    // Q fragment: pair = gid (+8) is (row, head) = (pair>>2, pair&3)
    uint32_t q_reg[16][4];
    #pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
        #pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const int pair = gid + hf * 8;
            const int r = pair >> 2, h = pair & 3;
            const bool ok = r < M;
            const float* qp = q + (size_t)(ok ? r : 0) * N_HEAD * HD
                                + (size_t)(kvh * 4 + h) * HD + kk * 16 + 2 * t4;
            const float sc = ok ? 0.0625f : 0.f;
            const float2 v0 = *(const float2*)(qp), v1 = *(const float2*)(qp + 8);
            q_reg[kk][hf]     = pack2(v0.x * sc, v0.y * sc);
            q_reg[kk][hf + 2] = pack2(v1.x * sc, v1.y * sc);
        }
    }

    float m[2] = {-1e30f, -1e30f}, l[2] = {0.f, 0.f};
    uint32_t o[HD / 8][2];
    #pragma unroll
    for (int j = 0; j < HD / 8; ++j) { o[j][0] = 0u; o[j][1] = 0u; }

    const bool row0_ok = (gid >> 2) < M;
    const bool row1_ok = ((gid + 8) >> 2) < M;
    const int lim0 = p0 + (gid >> 2), lim1 = p0 + ((gid + 8) >> 2);
    const int lo0 = is_swa ? max(0, lim0 - SWA + 1) : 0;
    const int lo1 = is_swa ? max(0, lim1 - SWA + 1) : 0;

    for (int t0 = a0; t0 < a1; t0 += TC_T) {
        // ---- dequantize 16 keys of K into this warp's shared tile
        __syncwarp();
        {
            const int key = t0 + (lane >> 1);
            const int kk = key < a1 ? key : a1 - 1;
            const int slot = is_swa ? (kk & (SWA_SLOTS - 1)) : kk;
            const size_t kb0 = (size_t)kvh * pstride * KV_KVH_BYTES + (size_t)slot * KV_KVH_BYTES;
            const int half8 = lane & 1;              // this lane's 128-dim half
            const int8_t* src = (const int8_t*)(kvk + kb0 + KV_SCALES_BYTES) + half8 * 128;
            const __half* sc = (const __half*)(kvk + kb0) + half8 * 4;
            uint4* dst = (uint4*)(ws + (lane >> 1) * TC_STRIDE + half8 * 128);
            #pragma unroll
            for (int u = 0; u < 16; ++u) {
                const int8_t* p = src + u * 8;
                const __half2 d2 = __half2half2(sc[u >> 2]);
                uint4 v;
                v.x = *(uint32_t*)&__hmul2(d2, __floats2half2_rn((float)p[0], (float)p[1]));
                v.y = *(uint32_t*)&__hmul2(d2, __floats2half2_rn((float)p[2], (float)p[3]));
                v.z = *(uint32_t*)&__hmul2(d2, __floats2half2_rn((float)p[4], (float)p[5]));
                v.w = *(uint32_t*)&__hmul2(d2, __floats2half2_rn((float)p[6], (float)p[7]));
                dst[u] = v;
            }
        }
        __syncwarp();

        // ---- QK^T: sh16[0] = keys 0..7, sh16[1] = keys 8..15
        uint32_t sh16[2][2] = {{0u, 0u}, {0u, 0u}};
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            uint32_t b0[2], b1[2];
            ldmatrix2(b0, &ws[(0 + ldm_r8) * TC_STRIDE + kk * 16 + ldm_half]);
            ldmatrix2(b1, &ws[(8 + ldm_r8) * TC_STRIDE + kk * 16 + ldm_half]);
            mma_16816_f16(sh16[0], q_reg[kk], b0);
            mma_16816_f16(sh16[1], q_reg[kk], b1);
        }
        const float2 s0l = __half22float2(*(const __half2*)&sh16[0][0]);
        const float2 s0h = __half22float2(*(const __half2*)&sh16[0][1]);
        const float2 s1l = __half22float2(*(const __half2*)&sh16[1][0]);
        const float2 s1h = __half22float2(*(const __half2*)&sh16[1][1]);
        float sv[8] = {s0l.x, s0l.y, s0h.x, s0h.y, s1l.x, s1l.y, s1h.x, s1h.y};
        const int kA = t0 + 2 * t4, kB = kA + 8;
        const bool full = (t0 + TC_T - 1 < a1) && (t0 + TC_T - 1 <= p0) &&
                          (!is_swa || t0 >= p0 + M - 1 - SWA + 1);
        if (!full) {
            if (kA > lim0 || kA >= a1 || kA < lo0) sv[0] = -1e30f;
            if (kA + 1 > lim0 || kA + 1 >= a1 || kA + 1 < lo0) sv[1] = -1e30f;
            if (kA > lim1 || kA >= a1 || kA < lo1) sv[2] = -1e30f;
            if (kA + 1 > lim1 || kA + 1 >= a1 || kA + 1 < lo1) sv[3] = -1e30f;
            if (kB > lim0 || kB >= a1 || kB < lo0) sv[4] = -1e30f;
            if (kB + 1 > lim0 || kB + 1 >= a1 || kB + 1 < lo0) sv[5] = -1e30f;
            if (kB > lim1 || kB >= a1 || kB < lo1) sv[6] = -1e30f;
            if (kB + 1 > lim1 || kB + 1 >= a1 || kB + 1 < lo1) sv[7] = -1e30f;
        }
        float mx0 = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[4], sv[5]));
        float mx1 = fmaxf(fmaxf(sv[2], sv[3]), fmaxf(sv[6], sv[7]));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const bool hot = (row0_ok && mx0 - m[0] > -14.f) || (row1_ok && mx1 - m[1] > -14.f);
        const float mn0 = fmaxf(m[0], mx0), mn1 = fmaxf(m[1], mx1);
        float al0, al1, e0, e1, e2, e3, e4, e5, e6, e7;
        if (full) {
            al0 = __expf(m[0] - mn0); al1 = __expf(m[1] - mn1);
            m[0] = mn0; m[1] = mn1;
            e0 = __expf(sv[0] - mn0); e1 = __expf(sv[1] - mn0);
            e2 = __expf(sv[2] - mn1); e3 = __expf(sv[3] - mn1);
            e4 = __expf(sv[4] - mn0); e5 = __expf(sv[5] - mn0);
            e6 = __expf(sv[6] - mn1); e7 = __expf(sv[7] - mn1);
        } else {
            const bool ok0 = mn0 > -1e29f, ok1 = mn1 > -1e29f;
            al0 = ok0 ? __expf(m[0] - mn0) : 1.f;
            al1 = ok1 ? __expf(m[1] - mn1) : 1.f;
            m[0] = mn0; m[1] = mn1;
            const float z0 = ok0 ? 1.f : 0.f, z1 = ok1 ? 1.f : 0.f;
            e0 = __expf(sv[0] - mn0) * z0; e1 = __expf(sv[1] - mn0) * z0;
            e2 = __expf(sv[2] - mn1) * z1; e3 = __expf(sv[3] - mn1) * z1;
            e4 = __expf(sv[4] - mn0) * z0; e5 = __expf(sv[5] - mn0) * z0;
            e6 = __expf(sv[6] - mn1) * z1; e7 = __expf(sv[7] - mn1) * z1;
        }
        float sum0 = e0 + e1 + e4 + e5;
        float sum1 = e2 + e3 + e6 + e7;
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
        l[0] = l[0] * al0 + sum0;
        l[1] = l[1] * al1 + sum1;
        uint32_t pa[4];
        pa[0] = pack2(e0, e1);
        pa[1] = pack2(e2, e3);
        pa[2] = pack2(e4, e5);
        pa[3] = pack2(e6, e7);
        if (al0 != 1.f || al1 != 1.f) {
            const __half2 a0h = __floats2half2_rn(al0, al0);
            const __half2 a1h = __floats2half2_rn(al1, al1);
            #pragma unroll
            for (int j = 0; j < HD / 8; ++j) {
                *(__half2*)&o[j][0] = __hmul2(*(__half2*)&o[j][0], a0h);
                *(__half2*)&o[j][1] = __hmul2(*(__half2*)&o[j][1], a1h);
            }
        }

        if (!__any_sync(0xffffffffu, hot)) continue;
        // ---- dequantize the same 16 keys of V into this warp's shared tile
        __syncwarp();
        {
            const int key = t0 + (lane >> 1);
            const int kk = key < a1 ? key : a1 - 1;
            const int slot = is_swa ? (kk & (SWA_SLOTS - 1)) : kk;
            const size_t kb0 = (size_t)kvh * pstride * KV_KVH_BYTES + (size_t)slot * KV_KVH_BYTES;
            const int half8 = lane & 1;
            const int8_t* src = (const int8_t*)(kvv + kb0 + KV_SCALES_BYTES) + half8 * 128;
            const __half* sc = (const __half*)(kvv + kb0) + half8 * 4;
            uint4* dst = (uint4*)(ws + (lane >> 1) * TC_STRIDE + half8 * 128);
            #pragma unroll
            for (int u = 0; u < 16; ++u) {
                const int8_t* p = src + u * 8;
                const __half2 d2 = __half2half2(sc[u >> 2]);
                uint4 v;
                v.x = *(uint32_t*)&__hmul2(d2, __floats2half2_rn((float)p[0], (float)p[1]));
                v.y = *(uint32_t*)&__hmul2(d2, __floats2half2_rn((float)p[2], (float)p[3]));
                v.z = *(uint32_t*)&__hmul2(d2, __floats2half2_rn((float)p[4], (float)p[5]));
                v.w = *(uint32_t*)&__hmul2(d2, __floats2half2_rn((float)p[6], (float)p[7]));
                dst[u] = v;
            }
        }
        __syncwarp();
        #pragma unroll
        for (int j = 0; j < HD / 8; ++j) {
            uint32_t b[2];
            ldmatrix2_trans(b, &ws[(ldm_half + ldm_r8) * TC_STRIDE + j * 8]);
            mma_16816_f16(o[j], pa, b);
        }
    }

    #pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
        const int pair = gid + hf * 8;
        const int r = pair >> 2, h = pair & 3;
        if (r >= M) continue;
        const float il = l[hf] > 0.f ? 1.f / l[hf] : 0.f;
        float* base = 0;
        if (ns == 1) {
            base = out + ((size_t)r * N_HEAD + kvh * 4 + h) * HD;
        } else {
            base = part + (size_t)r * part_row_stride(ns) +
                   (((size_t)kvh * ns + split) * 4 + h) * (HD + 2);
        }
        const int d0 = 2 * t4;
        #pragma unroll
        for (int j = 0; j < HD / 8; ++j) {
            const float2 v = __half22float2(*(const __half2*)&o[j][hf]);
            if (ns == 1) {
                base[d0 + j * 8]     = v.x * il;
                base[d0 + j * 8 + 1] = v.y * il;
            } else {
                base[d0 + j * 8]     = v.x;
                base[d0 + j * 8 + 1] = v.y;
            }
        }
        if (ns != 1 && t4 == 0) {
            base[HD] = m[hf];
            base[HD + 1] = l[hf];
        }
    }
}

// ---------------------------------------------------------------------------
// Single-token decode attention: one warp per (kv head, split), warp-owned
// 16-key tile, no block barriers.
//
// The row-split kernel gives each of the four warps a different four-row band of
// the 16-row MMA tile, so at rows == 1 three warps compute padding and every warp
// re-reads the whole K/V tile from shared (4x the LDS traffic per key).  Here a
// warp owns its own key range: K is staged once, QK^T runs once, and the tile's
// max/sum chain is warp-local, so __syncwarp() replaces the two __syncthreads().
// The KV dequant is the same vectorised PRMT/HSUB2/HMUL2 form as the row-split
// kernel (four 32-value groups per lane, 16 B loads), and all four passes' loads
// are issued before the dequant so ~8 x 128-bit loads are in flight per warp.
// ---------------------------------------------------------------------------
__device__ float g_lazy_thr = 14.f;
template <bool LAZY, bool MS = false>
__global__ void __launch_bounds__(32) k_attn_m1_w(
        float* __restrict__ out, const float* __restrict__ q,
        const uint8_t* __restrict__ kvk, const uint8_t* __restrict__ kvv,
        const int* __restrict__ pos, float* __restrict__ part, int is_swa,
        const int* __restrict__ kslot = nullptr, const int* __restrict__ act = nullptr) {
    extern __shared__ __align__(16) __half ms_sh[];
    __half* ksh = ms_sh;                                  // [TC_T][TC_STRIDE]
    const int lane = threadIdx.x & 31;
    const int kvh = blockIdx.x, split = blockIdx.y, ns = gridDim.y;
    const int gid = lane >> 2, t4 = lane & 3;
    const int ldm_r8 = lane & 7, ldm_half = ((lane >> 3) & 1) * 8;
    const __half2 h1152 = __float2half2_rn(1152.0f);

    // Multi-slot decode: one row (an independent sequence) per blockIdx.z.  Rows
    // flagged idle produce no output and never read/write a live KV partition.
    const int r = MS ? (int)blockIdx.z : 0;
    if (MS) {
        if (!act[r]) return;
        q += (size_t)r * N_HEAD * HD;
        out += (size_t)r * N_HEAD * HD;
        part += (size_t)r * part_row_stride(ns);
    }

    // A operand: rows 0..3 are the four query heads of this kv head (the M=1
    // token), rows 4..15 are padding and stay zero.  Only a0/a2 are live.
    uint32_t qa[16][2];
    #pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
        if (gid < 4) {
            const float* qp = q + (size_t)(kvh * 4 + gid) * HD + kk * 16 + 2 * t4;
            const float2 v0 = *(const float2*)(qp), v1 = *(const float2*)(qp + 8);
            qa[kk][0] = pack2(v0.x * 0.0625f, v0.y * 0.0625f);
            qa[kk][1] = pack2(v1.x * 0.0625f, v1.y * 0.0625f);
        } else { qa[kk][0] = 0u; qa[kk][1] = 0u; }
    }

    const int p0 = MS ? pos[r] : *pos;
    const int kend = p0 + 1;
    const int kstart = is_swa ? max(0, p0 - SWA + 1) : 0;
    const int nkeys = kend - kstart;
    const int per = (nkeys + ns - 1) / ns;
    const int a0 = kstart + split * per, a1 = min(kend, a0 + per);
    const int pstride = is_swa ? SWA_SLOTS : g_kv_nctx;
    const int kvbase = MS ? kslot[r] : 0;
    const uint8_t* kb = kvk + (size_t)(kvbase * N_KV + kvh) * pstride * KV_KVH_BYTES;
    const uint8_t* vb = kvv + (size_t)(kvbase * N_KV + kvh) * pstride * KV_KVH_BYTES;
    const int dk = lane >> 3, sub = lane & 7;

    float m = -1e30f, l = 0.f;
    uint32_t o[HD / 8][2];
    #pragma unroll
    for (int j = 0; j < HD / 8; ++j) { o[j][0] = 0u; o[j][1] = 0u; }

    const uint32_t zero = 0u;
    for (int t0 = a0; t0 < a1; t0 += TC_T) {
        // ---- stage 16 keys of K: all loads issued, then the dequant chain
        uint4 wk[4][2];
        __half2 dk2[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int kk = min(t0 + dk + u * 4, a1 - 1);
            const int slot = is_swa ? (kk & (SWA_SLOTS - 1)) : kk;
            const size_t off = (size_t)slot * KV_KVH_BYTES;
            wk[u][0] = *(const uint4*)(kb + off + KV_SCALES_BYTES + sub * 32);
            wk[u][1] = *(const uint4*)(kb + off + KV_SCALES_BYTES + sub * 32 + 16);
            dk2[u] = __half2half2(((const __half*)(kb + off))[sub]);
        }
        __syncwarp();
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            __half2 hk[8];
            dequant_i8x4_to_h2(wk[u][0].x, dk2[u], h1152, hk[0], hk[1]);
            dequant_i8x4_to_h2(wk[u][0].y, dk2[u], h1152, hk[2], hk[3]);
            dequant_i8x4_to_h2(wk[u][0].z, dk2[u], h1152, hk[4], hk[5]);
            dequant_i8x4_to_h2(wk[u][0].w, dk2[u], h1152, hk[6], hk[7]);
            *(uint4*)(&ksh[(dk + u * 4) * TC_STRIDE + sub * 32])     = *(uint4*)&hk[0];
            *(uint4*)(&ksh[(dk + u * 4) * TC_STRIDE + sub * 32 + 8]) = *(uint4*)&hk[4];
            dequant_i8x4_to_h2(wk[u][1].x, dk2[u], h1152, hk[0], hk[1]);
            dequant_i8x4_to_h2(wk[u][1].y, dk2[u], h1152, hk[2], hk[3]);
            dequant_i8x4_to_h2(wk[u][1].z, dk2[u], h1152, hk[4], hk[5]);
            dequant_i8x4_to_h2(wk[u][1].w, dk2[u], h1152, hk[6], hk[7]);
            *(uint4*)(&ksh[(dk + u * 4) * TC_STRIDE + sub * 32 + 16]) = *(uint4*)&hk[0];
            *(uint4*)(&ksh[(dk + u * 4) * TC_STRIDE + sub * 32 + 24]) = *(uint4*)&hk[4];
        }
        __syncwarp();

        // ---- QK^T for the live rows
        uint32_t s0[2] = {0u, 0u}, s1[2] = {0u, 0u};
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            uint32_t av[4] = {qa[kk][0], zero, qa[kk][1], zero};
            uint32_t b0[2], b1[2];
            ldmatrix2(b0, &ksh[(0 + ldm_r8) * TC_STRIDE + kk * 16 + ldm_half]);
            ldmatrix2(b1, &ksh[(8 + ldm_r8) * TC_STRIDE + kk * 16 + ldm_half]);
            mma_16816_f16(s0, av, b0);
            mma_16816_f16(s1, av, b1);
        }
        // lane holds scores for head gid at keys 2t4, 2t4+1 (s0) and +8 (s1);
        // the upper half of each accumulator is padding.
        float sv[4];
        {
            const __half* p0h = (const __half*)&s0[0];
            const __half* p1h = (const __half*)&s1[0];
            sv[0] = __half2float(p0h[0]);
            sv[1] = __half2float(p0h[1]);
            sv[2] = __half2float(p1h[0]);
            sv[3] = __half2float(p1h[1]);
        }
        if (gid >= 4) { sv[0] = sv[1] = sv[2] = sv[3] = -1e30f; }
        const int kA = t0 + 2 * t4, kB = kA + 8;
        const bool full = (t0 + TC_T - 1 < a1) && (t0 + TC_T - 1 <= p0) &&
                          (!is_swa || t0 >= p0 + 1 - SWA + 1);
        if (!full) {
            if (kA > p0 || kA >= a1 || kA < kstart + split * per) sv[0] = -1e30f;
            if (kA + 1 > p0 || kA + 1 >= a1 || kA + 1 < kstart + split * per) sv[1] = -1e30f;
            if (kB > p0 || kB >= a1 || kB < kstart + split * per) sv[2] = -1e30f;
            if (kB + 1 > p0 || kB + 1 >= a1 || kB + 1 < kstart + split * per) sv[3] = -1e30f;
            if (is_swa) {
                const int lo = max(0, p0 + 1 - SWA);
                if (kA < lo) sv[0] = -1e30f;
                if (kA + 1 < lo) sv[1] = -1e30f;
                if (kB < lo) sv[2] = -1e30f;
                if (kB + 1 < lo) sv[3] = -1e30f;
            }
        }
        float mx = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3]));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        const float mn = fmaxf(m, mx);
        const bool ok = mn > -1e29f;
        const float al = ok ? __expf(m - mn) : 1.f;
        m = mn;
        const float z = ok ? 1.f : 0.f;
        float e0 = __expf(sv[0] - mn) * z, e1 = __expf(sv[1] - mn) * z;
        float e2 = __expf(sv[2] - mn) * z, e3 = __expf(sv[3] - mn) * z;
        if (gid >= 4) { e0 = e1 = e2 = e3 = 0.f; }
        float sum = (e0 + e1) + (e2 + e3);
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        l = l * al + sum;
        if (LAZY) {
            const bool hot = (gid < 4) && (mx > m - g_lazy_thr) && (mx > -1e29f);
            if (!__any_sync(0xffffffffu, hot)) continue;
        }
        if (al != 1.f) {
            const __half2 ah = __floats2half2_rn(al, al);
            #pragma unroll
            for (int j = 0; j < HD / 8; ++j) {
                *(__half2*)&o[j][0] = __hmul2(*(__half2*)&o[j][0], ah);
                *(__half2*)&o[j][1] = __hmul2(*(__half2*)&o[j][1], ah);
            }
        }
        uint32_t pa[4] = {pack2(e0, e1), zero, pack2(e2, e3), zero};

        // ---- stage V into the same tile (lazy: only when the tile is hot)
        __syncwarp();
        uint4 wv[4][2];
        __half2 dv2[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int kk = min(t0 + dk + u * 4, a1 - 1);
            const int slot = is_swa ? (kk & (SWA_SLOTS - 1)) : kk;
            const size_t off = (size_t)slot * KV_KVH_BYTES;
            wv[u][0] = *(const uint4*)(vb + off + KV_SCALES_BYTES + sub * 32);
            wv[u][1] = *(const uint4*)(vb + off + KV_SCALES_BYTES + sub * 32 + 16);
            dv2[u] = __half2half2(((const __half*)(vb + off))[sub]);
        }
        __syncwarp();
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            __half2 hv[8];
            dequant_i8x4_to_h2(wv[u][0].x, dv2[u], h1152, hv[0], hv[1]);
            dequant_i8x4_to_h2(wv[u][0].y, dv2[u], h1152, hv[2], hv[3]);
            dequant_i8x4_to_h2(wv[u][0].z, dv2[u], h1152, hv[4], hv[5]);
            dequant_i8x4_to_h2(wv[u][0].w, dv2[u], h1152, hv[6], hv[7]);
            *(uint4*)(&ksh[(dk + u * 4) * TC_STRIDE + sub * 32])     = *(uint4*)&hv[0];
            *(uint4*)(&ksh[(dk + u * 4) * TC_STRIDE + sub * 32 + 8]) = *(uint4*)&hv[4];
            dequant_i8x4_to_h2(wv[u][1].x, dv2[u], h1152, hv[0], hv[1]);
            dequant_i8x4_to_h2(wv[u][1].y, dv2[u], h1152, hv[2], hv[3]);
            dequant_i8x4_to_h2(wv[u][1].z, dv2[u], h1152, hv[4], hv[5]);
            dequant_i8x4_to_h2(wv[u][1].w, dv2[u], h1152, hv[6], hv[7]);
            *(uint4*)(&ksh[(dk + u * 4) * TC_STRIDE + sub * 32 + 16]) = *(uint4*)&hv[0];
            *(uint4*)(&ksh[(dk + u * 4) * TC_STRIDE + sub * 32 + 24]) = *(uint4*)&hv[4];
        }
        __syncwarp();
        #pragma unroll
        for (int j = 0; j < HD / 8; ++j) {
            uint32_t b[2];
            ldmatrix2_trans(b, &ksh[(ldm_half + ldm_r8) * TC_STRIDE + j * 8]);
            mma_16816_f16(o[j], pa, b);
        }
    }

    if (gid >= 4) return;
    const float il = l > 0.f ? 1.f / l : 0.f;
    float* base;
    if (ns == 1) {
        base = out + ((size_t)kvh * 4 + gid) * HD;
    } else {
        base = part + (((size_t)kvh * ns + split) * 4 + gid) * (HD + 2);
    }
    const int d0 = 2 * t4;
    #pragma unroll
    for (int j = 0; j < HD / 8; ++j) {
        const float2 v = __half22float2(*(const __half2*)&o[j][0]);
        if (ns == 1) {
            base[d0 + j * 8]     = v.x * il;
            base[d0 + j * 8 + 1] = v.y * il;
        } else {
            base[d0 + j * 8]     = v.x;
            base[d0 + j * 8 + 1] = v.y;
        }
    }
    if (ns != 1 && t4 == 0) { base[HD] = m; base[HD + 1] = l; }
}

// Shared KV tile dequant: 16 keys x 272 B q8_0 -> 16 x TC_STRIDE fp16 in smem.
// All four passes' 128-bit loads are issued before the dequant chain so the warp
// keeps ~8 loads in flight.
__device__ __forceinline__ void kv_tile_dequant(__half* __restrict__ ksh, const uint8_t* __restrict__ base,
                                                int t0, int a1, int is_swa, int dk, int sub) {
    const __half2 h1152 = __float2half2_rn(1152.0f);
    uint4 w[4][2];
    __half2 d2[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        const int kk = min(t0 + dk + u * 4, a1 - 1);
        const int slot = is_swa ? (kk & (SWA_SLOTS - 1)) : kk;
        const size_t off = (size_t)slot * KV_KVH_BYTES;
        w[u][0] = *(const uint4*)(base + off + KV_SCALES_BYTES + sub * 32);
        w[u][1] = *(const uint4*)(base + off + KV_SCALES_BYTES + sub * 32 + 16);
        d2[u] = __half2half2(((const __half*)(base + off))[sub]);
    }
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        __half2 h[8];
        dequant_i8x4_to_h2(w[u][0].x, d2[u], h1152, h[0], h[1]);
        dequant_i8x4_to_h2(w[u][0].y, d2[u], h1152, h[2], h[3]);
        dequant_i8x4_to_h2(w[u][0].z, d2[u], h1152, h[4], h[5]);
        dequant_i8x4_to_h2(w[u][0].w, d2[u], h1152, h[6], h[7]);
        *(uint4*)(&ksh[(dk + u * 4) * TC_STRIDE + sub * 32])     = *(uint4*)&h[0];
        *(uint4*)(&ksh[(dk + u * 4) * TC_STRIDE + sub * 32 + 8]) = *(uint4*)&h[4];
        dequant_i8x4_to_h2(w[u][1].x, d2[u], h1152, h[0], h[1]);
        dequant_i8x4_to_h2(w[u][1].y, d2[u], h1152, h[2], h[3]);
        dequant_i8x4_to_h2(w[u][1].z, d2[u], h1152, h[4], h[5]);
        dequant_i8x4_to_h2(w[u][1].w, d2[u], h1152, h[6], h[7]);
        *(uint4*)(&ksh[(dk + u * 4) * TC_STRIDE + sub * 32 + 16]) = *(uint4*)&h[0];
        *(uint4*)(&ksh[(dk + u * 4) * TC_STRIDE + sub * 32 + 24]) = *(uint4*)&h[4];
    }
}

// ---------------------------------------------------------------------------
// 2..4-row speculative verification attention: one warp per (kv head, split).
// The 16-row mma tile is exactly (row, head) for 4 verified rows x 4 kv heads, so
// one warp carries all of them and no warp computes a second copy of QK^T.  Same
// warp-owned tile and __syncwarp-only structure as the single-row kernel.
// ---------------------------------------------------------------------------
template <int R>
__global__ void __launch_bounds__(32) k_attn_mv_w(
        float* __restrict__ out, const float* __restrict__ q,
        const uint8_t* __restrict__ kvk, const uint8_t* __restrict__ kvv,
        const int* __restrict__ pos, float* __restrict__ part, int is_swa) {
    extern __shared__ __align__(16) __half mv_sh[];
    __half* ksh = mv_sh;
    const int lane = threadIdx.x & 31;
    const int kvh = blockIdx.x, split = blockIdx.y, ns = gridDim.y;
    const int gid = lane >> 2, t4 = lane & 3;
    const int ldm_r8 = lane & 7, ldm_half = ((lane >> 3) & 1) * 8;

    uint32_t qa[16][4];
    #pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
        #pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const int pair = gid + hf * 8;
            const int r = pair >> 2, h = pair & 3;
            if (r < R) {
                const float* qp = q + (size_t)(r * N_HEAD + kvh * 4 + h) * HD + kk * 16 + 2 * t4;
                const float2 v0 = *(const float2*)(qp), v1 = *(const float2*)(qp + 8);
                qa[kk][hf]     = pack2(v0.x * 0.0625f, v0.y * 0.0625f);
                qa[kk][hf + 2] = pack2(v1.x * 0.0625f, v1.y * 0.0625f);
            } else { qa[kk][hf] = 0u; qa[kk][hf + 2] = 0u; }
        }
    }

    const int p0 = *pos;
    const int kend = p0 + R;
    const int kstart = is_swa ? max(0, p0 - SWA + 1) : 0;
    const int nkeys = kend - kstart;
    const int per = (nkeys + ns - 1) / ns;
    const int a0 = kstart + split * per, a1 = min(kend, a0 + per);
    const int pstride = is_swa ? SWA_SLOTS : g_kv_nctx;
    const uint8_t* kb = kvk + (size_t)kvh * pstride * KV_KVH_BYTES;
    const uint8_t* vb = kvv + (size_t)kvh * pstride * KV_KVH_BYTES;
    const int dk = lane >> 3, sub = lane & 7;

    float m[2] = {-1e30f, -1e30f}, l[2] = {0.f, 0.f};
    uint32_t o[HD / 8][2];
    #pragma unroll
    for (int j = 0; j < HD / 8; ++j) { o[j][0] = 0u; o[j][1] = 0u; }

    const int r0 = gid >> 2, r1 = r0 + 2;
    const int lim0 = p0 + r0, lim1 = p0 + r1;
    const int lo0 = is_swa ? max(0, lim0 - SWA + 1) : 0;
    const int lo1 = is_swa ? max(0, lim1 - SWA + 1) : 0;

    for (int t0 = a0; t0 < a1; t0 += TC_T) {
        __syncwarp();
        kv_tile_dequant(ksh, kb, t0, a1, is_swa, dk, sub);
        __syncwarp();

        uint32_t sh[2][2] = {{0u, 0u}, {0u, 0u}};
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            uint32_t b0[2], b1[2];
            ldmatrix2(b0, &ksh[(0 + ldm_r8) * TC_STRIDE + kk * 16 + ldm_half]);
            ldmatrix2(b1, &ksh[(8 + ldm_r8) * TC_STRIDE + kk * 16 + ldm_half]);
            mma_16816_f16(sh[0], qa[kk], b0);
            mma_16816_f16(sh[1], qa[kk], b1);
        }
        float sv[8];
        {
            const __half* h0 = (const __half*)&sh[0][0];
            const __half* h1 = (const __half*)&sh[1][0];
            sv[0] = __half2float(h0[0]); sv[1] = __half2float(h0[1]);
            sv[2] = __half2float(h0[2]); sv[3] = __half2float(h0[3]);
            sv[4] = __half2float(h1[0]); sv[5] = __half2float(h1[1]);
            sv[6] = __half2float(h1[2]); sv[7] = __half2float(h1[3]);
        }
        const int kA = t0 + 2 * t4, kB = kA + 8;
        const bool full = (t0 + TC_T - 1 < a1) && (t0 + TC_T - 1 <= p0) &&
                          (!is_swa || t0 >= p0 + R - SWA);
        if (!full) {
            const int lo = a0;
            if (kA > lim0 || kA >= a1 || kA < lo || kA < lo0) sv[0] = -1e30f;
            if (kA + 1 > lim0 || kA + 1 >= a1 || kA + 1 < lo || kA + 1 < lo0) sv[1] = -1e30f;
            if (kA > lim1 || kA >= a1 || kA < lo || kA < lo1) sv[2] = -1e30f;
            if (kA + 1 > lim1 || kA + 1 >= a1 || kA + 1 < lo || kA + 1 < lo1) sv[3] = -1e30f;
            if (kB > lim0 || kB >= a1 || kB < lo || kB < lo0) sv[4] = -1e30f;
            if (kB + 1 > lim0 || kB + 1 >= a1 || kB + 1 < lo || kB + 1 < lo0) sv[5] = -1e30f;
            if (kB > lim1 || kB >= a1 || kB < lo || kB < lo1) sv[6] = -1e30f;
            if (kB + 1 > lim1 || kB + 1 >= a1 || kB + 1 < lo || kB + 1 < lo1) sv[7] = -1e30f;
            if (R < 3) { sv[2] = sv[3] = sv[6] = sv[7] = -1e30f; }
            if (R < 2) { /* r0 only */ }
        }
        float mx0 = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[4], sv[5]));
        float mx1 = fmaxf(fmaxf(sv[2], sv[3]), fmaxf(sv[6], sv[7]));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m[0], mx0), mn1 = fmaxf(m[1], mx1);
        const bool ok0 = mn0 > -1e29f, ok1 = mn1 > -1e29f;
        const float al0 = ok0 ? __expf(m[0] - mn0) : 1.f;
        const float al1 = ok1 ? __expf(m[1] - mn1) : 1.f;
        m[0] = mn0; m[1] = mn1;
        const float z0 = ok0 ? 1.f : 0.f, z1 = ok1 ? 1.f : 0.f;
        const float e0 = __expf(sv[0] - mn0) * z0, e1 = __expf(sv[1] - mn0) * z0;
        const float e2 = __expf(sv[2] - mn1) * z1, e3 = __expf(sv[3] - mn1) * z1;
        const float e4 = __expf(sv[4] - mn0) * z0, e5 = __expf(sv[5] - mn0) * z0;
        const float e6 = __expf(sv[6] - mn1) * z1, e7 = __expf(sv[7] - mn1) * z1;
        float sum0 = (e0 + e1) + (e4 + e5);
        float sum1 = (e2 + e3) + (e6 + e7);
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
        l[0] = l[0] * al0 + sum0;
        l[1] = l[1] * al1 + sum1;
        if (al0 != 1.f || al1 != 1.f) {
            const __half2 a0h = __floats2half2_rn(al0, al0);
            const __half2 a1h = __floats2half2_rn(al1, al1);
            #pragma unroll
            for (int j = 0; j < HD / 8; ++j) {
                *(__half2*)&o[j][0] = __hmul2(*(__half2*)&o[j][0], a0h);
                *(__half2*)&o[j][1] = __hmul2(*(__half2*)&o[j][1], a1h);
            }
        }
        const uint32_t pa[4] = {pack2(e0, e1), pack2(e2, e3), pack2(e4, e5), pack2(e6, e7)};

        __syncwarp();
        kv_tile_dequant(ksh, vb, t0, a1, is_swa, dk, sub);
        __syncwarp();
        #pragma unroll
        for (int j = 0; j < HD / 8; ++j) {
            uint32_t b[2];
            ldmatrix2_trans(b, &ksh[(ldm_half + ldm_r8) * TC_STRIDE + j * 8]);
            mma_16816_f16(o[j], pa, b);
        }
    }

    #pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
        const int pair = gid + hf * 8;
        const int r = pair >> 2, h = pair & 3;
        if (r >= R) continue;
        const float il = l[hf] > 0.f ? 1.f / l[hf] : 0.f;
        float* base;
        if (ns == 1) {
            base = out + ((size_t)r * N_HEAD + kvh * 4 + h) * HD;
        } else {
            base = part + (size_t)r * part_row_stride(ns) +
                   (((size_t)kvh * ns + split) * 4 + h) * (HD + 2);
        }
        const int d0 = 2 * t4;
        #pragma unroll
        for (int j = 0; j < HD / 8; ++j) {
            const float2 v = __half22float2(*(const __half2*)&o[j][hf]);
            if (ns == 1) {
                base[d0 + j * 8]     = v.x * il;
                base[d0 + j * 8 + 1] = v.y * il;
            } else {
                base[d0 + j * 8]     = v.x;
                base[d0 + j * 8 + 1] = v.y;
            }
        }
        if (ns != 1 && t4 == 0) { base[HD] = m[hf]; base[HD + 1] = l[hf]; }
    }
}

static int g_attn_rs = 1;   // 1 = prompt attention uses the row-split kernel
// Warps per prompt-attention CTA on the full (non-windowed) layers.  The row
// block is 4 rows per warp, so 12 warps = 48 rows: half the KV passes and half
// the per-tile dequant/barrier overhead per row of 6 warps, and it still runs
// at one CTA/SM (168 regs x 384 threads = the whole register file).  Deep
// chunks win from the wider block; shallow ones (where a block covers only a
// couple of KV tiles) lose to the extra Q loads, so the dispatcher switches at
// ~12K (measured crossover 8-16K).
static int g_full_nw = -1;
static inline int prefill_full_nw() {
    if (g_full_nw < 0) {
        const char* e = getenv("BRUCE_ATTN_FULL_NW");
        g_full_nw = e ? atoi(e) : 12;
    }
    return g_full_nw;
}
static int g_batch_pos0 = 0;    // host-side first position of the current batch
static int g_full_nw_use = 6;   // resolved row-block width for full layers
static int g_full_nw_depth = 12288;
static inline int prefill_full_nw_use() { return g_full_nw_use; }
static int g_opt_norms = -1; // register-resident norms for rows <= 8 (measurement switch)
static inline bool opt_norms() {
    if (g_opt_norms < 0) {
        const char* e = getenv("BRUCE_OPT_NORMS");
        g_opt_norms = (!e || atoi(e)) ? 1 : 0;
    }
    return g_opt_norms != 0;
}
static int g_fast_norms = -1;   // float4 quarter-warp norms (A/B switch)
static inline bool fast_norms() {
    if (g_fast_norms < 0) {
        const char* e = getenv("BRUCE_FAST_NORMS");
        g_fast_norms = (!e || atoi(e)) ? 1 : 0;
    }
    return g_fast_norms != 0;
}
static int g_fast_qkv = -1;     // one-warp-per-head qkv prep (A/B switch)
static inline bool fast_qkv() {
    if (g_fast_qkv < 0) {
        const char* e = getenv("BRUCE_FAST_QKV");
        g_fast_qkv = (!e || atoi(e)) ? 1 : 0;
    }
    return g_fast_qkv != 0;
}
static int g_skip_attn = -1, g_skip_logits = -1;   // dev timing switches
static inline bool skip_attn() {
    if (g_skip_attn < 0) g_skip_attn = getenv("BRUCE_SKIP_ATTN") ? 1 : 0;
    return g_skip_attn != 0;
}
static inline bool skip_logits() {
    if (g_skip_logits < 0) g_skip_logits = getenv("BRUCE_SKIP_LOGITS") ? 1 : 0;
    return g_skip_logits != 0;
}
// ---- dev probe: phase mask for --probe (0 = run normally)
enum {
    PB_QKV = 1 << 0, PB_PREP = 1 << 1, PB_ATTN = 1 << 2, PB_GATE = 1 << 3,
    PB_WO = 1 << 4, PB_FFGU = 1 << 5, PB_SILU = 1 << 6, PB_WFD = 1 << 7,
    PB_RESID = 1 << 8, PB_NORM = 1 << 9, PB_LOGITS = 1 << 10, PB_EMBED = 1 << 11,
    PB_ALL = (1 << 12) - 1
};
static unsigned g_poff = 0;          // phases to skip (dev probe only)
static inline bool p_on(unsigned bit) { return (g_poff & bit) == 0; }

static int g_fused_combine = -1;
static inline bool fused_combine() {
    if (g_fused_combine < 0) {
        const char* e = getenv("BRUCE_FUSED_COMBINE");
        g_fused_combine = (e && atoi(e)) ? 1 : 0;
    }
    return g_fused_combine != 0;
}

// Loop-1 attention reuse (looped-transformer lever): the 5 recurrent full-attention
// blocks run twice per token over the *same* KV cache, and the loop-1 query is a
// tiny perturbation of the loop-0 query (measured cos 0.985-0.999).  Decode and
// verification therefore reuse the loop-0 attention output instead of streaming
// 5/14 of the full-attention KV a second time.  Prefill (rows > 8) never uses it,
// so prompt numerics and the KV writes are unchanged.
// Default on: measured 128/128 token-identical on the code prompt (plain and
// --draft 7), 48/48 on prose, and a 0.076-nat rms logit perturbation that flips
// documented ~0.2-nat near-ties on chat.  --no-loop1-reuse restores the exact path.
static int g_l1reuse = -1;
static inline int l1reuse() {
    if (g_l1reuse < 0) { const char* e = getenv("BRUCE_L1REUSE"); g_l1reuse = e ? (atoi(e) ? 1 : 0) : 1; }
    return g_l1reuse;
}

static int g_attn_wi = -1;   // experimental warp-independent verify attention (off: the CTA kernel is faster)
static inline bool attn_wi() {
    if (g_attn_wi < 0) { const char* e = getenv("BRUCE_ATTN_WI"); g_attn_wi = (e && atoi(e)) ? 1 : 0; }
    return g_attn_wi != 0;
}
static int g_rs_verify = -1; // row-split kernel for 2..8 verify rows (score computed once)
static inline bool rs_verify() {
    if (g_rs_verify < 0) { const char* e = getenv("BRUCE_RS_VERIFY"); g_rs_verify = (e && atoi(e) == 0) ? 0 : 1; }
    return g_rs_verify != 0;
}
static int g_verify_splits = -1;
static inline int verify_splits() {
    if (g_verify_splits < 0) {
        const char* e = getenv("BRUCE_VERIFY_SPLITS");
        g_verify_splits = e ? std::max(1, std::min(PART_MAX_SPLITS, atoi(e))) : 21;
    }
    return g_verify_splits;
}
static int g_swa_verify_splits = -1;
static inline int swa_verify_splits() {
    if (g_swa_verify_splits < 0) {
        const char* e = getenv("BRUCE_SWA_NS");
        g_swa_verify_splits = e ? std::max(1, std::min(PART_MAX_SPLITS, atoi(e))) : verify_splits();
    }
    return g_swa_verify_splits;
}

static int g_m1_splits = -1;
static inline int m1_splits(int is_swa) {
    if (g_m1_splits < 0) {
        const char* e = getenv("BRUCE_M1_SPLITS");
        g_m1_splits = e ? std::max(1, std::min(PART_MAX_SPLITS, atoi(e))) : 0;
    }
    if (g_m1_splits) return g_m1_splits;
    return is_swa ? 28 : 56;   // one warp per split; 4*56 = 224 = one wave at 8 blocks/SM
}

static int g_mv_splits = -1;
static inline int mv_splits(int is_swa) {
    if (g_mv_splits < 0) {
        const char* e = getenv("BRUCE_MV_SPLITS");
        g_mv_splits = e ? std::max(1, std::min(PART_MAX_SPLITS, atoi(e))) : 0;
    }
    if (g_mv_splits) return g_mv_splits;
    return is_swa ? 21 : 42;
}

// --attn-tc 0 keeps the pre-round-3 behaviour (scalar kernels + the m1path=2 split rule)
static inline int attn_m1_path() { return g_attn_tc ? attn_m1() : 2; }

static void launch_attn_m(float* out, const float* q, const uint8_t* kvk, const uint8_t* kvv,
                          const int* pos, float* part, int rows, int ns, int is_swa,
                          cudaStream_t st) {
    dim3 g(N_KV, ns);
    const int m1path = attn_m1_path();
    if (rows == 1 && m1path == 3 && g_attn_tc) {
        static bool m1w_attr = false;
        if (!m1w_attr) {
            m1w_attr = true;
            CK(cudaFuncSetAttribute((const void*)&k_attn_m1_w<true>,
                                    cudaFuncAttributePreferredSharedMemoryCarveout, 100));
            CK(cudaFuncSetAttribute((const void*)&k_attn_m1_w<false>,
                                    cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        }
        const size_t shw = (size_t)TC_T * TC_STRIDE * sizeof(__half);
        if (lazy_v())
            k_attn_m1_w<true><<<g, 32, shw, st>>>(out, q, kvk, kvv, pos, part, is_swa);
        else
            k_attn_m1_w<false><<<g, 32, shw, st>>>(out, q, kvk, kvv, pos, part, is_swa);
        cudaError_t ew = cudaGetLastError();
        if (ew) { fprintf(stderr, "attn m1w launch failed ns=%d: %s\n", ns, cudaGetErrorString(ew)); exit(1); }
        return;
    }
    if (rows >= 2 && rows <= 4 && m1path == 3 && g_attn_tc) {
        static bool mvw_attr = false;
        if (!mvw_attr) {
            mvw_attr = true;
            CK(cudaFuncSetAttribute((const void*)&k_attn_mv_w<2>,
                                    cudaFuncAttributePreferredSharedMemoryCarveout, 100));
            CK(cudaFuncSetAttribute((const void*)&k_attn_mv_w<3>,
                                    cudaFuncAttributePreferredSharedMemoryCarveout, 100));
            CK(cudaFuncSetAttribute((const void*)&k_attn_mv_w<4>,
                                    cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        }
        const size_t shw = (size_t)TC_T * TC_STRIDE * sizeof(__half);
        if (rows == 2)      k_attn_mv_w<2><<<g, 32, shw, st>>>(out, q, kvk, kvv, pos, part, is_swa);
        else if (rows == 3) k_attn_mv_w<3><<<g, 32, shw, st>>>(out, q, kvk, kvv, pos, part, is_swa);
        else                k_attn_mv_w<4><<<g, 32, shw, st>>>(out, q, kvk, kvv, pos, part, is_swa);
        cudaError_t ev = cudaGetLastError();
        if (ev) { fprintf(stderr, "attn mv launch failed rows=%d ns=%d: %s\n", rows, ns, cudaGetErrorString(ev)); exit(1); }
        return;
    }
    const bool allow1 = (rows == 1 && m1path != 0);
    if ((g_attn_tc || rows > 8) && (rows >= 2 || allow1) && rows <= MBUF) {
        static bool tc_smem_set = false;
        if (!tc_smem_set) {
            tc_smem_set = true;
            const int smx = (4 * 16 + 2 * TC_T) * TC_STRIDE * (int)sizeof(__half);
            CK(cudaFuncSetAttribute((const void*)&k_attn_tc<1, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, smx));
            CK(cudaFuncSetAttribute((const void*)&k_attn_tc<2, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, smx));
            CK(cudaFuncSetAttribute((const void*)&k_attn_tc<4, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, smx));
        }
        if ((rows >= 2 || allow1) && rows <= 8 && rs_verify() && (rows >= 2 || m1path == 2)) {
            // Same row-split layout as the prompt path: one score matrix per
            // (row, head) pair instead of recomputing QK^T in every dim-split
            // warp.  For a 4-row round the 16 MMA m-slots are fully occupied.
            const int rpb = 16;
            g = dim3(N_KV * ((rows + rpb - 1) / rpb), ns);
            const size_t sh = (size_t)(2 * TC_T) * TC_STRIDE * sizeof(__half);
            static bool rsv_attr = false;
            if (!rsv_attr) { rsv_attr = true;
                CK(cudaFuncSetAttribute((const void*)&k_attn_tc_rs_opt<4, 3, true>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
                CK(cudaFuncSetAttribute((const void*)&k_attn_tc_rs_opt<4, 3, false>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh)); }
            static int decpipe = -1;
            if (decpipe < 0) { const char* e = getenv("BRUCE_ATTN_DECPIPE"); decpipe = (!e || atoi(e)) ? 1 : 0; }
            if (decpipe) {
                const size_t shp = sh + (size_t)2 * TC_T * KV_KVH_BYTES;
                k_attn_tc_rs_opt<4, 3, false, true><<<g, 128, shp, st>>>(out, q, kvk, kvv, pos, part, rows, is_swa);
            } else if (lazy_v() && rows <= 1)
                k_attn_tc_rs_opt<4, 3, true><<<g, 128, sh, st>>>(out, q, kvk, kvv, pos, part, rows, is_swa);
            else
                k_attn_tc_rs_opt<4, 3, false><<<g, 128, sh, st>>>(out, q, kvk, kvv, pos, part, rows, is_swa);
            cudaError_t e = cudaGetLastError();
            if (e) { fprintf(stderr, "attn rs verify launch failed rows=%d ns=%d: %s\n",
                             rows, ns, cudaGetErrorString(e)); exit(1); }
            return;
        }
        if (rows > 8 && g_attn_rs) {
            const int nw = is_swa ? 4 : prefill_full_nw_use();
            const int rpb = 4 * nw;
            g = dim3(N_KV * ((rows + rpb - 1) / rpb), ns);
            static int pipe_on = -1;
            if (pipe_on < 0) { const char* e = getenv("BRUCE_ATTN_PIPE"); pipe_on = (e && atoi(e) == 0) ? 0 : 1; }
            static int dbuf_on = -1;
            // Measured negative (2026-10-28): the two-buffer one-barrier tile loop
            // is 1.7% slower at 128K and 5% slower at 32K than the two-barrier
            // version -- the dequant is already off the critical path, and issuing
            // it before the tile's MMA delays the 8 dequant warps instead of
            // overlapping with the other 4.  Kept behind the switch for reference.
            if (dbuf_on < 0) { const char* e = getenv("BRUCE_ATTN_DBUF"); dbuf_on = (e && atoi(e)) ? 1 : 0; }
            // one fp16 K+V tile pair + one raw q8 tile pair; the DBUF variant
            // carries two of each so the next tile's dequant overlaps the compute
            const bool dbuf = pipe_on && dbuf_on && !is_swa && nw == 12;
            const size_t sh = (size_t)(dbuf ? 2 : 1) * 2 * TC_T * TC_STRIDE * sizeof(__half) +
                              (size_t)(dbuf ? 2 : 1) * 2 * TC_T * KV_KVH_BYTES;
            static bool rs_attr = false;      // set outside graph capture (first call is warmup)
            if (!rs_attr) { rs_attr = true;
                CK(cudaFuncSetAttribute((const void*)&k_attn_tc_rs_opt<4, 3>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, 98 * 1024));
                CK(cudaFuncSetAttribute((const void*)&k_attn_tc_rs_opt<6, 2>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, 98 * 1024));
                CK(cudaFuncSetAttribute((const void*)&k_attn_tc_rs_opt<4, 3, false, true>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, 98 * 1024));
                CK(cudaFuncSetAttribute((const void*)&k_attn_tc_rs_opt<6, 2, false, true>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, 98 * 1024));
                CK(cudaFuncSetAttribute((const void*)&k_attn_tc_rs_opt<12, 1, false, true>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, 98 * 1024));
                CK(cudaFuncSetAttribute((const void*)&k_attn_tc_rs_opt<12, 1, false, true, true>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, 98 * 1024));
                CK(cudaFuncSetAttribute((const void*)&k_attn_tc_rs_opt<12, 1, false, false>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, 98 * 1024)); }
            if (pipe_on) {
                if (is_swa)
                    k_attn_tc_rs_opt<4, 3, false, true><<<g, 32 * nw, sh, st>>>(out, q, kvk, kvv, pos, part, rows, is_swa);
                else if (nw == 12) {
                    if (dbuf)
                        k_attn_tc_rs_opt<12, 1, false, true, true><<<g, 32 * nw, sh, st>>>(out, q, kvk, kvv, pos, part, rows, is_swa);
                    else
                        k_attn_tc_rs_opt<12, 1, false, true><<<g, 32 * nw, sh, st>>>(out, q, kvk, kvv, pos, part, rows, is_swa);
                }
                else if (nw == 4)
                    k_attn_tc_rs_opt<4, 3, false, true><<<g, 32 * nw, sh, st>>>(out, q, kvk, kvv, pos, part, rows, is_swa);
                else
                    k_attn_tc_rs_opt<6, 2, false, true><<<g, 32 * nw, sh, st>>>(out, q, kvk, kvv, pos, part, rows, is_swa);
            }
            if (!pipe_on) {
                if (is_swa || nw == 4)
                    k_attn_tc_rs_opt<4, 3, false, false><<<g, 32 * nw, sh, st>>>(out, q, kvk, kvv, pos, part, rows, is_swa);
                else if (nw == 12)
                    k_attn_tc_rs_opt<12, 1, false, false><<<g, 32 * nw, sh, st>>>(out, q, kvk, kvv, pos, part, rows, is_swa);
                else
                    k_attn_tc_rs_opt<6, 2, false, false><<<g, 32 * nw, sh, st>>>(out, q, kvk, kvv, pos, part, rows, is_swa);
            }
            { cudaError_t e = cudaGetLastError();
              if (e) { fprintf(stderr, "attn rs launch failed rows=%d ns=%d sh=%zu: %s\n", rows, ns, sh, cudaGetErrorString(e)); exit(1); } }
        } else if (rows > 8) {
            g = dim3(N_KV * ((rows + 7) / 8), ns);
            const size_t sh = (size_t)(2 * 16 + 2 * TC_T) * TC_STRIDE * sizeof(__half);
            k_attn_tc<2, 4><<<g, 128, sh, st>>>(out, q, kvk, kvv, pos, part, rows, is_swa);
        } else if (rows <= 4) {
            if (!attn_wi()) {
                g = dim3(N_KV, ns);
                const size_t sh = (size_t)(1 * 16 + 2 * TC_T) * TC_STRIDE * sizeof(__half);
                k_attn_tc<1, 4><<<g, 128, sh, st>>>(out, q, kvk, kvv, pos, part, rows, is_swa);
            } else {
            g = dim3(N_KV, (ns + 3) / 4);
            const size_t sh = (size_t)(4 * 16) * TC_STRIDE * sizeof(__half);
            static bool wi_attr = false;
            if (!wi_attr) { wi_attr = true;
                CK(cudaFuncSetAttribute((const void*)&k_attn_tc_wi<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
                CK(cudaFuncSetAttribute((const void*)&k_attn_tc_wi<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
                CK(cudaFuncSetAttribute((const void*)&k_attn_tc_wi<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
                CK(cudaFuncSetAttribute((const void*)&k_attn_tc_wi<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
            }
            switch (rows) {
                case 1: k_attn_tc_wi<1><<<g, 128, sh, st>>>(out, q, kvk, kvv, pos, part, is_swa, ns); break;
                case 2: k_attn_tc_wi<2><<<g, 128, sh, st>>>(out, q, kvk, kvv, pos, part, is_swa, ns); break;
                case 3: k_attn_tc_wi<3><<<g, 128, sh, st>>>(out, q, kvk, kvv, pos, part, is_swa, ns); break;
                default: k_attn_tc_wi<4><<<g, 128, sh, st>>>(out, q, kvk, kvv, pos, part, is_swa, ns); break;
            }
            }
        } else {
            g = dim3(N_KV, ns);
            const size_t sh = (size_t)(2 * 16 + 2 * TC_T) * TC_STRIDE * sizeof(__half);
            k_attn_tc<2, 4><<<g, 128, sh, st>>>(out, q, kvk, kvv, pos, part, rows, is_swa);
        }
        return;
    }
    switch (rows) {
        case 1: k_attn<<<g, 128, 0, st>>>(out, q, kvk, kvv, pos, 0, part); break;
        case 2: k_attn_m<2><<<g, 128, 0, st>>>(out, q, kvk, kvv, pos, part); break;
        case 3: k_attn_m<3><<<g, 128, 0, st>>>(out, q, kvk, kvv, pos, part); break;
        case 4: k_attn_m<4><<<g, 128, 0, st>>>(out, q, kvk, kvv, pos, part); break;
        case 5: k_attn_m<5><<<g, 128, 0, st>>>(out, q, kvk, kvv, pos, part); break;
        case 6: k_attn_m<6><<<g, 128, 0, st>>>(out, q, kvk, kvv, pos, part); break;
        case 7: k_attn_m<7><<<g, 128, 0, st>>>(out, q, kvk, kvv, pos, part); break;
        default: k_attn_m<8><<<g, 128, 0, st>>>(out, q, kvk, kvv, pos, part); break;
    }
}

// Multi-slot decode attention.  One (kv head, split, row) warp per block: each row
// is an independent sequence with its own KV partition and local position, so this
// is exactly the M=1 warp kernel with a row dimension.  The split count matches the
// M=1 kernel, so a row's scores go through the same reduction order whether it is
// decoded alone or in a batch; the grid simply grows with the number of rows.
static bool g_splits_explicit = false;   // user passed --splits* (declared here: MS attention reads it)
static int attn_ms_splits(int rows, int is_swa) {
    // The M=1 split count exists to fill the GPU with one-warp blocks (4 kv heads
    // x 56 splits = one wave).  A batched step multiplies the block count by
    // `rows`, so keeping the M=1 count would run 4-8 waves and make every split's
    // (m, l) partial cost a full combine read: at 8 rows that overhead is ~1 ms
    // of the 22 ms step.  Shrink the split count so the grid stays ~one wave;
    // the per-row KV bytes are unchanged (disjoint key ranges) and only the
    // reduction order moves, which is the batch path's existing rounding class.
    int ns = m1_splits(is_swa);
    if (rows > 1) {
        // BRUCE_MS_SPLITS: 0 = old M=1 count, 1 = scale with rows (default),
        // N>=2 = that fixed count for every batch size (batch-size-invariant
        // reduction order; useful when reproducibility across load patterns
        // matters more than the last few percent).
        static int ms_split = -1;
        if (ms_split < 0) {
            const char* e = getenv("BRUCE_MS_SPLITS");
            ms_split = e ? std::max(0, atoi(e)) : 1;
        }
        if (ms_split >= 2) ns = ms_split;
        else if (ms_split == 1) ns = std::max(1, (ns + rows - 1) / rows);
    }
    if (g_splits_explicit) ns = is_swa ? M.swa_splits : M.full_splits;
    ns = std::min(ns, PART_MAX_SPLITS);
    if ((size_t)rows * ns > PART_ROWS_MAX) ns = std::max(1, PART_ROWS_MAX / rows);
    return ns;
}

static void launch_attn_ms(float* out, const float* q, const uint8_t* kvk, const uint8_t* kvv,
                           const int* pos, const int* kslot, const int* act,
                           float* part, int rows, int ns, int is_swa, cudaStream_t st) {
    static bool m1w_attr = false;
    if (!m1w_attr) {
        m1w_attr = true;
        CK(cudaFuncSetAttribute((const void*)&k_attn_m1_w<true, true>,
                                cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        CK(cudaFuncSetAttribute((const void*)&k_attn_m1_w<false, true>,
                                cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    }
    const dim3 g(N_KV, ns, rows);
    const size_t shw = (size_t)TC_T * TC_STRIDE * sizeof(__half);
    if (lazy_v())
        k_attn_m1_w<true, true><<<g, 32, shw, st>>>(out, q, kvk, kvv, pos, part, is_swa, kslot, act);
    else
        k_attn_m1_w<false, true><<<g, 32, shw, st>>>(out, q, kvk, kvv, pos, part, is_swa, kslot, act);
    cudaError_t e = cudaGetLastError();
    if (e) { fprintf(stderr, "attn ms launch failed rows=%d ns=%d: %s\n", rows, ns, cudaGetErrorString(e)); exit(1); }
}

// The live KV cache is head-major; the debug taps and the reference comparison files keep the
// old slot-major order ([slot][kvh][272]), so reassemble that for the first two slots.
static void kv_tap(float* tapf, const uint8_t* src, int pstride, cudaStream_t st) {
    uint8_t* taps = (uint8_t*)tapf;
    for (int h = 0; h < N_KV; ++h)
        for (int s = 0; s < 2; ++s)
            CK(cudaMemcpyAsync(taps + (size_t)(s * N_KV + h) * KV_KVH_BYTES,
                               src + (size_t)h * pstride * KV_KVH_BYTES + (size_t)s * KV_KVH_BYTES,
                               KV_KVH_BYTES, cudaMemcpyDeviceToDevice, st));
}

// one row per warp, M activation rows share each weight load;
// R rows are unrolled so the staged activations are loaded once per R rows
template <int M, int WARPS, int R>
__global__ void __launch_bounds__(512) k_gemm_q5k(float* __restrict__ y, const uint8_t* __restrict__ W,
                           const int8_t* __restrict__ xq, const float* __restrict__ xd,
                           const int* __restrict__ xsum, int K, int N) {
    extern __shared__ int sxi[];
    const int nw = K >> 2;
    const int nsb = K >> 8;
    // padded layout: group of 4 superblocks = 4*8 groups of 9 words (9 gives 32 distinct banks)
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
    const int nb32 = K >> 5;
    const int stride = gridDim.x * WARPS * R;
    for (int row0 = (blockIdx.x * WARPS + warp) * R; row0 < N; row0 += stride) {
        float acc[R][M];
        #pragma unroll
        for (int r = 0; r < R; ++r)
            #pragma unroll
            for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
        for (int sb0 = 0; sb0 < nsb; sb0 += 4) {
            const int nch = min(4, nsb - sb0);
            if (sl < nch) {
                const int sb = sb0 + sl;
                const int* xb = sxi + (sb0 >> 2) * 288 + xgrp;
                int xw[M][8];
                #pragma unroll
                for (int m = 0; m < M; ++m)
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) xw[m][i] = xb[m * nwp + i];
                const int isb = sb * 8 + js;
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int row = row0 + r;
                    if (row >= N) break;
                    const uint8_t* b = W + (size_t)row * (size_t)nsb * 176 + sb * 176;
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
        for (int r = 0; r < R; ++r) {
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

// q6_K: symmetric 6-bit, dot with (q - 32) needs no per-block sum
template <int M, int WARPS, int R>
__global__ void __launch_bounds__(512) k_gemm_q6k(float* __restrict__ y, const uint8_t* __restrict__ W,
                           const int8_t* __restrict__ xq, const float* __restrict__ xd, int K, int N) {
    extern __shared__ int8_t sxq[];
    for (int i = threadIdx.x; i < K * M; i += blockDim.x) sxq[i] = xq[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int half = lane >> 3;
    const int o = (lane & 7) * 4;
    const int is = o >> 4;
    const int sb = half >> 1;
    const int h = half & 1;
    const int nsb = K >> 8;
    const int nb32 = K >> 5;
    const int stride = gridDim.x * WARPS * R;
    for (int row0 = (blockIdx.x * WARPS + warp) * R; row0 < N; row0 += stride) {
        float acc[R][M];
        #pragma unroll
        for (int r = 0; r < R; ++r)
            #pragma unroll
            for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
        for (int s2 = 0; s2 < nsb; s2 += 2) {
            // staged activations for this superblock pair, once per R rows
            int xw[M][4];
            const int xo = (s2 + sb) * 256 + h * 128 + o;
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                const int8_t* xr = sxq + m * K + xo;
                xw[m][0] = *(const int*)(xr);
                xw[m][1] = *(const int*)(xr + 32);
                xw[m][2] = *(const int*)(xr + 64);
                xw[m][3] = *(const int*)(xr + 96);
            }
            const int xdbase = (s2 + sb) * 8 + h * 4;
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = row0 + r;
                if (row >= N) break;
                const uint8_t* b = W + (size_t)row * (size_t)nsb * 210 + (size_t)(s2 + sb) * 210;
                const uint8_t* base = b + h * 64;
                uint32_t qla = ld32u(base + o);
                uint32_t qlb = ld32u(base + 32 + o);
                uint32_t qh4 = ld32u(b + 128 + h * 32 + o);
                const int8_t* sc = (const int8_t*)(b + 192 + h * 8);
                uint32_t s0 = ld32u((const uint8_t*)sc);
                uint32_t s1 = ld32u((const uint8_t*)sc + 4);
                float d = __half2float(*(const __half*)(b + 208));
                float scl0 = d * (float)(int8_t)((s0 >> (8 * is)) & 0xFF);
                float scl1 = d * (float)(int8_t)((s0 >> (8 * is + 16)) & 0xFF);
                float scl2 = d * (float)(int8_t)((s1 >> (8 * is)) & 0xFF);
                float scl3 = d * (float)(int8_t)((s1 >> (8 * is + 16)) & 0xFF);
                // byte-wise (q - 32): q in 0..63, (q+0x60) <= 0x9F so no cross-byte carry
                const uint32_t q1 = (((qla & 0x0F0F0F0Fu) | (((qh4 >> 0) & 0x03030303u) << 4)) + 0x60606060u) ^ 0x80808080u;
                const uint32_t q2 = (((qlb & 0x0F0F0F0Fu) | (((qh4 >> 2) & 0x03030303u) << 4)) + 0x60606060u) ^ 0x80808080u;
                const uint32_t q3 = ((((qla >> 4) & 0x0F0F0F0Fu) | (((qh4 >> 4) & 0x03030303u) << 4)) + 0x60606060u) ^ 0x80808080u;
                const uint32_t q4 = ((((qlb >> 4) & 0x0F0F0F0Fu) | (((qh4 >> 6) & 0x03030303u) << 4)) + 0x60606060u) ^ 0x80808080u;
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    const int d1 = __dp4a(xw[m][0], (int)q1, 0);
                    const int d2 = __dp4a(xw[m][1], (int)q2, 0);
                    const int d3 = __dp4a(xw[m][2], (int)q3, 0);
                    const int d4 = __dp4a(xw[m][3], (int)q4, 0);
                    const float* xdr = xd + m * nb32 + xdbase;
                    acc[r][m] += xdr[0] * scl0 * (float)d1 + xdr[1] * scl1 * (float)d2 +
                                 xdr[2] * scl2 * (float)d3 + xdr[3] * scl3 * (float)d4;
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < R; ++r) {
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

// bf16 weights, M activation rows in f16 shared
template <int M, int WARPS>
__global__ void k_gemm_bf16(float* __restrict__ y, const __nv_bfloat16* __restrict__ W,
                            const float* __restrict__ x, int K, int N, const float* __restrict__ rs) {
    extern __shared__ __half sxh[];
    for (int i = threadIdx.x; i < M * K; i += blockDim.x) sxh[i] = __float2half(x[i]);
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nchunk = K / 256;
    const int stride = gridDim.x * WARPS;
    for (int row = blockIdx.x * WARPS + warp; row < N; row += stride) {
        const __nv_bfloat16* w = W + (size_t)row * K;
        float acc[M];
        #pragma unroll
        for (int m = 0; m < M; ++m) acc[m] = 0.f;
        for (int c = 0; c < nchunk; ++c) {
            const int base = c * 256 + lane * 8;
            uint4 pk = *(const uint4*)(w + base);
            const __nv_bfloat16* h = (const __nv_bfloat16*)&pk;
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                float s = 0.f;
                #pragma unroll
                for (int k = 0; k < 8; ++k)
                    s = fmaf(__bfloat162float(h[k]), __half2float(sxh[m * K + base + k]), s);
                acc[m] += s;
            }
        }
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            float s = warp_sum(acc[m]);
            if (lane == 0) y[(size_t)m * N + row] = rs ? s * rs[row] : s;
        }
    }
}

// ---------------------------------------------------------------------------
// Q8_0 logits GEMV/GEMM.  output.weight is quantised at load time from BF16
// (671 MB) into head-aligned Q8_0 (356.5 MB): per (row, 256 values) 16 B of
// eight fp16 group scales followed by 256 B of int8, i.e. 272 B / 256 values.
// The final hidden state is already q8_0 (M.xq / M.bm_xq) at this point, so the
// dot products are plain dp4a and the weight stream is 1.88x smaller.
// ---------------------------------------------------------------------------
#define Q8_SROW 272

template <int M, int WARPS>
__global__ void __launch_bounds__(32 * WARPS) k_gemm_q8_logits(
        float* __restrict__ y, const uint8_t* __restrict__ W,
        const int8_t* __restrict__ xq, const float* __restrict__ xd,
        int K, int N, const float* __restrict__ rs) {
    extern __shared__ int8_t sxq[];
    for (int i = threadIdx.x; i < K * M; i += blockDim.x) sxq[i] = xq[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int half = lane >> 4;          // which of the two superblocks this lane covers
    const int l16 = lane & 15;           // 16 values (half of a 32-value group)
    const int grp16 = l16 >> 1;          // 32-value group inside the superblock
    const int nsb = K >> 8;
    const int stride = gridDim.x * WARPS;
    for (int row = blockIdx.x * WARPS + warp; row < N; row += stride) {
        const uint8_t* rp = W + (size_t)row * nsb * Q8_SROW;
        float acc[M];
        #pragma unroll
        for (int m = 0; m < M; ++m) acc[m] = 0.f;
        for (int s0 = 0; s0 < nsb; s0 += 2) {
            const int sb = s0 + half;
            const uint8_t* bp = rp + (size_t)sb * Q8_SROW;
            const uint4 wv = *(const uint4*)(bp + 16 + l16 * 16);
            const float ws = __half2float(((const __half*)bp)[grp16]);
            const int g = sb * 8 + grp16;
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                const uint4 xv = *(const uint4*)(sxq + (size_t)m * K + sb * 256 + l16 * 16);
                int d = __dp4a((int)wv.x, (int)xv.x, 0);
                d = __dp4a((int)wv.y, (int)xv.y, d);
                d = __dp4a((int)wv.z, (int)xv.z, d);
                d = __dp4a((int)wv.w, (int)xv.w, d);
                acc[m] += ws * xd[(size_t)m * (K >> 5) + g] * (float)d;
            }
        }
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            float s = warp_sum(acc[m]);
            if (lane == 0) y[(size_t)m * N + row] = s * rs[row];
        }
    }
}

// ---------------------------------------------------------------------------
// Q6_K decode kernel on the load-time repacked form (`w.dp`, 160 B per
// (row, 128 values): 128 int8 values + 8 fp32 d*sc scale products).  Unlike the
// packed 210 B q6_K blocks this is 16-byte aligned, so each lane fetches 16
// values with one uint4 and feeds four exact dp4a dots.  One 16-value group per
// lane means the scale group boundary lines up with the load.  M activation
// rows share each weight load; up to 8 warps/block.
// ---------------------------------------------------------------------------
template <int M, int WARPS>
__global__ void __launch_bounds__(32 * WARPS) k_gemv_q6k_dp(
        float* __restrict__ y, const uint8_t* __restrict__ W,
        const int8_t* __restrict__ xq, const float* __restrict__ xd, int K, int N, int xoff) {
    extern __shared__ int8_t sxq[];
    for (int i = threadIdx.x; i < K * M; i += blockDim.x) sxq[i] = xq[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nsb = K >> 8;
    const int nst = K >> 7;
    const int sb_off = lane >> 3;       // 0..3: four superblocks per iteration
    const int r8 = lane & 7;
    const int stage = r8 >> 2;          // 128-value stage inside the superblock
    const int j32 = r8 & 3;             // 32-value group inside the stage
    const int j16 = j32 * 2;            // first of its two 16-value scale groups
    const int stride = gridDim.x * WARPS;
    for (int row = blockIdx.x * WARPS + warp; row < N; row += stride) {
        const uint8_t* rp = W + (size_t)row * nst * Q6K_SROW;
        float acc[M];
        #pragma unroll
        for (int m = 0; m < M; ++m) acc[m] = 0.f;
        for (int s0 = 0; s0 < nsb; s0 += 4) {
            const int sb = s0 + sb_off;
            if (sb >= nsb) continue;
            const uint8_t* bp = rp + (size_t)(sb * 2 + stage) * Q6K_SROW;
            const uint8_t* wq = bp + j16 * 16;
            const uint4 wv0 = *(const uint4*)(wq);
            const uint4 wv1 = *(const uint4*)(wq + 16);
            const float2 wd = *(const float2*)(bp + 128 + j16 * 4);
            const int g = sb * 8 + stage * 4 + j32;
            const int8_t* xb = sxq + sb * 256 + stage * 128 + j32 * 32;
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                const uint4 xv0 = *(const uint4*)(xb + (size_t)m * K);
                const uint4 xv1 = *(const uint4*)(xb + (size_t)m * K + 16);
                int d0 = __dp4a((int)wv0.x, (int)xv0.x, 0);
                d0 = __dp4a((int)wv0.y, (int)xv0.y, d0);
                d0 = __dp4a((int)wv0.z, (int)xv0.z, d0);
                d0 = __dp4a((int)wv0.w, (int)xv0.w, d0);
                int d1 = __dp4a((int)wv1.x, (int)xv1.x, 0);
                d1 = __dp4a((int)wv1.y, (int)xv1.y, d1);
                d1 = __dp4a((int)wv1.z, (int)xv1.z, d1);
                d1 = __dp4a((int)wv1.w, (int)xv1.w, d1);
                acc[m] += xd[(size_t)(xoff + m) * (K >> 5) + g] *
                          (wd.x * (float)d0 + wd.y * (float)d1);
            }
        }
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            float s = warp_sum(acc[m]);
            if (lane == 0) y[(size_t)m * N + row] = s;
        }
    }
}

// Fully unrolled variant for the two shapes this engine actually has
// (K=2560 attention v, K=7168 ffn_down).  Two weight rows per warp double the
// per-warp memory-level parallelism while the activation tile is reused; NSB is
// a compile-time constant so the superblock loop unrolls cleanly.
template <int M, int WARPS, int NSB, int R>
__global__ void __launch_bounds__(32 * WARPS) k_gemv_q6k_dp_fix(
        float* __restrict__ y, const uint8_t* __restrict__ W,
        const int8_t* __restrict__ xq, const float* __restrict__ xd, int N, int xoff) {
    extern __shared__ int8_t sxq[];
    const int K = NSB * 256;
    for (int i = threadIdx.x; i < K * M; i += blockDim.x) sxq[i] = xq[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int sb_off = lane >> 3, r8 = lane & 7;
    const int stage = r8 >> 2, j32 = r8 & 3, j16 = j32 * 2;
    const int rowstride = NSB * 2 * Q6K_SROW;
    const int stride = gridDim.x * WARPS * R;
    for (int row0 = (blockIdx.x * WARPS + warp) * R; row0 < N; row0 += stride) {
        const uint8_t* rp = W + (size_t)row0 * rowstride;
        float acc[R][M];
        #pragma unroll
        for (int r = 0; r < R; ++r)
            #pragma unroll
            for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
        #pragma unroll
        for (int s0 = 0; s0 < NSB; s0 += 4) {
            const int sb = s0 + sb_off;
            if (sb >= NSB) continue;
            const int8_t* xb = sxq + sb * 256 + stage * 128 + j32 * 32;
            const int g = sb * 8 + stage * 4 + j32;
            #pragma unroll
            for (int rr = 0; rr < R; ++rr) {
                const int row = row0 + rr;
                if (row >= N) continue;
                const uint8_t* bp = rp + (size_t)rr * rowstride + (size_t)(sb * 2 + stage) * Q6K_SROW;
                const uint4 wv0 = *(const uint4*)(bp + j16 * 16);
                const uint4 wv1 = *(const uint4*)(bp + j16 * 16 + 16);
                const float2 wd = *(const float2*)(bp + 128 + j16 * 4);
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    const uint4 xv0 = *(const uint4*)(xb + (size_t)m * K);
                    const uint4 xv1 = *(const uint4*)(xb + (size_t)m * K + 16);
                    int d0 = __dp4a((int)wv0.x, (int)xv0.x, 0);
                    d0 = __dp4a((int)wv0.y, (int)xv0.y, d0);
                    d0 = __dp4a((int)wv0.z, (int)xv0.z, d0);
                    d0 = __dp4a((int)wv0.w, (int)xv0.w, d0);
                    int d1 = __dp4a((int)wv1.x, (int)xv1.x, 0);
                    d1 = __dp4a((int)wv1.y, (int)xv1.y, d1);
                    d1 = __dp4a((int)wv1.z, (int)xv1.z, d1);
                    d1 = __dp4a((int)wv1.w, (int)xv1.w, d1);
                    acc[rr][m] += xd[(size_t)(xoff + m) * (K >> 5) + g] *
                                  (wd.x * (float)d0 + wd.y * (float)d1);
                }
            }
        }
        #pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            const int row = row0 + rr;
            if (row >= N) continue;
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                float s = warp_sum(acc[rr][m]);
                if (lane == 0) y[(size_t)m * N + row] = s;
            }
        }
    }
}

// ---- sampling
__global__ void k_logit_max(float* __restrict__ out, const float* __restrict__ logits,
                            int n, float temp) {
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, logits[i] / temp);
    __shared__ float red[32];
    mx = block_max(mx, red);
    if (threadIdx.x == 0) out[0] = mx;
}

__global__ void k_probs(float* __restrict__ prob, const float* __restrict__ logits,
                        const float* __restrict__ mx, int n, float temp) {
    float m = mx[0];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        prob[i] = __expf(logits[i] / temp - m);
}

__global__ void k_argmax(int* __restrict__ tok_out, const float* __restrict__ logits, int n) {
    float mx = -INFINITY; int mi = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (logits[i] > mx) { mx = logits[i]; mi = i; }
    __shared__ float sm[32];
    __shared__ int si[32];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, mx, o);
        int oi = __shfl_down_sync(0xffffffffu, mi, o);
        if (ov > mx) { mx = ov; mi = oi; }
    }
    if (lane == 0) { sm[w] = mx; si[w] = mi; }
    __syncthreads();
    if (w == 0) {
        int nw = (blockDim.x + 31) >> 5;
        mx = (lane < nw) ? sm[lane] : -INFINITY;
        mi = (lane < nw) ? si[lane] : 0;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, mx, o);
            int oi = __shfl_down_sync(0xffffffffu, mi, o);
            if (ov > mx) { mx = ov; mi = oi; }
        }
        if (lane == 0) *tok_out = mi;
    }
}

// one block per row: argmax of each of `rows` logit rows
__global__ void k_argmax_rows(int* __restrict__ tok_out, const float* __restrict__ logits,
                              int n, int rows) {
    const int row = blockIdx.x;
    const float* lg = logits + (size_t)row * n;
    float mx = -INFINITY; int mi = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (lg[i] > mx) { mx = lg[i]; mi = i; }
    __shared__ float sm[32];
    __shared__ int si[32];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, mx, o);
        int oi = __shfl_down_sync(0xffffffffu, mi, o);
        if (ov > mx) { mx = ov; mi = oi; }
    }
    if (lane == 0) { sm[w] = mx; si[w] = mi; }
    __syncthreads();
    if (w == 0) {
        int nw = (blockDim.x + 31) >> 5;
        mx = (lane < nw) ? sm[lane] : -INFINITY;
        mi = (lane < nw) ? si[lane] : 0;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, mx, o);
            int oi = __shfl_down_sync(0xffffffffu, mi, o);
            if (ov > mx) { mx = ov; mi = oi; }
        }
        if (lane == 0) tok_out[row] = mi;
    }
    (void)rows;
}

__global__ void k_sample_top_p(int* __restrict__ tok_out, const float* __restrict__ p,
                               const int* __restrict__ idx, int n, float topp,
                               int topk, float minp, unsigned long long* __restrict__ rng) {
    extern __shared__ float sh[];
    const int T = blockDim.x;
    const int per = n / T;
    int base = threadIdx.x * per;
    float s = 0.f;
    for (int i = 0; i < per; ++i) s += p[base + i];
    sh[threadIdx.x] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int t = 1; t < T; ++t) sh[t] += sh[t - 1];
        float total = sh[T - 1];

        const bool restricted = (topk > 0 || minp > 0.f);
        int limit = n;
        float mass = total;
        if (restricted) {
            if (topk > 0 && topk < limit) limit = topk;
            if (minp > 0.f) {
                float thr = p[0] * minp;
                int i = 0;
                while (i < limit && p[i] >= thr) ++i;
                limit = i > 0 ? i : 1;
            }
            if (limit < 1) limit = 1;
            if (limit < n) {
                mass = 0.f;
                for (int i = 0; i < limit; ++i) mass += p[i];
            }
            if (!(mass > 0.f)) mass = total;
        }

        float tgt = topp * mass;
        if (!(tgt > 0.f)) tgt = mass;

        int chosen = 0;
        float pmass = mass;
        if (restricted) {
            float cm = 0.f;
            chosen = limit - 1;
            for (int i = 0; i < limit; ++i) {
                cm += p[i];
                if (cm >= tgt) { chosen = i; break; }
            }
            pmass = 0.f;
            for (int i = 0; i <= chosen; ++i) pmass += p[i];
        } else {
            int ci = 0;
            while (ci < T - 1 && sh[ci] < tgt) ++ci;
            float pre = ci ? sh[ci - 1] : 0.f;
            int b0 = ci * per, kk = 0;
            float cm = pre;
            while (kk < per) { cm += p[b0 + kk]; if (cm >= tgt) break; ++kk; }
            if (kk >= per) kk = per - 1;
            chosen = b0 + kk;
            pmass = cm;
        }

        unsigned long long r = splitmix64(*rng);
        *rng = r;
        float rnd = (float)((r >> 40) * (1.0 / 16777216.0));
        float rt = rnd * pmass;
        int pick = chosen;
        if (restricted) {
            float cm = 0.f;
            pick = limit - 1;
            for (int i = 0; i < limit; ++i) {
                cm += p[i];
                if (cm > rt) { pick = i; break; }
            }
        } else {
            int ci = 0;
            while (ci < T - 1 && sh[ci] < rt) ++ci;
            float pre = ci ? sh[ci - 1] : 0.f;
            int b0 = ci * per, kk = 0;
            float cm = pre;
            while (kk < per) { cm += p[b0 + kk]; if (cm > rt) break; ++kk; }
            if (kk >= per) kk = per - 1;
            pick = b0 + kk;
        }
        *tok_out = idx[pick];
    }
}

__global__ void k_apply_penalties(float* __restrict__ logits, int n,
                                  const int* __restrict__ hist, int nhist,
                                  float repeat_penalty, float freq_penalty, float pres_penalty,
                                  const int* __restrict__ bias_ids,
                                  const float* __restrict__ bias_vals, int nbias) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int cnt = 0;
    for (int j = 0; j < nhist; ++j) if (hist[j] == i) ++cnt;
    if (cnt == 0 && nbias <= 0) return;
    float x = logits[i];
    if (cnt > 0 && repeat_penalty != 1.0f) x = x > 0.f ? x / repeat_penalty : x * repeat_penalty;
    if (cnt > 0) x -= freq_penalty * (float)cnt + pres_penalty;
    for (int j = 0; j < nbias; ++j) if (bias_ids[j] == i) x += bias_vals[j];
    logits[i] = x;
}

// ---------------------------------------------------------------- host

static inline float q6k_h2f(uint16_t h) {
    const uint32_t sg = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FFu;
    // q6_K super-block scales are routinely denormal in fp16 (|d| ~ 1e-5),
    // so the zero exponent must be decoded as m * 2^-24 rather than as zero.
    if (e == 0) {
        const float f = (float)m * 5.9604644775390625e-8f;
        return (h & 0x8000u) ? -f : f;
    }
    uint32_t x;
    if (e == 31) x = sg | 0x7F800000u | (m << 13);
    else         x = sg | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4); return f;
}

// q6_K (210 B / 256 values) -> per (row, 128-value stage) 128 B int8 + 32 B fp32 scales.
// Values are the stored 6-bit codes minus 32 (exact); scales are the exact d*sc products.
static void q6k_repack_host(const uint8_t* src, uint8_t* dst, int K, int N) {
    const int nsb = K >> 8, nst = K >> 7;
    for (int r = 0; r < N; ++r) {
        const uint8_t* row = src + (size_t)r * nsb * 210;
        for (int h = 0; h < nst; ++h) {
            const uint8_t* b = row + (size_t)(h >> 1) * 210;
            const int hh = h & 1;
            const uint8_t* ql = b + 64 * hh;
            const uint8_t* qh = b + 128 + 32 * hh;
            const int8_t* sc = (const int8_t*)(b + 192 + 8 * hh);
            const float d = q6k_h2f(*(const uint16_t*)(b + 208));
            uint8_t* out = dst + ((size_t)r * nst + h) * Q6K_SROW;
            int8_t* oi = (int8_t*)out;
            float* os = (float*)(out + 128);
            for (int u = 0; u < 4; ++u) {
                const uint8_t* qp = ql + (u & 1) * 32;
                const int sh = 2 * u;
                for (int l = 0; l < 32; ++l) {
                    const int v = ((u < 2) ? (int)(qp[l] & 0xF) : (int)(qp[l] >> 4))
                                | (((qh[l] >> sh) & 3) << 4);
                    oi[u * 32 + l] = (int8_t)(v - 32);
                }
            }
            for (int g = 0; g < 8; ++g) os[g] = d * (float)sc[g];
        }
    }
}

static size_t g_weight_bytes = 0;
static size_t g_prepack_bytes = 0;   // Q5_K 144 B copy: prefill-only, not in the decode stream
static int g_q5k_prepack = -1;   // -1 auto, 0 off, 1 on
static cudaStream_t g_load_st = nullptr;   // stream the weight loader feeds

__global__ void k_q5k_prepack144(const uint8_t* __restrict__ W, uint8_t* __restrict__ W144,
                                 __half* __restrict__ Dm, int K, int N);

static bool q5k_prepack_enabled() {
    if (g_q5k_prepack < 0) {
        const char* e = getenv("BRUCE_Q5K_PREPACK");
        if (e) g_q5k_prepack = atoi(e) ? 1 : 0;
        else return true;   // auto: the up-front budget check reserves the KV cache
    }
    return g_q5k_prepack != 0;
}

// Allocation happens in the load plan (one up-front budget check); this enqueues
// the re-encode on the load stream, so it is ordered after that tensor's copy.
static void q5k_prepack_launch(TensorW& w) {
    if (w.type != GGML_TYPE_Q5_K || !w.dp || !w.dm) return;
    k_q5k_prepack144<<<w.N, 256, 0, g_load_st>>>((const uint8_t*)w.d, (uint8_t*)w.dp,
                                                 (__half*)w.dm, w.K, w.N);
    cudaError_t e = cudaGetLastError();
    if (e) { fprintf(stderr, "q5k prepack failed: %s\n", cudaGetErrorString(e)); exit(1); }
}


// BF16 [N][K] -> head-aligned Q8_0: per (row, 256 values) 8 fp16 group scales
// (16 B) followed by 256 int8 (256 B).  K must be a multiple of 256.
static void q8k_quant_host(const __nv_bfloat16* src, uint8_t* dst, int K, int N) {
    const int nsb = K >> 8;
    for (int r = 0; r < N; ++r) {
        const __nv_bfloat16* row = src + (size_t)r * K;
        uint8_t* out = dst + (size_t)r * nsb * Q8_SROW;
        for (int s = 0; s < nsb; ++s) {
            uint8_t* blk = out + (size_t)s * Q8_SROW;
            __half* sc = (__half*)blk;
            int8_t* q = (int8_t*)(blk + 16);
            for (int g = 0; g < 8; ++g) {
                const __nv_bfloat16* p = row + s * 256 + g * 32;
                float amax = 0.f;
                for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(__bfloat162float(p[i])));
                const float d = amax * (1.f / 127.f);
                const float id = d > 0.f ? 1.f / d : 0.f;
                sc[g] = __float2half(d);
                for (int i = 0; i < 32; ++i) {
                    int v = (int)lrintf(__bfloat162float(p[i]) * id);
                    v = max(-127, min(127, v));
                    q[g * 32 + i] = (int8_t)v;
                }
            }
        }
    }
}

// ---------------------------------------------------------------- streamed weight load
//
// The model drive sustains ~425 MB/s, while a pageable cudaMemcpy out of the
// mmap'ed GGUF only reaches ~310 MB/s and serialises the whole load: each byte
// is faulted in, staged and DMA'd before the next tensor is touched.  Q6_K was
// also read twice (raw upload + host repack) and the BF16 logits head was
// uploaded only to be freed right after its Q8_0 conversion.
//
// This loader reads every file range exactly once into pinned staging slots,
// runs the Q6_K / Q8_0 conversions on worker threads while the next chunk is
// still being read, and feeds one async copy stream.  The load becomes
// disk-bound.  The bytes that land in VRAM are bit-identical to the previous
// path: the same repack/quantise functions are applied to the same rows, only
// chunked and overlapped with the read.
#define LS_SLOT (16u << 20)
#define LS_NIN  5
#define LS_NOUT 3

struct LoadJob {
    int kind = 0;            // 1 = Q6_K 160 B repack, 2 = BF16 -> Q8_0 logits
    int in_i = 0, out_i = 0;
    const uint8_t* src = nullptr;
    void* dst = nullptr;
    size_t out_bytes = 0;
    int K = 0, rows = 0;
    bool wait_copy = false;  // input slot also feeds a raw H2D: free it afterwards
};

struct LoadStream {
    uint8_t* in[LS_NIN] = {};
    std::atomic<bool> in_busy[LS_NIN];
    std::atomic<bool> in_cpu[LS_NIN];    // transform worker is reading the slot
    std::atomic<bool> in_copy[LS_NIN];   // a raw H2D from the slot is queued/running
    uint8_t* out[LS_NOUT] = {};
    // 0 = free, 1 = owned by a transform worker, 2 = queued for H2D
    std::atomic<int> out_state[LS_NOUT];
    cudaStream_t st = nullptr;

    struct LoadJob {
        int kind = 0;            // 1 = Q6_K 160 B repack, 2 = BF16 -> Q8_0 logits
        int in_i = 0, out_i = 0;
        const uint8_t* src = nullptr;
        void* dst = nullptr;
        size_t out_bytes = 0;
        int K = 0, rows = 0;
    };
    struct CopyJob {
        const void* src = nullptr;
        void* dst = nullptr;
        size_t bytes = 0;
        int in_i = -1;    // slot whose raw copy this is (cleared when done)
        int out_i = -1;   // transform output slot (returned to the pool when done)
    };

    std::deque<LoadJob> jobs;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::thread> workers;

    // Every CUDA copy is issued and waited on by this one thread.  On WDDM a
    // host->device memcpy holds the driver's API lock until it has staged the
    // copy, so issuing copies from the reader thread (or polling events there)
    // serialises the disk read with the DMA.  Keeping the reader purely on
    // ReadFile lets the two overlap.
    std::deque<CopyJob> copies;
    std::mutex cp_mu;
    std::condition_variable cp_cv;
    std::thread copy_thread;

    bool stop_w = false, stop_c = false;
    int in_rr = 0, out_rr = 0;
    bool ready = false;

    double t_read = 0, t_wait = 0;   // profile: pread time vs slot-wait time
    std::atomic<double> t_xf{0};      // profile: CPU transform time (sum over workers)
    std::atomic<long long> n_xf{0};

    void init() {
        if (ready) return;
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        g_load_st = st;
        for (int i = 0; i < LS_NIN; ++i) {
            CK(cudaHostAlloc((void**)&in[i], LS_SLOT, cudaHostAllocDefault));
            in_busy[i].store(false); in_cpu[i].store(false); in_copy[i].store(false);
        }
        for (int i = 0; i < LS_NOUT; ++i) {
            CK(cudaHostAlloc((void**)&out[i], LS_SLOT, cudaHostAllocDefault));
            out_state[i].store(0);
        }
        for (int i = 0; i < 2; ++i) workers.emplace_back([this] { worker(); });
        copy_thread = std::thread([this] { copy_worker(); });
        ready = true;
    }

    int take_in() {
        const double t0 = now_ms();
        for (;;) {
            for (int k = 0; k < LS_NIN; ++k) {
                int i = (in_rr + k) % LS_NIN;
                if (in_busy[i].load(std::memory_order_acquire) &&
                    !in_cpu[i].load(std::memory_order_acquire) &&
                    !in_copy[i].load(std::memory_order_acquire))
                    in_busy[i].store(false, std::memory_order_release);
                if (!in_busy[i].exchange(true)) { in_rr = i + 1; t_wait += now_ms() - t0; return i; }
            }
            std::unique_lock<std::mutex> lk(mu);
            cv.wait_for(lk, std::chrono::milliseconds(1));
        }
    }
    int take_out() {
        const double t0 = now_ms();
        for (;;) {
            for (int k = 0; k < LS_NOUT; ++k) {
                int i = (out_rr + k) % LS_NOUT;
                if (out_state[i].load(std::memory_order_acquire) == 0) {
                    int expect = 0;
                    if (out_state[i].compare_exchange_strong(expect, 1, std::memory_order_acq_rel)) {
                        out_rr = i + 1;
                        t_wait += now_ms() - t0;
                        return i;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }

    void worker() {
        for (;;) {
            LoadJob j;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [this] { return stop_w || !jobs.empty(); });
                if (jobs.empty()) return;
                j = jobs.front();
                jobs.pop_front();
            }
            const double tx = now_ms();
            if (j.kind == 1) q6k_repack_host(j.src, out[j.out_i], j.K, j.rows);
            else              q8k_quant_host((const __nv_bfloat16*)j.src, out[j.out_i], j.K, j.rows);
            t_xf.store(t_xf.load(std::memory_order_relaxed) + (now_ms() - tx), std::memory_order_relaxed);
            n_xf.fetch_add(j.rows, std::memory_order_relaxed);
            // the slot is free for the reader as soon as the repack has read it
            in_cpu[j.in_i].store(false, std::memory_order_release);
            cv.notify_all();
            out_state[j.out_i].store(2, std::memory_order_release);
            submit_copy(CopyJob{ out[j.out_i], j.dst, j.out_bytes, -1, j.out_i });
        }
    }

    void copy_worker() {
        for (;;) {
            CopyJob j;
            {
                std::unique_lock<std::mutex> lk(cp_mu);
                cp_cv.wait(lk, [this] { return stop_c || !copies.empty(); });
                if (copies.empty()) return;
                j = copies.front();
                copies.pop_front();
            }
            CK(cudaMemcpyAsync(j.dst, j.src, j.bytes, cudaMemcpyHostToDevice, st));
            CK(cudaStreamSynchronize(st));
            if (j.out_i >= 0) out_state[j.out_i].store(0, std::memory_order_release);
            if (j.in_i >= 0)  in_copy[j.in_i].store(false, std::memory_order_release);
            cv.notify_all();
        }
    }
    void submit_copy(const CopyJob& j) {
        {
            std::lock_guard<std::mutex> lk(cp_mu);
            copies.push_back(j);
        }
        cp_cv.notify_one();
    }
    void submit_raw(int i, void* dst, size_t bytes) {
        in_copy[i].store(true, std::memory_order_release);
        submit_copy(CopyJob{ in[i], dst, bytes, i, -1 });
    }

    // absolute file offset -> VRAM, streamed through the pinned ring
    void raw(const GgufFile& g, uint64_t foff, size_t n, void* dst) {
        size_t done = 0;
        while (done < n) {
            int i = take_in();
            size_t c = std::min(n - done, (size_t)LS_SLOT);
            { const double tr = now_ms();
              if (!g.read_at(in[i], c, foff + done)) { fprintf(stderr, "bruce: read failed\n"); exit(1); }
              t_read += now_ms() - tr; }
            submit_raw(i, (uint8_t*)dst + done, c);
            done += c;
        }
    }

    // Q6_K: one read; optional raw copy + 160 B/128-value repack on a worker.
    void q6k(const GgufFile& g, uint64_t foff, int K, int N, void* raw_dst, void* rep_dst) {
        const size_t srow = (size_t)(K >> 8) * 210;               // GGUF q6_K row
        const size_t orow = (size_t)(K >> 7) * Q6K_SROW;          // repacked row
        int rows = (int)std::max<size_t>(1, std::min<size_t>(N, std::min(LS_SLOT / srow, LS_SLOT / orow)));
        for (int r0 = 0; r0 < N; r0 += rows) {
            int nr = std::min(rows, N - r0);
            int ii = take_in();
            in_cpu[ii].store(true, std::memory_order_release);
            size_t sb = (size_t)nr * srow;
            { const double tr = now_ms();
              if (!g.read_at(in[ii], sb, foff + (size_t)r0 * srow)) { fprintf(stderr, "bruce: read failed\n"); exit(1); }
              t_read += now_ms() - tr; }
            if (raw_dst) submit_raw(ii, (uint8_t*)raw_dst + (size_t)r0 * srow, sb);
            int oi = take_out();
            LoadJob j;
            j.kind = 1; j.in_i = ii; j.out_i = oi;
            j.src = in[ii];
            j.dst = (uint8_t*)rep_dst + (size_t)r0 * orow;
            j.out_bytes = (size_t)nr * orow;
            j.K = K; j.rows = nr;
            { std::lock_guard<std::mutex> lk(mu); jobs.push_back(j); }
            cv.notify_one();
        }
    }

    // BF16 [N][K] -> head-aligned Q8_0 logits: one read, no BF16 VRAM copy.
    void q8logits(const GgufFile& g, uint64_t foff, int K, int N, void* dst) {
        const size_t srow = (size_t)K * 2;
        const size_t orow = (size_t)(K >> 8) * Q8_SROW;
        int rows = (int)std::max<size_t>(1, std::min<size_t>(N, std::min(LS_SLOT / srow, LS_SLOT / orow)));
        for (int r0 = 0; r0 < N; r0 += rows) {
            int nr = std::min(rows, N - r0);
            int ii = take_in();
            in_cpu[ii].store(true, std::memory_order_release);
            { const double tr = now_ms();
              if (!g.read_at(in[ii], (size_t)nr * srow, foff + (size_t)r0 * srow)) {
                  fprintf(stderr, "bruce: read failed\n"); exit(1);
              }
              t_read += now_ms() - tr; }
            int oi = take_out();
            LoadJob j;
            j.kind = 2; j.in_i = ii; j.out_i = oi;
            j.src = in[ii];
            j.dst = (uint8_t*)dst + (size_t)r0 * orow;
            j.out_bytes = (size_t)nr * orow;
            j.K = K; j.rows = nr;
            { std::lock_guard<std::mutex> lk(mu); jobs.push_back(j); }
            cv.notify_one();
        }
    }

    void finish() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop_w = true;
        }
        cv.notify_all();
        for (auto& t : workers) t.join();
        {
            std::lock_guard<std::mutex> lk(cp_mu);
            stop_c = true;
        }
        cp_cv.notify_all();
        if (copy_thread.joinable()) copy_thread.join();
        CK(cudaStreamSynchronize(st));
    }
};

static LoadStream g_ld;

// One tensor range in the load plan.  The plan is sorted by file offset before
// it runs so the reader walks the GGUF strictly forward; the old per-block bind
// order jumped backwards inside every block (q,k,v,o,gate vs the file's
// gate,k,output,q,v) and cost ~10% of the achievable disk rate.
struct LoadItem {
    uint64_t foff = 0;
    int kind = 0;          // 0 = raw, 1 = q6_K raw + 160 B repack, 2 = BF16 -> Q8_0 logits
    const GgufTensor* t = nullptr;
    void* dst = nullptr;
    void* dst2 = nullptr;
    int K = 0, N = 0;
};

enum ScaleKind { SK_ALPHA = 0, SK_SUV, SK_QK, SK_GATE, SK_OUT };
struct ScaleItem { uint64_t foff; int n; int kind; const float** dst; };

static float scale_value(int kind, float v) {
    switch (kind) {
        case SK_ALPHA: return fabsf(v * (ALPHA_INIT / INIT_STD));
        case SK_SUV:   return v * sqrtf((float)N_EMBD);
        case SK_QK:    { const float c = sqrtf((float)HD) / INIT_STD;      const float t = v * c; return t * t; }
        case SK_GATE:  { const float c = sqrtf(GATE_INIT) / INIT_STD;      const float t = v * c; return t * t; }
        default:       return v / INIT_STD;
    }
}

// BF16 [N][K] -> head-aligned Q8_0, exactly the layout q8k_quant_host builds:
// per (row, 256 values) 8 fp16 group scales (16 B) followed by 256 int8 (256 B).
// One warp per 32-value group, so 335M values cost a few ms instead of the
// ~5 s the scalar host loop took (it was the last real cost in the load path).
__global__ void k_bf16_to_q8(const __nv_bfloat16* __restrict__ src, uint8_t* __restrict__ dst,
                             int K, int N) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nsb = K >> 8, groups = nsb * 8;
    const int gx = (groups + 7) >> 3;
    const int row = blockIdx.x / gx;
    const int g = (blockIdx.x % gx) * 8 + warp;
    if (row >= N || g >= groups) return;
    const int s = g >> 3, grp = g & 7;
    const __nv_bfloat16* p = src + (size_t)row * K + (size_t)s * 256 + grp * 32;
    const float v = __bfloat162float(p[lane]);
    float amax = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax * (1.f / 127.f);
    const float id = d > 0.f ? 1.f / d : 0.f;
    uint8_t* blk = dst + (size_t)row * nsb * Q8_SROW + (size_t)s * Q8_SROW;
    if (lane == 0) ((__half*)blk)[grp] = __float2half(d);
    int q = __float2int_rn(v * id);
    q = max(-127, min(127, q));
    ((int8_t*)(blk + 16))[grp * 32 + lane] = (int8_t)q;
}

static void bind_weights(const GgufFile& g, int n_loops) {
    const double t_bind = now_ms();
    log_line("[bruce] uploading weights to VRAM ...");

    std::vector<LoadItem> items;
    std::vector<ScaleItem> scales;
    std::vector<TensorW*> prepack_q5;

    // ---- plan: allocate every destination, record every file range
    auto plan = [&](TensorW& w, const std::string& name) {
        const GgufTensor* t = g.find(name);
        if (!t) { fprintf(stderr, "missing %s\n", name.c_str()); exit(1); }
        w.type = t->type;
        w.K = (int)t->ne[0];
        w.N = (int)t->ne[1];
        w.d = dev_alloc((size_t)t->nbytes);
        g_weight_bytes += (size_t)t->nbytes;
        LoadItem it;
        it.foff = g.data_offset + t->offset;
        it.t = t; it.K = w.K; it.N = w.N;
        it.dst = (void*)w.d;
        if (t->type == GGML_TYPE_Q6_K && (w.K % 256) == 0 && (w.N % 64) == 0) {
            it.kind = 1;
            const size_t nst = (size_t)w.N * (w.K >> 7) * Q6K_SROW;
            w.dp = dev_alloc(nst);
            it.dst2 = (void*)w.dp;
            g_weight_bytes += nst;
        } else {
            it.kind = 0;
            if (t->type == GGML_TYPE_Q5_K) prepack_q5.push_back(&w);
        }
        items.push_back(it);
    };
    auto plan_scale = [&](const std::string& name, int n, int kind, const float** dst) {
        const GgufTensor* t = g.find(name);
        if (!t) { fprintf(stderr, "missing tensor %s\n", name.c_str()); exit(1); }
        scales.push_back(ScaleItem{ g.data_offset + t->offset, n, kind, dst });
    };

    const GgufTensor* te = g.find("token_embd.weight");
    const GgufTensor* to = g.find("output.weight");
    if (!te || !to) { fprintf(stderr, "missing embedding/output\n"); exit(1); }

    M.emb = (const __nv_bfloat16*)dev_alloc((size_t)te->nbytes);
    g_weight_bytes += (size_t)te->nbytes;
    { LoadItem it; it.foff = g.data_offset + te->offset; it.t = te; it.kind = 0;
      it.dst = (void*)M.emb; items.push_back(it); }

    plan_scale("output_scale.weight", VOCAB, SK_OUT, &M.out_scale);

    static int q8_logits = -1;
    if (q8_logits < 0) {
        const char* e = getenv("BRUCE_Q8_LOGITS");
        q8_logits = (!e || atoi(e)) ? 1 : 0;
    }
    const int K_OUTW = (int)to->ne[0], N_OUTW = (int)to->ne[1];
    std::string q8_msg;
    void* out_bf16_tmp = nullptr;
    static int q8_gpu = -1;
    if (q8_gpu < 0) { const char* e = getenv("BRUCE_Q8_GPU"); q8_gpu = (!e || atoi(e)) ? 1 : 0; }
    {
        LoadItem it; it.foff = g.data_offset + to->offset; it.t = to; it.K = K_OUTW; it.N = N_OUTW;
        if (q8_logits && to->type == GGML_TYPE_BF16 && (K_OUTW % 256) == 0) {
            const size_t nb = (size_t)N_OUTW * (K_OUTW >> 8) * Q8_SROW;
            M.out_q8 = (const uint8_t*)dev_alloc(nb);
            if (q8_gpu) {
                // the BF16 head is transient: the quantiser runs on the GPU right
                // after the upload, and the buffer is released before the KV cache
                // is sized, so it never raises the steady-state footprint
                out_bf16_tmp = dev_alloc((size_t)to->nbytes);
                it.kind = 0;
                it.dst = out_bf16_tmp;
            } else {
                it.kind = 2;
                it.dst = (void*)M.out_q8;
            }
            g_weight_bytes += nb;
            char b[128];
            snprintf(b, sizeof b, "output.weight -> Q8_0 logits (%.1f MiB, frees %.1f MiB)",
                     nb / 1048576.0, ((size_t)to->nbytes - nb) / 1048576.0);
            q8_msg = b;
        } else {
            M.out_w = (const __nv_bfloat16*)dev_alloc((size_t)to->nbytes);
            it.kind = 0;
            it.dst = (void*)M.out_w;
            g_weight_bytes += (size_t)to->nbytes;
        }
        items.push_back(it);
    }

    std::string pattern = g.str("dory.layer_pattern");
    std::vector<int> profile;
    {
        auto it = g.kv.find("dory.rope.profile");
        if (it != g.kv.end()) {
            if (!it->second.fa.empty())
                for (float f : it->second.fa) profile.push_back((int)f);
            else
                for (int64_t v : it->second.ia) profile.push_back((int)v);
        }
    }
    if ((int)pattern.size() < 72 || (int)profile.size() < 72) {
        fprintf(stderr, "bad layer pattern/profile\n"); exit(1);
    }

    for (int b = 0; b < 72; ++b) {
        BlockW& w = M.blocks[b];
        char nm[96];
        snprintf(nm, sizeof nm, "blk.%d.residual_alpha.weight", b);
        plan_scale(nm, N_EMBD, SK_ALPHA, &w.alpha);
        char pc = pattern[b];
        w.is_ffn = (pc == '-');
        w.full = (pc == '*');
        w.theta_sel = ((int)profile[b] == 2) ? 2 : 1;
        if (w.is_ffn) {
            snprintf(nm, sizeof nm, "blk.%d.ffn_gate.weight", b); plan(w.wfg, nm);
            snprintf(nm, sizeof nm, "blk.%d.ffn_up.weight", b);   plan(w.wfu, nm);
            snprintf(nm, sizeof nm, "blk.%d.ffn_down.weight", b); plan(w.wfd, nm);
            snprintf(nm, sizeof nm, "blk.%d.ffn_gate_scale.weight", b);
            plan_scale(nm, N_FF, SK_SUV, &w.suv_gate);
            snprintf(nm, sizeof nm, "blk.%d.ffn_up_scale.weight", b);
            plan_scale(nm, N_FF, SK_SUV, &w.suv_up);
        } else {
            snprintf(nm, sizeof nm, "blk.%d.attn_q.weight", b); plan(w.wq, nm);
            snprintf(nm, sizeof nm, "blk.%d.attn_k.weight", b); plan(w.wk, nm);
            snprintf(nm, sizeof nm, "blk.%d.attn_v.weight", b); plan(w.wv, nm);
            snprintf(nm, sizeof nm, "blk.%d.attn_output.weight", b); plan(w.wo, nm);
            snprintf(nm, sizeof nm, "blk.%d.attn_gate.weight", b); plan(w.wg, nm);
            snprintf(nm, sizeof nm, "blk.%d.attn_qk_scale.weight", b);
            plan_scale(nm, N_HEAD * HD, SK_QK, &w.sqk);
            snprintf(nm, sizeof nm, "blk.%d.attn_gate_scale.weight", b);
            plan_scale(nm, N_HEAD * HD, SK_GATE, &w.sgate);
        }
    }

    // ---- Q5_K 144 B prefill pre-unpack: allocate now, launch after the uploads
    bool do_prepack = q5k_prepack_enabled();
    if (do_prepack) {
        size_t need = 0;
        for (TensorW* w : prepack_q5)
            if ((w->K % 512) == 0 && (w->N % 128) == 0)
                need += (size_t)w->N * (w->K >> 7) * 144 + (size_t)w->N * (w->K >> 5) * 2;
        size_t free_b = 0, total_b = 0;
        CK(cudaMemGetInfo(&free_b, &total_b));
        // The KV cache is allocated *after* the weights, so reserve it here:
        // otherwise a long context can consume the headroom this prepack just
        // took and the card ends up with 0 MiB free (which on WDDM silently
        // pages the KV cache and makes prefill ~20x slower).
        const size_t kv_need = (size_t)9 * (size_t)M.n_ctx * KV_POS_BYTES * 2 +
                               (size_t)27 * (size_t)std::max(1, M.n_slots) * SWA_SLOTS * KV_POS_BYTES * 2;
        const size_t reserve = kv_need + (size_t)512 * 1024 * 1024;
        if (need == 0 || free_b < need + reserve) {
            log_line("[bruce] Q5_K pre-pack skipped: needs %.0f MiB + kv %.0f MiB, %.0f MiB free",
                     need / 1048576.0, kv_need / 1048576.0, free_b / 1048576.0);
            do_prepack = false;
        }
        if (do_prepack) {
            for (TensorW* w : prepack_q5) {
                if ((w->K % 512) != 0 || (w->N % 128) != 0) continue;
                const size_t n144 = (size_t)w->N * (w->K >> 7) * 144;
                const size_t ndm  = (size_t)w->N * (w->K >> 5) * 2;
                w->dp = dev_alloc(n144);
                w->dm = dev_alloc(ndm);
                g_weight_bytes += n144 + ndm;
                g_prepack_bytes += n144 + ndm;
            }
        }
    }

    // ---- run: every file byte is read exactly once, in offset order
    std::sort(items.begin(), items.end(),
              [](const LoadItem& a, const LoadItem& b) { return a.foff < b.foff; });
    LoadStream& ld = g_ld;
    ld.init();
    size_t read_bytes = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        const LoadItem& it = items[i];
        const double tq = now_ms();
        if (it.kind == 0)      ld.raw(g, it.foff, (size_t)it.t->nbytes, it.dst);
        else if (it.kind == 1) ld.q6k(g, it.foff, it.K, it.N, it.dst, it.dst2);
        else                   ld.q8logits(g, it.foff, it.K, it.N, it.dst);
        g_lp_h2d += now_ms() - tq;
        read_bytes += (size_t)it.t->nbytes;
        if ((i % 24) == 23 || i + 1 == items.size()) {
            const double el = (now_ms() - t_bind) / 1000.0;
            status_line("[bruce] load %.2f GiB | %.2f GiB/s | %.1fs",
                        read_bytes / 1073741824.0,
                        el > 1e-6 ? read_bytes / (el * 1073741824.0) : 0.0, el);
        }
    }
    status_end();

    // ---- per-layer scales: ~4.4 MB of tiny BF16 tensors, transformed here and
    // uploaded as one buffer (the old path did one cudaMalloc + sync copy each)
    {
        const double tq = now_ms();
        std::sort(scales.begin(), scales.end(),
                  [](const ScaleItem& a, const ScaleItem& b) { return a.foff < b.foff; });
        size_t tot = 0;
        for (const ScaleItem& s : scales) tot += (size_t)s.n;
        std::vector<float> host(tot);
        std::vector<uint8_t> raw;
        size_t off = 0;
        for (const ScaleItem& s : scales) {
            raw.resize((size_t)s.n * 2);
            if (!g.read_at(raw.data(), raw.size(), s.foff)) { fprintf(stderr, "bruce: read failed\n"); exit(1); }
            const __nv_bfloat16* src = (const __nv_bfloat16*)raw.data();
            for (int i = 0; i < s.n; ++i) host[off + i] = scale_value(s.kind, __bfloat162float(src[i]));
            off += (size_t)s.n;
        }
        float* d = (float*)dev_alloc(tot * 4);
        CK(cudaMemcpy(d, host.data(), tot * 4, cudaMemcpyHostToDevice));
        off = 0;
        for (const ScaleItem& s : scales) { *s.dst = d + off; off += (size_t)s.n; }
        g_lp_scale += now_ms() - tq;
    }
    if (!q8_msg.empty()) log_line("[bruce] %s", q8_msg.c_str());

    // ---- prefill pre-unpack runs on the same stream, after the raw copies
    {
        const double tq = now_ms();
        for (TensorW* w : prepack_q5) if (w->dp) q5k_prepack_launch(*w);
        g_lp_prepack += now_ms() - tq;
    }
    {
        const double tq = now_ms();
        ld.finish();
        g_lp_drain += now_ms() - tq;
    }
    if (out_bf16_tmp) {
        const double tq = now_ms();
        const int gx = (((K_OUTW >> 8) * 8) + 7) >> 3;
        const unsigned blocks = (unsigned)((size_t)N_OUTW * (size_t)gx);
        k_bf16_to_q8<<<blocks, 256, 0, g_load_st>>>((const __nv_bfloat16*)out_bf16_tmp,
                                                    (uint8_t*)M.out_q8, K_OUTW, N_OUTW);
        cudaError_t e = cudaGetLastError();
        if (e) { fprintf(stderr, "q8 logits kernel failed: %s\n", cudaGetErrorString(e)); exit(1); }
        CK(cudaStreamSynchronize(g_load_st));
        CK(cudaFree(out_bf16_tmp));
        g_lp_q8 += now_ms() - tq;
    }
    CK(cudaStreamSynchronize(nullptr));

    int nv = 0;
    for (int i = 0; i < N_INPUT; ++i) M.order[nv++] = i;
    for (int l = 0; l < n_loops; ++l)
        for (int i = 0; i < N_REC; ++i) M.order[nv++] = N_INPUT + i;
    for (int i = 0; i < N_OUT; ++i) M.order[nv++] = N_INPUT + N_REC + i;
    M.n_virtual = nv;
    M.n_loops = n_loops;
    // slot table for loop-1 attention reuse: the recurrent full-attention blocks
    // (executed once per loop) get 0..4; every other block gets -1.
    for (int b = 0; b < 72; ++b) M.l1_slot[b] = -1;
    int slot = 0;
    for (int b = N_INPUT; b < N_INPUT + N_REC; ++b)
        if (M.blocks[b].full) M.l1_slot[b] = slot++;
    if (slot != 5) fprintf(stderr, "[bruce] warning: %d recurrent full-attention blocks (expected 5)\n", slot);
}


static size_t g_kv_bytes = 0;

// Point the legacy single-sequence globals (M.kvk_ptr / M.kvv_ptr) at one slot's
// KV partition.  All existing forward paths then run unchanged inside that slot.
static int g_cur_slot = -1;
static void set_cur_slot(int s) {
    if (M.n_slots <= 0) s = 0;
    if (s < 0) s = 0;
    if (s >= M.n_slots) s = M.n_slots - 1;
    if (s == g_cur_slot) return;
    g_cur_slot = s;
    for (int b = 0; b < 72; ++b) { M.kvk_ptr[b] = M.kvk_slot[s][b]; M.kvv_ptr[b] = M.kvv_slot[s][b]; }
}

static void alloc_buffers() {
    int nfull = 0, nswa = 0;
    for (int b = 0; b < 72; ++b) {
        M.kvk_ptr[b] = nullptr;
        M.kvv_ptr[b] = nullptr;
        if (M.blocks[b].is_ffn) continue;
        if (M.blocks[b].full) nfull++; else nswa++;
    }
    if (nfull != 9 || nswa != 27) { fprintf(stderr, "bad kv counts %d %d\n", nfull, nswa); exit(1); }
    if (M.n_slots < 1) M.n_slots = 1;
    if (M.n_slots > MAX_SLOTS) M.n_slots = MAX_SLOTS;
    M.slot_ctx = M.n_ctx / M.n_slots;
    if (M.slot_ctx < 1) M.slot_ctx = 1;
    // Full layers keep one [kvh][slot][272] block per slot, so the pool is the same
    // size as the single-slot engine (n_ctx total).  SWA layers need a disjoint
    // 2*SWA ring per slot because the ring index is position & (SWA_SLOTS-1).
    g_kv_bytes = (size_t)nfull * M.n_slots * M.slot_ctx * KV_POS_BYTES * 2 +
                 (size_t)nswa * M.n_slots * SWA_SLOTS * KV_POS_BYTES * 2;
    CK(cudaMalloc(&M.kvf_k, (size_t)nfull * M.n_slots * M.slot_ctx * KV_POS_BYTES));
    CK(cudaMalloc(&M.kvf_v, (size_t)nfull * M.n_slots * M.slot_ctx * KV_POS_BYTES));
    CK(cudaMalloc(&M.kvs_k, (size_t)nswa * M.n_slots * SWA_SLOTS * KV_POS_BYTES));
    CK(cudaMalloc(&M.kvs_v, (size_t)nswa * M.n_slots * SWA_SLOTS * KV_POS_BYTES));
    { const int nctx_ = M.slot_ctx; CK(cudaMemcpyToSymbol(g_kv_nctx, &nctx_, sizeof(nctx_))); }
    int fi = 0, si = 0;
    for (int b = 0; b < 72; ++b) {
        if (M.blocks[b].is_ffn) continue;
        if (M.blocks[b].full) {
            M.kvk_layer[b] = M.kvf_k + (size_t)fi * M.n_slots * M.slot_ctx * KV_POS_BYTES;
            M.kvv_layer[b] = M.kvf_v + (size_t)fi * M.n_slots * M.slot_ctx * KV_POS_BYTES;
        } else {
            M.kvk_layer[b] = M.kvs_k + (size_t)si * M.n_slots * SWA_SLOTS * KV_POS_BYTES;
            M.kvv_layer[b] = M.kvs_v + (size_t)si * M.n_slots * SWA_SLOTS * KV_POS_BYTES;
        }
        for (int s = 0; s < M.n_slots; ++s) {
            if (M.blocks[b].full) {
                const size_t off = (size_t)s * M.slot_ctx * KV_POS_BYTES;
                M.kvk_slot[s][b] = M.kvk_layer[b] + off;
                M.kvv_slot[s][b] = M.kvv_layer[b] + off;
            } else {
                const size_t off = (size_t)s * SWA_SLOTS * KV_POS_BYTES;
                M.kvk_slot[s][b] = M.kvk_layer[b] + off;
                M.kvv_slot[s][b] = M.kvv_layer[b] + off;
            }
        }
        if (M.blocks[b].full) fi++; else si++;
    }
    g_cur_slot = -1;
    set_cur_slot(0);
    M.x = (float*)dev_alloc(N_EMBD * 4);
    M.x_prev = (float*)dev_alloc(N_EMBD * 4);
    M.d_delta = (float*)dev_alloc(8);
    M.input_h = (float*)dev_alloc(N_EMBD * 4);
    M.q = (float*)dev_alloc(N_HEAD * HD * 4);
    M.k = (float*)dev_alloc(N_KV * HD * 4);
    M.v = (float*)dev_alloc(N_KV * HD * 4);
    M.gate = (float*)dev_alloc(N_HEAD * HD * 4);
    M.aout = (float*)dev_alloc(N_HEAD * HD * 4);
    M.attn_in = (float*)dev_alloc(N_HEAD * HD * 4);
    M.ffn_g = (float*)dev_alloc(N_FF * 4);
    M.ffn_u = (float*)dev_alloc(N_FF * 4);
    M.ffn_t = (float*)dev_alloc(N_FF * 4);
    M.proj = (float*)dev_alloc(N_EMBD * 4);
    M.xq = (int8_t*)dev_alloc(N_FF);
    M.xd = (float*)dev_alloc((N_FF / 32) * 4);
    M.xsum = (int*)dev_alloc((N_FF / 32) * 4);
    M.bm_x = (float*)dev_alloc((size_t)MBUF * N_EMBD * 4);
    M.bm_input_h = (float*)dev_alloc((size_t)MBUF * N_EMBD * 4);
    M.bm_q = (float*)dev_alloc((size_t)MBUF * N_HEAD * HD * 4);
    M.bm_k = (float*)dev_alloc((size_t)MBUF * N_KV * HD * 4);
    M.bm_v = (float*)dev_alloc((size_t)MBUF * N_KV * HD * 4);
    M.bm_gate = (float*)dev_alloc((size_t)MBUF * N_HEAD * HD * 4);
    M.bm_aout = (float*)dev_alloc((size_t)MBUF * N_HEAD * HD * 4);
    // 5 recurrent full-attention blocks x 8 decode rows (loop-1 reuse table)
    M.aout_l0 = (float*)dev_alloc((size_t)5 * 8 * N_HEAD * HD * 4);
    // bm_attn_in is only a debug tap, bm_ffn_t was never read at all
    M.bm_attn_in = g_debug ? (float*)dev_alloc((size_t)MBUF * N_HEAD * HD * 4) : nullptr;
    M.bm_ffn_g = (float*)dev_alloc((size_t)MBUF * N_FF * 4);
    M.bm_ffn_u = (float*)dev_alloc((size_t)MBUF * N_FF * 4);
    M.bm_ffn_t = nullptr;
    M.bm_proj = (float*)dev_alloc((size_t)MBUF * N_EMBD * 4);
    // only speculative verification (<= 8 rows) materialises all logits
    M.bm_logits = (float*)dev_alloc((size_t)9 * VOCAB * 4);
    M.bm_cos = (float*)dev_alloc((size_t)MBUF * 256 * 4);
    M.bm_sin = (float*)dev_alloc((size_t)MBUF * 256 * 4);
    M.bm_xq = (int8_t*)dev_alloc((size_t)MBUF * N_FF);
    M.bm_xd = (float*)dev_alloc((size_t)MBUF * (N_FF / 32) * 4);
    M.bm_xsum = (int*)dev_alloc((size_t)MBUF * (N_FF / 32) * 4);
    M.bm_xdh = (__half*)dev_alloc((size_t)MBUF * (N_FF / 32) * 2);
    M.bm_tok = (int*)dev_alloc((size_t)MBUF * 4);
    M.d_row_pos = (int*)dev_alloc(sizeof(int) * MAX_ROWS);
    M.d_row_slot = (int*)dev_alloc(sizeof(int) * MAX_ROWS);
    M.d_row_act = (int*)dev_alloc(sizeof(int) * MAX_ROWS);
    for (int s = 0; s < M.n_slots; ++s) {
        M.slot_logits[s] = (float*)dev_alloc(VOCAB * 4);
        M.slot_rng[s] = (unsigned long long*)dev_alloc(8);
    }
    M.logits = (float*)dev_alloc(VOCAB * 4);
    M.rope_cos = (float*)dev_alloc(256 * 4);
    M.rope_sin = (float*)dev_alloc(256 * 4);
    M.probs = (float*)dev_alloc(VOCAB * 4);
    M.sort_key = (float*)dev_alloc(VOCAB * 4);
    M.sort_idx = (int*)dev_alloc(VOCAB * 4);
    M.idx_in = (int*)dev_alloc(VOCAB * 4);
    M.samp_max = (float*)dev_alloc(64);
    M.part = (float*)dev_alloc((size_t)PART_ROWS_MAX * PART_ROW_FLOATS * 4);
    M.d_pos = (int*)dev_alloc(4);
    M.d_tok = (int*)dev_alloc(4);
    M.d_amax = (int*)dev_alloc(sizeof(int) * (MBUF + 1));
    M.d_rng = (unsigned long long*)dev_alloc(8);
    {
        std::vector<int> ids(VOCAB);
        for (int i = 0; i < VOCAB; ++i) ids[i] = i;
        CK(cudaMemcpy(M.idx_in, ids.data(), VOCAB * 4, cudaMemcpyHostToDevice));
    }
    {
        int one[MAX_ROWS];
        for (int i = 0; i < MAX_ROWS; ++i) one[i] = 1;
        CK(cudaMemcpy(M.d_row_act, one, sizeof(one), cudaMemcpyHostToDevice));
    }
    size_t tsz = 0;
    cub::DeviceRadixSort::SortPairsDescending(nullptr, tsz, M.probs, M.sort_key,
                                              M.idx_in, M.sort_idx, VOCAB);
    M.sort_tmp_sz = tsz;
    M.sort_tmp = dev_alloc(tsz);
    if (g_debug) {
        M.dbg_x = (float*)dev_alloc((size_t)448 * N_EMBD * 4);
        M.dbg_taps = (float*)dev_alloc((size_t)TAP_N * 4);
    }
}

static int g_gemv_blocks = 112;
static int g_gemm_blocks = 0;
static int gemv_grid(int N, int K) {
    (void)K;
    int warps = (N + 7) / 8;          // one row per warp -> blocks of 8 warps
    const int cap = g_gemv_blocks;
    if (warps <= cap) return warps;
    int passes = (warps + cap - 1) / cap;
    return (warps + passes - 1) / passes;   // balanced tail
}

// batched GEMM grid: one row per warp, WARPS=8 warps per block
static int gemm_grid(int N, int warps_per_block) {
    int warps = (N + 7) / 8;
    int blocks = (warps + warps_per_block - 1) / warps_per_block;
    int cap = g_gemm_blocks > 0 ? g_gemm_blocks : 28;   // one block per SM by default
    return std::max(1, std::min(blocks, cap));
}

// batched GEMM with R rows unrolled per warp
static int gemm_grid2(int N, int warps_per_block, int R) {
    int warps = (N + R - 1) / R;
    int blocks = (warps + warps_per_block - 1) / warps_per_block;
    int cap = g_gemm_blocks > 0 ? g_gemm_blocks : 28;
    return std::max(1, std::min(blocks, cap));
}

static void dbg_copy(float* dst, const float* src, int n) {
    k_copy<<<(n + 255) / 256, 256, 0, S>>>(dst, src, n);
}

static void launch_gemm_q6k_dp(float* y, const TensorW& w, int rows,
                                const int8_t* xq, const float* xd, int xoff, cudaStream_t st);

static void launch_gemv(float* y, const TensorW& w, const float* x, cudaStream_t st) {
    int K = w.K, N = w.N;
    if (w.type == GGML_TYPE_Q6_K && w.dp) {
        launch_gemm_q6k_dp(y, w, 1, M.xq, M.xd, 0, st);
        return;
    }
    int blocks = gemv_grid(N, K);
    if (w.type == GGML_TYPE_Q5_K)
        k_gemv_q5k<8><<<blocks, 256, (size_t)K, st>>>(y, (const uint8_t*)w.d, M.xq, M.xd, M.xsum, K, N);
    else if (w.type == GGML_TYPE_Q6_K)
        k_gemm_q6k<1,16,8><<<gemm_grid2(N,16,8), 512, (size_t)K, st>>>(
            y, (const uint8_t*)w.d, M.xq, M.xd, K, N);
    else {
        fprintf(stderr, "unsupported gemv type %u\n", w.type);
        exit(1);
    }
}

static void set_batched_attrs() {
    const int sm = 98 * 1024;   // widest Q6_K row group (32 rows x 2560) is 80 KB
    #define SETA(k, M, R) CK(cudaFuncSetAttribute((const void*)&k<M,16,R>, \
        cudaFuncAttributeMaxDynamicSharedMemorySize, sm))
    SETA(k_gemm_q5k, 1, 8); SETA(k_gemm_q5k, 2, 8); SETA(k_gemm_q5k, 3, 8); SETA(k_gemm_q5k, 4, 8);
    SETA(k_gemm_q5k, 5, 2); SETA(k_gemm_q5k, 6, 2); SETA(k_gemm_q5k, 7, 2); SETA(k_gemm_q5k, 8, 2);
    SETA(k_gemm_q6k, 1, 8); SETA(k_gemm_q6k, 2, 8); SETA(k_gemm_q6k, 3, 8); SETA(k_gemm_q6k, 4, 8);
    SETA(k_gemm_q6k, 5, 2); SETA(k_gemm_q6k, 6, 2); SETA(k_gemm_q6k, 7, 2); SETA(k_gemm_q6k, 8, 2);
SETA(k_gemm_q6k, 16, 2); SETA(k_gemm_q6k, 32, 1);
    #undef SETA
    size_t smb = 48 * 1024;
    #define SETD(M) CK(cudaFuncSetAttribute((const void*)&k_gemv_q6k_dp<M,8>, \
        cudaFuncAttributeMaxDynamicSharedMemorySize, sm))
    SETD(1); SETD(2); SETD(3); SETD(4); SETD(5); SETD(6); SETD(7); SETD(8);
    #undef SETD
    CK(cudaFuncSetAttribute((const void*)&k_gemm_bf16<1,8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smb));
    CK(cudaFuncSetAttribute((const void*)&k_gemm_bf16<2,8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smb));
    CK(cudaFuncSetAttribute((const void*)&k_gemm_bf16<3,8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smb));
    CK(cudaFuncSetAttribute((const void*)&k_gemm_bf16<4,8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smb));
    CK(cudaFuncSetAttribute((const void*)&k_gemm_bf16<5,8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smb));
    CK(cudaFuncSetAttribute((const void*)&k_gemm_bf16<6,8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smb));
    CK(cudaFuncSetAttribute((const void*)&k_gemm_bf16<7,8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smb));
    CK(cudaFuncSetAttribute((const void*)&k_gemm_bf16<8,8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smb));
}

// ---------------------------------------------------------------------------
// Prefill GEMM: int8 tensor-core Q5_K matrix product over MBUF prompt rows.
//
// 64 weight rows x 64 activation rows per block, 8 warps (warp tile 32x16),
// mma.m16n8k32.s8 with the int8 activations the q8_0 path already produces.
// K is walked in half-superblock stages (128 values) that are double buffered:
// the raw 5-bit bytes for stage s+1 are copied with cp.async at the top of
// stage s and unpacked in shared memory in place after the compute, so the
// DRAM latency of the weight stream hides behind the tensor cores.
//
// The int8 dot per 32-value group is exact, so the only rounding differences
// versus k_gemm_q5k are the fp32 scale application and the fp16->fp32 decode of
// d/dmin (same order as the reference), i.e. results match to fp32 rounding.
#define MMA_BKP 144
#define MMA_BK  128

__device__ __forceinline__ void mma_q5k_scale(uint32_t w0, uint32_t w1, uint32_t w2, int js,
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

__device__ __forceinline__ void mma_f16_acc(float (&d)[4], const uint32_t (&a)[4],
                                            const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// int8 MMA whose integer accumulator starts at the 2^23 offset, so the result is
// already an IEEE float bit pattern (12582912 + dot) and the epilogue needs one
// FADD instead of four quarter-rate I2FP conversions.  Exact for |dot| < 2^22.
__device__ __forceinline__ void mma_s8_bias(float (&cf)[4], const uint32_t (&a)[4],
                                            const uint32_t (&b)[2]) {
    int c0, c1, c2, c3;
    const int bias = 0x4B400000;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(bias));
    cf[0] = __int_as_float(c0) - 12582912.0f;
    cf[1] = __int_as_float(c1) - 12582912.0f;
    cf[2] = __int_as_float(c2) - 12582912.0f;
    cf[3] = __int_as_float(c3) - 12582912.0f;
}

__device__ __forceinline__ void mma_s8_k16_bias(float (&cf)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    int c0, c1, c2, c3;
    const int bias = 0x4B400000;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(bias));
    cf[0] = __int_as_float(c0) - 12582912.0f;
    cf[1] = __int_as_float(c1) - 12582912.0f;
    cf[2] = __int_as_float(c2) - 12582912.0f;
    cf[3] = __int_as_float(c3) - 12582912.0f;
}

// ---------------------------------------------------------------------------
// Load-time Q5_K pre-unpack (144 B per (row, 128 values)):
//   128 B int8 q (the 5-bit code, zero-extended)
//    16 B float4 ds = d_scalar * sc_{row,32-group}
// separately, one __half per (row, 32-group): dm = -dmin_scalar * m_{row,32-group}.
// The values are the stored codes, not a re-quantisation, so the int8 dot is
// identical to the shipping kernel.  Runs once at load on the GPU (~3.3 GB of
// writes for the 4B model).
// ---------------------------------------------------------------------------
__global__ void k_q5k_prepack144(const uint8_t* __restrict__ W, uint8_t* __restrict__ W144,
                                 __half* __restrict__ Dm, int K, int N) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nsb = K >> 8, nst = K >> 7, nb = K >> 5;
    for (int sb = warp; sb < nsb; sb += 8) {
        const uint8_t* b = W + ((size_t)row * nsb + sb) * 176;
        const uint32_t w0 = *(const uint32_t*)(b + 4);
        const uint32_t w1 = *(const uint32_t*)(b + 8);
        const uint32_t w2 = *(const uint32_t*)(b + 12);
        const float d    = __half2float(*(const __half*)b);
        const float dmin = __half2float(*(const __half*)(b + 2));
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int j = 4 * hh + (lane >> 3);
            const int o4 = (lane & 7) * 4;
            const uint32_t qs = *(const uint32_t*)(b + 48 + 32 * (j >> 1) + o4);
            const uint32_t qh = *(const uint32_t*)(b + 16 + o4);
            const uint32_t val = ((qs >> (4 * (j & 1))) & 0x0F0F0F0Fu)
                               | (((qh >> j) & 0x01010101u) << 4);
            // stage-major: [stage][row], so one CTA's whole 128-row stage is a
            // contiguous 18,432 B run and cp.async copies it with full 512 B
            // coalescing (a row-major layout costs 4 segments per warp copy).
            uint8_t* out = W144 + ((size_t)(sb * 2 + hh) * N + row) * 144;
            *(uint32_t*)(out + lane * 4) = val;
            if (lane < 4) {
                float ds, dm;
                mma_q5k_scale(w0, w1, w2, 4 * hh + lane, d, dmin, &ds, &dm);
                ((float*)(out + 128))[lane] = ds;
                Dm[(size_t)row * nb + sb * 8 + 4 * hh + lane] = __float2half(-dm);
            }
        }
    }
}


// half-width int8 MMA (9 cycles vs 18 on SM86): the Q6_K path needs one
// instruction per 16-value scale group anyway, so the k=32 form only wastes
// half of each instruction's slots.
__device__ __forceinline__ void mma_s8_k16(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ void mma_cp16(void* dst, const void* src) {
    unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(src));
}
__device__ __forceinline__ void mma_cp16_ca(void* dst, const void* src) {
    unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(src));
}
__device__ __forceinline__ void mma_cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void mma_cp_wait()   { asm volatile("cp.async.wait_group 0;\n"); }
__device__ __forceinline__ void mma_cp_wait1()  { asm volatile("cp.async.wait_group 1;\n"); }

// ---------------------------------------------------------------------------
// Fused decode GEMV: the same one-row-per-warp dp4a loop as k_gemv_q5k, but up
// to four Q5_K matrices that share the activation vector are concatenated into
// one launch (wq+wk+wg+wv, wfg+wfu).  That removes the per-matrix wave tail
// (N=1024, 2560, 4096 are not multiples of the 64-row block tile) and 4 launches
// per block become one.
// ---------------------------------------------------------------------------
template <int WARPS>
__global__ void __launch_bounds__(32 * WARPS) k_gemv_q5k_fused(
        const uint8_t* __restrict__ W0, const uint8_t* __restrict__ W1,
        const uint8_t* __restrict__ W2, const uint8_t* __restrict__ W3,
        float* __restrict__ Y0, float* __restrict__ Y1,
        float* __restrict__ Y2, float* __restrict__ Y3,
        int N0, int N1, int N2, int N3, int nparts, int Ntot,
        const int8_t* __restrict__ xq, const float* __restrict__ xd,
        const int* __restrict__ xsum, int K) {
    extern __shared__ int sxi[];
    const int nw = K >> 2;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) {
        const int sb = i >> 6, u = i & 63;
        const int js = 2 * ((u >> 3) & 3) + (u >> 5), q = u & 7;
        sxi[i] = *(const int*)(xq + sb * 256 + js * 32 + q * 4);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int js = lane & 7, sl = lane >> 3;
    const int sbh = js >> 1;
    const bool nibl = js & 1;
    const int xsbase = (js >> 1) * 8 + (js & 1) * 32;
    const int nsb = K >> 8;
    const int stride = gridDim.x * WARPS;
    for (int row0 = blockIdx.x * WARPS + warp; row0 < Ntot; row0 += stride) {
        int row = row0;
        const uint8_t* Wp = W0; float* Yp = Y0; int Nloc = N0;
        if (nparts > 1 && row >= Nloc) { row -= Nloc; Wp = W1; Yp = Y1; Nloc = N1; }
        if (nparts > 2 && row >= Nloc) { row -= Nloc; Wp = W2; Yp = Y2; Nloc = N2; }
        if (nparts > 3 && row >= Nloc) { row -= Nloc; Wp = W3; Yp = Y3; Nloc = N3; }
        const uint8_t* rp = Wp + (size_t)row * (size_t)nsb * 176;
        float acc = 0.f;
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
                int dot = 0;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t nib = nibl ? ((qsl[i] >> 4) & 0x0F0F0F0Fu) : (qsl[i] & 0x0F0F0F0Fu);
                    const uint32_t hi = ((qhl[i] >> js) & 0x01010101u) << 4;
                    dot = __dp4a(sxi[sb * 64 + xsbase + i], (int)(nib | hi), dot);
                }
                const uint32_t w0s = *(const uint32_t*)(b + 4);
                const uint32_t w1s = *(const uint32_t*)(b + 8);
                const uint32_t w2s = *(const uint32_t*)(b + 12);
                float ds, dm;
                q5k_unpack_sc(w0s, w1s, w2s, js, __half2float(*(const __half*)(b)),
                              __half2float(*(const __half*)(b + 2)), &ds, &dm);
                const int isb = sb * 8 + js;
                acc += xd[isb] * (ds * (float)dot - dm * (float)xsum[isb]);
            }
        }
        acc = warp_sum(acc);
        if (lane == 0) Yp[row] = acc;
    }
}

// ---------------------------------------------------------------------------
// Barrier-free register-direct tensor-core Q5_K GEMM for 2..8 verified rows.
//
// mma.m16n8k32.s8 has M_mma=16, N_mma=8, K_mma=32 and runs at 4x dp4a on SM86.
// The 16 weight rows go on operand A (fully utilised because N is a multiple of
// 16) and the M<=8 activation rows on operand B.  The activation matrix and its
// pre-combined scales are staged in shared once; weights stream from global to
// registers, so there is exactly one __syncthreads() in the whole kernel and
// the per-row cost is ~0.06 ms instead of ~2.9 ms.
//
// Up to four Q5_K matrices that share the activation tile are fused: the row
// index space is their concatenated N and a block resolves its part with three
// compares (exactly like the prefill MMQ kernel).
// ---------------------------------------------------------------------------
template <int M, int WARPS, bool PF, bool STAGE = false>
__global__ void __launch_bounds__(32 * WARPS, 2) k_gemm_tc_verify_q5k(
        const uint8_t* __restrict__ W0, const uint8_t* __restrict__ W1,
        const uint8_t* __restrict__ W2, const uint8_t* __restrict__ W3,
        float* __restrict__ Y0, float* __restrict__ Y1,
        float* __restrict__ Y2, float* __restrict__ Y3,
        int N0, int N1, int N2, int N3, int nparts, int Ntot,
        const int8_t* __restrict__ xq, const float* __restrict__ xd,
        const int* __restrict__ xsum, int K, int xoff) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int nb32 = K >> 5, nsb = K >> 8;
    int2* sX = (int2*)smem;                            // [nb32][32] int2
    float2* sXds = (float2*)(sX + nb32 * 32);          // [nb32][8] {xd, xd*xsum}

    for (int i = threadIdx.x; i < nb32 * 32; i += blockDim.x) {
        const int isb = i >> 5, lid = i & 31;
        const int m = lid >> 2, t4 = lid & 3;
        int2 v = make_int2(0, 0);
        if (m < M) {
            const int8_t* xp = xq + (size_t)(xoff + m) * K + isb * 32 + t4 * 4;
            v.x = *(const int*)(xp); v.y = *(const int*)(xp + 16);
        }
        sX[i] = v;
    }
    for (int i = threadIdx.x; i < nb32 * 8; i += blockDim.x) {
        const int isb = i >> 3, m = i & 7;
        float d = 0.f;
        int xs = 0;
        if (m < M) {
            d = xd[(size_t)(xoff + m) * (K >> 5) + isb];
            xs = xsum[(size_t)(xoff + m) * (K >> 5) + isb];
        }
        sXds[i] = make_float2(d, d * (float)xs);
    }
    __syncthreads();   // the only barrier in the kernel

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gid = lane >> 2, t4 = lane & 3;
    const int stride = gridDim.x * WARPS * 16;
    for (int row0 = (blockIdx.x * WARPS + warp) * 16; row0 < Ntot; row0 += stride) {
        int rloc = row0;
        const uint8_t* Wp = W0; float* Yp = Y0; int Nloc = N0;
        if (nparts > 1 && rloc >= Nloc) { rloc -= Nloc; Wp = W1; Yp = Y1; Nloc = N1; }
        if (nparts > 2 && rloc >= Nloc) { rloc -= Nloc; Wp = W2; Yp = Y2; Nloc = N2; }
        if (nparts > 3 && rloc >= Nloc) { rloc -= Nloc; Wp = W3; Yp = Y3; Nloc = N3; }

        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        const uint8_t* w0_row = Wp + (size_t)(rloc + gid) * nsb * 176;
        const uint8_t* w1_row = Wp + (size_t)(rloc + gid + 8) * nsb * 176;
        // One super-block of both rows lives in registers; the whole load set is
        // issued one iteration ahead of the MMA/epilogue that consumes it.  With
        // only one CTA per SM this prefetch is what keeps the weight stream at
        // ~290-315 GB/s instead of the ~250 GB/s the unpipelined form reaches
        // (measured, tmp/vb3.cu); no extra shared memory, no barrier.
        struct SbRegs {
            int4 h0, h1;
            uint32_t qh[2][2];
            uint32_t qs[2][4][2];
            float d[2], dm[2];
        };
        auto load_sb_ptr = [&](const uint8_t* b0, const uint8_t* b1, SbRegs& r) {
            r.h0 = *(const int4*)(b0); r.h1 = *(const int4*)(b1);
            r.d[0] = __half2float(*(const __half*)&r.h0.x);
            r.dm[0] = __half2float(*(const __half*)((const char*)&r.h0.x + 2));
            r.d[1] = __half2float(*(const __half*)&r.h1.x);
            r.dm[1] = __half2float(*(const __half*)((const char*)&r.h1.x + 2));
            r.qh[0][0] = *(const uint32_t*)(b0 + 16 + t4 * 4);
            r.qh[0][1] = *(const uint32_t*)(b0 + 32 + t4 * 4);
            r.qh[1][0] = *(const uint32_t*)(b1 + 16 + t4 * 4);
            r.qh[1][1] = *(const uint32_t*)(b1 + 32 + t4 * 4);
            #pragma unroll
            for (int sbh = 0; sbh < 4; ++sbh) {
                r.qs[0][sbh][0] = *(const uint32_t*)(b0 + 48 + 32 * sbh + t4 * 4);
                r.qs[0][sbh][1] = *(const uint32_t*)(b0 + 64 + 32 * sbh + t4 * 4);
                r.qs[1][sbh][0] = *(const uint32_t*)(b1 + 48 + 32 * sbh + t4 * 4);
                r.qs[1][sbh][1] = *(const uint32_t*)(b1 + 64 + 32 * sbh + t4 * 4);
            }
        };
        auto load_sb = [&](int sb, SbRegs& r) {
            load_sb_ptr(w0_row + sb * 176, w1_row + sb * 176, r);
        };
        (void)load_sb;
        auto compute_sb = [&](int sb, const SbRegs& r) {
            #pragma unroll
            for (int sbh = 0; sbh < 4; ++sbh) {
                #pragma unroll
                for (int nibl = 0; nibl < 2; ++nibl) {
                    const int js = 2 * sbh + nibl, isb = sb * 8 + js;
                    const uint32_t sh = 4 * nibl;
                    uint32_t a[4] = {
                        ((r.qs[0][sbh][0] >> sh) & 0x0F0F0F0Fu) | (((r.qh[0][0] >> js) & 0x01010101u) << 4),
                        ((r.qs[1][sbh][0] >> sh) & 0x0F0F0F0Fu) | (((r.qh[1][0] >> js) & 0x01010101u) << 4),
                        ((r.qs[0][sbh][1] >> sh) & 0x0F0F0F0Fu) | (((r.qh[0][1] >> js) & 0x01010101u) << 4),
                        ((r.qs[1][sbh][1] >> sh) & 0x0F0F0F0Fu) | (((r.qh[1][1] >> js) & 0x01010101u) << 4)
                    };
                    const int2 bx = sX[isb * 32 + lane];
                    uint32_t b[2] = {(uint32_t)bx.x, (uint32_t)bx.y};
                    int c[4] = {0, 0, 0, 0};
                    mma_s8(c, a, b);

                    float ds0, dmn0, ds1, dmn1;
                    q5k_unpack_sc(r.h0.y, r.h0.z, r.h0.w, js, r.d[0], r.dm[0], &ds0, &dmn0);
                    q5k_unpack_sc(r.h1.y, r.h1.z, r.h1.w, js, r.d[1], r.dm[1], &ds1, &dmn1);
                    const float2 xds0 = sXds[isb * 8 + 2 * t4], xds1 = sXds[isb * 8 + 2 * t4 + 1];
                    acc[0] = fmaf(ds0 * xds0.x, (float)c[0], fmaf(-dmn0, xds0.y, acc[0]));
                    acc[1] = fmaf(ds0 * xds1.x, (float)c[1], fmaf(-dmn0, xds1.y, acc[1]));
                    acc[2] = fmaf(ds1 * xds0.x, (float)c[2], fmaf(-dmn1, xds0.y, acc[2]));
                    acc[3] = fmaf(ds1 * xds1.x, (float)c[3], fmaf(-dmn1, xds1.y, acc[3]));
                }
            }
        };
        if constexpr (STAGE) {
            // The register-direct form issues 22 four-byte loads per lane per
            // super-block; each warp instruction touches eight rows x 16 B, i.e.
            // 2x the 32 B sectors it uses.  Staging the 16x176 B tile coalesced
            // (16 B per lane, 512 B per instruction) and reading it back from
            // shared turns that into ~18 sectors per 512 B and one cp.async
            // group per super-block.  Double buffered so the copy for sb+1 is in
            // flight during sb's MMA/epilogue; only __syncwarp() is needed
            // because each warp owns its two tiles.
            uint8_t* sW = (uint8_t*)(sXds + nb32 * 8);
            uint8_t* myW = sW + (size_t)warp * 2 * (16 * 176);
            auto issue = [&](int sb, int buf) {
                uint8_t* dst = myW + (size_t)buf * (16 * 176);
                #pragma unroll
                for (int i = 0; i < 6; ++i) {
                    const int c = i * 32 + lane;
                    if (c < 16 * 11) {
                        const int rr = c / 11, cc = (c % 11) * 16;
                        mma_cp16(dst + c * 16,
                                 Wp + (size_t)(rloc + rr) * nsb * 176 + (size_t)sb * 176 + cc);
                    }
                }
            };
            int buf = 0;
            issue(0, 0); mma_cp_commit();
            for (int sb = 0; sb < nsb; ++sb) {
                if (sb + 1 < nsb) { issue(sb + 1, buf ^ 1); mma_cp_commit(); mma_cp_wait1(); }
                else mma_cp_wait();
                __syncwarp();
                const uint8_t* b = myW + (size_t)buf * (16 * 176);
                SbRegs r;
                load_sb_ptr(b + gid * 176, b + (gid + 8) * 176, r);
                compute_sb(sb, r);
                __syncwarp();
                buf ^= 1;
            }
        } else if (PF) {
            SbRegs cur, nxt;
            load_sb(0, cur);
            for (int sb = 0; sb < nsb; ++sb) {
                if (sb + 1 < nsb) load_sb(sb + 1, nxt);
                compute_sb(sb, cur);
                cur = nxt;
            }
        } else {
            for (int sb = 0; sb < nsb; ++sb) {
                SbRegs r;
                load_sb(sb, r);
                compute_sb(sb, r);
            }
        }
        // the caller guarantees Nloc is a multiple of 16, so the tile is in one part
        const int m0 = 2 * t4, m1 = 2 * t4 + 1;
        if (m0 < M) {
            Yp[(size_t)m0 * Nloc + rloc + gid] = acc[0];
            Yp[(size_t)m0 * Nloc + rloc + gid + 8] = acc[2];
        }
        if (m1 < M) {
            Yp[(size_t)m1 * Nloc + rloc + gid] = acc[1];
            Yp[(size_t)m1 * Nloc + rloc + gid + 8] = acc[3];
        }
    }
}

// ---------------------------------------------------------------------------
// Q6_K verify GEMM on the aligned 160 B repack (`w.dp`: 128 int8 values + 8
// fp32 d*sc per row per 128 values).  Same shape as the Q5_K verify kernel --
// 16 weight rows on operand A, up to 8 verified rows on operand B, activations
// staged once in shared -- but Q6_K's scale group is 16 values wide, so one
// mma.m16n8k16.s8 covers exactly one group (half the cycles of k32 on SM86).
//
// The packed 210 B GGUF layout can only be read with two 16-bit loads per
// 4-byte group, which is what makes the raw Q6_K path slow at 5..8 rows
// (measured 120-143 GB/s in-engine); this form needs four 4-byte loads per
// 32 values and streams at the same rate as the Q5_K verify kernel.
// ---------------------------------------------------------------------------
template <int M, int WARPS, bool PF>
__global__ void __launch_bounds__(32 * WARPS, 2) k_gemm_tc_verify_q6k(
        const uint8_t* __restrict__ W0, const uint8_t* __restrict__ W1,
        const uint8_t* __restrict__ W2, const uint8_t* __restrict__ W3,
        float* __restrict__ Y0, float* __restrict__ Y1,
        float* __restrict__ Y2, float* __restrict__ Y3,
        int N0, int N1, int N2, int N3, int nparts, int Ntot,
        const int8_t* __restrict__ xq, const float* __restrict__ xd,
        int K, int xoff) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int nb32 = K >> 5, nst = K >> 7;
    int2* sX = (int2*)smem;                        // [nb32][32] two k16 fragments per lane
    float* sXd = (float*)(sX + nb32 * 32);         // [nb32][8] activation scales

    for (int i = threadIdx.x; i < nb32 * 32; i += blockDim.x) {
        const int isb = i >> 5, lid = i & 31;
        const int m = lid >> 2, t4 = lid & 3;
        int2 v = make_int2(0, 0);
        if (m < M) {
            const int8_t* xp = xq + (size_t)(xoff + m) * K + isb * 32 + t4 * 4;
            v.x = *(const int*)(xp); v.y = *(const int*)(xp + 16);
        }
        sX[i] = v;
    }
    for (int i = threadIdx.x; i < nb32 * 8; i += blockDim.x) {
        const int isb = i >> 3, m = i & 7;
        float d = 0.f;
        if (m < M) d = xd[(size_t)(xoff + m) * nb32 + isb];
        sXd[i] = d;
    }
    __syncthreads();   // the only barrier

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gid = lane >> 2, t4 = lane & 3;
    const int stride = gridDim.x * WARPS * 16;
    for (int row0 = (blockIdx.x * WARPS + warp) * 16; row0 < Ntot; row0 += stride) {
        int rloc = row0;
        const uint8_t* Wp = W0; float* Yp = Y0; int Nloc = N0;
        if (nparts > 1 && rloc >= Nloc) { rloc -= Nloc; Wp = W1; Yp = Y1; Nloc = N1; }
        if (nparts > 2 && rloc >= Nloc) { rloc -= Nloc; Wp = W2; Yp = Y2; Nloc = N2; }
        if (nparts > 3 && rloc >= Nloc) { rloc -= Nloc; Wp = W3; Yp = Y3; Nloc = N3; }

        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        const uint8_t* w0_row = Wp + (size_t)(rloc + gid) * nst * 160;
        const uint8_t* w1_row = Wp + (size_t)(rloc + gid + 8) * nst * 160;
        struct StRegs {
            uint32_t q[2][8];
            float ws[2][8];
        };
        auto load_st = [&](int st, StRegs& r) {
            const uint8_t* b0 = w0_row + st * 160;
            const uint8_t* b1 = w1_row + st * 160;
            #pragma unroll
            for (int g = 0; g < 8; ++g) {
                r.q[0][g] = *(const uint32_t*)(b0 + 16 * g + t4 * 4);
                r.q[1][g] = *(const uint32_t*)(b1 + 16 * g + t4 * 4);
                r.ws[0][g] = ((const float*)(b0 + 128))[g];
                r.ws[1][g] = ((const float*)(b1 + 128))[g];
            }
        };
        auto compute_st = [&](int st, const StRegs& r) {
            #pragma unroll
            for (int h = 0; h < 4; ++h) {
                const int isb = 4 * st + h;
                const int2 bx = sX[isb * 32 + lane];
                const float xds0 = sXd[isb * 8 + 2 * t4], xds1 = sXd[isb * 8 + 2 * t4 + 1];
                #pragma unroll
                for (int gg = 0; gg < 2; ++gg) {
                    const int g = 2 * h + gg;
                    const uint32_t b0 = gg ? (uint32_t)bx.y : (uint32_t)bx.x;
                    int c[4] = {0, 0, 0, 0};
                    mma_s8_k16(c, r.q[0][g], r.q[1][g], b0);
                    const float w0 = r.ws[0][g], w1 = r.ws[1][g];
                    acc[0] = fmaf(xds0 * w0, (float)c[0], acc[0]);
                    acc[1] = fmaf(xds1 * w0, (float)c[1], acc[1]);
                    acc[2] = fmaf(xds0 * w1, (float)c[2], acc[2]);
                    acc[3] = fmaf(xds1 * w1, (float)c[3], acc[3]);
                }
            }
        };
        if (PF) {
            StRegs cur, nxt;
            load_st(0, cur);
            for (int st = 0; st < nst; ++st) {
                if (st + 1 < nst) load_st(st + 1, nxt);
                compute_st(st, cur);
                cur = nxt;
            }
        } else {
            for (int st = 0; st < nst; ++st) {
                StRegs r;
                load_st(st, r);
                compute_st(st, r);
            }
        }
        const int m0 = 2 * t4, m1 = 2 * t4 + 1;
        if (m0 < M) {
            Yp[(size_t)m0 * Nloc + rloc + gid] = acc[0];
            Yp[(size_t)m0 * Nloc + rloc + gid + 8] = acc[2];
        }
        if (m1 < M) {
            Yp[(size_t)m1 * Nloc + rloc + gid] = acc[1];
            Yp[(size_t)m1 * Nloc + rloc + gid + 8] = acc[3];
        }
    }
}

__global__ void __launch_bounds__(256, 2)
k_gemm_mma_q5k(float* __restrict__ y, const uint8_t* __restrict__ W,
               const int8_t* __restrict__ xq, const float* __restrict__ xd,
               const int* __restrict__ xsum, int K, int N, int rows) {
    extern __shared__ __align__(16) int8_t mma_sh[];
    int8_t* sA  = mma_sh;                                  // [2][64][MMA_BKP] int8 activations
    int8_t* sB  = sA + 2 * 64 * MMA_BKP;               // [2][64][MMA_BKP] raw -> unpacked in place
    float*  sAd = (float*)(sB + 2 * 64 * MMA_BKP);     // [2][64][4] activation d
    float*  sAx = sAd + 2 * 256;                       // [2][64][4] activation d*sum
    float*  sBd = sAx + 2 * 256;                       // [2][64][4] weight ds
    float*  sBm = sBd + 2 * 256;                       // [2][64][4] weight dm

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int grp = lane >> 2, tg = lane & 3;
    const int wm = warp >> 2, wn = warp & 3;
    const int nstage = K / MMA_BK;
    const int nsb = K >> 8;
    const int nbase = blockIdx.x * 64;
    const int row0 = blockIdx.y * 64;
    const int srow = tid >> 2, sci = tid & 3;

    auto stage_async = [&](int s, int buf) {
        const int gk = s * MMA_BK, hh = s & 1, h2 = s >> 1;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int c = tid + i * 256;
            const int r = c >> 3, o = (c & 7) << 4;
            int8_t* d = sA + ((size_t)buf * 64 + r) * MMA_BKP + o;
            const int gm = row0 + r;
            if (gm < rows) mma_cp16(d, xq + (size_t)gm * K + gk + o);
            else           *(int4*)d = make_int4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int c = tid + i * 256;
            if (c < 448) {
                const int r = c / 7, k = c - r * 7;
                const int so = (k == 0) ? 0 : (k < 3 ? 16 + (k - 1) * 16 : 48 + 64 * hh + (k - 3) * 16);
                const uint8_t* wb = W + (size_t)(nbase + r) * (size_t)nsb * 176 + (size_t)h2 * 176;
                mma_cp16(sB + ((size_t)buf * 64 + r) * MMA_BKP + (k << 4), wb + so);
            }
        }
        if (tid < 64) {
            const int gm = row0 + tid;
            float* dd = sAd + (size_t)buf * 256 + tid * 4;
            if (gm < rows) mma_cp16(dd, xd + (size_t)gm * (K >> 5) + s * 4);
            else           *(int4*)dd = make_int4(0, 0, 0, 0);
        } else if (tid < 128) {
            const int r = tid - 64, gm = row0 + r;
            float* xx = sAx + (size_t)buf * 256 + r * 4;
            if (gm < rows) mma_cp16(xx, xsum + (size_t)gm * (K >> 5) + s * 4);
            else           *(int4*)xx = make_int4(0, 0, 0, 0);
        }
        mma_cp_commit();
    };

    auto unpack = [&](int s, int buf) {
        const int hh = s & 1;
        const int8_t* rowp = sB + ((size_t)buf * 64 + srow) * MMA_BKP;
        const int4 hdr = *(const int4*)(rowp);
        const int4 qh  = *(const int4*)(rowp + 16 + 16 * (sci & 1));
        const int4 qs  = *(const int4*)(rowp + 48 + 16 * sci);
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
        int8_t* db = sB + ((size_t)buf * 64 + srow) * MMA_BKP + 64 * (sci >> 1) + 16 * (sci & 1);
        *(int4*)(db) = *(const int4*)v0;
        *(int4*)(db + 32) = *(const int4*)v1;
        const float d = __half2float(*(const __half*)&hdr.x);
        const float dmin = __half2float(*(const __half*)((const char*)&hdr.x + 2));
        float ds, dm;
        mma_q5k_scale(hdr.y, hdr.z, hdr.w, 4 * hh + sci, d, dmin, &ds, &dm);
        sBd[buf * 256 + srow * 4 + sci] = ds;
        sBm[buf * 256 + srow * 4 + sci] = dm;
    };

    float acc[2][2][4];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.f;

    stage_async(0, 0);
    mma_cp_wait();
    __syncthreads();
    unpack(0, 0);
    __syncthreads();

    for (int s = 0; s < nstage; ++s) {
        const int buf = s & 1;
        if (s + 1 < nstage) stage_async(s + 1, buf ^ 1);

        const int8_t* Ab = sA + (size_t)buf * 64 * MMA_BKP;
        const int8_t* Bb = sB + (size_t)buf * 64 * MMA_BKP;
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            uint32_t a[2][4], bf[2][2];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int8_t* ap = Ab + (size_t)(wm * 32 + mt * 16 + grp) * MMA_BKP + (b << 5) + tg * 4;
                a[mt][0] = *(const uint32_t*)(ap);
                a[mt][1] = *(const uint32_t*)(ap + 8 * MMA_BKP);
                a[mt][2] = *(const uint32_t*)(ap + 16);
                a[mt][3] = *(const uint32_t*)(ap + 8 * MMA_BKP + 16);
            }
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int8_t* bp = Bb + (size_t)(wn * 16 + nt * 8 + grp) * MMA_BKP + (b << 5) + tg * 4;
                bf[nt][0] = *(const uint32_t*)(bp);
                bf[nt][1] = *(const uint32_t*)(bp + 16);
            }
            const int r0 = wm * 32 + grp, c0 = wn * 16 + tg * 2;
            const float xd0 = sAd[buf * 256 + r0 * 4 + b],         xd1 = sAd[buf * 256 + (r0 + 8) * 4 + b];
            const float xd2 = sAd[buf * 256 + (r0 + 16) * 4 + b],  xd3 = sAd[buf * 256 + (r0 + 24) * 4 + b];
            const int xs0 = *(const int*)&sAx[buf * 256 + r0 * 4 + b];
            const int xs1 = *(const int*)&sAx[buf * 256 + (r0 + 8) * 4 + b];
            const int xs2 = *(const int*)&sAx[buf * 256 + (r0 + 16) * 4 + b];
            const int xs3 = *(const int*)&sAx[buf * 256 + (r0 + 24) * 4 + b];
            const float bd00 = sBd[buf * 256 + c0 * 4 + b], bd01 = sBd[buf * 256 + (c0 + 1) * 4 + b];
            const float bm00 = sBm[buf * 256 + c0 * 4 + b], bm01 = sBm[buf * 256 + (c0 + 1) * 4 + b];
            const float bd10 = sBd[buf * 256 + (c0 + 8) * 4 + b], bd11 = sBd[buf * 256 + (c0 + 9) * 4 + b];
            const float bm10 = sBm[buf * 256 + (c0 + 8) * 4 + b], bm11 = sBm[buf * 256 + (c0 + 9) * 4 + b];
            const float xdv[2][2] = {{xd0, xd1}, {xd2, xd3}};
            const int xsv[2][2] = {{xs0, xs1}, {xs2, xs3}};
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
                        const int rr = i >> 1, cc = i & 1;
                        // same operation order as k_gemm_q5k: xd * (ds*dot - dm*xsum)
                        acc[mt][nt][i] += xdv[mt][rr] * (bdv[nt][cc] * (float)c[i]
                                                       - bmv[nt][cc] * (float)xsv[mt][rr]);
                    }
                }
        }

        if (s + 1 < nstage) {
            mma_cp_wait();
            __syncthreads();
            unpack(s + 1, buf ^ 1);
            __syncthreads();
        }
    }

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
// MMQ-style prefill Q5_K GEMM: 128 weight rows x 128 activation rows per CTA,
// 8 warps in a 4x2 grid (warp tile 32 weight rows x 64 activation rows = 16
// mma per k32 step).  The weight fragment A[2][4] and the activation scales are
// loaded once per k32 and reused by all 16 mma; the activation fragment B is
// streamed.  K is walked in 128-value stages, double buffered with cp.async;
// the 16-byte header (needed for each half's scales) is staged per stage and
// qh once per 256-value superblock, i.e. 192 B per superblock vs 176 B raw.
//
// Up to four independent weight matrices that share the same activation tile
// can be fused into one launch (QKVG, gate+up): grid.y spans the concatenated
// N and a block resolves its part with three compares.
//
// Epilogue: out += (d*sc)*xd*dot + (-dmin*m)*(xd*xsum).  Both scale pairs are
// precombined per (row, 32-group) in shared memory, so each output element
// costs one I2FP, two FMA and one multiply.
// ---------------------------------------------------------------------------
#define MMQ_TN 128
#define MMQ_TM 128
#define MMQ_BKP 144
#define MMQ_SMEM (2 * 128 * MMQ_BKP * 2 + 2 * 128 * 32 + 4 * 2 * 128 * 8 * 2)

__global__ void __launch_bounds__(256, 1)
k_gemm_mmq_q5k(const uint8_t* __restrict__ W0, const uint8_t* __restrict__ W1,
               const uint8_t* __restrict__ W2, const uint8_t* __restrict__ W3,
               float* __restrict__ Y0, float* __restrict__ Y1,
               float* __restrict__ Y2, float* __restrict__ Y3,
               int N0, int N1, int N2, int N3, int nparts,
               const int8_t* __restrict__ xq, const float* __restrict__ xd,
               const int* __restrict__ xsum, int K, int rows) {
    extern __shared__ __align__(16) int8_t mmq_sh[];
    int8_t* sW   = mmq_sh;                          // [2][128][144] raw -> int8
    int8_t* sX   = sW + 2 * 128 * MMQ_BKP;          // [2][128][144] int8 activations
    int8_t* sQh  = sX + 2 * 128 * MMQ_BKP;          // [2][128][32]  q5 high bits
    float2* sWdm = (float2*)(sQh + 2 * 128 * 32);   // [2][128][4]  {d*sc, -dmin*m}
    float2* sXds = sWdm + 2 * 128 * 4;              // [2][128][4]  {xd, xd*xsum}

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int grp = lane >> 2, tg = lane & 3;
    const int wm = warp >> 1, wn = warp & 1;        // 4 weight warps x 2 act warps
    const int nsb = K >> 8, nstage = K >> 7;
    const int row0 = blockIdx.x * MMQ_TM;

    int nb = blockIdx.y * MMQ_TN;
    const uint8_t* Wp = W0; float* Yp = Y0; int Nloc = N0;
    if (nparts > 1 && nb >= Nloc) { nb -= Nloc; Wp = W1; Yp = Y1; Nloc = N1; }
    if (nparts > 2 && nb >= Nloc) { nb -= Nloc; Wp = W2; Yp = Y2; Nloc = N2; }
    if (nparts > 3 && nb >= Nloc) { nb -= Nloc; Wp = W3; Yp = Y3; Nloc = N3; }
    const int nbase = nb;

    const int srow = tid >> 2, sci = tid & 3;

    auto stage_async = [&](int s, int buf) {
        const int hh = s & 1, h2 = s >> 1, gk = s * 128;
        // activations: 128 rows x 128 int8
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = tid + i * 256;
            const int r = c >> 3, o = (c & 7) << 4;
            int8_t* d = sX + ((size_t)buf * 128 + r) * MMQ_BKP + o;
            const int gm = row0 + r;
            if (gm < rows) mma_cp16(d, xq + (size_t)gm * K + gk + o);
            else           *(int4*)d = make_int4(0, 0, 0, 0);
        }
        // activation xd / xsum go to the unused 16-byte row tails
        if (tid < 128) {
            const int gm = row0 + tid;
            int8_t* d = sW + ((size_t)buf * 128 + tid) * MMQ_BKP + 128;
            if (gm < rows) mma_cp16(d, xd + (size_t)gm * (K >> 5) + s * 4);
            else           *(int4*)d = make_int4(0, 0, 0, 0);
        } else {
            const int r = tid - 128, gm = row0 + r;
            int8_t* d = sX + ((size_t)buf * 128 + r) * MMQ_BKP + 128;
            if (gm < rows) mma_cp16(d, xsum + (size_t)gm * (K >> 5) + s * 4);
            else           *(int4*)d = make_int4(0, 0, 0, 0);
        }
        // weight header (16 B) every stage
        if (tid < 128) {
            const uint8_t* wb = Wp + (size_t)(nbase + tid) * (size_t)nsb * 176 + (size_t)h2 * 176;
            mma_cp16(sW + ((size_t)buf * 128 + tid) * MMQ_BKP, wb);
        }
        // q5 high bits: 32 B per row, once per 256-value superblock
        if (hh == 0) {
            const int r = tid >> 1, o = (tid & 1) << 4;
            const uint8_t* wb = Wp + (size_t)(nbase + r) * (size_t)nsb * 176 + (size_t)h2 * 176;
            mma_cp16(sQh + ((size_t)(h2 & 1) * 128 + r) * 32 + o, wb + 16 + o);
        }
        // qs: 64 B per row for this half
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int c = tid + i * 256;
            const int r = c >> 2, k = c & 3;
            const uint8_t* wb = Wp + (size_t)(nbase + r) * (size_t)nsb * 176 + (size_t)h2 * 176;
            mma_cp16(sW + ((size_t)buf * 128 + r) * MMQ_BKP + 16 + (k << 4),
                     wb + 48 + 64 * hh + (k << 4));
        }
        mma_cp_commit();
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
            mma_q5k_scale(hdr.y, hdr.z, hdr.w, 4 * hh + sci, d, dmin, &ds, &dm);
            sWdm[buf * 512 + r * 4 + sci] = make_float2(ds, -dm);
        }
    };

    // raw xd/xsum tails -> {xd, xd*xsum} in sXds (no aliasing: separate rows)
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
    mma_cp_wait();
    __syncthreads();
    unpack(0, 0);
    scale_x(0, 0);
    __syncthreads();

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
            mma_cp_wait();
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
                if (ra < Nloc)     Yp[(size_t)ca * Nloc + ra] = acc[mt][nt][0];
                if (ra + 8 < Nloc) Yp[(size_t)ca * Nloc + ra + 8] = acc[mt][nt][2];
            }
            if (ca + 1 < rows) {
                if (ra < Nloc)     Yp[(size_t)(ca + 1) * Nloc + ra] = acc[mt][nt][1];
                if (ra + 8 < Nloc) Yp[(size_t)(ca + 1) * Nloc + ra + 8] = acc[mt][nt][3];
            }
        }
}

// Pre-combine the q8_0 activation scale and integer sum into the fp16 operand of
// the Q5_K min-term MMA.  One pass per GEMM call; the MMQ blocks then stage it
// with cp.async instead of recomputing xd*xsum on every weight tile.
__global__ void k_fill_xdh(__half* __restrict__ dst, const float* __restrict__ xd,
                           const int* __restrict__ xsum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(xd[i] * (float)xsum[i]);
}

// ---------------------------------------------------------------------------
// Pre-unpacked Q5_K prefill GEMM.  Same 128x128 CTA / 2x8 warp tile as
// k_gemm_mmq_q5k, but the 5-bit unpack and the fp32 I2FP/scale chain are gone:
//   * weights arrive as 144 B = 128 int8 + 4 fp32 (d*sc), so cp.async stages
//     them directly with no in-kernel unpack or qh staging;
//   * the min term is a rank-16 fp16 MMA over (row, 32-group) scales that is
//     accumulated once per 512 K values, outside the k32 dot-product loop.
// The main int8 MMA and its fp32 epilogue are unchanged.
// ---------------------------------------------------------------------------
#define MMQ144_SMEM (2 * 128 * 144 * 2 + 4 * 128 * 24 * 2)

__global__ void __launch_bounds__(256, 1)
k_gemm_mmq_q5k_144(const uint8_t* __restrict__ W0, const uint8_t* __restrict__ W1,
                   const uint8_t* __restrict__ W2, const uint8_t* __restrict__ W3,
                   const __half* __restrict__ D0, const __half* __restrict__ D1,
                   const __half* __restrict__ D2, const __half* __restrict__ D3,
                   float* __restrict__ Y0, float* __restrict__ Y1,
                   float* __restrict__ Y2, float* __restrict__ Y3,
                   int N0, int N1, int N2, int N3, int nparts,
                   const int8_t* __restrict__ xq, const float* __restrict__ xd,
                   const __half* __restrict__ xdh, int K, int rows) {
    extern __shared__ __align__(16) int8_t mmq144_sh[];
    int8_t* sW  = mmq144_sh;                            // [2][128][144]
    int8_t* sX  = sW + 2 * 128 * 144;                   // [2][128][144]
    __half* sWm = (__half*)(sX + 2 * 128 * 144);        // [2][128][24]
    __half* sXm = sWm + 2 * 128 * 24;                   // [2][128][24]

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int grp = lane >> 2, tg = lane & 3;
    const int wm = warp >> 1, wn = warp & 1;
    const int nb = K >> 5;   // stages are addressed linearly in the stage-major layout
    const int nstage = K >> 7;
    const int row0 = blockIdx.x * 128;

    int nbase = blockIdx.y * 128;
    const uint8_t* Wp = W0; const __half* Dp = D0; float* Yp = Y0; int Nloc = N0;
    if (nparts > 1 && nbase >= Nloc) { nbase -= Nloc; Wp = W1; Dp = D1; Yp = Y1; Nloc = N1; }
    if (nparts > 2 && nbase >= Nloc) { nbase -= Nloc; Wp = W2; Dp = D2; Yp = Y2; Nloc = N2; }
    if (nparts > 3 && nbase >= Nloc) { nbase -= Nloc; Wp = W3; Dp = D3; Yp = Y3; Nloc = N3; }

    auto stage_async = [&](int s, int buf) {
        const int gk = s * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = tid + i * 256;
            const int r = c >> 3, o = (c & 7) << 4;
            int8_t* d = sX + ((size_t)buf * 128 + r) * 144 + o;
            const int gm = row0 + r;
            if (gm < rows) mma_cp16(d, xq + (size_t)gm * K + gk + o);
            else           *(int4*)d = make_int4(0, 0, 0, 0);
        }
        // weights: the stage is [s][N][144], so rows nbase..nbase+127 are one
        // contiguous 18,432 B run; a warp copies 512 contiguous bytes.
        {
            const int8_t* wsrc = (const int8_t*)Wp + ((size_t)s * Nloc + nbase) * 144;
            #pragma unroll
            for (int i = 0; i < 5; ++i) {
                const int c = tid + i * 256;
                if (c >= 1152) break;
                mma_cp16(sW + (size_t)buf * 128 * 144 + c * 16, wsrc + c * 16);
            }
        }
        if (tid < 128) {
            const int gm = row0 + tid;
            int8_t* d = sX + ((size_t)buf * 128 + tid) * 144 + 128;
            if (gm < rows) mma_cp16(d, xd + (size_t)gm * nb + s * 4);
            else           *(int4*)d = make_int4(0, 0, 0, 0);
        }
        if ((s & 3) == 0) {
            const int mbuf = (s >> 2) & 1, gbase = s * 4;
            const int r = tid >> 1, h8 = (tid & 1) * 8;
            if (gbase + h8 < nb) {
                mma_cp16_ca(&sWm[(mbuf * 128 + r) * 24 + h8],
                            Dp + (size_t)(nbase + r) * nb + gbase + h8);
            }
            const int gm = row0 + r;
            __half* dst = &sXm[(mbuf * 128 + r) * 24 + h8];
            if (gm < rows && gbase + h8 < nb) {
                mma_cp16_ca(dst, xdh + (size_t)gm * nb + gbase + h8);
            } else {
                *(int4*)dst = make_int4(0, 0, 0, 0);
            }
        }
        mma_cp_commit();
    };

    float acc[2][8][4];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.f;

    stage_async(0, 0);
    mma_cp_wait();
    __syncthreads();

    for (int s = 0; s < nstage; ++s) {
        const int buf = s & 1;
        if (s + 1 < nstage) stage_async(s + 1, buf ^ 1);

        const int8_t* Wb = sW + (size_t)buf * 128 * 144;
        const int8_t* Xb = sX + (size_t)buf * 128 * 144;
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            uint32_t a[2][4], bf[8][2];
            float wds[2][2], xds[8][2];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int8_t* ap = Wb + (size_t)(wm * 32 + mt * 16 + grp) * 144 + (b << 5) + tg * 4;
                a[mt][0] = *(const uint32_t*)(ap);
                a[mt][1] = *(const uint32_t*)(ap + 8 * 144);
                a[mt][2] = *(const uint32_t*)(ap + 16);
                a[mt][3] = *(const uint32_t*)(ap + 8 * 144 + 16);
                const int r0 = wm * 32 + mt * 16 + grp;
                wds[mt][0] = ((const float*)(Wb + r0 * 144 + 128))[b];
                wds[mt][1] = ((const float*)(Wb + (r0 + 8) * 144 + 128))[b];
            }
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const int8_t* bp = Xb + (size_t)(wn * 64 + nt * 8 + grp) * 144 + (b << 5) + tg * 4;
                bf[nt][0] = *(const uint32_t*)(bp);
                bf[nt][1] = *(const uint32_t*)(bp + 16);
                const int j0 = wn * 64 + nt * 8 + 2 * tg;
                xds[nt][0] = ((const float*)(Xb + j0 * 144 + 128))[b];
                xds[nt][1] = ((const float*)(Xb + (j0 + 1) * 144 + 128))[b];
            }
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt)
                #pragma unroll
                for (int nt = 0; nt < 8; ++nt) {
                    int c[4] = {0, 0, 0, 0};
                    mma_s8(c, a[mt], bf[nt]);
                    acc[mt][nt][0] = fmaf(wds[mt][0] * xds[nt][0], (float)c[0], acc[mt][nt][0]);
                    acc[mt][nt][1] = fmaf(wds[mt][0] * xds[nt][1], (float)c[1], acc[mt][nt][1]);
                    acc[mt][nt][2] = fmaf(wds[mt][1] * xds[nt][0], (float)c[2], acc[mt][nt][2]);
                    acc[mt][nt][3] = fmaf(wds[mt][1] * xds[nt][1], (float)c[3], acc[mt][nt][3]);
                }
        }

        if ((s & 3) == 0) {
            const int mbuf = (s >> 2) & 1;
            const __half* Wmb = sWm + mbuf * 128 * 24;
            const __half* Xmb = sXm + mbuf * 128 * 24;
            uint32_t am[2][4], bm[8][2];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int r0 = wm * 32 + mt * 16 + grp;
                am[mt][0] = *(const uint32_t*)(&Wmb[r0 * 24 + tg * 2]);
                am[mt][1] = *(const uint32_t*)(&Wmb[(r0 + 8) * 24 + tg * 2]);
                am[mt][2] = *(const uint32_t*)(&Wmb[r0 * 24 + 8 + tg * 2]);
                am[mt][3] = *(const uint32_t*)(&Wmb[(r0 + 8) * 24 + 8 + tg * 2]);
            }
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const int c0 = wn * 64 + nt * 8 + grp;
                bm[nt][0] = *(const uint32_t*)(&Xmb[c0 * 24 + tg * 2]);
                bm[nt][1] = *(const uint32_t*)(&Xmb[c0 * 24 + 8 + tg * 2]);
            }
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt)
                #pragma unroll
                for (int nt = 0; nt < 8; ++nt)
                    mma_f16_acc(acc[mt][nt], am[mt], bm[nt]);
        }

        if (s + 1 < nstage) {
            mma_cp_wait();
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
                if (ra < Nloc)     Yp[(size_t)ca * Nloc + ra] = acc[mt][nt][0];
                if (ra + 8 < Nloc) Yp[(size_t)ca * Nloc + ra + 8] = acc[mt][nt][2];
            }
            if (ca + 1 < rows) {
                if (ra < Nloc)     Yp[(size_t)(ca + 1) * Nloc + ra] = acc[mt][nt][1];
                if (ra + 8 < Nloc) Yp[(size_t)(ca + 1) * Nloc + ra + 8] = acc[mt][nt][3];
            }
        }
}

//
// q6_K stores a scale per 16 values while mma.m16n8k32 spans 32, so the packed
// layout cannot feed the k=32 dot directly.  At load time each q6_K tensor is
// therefore re-encoded losslessly into
//     per 128 values: 128 B int8 (value = stored 6-bit code - 32) + 32 B fp32 scales
// i.e. 160 B per (row, 128-value stage).  The values are exactly the stored codes
// (no re-quantisation) and the scales are the exact fp32 d*sc products, so the
// only rounding left is the fp32 scale application in the epilogue.
//
// The 210-byte q6_K blocks are never 16-byte aligned, which is why the packed
// form cannot be staged with cp.async at all; 160 bytes per stage is aligned.
//
// mma granularity: each 32-value k block carries two 16-value scale groups, so
// two mma are issued per k block with the other half of the A fragment zeroed.
// The int32 dot of each 16-value group is exact; both halves share the B fragment.

__device__ __forceinline__ void q6k_stage(float* sBd, int8_t* sB, const uint8_t* base, int buf, int r, int c) {
    // c in 0..9: first 8 pieces feed the int8 tile, the last two the fp32 scales
    if (c < 8) mma_cp16(sB + ((size_t)buf * 64 + r) * Q6K_SROW + (c << 4), base + (c << 4));
    else       mma_cp16((int8_t*)(sBd + ((size_t)buf * 64 + r) * 8) + ((c - 8) << 4), base + 128 + ((c - 8) << 4));
}

__global__ void __launch_bounds__(256, 2)
k_gemm_mma_q6k(float* __restrict__ y, const uint8_t* __restrict__ W,
               const int8_t* __restrict__ xq, const float* __restrict__ xd,
               int K, int N, int rows) {
    extern __shared__ __align__(16) int8_t q6b_sh[];
    int8_t* sA  = q6b_sh;                             // [2][64][Q6K_SROW] activations
    int8_t* sB  = sA + 2 * 64 * Q6K_SROW;             // [2][64][Q6K_SROW] int8 weights (128 B) + pad
    float*  sAd = (float*)(sB + 2 * 64 * Q6K_SROW);   // [2][64][4] activation d
    float*  sBd = sAd + 2 * 256;                      // [2][64][8] weight d*scale per 16 values

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int grp = lane >> 2, tg = lane & 3;
    const int wm = warp >> 2, wn = warp & 3;
    const int nstage = K / 128;
    const int nbase = blockIdx.x * 64;
    const int row0 = blockIdx.y * 64;

    auto stage_async = [&](int s, int buf) {
        const int gk = s * 128;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int c = tid + i * 256;
            const int r = c >> 3, o = (c & 7) << 4;
            int8_t* d = sA + ((size_t)buf * 64 + r) * Q6K_SROW + o;
            const int gm = row0 + r;
            if (gm < rows) mma_cp16(d, xq + (size_t)gm * K + gk + o);
            else           *(int4*)d = make_int4(0, 0, 0, 0);
        }
        // 10 pieces per weight row: 128 B values + 32 B scales
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            const int c = tid + i * 256;
            if (c < 640) {
                const int r = c / 10, k = c - r * 10;   // r < 64
                const uint8_t* base = W + ((size_t)(nbase + r) * nstage + s) * Q6K_SROW;
                q6k_stage(sBd, sB, base, buf, r, k);
            }
        }
        if (tid < 64) {
            const int gm = row0 + tid;
            float* dd = sAd + (size_t)buf * 256 + tid * 4;
            if (gm < rows) mma_cp16(dd, xd + (size_t)gm * (K >> 5) + s * 4);
            else           *(int4*)dd = make_int4(0, 0, 0, 0);
        }
        mma_cp_commit();
    };

    float acc[2][2][4];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.f;

    stage_async(0, 0);
    mma_cp_wait();
    __syncthreads();

    for (int s = 0; s < nstage; ++s) {
        const int buf = s & 1;
        if (s + 1 < nstage) stage_async(s + 1, buf ^ 1);

        const int8_t* Ab = sA + (size_t)buf * 64 * Q6K_SROW;
        const int8_t* Bb = sB + (size_t)buf * 64 * Q6K_SROW;
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            uint32_t a[2][4], bf[2][2];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int8_t* ap = Ab + (size_t)(wm * 32 + mt * 16 + grp) * Q6K_SROW + (b << 5) + tg * 4;
                a[mt][0] = *(const uint32_t*)(ap);
                a[mt][1] = *(const uint32_t*)(ap + 8 * Q6K_SROW);
                a[mt][2] = *(const uint32_t*)(ap + 16);
                a[mt][3] = *(const uint32_t*)(ap + 8 * Q6K_SROW + 16);
            }
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int8_t* bp = Bb + (size_t)(wn * 16 + nt * 8 + grp) * Q6K_SROW + (b << 5) + tg * 4;
                bf[nt][0] = *(const uint32_t*)(bp);
                bf[nt][1] = *(const uint32_t*)(bp + 16);
            }
            const int r0 = wm * 32 + grp, c0 = wn * 16 + tg * 2;
            const float xd0 = sAd[buf * 256 + r0 * 4 + b],        xd1 = sAd[buf * 256 + (r0 + 8) * 4 + b];
            const float xd2 = sAd[buf * 256 + (r0 + 16) * 4 + b], xd3 = sAd[buf * 256 + (r0 + 24) * 4 + b];
            const float xdv[2][2] = {{xd0, xd1}, {xd2, xd3}};
            const float* bdl = sBd + (size_t)buf * 512;
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int cA = c0 + nt * 8;
                const float slv[2] = {bdl[cA * 8 + 2 * b],       bdl[(cA + 1) * 8 + 2 * b]};
                const float shv[2] = {bdl[cA * 8 + 2 * b + 1],   bdl[(cA + 1) * 8 + 2 * b + 1]};
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    int cl[4] = {0, 0, 0, 0}, ch[4] = {0, 0, 0, 0};
                    mma_s8_k16(cl, a[mt][0], a[mt][1], bf[nt][0]);
                    mma_s8_k16(ch, a[mt][2], a[mt][3], bf[nt][1]);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int rr = i >> 1, cc = i & 1;
                        acc[mt][nt][i] += xdv[mt][rr] * (slv[cc] * (float)cl[i] + shv[cc] * (float)ch[i]);
                    }
                }
            }
        }

        if (s + 1 < nstage) {
            mma_cp_wait();
            __syncthreads();
        }
    }

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

#define Q6K_SMEM (2 * (2 * 64 * Q6K_SROW) + 2 * 256 * 4 + 2 * 512 * 4)
#define Q6K128_SROW 160
#define Q6K128_SMEM (2 * 128 * Q6K128_SROW * 2)

// ---------------------------------------------------------------------------
// q6_K prefill GEMM on the same 128 x 128 CTA tile as the Q5_K MMQ kernel:
// 8 warps in a 4x2 grid, warp tile 32 activation rows x 64 weight columns,
// 16 mma.m16n8k16.s8 per 16-value scale group.  The 160-byte repacked stage
// (128 int8 values + 8 fp32 d*sc) is copied verbatim, so the weight scales and
// values need no in-kernel unpack.  Q6_K cannot use k=32 MMA because its scale
// group is 16 values wide; k=16 also runs in half the cycles on SM86.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 1)
k_gemm_mma_q6k_128(float* __restrict__ y, const uint8_t* __restrict__ W,
                   const int8_t* __restrict__ xq, const float* __restrict__ xd,
                   int K, int N, int rows) {
    extern __shared__ __align__(16) int8_t q6k128_sh[];
    int8_t* sA = q6k128_sh;                             // [2][128][160] activations
    int8_t* sB = sA + 2 * 128 * Q6K128_SROW;            // [2][128][160] weights

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int grp = lane >> 2, tg = lane & 3;
    const int wm = warp >> 1, wn = warp & 1;
    const int nstage = K >> 7, nb = K >> 5;
    const int row0 = blockIdx.y * 128;
    const int nbase = blockIdx.x * 128;

    auto stage_async = [&](int s, int buf) {
        const int gk = s * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = tid + i * 256;
            const int r = c >> 3, o = (c & 7) << 4;
            int8_t* d = sA + ((size_t)buf * 128 + r) * Q6K128_SROW + o;
            const int gm = row0 + r;
            if (gm < rows) mma_cp16(d, xq + (size_t)gm * K + gk + o);
            else           *(int4*)d = make_int4(0, 0, 0, 0);
        }
        // 10 x 16 B pieces per weight row: 8 of values + 2 of fp32 scales
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            const int c = tid + i * 256;
            if (c < 1280) {
                const int r = c / 10, k = c - r * 10;
                const int8_t* base = (const int8_t*)W + ((size_t)(nbase + r) * nstage + s) * Q6K128_SROW;
                int8_t* d = sB + ((size_t)buf * 128 + r) * Q6K128_SROW + (k < 8 ? (k << 4) : (128 + ((k - 8) << 4)));
                mma_cp16(d, base + (k < 8 ? (k << 4) : (128 + ((k - 8) << 4))));
            }
        }
        if (tid < 128) {
            const int gm = row0 + tid;
            int8_t* d = sA + ((size_t)buf * 128 + tid) * Q6K128_SROW + 128;
            if (gm < rows) mma_cp16(d, xd + (size_t)gm * nb + s * 4);
            else           *(int4*)d = make_int4(0, 0, 0, 0);
        }
        mma_cp_commit();
    };

    float acc[2][8][4];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.f;

    stage_async(0, 0);
    mma_cp_wait();
    __syncthreads();

    for (int s = 0; s < nstage; ++s) {
        const int buf = s & 1;
        if (s + 1 < nstage) stage_async(s + 1, buf ^ 1);
        const int8_t* Ab = sA + (size_t)buf * 128 * Q6K128_SROW;
        const int8_t* Bb = sB + (size_t)buf * 128 * Q6K128_SROW;
        #pragma unroll
        for (int kg = 0; kg < 8; ++kg) {
            uint32_t a[2][2], bf[8];
            float xds[2][2], wds[8][2];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int r0 = wm * 32 + mt * 16 + grp;
                const int8_t* ap = Ab + (size_t)r0 * Q6K128_SROW + kg * 16 + tg * 4;
                a[mt][0] = *(const uint32_t*)(ap);
                a[mt][1] = *(const uint32_t*)(ap + 8 * Q6K128_SROW);
                xds[mt][0] = ((const float*)(Ab + (size_t)r0 * Q6K128_SROW + 128))[kg >> 1];
                xds[mt][1] = ((const float*)(Ab + (size_t)(r0 + 8) * Q6K128_SROW + 128))[kg >> 1];
            }
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const int c0 = wn * 64 + nt * 8 + 2 * tg;
                const int8_t* bp = Bb + (size_t)(wn * 64 + nt * 8 + grp) * Q6K128_SROW + kg * 16 + tg * 4;
                bf[nt] = *(const uint32_t*)(bp);
                wds[nt][0] = ((const float*)(Bb + (size_t)c0 * Q6K128_SROW + 128))[kg];
                wds[nt][1] = ((const float*)(Bb + (size_t)(c0 + 1) * Q6K128_SROW + 128))[kg];
            }
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt)
                #pragma unroll
                for (int nt = 0; nt < 8; ++nt) {
                    int c[4] = {0, 0, 0, 0};
                    mma_s8_k16(c, a[mt][0], a[mt][1], bf[nt]);
                    const float x0 = xds[mt][0], x1 = xds[mt][1];
                    acc[mt][nt][0] = fmaf(x0 * wds[nt][0], (float)c[0], acc[mt][nt][0]);
                    acc[mt][nt][1] = fmaf(x0 * wds[nt][1], (float)c[1], acc[mt][nt][1]);
                    acc[mt][nt][2] = fmaf(x1 * wds[nt][0], (float)c[2], acc[mt][nt][2]);
                    acc[mt][nt][3] = fmaf(x1 * wds[nt][1], (float)c[3], acc[mt][nt][3]);
                }
        }
        if (s + 1 < nstage) {
            mma_cp_wait();
            __syncthreads();
        }
    }

    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int ra = row0 + wm * 32 + mt * 16 + grp;
            const int cb = nbase + wn * 64 + nt * 8 + 2 * tg;
            if (ra < rows)     *(float2*)(y + (size_t)ra * N + cb) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
            if (ra + 8 < rows) *(float2*)(y + (size_t)(ra + 8) * N + cb) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
        }
}


// shared bytes for k_gemm_mma_q5k
#define MMA_SMEM (2 * (2 * 64 * MMA_BKP) + 4 * 2 * 256 * 4)

static void launch_gemm_b_impl(float* y, const TensorW& w, int rows, int rowoff, cudaStream_t st) {
    const int K = w.K, N = w.N;
    const char* ewa = getenv("BRUCE_WA");     // dev sweep overrides (tile-shape tuning)
    const char* er  = getenv("BRUCE_R");
    const int WA = ewa ? atoi(ewa) : 16;     // warps per block for the batched kernels
    // weight rows unrolled per warp, per verified-row count (swept: see tools/README notes)
    const int Rdef = (rows == 2 || rows == 4 || rows == 6) ? 4 : ((rows == 1) ? 8 : 2);
    const int R  = er ? atoi(er) : Rdef;
    const int blocks = gemm_grid2(N, WA, R);
    const int nthr = 32 * WA;
    if (w.type == GGML_TYPE_Q5_K) {
        const size_t sm = (size_t)(K >> 8) * 72 * rows * 4;
        #define Q5(MR, WA_, RR) k_gemm_q5k<MR, WA_, RR><<<blocks, nthr, sm, st>>>(y, (const uint8_t*)w.d, \
        M.bm_xq + (size_t)rowoff * K, M.bm_xd + (size_t)rowoff * (K >> 5), M.bm_xsum + (size_t)rowoff * (K >> 5), K, N)
        #define Q5S(MR) switch (WA * 16 + R) { \
            case  8 * 16 + 2: Q5(MR, 8, 2); break;  case  8 * 16 + 4: Q5(MR, 8, 4); break; \
            case  8 * 16 + 8: Q5(MR, 8, 8); break;  case 16 * 16 + 2: Q5(MR, 16, 2); break; \
            case 16 * 16 + 4: Q5(MR, 16, 4); break; default:          Q5(MR, 16, 8); break; }
        if (ewa || er) {
            switch (rows) {
                case 1: Q5S(1); break;  case 2: Q5S(2); break;
                case 3: Q5S(3); break;  case 4: Q5S(4); break;
                case 5: Q5S(5); break;  case 6: Q5S(6); break;
                case 7: Q5S(7); break;  case 8: Q5S(8); break;
                default: fprintf(stderr, "bad gemm rows %d\n", rows); exit(1);
            }
        } else switch (rows) {
            case 1: Q5(1, 16, 8); break;
            case 2: Q5(2, 16, 4); break;
            case 3: Q5(3, 16, 2); break;
            case 4: Q5(4, 16, 4); break;
            case 5: Q5(5, 16, 2); break;
            case 6: Q5(6, 16, 4); break;
            case 7: Q5(7, 16, 2); break;
            case 8: Q5(8, 16, 2); break;
            default: fprintf(stderr, "bad gemm rows %d\n", rows); exit(1);
        }
        #undef Q5S
        #undef Q5
    } else if (w.type == GGML_TYPE_Q6_K) {
        const size_t sm = (size_t)K * rows;
        #define Q6(MR, RR) k_gemm_q6k<MR, 16, RR><<<blocks, nthr, sm, st>>>(y, (const uint8_t*)w.d, \
        M.bm_xq + (size_t)rowoff * K, M.bm_xd + (size_t)rowoff * (K >> 5), K, N)
        switch (rows) {
            case 1: Q6(1, 8); break;
            case 2: Q6(2, 8); break;
            case 3: Q6(3, 8); break;
            case 4: Q6(4, 8); break;
            case 5: Q6(5, 2); break;
            case 6: Q6(6, 2); break;
            case 7: Q6(7, 2); break;
            case 8: Q6(8, 2); break;
            case 16: Q6(16, 2); break;
            case 32: Q6(32, 1); break;
            default: fprintf(stderr, "bad gemm rows %d\n", rows); exit(1);
        }
        #undef Q6
    } else {
        fprintf(stderr, "unsupported gemm type %u\n", w.type);
        exit(1);
    }
    {
        cudaError_t e = cudaGetLastError();
        if (e) { fprintf(stderr, "gemm launch failed rows=%d K=%d N=%d blocks=%d: %s\n",
                             rows, K, N, blocks, cudaGetErrorString(e)); exit(1); }
    }
    (void)R;
}

static void launch_gemm_b_impl(float* y, const TensorW& w, int rows, int rowoff, cudaStream_t st);
static void launch_gemm_b(float* y, const TensorW& w, int rows, cudaStream_t st);

// Barrier-free tensor-core verify GEMM over up to four concatenated Q5_K
// matrices that share the activation rows.  Returns false when the shapes or
// types do not fit, so callers can fall back to per-tensor launches.
static bool launch_verify_fused(float* const* ys, const TensorW* const* ws, int nparts, int rows,
                                const int8_t* xq, const float* xd, const int* xsum, int xoff,
                                cudaStream_t st) {
    if (rows < 1 || rows > 8 || nparts < 1 || nparts > 4) return false;
    const int K = ws[0]->K;
    if ((K % 256) != 0) return false;
    int ntot = 0;
    const uint8_t* W[4] = {nullptr, nullptr, nullptr, nullptr};
    float* Y[4] = {nullptr, nullptr, nullptr, nullptr};
    int N[4] = {0, 0, 0, 0};
    for (int p = 0; p < nparts; ++p) {
        if (!ws[p]->d || !ys[p] || ws[p]->type != GGML_TYPE_Q5_K) return false;
        if (ws[p]->K != K || (ws[p]->N % 16) != 0) return false;
        W[p] = (const uint8_t*)ws[p]->d;
        Y[p] = ys[p];
        N[p] = ws[p]->N;
        ntot += ws[p]->N;
    }
    const int nb32 = K >> 5;
    static int tc1 = -1;
    if (tc1 < 0) { const char* e = getenv("BRUCE_TC1"); tc1 = (e && atoi(e)) ? 1 : 0; }
    if (rows == 1 && !tc1) {
        const size_t sm = (size_t)K;
        k_gemv_q5k_fused<8><<<gemv_grid(ntot, K), 256, sm, st>>>(W[0], W[1], W[2], W[3],
            Y[0], Y[1], Y[2], Y[3], N[0], N[1], N[2], N[3], nparts, ntot, xq, xd, xsum, K);
    } else {
        static int vstage = -1;
        if (vstage < 0) {
            const char* e = getenv("BRUCE_VERIFY_STAGE");
            vstage = (e && atoi(e) == 0) ? 0 : 1;
        }
        // Two 16x176 B tiles per warp cost 45 KB; that fits beside the K=2560
        // activation stage but not beside the K=7168 one, where the old
        // register-direct loop stays.
        const size_t sm_base = (size_t)nb32 * 320;
        const size_t stage_bytes = (size_t)8 * 2 * 16 * 176;
        const bool use_stage = vstage && sm_base + stage_bytes <= 96 * 1024;
        const size_t sm = sm_base + (use_stage ? stage_bytes : 0);
        static bool vtc_attr = false;
        if (!vtc_attr) {
            vtc_attr = true;
            #define SETV(M) CK(cudaFuncSetAttribute((const void*)&k_gemm_tc_verify_q5k<M,8,true>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98 * 1024)); \
                CK(cudaFuncSetAttribute((const void*)&k_gemm_tc_verify_q5k<M,8,false>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98 * 1024)); \
                CK(cudaFuncSetAttribute((const void*)&k_gemm_tc_verify_q5k<M,8,true,true>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98 * 1024)); \
                CK(cudaFuncSetAttribute((const void*)&k_gemm_tc_verify_q5k<M,8,false,true>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98 * 1024))
            SETV(1); SETV(2); SETV(3); SETV(4); SETV(5); SETV(6); SETV(7); SETV(8);
            #undef SETV
        }
        static int vpf = -1;
        if (vpf < 0) { const char* e = getenv("BRUCE_VERIFYPF"); vpf = (e && atoi(e) == 0) ? 0 : 1; }
        const int ntiles = ntot / 16;
        // Measured (tmp/vb3.cu, M=8 on this card): one CTA per SM streams the
        // strided 16-row weight tile fastest; 2 CTAs/SM resort to ~15% lower
        // bandwidth, and small-N matrices lose more to wave quantization than they
        // gain from more blocks.  Large N: 28 blocks.  Small N (<= 224 tiles):
        // one warp per tile, at most 20 blocks.
        int blocks = (ntiles >= 224) ? 28 : std::max(1, std::min(20, ntiles));
        #define VTC(MR, PF, ST) k_gemm_tc_verify_q5k<MR, 8, PF, ST><<<blocks, 256, sm, st>>>( \
            W[0], W[1], W[2], W[3], Y[0], Y[1], Y[2], Y[3], \
            N[0], N[1], N[2], N[3], nparts, ntot, xq, xd, xsum, K, xoff)
        #define VTCS(MR) do { \
            if (use_stage) { if (vpf) VTC(MR, true, true); else VTC(MR, false, true); } \
            else           { if (vpf) VTC(MR, true, false); else VTC(MR, false, false); } \
        } while (0)
        switch (rows) {
            case 1: VTCS(1); break;
            case 2: VTCS(2); break;
            case 3: VTCS(3); break;
            case 4: VTCS(4); break;
            case 5: VTCS(5); break;
            case 6: VTCS(6); break;
            case 7: VTCS(7); break;
            default: VTCS(8); break;
        }
        #undef VTCS
        #undef VTC
    }
    cudaError_t e = cudaGetLastError();
    if (e) { fprintf(stderr, "verify gemm launch failed rows=%d K=%d N=%d: %s\n",
                     rows, K, ntot, cudaGetErrorString(e)); exit(1); }
    return true;
}

// Q6_K verify path on the aligned 160 B repack (BRUCE_Q6TC=0 keeps the old raw
// 210 B / dp4a choice).  Only rows 2..8 ever enter here.
static bool launch_q6_verify_fused(float* const* ys, const TensorW* const* ws, int nparts, int rows,
                                   const int8_t* xq, const float* xd, int xoff, cudaStream_t st) {
    if (rows < 2 || rows > 8 || nparts < 1 || nparts > 4) return false;
    const int K = ws[0]->K;
    if ((K % 128) != 0) return false;
    int ntot = 0;
    const uint8_t* W[4] = {nullptr, nullptr, nullptr, nullptr};
    float* Y[4] = {nullptr, nullptr, nullptr, nullptr};
    int N[4] = {0, 0, 0, 0};
    for (int p = 0; p < nparts; ++p) {
        if (!ws[p]->dp || !ys[p] || ws[p]->type != GGML_TYPE_Q6_K) return false;
        if (ws[p]->K != K || (ws[p]->N % 16) != 0) return false;
        W[p] = (const uint8_t*)ws[p]->dp;
        Y[p] = ys[p];
        N[p] = ws[p]->N;
        ntot += ws[p]->N;
    }
    const size_t sm = (size_t)(K >> 5) * 288;
    static bool attr = false;
    if (!attr) {
        attr = true;
        #define SETV6(M) CK(cudaFuncSetAttribute((const void*)&k_gemm_tc_verify_q6k<M,8,true>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, 98 * 1024)); \
            CK(cudaFuncSetAttribute((const void*)&k_gemm_tc_verify_q6k<M,8,false>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, 98 * 1024))
        SETV6(2); SETV6(3); SETV6(4); SETV6(5); SETV6(6); SETV6(7); SETV6(8);
        #undef SETV6
    }
    static int vpf = -1;
    if (vpf < 0) { const char* e = getenv("BRUCE_VERIFYPF"); vpf = (e && atoi(e) == 0) ? 0 : 1; }
    const int ntiles = ntot / 16;
    const int blocks = (ntiles >= 224) ? 28 : std::max(1, std::min(20, ntiles));
    #define V6(MR, PF) k_gemm_tc_verify_q6k<MR, 8, PF><<<blocks, 256, sm, st>>>(W[0], W[1], W[2], W[3], \
        Y[0], Y[1], Y[2], Y[3], N[0], N[1], N[2], N[3], nparts, ntot, xq, xd, K, xoff)
    #define V6S(MR) do { if (vpf) V6(MR, true); else V6(MR, false); } while (0)
    switch (rows) {
        case 2: V6S(2); break;
        case 3: V6S(3); break;
        case 4: V6S(4); break;
        case 5: V6S(5); break;
        case 6: V6S(6); break;
        case 7: V6S(7); break;
        default: V6S(8); break;
    }
    #undef V6S
    #undef V6
    cudaError_t e = cudaGetLastError();
    if (e) { fprintf(stderr, "q6 verify gemm launch failed rows=%d K=%d N=%d: %s\n",
                     rows, K, ntot, cudaGetErrorString(e)); exit(1); }
    return true;
}

static void launch_gemm_q6k_dp(float* y, const TensorW& w, int rows,
                                const int8_t* xq, const float* xd, int xoff, cudaStream_t st) {
    const int K = w.K, N = w.N;
    const size_t sm = (size_t)K * rows;
    const int blocks = rows == 1 ? gemv_grid(N, K) : 56;
    static bool fix_attr = false;
    if (!fix_attr) {
        fix_attr = true;
        #define SETF(M,R) CK(cudaFuncSetAttribute((const void*)&k_gemv_q6k_dp_fix<M,8,10,R>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, 98 * 1024)); \
            CK(cudaFuncSetAttribute((const void*)&k_gemv_q6k_dp_fix<M,8,28,R>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, 98 * 1024))
        SETF(1,1); SETF(1,2); SETF(1,4); SETF(1,8);
        SETF(2,1); SETF(2,2); SETF(2,4); SETF(2,8);
        SETF(3,1); SETF(3,2); SETF(3,4); SETF(3,8);
        SETF(4,1); SETF(4,2); SETF(4,4); SETF(4,8);
        SETF(5,1); SETF(5,2); SETF(5,4); SETF(5,8);
        SETF(6,1); SETF(6,2); SETF(6,4); SETF(6,8);
        SETF(7,1); SETF(7,2); SETF(7,4); SETF(7,8);
        SETF(8,1); SETF(8,2); SETF(8,4); SETF(8,8);
        #undef SETF
    }
    if (K == 10 * 256 || K == 28 * 256) {
        // decode: one row per warp, R rows unrolled per warp.  R is chosen so the
        // warp-task count divides evenly into full blocks (no tail) and there are
        // enough independent 16-byte loads in flight to cover DRAM latency.
        static int q6r = -1, q6b = -1;
        if (q6r < 0) {
            const char* e = getenv("BRUCE_Q6R"); q6r = e ? std::max(1, std::min(8, atoi(e))) : 0;
            const char* b = getenv("BRUCE_Q6B"); q6b = b ? std::max(1, atoi(b)) : 0;
        }
        int R = q6r;
        if (!R) R = 2;
        int bl = q6b;
        if (!bl) bl = rows > 1 ? 56 : gemv_grid(N, K);
        #define Q6F(MR, NSB, RR) k_gemv_q6k_dp_fix<MR,8,NSB,RR><<<bl, 256, sm, st>>>(y, \
            (const uint8_t*)w.dp, xq, xd, N, xoff)
        #define Q6FR(MR, NSB) switch (R) { \
            case 1: Q6F(MR, NSB, 1); break;  case 2: Q6F(MR, NSB, 2); break; \
            case 4: Q6F(MR, NSB, 4); break;  default: Q6F(MR, NSB, 8); break; }
        #define Q6FS(MR) do { \
            if (K == 2560) { Q6FR(MR, 10) } else { Q6FR(MR, 28) } \
        } while (0)
        switch (rows) {
            case 1: Q6FS(1); break;
            case 2: Q6FS(2); break;
            case 3: Q6FS(3); break;
            case 4: Q6FS(4); break;
            case 5: Q6FS(5); break;
            case 6: Q6FS(6); break;
            case 7: Q6FS(7); break;
            case 8: Q6FS(8); break;
            default: fprintf(stderr, "bad q6k dp rows %d\n", rows); exit(1);
        }
        #undef Q6FS
        #undef Q6FR
        #undef Q6F
        cudaError_t e = cudaGetLastError();
        if (e) { fprintf(stderr, "q6k dp fix launch failed rows=%d K=%d N=%d: %s\n",
                         rows, K, N, cudaGetErrorString(e)); exit(1); }
        return;
    }
    #define Q6D(MR) k_gemv_q6k_dp<MR, 8><<<blocks, 256, sm, st>>>(y, (const uint8_t*)w.dp, \
        xq, xd, K, N, xoff)
    switch (rows) {
        case 1: Q6D(1); break;
        case 2: Q6D(2); break;
        case 3: Q6D(3); break;
        case 4: Q6D(4); break;
        case 5: Q6D(5); break;
        case 6: Q6D(6); break;
        case 7: Q6D(7); break;
        case 8: Q6D(8); break;
        default: fprintf(stderr, "bad q6k dp rows %d\n", rows); exit(1);
    }
    #undef Q6D
    cudaError_t e = cudaGetLastError();
    if (e) { fprintf(stderr, "q6k dp launch failed rows=%d K=%d N=%d: %s\n",
                     rows, K, N, cudaGetErrorString(e)); exit(1); }
}

// Launch the 128x128 MMQ kernel over `nparts` horizontally concatenated Q5_K
// matrices.  Returns false (without touching the stream) when the shapes or
// types do not fit, so callers can fall back to per-tensor launches.
static bool try_launch_mmq(float* const* ys, const TensorW* const* ws, int nparts, int rows, cudaStream_t st) {
    static int enabled = -1;
    if (enabled < 0) { const char* e = getenv("BRUCE_MMQ"); enabled = (!e || atoi(e)) ? 1 : 0; }
    if (!enabled || nparts < 1 || nparts > 4 || rows < 96) return false;
    const int K = ws[0]->K;
    int ntot = 0;
    const uint8_t* W[4] = {nullptr, nullptr, nullptr, nullptr};
    const __half* D[4] = {nullptr, nullptr, nullptr, nullptr};
    float* Y[4] = {nullptr, nullptr, nullptr, nullptr};
    int N[4] = {0, 0, 0, 0};
    bool prepacked = true;
    for (int p = 0; p < nparts; ++p) {
        if (!ws[p]->d || !ys[p] || ws[p]->type != GGML_TYPE_Q5_K) return false;
        if (ws[p]->K != K || (ws[p]->N % 128) != 0) return false;
        W[p] = (const uint8_t*)ws[p]->d;
        D[p] = (const __half*)ws[p]->dm;
        Y[p] = ys[p];
        N[p] = ws[p]->N;
        ntot += ws[p]->N;
        if (!ws[p]->dp || !ws[p]->dm || (K % 512) != 0) prepacked = false;
    }
    if ((K % 256) != 0 || (ntot % 128) != 0) return false;
    static bool attr = false;
    if (!attr) {
        attr = true;
        CK(cudaFuncSetAttribute((const void*)&k_gemm_mmq_q5k,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, MMQ_SMEM));
        CK(cudaFuncSetAttribute((const void*)&k_gemm_mmq_q5k_144,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, MMQ144_SMEM));
    }
    dim3 g((rows + 127) / 128, ntot / 128);
    if (prepacked) {
        const int nx = rows * (K >> 5);
        k_fill_xdh<<<(nx + 255) / 256, 256, 0, st>>>(M.bm_xdh, M.bm_xd, M.bm_xsum, nx);
        k_gemm_mmq_q5k_144<<<g, 256, MMQ144_SMEM, st>>>((const uint8_t*)ws[0]->dp,
            nparts > 1 ? (const uint8_t*)ws[1]->dp : nullptr,
            nparts > 2 ? (const uint8_t*)ws[2]->dp : nullptr,
            nparts > 3 ? (const uint8_t*)ws[3]->dp : nullptr,
            D[0], D[1], D[2], D[3], Y[0], Y[1], Y[2], Y[3],
            N[0], N[1], N[2], N[3], nparts,
            M.bm_xq, M.bm_xd, M.bm_xdh, K, rows);
    } else {
        k_gemm_mmq_q5k<<<g, 256, MMQ_SMEM, st>>>(W[0], W[1], W[2], W[3], Y[0], Y[1], Y[2], Y[3],
                                                  N[0], N[1], N[2], N[3], nparts,
                                                  M.bm_xq, M.bm_xd, M.bm_xsum, K, rows);
    }
    cudaError_t e = cudaGetLastError();
    if (e) { fprintf(stderr, "mmq q5k launch failed rows=%d K=%d N=%d: %s\n",
                     rows, K, ntot, cudaGetErrorString(e)); exit(1); }
    return true;
}

// Fused prompt GEMM: one MMQ launch for the concatenated outputs when possible,
// otherwise the ordinary per-tensor path.
static void launch_gemm_b_fused(float* const* ys, const TensorW* const* ws, int nparts, int rows, cudaStream_t st) {
    if (rows > 8 && try_launch_mmq(ys, ws, nparts, rows, st)) return;
    for (int p = 0; p < nparts; ++p) launch_gemm_b(ys[p], *ws[p], rows, st);
}

// prompt-batch entry: one tensor-core pass for the whole chunk, else 8-row sub-chunks
static void launch_gemm_b(float* y, const TensorW& w, int rows, cudaStream_t st) {
    if (rows <= 8) {
        if (w.type == GGML_TYPE_Q6_K && w.dp) {
            // dev A/B: the raw 210 B layout via the batched GEMM (34% fewer bytes,
            // unaligned narrow loads) vs the aligned 160 B dp4a decode kernel.
            // Measured: the raw 210 B layout wins as the row count grows (it moves
            // 0.84 GB/token instead of 1.28 GB) -- tie at 2 rows, -0.7% at 4,
            // -1.2% at 6, -10% at 8.  BRUCE_Q6GEMM=0 forces the dp kernel.
            static int q6gemm = -1;
            if (q6gemm < 0) { const char* e = getenv("BRUCE_Q6GEMM"); q6gemm = e ? (atoi(e) ? 1 : 0) : 1; }
            // rows >= 6 now use the tensor-core kernel on the aligned repack
            // (m16n8k16 per 16-value scale group); at 3..5 rows the raw layout's
            // 34% fewer bytes still wins (measured: 4K row4 17.33 vs 17.48,
            // row8 18.42 vs 17.62).  BRUCE_Q6TC=0 keeps the old choice.
            static int q6tc = -1;
            if (q6tc < 0) { const char* e = getenv("BRUCE_Q6TC"); q6tc = (!e || atoi(e)) ? 1 : 0; }
            if (rows >= 6 && q6gemm && q6tc) {
                float* ys[1] = {y};
                const TensorW* ws[1] = {&w};
                if (launch_q6_verify_fused(ys, ws, 1, rows, M.bm_xq, M.bm_xd, 0, st)) return;
            }
            if (rows < 3 || !q6gemm) {
                launch_gemm_q6k_dp(y, w, rows, M.bm_xq, M.bm_xd, 0, st);
                return;
            }
        }
        if (w.type == GGML_TYPE_Q5_K) {
            float* ys[1] = {y};
            const TensorW* ws[1] = {&w};
            if (launch_verify_fused(ys, ws, 1, rows, M.bm_xq, M.bm_xd, M.bm_xsum, 0, st)) return;
        }
    }
    if (rows > 8) {
        const int K = w.K, N = w.N;
        static int nimm = -1, nsel = -1, ksel = -1;
        if (nimm < 0) {
            const char* e = getenv("BRUCE_MMA");   nimm = (!e || atoi(e)) ? 1 : 0;
            const char* en = getenv("BRUCE_MMA_N"); nsel = en ? atoi(en) : 0;
            const char* ek = getenv("BRUCE_MMA_K"); ksel = ek ? atoi(ek) : 0;
        }
        const bool mma_ok = nimm && (!nsel || nsel == N) && (!ksel || ksel == K);
        if (mma_ok && w.type == GGML_TYPE_Q5_K && (K % 256) == 0 && (N % 128) == 0) {
            float* ys[1] = {y};
            const TensorW* ws[1] = {&w};
            if (try_launch_mmq(ys, ws, 1, rows, st)) return;
        }
        if (mma_ok && w.type == GGML_TYPE_Q6_K && w.dp && (K % 256) == 0 && (N % 128) == 0) {
            static int q6_128 = -1;
            if (q6_128 < 0) { const char* e = getenv("BRUCE_Q6_128"); q6_128 = (!e || atoi(e)) ? 1 : 0; }
            if (q6_128) {
                static bool q6_attr = false;
                if (!q6_attr) {
                    q6_attr = true;
                    CK(cudaFuncSetAttribute((const void*)&k_gemm_mma_q6k_128,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, Q6K128_SMEM));
                }
                dim3 g(N / 128, (rows + 127) / 128);
                k_gemm_mma_q6k_128<<<g, 256, Q6K128_SMEM, st>>>(y, (const uint8_t*)w.dp,
                    M.bm_xq, M.bm_xd, K, N, rows);
            } else {
                dim3 g(N / 64, (rows + 63) / 64);
                k_gemm_mma_q6k<<<g, 256, Q6K_SMEM, st>>>(y, (const uint8_t*)w.dp, M.bm_xq, M.bm_xd, K, N, rows);
            }
            if (getenv("BRUCE_Q6_CHK") && atoi(getenv("BRUCE_Q6_CHK")) != 0) {
                float* tmp = 0; CK(cudaMalloc(&tmp, (size_t)rows * N * 4));
                for (int r = 0; r < rows; r += 8)
                    launch_gemm_b_impl(tmp + (size_t)r * N, w, std::min(8, rows - r), r, st);
                CK(cudaDeviceSynchronize());
                float* a = (float*)malloc((size_t)rows * N * 4);
                float* b = (float*)malloc((size_t)rows * N * 4);
                CK(cudaMemcpy(a, y, (size_t)rows * N * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(b, tmp, (size_t)rows * N * 4, cudaMemcpyDeviceToHost));
                double mx = 0, mg = 0; int bad = 0, nf = 0;
                for (size_t i = 0; i < (size_t)rows * N; ++i) {
                    if (!isfinite(a[i])) ++nf;
                    mx = std::max(mx, (double)fabsf(a[i] - b[i]));
                    mg = std::max(mg, (double)fabsf(b[i]));
                    if (fabsf(a[i] - b[i]) > 1e-2f * fabsf(b[i]) + 1e-3f) ++bad;
                }
                fprintf(stderr, "[q6k] K=%d N=%d rows=%d: max |diff| %.4g (max |y| %.4g) bad %d/%zu nonfinite %d\n",
                        K, N, rows, mx, mg, bad, (size_t)rows * N, nf);
                static int once = 0;
                if (!once++) {
                    fprintf(stderr, "[q6k] mma: "); for (int i = 0; i < 6; ++i) fprintf(stderr, "%.5g ", a[i]);
                    fprintf(stderr, "\n[q6k] ref: "); for (int i = 0; i < 6; ++i) fprintf(stderr, "%.5g ", b[i]);
                    fprintf(stderr, "\n");
                }
                free(a); free(b); CK(cudaFree(tmp));
            }
            cudaError_t e = cudaGetLastError();
            if (e) { fprintf(stderr, "mma q6k launch failed rows=%d K=%d N=%d: %s\n", rows, K, N, cudaGetErrorString(e)); exit(1); }
            return;
        }
        if (mma_ok && w.type == GGML_TYPE_Q5_K && (K % 256) == 0 && (N % 64) == 0) {
            dim3 g(N / 64, (rows + 63) / 64);
            k_gemm_mma_q5k<<<g, 256, MMA_SMEM, st>>>(y, (const uint8_t*)w.d, M.bm_xq, M.bm_xd, M.bm_xsum, K, N, rows);
            if (getenv("BRUCE_MMA_CHK")) {
                float* tmp = 0; CK(cudaMalloc(&tmp, (size_t)rows * N * 4));
                for (int r = 0; r < rows; r += 8)
                    launch_gemm_b_impl(tmp + (size_t)r * N, w, std::min(8, rows - r), r, st);
                CK(cudaDeviceSynchronize());
                float* a = (float*)malloc((size_t)rows * N * 4);
                float* b = (float*)malloc((size_t)rows * N * 4);
                CK(cudaMemcpy(a, y, (size_t)rows * N * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(b, tmp, (size_t)rows * N * 4, cudaMemcpyDeviceToHost));
                double mx = 0, mg = 0; int bad = 0, nan_a = 0, nan_b = 0, first = -1;
                for (size_t i = 0; i < (size_t)rows * N; ++i) {
                    if (!isfinite(a[i])) { if (!nan_a && first < 0) first = (int)(i / N); ++nan_a; }
                    if (!isfinite(b[i])) ++nan_b;
                    mx = std::max(mx, (double)fabsf(a[i] - b[i]));
                    mg = std::max(mg, (double)fabsf(b[i]));
                    if (fabsf(a[i] - b[i]) > 1e-2f * fabsf(b[i]) + 1e-3f) ++bad;
                }
                fprintf(stderr, "[mma] K=%d N=%d rows=%d: max |diff| %.4g (max |y| %.4g), bad %d/%zu, nonfinite mma %d ref %d (first row %d)\n",
                        K, N, rows, mx, mg, bad, (size_t)rows * N, nan_a, nan_b, first);
                free(a); free(b); CK(cudaFree(tmp));
            }
            cudaError_t e = cudaGetLastError();
            if (e) { fprintf(stderr, "mma gemm launch failed rows=%d K=%d N=%d: %s\n", rows, K, N, cudaGetErrorString(e)); exit(1); }
            return;
        }
        if (getenv("BRUCE_MMA_CHK"))
            fprintf(stderr, "[gemm-fallback] type=%d K=%d N=%d rows=%d\n", (int)w.type, K, N, rows);
        // Q6_K stages K*rows bytes of int8 activations: widen the row group as far as
        // shared memory allows so each weight pass serves as many rows as possible.
        int chunk = 8;
        if (w.type == GGML_TYPE_Q6_K) {
            if ((size_t)K * 32 <= 90000) chunk = 32;
            else if ((size_t)K * 16 <= 90000) chunk = 16;
        }
        for (int r = 0; r < rows; ) {
            int c = std::min(chunk, rows - r);
            if (c > 8 && c != chunk) c = 8;   // remainders use the always-available 8-row kernel
            launch_gemm_b_impl(y + (size_t)r * N, w, c, r, st);
            r += c;
        }
        return;
    }
    launch_gemm_b_impl(y, w, rows, 0, st);
}

static void launch_logits_b(int rows, cudaStream_t st) {
    if (skip_logits() || !p_on(PB_LOGITS)) return;
    if (M.out_q8) {
        const size_t sm = (size_t)rows * N_EMBD;
        const int blocks = gemv_grid(VOCAB, N_EMBD);
        #define Q8L(R) k_gemm_q8_logits<R, 8><<<blocks, 256, sm, st>>>(M.bm_logits, M.out_q8, \
            M.bm_xq, M.bm_xd, N_EMBD, VOCAB, M.out_scale)
        switch (rows) {
            case 1: Q8L(1); break;
            case 2: Q8L(2); break;
            case 3: Q8L(3); break;
            case 4: Q8L(4); break;
            case 5: Q8L(5); break;
            case 6: Q8L(6); break;
            case 7: Q8L(7); break;
            case 8: Q8L(8); break;
            default: fprintf(stderr, "bad logits rows %d\n", rows); exit(1);
        }
        #undef Q8L
        return;
    }
    const size_t sm = (size_t)rows * N_EMBD * 2;
    const int blocks = gemv_grid(VOCAB, N_EMBD);
    switch (rows) {
        case 1: k_gemm_bf16<1,8><<<blocks, 256, sm, st>>>(M.bm_logits, M.out_w, M.bm_x, N_EMBD, VOCAB, M.out_scale); break;
        case 2: k_gemm_bf16<2,8><<<blocks, 256, sm, st>>>(M.bm_logits, M.out_w, M.bm_x, N_EMBD, VOCAB, M.out_scale); break;
        case 3: k_gemm_bf16<3,8><<<blocks, 256, sm, st>>>(M.bm_logits, M.out_w, M.bm_x, N_EMBD, VOCAB, M.out_scale); break;
        case 4: k_gemm_bf16<4,8><<<blocks, 256, sm, st>>>(M.bm_logits, M.out_w, M.bm_x, N_EMBD, VOCAB, M.out_scale); break;
        case 5: k_gemm_bf16<5,8><<<blocks, 256, sm, st>>>(M.bm_logits, M.out_w, M.bm_x, N_EMBD, VOCAB, M.out_scale); break;
        case 6: k_gemm_bf16<6,8><<<blocks, 256, sm, st>>>(M.bm_logits, M.out_w, M.bm_x, N_EMBD, VOCAB, M.out_scale); break;
        case 7: k_gemm_bf16<7,8><<<blocks, 256, sm, st>>>(M.bm_logits, M.out_w, M.bm_x, N_EMBD, VOCAB, M.out_scale); break;
        case 8: k_gemm_bf16<8,8><<<blocks, 256, sm, st>>>(M.bm_logits, M.out_w, M.bm_x, N_EMBD, VOCAB, M.out_scale); break;
        default: fprintf(stderr, "bad logits rows %d\n", rows); exit(1);
    }
}

// one pass over `rows` tokens at positions pos0..pos0+rows-1 (batched/chunk semantics)
// MS=false: all rows belong to one sequence (prompt chunk / speculative verify).
// MS=true:  each row is a different slot, with its own local position and KV
//           partition taken from M.d_row_pos / M.d_row_slot / M.d_row_act.
template <bool MS>
static void forward_batch_impl(int rows, bool all_logits, bool need_logits, cudaStream_t st,
                               bool skip_loop1 = false) {
    const int nsm = (N_EMBD + 32) * 4;
    // Prompt-attention row block for the full layers: wide at depth, narrow when
    // the chunk is short (a CTA then covers only a tile or two and the 48-row
    // Q load overhead dominates).  Prefill-only; decode/verify use their own path.
    g_full_nw_use = (g_batch_pos0 + rows >= g_full_nw_depth) ? prefill_full_nw() : 6;
    const int rec_end = N_INPUT + N_REC * M.n_loops;
    if (p_on(PB_EMBED)) {
        k_embed_rows<<<rows, 320, 0, st>>>(M.bm_x, M.emb, M.bm_tok, rows);
        k_q8_rows<<<rows, 256, 0, st>>>(M.bm_xq, M.bm_xd, M.bm_xsum, M.bm_x, N_EMBD, rows);
        if constexpr (MS)
            k_rope_tables_rows_ms<<<rows, 128, 0, st>>>(M.bm_cos, M.bm_sin, M.d_row_pos, rows);
        else
            k_rope_tables_rows<<<rows, 128, 0, st>>>(M.bm_cos, M.bm_sin, M.d_pos, rows);
    }
    for (int vi = 0; vi < M.n_virtual; ++vi) {
        if (skip_loop1 && vi >= N_INPUT + N_REC && vi < rec_end) continue;
        const int b = M.order[vi];
        BlockW& w = M.blocks[b];
        if (g_debug && vi == 0) { CK(cudaMemcpyAsync(M.dbg_taps + TAP_TOK, M.bm_tok, 4, cudaMemcpyDeviceToDevice, st)); CK(cudaMemcpyAsync(M.dbg_taps + TAP_POS, M.d_pos, 4, cudaMemcpyDeviceToDevice, st)); }
        if (vi == N_INPUT) k_copy<<<8, 320, 0, st>>>(M.bm_input_h, M.bm_x, N_EMBD * rows);
        if (vi >= N_INPUT && vi < rec_end && ((vi - N_INPUT) % N_REC) == 0)
            if (p_on(PB_NORM)) k_add_norm_rows<<<rows, 512, nsm, st>>>(M.bm_x, M.bm_input_h, N_EMBD, M.bm_xq, M.bm_xd, M.bm_xsum);
        if (!w.is_ffn) {
            if (p_on(PB_QKV)) {
            if (rows > 8) {
                // q/k/gate share one activation tile.  Half the layers have a Q5_K
                // v, which joins the same fused launch; the rest stream it as q6_K.
                float* ys[4] = {M.bm_q, M.bm_k, M.bm_gate, M.bm_v};
                const TensorW* ws[4] = {&w.wq, &w.wk, &w.wg, &w.wv};
                const int nf = (w.wv.type == GGML_TYPE_Q5_K) ? 4 : 3;
                launch_gemm_b_fused(ys, ws, nf, rows, st);
                if (nf == 3) launch_gemm_b(M.bm_v, w.wv, rows, st);
            } else {
                float* ys[4] = {M.bm_q, M.bm_k, M.bm_gate, M.bm_v};
                const TensorW* ws[4] = {&w.wq, &w.wk, &w.wg, &w.wv};
                const int nf = (w.wv.type == GGML_TYPE_Q5_K) ? 4 : 3;
                if (!launch_verify_fused(ys, ws, nf, rows, M.bm_xq, M.bm_xd, M.bm_xsum, 0, st)) {
                    for (int p = 0; p < nf; ++p) launch_gemm_b(ys[p], *ws[p], rows, st);
                }
                if (nf == 3) launch_gemm_b(M.bm_v, w.wv, rows, st);
            }
            }
            const bool dbg = g_debug && vi == g_debug_block;
            const size_t lr = (size_t)(rows - 1);
            if (dbg) dbg_copy(M.dbg_taps + TAP_XIN, M.bm_x + lr * N_EMBD, N_EMBD);
            if (dbg) {
                dbg_copy(M.dbg_taps + TAP_Q, M.bm_q + lr * N_HEAD * HD, N_HEAD * HD);
                dbg_copy(M.dbg_taps + TAP_K, M.bm_k + lr * N_KV * HD, N_KV * HD);
                dbg_copy(M.dbg_taps + TAP_V, M.bm_v + lr * N_KV * HD, N_KV * HD);
                dbg_copy(M.dbg_taps + TAP_GATE, M.bm_gate + lr * N_HEAD * HD, N_HEAD * HD);
            }
            const int is_swa = w.full ? 0 : 1;
            if (p_on(PB_PREP)) {
            if constexpr (MS) {
                k_qkv_prep_fast_ms<<<rows, 256, 0, st>>>(M.bm_q, M.bm_k, M.bm_v, w.sqk,
                    M.kvk_layer[b], M.kvv_layer[b], M.d_row_pos, M.d_row_slot, M.d_row_act,
                    is_swa, w.theta_sel, M.bm_cos, M.bm_sin);
            } else if (fast_qkv())
                k_qkv_prep_fast<<<rows, 256, 0, st>>>(M.bm_q, M.bm_k, M.bm_v, w.sqk,
                    M.kvk_ptr[b], M.kvv_ptr[b], M.d_pos, is_swa, w.theta_sel, M.bm_cos, M.bm_sin);
            else
                k_qkv_prep_rows<<<dim3(N_HEAD, rows), 256, 0, st>>>(M.bm_q, M.bm_k, M.bm_v, w.sqk,
                    M.kvk_ptr[b], M.kvv_ptr[b], M.d_pos, is_swa, w.theta_sel, M.bm_cos, M.bm_sin);
            }
            if (dbg) dbg_copy(M.dbg_taps + TAP_Q2, M.bm_q + lr * N_HEAD * HD, N_HEAD * HD);
            if (dbg) { kv_tap(M.dbg_taps + TAP_KVK, M.kvk_ptr[b], is_swa ? SWA_SLOTS : M.slot_ctx, st);
                       kv_tap(M.dbg_taps + TAP_KVV, M.kvv_ptr[b], is_swa ? SWA_SLOTS : M.slot_ctx, st); }
            // loop-1 attention reuse (decode/verify only, see l1reuse())
            const int l1s = (rows <= 8 && w.full && l1reuse() && !g_debug && !fused_combine())
                            ? M.l1_slot[b] : -1;
            const bool l1_reuse = l1s >= 0 && vi >= N_INPUT + N_REC;
            float* aout_p = (l1s >= 0) ? (M.aout_l0 + (size_t)l1s * 8 * N_HEAD * HD) : M.bm_aout;
            int ns = w.full ? M.full_splits : M.swa_splits;
            if (rows >= 2 && rows <= 4 && attn_m1_path() == 3 && !g_splits_explicit) ns = mv_splits(is_swa);
            else if (rows >= 2 && rows <= 8 && rs_verify()) ns = std::min(ns, verify_splits());
            // the split count exists to fill the GPU for single rows; a prompt chunk
            // already brings rows/8 * N_KV blocks, so keep ~256 blocks and no more
            const bool tc_path = (rows > 8) && (w.full ? (all_logits || g_prefill_tc) : true);
            if (rows > 8) {
                const int rpb = (w.full && g_attn_rs) ? 4 * prefill_full_nw_use() : 16;
                const int rdiv = tc_path ? (((rows > 8) && g_attn_rs) ? ((rows + rpb - 1) / rpb)
                                                                     : ((rows + 7) / 8)) : rows;
                ns = std::max(1, ns / rdiv);
                // Full layers: one split per row-chunk leaves 4*ceil(rows/24) blocks,
                // i.e. ~6 waves on 28 SMs, so the last wave is a 20-30% tail.  More
                // splits (disjoint key ranges, so the same KV bytes) fill the machine:
                // measured +1.8% at 32K, +0.5% at 16K, +1% at 8K.  Decode/verify keep
                // their tuned count.  BRUCE_PREFILL_SPLITS_MIN=0 restores the old rule.
                static int pmin = -1;
                if (pmin < 0) { const char* e = getenv("BRUCE_PREFILL_SPLITS_MIN"); pmin = e ? atoi(e) : -1; }
                if (w.full && g_attn_rs && pmin != 0) {
                    const int want = (12 * 28 + N_KV * rdiv - 1) / (N_KV * rdiv);
                    ns = std::max(ns, std::min(8, pmin > 0 ? std::min(pmin, want) : want));
                }
            }
            if ((size_t)rows * ns > PART_ROWS_MAX) {
                ns = std::max(1, PART_ROWS_MAX / rows);
            }
            if ((size_t)rows * ns > PART_ROWS_MAX) {
                fprintf(stderr, "part overflow rows=%d ns=%d (max %d row-splits)\n",
                        rows, ns, PART_ROWS_MAX);
                exit(1);
            }
            if (!skip_attn() && p_on(PB_ATTN) && !l1_reuse) {
            if constexpr (MS) {
                // each row is its own sequence: same M=1 warp kernel with a row axis
                ns = attn_ms_splits(rows, is_swa);
                launch_attn_ms(aout_p, M.bm_q, M.kvk_layer[b], M.kvv_layer[b], M.d_row_pos,
                               M.d_row_slot, M.d_row_act, M.part, rows, ns, is_swa, st);
            } else if (w.full && (all_logits || g_prefill_tc)) {
                // One KV pass shared by every row of the chunk: speculative verification and
                // prompt processing.  At depth the per-row kernel re-reads the whole KV cache
                // once per row; sharing it across the chunk cuts that traffic by `rows`.
                // The fused kernel rounds QK^T through fp16, so prompt logits shift by ~1e-3
                // at near-ties (same trade-off llama.cpp's -fa makes).  --prefill-tc 0 keeps
                // the bit-exact per-row path.
                launch_attn_m(aout_p, M.bm_q, M.kvk_ptr[b], M.kvv_ptr[b], M.d_pos, M.part, rows, ns, 0, st);
            } else if (!w.full && rows >= 2 && g_attn_tc) {
                // windowed layers: same shared-KV tensor-core pass, clamped to the window
                launch_attn_m(aout_p, M.bm_q, M.kvk_ptr[b], M.kvv_ptr[b], M.d_pos, M.part, rows, ns, 1, st);
            } else {
                // Prompt processing, and the windowed layers (which touch only SWA keys).
                k_attn_rows<<<dim3(N_KV, ns, rows), 128, 0, st>>>(aout_p, M.bm_q, M.kvk_ptr[b], M.kvv_ptr[b],
                                                                  M.d_pos, is_swa, M.part);
            }
            }
            if (p_on(PB_GATE)) {
            if (!dbg && (rows > 8 || fused_combine())) {
                // register-resident combine + gate norm (prompt path)
                if (ns > 1) {
                    k_combine_gate_norm_rows_opt<16><<<rows, 256, 0, st>>>(nullptr, M.part, M.bm_gate, w.sgate, ns,
                                                                           N_HEAD * HD, M.bm_xq, M.bm_xd, M.bm_xsum);
                } else if (rows > 8) {
                    k_gate_norm_vec4<<<rows, 256, 0, st>>>(nullptr, M.bm_aout, M.bm_gate, w.sgate,
                                                                   N_HEAD * HD, M.bm_xq, M.bm_xd, M.bm_xsum);
                } else {
                    k_gate_norm_vec4<<<rows, 256, 0, st>>>(nullptr, M.bm_aout, M.bm_gate, w.sgate,
                                                                   N_HEAD * HD, M.bm_xq, M.bm_xd, M.bm_xsum);
                }
            } else {
                if (ns > 1 && !l1_reuse) k_attn_combine_rows<<<dim3(N_HEAD, rows), 32, 0, st>>>(aout_p, M.part, ns);
                if (fast_norms())
                    k_gate_norm_vec4<<<rows, 256, 0, st>>>(M.bm_attn_in, aout_p, M.bm_gate, w.sgate,
                                                           N_HEAD * HD, M.bm_xq, M.bm_xd, M.bm_xsum);
                else
                    k_gate_norm<<<rows, 512, (N_HEAD * HD + 32) * 4, st>>>(M.bm_attn_in, aout_p, M.bm_gate, w.sgate,
                                                                           N_HEAD * HD, M.bm_xq, M.bm_xd, M.bm_xsum);
            }
            }
            if (p_on(PB_WO)) launch_gemm_b(M.bm_proj, w.wo, rows, st);
            if (dbg) {
                dbg_copy(M.dbg_taps + TAP_AOUT, aout_p + lr * N_HEAD * HD, N_HEAD * HD);
                dbg_copy(M.dbg_taps + TAP_AIN, M.bm_attn_in + lr * N_HEAD * HD, N_HEAD * HD);
                dbg_copy(M.dbg_taps + TAP_PROJ, M.bm_proj + lr * N_EMBD, N_EMBD);
            }
        } else {
            if (p_on(PB_FFGU)) {
            if (rows > 8) {
                float* ys[2] = {M.bm_ffn_g, M.bm_ffn_u};
                const TensorW* ws[2] = {&w.wfg, &w.wfu};
                launch_gemm_b_fused(ys, ws, 2, rows, st);
            } else {
                float* ys[2] = {M.bm_ffn_g, M.bm_ffn_u};
                const TensorW* ws[2] = {&w.wfg, &w.wfu};
                if (!launch_verify_fused(ys, ws, 2, rows, M.bm_xq, M.bm_xd, M.bm_xsum, 0, st)) {
                    for (int p = 0; p < 2; ++p) launch_gemm_b(ys[p], *ws[p], rows, st);
                }
            }
            }
            if (p_on(PB_SILU)) {
            if (fast_norms()) {
                k_silu_mul_norm_vec4<<<rows, 256, 0, st>>>(nullptr, M.bm_ffn_g, M.bm_ffn_u,
                                                                   w.suv_gate, w.suv_up, N_FF,
                                                                   M.bm_xq, M.bm_xd, M.bm_xsum);
            } else if (rows > 8) {
                k_silu_mul_norm_rows_opt<28><<<rows, 256, 0, st>>>(nullptr, M.bm_ffn_g, M.bm_ffn_u,
                                                                   w.suv_gate, w.suv_up, N_FF,
                                                                   M.bm_xq, M.bm_xd, M.bm_xsum);
            } else if (opt_norms()) {
                k_silu_mul_norm_rows_opt<7, 32><<<rows, 1024, 0, st>>>(nullptr, M.bm_ffn_g, M.bm_ffn_u,
                                                                       w.suv_gate, w.suv_up, N_FF,
                                                                       M.bm_xq, M.bm_xd, M.bm_xsum);
            } else {
                k_silu_mul_norm_rows<<<rows, 1024, (N_FF + 32) * 4, st>>>(M.bm_ffn_t, M.bm_ffn_g, M.bm_ffn_u,
                                                                         w.suv_gate, w.suv_up, N_FF,
                                                                         M.bm_xq, M.bm_xd, M.bm_xsum);
            }
            }
            if (p_on(PB_WFD)) launch_gemm_b(M.bm_proj, w.wfd, rows, st);
        }
        if (p_on(PB_RESID)) {
        if (fast_norms()) {
            k_resid_vec4<<<rows, 160, 0, st>>>(M.bm_x, M.bm_proj, w.alpha, N_EMBD,
                                                       M.bm_xq, M.bm_xd, M.bm_xsum);
        } else if (rows > 8) {
            k_resid_rows_opt<10><<<rows, 256, 0, st>>>(M.bm_x, M.bm_proj, w.alpha, N_EMBD,
                                                       M.bm_xq, M.bm_xd, M.bm_xsum);
        } else if (opt_norms()) {
            k_resid_rows_opt<3, 32><<<rows, 1024, 0, st>>>(M.bm_x, M.bm_proj, w.alpha, N_EMBD,
                                                           M.bm_xq, M.bm_xd, M.bm_xsum);
        } else {
            k_resid_rows<<<rows, 1024, (2 * N_EMBD + 32) * 4, st>>>(M.bm_x, M.bm_proj, w.alpha, N_EMBD,
                                                                    M.bm_xq, M.bm_xd, M.bm_xsum);
        }
        }
        if (vi >= N_INPUT && vi < rec_end && ((vi + 1 - N_INPUT) % N_REC) == 0)
            if (p_on(PB_NORM)) k_norm_rows<<<rows, 512, nsm, st>>>(M.bm_x, M.bm_x, N_EMBD, M.bm_xq, M.bm_xd, M.bm_xsum);
        if (vi == rec_end - N_REC - 1)
            k_copy<<<8, 320, 0, st>>>(M.x_prev, M.bm_x + (size_t)(rows - 1) * N_EMBD, N_EMBD);
        if (!skip_loop1 && M.n_loops > 1 && vi == rec_end - 1)
            k_l2_delta<<<1, 256, 0, st>>>(M.d_delta, M.bm_x + (size_t)(rows - 1) * N_EMBD, M.x_prev, N_EMBD);
        if (g_debug)
            k_copy<<<8, 320, 0, st>>>(M.dbg_x + (size_t)vi * N_EMBD,
                                      M.bm_x + (size_t)(rows - 1) * N_EMBD, N_EMBD);
    }
    if (!need_logits) return;
    if (all_logits) {
        launch_logits_b(rows, st);
    } else if (M.out_q8) {
        // only the last row's logits are needed (prefill): the last row's q8
        // activations are already in bm_xq, so no fp32 copy is needed.
        const size_t off = (size_t)(rows - 1) * N_EMBD;
        k_gemm_q8_logits<1, 8><<<gemv_grid(VOCAB, N_EMBD), 256, (size_t)N_EMBD, st>>>(
            M.logits, M.out_q8, M.bm_xq + off, M.bm_xd + off / 32, N_EMBD, VOCAB, M.out_scale);
    } else {
        // only the last row's logits are needed (prefill): reuse the exact M=1 path
        k_copy<<<8, 320, 0, st>>>(M.x, M.bm_x + (size_t)(rows - 1) * N_EMBD, N_EMBD);
        k_gemv_bf16<8, true><<<gemv_grid(VOCAB, N_EMBD), 256, (size_t)N_EMBD * 4, st>>>(
            M.logits, M.out_w, M.x, N_EMBD, VOCAB, M.out_scale);
    }
}

static void forward_batch(int rows, bool all_logits, bool need_logits, cudaStream_t st,
                          bool skip_loop1 = false) {
    forward_batch_impl<false>(rows, all_logits, need_logits, st, skip_loop1);
}

// One decode step over `rows` independent sequences (one row per active slot).
static void forward_batch_ms(int rows, bool all_logits, bool need_logits, cudaStream_t st) {
    forward_batch_impl<true>(rows, all_logits, need_logits, st, false);
}

static bool g_last_verify_1 = false;

// Speculative verification round: forward_batch + per-row argmax.  For 2..8 rows
// this runs as a captured CUDA graph (M.ex_verify), which removes ~800 CPU kernel
// launches per round.  The graph reads bm_tok / d_pos, so the caller only copies
// those two small buffers before replaying.  With --adaptive-loops a second set of
// graphs skips loop 1.
static void verify_launch(int rows, cudaStream_t st) {
    const int cs = (g_cur_slot >= 0 && g_cur_slot < MAX_SLOTS) ? g_cur_slot : 0;
    bool use1 = g_adaptive_loops && g_use_1loop && rows >= 2 && rows <= 8 && M.ex_verify1[cs][rows];
    if (use1 && g_skip_used >= g_loop_probe) { g_use_1loop = false; use1 = false; g_skip_used = 0; }
    if (rows >= 2 && rows <= 8 && M.ex_verify[cs][rows]) {
        g_last_verify_1 = use1;
        CK(cudaGraphLaunch(use1 ? M.ex_verify1[cs][rows] : M.ex_verify[cs][rows], st));
        if (use1) g_skip_used++;
        return;
    }
    g_last_verify_1 = false;
    forward_batch(rows, true, true, st);
    k_argmax_rows<<<rows, 1024, 0, st>>>(M.d_amax, M.bm_logits, VOCAB, rows);
}

static void capture_verify_graphs() {
    // warm the lazy attributes/launch state outside capture
    forward_batch(8, true, true, S);
    CK(cudaStreamSynchronize(S));
    const int nslots = std::max(1, M.n_slots);
    const int passes = g_adaptive_loops ? 2 : 1;
    for (int s = 0; s < nslots; ++s) {
        set_cur_slot(s);
        for (int pass = 0; pass < passes; ++pass) {
            const bool skip = pass == 1;
            for (int rows = 2; rows <= 8; ++rows) {
                cudaGraph_t graph = nullptr;
                CK(cudaStreamBeginCapture(S, cudaStreamCaptureModeThreadLocal));
                forward_batch(rows, true, true, S, skip);
                k_argmax_rows<<<rows, 1024, 0, S>>>(M.d_amax, M.bm_logits, VOCAB, rows);
                CK(cudaStreamEndCapture(S, &graph));
                cudaGraphExec_t* ex = skip ? &M.ex_verify1[s][rows] : &M.ex_verify[s][rows];
                CK(cudaGraphInstantiate(ex, graph, nullptr, nullptr, 0));
                CK(cudaGraphDestroy(graph));
            }
        }
    }
    set_cur_slot(0);
}

static void forward_launch(cudaStream_t st, bool skip_loop1 = false) {
    const int nsm = (N_EMBD + 32) * 4;
    const int rec_end = N_INPUT + N_REC * M.n_loops;
    if (p_on(PB_EMBED)) {
    k_embed<<<8, 320, 0, st>>>(M.x, M.emb, M.d_tok);
    k_q8_quant<<<1, 256, 0, st>>>(M.xq, M.xd, M.xsum, M.x, N_EMBD);
    k_rope_tables<<<1, 128, 0, st>>>(M.rope_cos, M.rope_sin, M.d_pos);
    }    for (int vi = 0; vi < M.n_virtual; ++vi) {
        if (skip_loop1 && vi >= N_INPUT + N_REC && vi < rec_end) continue;
        const int b = M.order[vi];
        BlockW& w = M.blocks[b];
        if (g_debug && vi == 0) { CK(cudaMemcpyAsync(M.dbg_taps + TAP_TOK, M.d_tok, 4, cudaMemcpyDeviceToDevice, st)); CK(cudaMemcpyAsync(M.dbg_taps + TAP_POS, M.d_pos, 4, cudaMemcpyDeviceToDevice, st)); }
        if (vi == N_INPUT) k_copy<<<8, 320, 0, st>>>(M.input_h, M.x, N_EMBD);
        if (g_debug && vi == N_INPUT) { dbg_copy(M.dbg_taps + TAP_IH, M.input_h, N_EMBD); dbg_copy(M.dbg_taps + TAP_X0, M.x, N_EMBD); }
        if (vi >= N_INPUT && vi < rec_end && ((vi - N_INPUT) % N_REC) == 0)
            if (p_on(PB_NORM)) k_add_norm<<<1, 512, nsm, st>>>(M.x, M.input_h, N_EMBD, M.xq, M.xd, M.xsum);
        if (!w.is_ffn) {
            if (p_on(PB_QKV)) {
                float* ys[4] = {M.q, M.k, M.gate, M.v};
                const TensorW* ws[4] = {&w.wq, &w.wk, &w.wg, &w.wv};
                const int nf = (w.wv.type == GGML_TYPE_Q5_K) ? 4 : 3;
                if (!launch_verify_fused(ys, ws, nf, 1, M.xq, M.xd, M.xsum, 0, st)) {
                    for (int p = 0; p < nf; ++p) launch_gemv(ys[p], *ws[p], M.x, st);
                }
                if (nf == 3) launch_gemv(M.v, w.wv, M.x, st);
            }
            const bool dbg = g_debug && vi == g_debug_block;
            if (dbg) dbg_copy(M.dbg_taps + TAP_XIN, M.x, N_EMBD);
            if (dbg) {
                dbg_copy(M.dbg_taps + TAP_Q, M.q, N_HEAD * HD);
                dbg_copy(M.dbg_taps + TAP_K, M.k, N_KV * HD);
                dbg_copy(M.dbg_taps + TAP_V, M.v, N_KV * HD);
                dbg_copy(M.dbg_taps + TAP_GATE, M.gate, N_HEAD * HD);
            }
            const int is_swa = w.full ? 0 : 1;
            // M=1 keeps the original per-head prep so plain decode stays
            // bit-identical to the pre-plan engine (the fast form's reassociated
            // norm can flip a documented prose/chat near-tie).
            if (p_on(PB_PREP))
            k_qkv_prep<<<N_HEAD, 256, 0, st>>>(M.q, M.k, M.v, w.sqk,
                M.kvk_ptr[b], M.kvv_ptr[b], M.d_pos, is_swa, w.theta_sel,
                M.rope_cos, M.rope_sin);if (dbg) dbg_copy(M.dbg_taps + TAP_Q2, M.q, N_HEAD * HD);
            if (dbg) { kv_tap(M.dbg_taps + TAP_KVK, M.kvk_ptr[b], is_swa ? SWA_SLOTS : M.slot_ctx, st);
                       kv_tap(M.dbg_taps + TAP_KVV, M.kvv_ptr[b], is_swa ? SWA_SLOTS : M.slot_ctx, st); }
            int ns = w.full ? M.full_splits : M.swa_splits;
            const int m1 = attn_m1_path();
            if (m1 == 2) ns = std::min(ns, is_swa ? swa_verify_splits() : verify_splits());
            if (m1 == 3) { if (!g_splits_explicit) ns = m1_splits(is_swa); }
            // loop-1 attention reuse: see l1reuse().  l1s >= 0 means this block's
            // attention output is shared between loops; the first loop writes it,
            // later loops read it instead of re-streaming the KV cache.
            const int l1s = (w.full && !g_debug && l1reuse()) ? M.l1_slot[b] : -1;
            const bool l1_reuse = l1s >= 0 && vi >= N_INPUT + N_REC;
            float* aout_p = (l1s >= 0) ? (M.aout_l0 + (size_t)l1s * N_HEAD * HD) : M.aout;
            if (p_on(PB_ATTN) && !l1_reuse) {
            if (m1)
                launch_attn_m(aout_p, M.q, M.kvk_ptr[b], M.kvv_ptr[b], M.d_pos, M.part, 1, ns, is_swa, st);
            else if (attn_u4())
                k_attn_u4<<<dim3(N_KV, ns), 128, 0, st>>>(aout_p, M.q, M.kvk_ptr[b], M.kvv_ptr[b],
                                                          M.d_pos, is_swa, M.part);
            else
                k_attn<<<dim3(N_KV, ns), 128, 0, st>>>(aout_p, M.q, M.kvk_ptr[b], M.kvv_ptr[b],
                                                       M.d_pos, is_swa, M.part);
            }
            // M=1: original split combine + shared-memory gate norm (bit-identical).
            if (p_on(PB_GATE)) {
            if (ns > 1 && !l1_reuse) {
                if (attn_comb_fast()) k_attn_combine_fast<<<N_HEAD, 256, 0, st>>>(aout_p, M.part, ns);
                else k_attn_combine<<<N_HEAD, 32, 0, st>>>(aout_p, M.part, ns);
            }
            static int fastn_m1 = -1;
            if (fastn_m1 < 0) { const char* e = getenv("BRUCE_FAST_NORMS_M1"); fastn_m1 = (!e || atoi(e)) ? 1 : 0; }
            if (fastn_m1)
                k_gate_norm_vec4<<<1, 256, 0, st>>>(M.attn_in, aout_p, M.gate, w.sgate, N_HEAD * HD,
                                                   M.xq, M.xd, M.xsum);
            else
            k_gate_norm<<<1, 512, (N_HEAD * HD + 32) * 4, st>>>(M.attn_in, aout_p, M.gate, w.sgate, N_HEAD * HD,
                                                               M.xq, M.xd, M.xsum);
            }
            if (p_on(PB_WO))
            launch_gemv(M.proj, w.wo, M.attn_in, st);
            if (dbg) {
                dbg_copy(M.dbg_taps + TAP_AOUT, aout_p, N_HEAD * HD);
                dbg_copy(M.dbg_taps + TAP_AIN, M.attn_in, N_HEAD * HD);
                dbg_copy(M.dbg_taps + TAP_PROJ, M.proj, N_EMBD);
            }
        } else {
            if (p_on(PB_FFGU)) {
                float* ys[2] = {M.ffn_g, M.ffn_u};
                const TensorW* ws[2] = {&w.wfg, &w.wfu};
                if (!launch_verify_fused(ys, ws, 2, 1, M.xq, M.xd, M.xsum, 0, st)) {
                    for (int p = 0; p < 2; ++p) launch_gemv(ys[p], *ws[p], M.x, st);
                }
            }
            // M=1: original kernels (bit-identical decode).  The float4 forms are
            // used by the 2..8 row verify path and by prompt chunks.
            if (p_on(PB_SILU)) {
            static int fasts_m1 = -1;
            if (fasts_m1 < 0) { const char* e = getenv("BRUCE_FAST_NORMS_M1"); fasts_m1 = (!e || atoi(e)) ? 1 : 0; }
            if (fasts_m1)
                k_silu_mul_norm_vec4<<<1, 256, 0, st>>>(g_debug ? M.ffn_t : nullptr,
                                                        M.ffn_g, M.ffn_u, w.suv_gate, w.suv_up, N_FF,
                                                        M.xq, M.xd, M.xsum);
            else
            k_silu_mul_norm_rows_opt<7, 32><<<1, 1024, 0, st>>>(g_debug ? M.ffn_t : nullptr,
                                                                 M.ffn_g, M.ffn_u, w.suv_gate, w.suv_up, N_FF,
                                                                 M.xq, M.xd, M.xsum);
            }
            if (p_on(PB_WFD))
            launch_gemv(M.proj, w.wfd, M.x, st);
        }
        if (p_on(PB_RESID)) {
        static int fastr_m1 = -1;
        if (fastr_m1 < 0) { const char* e = getenv("BRUCE_FAST_NORMS_M1"); fastr_m1 = (!e || atoi(e)) ? 1 : 0; }
        if (fastr_m1)
            k_resid_vec4<<<1, 160, 0, st>>>(M.x, M.proj, w.alpha, N_EMBD, M.xq, M.xd, M.xsum);
        else
        k_resid_rows_opt<3, 32><<<1, 1024, 0, st>>>(M.x, M.proj, w.alpha, N_EMBD, M.xq, M.xd, M.xsum);
        }
        if (vi >= N_INPUT && vi < rec_end && ((vi + 1 - N_INPUT) % N_REC) == 0)
            if (p_on(PB_NORM)) k_norm<<<1, 512, nsm, st>>>(M.x, M.x, N_EMBD, M.xq, M.xd, M.xsum);
        if (vi == rec_end - N_REC - 1) k_copy<<<8, 320, 0, st>>>(M.x_prev, M.x, N_EMBD);
        if (M.n_loops > 1 && vi == rec_end - 1)
            k_l2_delta<<<1, 256, 0, st>>>(M.d_delta, M.x, M.x_prev, N_EMBD);
        if (g_debug) dbg_copy(M.dbg_x + (size_t)vi * N_EMBD, M.x, N_EMBD);
    }
    if (M.out_q8) {
        if (p_on(PB_LOGITS))
        k_gemm_q8_logits<1, 8><<<gemv_grid(VOCAB, N_EMBD), 256, (size_t)N_EMBD, st>>>(
            M.logits, M.out_q8, M.xq, M.xd, N_EMBD, VOCAB, M.out_scale);
    } else {
        if (p_on(PB_LOGITS))
        k_gemv_bf16<8, true><<<gemv_grid(VOCAB, N_EMBD), 256, (size_t)N_EMBD * 4, st>>>(
            M.logits, M.out_w, M.x, N_EMBD, VOCAB, M.out_scale);
    }
}

// ---------------------------------------------------------------- runtime

static void set_tok_pos(int tok, int pos) {
    CK(cudaMemcpyAsync(M.d_tok, &tok, 4, cudaMemcpyHostToDevice, S));
    CK(cudaMemcpyAsync(M.d_pos, &pos, 4, cudaMemcpyHostToDevice, S));
}

static void capture_graph(cudaGraphExec_t* ex, bool advance, bool skip_loop1 = false) {
    cudaGraph_t graph = nullptr;
    CK(cudaStreamSynchronize(S));
    CK(cudaStreamBeginCapture(S, cudaStreamCaptureModeThreadLocal));
    forward_launch(S, skip_loop1);
    if (advance) k_advance<<<1, 1, 0, S>>>(M.d_pos);
    CK(cudaStreamEndCapture(S, &graph));
    CK(cudaGraphInstantiate(ex, graph, nullptr, nullptr, 0));
    CK(cudaGraphDestroy(graph));
}

// Adaptive decode: replay the loop-1-skipping graph when the previous token's
// loop-0 fixed-point step was below threshold.  Both graphs are the same code
// up to the end of loop 0, so the delta stays fresh either way.
static void launch_decode_graph(cudaStream_t st) {
    if (g_adaptive_loops && g_use_1loop) {
        if (g_skip_used >= g_loop_probe) { g_use_1loop = false; g_skip_used = 0; }
    }
    g_last_ran_1 = g_adaptive_loops && g_use_1loop && M.ex_fwd1;
    if (g_last_ran_1) {
        g_skip_used++;
        CK(cudaGraphLaunch(M.ex_fwd1, st));
    } else if (M.ex_fwd) {
        CK(cudaGraphLaunch(M.ex_fwd, st));
    } else {
        forward_launch(st);
    }
}

static void update_adaptive() {
    if (!g_adaptive_loops) return;
    if (g_last_delta < g_loop_tau) g_use_1loop = true;
    else if (g_last_delta > g_loop_tau * 1.5f) g_use_1loop = false;
    static int dbg = -1;
    if (dbg < 0) dbg = getenv("BRUCE_DELTA_LOG") ? 1 : 0;
    if (dbg) fprintf(stderr, "[delta] loop0-vs-loop1 1-cos %.6f rel %.5f -> %s\n",
                     g_last_rel, g_last_delta, g_use_1loop ? "skip loop 1" : "2 loops");
}

// Read the loop-convergence metric after a verification round (the caller has
// already synchronised for the argmax).  A round that skipped loop 1 did not
// refresh the metric, so keep the previous decision.
static void verify_update() {
    if (!g_adaptive_loops || g_last_verify_1) return;
    float d[2] = {g_last_rel, g_last_delta};
    CK(cudaMemcpyAsync(d, M.d_delta, 8, cudaMemcpyDeviceToHost, S));
    CK(cudaStreamSynchronize(S));
    g_last_rel = d[0]; g_last_delta = d[1];
    update_adaptive();
}

// ---------------------------------------------------------------- multi-slot serving
// A serving slot is an independent sequence: its own KV partition, its own
// positions 0..slot_ctx-1, its own prefix cache and sampler state.  A decode step
// batches one row per active slot through forward_batch_ms (one weight pass for all
// of them); prefill stays per-slot and is chunked by the server scheduler.

// Dev check: batch N independent sequences in N slots, then recompute each row
// alone through the M=1 path and compare logits.  Both use the same attention
// kernel and split count, so any difference is the batched GEMM's rounding.
static void bruce_prefill_chunk(int slot, const int* toks, int n, int pos, bool need_logits);
static void bruce_decode_step(int rows, const int* toks, const int* pos, const int* slots);
static void capture_ms_graphs();
static void ms_selftest(int nrows) {
    nrows = std::max(2, std::min(nrows, M.n_slots));
    if (!M.ex_ms[nrows]) capture_ms_graphs();
    const int plen = 16;
    std::vector<int> toks((size_t)plen + 1);
    for (int i = 0; i < plen + 1; ++i) toks[(size_t)i] = 100 + i * 7;
    for (int s = 0; s < nrows; ++s) bruce_prefill_chunk(s, toks.data(), plen, 0, false);
    CK(cudaStreamSynchronize(S));
    int next[MAX_ROWS], pos[MAX_ROWS], slots[MAX_ROWS];
    for (int s = 0; s < nrows; ++s) { next[s] = toks[(size_t)plen]; pos[s] = plen; slots[s] = s; }
    bruce_decode_step(nrows, next, pos, slots);
    CK(cudaStreamSynchronize(S));
    std::vector<float> ms((size_t)nrows * VOCAB), one(VOCAB);
    CK(cudaMemcpy(ms.data(), M.bm_logits, sizeof(float) * ms.size(), cudaMemcpyDeviceToHost));
    for (int s = 0; s < nrows; ++s) {
        set_cur_slot(s);
        CK(cudaMemcpyAsync(M.d_tok, &next[s], 4, cudaMemcpyHostToDevice, S));
        CK(cudaMemcpyAsync(M.d_pos, &pos[s], 4, cudaMemcpyHostToDevice, S));
        forward_launch(S);
        CK(cudaMemcpyAsync(one.data(), M.logits, sizeof(float) * VOCAB, cudaMemcpyDeviceToHost, S));
        CK(cudaStreamSynchronize(S));
        double mx = 0;
        int bad = 0, amax_ms = 0, amax_one = 0;
        for (int i = 0; i < VOCAB; ++i) {
            const float a = ms[(size_t)s * VOCAB + i], b = one[(size_t)i];
            const double d = fabs((double)a - (double)b);
            if (d > mx) mx = d;
            if (d > 1e-2 * fabs((double)b) + 1e-3) ++bad;
            if (a > ms[(size_t)s * VOCAB + amax_ms]) amax_ms = i;
            if (b > one[(size_t)amax_one]) amax_one = i;
        }
        printf("ms row %d (slot %d): max |diff| %.5g, bad %d/%d, argmax %d vs %d %s\n",
               s, s, mx, bad, VOCAB, amax_ms, amax_one, amax_ms == amax_one ? "OK" : "DIFF");
        printf("   top: ms %d=%.4f %d=%.4f %d=%.4f | m1 %d=%.4f %d=%.4f %d=%.4f | max|m1| %.3f\n",
               amax_ms, ms[(size_t)s * VOCAB + amax_ms],
               (amax_ms + 1) % VOCAB, ms[(size_t)s * VOCAB + (amax_ms + 1) % VOCAB],
               (amax_ms + 2) % VOCAB, ms[(size_t)s * VOCAB + (amax_ms + 2) % VOCAB],
               amax_one, one[(size_t)amax_one],
               (amax_one + 1) % VOCAB, one[(size_t)((amax_one + 1) % VOCAB)],
               (amax_one + 2) % VOCAB, one[(size_t)((amax_one + 2) % VOCAB)],
               [&] { double m = 0; for (float v : one) m = std::max(m, (double)fabsf(v)); return m; }());
        // same row through forward_batch(1): batched verify GEMM + M=1 attention
        set_cur_slot(s);
        CK(cudaMemcpyAsync(M.bm_tok, &next[s], 4, cudaMemcpyHostToDevice, S));
        CK(cudaMemcpyAsync(M.d_pos, &pos[s], 4, cudaMemcpyHostToDevice, S));
        forward_batch(1, true, true, S);
        CK(cudaMemcpyAsync(one.data(), M.bm_logits, sizeof(float) * VOCAB, cudaMemcpyDeviceToHost, S));
        CK(cudaStreamSynchronize(S));
        double mxb = 0;
        for (int i = 0; i < VOCAB; ++i) {
            mxb = std::max(mxb, fabs((double)ms[(size_t)s * VOCAB + i] - (double)one[(size_t)i]));
        }
        printf("   ms vs forward_batch(1): max |diff| %.5g\n", mxb);
    }
    set_cur_slot(0);
    // raw step cost (no HTTP, no sampler): weight stream + attention + logits
    for (int i = 0; i < 3; ++i) bruce_decode_step(nrows, next, pos, slots);
    CK(cudaStreamSynchronize(S));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, S));
    for (int i = 0; i < 20; ++i) bruce_decode_step(nrows, next, pos, slots);
    CK(cudaEventRecord(e1, S));
    CK(cudaEventSynchronize(e1));
    float step_ms = 0;
    CK(cudaEventElapsedTime(&step_ms, e0, e1));
    printf("ms step: %d rows, %.3f ms/step (%.1f tok/s aggregate)\n", nrows, step_ms / 20.0,
           nrows * 20000.0 / step_ms);
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));

    // Attention-only A/B: the same 8 consecutive positions of one sequence through
    // the verify path (shared-KV row-split attention) and the MS path (per-row M=1
    // attention).  Both use the same 8-row GEMM, so this isolates the attention
    // kernel: the two must agree to rounding.
    {
        const int plen2 = 16;
        int tk[24];
        for (int i = 0; i < 24; ++i) tk[i] = 100 + i * 7;
        bruce_prefill_chunk(0, tk, plen2, 0, false);
        CK(cudaStreamSynchronize(S));
        set_cur_slot(0);
        CK(cudaMemcpyAsync(M.bm_tok, tk + plen2, sizeof(int) * 8, cudaMemcpyHostToDevice, S));
        int p0 = plen2;
        CK(cudaMemcpyAsync(M.d_pos, &p0, 4, cudaMemcpyHostToDevice, S));
        forward_batch(8, true, true, S);
        CK(cudaStreamSynchronize(S));
        std::vector<float> vb((size_t)8 * VOCAB);
        CK(cudaMemcpy(vb.data(), M.bm_logits, sizeof(float) * vb.size(), cudaMemcpyDeviceToHost));
        int pos8[MAX_ROWS], slot8[MAX_ROWS];
        for (int i = 0; i < 8; ++i) { pos8[i] = plen2 + i; slot8[i] = 0; }
        bruce_decode_step(8, tk + plen2, pos8, slot8);
        CK(cudaStreamSynchronize(S));
        std::vector<float> mb((size_t)8 * VOCAB);
        CK(cudaMemcpy(mb.data(), M.bm_logits, sizeof(float) * mb.size(), cudaMemcpyDeviceToHost));
        double worst = 0;
        int arg_bad = 0;
        for (int r = 0; r < 8; ++r) {
            double d = 0;
            int a1 = 0, a2 = 0;
            for (int i = 0; i < VOCAB; ++i) {
                const float x = vb[(size_t)r * VOCAB + i], y = mb[(size_t)r * VOCAB + i];
                d = std::max(d, (double)fabs((double)x - (double)y));
                if (x > vb[(size_t)r * VOCAB + a1]) a1 = i;
                if (y > mb[(size_t)r * VOCAB + a2]) a2 = i;
            }
            if (a1 != a2) ++arg_bad;
            worst = std::max(worst, d);
        }
        printf("attn A/B (verify vs ms, same 8-row GEMM): worst |diff| %.5g, argmax mismatches %d/8\n",
               worst, arg_bad);
        set_cur_slot(0);
    }
}

static void capture_ms_graphs() {
    if (M.n_slots <= 1) return;
    int pos[MAX_ROWS], slot[MAX_ROWS];
    for (int i = 0; i < MAX_ROWS; ++i) { pos[i] = i; slot[i] = i % M.n_slots; }
    CK(cudaMemcpy(M.d_row_pos, pos, sizeof(pos), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(M.d_row_slot, slot, sizeof(slot), cudaMemcpyHostToDevice));
    // warm the lazy attributes / launch state outside capture
    forward_batch_ms(MAX_ROWS, true, true, S);
    CK(cudaStreamSynchronize(S));
    for (int rows = 2; rows <= MAX_ROWS; ++rows) {
        cudaGraph_t graph = nullptr;
        CK(cudaStreamBeginCapture(S, cudaStreamCaptureModeThreadLocal));
        forward_batch_ms(rows, true, true, S);
        k_argmax_rows<<<rows, 1024, 0, S>>>(M.d_amax, M.bm_logits, VOCAB, rows);
        CK(cudaStreamEndCapture(S, &graph));
        CK(cudaGraphInstantiate(&M.ex_ms[rows], graph, nullptr, nullptr, 0));
        CK(cudaGraphDestroy(graph));
    }
}

// One decode token for each of `rows` active slots.  Copies the small per-row
// arrays, replays the batch graph, then moves each row's logits into that slot's
// private buffer so the request threads can sample without racing the next step.
static void bruce_decode_step(int rows, const int* toks, const int* pos, const int* slots) {
    if (rows <= 1) {
        const int s = slots[0];
        set_cur_slot(s);
        CK(cudaMemcpyAsync(M.d_tok, toks, 4, cudaMemcpyHostToDevice, S));
        CK(cudaMemcpyAsync(M.d_pos, pos, 4, cudaMemcpyHostToDevice, S));
        if (M.ex_fwd_slot[s]) CK(cudaGraphLaunch(M.ex_fwd_slot[s], S));
        else forward_launch(S);
        CK(cudaMemcpyAsync(M.slot_logits[s], M.logits, (size_t)VOCAB * 4,
                           cudaMemcpyDeviceToDevice, S));
        return;
    }
    CK(cudaMemcpyAsync(M.bm_tok, toks, sizeof(int) * rows, cudaMemcpyHostToDevice, S));
    CK(cudaMemcpyAsync(M.d_row_pos, pos, sizeof(int) * rows, cudaMemcpyHostToDevice, S));
    CK(cudaMemcpyAsync(M.d_row_slot, slots, sizeof(int) * rows, cudaMemcpyHostToDevice, S));
    if (M.ex_ms[rows]) {
        CK(cudaGraphLaunch(M.ex_ms[rows], S));
    } else {
        forward_batch_ms(rows, true, true, S);
        k_argmax_rows<<<rows, 1024, 0, S>>>(M.d_amax, M.bm_logits, VOCAB, rows);
    }
    for (int r = 0; r < rows; ++r) {
        CK(cudaMemcpyAsync(M.slot_logits[slots[r]], M.bm_logits + (size_t)r * VOCAB,
                           (size_t)VOCAB * 4, cudaMemcpyDeviceToDevice, S));
    }
}

// One prompt chunk for one slot.  `n` <= n_batch; positions are local to the slot.
static void bruce_prefill_chunk(int slot, const int* toks, int n, int pos, bool need_logits) {
    set_cur_slot(slot);
    g_batch_pos0 = pos;
    if (n <= 1) {
        CK(cudaMemcpyAsync(M.d_tok, toks, 4, cudaMemcpyHostToDevice, S));
        CK(cudaMemcpyAsync(M.d_pos, &pos, 4, cudaMemcpyHostToDevice, S));
        if (M.ex_fwd_slot[slot]) CK(cudaGraphLaunch(M.ex_fwd_slot[slot], S));
        else forward_launch(S);
    } else {
        CK(cudaMemcpyAsync(M.bm_tok, toks, sizeof(int) * n, cudaMemcpyHostToDevice, S));
        CK(cudaMemcpyAsync(M.d_pos, &pos, 4, cudaMemcpyHostToDevice, S));
        forward_batch(n, false, need_logits, S);
    }
    if (need_logits) {
        CK(cudaMemcpyAsync(M.slot_logits[slot], M.logits, (size_t)VOCAB * 4,
                           cudaMemcpyDeviceToDevice, S));
    }
}

// Speculative verification for one slot (only when it is the only active session).
// Returns the per-row argmax in `out` after the stream is synchronised.
static void bruce_verify_round(int slot, const int* toks, int n, int pos, int* out) {
    set_cur_slot(slot);
    CK(cudaMemcpyAsync(M.bm_tok, toks, sizeof(int) * n, cudaMemcpyHostToDevice, S));
    CK(cudaMemcpyAsync(M.d_pos, &pos, 4, cudaMemcpyHostToDevice, S));
    verify_launch(n, S);
    CK(cudaMemcpyAsync(out, M.d_amax, sizeof(int) * n, cudaMemcpyDeviceToHost, S));
    CK(cudaStreamSynchronize(S));
    verify_update();
}

// Sampling is per slot but the scratch buffers (probs/sort) are shared, so the
// server serialises it with this lock.  `logits` and `rng` are the slot's own.
static std::mutex g_sample_mu;

static void sample_tokens_ex(const float* logits, unsigned long long* rng,
                             bool greedy, float temp, float topp, int topk, float minp) {
    if (greedy || temp <= 0.f) {
        k_argmax<<<1, 1024, 0, S>>>(M.d_tok, logits, VOCAB);
        return;
    }
    k_logit_max<<<1, 1024, 0, S>>>(M.samp_max, logits, VOCAB, temp);
    k_probs<<<128, 1024, 0, S>>>(M.probs, logits, M.samp_max, VOCAB, temp);
    cub::DeviceRadixSort::SortPairsDescending(M.sort_tmp, M.sort_tmp_sz, M.probs, M.sort_key,
                                              M.idx_in, M.sort_idx, VOCAB, 0, sizeof(float) * 8, S);
    k_sample_top_p<<<1, 1024, 1024 * 4, S>>>(M.d_tok, M.sort_key, M.sort_idx, VOCAB, topp,
                                             topk, minp, rng);
}

static void sample_tokens_ex(bool greedy, float temp, float topp, int topk, float minp) {
    sample_tokens_ex(M.logits, M.d_rng, greedy, temp, topp, topk, minp);
}

static void sample_tokens(bool greedy, float temp, float topp) {
    sample_tokens_ex(greedy, temp, topp, 0, 0.f);
}

// ---- dev probe: time one forward pass with individual phases disabled, so the
// delta is that phase's contribution to the decode step.  Used to attribute the
// non-weight milliseconds; not part of the normal path (g_poff stays 0).
static void probe_mode(int pos, int reps, int rows = 1, bool ms = false) {
    pos = std::min(pos, std::max(0, M.slot_ctx - std::max(rows, 1)));   // stay inside the KV cache
    g_batch_pos0 = pos;
    if (rows <= 1 && !ms) {
        set_tok_pos(1, pos);
    } else if (!ms) {
        int rk[MBUF + 1];
        for (int i = 0; i < rows; ++i) rk[i] = 1000 + i;
        CK(cudaMemcpyAsync(M.bm_tok, rk, sizeof(int) * rows, cudaMemcpyHostToDevice, S));
        CK(cudaMemcpyAsync(M.d_pos, &pos, 4, cudaMemcpyHostToDevice, S));
    } else {
        // Multi-slot decode: one row per independent slot, all at the same local depth.
        rows = std::max(1, std::min(rows, M.n_slots));
        int rk[MAX_ROWS], rp[MAX_ROWS], rs[MAX_ROWS], ra[MAX_ROWS];
        for (int i = 0; i < rows; ++i) { rk[i] = 1000 + i; rp[i] = pos; rs[i] = i; ra[i] = 1; }
        CK(cudaMemcpyAsync(M.bm_tok, rk, sizeof(int) * rows, cudaMemcpyHostToDevice, S));
        CK(cudaMemcpyAsync(M.d_row_pos, rp, sizeof(int) * rows, cudaMemcpyHostToDevice, S));
        CK(cudaMemcpyAsync(M.d_row_slot, rs, sizeof(int) * rows, cudaMemcpyHostToDevice, S));
        CK(cudaMemcpyAsync(M.d_row_act, ra, sizeof(int) * rows, cudaMemcpyHostToDevice, S));
    }
    CK(cudaStreamSynchronize(S));
    auto pass = [&]() {
        if (ms && rows >= 1) forward_batch_ms(rows, true, true, S);
        else if (rows <= 1) forward_launch(S);
        else forward_batch(rows, rows <= 8, true, S);
    };
    auto timeit = [&](unsigned off) {
        g_poff = off;
        for (int i = 0; i < 3; ++i) pass();
        CK(cudaStreamSynchronize(S));
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a, S));
        for (int i = 0; i < reps; ++i) pass();
        CK(cudaEventRecord(b, S));
        CK(cudaEventSynchronize(b));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, a, b));
        CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
        g_poff = 0;
        return (double)ms / reps;
    };
    const double full = timeit(0);
    printf("probe pos %d rows %d reps %d%s: full %.3f ms\n", pos, rows, reps,
           ms ? " multi-slot" : "", full);
    struct { const char* n; unsigned bit; } list[] = {
        {"qkv gemm", PB_QKV}, {"qkv prep", PB_PREP}, {"attention", PB_ATTN},
        {"combine+gate", PB_GATE}, {"wo gemv", PB_WO}, {"ffn gate/up", PB_FFGU},
        {"silu", PB_SILU}, {"ffn down", PB_WFD}, {"resid", PB_RESID},
        {"block norms", PB_NORM}, {"logits", PB_LOGITS}, {"embed/rope", PB_EMBED},
    };
    double sum = 0;
    for (auto& e : list) {
        const double t = timeit(e.bit);
        sum += full - t;
        printf("  -%-14s %8.3f ms  (%7.3f saved)\n", e.n, t, full - t);
    }
    printf("  sum of deltas %.3f ms\n", sum);
    printf("only-phase times (host launch overhead included):\n");
    for (auto& e : list) {
        const double t = timeit(PB_ALL & ~e.bit);
        printf("  %-15s %8.3f ms\n", e.n, t);
    }
}

static int get_tok() {    // S is a non-blocking stream, so a synchronous cudaMemcpy on the null stream is
    // NOT ordered after the sampler kernel.  Read through S and wait for it.
    int t = 0;
    float d[2] = {g_last_delta, g_last_rel};
    CK(cudaMemcpyAsync(&t, M.d_tok, 4, cudaMemcpyDeviceToHost, S));
    if (g_adaptive_loops) CK(cudaMemcpyAsync(d, M.d_delta, 8, cudaMemcpyDeviceToHost, S));
    CK(cudaStreamSynchronize(S));
    if (g_adaptive_loops) {
        g_last_rel = d[0]; g_last_delta = d[1];
        if (!g_last_ran_1) update_adaptive();
    }
    return t;
}

static void print_mem(double t0) {
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    fprintf(stderr, "[bruce] weights %.1f MiB, kv %.1f MiB, free %.1f MiB of %.1f MiB (load %.2f s)\n",
            g_weight_bytes / 1048576.0, g_kv_bytes / 1048576.0,
            free_b / 1048576.0, total_b / 1048576.0, (now_ms() - t0) / 1000.0);
}

static std::string default_gguf_path() {
    const char* env = getenv("BRUCE_GGUF");
    return std::string(env && *env ? env : "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf");
}

struct Args {
    std::string gguf = default_gguf_path();
    int ctx = 262144;
    int loops = 2;
    std::string tokens;
    std::string token_file;
    int gen = 0;
    bool greedy = false;
    float temp = 1.0f;
    float top_p = 0.95f;
    int top_k = 0;
    float min_p = 0.0f;
    unsigned long long seed = 0x1234567890ABCDEFull;
    std::string dump_logits;
    std::string dump_x;
    std::string dump_taps;
    int top_logits = 0;
    int bench = 0;
    bool no_graph = false;
    int gemv_blocks = 112;
    int gemm_blocks = 0;
    int splits = 0;
    int splits_full = 0;
    int splits_swa = 0;
    int bench_at = 0;
    int bench_rows = 1;   // >1: time the batched (verify) path with this many rows
    int prefill_batch = MBUF;
    bool force_batch = false;
    bool print_ids = false;
    bool probe = false;
    bool probe_ms = false;  // dev: --probe through the multi-slot batched decode path
    bool bench_ms = false;  // dev: time the multi-slot batched decode step directly
    bool ms_test = false;   // dev: compare multi-slot batch logits against per-slot M=1 logits
    int slots = 0;          // concurrent serving slots (0 = auto: 4 for --serve, else 1)
    int draft = 0;          // max draft length for prompt-lookup speculation (0 = off)
    int draft_ngram = 4;    // longest n-gram used for the lookup
    int draft_nmin = -1;    // shortest n-gram (-1 = nmax-2, i.e. 2)
    int q5k_prepack = -1;   // -1 auto, 0 off, 1 on

    // ---- server / tokenizer modes
    bool serve = false;
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string alias = "bruce";
    std::string api_key;
    int n_predict = 256;
    int n_keep = 0;
    float repeat_penalty = 1.0f;
    int repeat_last_n = 64;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    bool cache_prompt = true;
    bool no_live = false;
    bool enable_thinking = true;
    bool medium_effort = false;
    bool truncate_history_thinking = true;
    bool verbose = false;
    bool seed_set = false;
    bool top_k_set = false;

    bool tokenize_set = false;
    std::string tokenize_text;
    std::string detok_text;
    bool parse_special = true;

    std::string prompt_text;
    std::string prompt_file;
    bool chat_mode = false;
    bool text_out = false;
};

// ---- prompt-lookup (n-gram) drafter: drafts are always verified by the target model,
//      so a hash collision or a bad match can only cost speed, never correctness.
struct NgramDrafter {
    std::unordered_map<unsigned long long, int> tab[12];
    std::vector<int> hist, last;
    int dmax = 7, nmax = 4, nmin = 1;
    int lastg = 0;                 // n-gram length that produced the current draft

    static unsigned long long hash_n(const int* p, int n) {
        unsigned long long k = 1469598103934665603ull;
        for (int i = 0; i < n; ++i) { k ^= (unsigned long long)(unsigned)p[i]; k *= 1099511628211ull; }
        return k;
    }
    void clear() { for (int i = 0; i < 12; ++i) tab[i].clear(); hist.clear(); last.clear(); }
    // append a token, then draft from the most recent earlier occurrence of the suffix
    void push(int t) {
        hist.push_back(t);
        const int n = (int)hist.size();
        last.clear();
        for (int g = nmax; g >= nmin && last.empty(); --g) {
            if (n < g) continue;
            auto it = tab[g].find(hash_n(&hist[n - g], g));   // lookup before inserting this position
            if (it == tab[g].end()) continue;
            const int j = it->second;                          // end position of the previous match
            if (j + 1 >= n) continue;
            lastg = g;
            const int span = n - 1 - j;                        // tokens known to follow the match
            for (int i = 0; i < dmax; ++i) {
                if (j + 1 + i <= n - 1) last.push_back(hist[j + 1 + i]);
                else last.push_back(hist[j + 1 + (i % span)]); // repeat a periodic continuation
            }
        }
        for (int g = nmin; g <= nmax; ++g)
            if (n >= g) tab[g][hash_n(&hist[n - g], g)] = n - 1;
    }
    // Verification rows are near-free on the tensor-core path (+0.06 ms/row),
    // so a short n-gram match no longer has to cap the draft length.
    void draft(std::vector<int>& out) const {
        out.assign(last.begin(), last.begin() + std::min((int)last.size(), dmax));
    }
};

#include "server.h"

static bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", what); exit(2); }
            return argv[++i];
        };
        if (s == "--gguf" || s == "--model" || s == "-m") a.gguf = next("--model");
        else if (s == "--ctx" || s == "--n-ctx" || s == "--ctx-size" || s == "-c") {
            a.ctx = atoi(next("--ctx").c_str());
        }
        else if (s == "--loops") a.loops = atoi(next("--loops").c_str());
        else if (s == "--tokens") a.tokens = next("--tokens");
        else if (s == "--token-file") a.token_file = next("--token-file");
        else if (s == "--gen") a.gen = atoi(next("--gen").c_str());
        else if (s == "--greedy") a.greedy = true;
        else if (s == "--temp") a.temp = (float)atof(next("--temp").c_str());
        else if (s == "--top-p") a.top_p = (float)atof(next("--top-p").c_str());
        else if (s == "--seed") {
            a.seed = strtoull(next("--seed").c_str(), nullptr, 10);
            a.seed_set = true;
        }
        else if (s == "--dump-logits") a.dump_logits = next("--dump-logits");
        else if (s == "--dump-x") { a.dump_x = next("--dump-x"); g_debug = true; }
        else if (s == "--dump-taps") { a.dump_taps = next("--dump-taps"); g_debug = true; }
        else if (s == "--debug-block") g_debug_block = atoi(next("--debug-block").c_str());
        else if (s == "--top-logits") a.top_logits = atoi(next("--top-logits").c_str());
        else if (s == "--bench") a.bench = atoi(next("--bench").c_str());
        else if (s == "--no-graph") a.no_graph = true;
        else if (s == "--gemv-blocks") a.gemv_blocks = atoi(next("--gemv-blocks").c_str());
        else if (s == "--gemm-blocks") a.gemm_blocks = atoi(next("--gemm-blocks").c_str());
        else if (s == "--splits") a.splits = atoi(next("--splits").c_str());
        else if (s == "--splits-full") a.splits_full = atoi(next("--splits-full").c_str());
        else if (s == "--splits-swa") a.splits_swa = atoi(next("--splits-swa").c_str());
        else if (s == "--bench-at") a.bench_at = atoi(next("--bench-at").c_str());
        else if (s == "--bench-rows") a.bench_rows = atoi(next("--bench-rows").c_str());
        else if (s == "--prefill-batch" || s == "--n-batch" || s == "--batch-size" || s == "-b") {
            a.prefill_batch = atoi(next("--prefill-batch").c_str());
        }
        else if (s == "--force-batch") a.force_batch = true;
        else if (s == "--probe") a.probe = true;
        else if (s == "--probe-ms") a.probe_ms = true;
        else if (s == "--bench-ms") a.bench_ms = true;
        else if (s == "--ms-test") a.ms_test = true;
        else if (s == "--print-ids") a.print_ids = true;
        else if (s == "--slots" || s == "--parallel" || s == "-np") a.slots = atoi(next("--slots").c_str());
        else if (s == "--draft") a.draft = atoi(next("--draft").c_str());
        else if (s == "--q5k-prepack") {
            const std::string v = next("--q5k-prepack");
            a.q5k_prepack = (v == "off" || v == "0") ? 0 : ((v == "on" || v == "1") ? 1 : -1);
        }
        else if (s == "--attn-tc") g_attn_tc = atoi(next("--attn-tc").c_str());
        else if (s == "--prefill-tc") g_prefill_tc = atoi(next("--prefill-tc").c_str());
        else if (s == "--adaptive-loops") g_adaptive_loops = 1;
        else if (s == "--no-adaptive-loops") g_adaptive_loops = 0;
        else if (s == "--loop1-reuse") g_l1reuse = 1;
        else if (s == "--no-loop1-reuse") g_l1reuse = 0;
        else if (s == "--loop-tau") g_loop_tau = (float)atof(next("--loop-tau").c_str());
        else if (s == "--occ") {
            CK(cudaFuncSetAttribute((const void*)&k_gemm_mmq_q5k, cudaFuncAttributeMaxDynamicSharedMemorySize, MMQ_SMEM));
            CK(cudaFuncSetAttribute((const void*)&k_gemm_mmq_q5k_144, cudaFuncAttributeMaxDynamicSharedMemorySize, MMQ144_SMEM));
            CK(cudaFuncSetAttribute((const void*)&k_gemm_mma_q6k, cudaFuncAttributeMaxDynamicSharedMemorySize, Q6K_SMEM));
            CK(cudaFuncSetAttribute((const void*)&k_gemm_mma_q6k_128, cudaFuncAttributeMaxDynamicSharedMemorySize, Q6K128_SMEM));
            struct Entry { const char* n; const void* f; int t; int sh; };
            const Entry ks[] = {
                {"k_attn_tc_warp_indep", (const void*)&k_attn_tc_wi<1>, 128, 4 * 16 * TC_STRIDE * 2},
                {"k_attn_tc<1,4>", (const void*)&k_attn_tc<1, 4>,        128, 0},
                {"k_attn_tc<2,4>", (const void*)&k_attn_tc<2, 4>,        128, 0},
                {"k_attn_rs_opt",  (const void*)&k_attn_tc_rs_opt<4, 3>,    128, 2 * TC_T * TC_STRIDE * 2},
                {"k_attn_rs_opt6", (const void*)&k_attn_tc_rs_opt<6, 2>,    192, 2 * TC_T * TC_STRIDE * 2},
                {"k_attn_tc_rs",   (const void*)&k_attn_tc_rs<4>,       256, 0},
                {"k_attn_tc<4,8>", (const void*)&k_attn_tc<4, 8>,        256, 0},
                {"mmq_q5k",        (const void*)&k_gemm_mmq_q5k,         256, MMQ_SMEM},
                {"mmq_q5k_144",    (const void*)&k_gemm_mmq_q5k_144,     256, MMQ144_SMEM},
                {"mmq_q6k",        (const void*)&k_gemm_mma_q6k,         256, Q6K_SMEM},
                {"mmq_q6k_128",    (const void*)&k_gemm_mma_q6k_128,     256, Q6K128_SMEM},
                {"silu_vec4",      (const void*)&k_silu_mul_norm_vec4,   256, 0},
                {"resid_vec4",     (const void*)&k_resid_vec4,           160, 0},
                {"gate_vec4",      (const void*)&k_gate_norm_vec4,       256, 0},
                {"qkv_prep_fast",  (const void*)&k_qkv_prep_fast,        256, 0},
                {"q5k<1,16,8>",   (const void*)&k_gemm_q5k<1, 16, 8>,   512, 0},
                {"q5k<2,16,4>",   (const void*)&k_gemm_q5k<2, 16, 4>,   512, 0},
                {"q5k<3,16,2>",   (const void*)&k_gemm_q5k<3, 16, 2>,   512, 0},
                {"q5k<4,16,4>",   (const void*)&k_gemm_q5k<4, 16, 4>,   512, 0},
                {"q5k<4,16,2>",   (const void*)&k_gemm_q5k<4, 16, 2>,   512, 0},
                {"q5k<8,16,2>",   (const void*)&k_gemm_q5k<8, 16, 2>,   512, 0},
                {"q6k<4,16,8>",   (const void*)&k_gemm_q6k<4, 16, 8>,   512, 0},
                {"bf16<4,8>",     (const void*)&k_gemm_bf16<4, 8>,      512, 0},
            };
            for (const Entry& e : ks) {
                int nb = 0;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, e.f, e.t, e.sh);
                cudaFuncAttributes at{};
                cudaFuncGetAttributes(&at, e.f);
                printf("%-12s %2d blk/SM  %3d regs  %6zu B smem  %2d warps/SM\n",
                       e.n, nb, at.numRegs, at.sharedSizeBytes, nb * (e.t / 32));
            }
            exit(0);
        }
        else if (s == "--draft-ngram") a.draft_ngram = atoi(next("--draft-ngram").c_str());
        else if (s == "--draft-nmin") a.draft_nmin = atoi(next("--draft-nmin").c_str());
        else if (s == "--top-k") { a.top_k = atoi(next("--top-k").c_str()); a.top_k_set = true; }
        else if (s == "--min-p") a.min_p = (float)atof(next("--min-p").c_str());
        else if (s == "--serve" || s == "--server") a.serve = true;
        else if (s == "--host") a.host = next("--host");
        else if (s == "--port") a.port = atoi(next("--port").c_str());
        else if (s == "--alias") a.alias = next("--alias");
        else if (s == "--api-key") a.api_key = next("--api-key");
        else if (s == "--n-predict" || s == "--predict" || s == "-n") {
            a.n_predict = atoi(next("--n-predict").c_str());
        }
        else if (s == "--n-keep" || s == "--keep") a.n_keep = atoi(next("--n-keep").c_str());
        else if (s == "--repeat-penalty" || s == "--repetition-penalty") {
            a.repeat_penalty = (float)atof(next("--repeat-penalty").c_str());
        }
        else if (s == "--repeat-last-n" || s == "--repeat-penalty-last-n" || s == "--penalty-last-n") {
            a.repeat_last_n = atoi(next("--repeat-last-n").c_str());
        }
        else if (s == "--frequency-penalty") a.frequency_penalty = (float)atof(next("--frequency-penalty").c_str());
        else if (s == "--presence-penalty") a.presence_penalty = (float)atof(next("--presence-penalty").c_str());
        else if (s == "--cache-prompt") a.cache_prompt = true;
        else if (s == "--no-cache-prompt") a.cache_prompt = false;
        else if (s == "--medium-effort") a.medium_effort = true;
        else if (s == "--no-medium-effort") a.medium_effort = false;
        else if (s == "--truncate-thinking") a.truncate_history_thinking = true;
        else if (s == "--no-truncate-thinking") a.truncate_history_thinking = false;
        else if (s == "--no-live") a.no_live = true;
        else if (s == "--no-thinking") a.enable_thinking = false;
        else if (s == "--thinking") a.enable_thinking = true;
        else if (s == "--tokenize") { a.tokenize_text = next("--tokenize"); a.tokenize_set = true; }
        else if (s == "--detokenize") a.detok_text = next("--detokenize");
        else if (s == "--parse-special") a.parse_special = atoi(next("--parse-special").c_str()) != 0;
        else if (s == "--prompt-text") a.prompt_text = next("--prompt-text");
        else if (s == "--prompt-file") a.prompt_file = next("--prompt-file");
        else if (s == "--chat") a.chat_mode = true;
        else if (s == "--text") a.text_out = true;
        else if (s == "--jinja" || s == "--no-webui" || s == "--metrics" ||
                 s == "--flash-attn" || s == "-fa" || s == "--no-flash-attn" ||
                 s == "--mlock" || s == "--no-mmap" || s == "--numa" ||
                 s == "--cont-batching" || s == "-cb" || s == "--no-cont-batching" ||
                 s == "--no-warmup" || s == "--no-context-shift" || s == "--context-shift" ||
                 s == "--no-display-prompt" || s == "--simple-io" || s == "--escape" ||
                 s == "--embeddings" || s == "--embedding" || s == "--reranking" ||
                 s == "--op-offload" || s == "--no-op-offload" || s == "--list-devices" ||
                 s == "--log-colors" || s == "--log-disable" || s == "--no-perf") {
            /* accepted for llama.cpp-style launches; this engine has its own backend */
        }
        else if (s == "--threads" || s == "-t" || s == "--threads-batch" ||
                 s == "--gpu-layers" || s == "-ngl" || s == "--ubatch-size" || s == "-ub" ||
                 s == "--split-mode" || s == "--main-gpu" || s == "--tensor-split" ||
                 s == "--cache-type-k" || s == "--cache-type-v" || s == "--timeout" ||
                 s == "--threads-http" || s == "--slot-save-path" || s == "--log-file" ||
                 s == "--reasoning-format" || s == "--reasoning-budget" || s == "--prio" ||
                 s == "--device" || s == "--rope-scaling" || s == "--rope-freq-base" ||
                 s == "--rope-freq-scale" || s == "--defrag-thold" || s == "--lora" ||
                 s == "--lora-scaled" || s == "--control-vector" || s == "--control-vector-scaled" ||
                 s == "--override-kv" || s == "--chat-template" || s == "--chat-template-file" ||
                 s == "--pooling") {
            (void)next(s.c_str());
        }
        else if (s == "--parallel" || s == "-np") {
            (void)next(s.c_str());
            log_line("[bruce] --parallel is accepted but this engine serves one sequence at a time");
        }
        else if (s == "-v" || s == "--verbose") { a.verbose = true; g_verbose = true; }
        else if (s == "-h" || s == "--help") return false;
        else { fprintf(stderr, "unknown arg %s\n", s.c_str()); return false; }
    }
    return true;
}

static std::vector<int> load_token_ids(const Args& a) {
    std::vector<int> ids;
    std::string text = a.tokens;
    if (text.empty() && !a.token_file.empty()) {
        FILE* f = fopen(a.token_file.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", a.token_file.c_str()); exit(1); }
        char buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
        fclose(f);
    }
    if (!text.empty()) {
        const char* p = text.c_str();
        while (*p) {
            char* e = nullptr;
            long v = strtol(p, &e, 10);
            if (e == p) break;
            ids.push_back((int)v);
            p = e;
            while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') ++p;
        }
    }
    return ids;
}

static void dump_taps_now(const std::string& path) {
    std::vector<float> buf(TAP_N);
    CK(cudaMemcpy(buf.data(), M.dbg_taps, TAP_N * 4, cudaMemcpyDeviceToHost));
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); return; }
    fwrite(buf.data(), 4, buf.size(), f);
    fclose(f);
    fprintf(stderr, "[bruce] dumped taps for block %d to %s\n", g_debug_block, path.c_str());
    if (g_debug_block >= 0 && !M.blocks[g_debug_block].is_ffn) {
        const size_t nb = (size_t)4 * KV_POS_BYTES;
        const int pstride = M.blocks[g_debug_block].full ? M.slot_ctx : SWA_SLOTS;
        std::vector<uint8_t> kbuf(nb);
        for (int pass = 0; pass < 2; ++pass) {
            const uint8_t* src = pass ? M.kvv_ptr[g_debug_block] : M.kvk_ptr[g_debug_block];
            for (int h = 0; h < N_KV; ++h)
                for (int sl = 0; sl < 4; ++sl)
                    CK(cudaMemcpy(kbuf.data() + (size_t)(sl * N_KV + h) * KV_KVH_BYTES,
                                  src + (size_t)h * pstride * KV_KVH_BYTES + (size_t)sl * KV_KVH_BYTES,
                                  KV_KVH_BYTES, cudaMemcpyDeviceToHost));
            FILE* f = fopen((path + (pass ? ".v" : ".k")).c_str(), "wb");
            fwrite(kbuf.data(), 1, nb, f); fclose(f);
        }
    }
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        fprintf(stderr,
                "usage: bruce --gguf PATH [--tokens ids] [--gen N] [--greedy] [--temp T] [--top-p P]\n"
                "       (model default: $BRUCE_GGUF, else NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf)\n"
                "            [--prompt-text TEXT [--chat] [--text]] [--tokenize TEXT] [--detokenize IDS]\n"
                "            [--dump-logits FILE] [--bench N] [--ctx N]\n"
                "\n"
                "server: bruce --model MODEL.gguf --serve [options]\n"
                "  --host H --port P --alias NAME --api-key KEY\n"
                "  --ctx N | --n-ctx N | -c N        KV pool size (default 262144)\n"
                "  --slots N | --parallel N          concurrent sequences (default 4 when ctx >= 16384, else 1; ctx/N each)\n"
                "  --n-batch N | --batch-size N      prompt chunk size (default %d)\n"
                "  --n-predict N | -n N              default max tokens (default 256)\n"
                "  --temp T --top-p P --top-k K --min-p M --seed S\n"
                "  --repeat-penalty R --repeat-last-n N\n"
                "  --frequency-penalty F --presence-penalty P\n"
                "  --thinking | --no-thinking        enable/disable reasoning prompt\n"
                "  --medium-effort                   append the efficient-effort hint\n"
                "  --no-truncate-thinking            keep old reasoning in chat history\n"
                "  --cache-prompt | --no-cache-prompt\n"
                "  --draft N --draft-ngram N --draft-nmin N   prompt-lookup speculation\n"
                "  --q5k-prepack auto|on|off         load-time Q5_K pre-unpack (default auto, ctx<=32K)\n"
                "  --adaptive-loops [--loop-tau T]   skip loop 1 when the loop-0 fixed point converged\n"
                "  --loop1-reuse | --no-loop1-reuse  reuse loop-0 attention in loop 1 (decode only)\n"
                "  --no-live                         disable live progress lines\n",
                MBUF);
        return 1;
    }
    init_live_log();
    g_no_live = args.no_live;
    double t0 = now_ms();
    GgufFile g;
    {
        double tq = now_ms();
        g.open(args.gguf.c_str());
        g_lp_open = now_ms() - tq;
    }
    if (g.str("general.architecture") != "dory") { fprintf(stderr, "not a dory model\n"); return 1; }
    if ((int)g.i("dory.embedding_length") != N_EMBD || (int)g.i("dory.feed_forward_length") != N_FF ||
        (int)g.i("dory.attention.head_count") != N_HEAD || (int)g.i("dory.attention.head_count_kv") != N_KV ||
        (int)g.i("dory.attention.key_length") != HD || (int)g.i("dory.attention.sliding_window") != SWA ||
        (int)g.i("dory.input_layer_count") != N_INPUT || (int)g.i("dory.recurrent_layer_count") != N_REC ||
        (int)g.i("dory.output_layer_count") != N_OUT) {
        fprintf(stderr, "unexpected model dimensions\n"); return 1;
    }
    if (args.loops < 1 || args.loops > 4) { fprintf(stderr, "loops must be 1..4\n"); return 1; }
    if (args.ctx < 1 || args.ctx > (int)g.i("dory.context_length")) { fprintf(stderr, "bad ctx\n"); return 1; }
    args.prefill_batch = std::max(1, std::min(MBUF, args.prefill_batch));
    args.bench_rows = std::max(1, std::min(args.probe ? MBUF : 8, args.bench_rows));

    Tokenizer tokenizer;
    {
        double tq = now_ms();
        tokenizer.init(g);
        g_lp_tok = now_ms() - tq;
    }
    // From here on the model bytes come through the streamed loader's own file
    // handle (no mapping stays alive: a mapped file forces ReadFile back to disk
    // speed instead of the OS cache).
    std::string model_display = g.str("general.name", "Dory 1.0 4B");
    log_line("[bruce] tokenizer: %d tokens, %zu merges, %zu special, eos=%d",
             tokenizer.n_vocab, tokenizer.bpe_rank.size(), tokenizer.special_ids.size(), tokenizer.eos_id);

    auto parse_ids = [](const std::string& text) {
        std::vector<int> out;
        const char* p = text.c_str();
        while (*p) {
            char* e = nullptr;
            long v = strtol(p, &e, 10);
            if (e == p) break;
            out.push_back((int)v);
            p = e;
            while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') ++p;
        }
        return out;
    };

    if (args.tokenize_set) {
        std::vector<int> ids = tokenizer.encode(args.tokenize_text, false, args.parse_special);
        if (args.print_ids) {
            for (int id : ids) printf("%d\n", id);
        } else {
            for (size_t i = 0; i < ids.size(); ++i) printf("%s%d", i ? " " : "", ids[i]);
            printf("\n");
        }
        return 0;
    }
    if (!args.detok_text.empty()) {
        std::vector<int> ids = parse_ids(args.detok_text);
        printf("%s\n", tokenizer.decode(ids, false).c_str());
        return 0;
    }

    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    log_line("[bruce] GPU: %s | sm_%d%d | %.1f GiB | CUDA %d.%d",
             prop.name, prop.major, prop.minor, prop.totalGlobalMem / 1073741824.0,
             CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    log_line("[bruce] decode: loop-1 attention reuse %s | %s",
             l1reuse() ? "on (5 recurrent full blocks)" : "off",
             g_adaptive_loops ? "adaptive loops" : "2 loops");

    M.n_ctx = args.ctx;
    // Multi-slot serving is on by default when the pool is large enough to give
    // every slot a usable window; small-context servers stay single-slot.
    const int auto_slots = (args.serve && args.ctx >= 16384) ? 4 : 1;
    M.n_slots = std::max(1, std::min(MAX_SLOTS, args.slots > 0 ? args.slots : auto_slots));
    log_line("[bruce] model: %s | ctx %d (%d/slot x %d) | loops %d | q8_0 last-loop KV",
             args.gguf.c_str(), args.ctx, args.ctx / M.n_slots, M.n_slots, args.loops);
    if (M.n_slots > 1 && g_adaptive_loops) {
        g_adaptive_loops = 0;
        log_line("[bruce] --adaptive-loops is single-slot only; disabled with --slots %d", M.n_slots);
    }
    if (M.n_slots > 1 && args.draft > 0) {
        log_line("[bruce] speculative decode stays enabled per slot when it is the only active request");
    }
    g_q5k_prepack = args.q5k_prepack;
    { const char* e = getenv("BRUCE_ATTN_NW_DEPTH"); if (e) g_full_nw_depth = atoi(e); }
    CK(cudaStreamCreateWithFlags(&S, cudaStreamNonBlocking));

    bind_weights(g, args.loops);
    {
        double tq = now_ms();
        alloc_buffers();
        g_lp_alloc = now_ms() - tq;
    }
    set_batched_attrs();
    print_mem(t0);
    g.close();
    if (getenv("BRUCE_LOAD_PROF")) {
        log_line("[load-prof] open %.2f tok %.2f | loop %.2f pread %.2f slotwait %.2f scale %.2f "
                 "prepack %.2f drain %.2f alloc %.2f | xf %.2f rows %lld | total %.2f s",
                 g_lp_open / 1000, g_lp_tok / 1000, g_lp_h2d / 1000, g_ld.t_read / 1000,
                 g_ld.t_wait / 1000, g_lp_scale / 1000, g_lp_prepack / 1000, g_lp_drain / 1000,
                 g_lp_alloc / 1000, g_ld.t_xf.load() / 1000, g_ld.n_xf.load(),
                 (now_ms() - t0) / 1000);
    }

    CK(cudaMemcpy(M.d_rng, &args.seed, 8, cudaMemcpyHostToDevice));

    // warmup
    set_tok_pos(1, 0);
    forward_launch(S);
    forward_launch(S);
    CK(cudaStreamSynchronize(S));

    if (args.ms_test) {
        ms_selftest(args.slots);
        return 0;
    }

    std::vector<int> ids = load_token_ids(args);
    if (!args.prompt_file.empty()) {
        FILE* f = fopen(args.prompt_file.c_str(), "rb");
        if (!f) { log_line("[bruce] cannot open %s", args.prompt_file.c_str()); return 1; }
        std::string text;
        char buf[8192];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
        fclose(f);
        args.prompt_text = std::move(text);
    }
    if (!args.prompt_text.empty()) {
        std::string text = args.prompt_text;
        if (args.chat_mode) text = simple_chat_prompt(text, args.enable_thinking);
        ids = tokenizer.encode(text, false, true);
    }
    if ((int)ids.size() >= M.slot_ctx) {
        log_line("[bruce] prompt %zu tokens >= per-slot context %d (use --slots 1 or a larger --ctx)",
                 ids.size(), M.slot_ctx);
        return 1;
    }
    if (!ids.empty()) log_line("[bruce] prompt: %zu tokens", ids.size());
    g_gemv_blocks = std::max(8, args.gemv_blocks);
    g_gemm_blocks = args.gemm_blocks;
    {   // lazy-V tile-skip threshold (nats below the running max); dev sweep knob
        const char* e = getenv("BRUCE_LAZY_THR");
        const float thr = e ? (float)atof(e) : 14.f;
        CK(cudaMemcpyToSymbol(g_lazy_thr, &thr, sizeof(float)));
    }
    if (args.ctx >= 65536 && args.splits == 0 && args.splits_full == 0) M.full_splits = 128;
    if (args.splits > 0) { M.full_splits = args.splits; M.swa_splits = args.splits; g_splits_explicit = true; }
    if (args.splits_full > 0 || args.splits_swa > 0) g_splits_explicit = true;
    if (args.splits_full > 0) M.full_splits = args.splits_full;
    if (args.splits_swa > 0) M.swa_splits = args.splits_swa;
    M.full_splits = std::max(1, std::min(PART_MAX_SPLITS, M.full_splits));
    M.swa_splits = std::max(1, std::min(PART_MAX_SPLITS, M.swa_splits));
    if (args.probe || args.probe_ms) {
        probe_mode(args.bench_at > 0 ? args.bench_at : 4096, args.bench > 0 ? args.bench : 20,
                   args.bench_rows, args.probe_ms);
        return 0;
    }
    if (args.bench_ms) {
        const int rows = std::max(1, std::min(args.bench_rows, M.n_slots));
        const int reps = args.bench > 0 ? args.bench : 20;
        const int p0 = std::min(args.bench_at, std::max(0, M.slot_ctx - rows));
        int rk[MAX_ROWS], rp[MAX_ROWS], rs[MAX_ROWS], ra[MAX_ROWS];
        for (int i = 0; i < rows; ++i) { rk[i] = 1000 + i; rp[i] = p0; rs[i] = i; ra[i] = 1; }
        CK(cudaMemcpyAsync(M.bm_tok, rk, sizeof(int) * rows, cudaMemcpyHostToDevice, S));
        CK(cudaMemcpyAsync(M.d_row_pos, rp, sizeof(int) * rows, cudaMemcpyHostToDevice, S));
        CK(cudaMemcpyAsync(M.d_row_slot, rs, sizeof(int) * rows, cudaMemcpyHostToDevice, S));
        CK(cudaMemcpyAsync(M.d_row_act, ra, sizeof(int) * rows, cudaMemcpyHostToDevice, S));
        auto step = [&]() {
            forward_batch_ms(rows, true, true, S);
            k_argmax_rows<<<rows, 1024, 0, S>>>(M.d_amax, M.bm_logits, VOCAB, rows);
        };
        for (int i = 0; i < 3; ++i) step();
        CK(cudaStreamSynchronize(S));
        cudaEvent_t b0, b1;
        CK(cudaEventCreate(&b0)); CK(cudaEventCreate(&b1));
        CK(cudaEventRecord(b0, S));
        for (int i = 0; i < reps; ++i) step();
        CK(cudaEventRecord(b1, S));
        CK(cudaEventSynchronize(b1));
        float bms = 0.f;
        CK(cudaEventElapsedTime(&bms, b0, b1));
        const double per = bms / reps;
        printf("bench-ms: %d rows at pos %d, %d steps in %.2f ms, %.3f ms/step, %.1f tok/s aggregate\n",
               rows, p0, reps, bms, per, rows * 1000.0 / per);
        CK(cudaEventDestroy(b0)); CK(cudaEventDestroy(b1));
        return 0;
    }
    if (!args.no_graph && !g_debug) capture_verify_graphs();

    if (args.bench > 0) {
        if (args.bench_at > 0) {
            const size_t fbytes = (size_t)9 * M.n_slots * M.slot_ctx * KV_POS_BYTES;
            const size_t sbytes = (size_t)27 * M.n_slots * SWA_SLOTS * KV_POS_BYTES;
            CK(cudaMemset(M.kvf_k, 0, fbytes));
            CK(cudaMemset(M.kvf_v, 0, fbytes));
            CK(cudaMemset(M.kvs_k, 0, sbytes));
            CK(cudaMemset(M.kvs_v, 0, sbytes));
        }
        if (args.bench_rows > 1) {
            // Time the batched forward the way a speculative round uses it: `rows` tokens
            // verified in one pass, then an argmax + host sync (as the accept check does).
            const int rows = std::min(MBUF, args.bench_rows);
            int* d_amax = nullptr;
            CK(cudaMalloc(&d_amax, sizeof(int) * (MBUF + 1)));
            int rtok[MBUF + 1], amax[MBUF + 1];
            for (int i = 0; i < rows; ++i) rtok[i] = 1000 + i;
            CK(cudaMemcpyAsync(M.bm_tok, rtok, sizeof(int) * rows, cudaMemcpyHostToDevice, S));
            int p0 = std::min(args.bench_at, std::max(0, M.slot_ctx - rows));
            auto step = [&]() {
                CK(cudaMemcpyAsync(M.d_pos, &p0, 4, cudaMemcpyHostToDevice, S));
                verify_launch(rows, S);
                CK(cudaMemcpyAsync(amax, M.d_amax, sizeof(int) * rows, cudaMemcpyDeviceToHost, S));
                CK(cudaStreamSynchronize(S));
                verify_update();
            };
            for (int i = 0; i < 3; ++i) step();
            cudaEvent_t b0, b1;
            CK(cudaEventCreate(&b0)); CK(cudaEventCreate(&b1));
            CK(cudaEventRecord(b0, S));
            for (int i = 0; i < args.bench; ++i) step();
            CK(cudaEventRecord(b1, S));
            CK(cudaEventSynchronize(b1));
            float bms = 0.f;
            CK(cudaEventElapsedTime(&bms, b0, b1));
            double per = bms / args.bench;
            printf("bench-rows %d: %d rounds in %.2f ms, %.3f ms/round, %.2f row-passes/s\n",
                   rows, args.bench, bms, per, 1000.0 / per);
            cudaFree(d_amax);
            return 0;
        }
        capture_graph(&M.ex_fwd_adv, true);
        // The bench graph advances the position counter on every replay, so
        // leave room for the warm-up plus all timed replays: writing KV past
        // n_ctx used to land in free VRAM and now faults.
        const int warm = 3;
        int p0 = std::min(args.bench_at, std::max(0, M.slot_ctx - 1 - warm - args.bench));
        set_tok_pos(1, p0);
        CK(cudaStreamSynchronize(S));
        cudaEvent_t ev0, ev1;
        CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
        for (int i = 0; i < warm; ++i) CK(cudaGraphLaunch(M.ex_fwd_adv, S));
        CK(cudaStreamSynchronize(S));
        CK(cudaEventRecord(ev0, S));
        for (int i = 0; i < args.bench; ++i) CK(cudaGraphLaunch(M.ex_fwd_adv, S));
        CK(cudaEventRecord(ev1, S));
        CK(cudaEventSynchronize(ev1));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, ev0, ev1));
        double per = ms / args.bench;
        double wbytes = (double)(g_weight_bytes - g_prepack_bytes);
        printf("bench: %d steps in %.2f ms, %.3f ms/token, %.2f tok/s, weights %.2f GB/s\n",
               args.bench, ms, per, 1000.0 / per, wbytes / (per * 1e6));
        return 0;
    }

    if (!args.no_graph && !g_debug) {
        capture_graph(&M.ex_fwd, false);
        if (g_adaptive_loops) capture_graph(&M.ex_fwd1, false, true);
        if (M.n_slots > 1) {
            // one M=1 graph per slot (the KV base pointers are baked into the graph),
            // plus the shared multi-row batch graphs
            for (int s = 0; s < M.n_slots; ++s) {
                set_cur_slot(s);
                capture_graph(&M.ex_fwd_slot[s], false);
            }
            set_cur_slot(0);
            capture_ms_graphs();
        }
    }

    if (args.serve) {
        ServerConfig sc;
        sc.host = args.host;
        sc.port = args.port;
        sc.model_id = args.alias;
        sc.model_path = args.gguf;
        sc.model_name = model_display;
        sc.api_key = args.api_key;
        sc.n_ctx = args.ctx;
        sc.n_ctx_slot = M.slot_ctx;
        sc.slots = M.n_slots;
        sc.n_predict = args.n_predict;
        sc.n_batch = args.prefill_batch;
        sc.n_loops = args.loops;
        sc.n_keep = args.n_keep;
        sc.draft = args.draft;
        sc.draft_ngram = args.draft_ngram;
        sc.draft_nmin = args.draft_nmin;
        sc.greedy = args.greedy || args.temp <= 0.0f;
        sc.no_live = args.no_live;
        sc.verbose = args.verbose;
        sc.enable_thinking = args.enable_thinking;
        sc.medium_effort = args.medium_effort;
        sc.truncate_history_thinking = args.truncate_history_thinking;
        sc.default_cache_prompt = args.cache_prompt;
        sc.default_temp = args.temp;
        sc.default_top_p = args.top_p;
        sc.default_top_k = args.top_k_set ? args.top_k : 40;
        sc.default_min_p = args.min_p;
        sc.default_repeat_penalty = args.repeat_penalty;
        sc.default_repeat_last_n = args.repeat_last_n;
        sc.default_frequency_penalty = args.frequency_penalty;
        sc.default_presence_penalty = args.presence_penalty;
        sc.default_seed = args.seed;
        sc.seed_set = args.seed_set;
        return run_server(tokenizer, sc);
    }

    // prefill
    double tp = now_ms();
    double t_last_log = tp;
    int last_rows = 0;
    // Row-split attention is now 2 blocks/SM, so it wins at every prompt depth.
    g_attn_rs = 1;
    for (size_t i = 0; i < ids.size();) {
        size_t left = ids.size() - i;
        int n = args.no_graph ? 1 : (int)std::min((size_t)args.prefill_batch, left);
        if (n <= 1 && !args.force_batch) {
            set_tok_pos(ids[i], (int)i);
            if (!args.no_graph && !g_debug) CK(cudaGraphLaunch(M.ex_fwd, S));
            else forward_launch(S);
            last_rows = 0;
            i += 1;
        } else {
            CK(cudaMemcpyAsync(M.bm_tok, ids.data() + i, (size_t)n * 4, cudaMemcpyHostToDevice, S));
            int p0 = (int)i;
            g_batch_pos0 = p0;
            CK(cudaMemcpyAsync(M.d_pos, &p0, 4, cudaMemcpyHostToDevice, S));
            forward_batch(n, false, i + n >= ids.size(), S);
            last_rows = n;
            i += n;
        }
        double now = now_ms();
        if (i < ids.size() && (g_tty || now - t_last_log > 500.0)) {
            double el = now - tp;
            status_line("[bruce] prefill %zu/%zu | %.0f tok/s | %.2fs | eta %.1fs",
                        i, ids.size(), el > 1.0 ? i * 1000.0 / el : 0.0, el / 1000.0,
                        el > 1.0 ? (ids.size() - i) / (i / el) / 1000.0 : 0.0);
            t_last_log = now;
        }
    }
    CK(cudaStreamSynchronize(S));
    status_end();
    fprintf(stderr, "[bruce] prefill %zu tokens in %.2f ms (%.2f tok/s)\n",
            ids.size(), now_ms() - tp, ids.empty() ? 0.0 : ids.size() * 1000.0 / (now_ms() - tp));
    int pos = (int)ids.size();
    CK(cudaMemcpyAsync(M.d_pos, &pos, 4, cudaMemcpyHostToDevice, S));
    CK(cudaStreamSynchronize(S));

    if (!args.dump_logits.empty()) {
        std::vector<float> lg(VOCAB);
        CK(cudaMemcpy(lg.data(), M.logits, VOCAB * 4, cudaMemcpyDeviceToHost));
        FILE* f = fopen(args.dump_logits.c_str(), "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", args.dump_logits.c_str()); return 1; }
        fwrite(lg.data(), 4, VOCAB, f);
        fclose(f);
        fprintf(stderr, "[bruce] dumped logits to %s\n", args.dump_logits.c_str());
    }
    if (!args.dump_x.empty()) {
        std::vector<float> buf((size_t)M.n_virtual * N_EMBD);
        CK(cudaMemcpy(buf.data(), M.dbg_x, buf.size() * 4, cudaMemcpyDeviceToHost));
        FILE* f = fopen(args.dump_x.c_str(), "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", args.dump_x.c_str()); return 1; }
        fwrite(buf.data(), 4, buf.size(), f);
        fclose(f);
        fprintf(stderr, "[bruce] dumped %d layer activations to %s\n", M.n_virtual, args.dump_x.c_str());
    }
    if (!args.dump_taps.empty() && g_debug_block >= 0) dump_taps_now(args.dump_taps);
    if (args.top_logits > 0) {
        std::vector<float> lg(VOCAB);
        CK(cudaMemcpy(lg.data(), M.logits, VOCAB * 4, cudaMemcpyDeviceToHost));
        std::vector<int> ord(VOCAB);
        for (int i = 0; i < VOCAB; ++i) ord[i] = i;
        int k = std::min(args.top_logits, VOCAB);
        std::partial_sort(ord.begin(), ord.begin() + k, ord.end(),
                          [&](int a, int b) { return lg[a] > lg[b]; });
        for (int i = 0; i < k; ++i) printf("%d %.6f\n", ord[i], lg[ord[i]]);
    }
    if (args.gen <= 0) return 0;

    // prompt-lookup (n-gram) speculative decoding with target-model verification.
    // A draft token is only kept when the target model's own argmax agrees, otherwise
    // the target token is emitted instead, so the output is always a sequence of target
    // argmaxes.  Note that the target argmax is evaluated on a *chunk* of rows, which in
    // the reference `last_loop` KV mode sees the in-chunk positions through the earlier
    // loop's KV (llama.cpp has the same property with n_ubatch > 1); near-ties can
    // therefore resolve differently than strictly sequential (batch-of-1) decoding.
    const bool spec = args.draft > 0 && (args.greedy || args.temp <= 0.f);
    NgramDrafter dr;
    if (spec) {
        dr.dmax = std::max(1, std::min(MBUF - 1, args.draft));
        dr.nmax = std::max(2, std::min(11, args.draft_ngram));
        dr.nmin = std::max(1, std::min(dr.nmax, args.draft_nmin < 0 ? 2 : args.draft_nmin));
        for (int tk : ids) dr.push(tk);
    }
    int* d_amax = nullptr;
    if (spec) CK(cudaMalloc(&d_amax, sizeof(int) * (MBUF + 1)));

    sample_tokens_ex(args.greedy, args.temp, args.top_p, args.top_k, args.min_p);
    int t = get_tok();
    int pos_now = (int)ids.size();
    int produced = 1;
    Utf8Carry cli_carry;
    auto emit = [&](int tok) {
        if (args.print_ids) {
            printf("%d\n", tok);
        } else if (args.text_out) {
            std::string out;
            cli_carry.feed(tokenizer.decode_token(tok, false), out);
            if (!out.empty()) { fwrite(out.data(), 1, out.size(), stdout); fflush(stdout); }
        } else {
            printf("%d ", tok);
            fflush(stdout);
        }
        if (spec) dr.push(tok);
    };
    emit(t);
    std::vector<int> draft;
    int kmax = spec ? std::min(dr.dmax, 4) : 0;   // adaptive speculation depth
    long spec_rounds = 0, spec_tokens = 0;
    long acc_hist[9] = {0};
    long k_hist[9] = {0};
    const double tg0 = now_ms();
    double t_last_status = tg0;
    auto progress = [&]() {
        double now = now_ms();
        if (g_tty || now - t_last_status > 500.0) {
            double el = now - tg0;
            double tps = el > 1.0 ? produced * 1000.0 / el : 0.0;
            status_line("[bruce] decode %d/%d | %.1f tok/s | ctx %d/%d | %.2f ms/token | eta %.1fs",
                        produced, args.gen, tps, pos_now, args.ctx,
                        produced > 0 ? el / produced : 0.0,
                        tps > 1.0 ? (args.gen - produced) / tps : 0.0);
            t_last_status = now;
        }
    };
    progress();
    while (produced < args.gen) {
        draft.clear();
        if (spec) dr.draft(draft);
        int k = (int)draft.size();
        if (k > kmax) k = kmax;
        if (k > MBUF - 1) k = MBUF - 1;
        if (k > args.gen - produced - 1) k = args.gen - produced - 1;
        if (k <= 0) {
            set_tok_pos(t, pos_now);
            if (!args.no_graph && !g_debug) launch_decode_graph(S);
            else forward_launch(S);
            pos_now += 1;
            sample_tokens_ex(args.greedy, args.temp, args.top_p, args.top_k, args.min_p);
            t = get_tok();
            emit(t);
            produced += 1;
            progress();
            continue;
        }
        int rtok[MBUF + 1];
        rtok[0] = t;
        for (int i = 0; i < k; ++i) rtok[i + 1] = draft[i];
        CK(cudaMemcpyAsync(M.bm_tok, rtok, sizeof(int) * (k + 1), cudaMemcpyHostToDevice, S));
        int p0 = pos_now;
        CK(cudaMemcpyAsync(M.d_pos, &p0, 4, cudaMemcpyHostToDevice, S));
        verify_launch(k + 1, S);
        int amax[MBUF + 1];
        CK(cudaMemcpyAsync(amax, M.d_amax, sizeof(int) * (k + 1), cudaMemcpyDeviceToHost, S));
        CK(cudaStreamSynchronize(S));
        verify_update();
        int j = k;
        for (int i = 0; i < k; ++i) if (amax[i] != draft[i]) { j = i; break; }
        for (int i = 0; i < j; ++i) { emit(draft[i]); produced += 1; }
        t = amax[j];
        emit(t);
        produced += 1;
        pos_now = pos_now + j + 1;
        spec_rounds += 1;
        spec_tokens += j + 1;
        acc_hist[j] += 1;
        k_hist[k] += 1;
        if (j == k) kmax = std::min(dr.dmax, kmax + 1);
        else if (j == 0) kmax = std::max(1, kmax - 1);
        CK(cudaMemcpyAsync(M.d_pos, &pos_now, 4, cudaMemcpyHostToDevice, S));
        CK(cudaStreamSynchronize(S));
        progress();
    }
    status_end();
    if (spec) {
        fprintf(stderr, "[bruce] spec: %ld rounds, %ld tokens, %.2f tokens/round; accepted ",
                spec_rounds, spec_tokens,
                spec_rounds ? (double)spec_tokens / (double)spec_rounds : 0.0);
        for (int i = 0; i <= 7; ++i) fprintf(stderr, "%ld ", acc_hist[i]);
        fprintf(stderr, "; draftlen ");
        for (int i = 0; i <= 7; ++i) fprintf(stderr, "%ld ", k_hist[i]);
        fprintf(stderr, "\n");
        cudaFree(d_amax);
    }
    CK(cudaStreamSynchronize(S));
    const double tgen = now_ms() - tg0;
    fprintf(stderr, "[bruce] gen %d tokens in %.1f ms (%.2f tok/s)\n", produced, tgen,
            produced * 1000.0 / tgen);
    if (args.text_out && !args.print_ids) {
        std::string out;
        cli_carry.flush(out);
        if (!out.empty()) fwrite(out.data(), 1, out.size(), stdout);
    }
    if (!args.print_ids) printf("\n");
    if (!args.dump_logits.empty() && args.gen > 0) {
        std::vector<float> lg(VOCAB);
        CK(cudaMemcpy(lg.data(), M.logits, VOCAB * 4, cudaMemcpyDeviceToHost));
        FILE* fl = fopen((args.dump_logits + ".late").c_str(), "wb");
        if (fl) { fwrite(lg.data(), 4, VOCAB, fl); fclose(fl); }
        fprintf(stderr, "[bruce] dumped late logits to %s.late\n", args.dump_logits.c_str());
    }
    if (!args.dump_x.empty() && args.gen > 0) {
        std::vector<float> buf((size_t)M.n_virtual * N_EMBD);
        CK(cudaMemcpy(buf.data(), M.dbg_x, buf.size() * 4, cudaMemcpyDeviceToHost));
        FILE* fx = fopen((args.dump_x + ".late").c_str(), "wb");
        if (fx) { fwrite(buf.data(), 4, buf.size(), fx); fclose(fx); }
        fprintf(stderr, "[bruce] dumped late layer activations to %s.late\n", args.dump_x.c_str());
    }
    if (!args.dump_taps.empty() && g_debug_block >= 0 && args.gen > 0)
        dump_taps_now(args.dump_taps + ".late");
    return 0;
}
