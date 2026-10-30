// Bruce inference engine for Dory 1.0 4B.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#define N_EMBD 2560
#define N_HEAD 16
#define N_KV   4
#define HD     256
#define N_FF   7168
#define VOCAB  131072
#define SWA    1024
#define QK_K   256
#define QK8_0  32

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)

struct TensorW {
    const void* d = nullptr;
    uint32_t type = 0;
    int K = 0, N = 0;
};

// One attention or FFN block of the physical stack (72 total).
struct BlockW {
    int is_ffn = 0;
    int full = 0;          // full attention (else sliding window)
    int theta_sel = 1;     // 1 => 1e7, 2 => 1e4
    TensorW wq, wk, wv, wo, wg;      // attn
    TensorW wfg, wfu, wfd;           // ffn
    const float* sqk = nullptr;      // [4096] pre-scaled, squared
    const float* sgate = nullptr;    // [4096]
    const float* suv_gate = nullptr; // [7168]
    const float* suv_up = nullptr;   // [7168]
    const float* alpha = nullptr;    // [2560]
    __nv_bfloat16* alpha_src = nullptr;
};

struct Model {
    // weights
    const __nv_bfloat16* emb = nullptr;
    const __nv_bfloat16* out_w = nullptr;
    const float* out_scale = nullptr;  // [VOCAB] pre-scaled
    BlockW blocks[72];
    int order[112];        // virtual block -> physical block
    int n_virtual = 112;
    int n_loops = 2;
    // kv
    uint8_t* kv_full_k = nullptr;   // [9][ctx][1088]
    uint8_t* kv_full_v = nullptr;
    uint16_t* kv_swa_k = nullptr;   // [27][1024][1024]
    uint16_t* kv_swa_v = nullptr;
    int kv_full_idx[72];  // per physical block -> full-layer slot or -1
    int kv_swa_idx[72];
    int ctx = 0;
    // scratch
    float* x = nullptr;       // [2560]
    float* upd = nullptr;     // [2560]
    float* q = nullptr;       // [4096]
    float* k = nullptr;       // [1024]
    float* v = nullptr;       // [1024]
    float* gate = nullptr;    // [4096]
    float* aout = nullptr;    // [4096]
    float* ffn_g = nullptr;   // [7168]
    float* ffn_u = nullptr;   // [7168]
    float* logits = nullptr;  // [VOCAB]
    float* rope_cos = nullptr; // [2][128]
    float* rope_sin = nullptr;
    float* attn_part = nullptr; // split-k partials
    int* d_pos = nullptr;
    int* d_tok = nullptr;
    double* d_rand = nullptr;
    int* d_out_tokens = nullptr;
    int  out_tokens_cap = 0;
};

// kernels.cu
void bruce_init(Model& m, int ctx, int n_loops);
void bruce_upload(Model& m, const void* gguf_base, uint64_t data_offset, const void* tensors_desc);
void bruce_forward_stage(Model& m, int stage, int pos);
