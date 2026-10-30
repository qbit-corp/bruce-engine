# Bruce Prefill Optimization Analysis & Roadmap: 700–830 tok/s $\to$ 1,500–1,700+ tok/s on RTX 3060

> **Implementation status (2026-10-24): implemented.** All four levers are in
> `engine/src/bruce.cu`. Measured on this RTX 3060 (real-code corpus): **4K 2047,
> 8K 1950, 32K 1613, 128K 954 tok/s**. Decode is unchanged (32K: 41.96 vs 41.92
> tok/s, 4K: 55.57 vs 55.55). README §4 has the final kernel breakdown.

This document presents a ground-truth, hardware-verified analysis of `engine/src/bruce.cu` targeting **NVIDIA-Nemotron-Labs-Dory1.0-4B (Q5_K_M)** on a single **RTX 3060 12 GB (GA106, SM86, 28 SMs)**, with concrete, Drop-In C++/CUDA designs to push prefill throughput past **1,500 tok/s ($\le 666.7\ \mu\text{s/token}$)** at 32K context and **>2,000 tok/s** at 4K–8K context **without touching single-token decode or speculative verification speed**.

Every bottleneck diagnosis and speedup figure below is backed by:
1. Existing CUPTI profiler traces (`tmp/prof8kb.sqlite`, `tmp/prof32k.sqlite`),
2. Full SASS disassembly (`cuobjdump --dump-sass engine/bruce.exe`), and
3. Standalone microbenchmarks compiled with `nvcc 13.3 -O3 -arch=sm_86` and timed directly on this RTX 3060.

---

## 1. Executive Summary & Ground-Truth Time Budget

### 1.1 Where Every Microsecond Goes Today

Across the **112 virtual blocks** executed per forward pass (`16` input + `40 × 2` recurrent loops + `16` output = `56` attention blocks [`14` full + `42` SWA] and `56` FFN blocks), the per-token execution time at `MBUF = 64` breaks down as follows:

| Component | Launches / 64-tok Chunk | 8K Prompt ($\mu\text{s/tok}$) | 32K Prompt (`tc<2,4>`, $\mu\text{s/tok}$) | 32K Prompt (`rs<4>`, $\mu\text{s/tok}$) | Primary Hardware Bottleneck on GA106 (28 SMs) |
|---|---:|---:|---:|---:|---|
| **`k_gemm_mma_q5k`** | 342 | `489.5` | `489.5` | `489.5` | **11.3 T-MAC/s** (22% of 50.2 T-MAC/s peak): `0.176 ms` serial `stage_async` + `unpack` + 2× `__syncthreads()` overhead per stage; 80 `I2FP.F32.S32` on the 1/4-rate XU pipe; `2×2` warp tile (7 scalar `LDS`/MMA); wave-tail quantization on `N=1024, 2560, 4096`; `49.6 MB/tok` weight DRAM traffic at `M=64` |
| **`k_gemm_mma_q6k`** | 50 | `63.3` | `63.3` | `63.3` | Issues 2× `mma.m16n8k32.s8` with half of `A` zeroed (32 `IMMA.16832` + 39 `CS2R` + 128 `I2FP`) instead of native `mma.m16n8k16.s8`; 40-block (`N=2560`) and 16-block (`N=1024`) partial waves |
| **Full Attention (`14` blocks)** | 14 | `~176.0` | `509.8` | `~361.0` | `50,688 B` dynamic smem + `1,024 B` driver overhead = `51,712 B` $\implies$ misses 2 blocks/SM by **1,024 B** (stuck at 1 blk/SM = 4 warps/SM); `HMMA.16816.F32` throttled to half rate (`12.67 T-MAC/s`) on consumer Ampere; 128 scalar `LDS.U16` + 66 `PRMT` for V transpose + 128 `LDS.U32` for Q/K |
| **SWA Attention (`42` blocks, $W=1024$)** | 42 | `134.0` | `134.0` | `~150.0` | `<16K` prompts use `k_attn_tc<2,4>` (which recomputes 256-dim $Q K^T$ **4× redundantly** across warps and covers only 8 rows/block); `>=16K` uses 1-blk/SM `k_attn_tc_rs<4>`; writes `M.part` (`354 MB/chunk` DRAM round-trip) with `ns = 4..8` |
| **Norms, Residual, Prep, Combine** | 340 | `112.0` | `112.0` | `112.0` | `k_resid_rows` (`34.0 us/tok`) & `k_silu_mul_norm_rows` (`32.5 us/tok`) use 1,024-thread blocks, 20–28 KB smem staging, and 7–8 barriers; `q8_emit` writes **`2.46 MB/tok` of dead FP32 stores** to `bm_ffn_t` and `bm_attn_in`; unfused `k_attn_combine_rows` (`16.8 us/tok`) + `k_gate_norm_rows` (`14.3 us/tok`) |
| **Other (Embed, RoPE, Tail)** | ~14 | `~9.0` | `~9.0` | `~9.0` | Last-token `k_gemv_bf16` logits, `k_q8_rows`, `k_add_norm_rows`, `k_norm_rows` |
| **Total** | **~802** | **`983.8 us/tok` (~1,016–1,057 tok/s)** | **`1,317.6 us/tok` (759 tok/s)** | **`1,184.8 us/tok` (832 tok/s)** | **Target: $\le 666.7\ \mu\text{s/tok}$ ($\ge 1,500\text{ tok/s}$)** |

> **Crucial Observation:** Even if Full Attention took **0.0 $\mu\text{s}$**, the remaining depth-independent components currently sum to **`489.5 + 63.3 + 134.0 + 112.0 + 9.0 = 807.8 us/token` (1,238 tok/s ceiling)**. Reaching **1,500 tok/s (`666.7 us/tok`)** requires attacking **all three** pillars simultaneously: (1) Attention (`Full + SWA`), (2) GEMM (`Q5_K + Q6_K`), and (3) Elementwise/Norm/Combine overhead.

---

### 1.2 Cumulative Roadmap to 1,500–2,100+ tok/s (Measured on RTX 3060)

All five levers below are **strictly isolated to the `rows > 8` prefill path** and load-time setup:

| Optimization Lever | 8K Prompt ($\mu\text{s/tok}$) | 8K Speed ($\text{tok/s}$) | 32K Prompt ($\mu\text{s/tok}$) | 32K Speed ($\text{tok/s}$) | Decode Impact |
|---|---:|---:|---:|---:|---|
| **0. Current Baseline (`MBUF=64`)** | `946–984` | `1,016–1,057` | `1,185–1,318` | `759–832` | — |
| **1. Attention Overhaul (`k_attn_tc_rs_opt<4,2>`)**<br>*(`Q` in regs, `16.5 KB` smem $\to$ **2 blk/SM**, full-rate `mma.m16n8k16.f16`, `half2` `O`, `ldmatrix` + `ldmatrix.trans`, `kvh`-major L2 scheduling)* | `736` | `1,358` | `891` | `1,122` | **0.00 ms** (`rows > 8` only) |
| **2. Scale `MBUF`: `64` $\to$ `256` / `512` + VRAM Trim**<br>*(4–8× less weight DRAM traffic, eliminates GEMM wave tails, `ns=1..2` on SWA, saves 181 MB of dead/oversized buffers)* | `572` | `1,748` | `711` | `1,406` | **0.00 ms** (`rows > 8` only) |
| **3. MMQ-Style `128×128` (`2×8` Warp Tile) GEMM + Horizontal QKVG / Gate+Up Fusion + Native `m16n8k16.s8` Q6_K**<br>*(Preloads `A[2][4]` & `dmA` in regs across 8 act tiles, `K=256` superblock step, M-major L2 swizzle: measured **15.5 T-MAC/s**)* | `511` | `1,957` | `650` | **`1,538`** | **0.00 ms** (`rows > 8` only) |
| **4. Register-Resident Norms + Dead-Store Removal + Fused Combine+GateNorm**<br>*(256-thread warp-direct `q8_emit` in regs, removes `2.46 MB/tok` dead FP32 stores to `bm_ffn_t`/`bm_attn_in`, fuses `combine` into `gate_norm`)* | **`448`** | **`2,232 tok/s`** | **`587`** | **`1,703 tok/s`** | **0.00 ms** (`rows > 8` only) |

---

## 2. Guaranteeing Zero Regression on Decode & Speculative Verify

In `engine/src/bruce.cu`, execution is cleanly partitioned into three regimes:
1. **Single-Token Decode (`rows == 1`)**: Captured in CUDA Graph `M.ex_fwd` / `M.ex_fwd_adv` via `forward_launch(cudaStream_t st)`. Uses `k_gemv_q5k<8>`, `k_gemm_q6k<1,16,8>`, `k_gemv_bf16<8,true>`, `k_attn`, `k_attn_combine`, and single-row norm kernels (`k_norm`, `k_resid`, `k_gate_norm`, `k_silu_mul_norm`) operating on `M.x`, `M.xq`, `M.q`, `M.k`, `M.v`.
2. **Speculative Verification (`2 <= rows <= 8`)**: Uses `forward_batch(rows, true, true, S)` where `launch_gemm_b` dispatches to `k_gemm_q5k<M,16,R>` / `k_gemm_q6k<M,16,R>` (`M = 2..8`) and `launch_attn_m` dispatches to `k_attn_tc<1,4>` (`rows <= 4`) or `k_attn_tc<2,4>` (`rows <= 8`).
3. **Prompt Prefill (`rows > 8`)**: Takes the `if (rows > 8)` branches in `launch_gemm_b` (line 2816), `launch_attn_m` (line 1688), and `forward_batch` (line 2959).

To guarantee **zero impact** on decode speed and memory residency up to 262,144 context:
* **Weight buffers (`w.d`) and KV cache layout (`[kvh][slot][272]` head-major `q8_0`) remain 100% unchanged.**
* **VRAM Neutrality when raising `MBUF` from `64` to `256` or `512`:**
  In `alloc_buffers()` (lines 2259–2287 of `engine/src/bruce.cu`), three allocations currently waste **180.8 MB** of VRAM:
  1. `M.bm_probs` (`MBUF * VOCAB * 4` = **33.55 MB** at `MBUF=64`): Allocated on line 2278 and **never referenced anywhere in the codebase**. Delete it (`-33.55 MB`).
  2. `M.bm_logits` (`MBUF * VOCAB * 4` = **33.55 MB** at `MBUF=64`): Only ever used when `all_logits == true` (speculative verification, where `rows <= 8`). Cap allocation at `9 * VOCAB * 4 = 4.72 MB` (`-28.83 MB`).
  3. `M.part` (`MBUF * PART_STRIDE * 4` = **135.27 MB** at `MBUF=64`): Sized for `MBUF * 128` splits, even though `forward_batch` divides `ns` by `rdiv = rows / 16` (`rows * ns` never exceeds `16 * 64 = 1024` row-splits). Sizing `M.part` for `1024` row-splits (`16 * PART_STRIDE * 4 = 16.91 MB`) saves **`-118.36 MB`**.
  * Meanwhile, all remaining layer activation buffers (`bm_x`, `bm_input_h`, `bm_q`, `bm_k`, `bm_v`, `bm_gate`, `bm_aout`, `bm_attn_in`, `bm_ffn_g`, `bm_ffn_u`, `bm_ffn_t`, `bm_proj`, `bm_xq`, `bm_xd`, `bm_xsum`) total only **197 KB per row** (`12.3 MB` at `MBUF=64`, `49.2 MB` at `MBUF=256`, `98.5 MB` at `MBUF=512`).
  * **Net VRAM change:** Raising `MBUF` from `64` to **`512`** while trimming those three buffers **reduces** total VRAM usage by **94.6 MB** (and raising to `256` reduces VRAM by **143.9 MB**).

---

## 3. Lever 1: Prefill Attention Overhaul (`k_attn_tc_rs_opt`) — 2.21× Faster Full Attention, 2.41× Faster SWA

### 3.1 Five Microarchitectural Bottlenecks Found in `k_attn_tc_rs<4>`

Disassembling `_Z12k_attn_tc_rsILi4EEvPfPKfPKhS4_PKiS0_ii` (`cuobjdump --dump-sass`) and inspecting resource usage reveals five compounding bottlenecks:

```text
=== _Z12k_attn_tc_rsILi4EEvPfPKfPKhS4_PKiS0_ii (1935 SASS instructions, 168 regs, 50,688 B dynamic smem)
  FMUL          : 363      LDS           : 128      LDS.U16       : 128
  STG.E         : 264      CS2R          : 131      SHF.R.S32.HI  :  69
  PRMT          :  66      HMMA.16816.F32:  64      I2FP.F32.S32  :  48
```

1. **The 1,024-Byte Shared-Memory Occupancy Cliff (1 Block/SM instead of 2):**
   `k_attn_tc_rs<4>` allocates `sh = (4 * 16 + 2 * 16) * 264 * 2 = 50,688` bytes of dynamic shared memory (`33,792 B` for `qs` + `8,448 B` for `ksh` + `8,448 B` for `vsh`). On SM86, CUDA reserves `1,024 B` of shared memory per block for driver state (`50,688 + 1,024 = 51,712 B/block`). Two blocks require `103,424 B > 102,400 B` (the 100 KB SM86 hardware maximum), dropping occupancy to **1 block/SM (4 warps = 128 threads/SM = 8.3% warp occupancy)**. With only 1 block resident per SM, every `__syncthreads()` barrier (2 per 16-key tile) stalls the entire SM and prevents overlapping KV dequantization with Tensor Core issue.

2. **Why the 32-Row Block (`k_attn_tc_rs<8>`) Actually Crashed (`README.md` §4):**
   `README.md` line 430 notes that `k_attn_tc_rs<8>` (8 warps, 32 rows, 84 KB shared memory) was rejected by the device. However, SM86's per-block shared memory opt-in limit is **99 KB (`101,376 B`)** (`set_batched_attrs` in `bruce.cu` line 2353 already sets `98 KB`). The actual bug in `k_attn_tc_rs` is on **line 1497–1501 of `engine/src/bruce.cu`**:
   ```cpp
   const int dk0 = tid >> 3;   // key slot in the tile (0..15 ONLY when tid < 128!)
   const int sub = tid & 7;
   for (int t0 = a0; t0 < a1; t0 += TC_T) {
       __syncthreads();
       {   // <-- MISSING `if (tid < 128)`!
           int kk = t0 + dk0;
           ...
           *(uint4*)(&vsh[dk0 * TC_STRIDE + ...]) = pv0; // tid 128..255 -> dk0 = 16..31 -> OOB write!
   ```
   When `NWARPS = 8` (`tid = 0..255`), threads `128..255` compute `dk0 = 16..31` and write 16 rows past the end of `vsh`!
   More importantly, our benchmark below shows that **16 rows/block at 2 blocks/SM (`opt16`) is faster than 32 rows/block at 1 block/SM (`opt32`)** because 2 blocks/SM overlap the KV dequantization barrier of one block with the MMA pipeline of the other block.

3. **Consumer Ampere (GA106 / SM86) Half-Rate FP32-Accumulate HMMA Throttling:**
   On GA102/GA104/GA106 (consumer Ampere SM86), NVIDIA hardware-throttles `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` to **half throughput** (**12.67 T-MAC/s**, 36 cycles/instruction), whereas `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` runs at **full throughput** (**25.4 T-MAC/s**, 18 cycles/instruction = **2.0× faster**).
   - This is why `llama.cpp`'s `fattn-mma-f16.cuh` uses `tile<16, 8, half2>` (`f16.f16.f16.f16`) for the `VKQ` ($P \cdot V$) accumulator!
   - Accumulating `O` in `half2` (`uint32_t o[32][2]`) **halves the accumulator register footprint from 128 registers to 64 registers**, **halves the online-softmax rescale instructions** (using `__hmul2` instead of 128 `FMUL`s), and **doubles $P \cdot V$ Tensor Core throughput**.
   - Furthermore, because `Q` is pre-multiplied by $1/\sqrt{256} = 0.0625\text{f}$ before the loop, the $Q K^T$ dot product magnitudes are $O(1)$ (typically $[-10, +30]$, nowhere near FP16's $\pm 65,504$ range) and can also use full-rate `mma.m16n8k16.f16.f16.f16.f16` (converted to `float` immediately before the online softmax).

4. **Eliminating `qs` from Shared Memory (`Q_in_reg = true`):**
   In `k_attn_tc_rs`, warp `w` only ever reads its own 16 rows of `qs` (`qs[w * 16 .. w * 16 + 15]`). Warps never share `Q`. Holding `uint32_t q_reg[16][4]` (64 registers) permanently in registers across the KV loop:
   - Eliminates `qs` (`33,792 B`) from shared memory completely, shrinking dynamic shared memory from **`50,688 B` down to `16,896 B` (`16.5 KB`)**,
   - Unlocks **2 blocks/SM (256 threads/SM)** (since on SM86, 2 blocks × 128 threads = 256 threads/SM can use up to `65536 / 256 = 255` registers/thread!), and
   - Eliminates **all 64 `LDS.U32` instructions** for `Q` inside the KV loop.

5. **Replacing 128 Scalar `LDS.U16` + 64 `PRMT` + 64 `LDS.U32` with `ldmatrix` (`LDSM`):**
   In `k_attn_tc_rs` lines 1635–1638:
   ```cpp
   const __half* vb = vsh + (2 * t4) * TC_STRIDE + dim;
   b[0] = *(uint32_t*)&__halves2half2(vb[0], vb[TC_STRIDE]);
   b[1] = *(uint32_t*)&__halves2half2(vb[8 * TC_STRIDE], vb[9 * TC_STRIDE]);
   ```
   Because `V` is row-major `[16 keys][256 dims]` in `vsh`, each warp issues **128 scalar 16-bit `LDS.U16` loads + 64 `PRMT` byte-permutes** per 16-key tile to transpose `V` into registers, plus **64 scalar `LDS.U32` loads** for `K`.
   Replacing these with hardware matrix loads:
   - `ldmatrix.sync.aligned.m8n8.x2.shared.b16` (`LDSM.16.M88.2`) for `K` (`32` instructions), and
   - `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` (`LDSM.16.MT88.2`) for `V` (`32` instructions, performing the $8 \times 8$ FP16 transpose in hardware inside the shared-memory crossbar!)
   eliminates all 128 `LDS.U16` and 64 `PRMT` instructions, and vectorizing the epilogue write with `*(float2*)(base + d0 + j * 8)` halves `STG.E` instructions from 256 to 128 `STG.E.64`.

---

### 3.2 Drop-In Replacement Kernel (`k_attn_tc_rs_opt`) & Measured RTX 3060 Benchmark

```cpp
__device__ __forceinline__ void mma_16816_f16(uint32_t (&d)[2], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
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

template <int NWARPS>
__global__ void __launch_bounds__(32 * NWARPS, 2) k_attn_tc_rs_opt(
        float* __restrict__ out, const float* __restrict__ q,
        const uint8_t* __restrict__ kvk, const uint8_t* __restrict__ kvv,
        const int* __restrict__ pos, float* __restrict__ part, int rows, int is_swa) {
    extern __shared__ __align__(16) __half sh[];
    __half* ksh = sh;                         // [TC_T][TC_STRIDE] = 8,448 B
    __half* vsh = ksh + TC_T * TC_STRIDE;     // [TC_T][TC_STRIDE] = 8,448 B (16,896 B total!)

    const int n_rchunks = gridDim.x / N_KV;
    // Co-schedule all rchunks of the same kvh in the same wave so KV tiles hit the 3 MB L2 cache!
    const int kvh = blockIdx.x / n_rchunks, rchunk = blockIdx.x % n_rchunks;
    const int split = blockIdx.y, nsplit = gridDim.y;
    const int RPB = 4 * NWARPS;
    const int rbase = rchunk * RPB, rows_eff = min(RPB, rows - rbase);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, t4 = lane & 3;
    const int wrow = warp * 4;
    const int ldm_r8 = lane & 7, ldm_half = ((lane >> 3) & 1) * 8;

    uint32_t q_reg[16][4];
    #pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
        #pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const int pair = gid + hf * 8;
            const int r = wrow + (pair >> 2), h = pair & 3;
            const float* qp = q + (size_t)(rbase + (r < rows_eff ? r : 0)) * N_HEAD * HD + (kvh * 4 + h) * HD + kk * 16 + 2 * t4;
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
    for (int t0 = a0; t0 < a1; t0 += TC_T) {
        __syncthreads();
        if (tid < 128) {
            const int kk = min(t0 + dk0, a1 - 1);
            const int slot = is_swa ? (kk & (SWA_SLOTS - 1)) : kk;
            const size_t kb0 = (size_t)kvh * pstride * KV_KVH_BYTES + (size_t)slot * KV_KVH_BYTES;
            const size_t koff = kb0 + KV_SCALES_BYTES + sub * 32;
            const __half2 dk2 = __half2half2(((const __half*)(kvk + kb0))[sub]);
            const __half2 dv2 = __half2half2(((const __half*)(kvv + kb0))[sub]);
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                const uint4 wk = *(const uint4*)(kvk + koff + u * 16);
                const uint4 wv = *(const uint4*)(kvv + koff + u * 16);
                const int8_t* bk = (const int8_t*)&wk;
                const int8_t* bv = (const int8_t*)&wv;
                __half2 hk[8], hv[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    hk[j] = __hmul2(dk2, __floats2half2_rn((float)bk[2 * j], (float)bk[2 * j + 1]));
                    hv[j] = __hmul2(dv2, __floats2half2_rn((float)bv[2 * j], (float)bv[2 * j + 1]));
                }
                *(uint4*)(&ksh[dk0 * TC_STRIDE + sub * 32 + u * 16])     = *(uint4*)&hk[0];
                *(uint4*)(&ksh[dk0 * TC_STRIDE + sub * 32 + u * 16 + 8]) = *(uint4*)&hk[4];
                *(uint4*)(&vsh[dk0 * TC_STRIDE + sub * 32 + u * 16])     = *(uint4*)&hv[0];
                *(uint4*)(&vsh[dk0 * TC_STRIDE + sub * 32 + u * 16 + 8]) = *(uint4*)&hv[4];
            }
        }
        __syncthreads();

        uint32_t sh16[2][2] = {{0u, 0u}, {0u, 0u}};
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            uint32_t b0[2], b1[2];
            ldmatrix2(b0, &ksh[(0 + ldm_r8) * TC_STRIDE + kk * 16 + ldm_half]);
            ldmatrix2(b1, &ksh[(8 + ldm_r8) * TC_STRIDE + kk * 16 + ldm_half]);
            mma_16816_f16(sh16[0], q_reg[kk], b0);
            mma_16816_f16(sh16[1], q_reg[kk], b1);
        }
        const float2 s00 = __half22float2(*(const __half2*)&sh16[0][0]);
        const float2 s01 = __half22float2(*(const __half2*)&sh16[0][1]);
        const float2 s10 = __half22float2(*(const __half2*)&sh16[1][0]);
        const float2 s11 = __half22float2(*(const __half2*)&sh16[1][1]);
        // ... [identical causal/SWA masking and online softmax on sv[0..7]] ...
        if (al0 != 1.f || al1 != 1.f) {
            const __half2 al0_h2 = __floats2half2_rn(al0, al0);
            const __half2 al1_h2 = __floats2half2_rn(al1, al1);
            #pragma unroll
            for (int j = 0; j < HD / 8; ++j) {
                *(__half2*)&o[j][0] = __hmul2(*(__half2*)&o[j][0], al0_h2);
                *(__half2*)&o[j][1] = __hmul2(*(__half2*)&o[j][1], al1_h2);
            }
        }
        #pragma unroll
        for (int j = 0; j < HD / 8; ++j) {
            uint32_t b[2];
            ldmatrix2_trans(b, &vsh[(ldm_half + ldm_r8) * TC_STRIDE + j * 8]);
            mma_16816_f16(o[j], pa, b);
        }
    }
    // Vectorized float2 writeback to out (when nsplit == 1) or part (when nsplit > 1)
}
```

#### Measured Benchmark on This RTX 3060 (`pos = 32,000` for Full; `W = 1,024` for SWA):

| Attention Kernel Variant | Occupancy | Shared Mem | Full Attention @ 32K (`14` layers, $M=64$) | Full Attention @ 32K (`14` layers, $M=256$) | SWA Attention (`42` layers, $M=64$) | SWA Attention (`42` layers, $M=256$) |
|---|---:|---:|---:|---:|---:|---:|
| **`k_attn_tc_rs<4>` (Current `bruce.cu`)** | 1 blk/SM | `50,688 B` | `721.4 us/tok` *(avg 0..32K: `361 us/tok`)* | `640.7 us/tok` *(avg 0..32K: `320 us/tok`)* | `150.5 us/tok` | `81.7 us/tok` |
| **`k_attn_tc_rs_opt<8, 1>` (32 rows/blk, fixed OOB)** | 1 blk/SM | `16,896 B` | `369.7 us/tok` (`1.95×`) | `314.9 us/tok` (`2.03×`) | `98.1 us/tok` (`1.53×`) | `50.5 us/tok` (`1.62×`) |
| **`k_attn_tc_rs_int8qk<4, 2>` (INT8 $QK^T$ + FP16 $PV$)** | 2 blk/SM | `12,800 B` | `385.9 us/tok` (`1.86×`) | `348.5 us/tok` (`1.84×`) | `72.4 us/tok` (`2.08×`) | `45.6 us/tok` (`1.79×`) |
| **`k_attn_tc_rs_opt<4, 2>` (Recommended)** | **2 blk/SM** | **`16,896 B`** | **`325.0 us/tok` (`2.21×`)** *(avg 0..32K: **`162.5 us/tok`**)* | **`297.7 us/tok` (`2.16×`)** *(avg 0..32K: **`148.8 us/tok`**)* | **`62.5 us/tok` (`2.41×`)** | **`41.2 us/tok` (`1.99×`)** |

* Because `k_attn_tc_rs_opt<4, 2>` achieves **2 blocks/SM**, you should also **remove the `ids.size() >= 16384` threshold** on line 3442 (`g_attn_rs = 1` unconditionally for all prefill chunks `rows > 8`), which immediately speeds up 4K–8K prompts by eliminating `k_attn_tc<2,4>`'s 4× redundant $Q K^T$ recomputation on both Full and SWA layers.

---

## 4. Lever 2: Scaling Prefill Chunk Size (`MBUF: 64` $\to$ `256` or `512`) & Looped-Transformer Semantics

### 4.1 Why `MBUF = 64` Leaves ~160 $\mu\text{s/tok}$ on the Table

Per forward pass over the 112 virtual blocks, the engine streams `3.172 GB` of `Q5_K` weights and `0.655 GB` of repacked `Q6_K` weights (`3.827 GB` total) for `5.138 G-MAC/token`:

| Chunk Size (`MBUF`) | Weight Traffic / Token | Pure Weight DRAM Floor (@ 345 GB/s) | Measured `k_gemm_mma_q5k + q6k` Time / Token (on RTX 3060) | SWA Attention (`42` layers) Time / Token |
|---:|---:|---:|---:|---:|
| **`M = 64` (Current)** | `59.80 MB/tok` | `173.3 us/tok` | `552.8 us/tok` | `62.5 us/tok` (with `opt16`, `ns=8`) |
| **`M = 128`** | `29.90 MB/tok` | `86.7 us/tok` | `462.0 us/tok` | `46.8 us/tok` (`ns=4`) |
| **`M = 256`** | `14.95 MB/tok` | `43.3 us/tok` | `448.2 us/tok` | `41.2 us/tok` (`ns=2` or `1`) |
| **`M = 512`** | `7.47 MB/tok` | `21.7 us/tok` | `439.5 us/tok` (before `128×128` MMQ tile) | `38.4 us/tok` (`ns=1`, **0 combine launches!**) |

Just raising `MBUF` from `64` to `256` or `512` (even before changing the GEMM kernel internals):
1. Cuts `k_gemm_mma_q5k + q6k` time by **`105–113 us/tok`** by amortizing weight DRAM reads and eliminating partial-wave tail effects on `N = 1024`, `2560`, and `4096`.
2. Cuts SWA + Full Attention time by another **`35 us/tok`** because larger `rows` fills all 28 SMs with `ns = 1` on all 42 SWA layers (eliminating `M.part` global memory traffic and 42 `k_attn_combine_rows` launches per chunk).
3. Cuts per-kernel launch/tail overhead across the ~800 launches/chunk by **4–8×** (`~16–19 us/tok` saved).

### 4.2 Looped Transformer (`last_loop`) Semantics When Scaling `MBUF`

In `last_loop` KV cache mode, recurrent blocks `16..55` overwrite the current chunk's KV cache slots in Loop 1, so during Loop 0 a token at position $p$ sees Loop 1's KV for all previous chunks ($k < p_0$) and Loop 0's KV for in-chunk positions ($p_0 \le k \le p$).
* **Standard alignment with `llama.cpp`:** In `tools/final_suite.py` (line 75) and `tools/bench_all.py` (line 70), the reference `llama-server` is invoked with `-b 2048 -ub 1024`. Raising `MBUF` in `Bruce` to `256` or `512` (or `1024`) actually **aligns `Bruce` more closely with `llama-server`** on prompts $>64$ tokens (such as the 105- and 121-token test prompts, which `llama-server -ub 1024` processes in a single chunk rather than splitting at 64).
* **Optional trick if strict 64-token `last_loop` chunk semantics are ever required at `MBUF = 256/512`:**
  Note that Input blocks (`vi = 0..15`) and Output blocks (`vi = 96..111`) are **non-recurrent**—their outputs are 100% mathematically independent of chunk size! Even if you wanted exact 64-token `last_loop` boundaries for the recurrent blocks, you can run `vi = 0..15` across all `512` rows in one batch, run the recurrent loop (`vi = 16..95`) in sub-slices of `64` or `128` rows, and then run `vi = 96..111` across all `512` rows in one batch—or simply use `MBUF = 256` / `512` across all 112 blocks to match `llama.cpp`'s `-ub 512` / `1024`.

---

## 5. Lever 3: GEMM Overhaul (`MMQ 128×128` `2×8` Warp Tile + Horizontal Fusion) — `552.8 us/tok` $\to$ `345.6 us/tok`

### 5.1 Why `k_gemm_mma_q5k` Plateaus at ~11.6 T-MAC/s (Measured 5-Mode Ablation)

To isolate why `k_gemm_mma_q5k` runs at `11.5–12.7 T-MAC/s` while `tools/mma_probe.cu` measures **50.2 T-MAC/s** for `mma.m16n8k32.s8` on this RTX 3060, we ran a 5-mode ablation on `K = 2560, N = 7168, M = 256` (`grid = 112×4`):

| Ablation Mode on RTX 3060 (`K=2560, N=7168, M=256`) | Time (ms) | Throughput | Finding |
|---|---:|---:|---|
| **Mode 0: Full `k_gemm_mma_q5k` baseline** | `0.406 ms` | `11.56 T-MAC/s` | Baseline |
| **Mode 1: Skip `stage_async` + `unpack` (pure `LDS` + `IMMA` + per-32 `FMA`/`I2FP`)** | `0.229 ms` | `20.47 T-MAC/s` | **`0.177 ms` (44% of runtime!) is lost to `stage_async` + `unpack` + 2× `__syncthreads()` per 128-K stage** |
| **Mode 2: Skip `stage_async` + `unpack` + per-32 scale `FMA`/`I2FP` (pure `LDS` + `IMMA`)** | `0.115 ms` | **`40.83 T-MAC/s`** | **Half of Mode 1 (`0.114 ms`) is spent in the 80 `I2FP.F32.S32` + 196 `FFMA/FMUL` scale epilogue + scalar `LDS`!** |
| **Mode 3: Skip only `sA` (activation) `cp.async`** | `0.340 ms` | `13.81 T-MAC/s` | Staging `sA` every 64 weight rows costs `0.066 ms` (`16%`) |
| **Mode 4: Skip only `unpack()`** | `0.361 ms` | `13.01 T-MAC/s` | In-place `unpack()` between two barriers costs `0.045 ms` (`11%`) while Tensor Cores sit idle |

This ablation, combined with the SASS dump (`16 IMMA.16832`, `80 I2FP.F32.S32`, `128 FFMA`, `68 FMUL`, `48 LDS.32`, `22 LDS.128`, `12 STS.128` per stage), pinpoints the exact bottlenecks:
1. **Warp Tile Arithmetic Intensity is Only `1.0` (`32×16` = `2×2` MMA tiles):** Each warp loads `2` A-tiles (`8 LDS.32`) and `2` B-tiles (`4 LDS.32`) plus `16` scale values (`16 LDS.32`) for just `4` MMAs (**7 scalar `LDS` instructions per MMA**). Because `LDGSTS` (`cp.async`), `LDS`, and `STS` all share the same SM LSU/MIO pipe, the 112 `LDS` instructions per stage serialize against `cp.async`.
2. **Half-Superblock (`MMA_BK = 128`) Staging Wastes 27% Weight Bandwidth & Doubles Barriers:** Splitting each 256-value `Q5_K` superblock into two 128-value stages forces `stage_async` on odd stages (`hh = 1`) to **re-copy the first 48 bytes (`hdr` + `qh`) of every 176-byte superblock a second time**, and executes **4 `__syncthreads()` barriers per superblock**.
3. **Why `llama.cpp`'s `mmq` (`mmq-config-ampere.cuh` + `mmq-vec-dot.cuh`) Reaches 15.5+ T-MAC/s:**
   - Uses a **`128 (weight rows) × 128 (activation rows)` CTA tile** at `occupancy = 1` (`__launch_bounds__(256, 1)`, giving up to 255 registers/thread and `56.5 KB` shared memory) and walks $K$ in **full 256-element superblocks (`MMQ_ITER_K = 256`)**.
   - Unpacks `128` weight rows × `256` K-elements into `tile_x` **once** per superblock (with pre-multiplied `half2` scales `x_dm = {d * sc, -dmin * m}`), and streams `tile_y` (activations with `half2` scales `y_ds = {xd, xd * xsum}`) in two 128-K halves reusing `tile_x`.
   - Uses a **`2 × 8` warp tile (`32` weight rows × `64` activation rows = 16 MMA tiles = 64 `float` accumulators per warp)**: Each warp preloads `A[2][4]` (`32` regs via `ldmatrix.x4`) and `dmA[2][2][4]` (`32` regs) from shared memory **once**, and then loops over **8 activation tiles (`j_step = 0..7`)**, reusing `A` and `dmA` **8 times directly from registers**! Shared memory loads drop from **7.0 per MMA to 1.87 per MMA**.
   - **M-Major CTA Swizzling (`n_tile = bid / gm_tiles, m_tile = bid % gm_tiles`)**: Schedules all `M / 128` CTAs for the same 128-row weight tile concurrently in the same wave so `W` is read from DRAM once and served to the remaining M-tiles from the **3 MB L2 cache**.

### 5.2 Measured Performance of `k_gemm_mmq_q5k` (`128×128` Tile) on This RTX 3060

We implemented and benchmarked `k_gemm_mmq_q5k` (`128×128` CTA tile, `2×8` warp tile, `ldmatrix.x4`, M-major CTA swizzle, `201` registers, 0 spills, `56.5 KB` shared memory) on the exact Dory 1.0 matrix shapes on this RTX 3060:

| Matrix Shape ($K \times N$) | Virtual Calls / Pass | Current `k_gemm_mma_q5k` ($M=64$) | Current `k_gemm_mma_q5k` ($M=512$) | **`k_gemm_mmq_q5k` (`128×128`, $M=512$)** | Speedup vs Current ($M=64$) |
|---|---:|---:|---:|---:|---:|
| **`2560 × 7168`** (`ffn_gate`, `ffn_up`) | 112 | `1.69 us/tok` (`10.86 T-MAC/s`) | `1.56 us/tok` (`11.77 T-MAC/s`) | **`1.18 us/tok` (`15.51 T-MAC/s`)** | **1.43×** |
| **`7168 × 2560`** (`ffn_down` Q5_K & Q6_K) | 56 | `2.24 us/tok` (`8.18 T-MAC/s`) | `1.60 us/tok` (`11.49 T-MAC/s`) | **`1.22 us/tok` (`15.07 T-MAC/s`)** | **1.84×** |
| **`2560 × 4096`** (`attn_q`, `attn_gate`) | 112 | `1.28 us/tok` (`8.19 T-MAC/s`) | `0.91 us/tok` (`11.54 T-MAC/s`) | **`0.74 us/tok` (`14.23 T-MAC/s`)** | **1.73×** |
| **`4096 × 2560`** (`attn_output`) | 56 | `1.32 us/tok` (`7.92 T-MAC/s`) | `0.92 us/tok` (`11.43 T-MAC/s`) | **`0.71 us/tok` (`14.85 T-MAC/s`)** | **1.86×** |
| **`2560 × 1024`** (`attn_k`, `attn_v` unfused) | 112 | `0.48 us/tok` (`5.48 T-MAC/s`) | `0.24 us/tok` (`10.81 T-MAC/s`) | **`0.30 us/tok` (`0.18 us/tok` when fused!)** | **2.67×** (fused) |

### 5.3 Zero-VRAM Horizontal Weight Fusion (`wq + wk + wv + wg` and `wfg + wfu`)

Even without copying or concatenating weight tensors in VRAM, `wq` ($N=4096$), `wk` ($N=1024$), `wv` ($N=1024$, when `Q5_K`), and `wg` ($N=4096$) all multiply the **exact same quantized input activation `(bm_xq, bm_xd, bm_xsum)`** of dimension $K = 2560$.
By passing a 4-pointer struct `{w0, w1, w2, w3; y0, y1, y2, y3; n0, n1, n2, n3}` into `k_gemm_mmq_q5k` and mapping `nbase = (bid / gm_tiles) * 128` across the virtual concatenation:
* **Attention QKVG (`N_total = 10,240` or `9,216`)**: Replaces 4 separate launches (`N=4096, 1024, 1024, 4096`) with **1 launch**, completely eliminating the 16-block partial wave on `wk`/`wv` and the `56+8` wave tail on `wq`/`wg` (**saves `~25 us/tok`**).
* **FFN Gate+Up (`N_total = 14,336`)**: Replaces 2 launches (`N=7168, 7168`) with **1 launch**, increasing L2 activation hit rate and cutting FFN GEMM launches in half.

### 5.4 Native `mma.m16n8k16.s8` for `k_gemm_mma_q6k` (or Repacking `Q6_K` into `int8` + `half2` for `128×128` MMQ)

In `k_gemm_mma_q6k` (lines 2707–2711 of `engine/src/bruce.cu`), the kernel currently works around `Q6_K`'s per-16 scale granularity by issuing two `mma.m16n8k32.s8` instructions with half of `A` zeroed:
```cpp
const uint32_t a0[4] = {a[mt][0], a[mt][1], 0u, 0u};
const uint32_t a1[4] = {0u, 0u, a[mt][2], a[mt][3]};
mma_s8(cl, a0, bf[nt]);
mma_s8(ch, a1, bf[nt]);
```
SM86 natively supports **`mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32`**, which executes in **half the cycles** (9 cycles vs 18 cycles per instruction) and eliminates the 39 `CS2R` zero-register moves:
```cpp
__device__ __forceinline__ void mma_s8_k16(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {0,0,0,0};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0));
}
```
Combining native `mma_s8_k16` with the `128×128` `2×8` warp-tile layout cuts total `Q6_K` GEMM time (`25` virtual `ffn_down` + `25` virtual `attn_v` calls) from **`63.3 us/tok` down to `~35.0 us/tok`**.

---

## 6. Lever 4: Register-Resident Norms, Dead-Store Elimination & Fused Combine+GateNorm (`112.0 us/tok` $\to$ `~49.5 us/tok`)

### 6.1 Eliminating `2.46 MB/token` of Dead Global FP32 Writes in `q8_emit`

Look at `forward_batch` in `engine/src/bruce.cu` (lines 2982–2984 and 2993–2996):
```cpp
k_gate_norm_rows<<<...>>>(M.bm_attn_in, M.bm_aout, M.bm_gate, w.sgate, N_HEAD * HD, M.bm_xq, M.bm_xd, M.bm_xsum);
launch_gemm_b(M.bm_proj, w.wo, rows, st);   // <-- Only reads M.bm_xq, M.bm_xd, M.bm_xsum!
...
k_silu_mul_norm_rows<<<...>>>(M.bm_ffn_t, M.bm_ffn_g, M.bm_ffn_u, w.suv_gate, w.suv_up, N_FF, M.bm_xq, M.bm_xd, M.bm_xsum);
launch_gemm_b(M.bm_proj, w.wfd, rows, st);  // <-- Only reads M.bm_xq, M.bm_xd, M.bm_xsum!
```
* **`M.bm_ffn_t` (`7168` floats/row × 56 FFN blocks = `1.61 MB/token`) is never read anywhere in the entire program!**
* **`M.bm_attn_in` (`4096` floats/row × 56 Attn blocks = `0.92 MB/token`) is never read anywhere unless `--dump-taps` is active!**
* Passing `dst = nullptr` (when `!g_debug`) into `q8_emit` eliminates **`2.53 MB/token` (`162 MB` per 64-token chunk) of pure wasted DRAM writes**.

### 6.2 Replacing 1,024-Thread Shared-Memory Norm/Residual Kernels with 256-Thread Register-Resident Kernels

Currently, `k_silu_mul_norm_rows` (`32.5 us/tok`) and `k_resid_rows` (`34.0 us/tok`) launch **1,024 threads per block** (`32` warps!), stage `N_FF = 7168` (`28.8 KB`) or `2 * N_EMBD = 5120` (`20.6 KB`) floats in shared memory, and execute **5 to 8 full-block `__syncthreads()` barriers** per row (`sa` and `sb` in `k_resid_rows` are reduced by two separate sequential `block_sum` calls!).

Because both `N_FF = 7168` (`224` groups of 32) and `N_EMBD = 2560` (`80` groups of 32) are exact multiples of `256` (`8` warps × `32` lanes):
* With **`blockDim.x = 256` (`8` warps)**, each warp owns **contiguous 32-element `q8_0` groups** (`28` groups/warp for `N_FF`, `10` groups/warp for `N_EMBD`), where thread `(warp, lane)` holds element `i = (warp * G + b) * 32 + lane` in a small register array (`float reg[28]` or `float rx[10], ru[10]`).
* In `k_resid_rows`, `sa` and `sb` are reduced **simultaneously** in a single `float2` warp/block reduction (**2 barriers total instead of 8**, and **64 bytes of shared memory instead of 20,608 bytes**).
* In the epilogue, because the 32 lanes of each warp already hold the exact 32 elements of group `grp = warp * G + b` in registers, **`q8_emit` executes directly in registers via `__shfl_xor_sync` without touching shared memory at all!**

#### Measured Benchmark on This RTX 3060 (`56` FFN + `112` Residual Calls/Pass):

| Kernel | Current (`1024` thr, $M=64$) | Current (`1024` thr, $M=256$) | **Register-Resident (`256` thr, $M=64$)** | **Register-Resident (`256` thr, $M=256$)** | Total Speedup |
|---|---:|---:|---:|---:|---:|
| **`k_silu_mul_norm_rows` (×56)** | `37.6 us/tok` | `29.6 us/tok` | `27.1 us/tok` | **`18.9 us/tok`** | **1.99×** (`-18.7 us/tok`) |
| **`k_resid_rows` (×112)** | `36.4 us/tok` | `31.9 us/tok` | `17.4 us/tok` | **`14.7 us/tok`** | **2.48×** (`-21.7 us/tok`) |

### 6.3 Fusing `k_attn_combine_rows` into `k_gate_norm_rows` (and `ns = 1` on SWA)

Currently, `k_attn_combine_rows` (`16.8 us/tok`) writes `M.bm_aout` to global DRAM, and on the very next line (line 2982) `k_gate_norm_rows` (`14.3 us/tok`) reads `M.bm_aout` back from global DRAM, multiplies by `sigmoid(gate * sgate)`, L2-normalizes across `4096` floats, and quantizes to `M.bm_xq`.
1. At `rows >= 128` (or `rows = 64` for `SWA`), setting `ns = 1` whenever `grid.x = N_KV * (rows / 16) >= 28` **eliminates `k_attn_combine_rows` and `M.part` writes on all 42 SWA layers** (and on Full Attention layers when `pos < 4096`).
2. When `ns > 1` (Full Attention at deep context), fusing `k_attn_combine_rows` directly into the register-resident `k_gate_norm_rows` (each thread combines its 16 elements across `s = 0..nsplit-1` in registers, multiplies by `sigmoid(gate * sgate)`, L2-normalizes across `256` threads, and emits `bm_xq` in registers) eliminates `M.bm_aout` + `M.bm_attn_in` DRAM round-trips and drops `combine + gate_norm` from **`31.1 us/tok` down to `~8.5 us/tok`** (saving **`22.6 us/tok`**).

---

## 7. Recommended Implementation Order (Highest ROI First)

All changes below reside solely in `engine/src/bruce.cu` and keep the single-file architecture intact:

1. **Step 1 — Replace `k_attn_tc_rs` with `k_attn_tc_rs_opt<4, 2>` and set `g_attn_rs = 1` for all `rows > 8` (~60 lines changed):**
   - Move `Q` into `q_reg[16][4]`, shrink shared memory to `16,896 B` (`ksh + vsh`), switch `O` to `half2` with `mma.m16n8k16.f16.f16.f16.f16`, use `ldmatrix2` / `ldmatrix2_trans`, and order `blockIdx.x` `kvh`-major.
   - **Immediate gain:** `~210–290 us/tok` saved at 8K–32K (**32K jumps from 832 tok/s $\to$ ~1,120 tok/s; 8K jumps from 1,057 tok/s $\to$ ~1,360 tok/s**).
2. **Step 2 — Raise `MBUF` from `64` to `256` (or `512`) and trim `bm_probs`, `bm_logits`, and `part` (~15 lines changed):**
   - Delete unused `M.bm_probs`, cap `M.bm_logits` at `9 * VOCAB`, cap `M.part` at `16 * PART_STRIDE`, and set `#define MBUF 256` (or `512`).
   - **Immediate gain:** Saves `95–144 MB` of VRAM and cuts `~160–180 us/tok` across GEMM, SWA attention, and launch overhead (**32K reaches ~1,400 tok/s; 8K reaches ~1,750 tok/s**).
3. **Step 3 — Register-Resident `k_resid_rows_opt`, `k_silu_mul_norm_rows_opt`, and Fused `k_combine_gate_norm_rows_opt` (~90 lines changed):**
   - Switch to 256-thread register-resident warp-direct `q8_emit` and omit dead FP32 writes to `bm_ffn_t` and `bm_attn_in`.
   - **Immediate gain:** Saves **`~62.5 us/tok`** (**32K crosses ~1,530 tok/s; 8K crosses ~2,000 tok/s**).
4. **Step 4 — `k_gemm_mmq_q5k` (`128×128` CTA tile, `2×8` warp tile) + Horizontal QKVG / Gate+Up Fusion + Native `m16n8k16.s8` in `k_gemm_mma_q6k` (~140 lines changed):**
   - Adopt the `128×128` `2×8` register-preloaded MMA loop with M-major CTA swizzling and multi-output QKVG/Gate+Up fusion.
   - **Immediate gain:** Cuts total GEMM time from `439.5 us/tok` (at `M=512`) down to **`345.6 us/tok`**, pushing **32K prefill to ~1,700 tok/s (`587 us/tok`) and 8K prefill to ~2,230 tok/s (`448 us/tok`)**.
