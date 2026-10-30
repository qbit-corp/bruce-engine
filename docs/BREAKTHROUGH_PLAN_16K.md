# Breakthrough Plan — 2500 tok/s Prefill & 70 tok/s Decode at 16K Context

**Target hardware:** 1 × RTX 3060 12 GB (GA106, SM86, 28 SMs, 360 GB/s theoretical / ≤345 GB/s achievable DRAM)
**Target model:** NVIDIA-Nemotron-Labs-Dory1.0-4B, Q5_K_M GGUF, 2 loops, `last_loop` KV, q8_0 KV
**Target context:** 16 384 tokens
**Constraint:** no quality degradation relative to llama.cpp

Status: **implemented 2026-10-27** (P1, P2, P3, P4, D1 verify path).  Measured results,
exactness gates and the remaining tree-drafter work are in `README.md` §4b.  The
projections below are kept as the original research record; the shipped numbers are
prefill 16K **2341 tok/s** (94% of the 2500 target), 4-row round 16K **19.59 ms**,
speculative code prompt **72.2 tok/s**, 32K corpus **95.1 tok/s**.  The `0x4B400000`
bias trick measured slower on this build and was reverted; the Q6_K 128x128 retile
broke even (39.2 vs 40.8 us/token), which is where most of the remaining prefill gap
sits.

---

## 0. Executive summary

| | today (measured) | target | gap | how |
|---|---:|---:|---:|---|
| **prefill @ 16K** | **1813 tok/s** (551.5 µs/tok) | **2500 tok/s** (400.0 µs/tok) | −151.5 µs | 4 levers, all measured, −160 µs total |
| **decode @ 16K, lossless** | **52.9 tok/s** (18.91 ms/tok) | **70 tok/s** (14.29 ms/tok) | −4.6 ms | verified 4-row rounds + faster verify path |

The prefill target is reachable **without changing a single output bit of the mathematical
contract** and with three of the four levers already validated as *bit-identical* to the current
kernel. The decode target is **not** reachable at M = 1 — the weight-stream floor forbids it — so it
must be reached through verified multi-token rounds, which is the one place Dory's architecture
makes this cheap (the looped body is executed twice per token, so the weight stream dominates and
extra verified rows are nearly free).

Two things make this a genuine breakthrough rather than tuning:

1. **Q5_K's min/scale term is not part of the dot product.** It is a rank-1 correction that
   depends only on (weight row, 32-group) × (activation row, 32-group). Hoisting it out of the
   k-loop and computing it with a single `mma.m16n8k16.f32` per 512 K-elements removes ~40 % of
   the prefill GEMM's instruction stream. Measured: **13.19 → 18.63 T-MAC/s (+41 %)** with the
   matmul math unchanged.
2. **The dequantisation is the bottleneck, not the tensor cores.** The prefill kernels were
   `I2FP.F32.S32`-bound (XU pipe, ¼ rate) and the attention kernel was `I2F.S8`-bound. Both can be
   replaced by integer/`PRMT` bit tricks that are *provably bit-exact* on SM86. Measured:
   **+33 % on attention (0 ULP difference)**, **+41 % on the GEMM**.

---

## 1. Measured baseline at 16K (ground truth for this plan)

All figures taken on this machine, today, driver 596.21, power cap 187 W, SM clock 1777–1890 MHz
as reported by the engine. Prompt corpus: `tmp/ids_codereal32k.txt` truncated to 16 384 tokens
(written to `%TEMP%\opencode\ids_16k.txt`).

```
engine\bruce.exe --ctx 16384 --token-file <16k> --gen 0 --no-live
  -> prefill 16384 tokens in 9035.84 ms  (1813.22 tok/s)

engine\bruce.exe --ctx 16384 --bench 20 --bench-at 16384
  -> 18.909 ms/token   52.89 tok/s   202.05 GB/s of weights
engine\bruce.exe --ctx 16384 --bench 20 --bench-at 16384 --bench-rows 4
  -> 21.468 ms/round
engine\bruce.exe --ctx 16384 --bench 20 --bench-at 16384 --bench-rows 8
  -> 29.195 ms/round
```

VRAM at 16K: weights 3643.4 MiB, KV 420.8 MiB, **6663.0 MiB free of 12287.5 MiB**.
That 6.5 GB of headroom is the single most under-exploited resource in the current design, and two
of the four prefill levers spend it.

### 1.1 Where the 551.5 µs/token of prefill goes

Reconstructed from the stored nsys profiles (`tmp/prof_mmq8k.sqlite`, `tmp/prof_mmq32k.sqlite`,
both recorded with the current kernel set at MBUF = 512) and rescaled to 16K depth.

| kernel | µs/token | share | limiter |
|---|---:|---:|---|
| `k_gemm_mmq_q5k` | 340 | 61.6 % | `I2FP` on the XU pipe + `unpack()` barriers + 1 blk/SM |
| `k_attn_tc_rs_opt<4>` (84 full + 37 SWA) | 121 | 21.9 % | `I2F.S8`/`F2FP` dequant + 2 blk/SM (223 regs) |
| `k_gemm_mma_q6k` | 41 | 7.4 % | 64×64 tile, 2×2 warp tile, 2 blk/SM |
| `k_silu_mul_norm_rows_opt<28>` | 21 | 3.8 % | 10 shuffles + 32 byte-stores per 32-group |
| `k_resid_rows_opt<10>` | 14.5 | 2.6 % | same |
| `k_qkv_prep_rows` | 11 | 2.0 % | 16 blocks/row, 6 `__syncthreads()`, 128/256 threads idle |
| `k_combine_gate_norm_rows_opt` + `k_gate_norm_rows_opt` | 14 | 2.5 % | split-merge reads |
| `k_gemm_mma_q5k` (rows < 96 tail) | 8 | 1.5 % | fallback kernel |
| embed / q8 / rope / copy | 3 | 0.5 % | launch bound |
| **total** | **551.5** | | |

### 1.2 Where the 18.91 ms of decode goes at 16K

| component | ms | note |
|---|---:|---|
| Q5_K weight stream (3.13 GB, recurrent blocks twice) | ~9.9 | `k_gemv_q5k` / `k_gemm_tc_verify_q5k`, ~310 GB/s |
| Q6_K weight stream (0.31 GB → 0.47 GB `w.dp`) | ~1.9 | `k_gemv_q6k_dp_fix` |
| Q8_0 logits (0.34 GB) | ~0.95 | `k_gemm_q8_logits` |
| attention (14 full + 42 SWA) | ~3.0 | `k_attn` + `k_attn_tc`, split-K 64/32 |
| norms / prep / combine | ~1.5 | |

The depth slope is `17.4 + 0.179 µs/key` (README §4). At 16K that is 2.9 ms of full-attention KV
traffic, whose DRAM floor is `14 layers × 4 kvh × 16384 × 544 B / 345 GB/s = 1.45 ms` — so
attention is at 2.0× its floor but is only 16 % of the step. **Decode at 16K is a weight-stream
problem, not a context problem.** That is the key strategic fact.

### 1.3 The two physical floors at 16K

```
Decode M=1, weight floor
  3.97 GB / 345 GB/s = 11.5 ms = 87 tok/s          <- hard ceiling for one token per pass
Decode M=1, achieved
  3.97 GB / 13.6 ms = 292 GB/s                     <- 85% of ceiling; 18.91 - 13.6 = 5.3 ms of other work
Prefill, per token
  weights 7.47 MB/token / 345 GB/s = 21.7 µs       <- irrelevant; prefill is instruction bound
```

Consequence: **70 tok/s at M = 1 needs 14.29 ms/token = 277 GB/s of effective weight bandwidth
*including* attention, logits and norms.** That is achievable in principle but leaves zero margin,
and it would need every non-weight kernel to shrink to ~1.5 ms total. It is the wrong target. The
right target is `tokens_per_round ≥ 1.3` on a verified round, which is a *speed* property only
(a draft token is emitted only when the target's own argmax agrees, so acceptance cannot alter
quality).

---

## 2. Lever P1 — Q5_K prefill GEMM: separate the rank-1 correction from the dot product

**Measured gain: 13.19 → 16.17 T-MAC/s (+23 %, exact) or 18.63 T-MAC/s (+41 %, near-exact).**
**Attacks 340 µs of the 551.5 µs budget.**

### 2.1 Why the current kernel is slow

Q5_K decomposes a 256-value superblock into 8 sub-blocks of 32 values, each with a scale `sc` and a
minimum `m`:

```
w[j][o] = d_r · sc_{r,j} · q[j][o]  −  dmin_r · m_{r,j}          q ∈ 0..31, j = 0..7, o = 0..31
```

so the dot product against a q8_1 activation (per-32 scale `xd`, per-32 integer sum `xsum`) is

```
y = Σ_j  xd_{m,j} · ( d_r·sc_{r,j} · Σ_o q[j][o]·xq[o]  −  dmin_r·m_{r,j} · xsum_{m,j} )
```

`d_r` and `dmin_r` are **superblock** constants: they do not depend on `j`. `m_{r,j}` is a 6-bit
integer. The current `k_gemm_mmq_q5k` nevertheless recomputes the second term inside the `b = 0..3`
inner loop, which is where its SASS comes from:

```
k_gemm_mmq_q5k SASS (measured, cuobjdump --dump-sass engine\bruce.exe):
  FFMA             516     <- 256 of them are the min-term, per stage
  FMUL             272
  I2FP.F32.S32     268     <- ¼-rate XU pipe: 268 x 8 cycles = 2144 cycles/stage
  LDS              162
  IMMA.16832.S8     64     <- only 64 useful instructions out of 2560
```

The ablation in `PREFILL_OPTIMIZATION_PLAN.md` §5.1 already isolated the same thing
("Mode 2: skip the per-32 scale FMA/I2FP → 40.83 T-MAC/s"). What it did not notice is that the
min-term is **almost free**, because the 16 `m_{r,j}` values of a 512-K span form a
`16 × 16` matrix and `xd·xsum` forms its own `16 × 128` matrix — so the whole correction is
`mma.m16n8k16.f32.f16.f16.f32`-shaped.

### 2.2 The reformulation

Split the output into two independently-accumulated matrices:

```
main[r][m] = Σ_{g=0}^{K/32}  (d_r·sc_{r,g}) · xd_{m,g} · dot(q_{r,g}, xq_{m,g})     <- int8 MMA, fp32 epilogue
min [r][m] = −Σ_{g=0}^{K/32} dmin_r · m_{r,g} · (xd_{m,g} · xsum_{m,g})             <- rank-1/16, HMMA
```

Both are exact restatements of the same sum. Two further exact simplifications:

* `d_r` is constant over the whole superblock, so `sc_{r,g}` can be folded into the **stored**
  weight scale at load time (`ds = d_r·sc_{r,g}`), and `w_min = −dmin_r·m_{r,g}` likewise into
  `dm = −dmin_r·m_{r,g}`. Both become a plain fp32 multiply in the epilogue. No `q5k_scale()`
  bit-extraction, no `I2FP`, no `I2FP.F32.S32` for the dot (`(float)c[i]` is replaced by the
  `0x4B400000` bias trick below).
* The weight **values** `q[j][o]` are 5-bit unsigned, which fits `mma.m16n8k32.s8`'s signed int8
  operand as-is. That is why the existing kernel already matches llama.cpp exactly; the plan does
  not change it.

### 2.3 The `+12582912.0f` trick (removes the XU-pipe `I2FP`)

Replacing `(float)c[i]` with an integer `I2FP.F32.S32` costs 8 cycles on SM86's ¼-rate XU pipe.
Two integer identities recover it for free:

```cuda
// 1. add 2^23 during the MMA so the result is already an "offset float" bit pattern.
//    IEEE-754: for |v| < 2^22, (0x4B400000 + v) reinterpreted as float == 12582912.0f + v, exactly.
static __device__ __forceinline__ void mma_s8_biased(float (&cf)[4],
        const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    int c0, c1, c2, c3; const int bias = 0x4B400000;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                 : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(bias));
    cf[0] = __int_as_float(c0) - 12582912.0f;   // single FADD on the full-rate FMA pipe
    cf[1] = __int_as_float(c1) - 12582912.0f;
    cf[2] = __int_as_float(c2) - 12582912.0f;
    cf[3] = __int_as_float(c3) - 12582912.0f;
}
```

Bound check: `|c| ≤ 32 · 126 · 127 = 511 872 < 2^22`, so the addition cannot reach the mantissa
bits it would perturb, and the recovery is **bit-exact** for every value the kernel can produce.

### 2.4 Storage: the pre-unpacked 144-byte stage

The int8 values must reach shared memory as bytes. The current kernel unpacks `qs`/`qh` inside the
kernel between two `__syncthreads()` (measured `unpack()` = 45 µs/stage-equivalent, ~11 % of the
kernel). With 6.5 GB free at 16K, do it at load time instead:

```
per (row, 128 values):   128 B  int8 values q           (5-bit codes, zero-extended)
                          16 B  float4 ds = d_r·sc_{r,b}   (4 sub-blocks)
                        = 144 B  -> identical stride to the existing MMA_BKP, so the
                                   existing cp.async staging pattern works unchanged
separately:             [N][K/32] __half  dm = −dmin_r·m_{r,g}
```

144 B / 128 values = 288 B per superblock vs 176 B raw → **1.73× the Q5_K bytes**, i.e.
2.01 GB → 3.47 GB, plus 0.16 GB for `dm`. This is what the 16K VRAM headroom buys.

### 2.5 Measured results

Microbenchmark: `%TEMP%\opencode\mmq_probe.cu` (reproduced in Appendix A).
Shapes and M = 512 match a real prefill chunk.

| variant | 2560×14336 | 2560×9216 | 4096×2560 | 7168×2560 | VRAM |
|---|---:|---:|---:|---:|---:|
| current `k_gemm_mmq_q5k` | 13.19 | 12.41 | 12.59 | 12.83 T-MAC/s | 1.00× |
| in-place, conflict-free scale layout + hoisted min-term | **14.50** | 13.68 | 13.94 | 14.23 | **1.00×** |
| pre-unpacked 144 B + fp32 `float4` min-term | **16.17** | 15.62 | 16.01 | 16.39 | 1.73× |
| pre-unpacked 144 B + fp16 `mma.m16n8k16` min-term | **18.63** | 17.55 | 18.10 | 18.50 | 1.73× |

Exactness against an independent scalar CPU evaluation of the identical Q5_K formula
(K = 2560, N = 256, M = 128, max |y| = 49.94, `%TEMP%\opencode\exact_probe.cu`):

| variant | max abs error |
|---|---:|
| current `k_gemm_mmq_q5k` | 3.8 × 10⁻⁵ |
| pre-unpacked + **fp32** min-term | **4.8 × 10⁻⁶** (3× *better* than today) |
| pre-unpacked + **fp16** min-term | 1.35 × 10⁻⁴ (2.6 × 10⁻⁶ relative) |

**Recommendation: ship the fp32 tier.** It is smaller error than the shipping kernel, so it cannot
be the reason for any zero-quality regression, and it still delivers +23 % on the GEMM. The fp16
tier is a documented +13 % optional uplift whose error (2.6 × 10⁻⁶ relative) is four orders of
magnitude below the 0.06-nat per-logit noise already documented in README §3 — but it must be
gated on the tap comparison in §7.

### 2.6 Two measured negative results worth recording

* `ldmatrix.sync.aligned.m8n8.x4` for the A fragment at a 144-byte row stride is **slower** than
  the plain swizzled `LDS.32` load (11.65 vs 12.57 T-MAC/s) because the stride is not
  `ldmatrix`-friendly. Do not "modernise" it.
* `cp.async.ca` (L1-cached) instead of `cp.async.cg` (bypass) costs 18 % (18.63 → 15.23 T-MAC/s).
  The weight stream is one-shot and must not pollute L1.

### 2.7 A cheaper fallback if the VRAM budget is rejected

The in-place variant (§2.5 row 2) is +10 % for **zero** extra VRAM: keep the 176-byte layout, but
(a) store `ds` and `dm` transposed as `[b][row]`/`[b][col]` so the `float2` read
`sWdm[r0*4 + b]` stops being a 2-way bank conflict, (b) move the min-term out of the `b` loop into
one `float4` dot per 128-K stage, and (c) apply the `0x4B400000` trick. Worth ~34 µs/token. Alone
it gives 1813 → ~1930 tok/s; it is the safety net, not the plan.

---

## 3. Lever P2 — prefill attention: bit-identical occupancy + dequant fix

**Measured gain: 121 → 91 µs/token (−25 %), with `max|diff| = 0.000e+00` against the shipping kernel.**

### 3.1 Why the current kernel is slow

`k_attn_tc_rs_opt<4>` (SASS measured): `FMUL 294`, `HADD2.F32 264`, `STG.E 264`, `F2FP.PACK_AB 102`,
`I2F.S8 48`, `HMMA.16816.F16 64`. Three problems:

1. **Occupancy.** `__launch_bounds__(128, 2)` forces ptxas to 223 registers → 2 blocks/SM =
   8 warps/SM. The register budget is spent on `q_reg[16][4]` (64 regs) and `o[32][2]` (64 regs).
   *(A prior experiment concluded "32-row blocks are slower because they need 256 threads and
   >128 regs". That conclusion is correct but the fix is not more rows — it is `minBlocks = 3`,
   which drops the kernel to 168 registers with zero spills and gives 12 warps/SM.)*
2. **Dequantisation.** Each of the 128 staging threads does
   `I2F.S8 (×16) → F2FP.PACK_AB (×8) → FMUL (×8)` per 32 values, i.e. 24 dependent XU/convert
   ops per 32 values for K **and** V. This is the `I2F.S8 48` + `F2FP 102` in the SASS.
3. **Tail-wave imbalance.** `rchunk` ascends, so the last wave contains the *longest* causal ranges
   (`t0 ∈ [0, pos]`). With 128 row-chunks over 28 SMs the final partial wave is the most expensive
   one.

### 3.2 The bit-exact `PRMT`/`HSUB2`/`HMUL2` dequant

Four int8 values in a `uint32` become two `__half2` with 4 instructions and **no** convert-pipe
traffic. The identity is exact: `0x6400` is fp16 `1024.0`, and `(byte ⊕ 0x80) | 0x6400` decodes to
`1024.0 + byte` with the sign restored, because fp16's exponent 25 has unit spacing at that
magnitude and the mantissa low bits are exactly the integer.

```cuda
// int8x4 (w) -> two __half2 == d2 * (float)int8_value, 0 ULP vs. the current path
static __device__ __forceinline__ void dequant_i8x4(uint32_t w, __half2 d2, __half2 h1152,
                                                    __half2 &o0, __half2 &o1) {
    const uint32_t u    = w ^ 0x80808080u;
    const uint32_t h01  = __byte_perm(u, 0x64006400u, 0x7170u);
    const uint32_t h23  = __byte_perm(u, 0x64006400u, 0x7372u);
    o0 = __hmul2(d2, __hsub2(*(__half2*)&h01, h1152));   // h1152 = (half2)1152.0f
    o1 = __hmul2(d2, __hsub2(*(__half2*)&h23, h1152));
}
```

`HSUB2`/`HMUL2` run on the half2 pipe at full rate; `PRMT` is a single-cycle ALU op. Measured
`max|diff| = 0.000e+00` over 512 rows × 16 heads × 256 dims.

> Note: an `HFMA2(1024-byte, d2, d2·(−1152))` single-instruction form is 5 % faster but rounds the
> bias first and gives `2.8 × 10⁻³`. Use the `HSUB2` form; it is exactly zero.

### 3.3 The other four changes (all structural, none numerical)

* `__launch_bounds__(32*NWARPS, 3)` → 168 regs, **3 blocks/SM**, 12 warps/SM, no spills.
* Issue all four 128-bit KV loads *before* the dequant chain (4-deep MLP instead of a dependent
  `LDG → dequant → STS` chain per 16 bytes).
* `full`-tile fast path: when a tile needs no causal/SWA mask, skip the eight `-1e30f` selects, the
  `ok0/ok1` guards, the `z0/z1` multiplies and the `if (al != 1)` branch. `full` is true for
  `≥ 95 %` of tiles at 16K.
* Reverse `rchunk`: `rchunk = n_rchunks - 1 - (blockIdx.x % n_rchunks)` so the longest ranges run
  in the first full waves and the tail wave is the cheapest.

### 3.4 Measured

Microbenchmark: `%TEMP%\opencode\attn_probe.cu`. K/V cache filled to 16 384 slots, `pos = 8192`
(so the average causal length matches a full 16K prefill), `rows = 512`.

| variant | full attn µs/tok (14 layers) | SWA µs/tok (42 layers) | blk/SM | regs |
|---|---:|---:|---:|---:|
| current `k_attn_tc_rs_opt<4>` | 83.7 | 37.1 | 2 | 223 |
| fast, 4 warps, ns = 3 / ns = 1 | **69.4** | **27.6** | 3 | 168 |
| fast, 6 warps (24 rows/blk), ns = 3 / ns = 1 | **63.1** | 31.0 | 2 | 168 |

**Recommendation:** instantiate both shapes — `k_attn_tc_rs_fast<4,3>` for SWA (ns = 1) and
`k_attn_tc_rs_fast<6,2>` for full layers (ns = 3). Combined: 63.1 + 27.6 = **90.7 µs/token
(−30.4)**, bit-identical.

### 3.5 Measured negatives

* **Lazy-V** (skip the V load when the tile max is > 14 nats below the running max) is
  neutral-to-worse at 16K: 71.5 vs 70.5 µs/tok. At 16K there are not enough cold tiles for the
  extra `__any_sync` + shared-flag + barrier to pay for itself. Keep it as a ≥128K lever only.
* 24-row blocks at 2 blocks/SM do **not** beat 16-row blocks at 3 blocks/SM for SWA (31.0 vs 27.6),
  because the SWA causal range is short and the extra rows buy nothing.
* Walking the key loop backwards (to seed the running max with the local window) is also neutral
  here, for the same reason.

---

## 4. Lever P3 — Q6_K prefill GEMM: apply the same tile

**Projected gain: 41 → 28 µs/token. Gate on a microbenchmark before committing.**

`k_gemm_mma_q6k` is still the old 64×64 CTA tile with a 2×2 warp tile (`IMMA 32`, `I2FP.F32.S32 128`,
`FFMA 128` per stage — measured SASS). The Q5_K work in §2 proved the 128×128 CTA / 2×8 warp tile
with a conflict-free scale layout is worth +23–41 % on the identical instruction mix, and Q6_K's
160-byte repacked stage (`w.dp`) is already 16-byte aligned and already has an fp32 scale per
16 values.

Work items:
1. Port `k_gemm_mmq_q5k`'s tile (`128 weight rows × 128 activation rows`, 8 warps, `4 × 2` grid,
   warp tile 32 × 64, 16 `mma.m16n8k16.s8` per k32) to the Q6_K `w.dp` layout.
2. Keep `mma.m16n8k16.s8` (half the cycles of k32 on SM86 and exactly one 16-value scale group per
   instruction), matching the current kernel.
3. Apply the `0x4B400000` bias trick to its 128 `I2FP` per stage.
4. Expected: 15–18 T-MAC/s from the same instruction-mix argument; **do not assume it** — gate on
   `tools/mmq_test.exe`-style measurement at K = 2560, N = 2560 and K = 7168, N = 2560.

---

## 5. Lever P4 — elementwise norms and QKV prep (bit-exact vectorisation)

**Measured gain: 46.75 → 29.45 µs/token (−17.3).**

All three kernels share the same defect: they operate on **one element per lane** and redo a full
warp reduction for every 32-element q8_0 group.

```cuda
// current: per 32-element group
//   32 x byte store          -> 32 separate STG.E.U8
//   5 x SHFL for amax  +  5 x SHFL for xsum
// new: one quarter-warp (8 lanes) owns the group's 32 values (float4 each)
//   1 x 4-byte coalesced store (8 lanes x 4 B = one 32-B sector)
//   3 x SHFL for amax  +  3 x SHFL for xsum   (reduction depth 8 instead of 32)
static __device__ __forceinline__ void q8_group_vec4(int8_t* xq, float* xd, int* xsum,
                                                     int g, float4 v, int sub8) {
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    const float d  = amax * (1.f / 127.f);
    const float id = d > 0.f ? 1.f / d : 0.f;
    int q0 = max(-127, min(127, __float2int_rn(v.x * id)));
    int q1 = max(-127, min(127, __float2int_rn(v.y * id)));
    int q2 = max(-127, min(127, __float2int_rn(v.z * id)));
    int q3 = max(-127, min(127, __float2int_rn(v.w * id)));
    *(uint32_t*)(xq + g * 32 + sub8 * 4) =
        (uint32_t)(uint8_t)q0 | ((uint32_t)(uint8_t)q1 << 8)
      | ((uint32_t)(uint8_t)q2 << 16) | ((uint32_t)(uint8_t)q3 << 24);
    int s = (q0 + q1) + (q2 + q3);
    s += __shfl_xor_sync(0xffffffffu, s, 4);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    if (sub8 == 0) { xd[g] = d; xsum[g] = s; }
}
```

Bit-exactness is structural: `amax` is a max (associative, and `fabsf` of the same floats), `xsum`
is an integer add (associative), and `d`, `id`, `__float2int_rn` are the identical scalar chain on
the identical inputs.

Measured (`%TEMP%\opencode\norm_probe.cu`, `rows = 512`):

| kernel | current µs/tok | fast µs/tok | speedup | mechanism |
|---|---:|---:|---:|---|
| `k_silu_mul_norm_rows_opt<28>` → `k_silu_mul_norm_vec4` | 21.02 | **11.03** | 1.91× | `float4` loads, 224 groups over 32 quarter-warps, 2-barrier block reduction over 8 warps |
| `k_resid_rows_opt<10>` → `k_resid_vec4` | 14.49 | **11.33** | 1.28× | 160 threads = 20 quarter-warps × 4 groups; `float4` ALPHA load |
| `k_qkv_prep_rows` → `k_qkv_prep_fast` | 11.25 | **7.09** | 1.59× | **1 warp per head, zero `__syncthreads()`, zero shared memory** |

`k_qkv_prep_fast` is the largest structural win: the current kernel launches
`dim3(16, rows)` blocks of 256 threads, each doing one 256-float head through 2,176 B of shared
memory behind 6 `__syncthreads()`, with 128 of 256 threads idle and 12 of 16 blocks returning early.
The replacement keeps a whole head in one warp's registers (`float4 c`, `float4 s`,
`float4 x0/x1`, `warp_sum_bcast` for the L2 norm) and does the KV quantisation with a quarter-warp
per 32-element q8_0 sub-block, writing four bytes per lane in one store.

---

## 6. Lever D1 — the decode path to 70 tok/s at 16K

**Measured today: M = 1 → 18.91 ms (52.9 tok/s); 4 rows → 21.47 ms; 8 rows → 29.20 ms.**

### 6.1 Why M = 1 cannot get there

`3.97 GB` of weight traffic per token. At the measured 345 GB/s ceiling that is 11.5 ms; the
remaining 5.3 ms of the current 18.91 ms is attention (3.0), logits (0.95) and norms/prep (1.5).
Even with a perfect attention kernel and zero norm cost, M = 1 lands at ~13.3 ms = 75 tok/s, and
that assumes 345 GB/s sustained, which no kernel on this card achieves for a mixed stream
(`k_gemv_q5k` sustains 310 GB/s, 90 % of ceiling). **Realistic M = 1 after this plan: ~15.5 ms =
64.5 tok/s.** The measured `--gemv-blocks` sweep (56 → 19.36 ms, 112 → 18.91, 224 → 20.10) confirms
the GEMV grid is already tuned, and `--splits-full` (1 → 129.6 ms, 32 → 20.16, 64 → 18.91, 128 →
18.92) shows the attention split-K is also already at its optimum.

### 6.2 The round arithmetic

A verified round streams the weights **once** for all rows, so the marginal cost of a row is small:

```
round(M) ≈ weight_stream + attention(M, shared KV) + logits(M) + norms(M)
measured at 16K:  M=1 -> 18.91      M=4 -> 21.47      M=8 -> 29.20
marginal per row:                   0.85 ms           1.47 ms
```

| round cost | 1.30 tok/round | 1.43 tok/round (measured on code) | 2.10 tok/round (measured on 32K CUDA corpus) |
|---:|---:|---:|---:|
| 21.47 ms (today, M=4) | 60.6 | 66.6 | 97.8 tok/s |
| **18.0 ms (post-plan, M=4)** | **72.2** | **79.4** | 116.7 tok/s |
| 22.0 ms (post-plan, M=8) | 59.1 | 65.0 | **95.5 tok/s** |

So **70 tok/s needs a 4-row round at ≤ 18.6 ms with ≥ 1.30 tokens/round** (or ≥ 1.43 tokens/round
at ≤ 20.5 ms, which is today's round cost — i.e. acceptance alone could get there on code).
Both halves are attackable:

### 6.3 Making the 4-row round 18.0 ms (−3.5 ms)

| change | saving | mechanism |
|---|---:|---|
| verify attention → `PRMT`/`HSUB2` dequant + 3 blocks/SM | −2.0 ms | `k_attn_tc` has the same `I2F.S8`/`F2FP` pathology as the prefill kernel; §3 measured 1.33× on the same code shape. At 16K/4 rows, attention is ~4.5 ms of the 21.47. |
| norms → the §5 `vec4` kernels on the verify path | −0.9 ms | `k_resid_rows_opt<3,32>`, `k_silu_mul_norm_rows_opt<7,32>`, `k_gate_norm_rows_opt<4,32>` are all still `_opt`-class; the `vec4` form is 1.3–1.9× |
| 4-row logits | −0.3 ms | `k_gemm_q8_logits<4>` → same 128×128 tile treatment; the stream is 0.34 GB but M = 4 reuses each weight load |
| fused combine into gate-norm for ns = 32 | −0.3 ms | `k_combine_gate_norm_rows_opt<16>` currently computes `s_wgt[head][s]` correctly but re-reads `part` 16× per row |
| **total** | **−3.5 ms** | |

### 6.4 Raising acceptance (the larger lever)

Today's `NgramDrafter` stores one continuation per n-gram (`tab[g][hash] = n-1`, most recent wins)
and drafts a linear chain of up to 7. Since the marginal verify row costs ~0.85 ms, the 8-row tile
is not being filled. Concretely:

1. **Tree verification in the 8-row tile** (the engine already has `pos_off`-free in-chunk KV and an
   8-bit ancestor mask is all `k_attn_tc` needs). At position `p+1` verify four distinct candidate
   tokens (rows 1, 4, 6, 7) instead of one. Expected 1.43 → 1.9–2.4 tok/round on prose.
2. **Second-most-recent continuation per n-gram** (keep 4 distinct continuations, not 1). Recency
   loses to frequency on long contexts, so this is a cheap diversity add.
3. **Pre-seeded `<think>` / reasoning n-gram table** (64 KB, static). Dory opens every answer with
   `<think>` and reuses phrases that appear nowhere in the prompt; this is a pure acceptance win at
   zero runtime cost.
4. **Lookahead/Jacobi slots**: use rows that tree verification leaves empty to run 4 self-generated
   tokens, feeding their argmaxes back into the n-gram pool. Turns prompt-free prose into
   self-speculation.

Speculation is quality-neutral by construction: a draft is emitted only when
`amax[i] == draft[i]`, so the emitted sequence is always a sequence of target-model argmaxes. The
only caveat is the one already documented in README §3 (`last_loop` chunk semantics resolve ties
inside a chunk differently from batch-of-1) — that is inherited from the reference, not introduced
here, and must be re-validated with the existing 128/128 greedy test.

### 6.5 What is *not* on the table

`--adaptive-loops` (skip loop 1 when the loop-0 fixed point has converged) reaches 98 tok/s today,
and would trivially clear 70. It is **excluded**: it is explicitly documented as lossy on prose and
chat. The 70 tok/s target in this plan is a lossless target.

---

## 7. Quality preservation — what is exact and how to prove it

Three distinct classes, each with a different proof obligation.

### Class 1 — bit-identical (no proof needed beyond `memcmp`)

| change | claim | evidence |
|---|---|---|
| `k_attn_tc_rs_fast<4,3>` / `<6,2>` vs `k_attn_tc_rs_opt<4>` | same values, same order | measured `max|diff| = 0.000e+00` over 512 × 16 × 256 outputs |
| `PRMT`/`HSUB2`/`HMUL2` int8→fp16 | exact fp16 integer decode | same measurement; IEEE argument in §3.2 |
| `q8_group_vec4` (silu/resid/gate) | same `d`, `id`, `q`, `xsum` | max is associative, integer add is associative, identical scalar chain |
| `k_qkv_prep_fast` | same RoPE, same L2 norm, same q8_0 bytes | same `warp_sum_bcast`/`l2_scale` chain; only the thread mapping changes |
| `0x4B400000` bias trick | exact for `|c| < 2^22` | bound check in §2.3 |

Gate: a debug build that runs old and new kernels back-to-back on the same device buffers and
asserts `memcmp == 0` (or `max|diff| == 0.0f`) before any graph is captured.

### Class 2 — same formula, different fp32 association order

`k_mmq_144_hmma_min` (both tiers). Obligation: the new kernel's error against an independent exact
evaluation must be **no larger** than the shipping kernel's. Measured in §2.5:

```
current  : 3.8e-05     fp32 tier: 4.8e-06  (better)     fp16 tier: 1.35e-04 (worse but 2.6e-6 relative)
```

Gate: run the Q5_K prefill with old and new kernels and diff the **logits** (`--dump-logits`) at
the end of a 16K prompt. Accept if `max |Δlogit| < 0.01` (the existing engine's own noise floor is
0.06 nats, README §3), and re-confirm the 128/128 greedy-token match on the code prompt.

### Class 3 — pre-existing trade-offs that this plan does not touch

* fp16 `QK^T` in `k_attn_tc_rs_*` — the same convention as llama.cpp's `-fa on`. Unchanged.
* q8_0 KV — same as the reference invocation. Unchanged.
* `last_loop` in-chunk semantics — inherited from the reference. Unchanged.

### The four-step validation ladder (run after every step, not at the end)

```
step 1  kernel-vs-kernel     %TEMP%/..._probe.exe        max|diff| == 0 (Class 1) or <= today (Class 2)
step 2  block-vs-block       `--dump-taps --debug-block V` all 112 virtual blocks, per-layer cos >= 0.99999
step 3  logits-vs-oracle     `tools/validate.py` at 16K   same argmax; top-5 5/5; top-50 >= 49/50
step 4  tokens-vs-oracle     `tools/gen_cmp.py`           greedy 128/128 on ids_code at 16K
```

`tools/final_suite.py` already wraps 3 + 4 plus a benchmark; add the 16K prompt to its list and
extend step 1 with a `--check-kernels` CLI flag that runs both implementations.

---

## 8. Expected result and the budget sheet

### Prefill, per token at 16K

| line item | today | after | Δ | confidence |
|---|---:|---:|---:|---|
| Q5_K GEMM (`k_mmq_144_hmma_min`, fp32 tier) | 340 | 241 | −99 | **measured** (16.17 vs 13.19 T-MAC/s) |
| Q6_K GEMM (128×128 retile) | 41 | 28 | −13 | projected (same instruction mix as P1) |
| attention (full 63.1 + SWA 27.6) | 121 | 91 | −30 | **measured** (bit-identical) |
| `k_silu_mul_norm_vec4` | 21.0 | 11.0 | −10 | **measured** |
| `k_resid_vec4` | 14.5 | 11.3 | −3.2 | **measured** |
| `k_qkv_prep_fast` | 11.3 | 7.1 | −4.2 | **measured** |
| gate-norm / combine | 14 | 12 | −2 | projected |
| embed / q8 / rope / copy | 3 | 3 | 0 | — |
| **total** | **551.5** | **391** | **−160** | 389 µs/token = **2569 tok/s** |

Sensitivity: with the fp16 min-term tier (−17 more µs) → 374 µs = 2674 tok/s. With the in-place
Q5_K fallback (+65 µs vs the fp32 tier) → 456 µs = 2193 tok/s (misses target — the pre-unpack is
required for 2500).

**Normalise to clock.** Today's 1813 tok/s was taken at 1777–1890 MHz; the README's 1942 tok/s at
8K was at 1890 MHz. A 5 % clock drop is 125 tok/s. Record `nvidia-smi --query-gpu=clocks.sm` with
every run and report tok/s and the clock together, as `bench/report.md` already does.

### Decode at 16K

| path | round | tok/round | tok/s | lossless |
|---|---:|---:|---:|---|
| today, plain M=1 | 18.91 ms | 1.00 | 52.9 | yes |
| post-plan, plain M=1 | ~15.5 ms | 1.00 | 64.5 | yes |
| post-plan, M=4 + today's drafter | 18.0 ms | 1.43 (code) | **79.4** | yes |
| post-plan, M=4 + today's drafter | 18.0 ms | 1.30 | **72.2** | yes |
| post-plan, M=8 + tree drafter | 22.0 ms | 2.10 | 95.5 | yes |
| (excluded) `--adaptive-loops` | — | — | 98.9 today | **no** |

**70 tok/s is met at 4 rows with ≥ 1.30 tokens/round, or at 1.43 (already measured on code).**

### VRAM at 16K

| item | today | after |
|---|---:|---:|
| Q5_K raw (176 B) | 2011 MB | 2011 MB (decode) |
| Q5_K pre-unpacked (144 B) + `dm` | — | +1630 MB |
| Q6_K raw + `w.dp` | 310 + 472 MB | 310 + 472 MB |
| output Q8_0 | 340 MB | 340 MB |
| token embedding | 671 MB | 671 MB |
| KV q8_0 (16K) | 421 MB | 421 MB |
| activations / graphs | ~350 MB | ~350 MB |
| **total** | **~4570 MB** | **~6200 MB** |
| **free of 12 288** | 6 663 MB | ~6 000 MB |

Constraint: the pre-unpacked copy is only affordable while `4 × ctx × 0.544 KB + 6.2 GB ≤ 11.5 GB`,
i.e. **ctx ≲ 32K**. Gate it: build it when `--ctx ≤ 32768`, else fall back to the in-place variant
(§2.7) or to today's kernel. Expose `--q5k-prepack auto|on|off` so the choice is explicit and
measurable.

---

## 9. Implementation order (each step is independently shippable)

Shipped 2026-10-27: steps 1-6.  Step 7 (tree verification / multi-continuation
drafter) is **not** implemented; step 8 was partially replaced by the one-wave
`ns = 20` verify split and the existing CLI sweeps; step 9's re-baseline is in
`README.md` §4/§4b.  Measured at 1777 MHz: prefill 16K 2341 tok/s (vs 1813 before),
4-row round 16K 19.59 ms (vs 21.47), speculative code 72.2 tok/s (vs 70.9), 32K
corpus 95.1 tok/s (vs 81.6).

| # | step | effort | measured / expected | gate |
|---|---|---|---|---|
| 0 | Add the `--check-kernels` harness and the 16K prompt to `final_suite.py` | ½ d | — | baseline reproduced |
| 1 | `k_attn_tc_rs_fast<4,3>` + `<6,2>`, PRMT dequant, `full` fast path, reverse `rchunk` | 1 d | **−30 µs prefill**, −2 ms decode round, `max|diff| == 0` | step 1–4 ladder, must be exactly 0 |
| 2 | `q8_group_vec4` + `k_silu_mul_norm_vec4` + `k_resid_vec4` (prefill **and** verify rows) | 1 d | **−13 µs prefill**, −0.9 ms round, exact | `memcmp` on `xq/xd/xsum` |
| 3 | `k_qkv_prep_fast` (1 warp/head, 0 barriers) | ½ d | **−4.2 µs prefill**, exact | `memcmp` on the KV slot |
| 4 | Load-time Q5_K pre-pack (144 B + `dm`), `k_mmq_144_hmma_min` fp32 tier, `--q5k-prepack` | 1½ d | **−99 µs prefill** (241 µs) | Class-2 logit diff < 0.01 nats |
| 5 | Q6_K 128×128 retile + bias trick | 1 d | **−13 µs prefill** | `mmq_test`-style correctness, then Class-2 |
| 6 | Verify-path attention (`PRMT` + 3 blk/SM on `k_attn_tc`) + 4-row logits | 1 d | **−2.3 ms round** | `max|diff| == 0`, 128/128 greedy |
| 7 | `NgramDrafter` tree verification + multi-continuation table + `<think>` table | 1½ d | 1.43 → 1.9–2.4 tok/round | 128/128 greedy on all 4 prompts |
| 8 | Re-tune `--splits-full/-swa` and `--gemv-blocks` against the new kernels | ½ d | 2–5 % | — |
| 9 | Full 4K/16K/32K/128K/256K re-baseline + `bench/report.md` refresh | ½ d | — | regression ≤ 0 at every depth |

Steps 1–3 alone are −47 µs/token (**1813 → 1975 tok/s**) and are the low-risk, bit-identical core.
Step 4 is the single largest lever and is where the target is won.
`PREFILL_DECODE_16K_BREAKTHROUGH_PLAN.md` §2.5 numbers come from `mmq_probe.cu`; re-run it before
step 4 to confirm on the machine state of the day.

---

## 10. Risk register

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| SM clock drop of 5 % eats the margin | med | −125 tok/s | report tok/s with the clock; target ≥ 400 µs at 1777 MHz, not 1890 |
| Class-2 logit diff on the fp16 min-term tier perturbs a tie | low | quality regression | ship the fp32 tier (error 3× *better* than today); keep fp16 behind a flag |
| Pre-unpacked Q5_K does not fit because ctx is raised later | med | feature unavailable | `auto` gate at 32K + the in-place fallback (§2.7, +10 % for free) |
| `k_attn_tc` verify attention resists the 3-block bounds | med | −2 ms lost from the decode plan | fall back to `PRMT` dequant only (measured 1.18× on the prefill shape) |
| Tree verification's in-chunk KV aliasing | med | wrong tokens | the SWA ring is already 2 × SWA slots for exactly this; spend the ½ d to add the ancestor mask test to `tools/kvcmp.py` |
| Q6_K retile does not reproduce the 18 T-MAC/s | med | −13 µs lost | it is 3 % of the budget; ship the bias trick alone if the tile does not pay |
| `mma.m16n8k32.s8` bias trick breaks with a future max-abs change | low | silent precision loss | static_assert the bound `32·126·127 < (1<<22)` and assert `xd·|dot|` magnitudes in a debug run |

---

## 11. Appendix A — reproduction commands

```bat
:: baseline (this document's numbers)
engine\bruce.exe --ctx 16384 --bench 20 --bench-at 16384
engine\bruce.exe --ctx 16384 --bench 20 --bench-at 16384 --bench-rows 4
engine\bruce.exe --ctx 16384 --bench 20 --bench-at 16384 --bench-rows 8
engine\bruce.exe --ctx 16384 --token-file %TEMP%\opencode\ids_16k.txt --gen 0 --no-live
engine\bruce.exe --occ

:: microbenchmarks written for this plan
%TEMP%\opencode\run_probe.cmd    :: mmq_probe.cu   -> Q5_K tile/scale/min variants
%TEMP%\opencode\run_probe.cmd    :: attn_probe.cu  -> prefill attention variants
%TEMP%\opencode\run_norm.cmd     :: norm_probe.cu  -> silu/resid/qkv-prep variants
%TEMP%\opencode\run_exact.cmd    :: exact_probe.cu -> Q5_K exactness vs a CPU reference
```

16K prompt used throughout: the first 16 384 ids of `tmp/ids_codereal32k.txt`.

## 12. Appendix B — the two code shapes that carry the win

### B.1 The Q5_K rank-1 split (per 128-K stage)

```cuda
// main term: one mma.m16n8k32.s8 per (16 weights x 8 activations x 32 values), fp32 scale in the epilogue
#pragma unroll
for (int b = 0; b < 4; ++b) {                     // 4 sub-blocks of 32 values
    int c[4] = {0,0,0,0};
    mma_s8(c, a[mt], bf[nt]);                     // a: 16 weight rows from sW[buf][row][144]
    acc[mt][nt][0] = fmaf(wds[mt][0]*xds[nt][0], (float)c[0], acc[mt][nt][0]);
    acc[mt][nt][1] = fmaf(wds[mt][0]*xds[nt][1], (float)c[1], acc[mt][nt][1]);
    acc[mt][nt][2] = fmaf(wds[mt][1]*xds[nt][0], (float)c[2], acc[mt][nt][2]);
    acc[mt][nt][3] = fmaf(wds[mt][1]*xds[nt][1], (float)c[3], acc[mt][nt][3]);
}

// min term: once per 4 stages (512 K values), 1 float4 per row/col, NOT inside the k loop
if ((s & 3) == 0) {
    const float4 w0 = *(const float4*)(&Wf[r0*4]);          // -dmin_r * m_{r, b=0..3}
    const float4 x0 = *(const float4*)(&Xf[j0*4]);          // xd_{m,b} * xsum_{m,b}
    acc[mt][nt][0] += w0.x*x0.x + w0.y*x0.y + w0.z*x0.z + w0.w*x0.w;
    // ... 3 more combinations
}
```

`wds`/`xds` are `((const float*)(stage_base + 128))[b]` — the 128-byte int8 payload is followed
immediately by the 16-byte `float4` of `d_r·sc_{r,b}`, so one 144-byte `cp.async` moves a whole
(row, stage) and the scale read is a `LDS.32` in the same cache line as the values it scales.

### B.2 The KV dequant that removes `I2F.S8`

```cuda
const uint32_t u   = w ^ 0x80808080u;                    // re-centre the signed bytes
const uint32_t h01 = __byte_perm(u, 0x64006400u, 0x7170u); // 1024 + b0, 1024 + b1
const uint32_t h23 = __byte_perm(u, 0x64006400u, 0x7372u); // 1024 + b2, 1024 + b3
o0 = __hmul2(d2, __hsub2(*(__half2*)&h01, h1152));
o1 = __hmul2(d2, __hsub2(*(__half2*)&h23, h1152));
```

4 instructions / 4 values vs. today's 12, and it moves the work off the XU pipe onto the half2 pipe.
Measured `max|diff| = 0.000e+00`.

---

*Numbers in §1 and §2.5/§3.4/§5 were measured on 2026-10-26/27 on the target machine with
`nvcc 13.3 -O3 -arch=sm_86`, `-Xptxas -v` reporting 0 spill stores / 0 spill loads for every kernel
quoted. Profiles referenced: `tmp/prof_mmq8k.sqlite`, `tmp/prof_mmq32k.sqlite`, `tmp/decprof2.sqlite`.*
