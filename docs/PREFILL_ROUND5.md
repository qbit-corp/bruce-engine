# Bruce prefill optimization: 1101 -> 1343 tok/s at 128K, 2803 at 4K (RTX 3060, M=512)

> Implementation status (2026-10-28): merged in `engine/src/bruce.cu`.  Measured
> prefill **2803 / 2667 / 2500 / 2213 / 1343 tok/s at 4K / 8K / 16K / 32K / 128K**
> (512-row chunks, defaults), **1824 tok/s at 64K** and **805 tok/s on a
> 262,000-token prompt**.  The same build through the HTTP server reports
> 1343.2 tok/s at 128K.  Decode 14.42 / 15.92 / 21.00 ms per token at 4K / 32K /
> 128K (unchanged, -0.3 % to +0.1 %); 4-row verify 17.31 / 18.79 / 24.07 ms
> (unchanged).  Quality class: prompt paths keep the fp16-QK^T convention; 128/128
> code-prompt tokens and 16/16 and 24/24 long-prompt tokens identical to the
> pre-change / alternate-shape builds.

## 0. Target card and regime

```text
Hardware:  RTX 3060 12 GB, GA106 SM86, 28 SMs, 1875-1897 MHz, 187 W cap.
Measured:  int8 MMA 53.9 T-MAC/s, f16 MMA 26.9 T-MAC/s, LDSM.x2 2.0 cycles,
           MMA 4.02 cycles/instruction, DRAM ~345 GB/s, cp.async ~32.5 GB/s/SM
           (L2-hot, 16 B/thread copies), smem 128 B/cycle.
Model:     Nemotron-Dory 1.0 4B, Q5_K_M (3.13 GB Q5_K + 0.73 GB Q6_K repacked),
           16 q heads / 4 kv heads / 256 dim, 56 attention + 56 FFN virtual
           blocks, 2 loops, q8_0 last-loop KV.
Regimes:   decode M=1 | verify M=2..8 | prefill M=9..512.
Chunks:    M=512 (MBUF), 512-row prompt batches, last chunk variable.
Target:    raise the whole curve, flatten the 4K->128K drop, no decode change.
```

## 1. Executive summary

### 1.1 Ceiling argument

Depth-independent work per 512-row chunk (measured, `--probe --bench-rows 512`, last
chunk at 4K):

```text
qkv 40.5 + ffn gate/up 51.3 + ffn down 31.1 + wo 15.5 + elementwise 21.1 = 159.5 ms
-> 311.5 us/token -> 3210 tok/s ceiling with full attention forced to zero.
```

Measured 4K pass is 190.3 ms (357 us/token), i.e. full+SWA attention adds 60.8 us.
So at short depth the GEMM block is 73 % of the budget and attention is not the
lever; at 128K the same 161 ms block is only 28 % and attention is 72 %.  Both had
to be examined, but only attention can flatten the depth curve.

### 1.2 Ground-truth breakdown (per 512-row chunk, last chunk at depth)

| component | 4K | 32K | 128K | T-MAC/s | % of measured peak | bound by |
|---|---:|---:|---:|---:|---:|---|
| qkv gemm (Q5_K fused + v) | 40.5 | 40.8 | 41.1 | 18.3 | 34 % int8 | staging + epilogue |
| ffn gate+up | 51.3 | 51.5 | 51.7 | 20.5 | 38 % | staging + epilogue |
| ffn down (Q5_K + Q6_K) | 31.1 | 31.2 | 31.4 | 16.9 | 31 % | Q6_K 2x MMA + staging |
| wo | 15.5 | 15.7 | 15.9 | 12.1 | 22 % | small-N wave tail |
| attention | 30.8 | 121.4 | 418.2 | 17.3 / 18.8 | 64 / 70 % f16 | softmax chain + barriers |
| elementwise/norm/quant | 21.1 | 22.4 | 23.5 | - | - | ~150-400 GB/s |

### 1.3 The physics

```text
T_weights(512) = 3.83e9 / (512 * 345e9) = 21.7 us/token   [derived]
T_tc(int8)     = 5.14e9 / 53.9e12       = 95.4 us/token   [derived]
M_crossover    = 3.83e9 * 53.9e12 / (5.14e9 * 345e9) = 116 rows
GEMM measured  = 376 us/token  (M=512) -> 3.9x the int8 MAC floor
attention KV   = 14*4*131072*544/48 = 83.3 MB/token at 128K
                 -> 241 us/token at 345 GB/s; measured 817 us/token at depth
                 -> 30 % KV-bound, 70 % MMA/issue-bound
launches       = ~800 per chunk * 1.9 us = 1.5 ms = 3 us/token
```

`M=512` is 4.4x above the crossover, so weight streaming (21.7 us/token) is not the
wall at any depth; the measured GEMM limit is the per-CTA staging stream and the
per-32 scale epilogue.

### 1.4 Cumulative roadmap (measured end-to-end, same session)

| stage | 4K | 8K | 32K | 128K | decode impact |
|---|---:|---:|---:|---:|---|
| 0 baseline (defaults before this round) | 2782 | 2640 | 2182 | 1101 | - |
| 1 Q5_K prepack budget reserves the KV cache | 2802 | 2666 | 2182 | 1252 | 0 |
| 2 + 48-row deep-chunk attention | 2802 | 2666 | 2179 | 1272 | 0 |
| 3 + 16-thread-per-key KV dequant | **2803** | **2667** | **2213** | **1343** | 0 |

Stage 3 is +22 % over the old default at 128K and +7.3 % over the old
*forced-prepack* best (1251.4 -> 1343).  The 4K/8K rows do not move because those
chunks use the 24-row shape below 12K depth by design.

## 2. Guaranteeing zero decode/verify regression

* All changes are reachable only from `forward_batch`, `rows > 8` (prompt) or from
  the weight-load budget.  Decode (`M.ex_fwd` graph, `forward_launch`) and verify
  (`M.ex_verify`, `rows <= 8`) dispatch to the old instantiations
  (`k_attn_tc_rs_opt<4,3,*,*>`, `<6,2,*,*>`, `<12,1,*>` is never selected for
  `rows <= 8`).
* The `k_attn_tc_rs_opt` refactor is template-only: the non-DBUF instantiation has
  the same instruction stream (verified: `nodb 32K = 120.5 ms` vs 120.5 before the
  refactor) and the DBUF instantiation is default-off.
* Weight buffers (`w.d`), the 144 B pre-unpack (`w.dp`), and the KV layout
  (`[kvh][slot][272]` q8_0) are unchanged.
* Decode/verify matrix before -> after (graph replay, 3 reps):

| measure | 4K | 32K | 128K |
|---|---:|---:|---:|
| M=1 ms/token | 14.443 -> **14.421** | 15.954 -> **15.921** | 21.001 -> **21.004** |
| M=4 ms/round | 17.351 -> **17.309** | 18.760 -> **18.793** | 24.166 -> **24.072** |
| M=8 ms/round | 17.649 -> **17.508** | 19.671 -> **19.650** | - |

All deltas are inside the +-0.4 % run band, in both directions.

## 3. Lever 1: the Q5_K prepack budget (host, +14 % default at 128K)

### 3.1 The bug

`q5k_prepack_enabled()` returned `M.n_ctx <= 32768` for `auto`, and the allocation
check in `bind_weights()` compared the 3.31 GiB pre-unpack against free VRAM
*before* `alloc_buffers()` allocated the KV cache.  Result: the fastest GEMM path
(the 144 B stage-major pre-unpack, 40.8 ms vs 58.2 ms for qkv) was off at every
context above 32K even when 1 GiB would have remained free, and forcing it on at
256K produced 0 MiB free and a 15.7x slowdown from WDDM paging.

### 3.2 Fix

```cpp
const size_t kv_need = (size_t)9 * M.n_ctx * KV_POS_BYTES * 2 +
                       (size_t)27 * SWA_SLOTS * KV_POS_BYTES * 2;
const size_t reserve = kv_need + (size_t)512 * 1024 * 1024;
if (need == 0 || free_b < need + reserve) { ... skip ... }
```

and `auto` now always attempts the prepack (the budget decides).  Result: on at
4K/32K/128K, off at 256K; the forced path is protected too.

## 4. Lever 2: the full-layer prompt attention tile

### 4.1 Why the shape was wrong at depth

The row-split kernel keeps Q (64 regs) and O (64 half2 regs) per lane, so 168
registers allow at most 12 warps (384 threads) and one CTA/SM.  At 6 warps the CTA
covers 24 rows, so the KV cache is walked `rows/24` times per chunk and every
16-key tile pays a dequant + two barriers shared by only 24 rows.  At 128K the
KV floor is 241 us/token and the measured attention was 817 us/token, with the
KV read repeated 21x per chunk.

### 4.2 The redesign

1. 12 warps (48 rows) for full layers when `pos + rows >= 12288`; SWA keeps the
   4-warp/16-row/3-CTA shape and shallow full layers keep 6 warps/24 rows.
2. The raw q8 tile dequant now uses 16 threads per key (16 B of K and 16 B of V
   each) instead of 8 threads x 32 B.  With 12 warps, 256 threads participate
   instead of 128, halving the serial dequant chain between the barriers.  The
   arithmetic and the bytes stored are identical per element.

Per-CTA shared memory is unchanged at 25,600 B (one fp16 K+V pair 16,896 B + one
raw q8 K+V pair 8,704 B).  The 12-warp CTA is register-bound, not smem-bound.

### 4.3 Measured result (attention-only, `--probe` delta, ms per 512-row chunk)

| position | 3.5K | 8K | 16K | 24K | 32K | 128K |
|---|---:|---:|---:|---:|---:|---:|
| 24-row, 8-thread dequant | **31.1** | **47.9** | 77.5 | 107.0 | 134.8 | 482.1 |
| 48-row, 8-thread dequant | 34.9 | 49.0 | 73.7 | 98.4 | 130.8 | 460.6 |
| 48-row, 16-thread dequant | - | - | - | - | **120.5** | **417.3** |

Attention throughput 17.3 T-MAC/s at 32K and 18.8 at 128K of the 26.9 T-MAC/s
f16 peak.  A standalone probe (`tmp/ioprobe.cu`) shows `ldmatrix.x2` (2 cycles) and
`mma.m16n8k16.f16` (4.02 cycles) fully overlap at a 1:1 mix, so the residual 30 %
is the score -> softmax -> PV dependency chain plus two barriers per 16-key tile.

## 5. Excluded ideas (measured)

| idea | measured result | why rejected |
|---|---|---|
| two fp16 K/V buffers + one barrier/tile (`BRUCE_ATTN_DBUF=1`) | 32K 126.8 vs **120.5**; 128K 426.6 vs **419.6** | the dequant is already off the critical path; issuing it before the tile's MMA delays the 8 dequant warps rather than overlapping |
| 48-row block below 12K | 3.5K 34.9 vs **31.1**; 8K 49.0 vs **47.9** | a block then covers only a few KV tiles; the 48-row Q load dominates |
| split count 0/2/4/8/16 at 128K (48-row) | 417.1 / 420.3 / 419.6 / 420.4 / 420.3 ms | inside noise |
| forcing Q5_K prepack at 256K | 10.98 s vs **0.70 s** per chunk | 0 MiB free, KV cache paged by WDDM |
| 128x64 tile at 2 CTAs/SM (round 4, rechecked) | 2.7-11 T-MAC/s | 128-register budget spills the accumulator tile |
| 3-deep cp.async, 16 warps, no-activation-staging (round 4) | 17.1 / 15.5-16.0 / 12.98 T-MAC/s | staging is copy-path bound; the smem bounce is cheaper than 4 B strided global loads |
| bias-trick epilogue (round 4) | 17.2 vs 17.1 T-MAC/s | the epilogue is not conversion-bound on GA106 |

## 6. What is left (measured)

* **GEMM (139 ms/chunk, 73 % at 4K).**  Per CTA stage the cost is additive: MMA
  2058 + epilogue 1360 + LDS 1120 + staging 2170 = 6708 cycles vs 6250 measured.
  The 128x128 tile with 64 accumulators/lane is forced by the 64K register file;
  it minimises bytes/MAC at that output budget.  Overlap needs 2 CTAs at <=128
  regs.  The measured no-staging/no-epilogue ceiling is 40 T-MAC/s (2.1x).
* **Attention (418 ms/chunk at 128K).**  Remaining gap is the softmax dependency
  chain; more tiles in flight needs registers that do not exist (168 of 170 at
  384 threads).  Q6_K is 8 % of MACs and ~20 % of GEMM time (16-value scale group
  forces 2 MMAs per 32 values).
* **Elementwise (21-23 ms/chunk).**  The silu/resid/quantize kernels move ~3.3 GB
  per chunk at 150-400 GB/s; the per-element gate/up scales are the largest
  removable read (28.7 MB/chunk through L2) if folded into the GEMM epilogue.
* **`wo` is the one GEMM outlier:** 15.9 ms/chunk (12.1 T-MAC/s) where the other
  Q5_K shapes run at 18-20.5 and per-stage scaling from the standalone harness
  predicts ~10.5 ms.  Its grid is 4 x 20 = 80 CTAs, i.e. 2.86 waves on 28 SMs, so
  a persistent / stream-K schedule (or fusing wo with the next block's qkv, which
  cannot share the activation) is the remaining candidate; worth at most ~1.5 %
  end-to-end at 4K.

## 7. Reproduce

```bat
cd engine && build.cmd
:: prefill matrix (512-row chunks)
bruce.exe --ctx 8192  --token-file ..\tmp\ids_codereal4k.txt  --gen 1 --greedy --no-live
bruce.exe --ctx 8192  --token-file ..\tmp\ids_code8k.txt       --gen 1 --greedy --no-live
bruce.exe --ctx 16384 --token-file ..\tmp\ids_16k.txt          --gen 1 --greedy --no-live
bruce.exe --ctx 32768 --token-file ..\tmp\ids_codereal32k.txt  --gen 1 --greedy --no-live
bruce.exe --ctx 131072 --token-file ..\tmp\ids_code128k.txt    --gen 1 --greedy --no-live
:: phase breakdown at a depth (prints one "probe pos ... full" line plus deltas)
bruce.exe --ctx 131072 --probe --bench-at 130560 --bench 2 --bench-rows 512 --no-live
:: decode/verify invariants
bruce.exe --ctx 4096  --bench 40 --bench-at 4096  --no-live
bruce.exe --ctx 32768 --bench 20 --bench-at 32768 --bench-rows 4 --no-live
```

A/B knobs: `BRUCE_ATTN_FULL_NW=6` (old shape), `BRUCE_ATTN_NW_DEPTH=0` (force wide),
`BRUCE_ATTN_DBUF=1`, `--q5k-prepack off`, `BRUCE_ATTN_PIPE=0`.

## Appendix A: environment

RTX 3060 12 GB (GA106, 28 SMs), driver 596.21, CUDA 13.3, Windows 10 22H2,
Ryzen 5 5600G, 187 W cap.  SM clock during the reported runs 1875-1897 MHz,
temperature 51-68 C.  GGUF NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M (3,673,288,864 B).

## Appendix B: measurement notes

End-to-end prefill numbers are one 512-row chunk stream from the token files above
with `--gen 1`; the reported tok/s is `prompt tokens / prefill wall time`.
`--probe` numbers are CUDA-event timings of one `forward_batch(512)` pass after
three warm-ups, per-phase by disabling that phase (and by running it alone); they
are the *last* chunk at the stated fixed depth, so they are up to 2x the
prompt-average attention cost.  Decode/verify numbers are 3-rep medians of captured
graph replays.  All A/B runs interleaved baseline and candidate on the same machine
state; the spread of a repeated prefill is <= 0.8 %.
