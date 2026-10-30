# Bruce — a from-scratch CUDA engine for NVIDIA-Nemotron-Labs-Dory1.0-4B

> **This is the engineering log, kept for the measurement history and the reasoning
> behind each kernel.** For the short version, install instructions, and the benchmark
> summary, start at [`../README.md`](../README.md). Paths in the older sections refer to
> the author's local model directory.

Single-file C++/CUDA inference engine (`engine/src/bruce.cu`, ~4400 lines) specialised for
the **Dory 1.0 4B** looped-transformer architecture and its Q5_K_M GGUF, targeting a
single RTX 3060 12 GB.  No ggml, no PyTorch: own GGUF loader, own dequantising GEMV/GEMM
kernels, own KV cache, own attention, own sampler, own speculative decoder.

The reference semantics are pinned to the llama.cpp PR #30306 implementation in
`tools/doryref` (binary commit `57b70d0`); the engine is validated against it token by
token.

---

## 1. Build & run

```bat
engine\build.cmd                     :: nvcc 13.3 + MSVC BuildTools, -O3 -arch=sm_86
```

```bat
engine\bruce.exe --ctx 4096 --token-file tmp\ids_code.txt --gen 128 --greedy --print-ids
engine\bruce.exe --ctx 4096 --token-file tmp\ids_code.txt --gen 128 --greedy --draft 7
engine\bruce.exe --ctx 4096 --token-file tmp\ids_code.txt --gen 128 --greedy --draft 7 --adaptive-loops
engine\bruce.exe --ctx 262144 --bench 20 --bench-at 262144
```

### CLI

| flag | meaning |
|---|---|
| `--gguf PATH` | model path (default: the Q5_K_M file) |
| `--ctx N` | KV/context capacity (default 262144) |
| `--loops N` | number of recurrent loops (default 2) |
| `--tokens a,b,c` / `--token-file F` | prompt token ids (space/comma/newline separated) |
| `--gen N` | tokens to generate |
| `--greedy` / `--temp T --top-p P --seed S` | sampler |
| `--print-ids` | print ids one per line (default: space separated) |
| `--bench N --bench-at POS` | pure decode benchmark (graph replays) at a context length |
| `--bench-rows M` | in `--bench` mode, time one speculative verification round with M rows |
| `--prefill-batch N` | prefill chunk size (1..512; default 512, 8 = previous 8-row path) |
| `--draft N --draft-ngram L --draft-nmin K` | prompt-lookup speculation |
| `--attn-tc N` | tensor-core verify attention (default 1, `0` = scalar fp32 path) |
| `--prefill-tc N` | prompt rows share one tensor-core KV pass on full layers (default 1; `0` = bit-exact per-row) |
| `--q5k-prepack auto\|on\|off` | load-time 144 B Q5_K pre-unpack for the prompt GEMM (default `auto`, enabled at ctx ≤ 32K; costs ~3.3 GB VRAM) |
| `--adaptive-loops` | skip loop 1 when the loop-0 fixed point has converged (`--loop-tau`, default 0.04) |
| `--no-adaptive-loops` | force both recurrent loops (default) |
| `--loop1-reuse` / `--no-loop1-reuse` | reuse the loop-0 attention output for the 5 recurrent full-attention blocks in loop 1 (default on, decode/verify only; §4f) |
| `--loop-tau T` | relative loop-0→loop-1 step threshold for `--adaptive-loops` |
| `--occ` | dev diagnostic: blocks/SM, registers, shared memory per hot kernel |
| `--probe` | dev diagnostic: attribute a decode pass per phase (`--bench` = reps, `--bench-rows` = rows) |
| `--splits[-full/-swa] N` | attention split-K factor (defaults 64/32, 128/32 for ctx>=64K) |
| `--gemv-blocks N`, `--gemm-blocks N` | grid overrides |
| `--no-graph`, `--force-batch` | debug: disable CUDA graph / force the batched path |
| `--dump-logits F`, `--top-logits K`, `--dump-x F`, `--dump-taps F --debug-block V` | debug taps (`*.late` variants dump after generation) |
| `--serve` | start the OpenAI-compatible HTTP server |
| `--host H --port P` | server bind address (default `127.0.0.1:8080`) |
| `--slots N` / `--parallel N` | concurrent serving slots, each with its own KV partition (default 4 when `--ctx >= 16384`, else 1; per-slot context = `--ctx / N`) |
| `--alias NAME` | model id returned by `/v1/models` (default `bruce`) |
| `--api-key KEY` | require `Authorization: Bearer KEY` or `X-API-Key: KEY` |
| `--n-predict N` | default `max_tokens` for server requests (default 256) |
| `--thinking` / `--no-thinking` | default chat reasoning mode (default on) |
| `--medium-effort` / `--no-medium-effort` | append `{reasoning effort: efficient}` to the last user message |
| `--truncate-thinking` / `--no-truncate-thinking` | strip reasoning from historical assistant turns (default on) |
| `--cache-prompt` / `--no-cache-prompt` | prefix-cache default for requests (default on) |
| `--no-live` | disable live prefill/decode status lines |

---

## 2. Architecture implemented

* 72 physical blocks: 36 attention (even) + 36 FFN (odd), executed as **112 virtual
  blocks** with 2 loops (order: 0-15, 16-55, 16-55, 56-71).
* `n_embd 2560`, 16 heads / 4 KV heads, `head_dim 256`, FFN 7168, vocab 131072.
* Every attention block: RMS-normalised input -> Q5_K `q/k/gate` + Q6_K `v` -> RoPE
  (theta 1e7 full / 1e4 SWA) -> cache -> attention -> Q5_K `output` -> gate (`sgate`) ->
  `resid` (normalise x, normalise upd, `lerp(alpha)`, normalise) -> q8 activations.
* FFN: Q5_K `gate/up` -> fused silu-mul-norm (`suv_gate`, `suv_up`) -> Q6_K `down` -> resid.
* Loop-boundary `input_h` injection exactly as in the reference.
* Precomputed per-layer scales (`sqk`, `sgate`, `suv_*`, `alpha`, `out_scale`) folded out of
  the graph at load time.

### KV cache

* **q8_0, per position, head-major**: K = 4 kvh x 8 sub-blocks x (fp16 scale + 32 int8) = 1088 B,
  V likewise.  The buffers are laid out `[kvh][slot][272]`, so a block that owns one KV head
  reads one contiguous 272 B record per slot instead of 272 B out of every 1088 B.
  Layout per kv head is `[8 fp16 scales = 16 B][8 x 32 int8 = 256 B]`, not the usual
  interleaved 34-byte sub-blocks: putting the scales first keeps the quantised data
  8-byte aligned, so a lane fetches its eight weights with one 64-bit load instead of
  eight byte loads (the 34-byte stride spread one warp access over 8+ sectors).  Same
  1088 B/position, same quantisation, bit-identical results; worth **1.5x** on long-context
  decode (262144: 96.8-103.3 -> 64.2 ms/token) because the reader is instruction bound.
* `last_loop` mode (the server's mode): recurrent blocks only keep the last loop's KV.
* Full-attention layers: one slot per position.  SWA layers: ring of `2*1024` slots so a
  speculative/prefill chunk can never alias a live window position.

### Kernels

* GEMV: int8 activations (per-32 scale + per-32 sum) with `__dp4a`; one row per warp,
  swizzled shared-memory activation copy, 8 warps/block; Q5_K stays in its GGUF layout.
  Decode horizontally fuses `q/k/g/v` and `gate+up` into one launch when the tensors are
  Q5_K, so the N=1024/2560/4096 wave tails and 4 launches per block disappear.
* Prompt GEMM (`k_gemm_mmq_q5k_144`, `--q5k-prepack auto`): int8
  `mma.m16n8k32.s8`, **128 weight rows x 128 activation rows** per block, 8 warps in a
  4x2 grid (warp tile 32x64 = 16 MMA per k32), 256 threads, 96 KB shared, occupancy 1.
  At load time each Q5_K tensor is re-encoded on the GPU into a 144 B stage per
  (row, 128 values): 128 B of int8 values (the stored 5-bit codes, zero-extended) plus
  16 B of `float4 ds = d*sc`; a separate `__half` array holds `dm = -dmin*m` per 32-group.
  `cp.async` therefore stages the weights without any in-kernel unpack or qh staging,
  and the min term is a rank-16 fp16 `mma.m16n8k16` accumulated once per 512 K values
  outside the int8 dot loop.  The epilogue is one I2FP + two FMA + one multiply per
  output.  Up to four Q5_K matrices that share an activation tile are fused into one
  launch (q/k/gate, gate+up), with a block-level part lookup and M-major CTA ordering
  for L2 reuse.  Costs ~3.3 GB VRAM and is enabled automatically at ctx ≤ 32K
  (`--q5k-prepack on|off` overrides); otherwise the original 128x128 raw MMQ kernel
  and the 64x64 `k_gemm_mma_q5k` tail remain as fallbacks.  Verified against a CPU
  reference (max |diff| ~7e-5 on outputs up to 72, no non-finite values).
* Prompt q6_K (`k_gemm_mma_q6k_128`): the load-time lossless 160 B repack (128 int8 +
  8 fp32 scales per 128 values) feeds native `mma.m16n8k16.s8` on a 128 weight x 128
  activation tile, one instruction per 16-value scale group.  The 96 KB stage costs the
  second block per SM, so the retile measures 39.2 us/token versus 40.8 for the old
  64x64 kernel: a small win, kept because it is lossless (max |diff| ~1e-6).
* Batched (M = 2..8, used by speculative verification and the tail of a prompt chunk):
  `k_gemm_tc_verify_q5k<M>` is the decode/verify path: a barrier-free, register-direct
  `mma.m16n8k32.s8` GEMM with the 16 weight rows on operand A and up to 8 verified rows on
  operand B.  The quantized activation tile and its pre-combined scales are staged in
  shared once; weights stream from global straight into registers, so the kernel has one
  `__syncthreads()` and the marginal cost of a verified row is ~0.06 ms (was ~2.9 ms).
  Up to four Q5_K matrices that share the activation tile are fused into one launch.
  Since §4g the whole 176 B superblock of both weight rows is prefetched into registers
  one iteration ahead of the MMA, which is what keeps the stream at 278-316 GB/s (a
  depth-2 buffer spills and costs 2x, see §4g).  The prefill path still uses the 128x128
  MMQ kernel when the chunk has >= 96 rows.
* Q6_K verify rows 6..8: `k_gemm_tc_verify_q6k` is the same structure on the aligned
  160 B repack with `mma.m16n8k16.s8` (one instruction per 16-value scale group).
  Rows 3..5 keep the raw 210 B batched kernel, whose 34 % smaller stream still wins
  there.  `BRUCE_Q6TC=0` restores the old choice for A/B.
* Q6_K decode (`k_gemv_q6k_dp`): reads the load-time aligned repacked form (128 int8 +
  8 fp32 scales per 128 values), one 16-value scale group per lane, so each weight byte
  is fetched with a 16-byte `uint4` and feeds exact `dp4a` dots.
* Logits: `output.weight` is quantized at load time from BF16 (671 MB) to head-aligned
  Q8_0 (`k_gemm_q8_logits`: 16 B of fp16 scales + 256 B of int8 per 256 values,
  356.5 MB).  The final hidden state is already q8_0, so the logit pass is dp4a over a
  1.88x smaller stream and the BF16 buffer is freed (saves 300 MiB).
* Decode/verify attention: split-K flash-style online softmax, 4 heads per block, int8 KV
  dot products, separate combine kernel.  Split factors tuned (see §5).
* Prompt attention (`k_attn_tc_rs_opt<4,3>` / `<6,2>`): row-split layout, each warp owns
  4 rows x 4 q-heads and walks all 256 dims, Q held in registers (64 regs/lane), K/V
  fragments from `ldmatrix` / `ldmatrix.trans`, fp16 `mma.m16n8k16` for both QK^T and PV
  with half2 output accumulators.  The KV dequant is the exact
  `PRMT 0x6400 + HSUB2 + HMUL2` form (no I2F.S8/F2FP on the XU pipe).  Windowed layers
  run 4 warps, 16 rows/block, 3 blocks/SM (168 regs); full layers run 6 warps, 24
  rows/block, 2 blocks/SM.  Blocks are ordered kv-head-major and the row chunks are
  reversed so the longest causal ranges run in the first full wave.  The fused
  `k_combine_gate_norm_rows_opt` consumes the split partials directly in registers, so
  `bm_aout` and `bm_attn_in` are not materialised.
* Speculative verification (2-8 rows) uses the same row-split kernel (`BRUCE_RS_VERIFY`,
  default on) with the split count capped at 20 so the grid is one wave.  This replaces
  the dim-split `k_attn_tc` verify kernel, which recomputed QK^T once per warp; it is
  0.7-7.5 ms/round faster from 4K to 256K.  `--attn-tc 0` still selects the scalar
  `k_attn_m<M>` fallback.  Plain M=1 decode deliberately keeps the original scalar
  `k_attn` + split combine + shared-memory norm path so it stays bit-identical to the
  pre-plan engine.
* Decode is captured in a **CUDA graph** (~700 nodes) with a device-side position counter;
  the sampler runs outside the graph.
* Loop-1 attention reuse (§4f): the five recurrent full-attention blocks keep their
  loop-0 attention output (640 KiB table) and loops >= 1 read it instead of streaming
  the same KV cache again.  Decode/verify only; prefill never enters the path.
* Speculative verification (`M = 2..8`) is captured in its own CUDA graphs
  (`M.ex_verify[2..8]`, plus `ex_verify1` with loop 1 skipped), so a round is one graph
  launch instead of ~800 CPU kernel launches.
* `--adaptive-loops`: at the end of loop 0 a device kernel measures
  `||x[95] - x[55]|| / ||x[95]||` (2-loop rounds) and the next token/round replays a graph
  with loop 1 skipped when it is below `--loop-tau`; a 2-loop probe runs every 8 skipped
  tokens.  The 1-loop graph is exactly the `--loops 1` block schedule (same block order,
  same loop-boundary norms).  On the code prompt this is 128/128 token-identical to the
  2-loop engine; on prose/chat it is a lossy speed mode (coherent continuation, different
  sampling).  Prefill never uses it.

---

### Tensor-core verify attention

`k_attn_tc` (full and windowed layers, 2-8 verify rows) packs the four heads of a KV head and up to four
rows into one 16-row `mma.m16n8k16` tile (`pair = row*4 + head`), so the MMA row dimension
carries the draft rows for free and each key is dequantized once per 16-key tile and reused by
all of them.  One block per (KV head, split), four warps, each owning 64 of the 256 output dims
and its own 32 fp32 accumulators; QK^T is recomputed per warp because at this arithmetic
intensity that is still cheaper than a cross-warp reduction.  K/V arrive as q8_0 and are
unpacked, scaled and written as fp16 shared tiles by all 128 threads (`uint4` stores, 528-byte
padded row stride, conflict free for both the QK and the PV fragment loads), the causal mask is
applied to the score fragments before the max, and a fully masked row keeps its running max
instead of producing a NaN.  Splits merge through the same `(O, m, l)` partials as before.

Measured effect on a verify round (see §4): attention drops from 64 ms to 17 ms at 128K/4 rows
and from 20 ms to 5 ms at 32K/4 rows, which puts the kernel back on the DRAM floor.

Score arithmetic follows the fp16 flash-attention convention that llama.cpp uses with `-fa on`
(q8_0 dequantized to fp16 operands, fp32 accumulate).  A verify pass can therefore branch away
from the fp32 plain path where the top-2 logits sit within ~1e-3 of each other.  That is
rounding, not a quality change: the fp32 plain path by itself already branches at token 49 of
the prose prompt when only the KV split count changes (`--splits-full 1` vs 64, i.e. pure fp32
reassociation), and the verify path reproduces plain decode 128/128 on the code prompt in both
split modes.

### Weight loading

The GGUF is no longer left mapped while the weights are read.  On Windows a live
mapping makes every `ReadFile` on the same file run at raw disk speed (measured
**392 MB/s vs 7 400 MB/s** from the OS cache), so the loader parses the metadata
from a 32 MB prefix buffer and streams the tensor bytes through its own
`FILE_FLAG_SEQUENTIAL_SCAN` handle.

Weights are read **exactly once**, in file-offset order, into 5 x 16 MB pinned
staging slots, and a single copy thread feeds one async stream.  All CUDA calls
live on that thread: on WDDM a pinned `cudaMemcpyAsync` holds the driver API lock
until the copy is staged, so issuing it (or polling its event) from the reader
thread serialises the disk read with the DMA.  Q6_K is repacked to the 160 B/128
layout by two worker threads while the next chunks are still being read; the BF16
logits head is quantised to Q8_0 on the GPU (`k_bf16_to_q8`, one warp per 32-value
group) from a transient 671 MB buffer that is freed before the KV cache is sized.
The previous path also uploaded the BF16 head (671 MB) only to free it, read Q6_K
twice (raw + repack), and quantised the head in a ~5 s scalar host loop.

Measured on the 3.67 GB Q5_K_M file (RTX 3060, SATA SSD at ~400 MB/s):

| load | before | **after** |
|---|---|---|
| cold OS cache (first load after boot / other file activity) | 17.0 s | **9.9 s** (9.2 s of it is the disk read at its 400 MB/s ceiling) |
| warm OS cache (restart, file resident) | 17.0 s | **1.2 s** (0.7 s read at 5.2 GB/s, 0.14 s slot wait) |

The bytes that land in VRAM are identical (same repack/quantise functions, same
rows); greedy 128-token output on the code prompt is 128/128 identical to the
pre-change build for plain and `--draft 7`, and the CPU/GPU logits quantisers
produce identical tokens.  Prefill and decode are untouched: 2790 / 2674 / 2499
tok/s at 4K / 8K / 16K and 13.96 ms/token at 4K.

The model file on this machine lives on a SATA SSD (~400 MB/s), which is the
cold floor.  The same file on the NVMe reads at ~2.0 GB/s and loads cold in
**~2.5-3 s** instead of 9.9 s; warm loads are already cache-bound at 1.2 s.
Beyond that, the only way to cut a cold load is to read fewer bytes.

`BRUCE_LOAD_PROF=1` prints the phase breakdown (open, tokenizer, pread, slot wait,
scales, drain).  `BRUCE_Q8_GPU=0` selects the old scalar BF16→Q8_0 host path.

## 3. Correctness

Validated against the pinned llama.cpp reference on the same weights, KV type (q8_0) and
KV mode (`last_loop`, 2 loops).

| check | result |
|---|---|
| Prefill logits, 105-token prompt (top-5 ids / logprobs) | identical ids, deltas <= 0.28 nats |
| Per-layer activations (all 112 blocks, batch vs sequential) | bit-identical |
| **Greedy generation, 121-token code prompt, 128 tokens vs `llama-server`** | **128/128 tokens identical** |
| Breakthrough build, 121-token code prompt, plain and `--draft 7` | **128/128 identical to the pre-plan engine** |
| Breakthrough build, 16K prompt logits | same argmax, top-5 5/5, top-50 49/50, max abs delta 0.56 |
| Greedy generation vs server, chat / prose prompts | 51/128 and 74/128 before the first tie |
| Full-logit vector at a divergence point | cosine 0.99976 / 0.99992, centred std 0.12 / 0.06 nats |
| Speculative greedy vs plain greedy (code prompt) | 128/128 identical |
| Speculative greedy + `--adaptive-loops` vs plain 2-loop (code prompt) | 128/128 identical |
| Logits after BF16→Q8_0 output quantisation | same argmax and top-5, max abs delta 0.117 |
| `--adaptive-loops` on prose / chat | coherent continuation, different sampling (lossy speed mode) |
| Prefill with shared-KV attention vs per-row (`--prefill-tc 0`), 8K real code | same argmax, top-5 4/5, top-50 48/50, max abs logit delta 0.50 (logits to ~26), centred std 0.076 nats |

The 128-token exact match on the code prompt is the strongest evidence: same prompt,
same greedy sampler, same KV quantisation, token-for-token identical text.

Divergences on other prompts occur only where the reference's own top-2 logits differ by
**0.003–0.12 nats** (measured with the sequential `tools/oracle_dump.exe` reference);
our per-logit numerical noise is ~0.06 nats, so these are coin flips, not errors.  A
full-logit comparison at such a point is in `tools/pos_cmp.py`.

### Semantics note (chunked verification)

In `last_loop` KV mode, tokens inside a *chunk* see the in-chunk positions through the
earlier loop's KV.  This is exactly what llama.cpp does for `n_ubatch > 1` (and therefore
for its own speculative verification), and the prefill path matches it bit-for-bit.  It
means speculative verification can resolve a near-tie differently than a strictly
batch-of-1 decode; it never submits a non-target token.

---

## 4. Performance (RTX 3060 12 GB)

> **Current numbers are in §4h** (plain decode 69.3 / 62.8 / 47.6 tok/s at
> 4K / 32K / 128K; a 4-row verify round costs 17.3 / 18.8 / 24.1 ms; prefill
> 2803 / 2667 / 2500 / 2213 / 1343 tok/s at 4K / 8K / 16K / 32K / 128K, and
> 805 tok/s on a 262K-token prompt).  Multi-slot serving numbers are in §4i
> (341 tok/s aggregate at 8 slots / 4K, up from 306).
> The tables in §4 and §4b-§4g are the historical record of each round.

| mode | tok/s | notes |
|---|---|---|
| prefill, 4K / 8K / 16K / 32K / 128K | **2796 / 2693 / 2500 / 2184 / 1101 tok/s** | 512-row chunks, stage-major Q5_K MMQ + row-split attention (§4d) |
| decode, greedy (121-token code prompt) | **66.6 tok/s** | 15.0 ms/token |
| decode, temp 0.8 / top-p 0.9 | 66.2 tok/s | sampler outside the graph |
| decode, speculative (code / chat / prose) | **72.2 / 68.4 / 66.8 tok/s** | 1.43 tokens/round (drafter-limited) |
| decode, speculative + `--adaptive-loops` | **98.9 / 95.2 / 95.1 tok/s** | loop-1 skip (lossy on prose/chat) |
| decode, speculative 32K CUDA corpus | **95.1 tok/s** (2.10 tok/round) | 21.9 ms/round |
| ... + `--adaptive-loops` @32K | **122 tok/s** (2.40 tok/round) | 2-loop probe every 8 tokens |
| decode, speculative (repetitive 8K code corpus) | **281 tok/s** | 7.00 tokens/round (64-token sample) |

The prompt path now uses the load-time Q5_K pre-unpack (144 B per 128 values + fp16
min-term scales) in a stage-major layout, the row-split attention kernel at 3 blocks/SM for
SWA and 2×6-warp blocks for full layers, float4 quarter-warp residual/SiLU norms and a
one-warp-per-head QKV prep.  Measured prefill is **2796 / 2693 / 2500 / 2184 tok/s at
4K/8K/16K/32K** (1777 MHz machine state) and **1101 tok/s at 128K**, up from
2047 / 1950 / 1813 / 1613 in the first prefill round; §4d has the round-3 lever and its
measured negative results.  The 32K+ fallback keeps
the raw Q5_K MMQ kernel (`--q5k-prepack off`, automatic above 32K context).

The 4-row verify round at 16K drops from **21.47 to 19.59 ms** (1.43 tokens/round
⇒ ~73.0 tok/s), and the same round at 4K/32K is **17.98 / 21.66 ms**.  On the 121-token
code prompt speculative decoding now measures **72.2 tok/s** (30 rounds, 1.43
tokens/round, 128/128 tokens identical to plain greedy and to the pre-plan engine).

Pure decode vs context (`--bench`, graph replays; `--bench-at ctx`):

| ctx | ms/token | tok/s | before this round |
|---|---|---|---|
| 4 096 | **16.39** | **61.0** | 18.02 / 55.5 |
| 32 768 | **22.62** | **44.2** | 23.65 / 42.3 |
| 131 072 | **40.95** | **24.4** | 41.16 / 24.3 |
| 262 144 | **64.82** | **15.4** | 64.21 / 15.6 |

A batch of 4 tokens in one weight pass (`--bench-rows`, host sync included):

| ctx \ rows | 4 | 8 |
|---|---|---|
| 4 096 | **17.98** (was 19.63) | **21.61** (was 25.86) |
| 16 384 | **19.59** (was 21.47) | **23.17** (was 29.20) |
| 32 768 | **21.66** (was 24.20) | **25.26** |
| 131 072 | **34.04** (was 38.88) | - |
| 262 144 | **50.42** (was 57.95) | - |

The 4-row round is now cheaper than 4 separate 1-row passes at every depth, and at
128K it is *below* the single-row path (34.1 vs 41.0 ms).  The verify attention now
uses the same row-split kernel as the prompt path (scores computed once per row/head
instead of once per dim-split warp) with the split count capped at 20 so the whole
grid is one wave; that is worth 0.7-7.5 ms/round depending on depth.  Plain M=1
decode deliberately keeps the original scalar attention, split combine and norm
kernels, so it stays bit-identical to the pre-plan engine (18.91 ms / 52.9 tok/s at
16K, unchanged).

Prefill no longer re-reads the cache once per row: each prompt chunk shares **one** KV pass
over the full layers (`--prefill-tc`, default on), and the default chunk is 512 rows.  The
remaining depth dependence is the attention work itself (the windowed layers are constant,
the full layers grow with depth).  On the real-code corpus:

| prompt tokens | 4K | 8K | 32K | 128K |
|---|---|---|---|---|
| prefill, 512-row chunk + MMQ (default) | **2047** | **1950** | **1613** | **954 tok/s** |
| prefill, per-row attention (`--prefill-tc 0`) | - | 877 | - | - |
| prefill, 8-row chunk (`--prefill-batch 8`, previous path) | 256* | - | - | - |

\* 121-token prompt.  The eight-row row is the previous generation of this engine: a dp4a
GEMM plus the shared KV pass.  The 512-row path adds the 128x128 int8 MMQ prompt GEMM, the
register-resident norms, and the two-block/SM row-split attention, and is **6-8x faster**
depending on depth.  The per-row attention path re-reads once per row, which is why it falls
away from the shared pass.  The shared pass rounds QK^T through fp16 (the same trade-off
llama.cpp's `-fa` makes); at 8K it keeps the argmax and 4 of the top 5 (top-50 48/50), and
`--prefill-tc 0` keeps the bit-exact reference path for oracle runs.

Speculative decoding keeps the exact verification guarantee (a draft token is emitted only
when the target model's own argmax agrees), so tokens/round is a pure speed property.  How
much it buys depends entirely on how repetitive the continuation is; long code contexts are
where it wins:

| context | prompt | tokens/round | spec tok/s | plain tok/s |
|---|---|---|---|---|
| 4 096 | 121-token code | 1.43 | **72.2** | 66.6 |
| 4 096 | same, `--adaptive-loops` | 1.43 | **98.9** | 93.3 |
| 8 192 | 8.2K-token repeated code (64-token sample) | 7.00 | **281** | ~53 |
| 32 768 | 32.7K-token CUDA corpus (64-token sample) | 2.10 | **95.1** | 44.2 |
| 32 768 | same, `--adaptive-loops` | 2.40 | **122.2** | - |
| 131 072 | (round cost measured; acceptance as above) | 3.23 | **>100** | 24.4 |

The 32K figures use a 64-token generation window on a 32,700-token CUDA corpus prompt
(`tmp/ids_codereal32k.txt`), so acceptance is a small-sample estimate.  The round cost is
the stable part: 21.9 ms at 32K vs 32.2 ms before, and 34.1 ms at 128K vs 46.1 ms.

The tensor-core verify kernel (§2) plus the shared-KV windowed pass removed most of the
attention tail: at 128K/4 rows a round now splits roughly into ~26 ms of weight streaming
and ~12 ms of attention (11.6 ms KV DRAM floor), so the round is bounded by
`weights + KV` at ~23 ms.

#### What changed in the decode pass (measured)

The decode improvements in this round, in the order they were applied (RTX 3060, 4K,
`--bench-at 4096`, 1-row / 4-row / 8-row):

| change | M=1 | M=4 | M=8 |
|---|---|---|---|
| baseline | 18.06 ms | 27.38 ms | 41.38 ms |
| `output.weight` BF16 -> aligned Q8_0 (logit GEMV, frees 300 MiB) | 17.25 | 25.43 | 36.08 |
| disable `k_combine_gate_norm` for small rows; 1024-thread register norms | 16.72 | 21.24 | 26.12 |
| aligned Q6_K `w.dp` decode kernel; fused QKVG / gate+up; TC verify GEMM | 16.70 | 21.39 | 26.12 |
| windowed verify layers move to the shared-KV tensor-core pass | 16.70 | 19.89 | 26.12 |
| register norms enabled (final default) | **16.39** | **19.63** | **25.86** |
| breakthrough-plan kernels: row-split verify attention (ns=20), gate vec4 | 16.39 | **17.98** | **21.61** |

The optional `--adaptive-loops` mode (skip loop 1 when converged) is measured on the
generation path instead: 93.3 tok/s plain and 98.9 tok/s with `--draft 7` on the
121-token code prompt, and 122 tok/s on the 32K CUDA corpus with `--draft 7`.

The Q5_K verify GEMM is the big one: `mma.m16n8k32.s8` with 16 weight rows on A and up to
8 verified rows on B, activations staged once in shared, weights streamed register-direct,
one barrier.  A verified row costs ~0.06 ms instead of ~2.9 ms, so a 4-row round went from
27.4 ms to 19.6 ms and an 8-row round from 41.4 ms to 25.9 ms.

Two measured negative results are worth keeping:

* the **warp-independent 256-dim attention kernel** (`k_attn_tc_wi`, `BRUCE_ATTN_WI=1`) is
  correct but slower than the existing dim-split CTA kernel at every depth: 4K/4 rows
  20.9 vs 19.6 ms, 32K/4 rows 30.4 vs 24.2 ms, 128K/4 rows 56.0 vs 38.9 ms.  With 2
  blocks/SM it has only 8 warps/SM, while the CTA kernel gets 4 blocks/SM (16 warps) and
  shares one K/V tile across the four rows; at long context occupancy beats the saved
  QK^T work.  It stays behind an env switch, default off.
* the **fused combine+gate norm** (`BRUCE_FUSED_COMBINE=1`) is slower than the existing
  16-block split-combine followed by the 512-thread gate norm for 1-8 rows (one block
  loses the 16-way head parallelism); it is used only for the prompt path, where it wins.

| `BRUCE_WA`/`BRUCE_R` override the batched-GEMM tile shape (dev sweep knobs).

**Decode numbers below are from the first breakthrough round; §4c has the current ones**
(single-token 4K/32K/128K/256K = 15.46 / 18.94 / 30.85 / 46.60 ms, prefill unchanged).

### Head-to-head with llama.cpp PR 30306 (`run_dory.bat`)

Same GGUF, same semantics (2 loops, `last_loop`, q8_0 KV, `-fa on`, 262144 ctx, `-ngl 999`,
`-ot` tensor overrides), same 262144 KV allocation, batch 1, greedy.  The reference is the
server from `run_dory.bat` driven through `/completion` with a token-id prompt of exactly the
depth and `n_predict 128` (its own prefill/generation timings); the engine is
`--ctx 262144 --bench 20 --bench-at DEPTH`.  Engine rows are pure CUDA-graph replays; the
server figure includes its per-token sampler/server loop (~3 ms/token, see §5).

| depth | engine ms/token | llama.cpp ms/token | speedup | engine tok/s | llama.cpp tok/s |
|---|---|---|---|---|---|
| 16 384 | 20.30 | 24.13 | 1.19x | **49.3** | 41.4 |
| 32 768 | 23.31 | 27.13 | 1.16x | **42.9** | 36.9 |
| 65 536 | 29.04 | 33.88 | 1.17x | **34.4** | 29.5 |
| 131 072 | 40.84 | 47.37 | 1.16x | **24.5** | 21.1 |

Fitting `ms/token = fixed + slope * depth` over those points: engine `17.4 + 0.179 us/key`,
llama.cpp `20.8 + 0.202 us/key`.  So the decode lead is ~13 % cheaper attention per key plus
~3.4 ms/token less fixed overhead; the weights pass (16 ms) is common to both.

With prompt-lookup speculation enabled (the reference server has no equivalent drafter), the
same engine measures 85.3 tok/s on the 32K real-code corpus and 70.0 tok/s at 128K depth
(round cost 34.1 ms at 4 rows, acceptance 3.23 tokens/round).

Prefill is now ahead of the reference server's rate at the same depth: this engine ingests
**1613 tok/s at 32K** and **954 tok/s at 128K** with a 512-token chunk and the 128x128 MMQ
tensor-core GEMM, versus the reference's 1501 tok/s at 32K and 934 tok/s at 128K with its
1024-token ubatch.  (The earlier 448 tok/s at 32K was the start of this round; the increase
is 3.6x and comes from the four levers below.)

#### The prompt GEMM: int8 tensor cores

Per token the engine applies 5.14 G MACs of q5_K/q6_K weights.  The old batched kernel ran at
1.4 T-MAC/s; a probe of this GPU's actual peaks (`tools/mma_probe.cu`, 28 SMs) said where the
ceiling is:

| path | measured peak | note |
|---|---|---|
| `dp4a` (CUDA cores, INT8) | 12.65 T-MAC/s | old kernel reached 11% of it |
| `mma.m16n8k32.s8` (INT8 tensor cores) | **50.2 T-MAC/s** | the path taken |
| `mma.m16n8k16.f16` (FP16 tensor cores) | 12.67 T-MAC/s | no advantage over dp4a |

`k_gemm_mmq_q5k` is the current prompt GEMM: a **128-weight-row x 128-activation-row**
tile, 8 warps in a 4x2 grid (warp tile 32x64, 16 `mma.m16n8k32.s8` per k32 step), 96 KB
shared, occupancy 1.  Each warp preloads its two weight A-fragments and their scale pairs
once per k32 and reuses them across all eight activation tiles, which drops shared-memory
traffic to ~1.9 LDS per MMA.  K is walked in 128-value double-buffered `cp.async` stages;
the 16-byte block header is staged per stage (each half needs its own scales) and `qh` once
per 256-value superblock, i.e. 192 B per superblock versus 176 B raw (+9%).  The int8 tile
is unpacked in place after each 128-value compute phase.  Weight and activation scales are
precombined per 32-value group as `{d*sc, -dmin*m}` and `{xd, xd*xsum}`, so the epilogue is
one I2FP, two FMA and one multiply per output instead of five operations.  Q/K/gate (plus v
on the layers where it is Q5_K) and FFN gate/up are launched as one horizontally fused
matrix each (a block resolves its part with three compares), and the grid is M-major so the
four activation-row tiles that share a weight tile are co-resident and hit L2.  Measured
13-14.6 T-MAC/s standalone on the real shapes; the smaller 64x64 `k_gemm_mma_q5k` remains
as the fallback for chunks below 96 rows.

The chunk is now 512 prompt rows (`MBUF` 512).  At 32K the weights stream 7.5 MB per token
instead of 59.8 MB at M=64, and 4-8x fewer GEMM launches are needed.

#### Where prefill time goes now (nsys, 32K prompt, approximate us per token)

| kernel | us/token | share | why |
|---|---|---|---|
| `k_gemm_mmq_q5k_144` | 251 | 51% | pre-unpacked 144 B stages, fp16 min-term MMA, 17.5 T-MAC/s |
| `k_attn_tc_rs_opt<6,2>` + `<4,3>` | 159 | 32% | 6-warp full / 4-warp SWA, 3 blocks/SM for SWA, 2 for full |
| `k_gemm_mma_q6k_128` | 39 | 8% | 128x128 tile, native m16n8k16 int8 |
| `k_silu_mul_norm_vec4` | 11 | 2.2% | float4 quarter-warp, 256 threads |
| `k_resid_vec4` | 10.7 | 2.2% | float4 quarter-warp, 160 threads |
| `k_qkv_prep_fast` | 6.9 | 1.4% | one warp per head, zero barriers |
| `k_gate_norm_vec4` + fused combine | 8.2 | 1.7% | register-resident |
| `k_fill_xdh` + embed / q8 / rope / copy | 3 | 0.6% | launch bound |
| **total** | **489** | | **2036 tok/s** |

At 8K the same build runs 2505 tok/s (399 us/token); attention falls to 45 us/token
because the full layers see half the depth.  The Q5_K GEMM is now at the measured
standalone rate for this clock (17.5 T-MAC/s at 1777 MHz, vs 18.6 at 1890 MHz), and
the 128x128 Q6_K retile is only marginally faster than the old 64x64 kernel (39.2 vs
40.8 us/token): its 96 KB stage costs the second block per SM, so the larger tile
mostly breaks even.  The two levers that moved the number are the pre-unpack
(340 -> 251 us) and the row-split attention (187 -> 159 us at 32K).

#### q6_K on tensor cores (lossless repack)

q6_K keeps a scale per **16** values while `mma.m16n8k32` covers 32, so its packed form cannot
feed the k=32 dot.  Two things block the obvious approach, and both are solved at load time:

* **Granularity.**  The 32-value k block spans two scale groups, so the kernel issues one
  native `mma.m16n8k16.s8` per 16-value group (`a[0],a[1]` carry k 0..15, `a[2],a[3]`
  carry k 16..31).  The k=16 form runs in half the cycles of k=32 on SM86 and needs no
  zeroed A fragments or `CS2R` zero moves; each int32 dot covers exactly one scale group,
  so it stays exact.
* **Alignment.**  A q6_K superblock is 210 bytes, so every row starts at an odd byte offset and
  `cp.async` (16-byte granularity, alignment enforced) cannot stage those bytes at all.  An
  in-kernel version of this kernel was written and fails exactly there.

So at load each q6_K tensor is re-encoded, losslessly, into 160 B per (row, 128 values):
128 B of int8 (`value = stored 6-bit code - 32`, exact) plus 32 B of fp32 `d*sc` products
(exact).  The values are the stored codes, not a re-quantisation.  Cost: +0.3 GB of VRAM,
+0.6 s of load time, and the packed form is kept for the single-row/decode path (which wants
the smaller representation).  Measured against the dp4a q6_K path: max |diff| 7e-8.

One trap worth recording: **q6_K super-block scales are routinely denormal fp16** (|d| ~ 1e-5,
exponent field 0).  A hand-written fp16->fp32 helper that only handles normal halves returns 0
for them and silently zeroes every FFN-down output; the host converter decodes exponent 0 as
`m * 2^-24`.

#### What limits each kernel now

The old dim-split prompt attention recomputed QK^T four times (once per warp) and, at 16
rows, could not keep two blocks resident; that is why attention used to dominate at depth.
The row-split kernel below computes scores once, holds Q in registers and fits **2 blocks/SM**,
taking attention from 0.644 ms/token at 32K (49%) to **0.187 ms (30%)**.

The MMQ GEMM is now issue-bound on the scale epilogue rather than on memory or tensor cores:
the int8 HMMA work is a small fraction of the kernel, while 64 I2FP conversions + 128 FMA +
64 multiplies per k32 step dominate.  The 3-op epilogue (precombined `{d*sc, -dmin*m}` and
`{xd, xd*xsum}`) is the minimum that keeps the per-32-group scales exact.  The 50.2 T-MAC/s
peak needs the scale application removed entirely (the 5-mode ablation's pure-LDS+IMMA mode
measured 40.8 T-MAC/s), which the quantisation does not allow.

Two earlier ideas remain excluded: 16-row attention blocks at 1 block/SM (losing the second
block per SM costs more than the KV traffic saved) and factoring the epilogue as a per-tile
precompute (the min term carries an `xd` factor that cannot be shared).

#### Row-split prompt attention (llama.cpp's fattn layout, adapted)

`k_attn_tc` splits the 256 head dims across four warps so 64 output floats/lane fit the
register file; every warp then needs the full QK^T, so **the score matrix is computed four
times** and only 8 rows share each KV sweep.  `k_attn_tc_rs_opt` instead gives each warp
**four rows x four q-heads = all 16 mma m-slots** and walks **all 256 dims**, so scores are
computed once and a block covers 16 rows per KV pass:

| | `k_attn_tc` (8 rows) | `k_attn_tc_rs_opt` (16 rows) |
|---|---|---|
| attention mma per prompt row | 40 | 16 |
| KV passes per 512-row chunk | 64 | 32 |
| accumulator floats per lane | 64 | 64 (half2) |
| Q storage | shared | 64 regs/lane |
| shared per block | 33.0 KB | 16.9 KB |
| blocks/SM | 2 | **2** |

The optimised kernel keeps Q in registers and accumulates PV in half2, which halves the
shared-memory footprint and the online-softmax rescale work while unlocking the second
block per SM.  It therefore wins at every depth, so `rows > 8` always dispatches to it
(the old dim-split kernel remains only for the 2-8 row verify path).

A 32-row block (8 warps x 4 rows) would halve the KV traffic again, but it needs 256
threads and >128 registers/lane just for Q+O, so it cannot keep the second block per SM;
the measured 1-block/SM variant was slower at both 8K and 32K.

#### Result vs the 1500 tok/s target

The target is met with margin at the depths that matter:

| prompt | before this round | now | target |
|---|---|---|---|
| 4K | ~1132 | **2047 tok/s** (488 us/tok) | 1500+ |
| 8K | ~1057 | **1950 tok/s** (513 us/tok) | 1500+ |
| 32K | 832 | **1613 tok/s** (620 us/tok) | 1500+ |
| 128K | - | **954 tok/s** | - |

The remaining 32K time is dominated by the 128x128 MMQ (55%) and the row-split attention
(30%).  Both are now at the measured limits of their layouts on this card: the MMQ is
issue-bound on the int->float scale epilogue (the int8 dot itself would run at
40+ T-MAC/s), and attention is bound by the fp16 MMA rate plus the split-K tail.  Further
gains would need stream-K/persistent GEMM scheduling to remove the 5-10% wave tails, or
load-time pre-unpacking of Q5_K into int8 to eliminate the in-kernel unpack.

Reaching 0.667 ms/token (1500 tok/s) needs both of those *and* a further ~1.4x, which on this
GPU means cutting the attention's fp16 mma work again (packing 4 rows x 4 heads into every
mma slot and keeping the output at 16 mma/warp) and getting the GEMM to its 20 T-MAC/s
stage-free core rate with the staging fully overlapped.

---

## 4b. Breakthrough plan: what shipped (2026-10-27)

`BREAKTHROUGH_PLAN_16K.md` levers P1, P2, P3, P4 and D1's verify path are implemented
and measured.  The changes are independent and each has a fallback:

| lever | implementation | measured |
|---|---|---|
| P1 Q5_K pre-unpack | `k_q5k_prepack144` (load-time, GPU) + `k_gemm_mmq_q5k_144`; 144 B stage, `dm` rank-16 fp16 MMA | Q5 GEMM **340 → 251 us/token** at 16K |
| P2 prefill attention | `PRMT/HSUB2/HMUL2` KV dequant, `<4,3>` SWA + `<6,2>` full, `full` fast path, reversed row chunks | **121 → 97 us/token** at 16K |
| P3 Q6_K retile | `k_gemm_mma_q6k_128`, 128x128 tile on the 160 B repack | **40.8 → 39.2 us/token** (marginal) |
| P4 elementwise | `k_silu_mul_norm_vec4`, `k_resid_vec4`, `k_gate_norm_vec4`, `k_qkv_prep_fast` | silu **21.0 → 11.2**, resid **14.5 → 11.0**, qkv **11.4 → 7.0 us/token** |
| D1 verify attention | row-split verify kernel + `ns = 20` one-wave split | 16K 4-row round **21.47 → 19.59 ms**; 32K **24.20 → 21.66**; 256K **57.95 → 50.42** |

Prefill at 16K is **2341 tok/s** (was 1813), and the 16K 4-row round is **19.59 ms**,
i.e. ~73.0 tok/s at the 1.43 tokens/round measured on the code prompt.  Speculative
generation on the 121-token code prompt is **72.2 tok/s** (was 70.9) and on the 32K
CUDA corpus **95.1 tok/s** (was 81.6), both with 128/128 token identity on the code
prompt.

The 2500 tok/s prefill target is **not fully met**: 2341 is 94% of it.  The remaining
gap is Q6_K (39 vs the projected 28 us/token; the 128x128 tile loses the second block
per SM) and the clock-normalised Q5/attention numbers (the probe's 18.6 T-MAC/s is
17.5 at 1777 MHz).  Closing it needs a Q6_K kernel that keeps two blocks resident, or
a stream-K/persistent schedule to remove the MMQ wave tails.

Quality gates actually run (not projected):

* 16K prompt: same argmax, top-5 5/5, top-50 49/50, max |Δlogit| 0.56 (at logits ~1,
  not at the top; the model's own near-tie noise is 0.06 nats).
* 121-token code prompt: plain greedy 128/128 identical to the pre-plan engine, and
  speculative greedy 128/128 identical as well.
* Q6_K retile vs the old kernel: max |diff| 1e-6 over the real shapes.
* `PRMT` attention dequant is bit-identical to the old `I2F` path (0 ULP).
* Plain M=1 decode is bit-identical by construction: it keeps `k_attn`, the split
  combine, `k_qkv_prep`, `k_silu_mul_norm_rows_opt<7,32>` and
  `k_resid_rows_opt<3,32>`.  The float4 norms and fast QKV prep are used only for
  prompt chunks and verify rows 2..8.

Runtime switches (all default-on unless noted):

| switch | meaning |
|---|---|
| `--q5k-prepack auto\|on\|off` | 144 B Q5_K pre-unpack; `auto` enables at ctx ≤ 32K (~3.3 GB) |
| `BRUCE_RS_VERIFY=0` | verify rows 2..8 use the old dim-split `k_attn_tc` |
| `BRUCE_VERIFY_SPLITS=N` | verify row-split count (default 20) |
| `BRUCE_Q6_128=0` | disable the 128x128 Q6_K retile |
| `BRUCE_FAST_NORMS=0` / `BRUCE_FAST_QKV=0` | A/B the float4 norms / fast QKV prep on the batch path |
| `BRUCE_MMQ=0`, `BRUCE_MMA=0`, `BRUCE_TC1=1` | existing fallback switches |

Measured negative results from this round: the `0x4B400000` biased-MMA prefix is
bit-identical but **slower** here (7193 vs 6998 ms prefill) because the extra FADD chain
outweighs the removed I2FP; the 128x128 Q6_K tile only breaks even because 96 KB of
shared memory costs the second block/SM; and the 6-warp full-attention shape is worth
only ~2 us/token over the 4-warp shape.  Tree verification / Jacobi lookahead slots
(decode plan §5.1-5.3) are **not** implemented: acceptance remains drafter-limited at
1.43 tokens/round on prose/code, which is already enough to hit 72 tok/s at the new
19.59 ms round.

## 4c. Decode pass 2026-10-27 (round 2): one-wave tensor-core decode attention

The first breakthrough round left a very uneven budget: the weight streams were already at
92-100 % of the card's measured 345 GB/s, but **single-token attention ran at 99 GB/s at 4K and
~160 GB/s at 128K**, and the split-K merge was a serial L2 chain.  This round attacks exactly
those two, and leaves prefill untouched (measured, not assumed).

### What changed

| change | mechanism | code |
|---|---|---|
| **M=1 decode attention → row-split tensor-core kernel** | Single-row decode now uses the same `k_attn_tc_rs_opt<4,3>` row-split kernel as verification and prefill attention: 16-key K/V tiles are dequantised once per CTA into shared fp16, scores are computed with `mma.m16n8k16.f16` and the score matrix is not recomputed per dim-split warp. One KV pass per layer instead of a scalar per-key dependent load chain. | `attn_m1()` (`BRUCE_ATTN_M1PATH`, default 2) |
| **one-wave verify split count** | `BRUCE_VERIFY_SPLITS` 20 → **21**: with 4 kv heads and 3 blocks/SM the grid is `4 × 21 = 84` blocks = exactly one full wave on 28 SMs (20 leaves 4 SMs idle, 22+ starts a partial second wave). Measured 17.50 → 16.77 ms of attention at 128K. | `verify_splits()` |
| **parallel split-K merge** | `k_attn_combine_fast`: one block per query head, one thread per output dim, per-split softmax weights computed once and broadcast through shared. The old form walked the split list serially in a single warp. | `attn_comb_fast()` (`BRUCE_COMB_FAST`, default 1) |
| **4-key pipelined scalar attention** (kept as a switch) | `k_attn_u4` keeps four K/V loads, four QK dots and four V accumulations in flight. 2.21 → 1.91 ms at 4K, but the row-split kernel is faster still (1.48 ms), so it is opt-in. | `BRUCE_ATTN_U4=1` |
| **lazy-V for single-row decode** | After the scores are known, a 16-key tile whose best score is >14 nats below the running max contributes `exp(-14) < 8e-7` to a numerator accumulated in fp16, so its V is never fetched, dequantised or multiplied. Block-wide vote (`__syncthreads_or`) because V is staged cooperatively. Single-row only: at 2..8 rows the union over 16 rows makes almost every tile hot and the extra vote barrier costs more than it saves (measured 17.898 → 17.924 ms at M=4). | `BRUCE_LAZYV=0` disables |
| **phase probe** | `--probe [--bench N] [--bench-at POS] [--bench-rows M]` times one pass with each phase disabled and then each phase alone, so the decode budget is attributed per component without a profiler. `--bench` is the rep count. | `probe_mode()` |

### Measured (RTX 3060, 1875-1897 MHz, graph replay, `--bench-at ctx`)

| ctx | before ms/tok | **now ms/tok** | before tok/s | **now tok/s** | delta |
|---:|---:|---:|---:|---:|---:|
| 4 096 | 16.374 | **15.411** | 61.07 | **64.89** | −5.9 % |
| 32 768 | 22.352 | **18.842** | 44.74 | **53.07** | −15.7 % |
| 131 072 | 40.568 | **30.689** | 24.65 | **32.58** | −24.3 % |
| 262 144 | 65.154 | **45.689** | 15.35 | **21.89** | −29.9 % |

Verify rounds (`--bench-rows M`, host sync + argmax included):

| ctx \ rows | 2 | 4 | 6 | 8 |
|---|---:|---:|---:|---:|
| 4 096 | **17.08** | **17.90** | **18.61** | **21.54** |
| 32 768 | – | **21.40** | – | **24.97** |
| 131 072 | – | **33.38** | – | – |
| 262 144 | – | **48.76** | – | – |

The 2..8-row rounds are unchanged (they already used the tensor-core kernels): the per-row
weight stream is still 12-40 % slower than the single-row path because the barrier-free verify
GEMM issues 4-byte strided weight loads.  A 4-row round at 4K is 17.90 ms, i.e. **79.9 tok/s at
the 1.43 tokens/round measured on the code prompt**.

End-to-end generation on the same prompts as §4:

| workload | before | **now** |
|---|---:|---:|
| 121-token code, plain greedy | 66.6 / 68.2 tok/s | **68.5 tok/s** |
| 121-token code, `--draft 7` | 72.2 tok/s | **73.9 tok/s** |
| 32.7K CUDA corpus, plain (64-token window) | 44.2 tok/s | **53.5 tok/s** |
| 32.7K CUDA corpus, `--draft 7` | 95.1 tok/s | **97.4 tok/s** |

Prefill is bit-for-bit the same path and measures **2633 / 2531 / 2341 / 2038 tok/s at
4K / 8K / 16K / 32K** (pre-change 2637 / 2505 / 2341 / 2036 — within run noise).

### Where the time goes now (`--probe`, 4K, M=1, eager launches, 1.9 µs/launch included)

| phase | ms/pass | note |
|---|---:|---|
| ffn gate+up (Q5_K) | 4.17 | 339 GB/s = 98 % of ceiling |
| qkv fused (Q5_K + q6_K v) | 3.35 | 313 GB/s |
| ffn down (q5_K/q6_K mix) | 3.12 | 319 GB/s of the repacked form |
| wo (Q5_K) | 1.38 | 293 GB/s |
| logits (Q8_0) | 0.97 | 351 GB/s |
| attention (56 layers) | **1.47** | was 2.21; full layers run at 332 GB/s, SWA layers are launch-bound (lazy-V trims 1.489 → 1.462) |
| combine + gate norm | 0.72 | was 0.95 |
| resid / silu / qkv prep | 0.68 / 0.55 / 0.27 | single-block kernels, latency-bound |
| block norms + embed | 0.05 | |

The weight streams are **4.20 GB per token** (3.13 GB q5_K + 0.73 GB q6_K in the 160 B repacked
form + 0.34 GB Q8_0 logits; the 0.48 GB in §5 is the raw `w.d` accounting) and 12.0 ms of the
15.5 ms: **the 4K single-token path is now within ~14 % of its 13.6 ms weight+logits+KV floor**.
What remains is ~2 ms of single-block elementwise latency that would need cross-kernel fusion to
remove, and the tensor-core verify gap on the 2..8-row path.

### Excluded / measured negative results this round

| idea | measured result | why rejected |
|---|---|---|
| multi-row dp4a Q5_K GEMV for verify rows (16-byte weight loads, M rows per warp) | qkv 4.30 vs 3.88 ms at M=2, **8.53 vs 4.11** at M=4, 13.82 vs 4.55 at M=8 | the M× shared-activation reads and dp4a issue dominate once M > 2; the tensor-core form is the right one |
| 3 blocks/SM for the verify GEMM (`__launch_bounds__(256,3)`, 85 regs) | qkv 4.09 → 5.18, gate+up 4.68 → 6.29 ms at M=4 | more resident CTAs thrash L2 on the weight stream; 0 spills, so it is not a register issue |
| Q6_K decode row-unroll R = 4 / 8 with exact-division grids | M=4 ffn-down 4.31 / 4.29 vs 4.23 at R=2; M=8 8.76 vs 7.42 | at M ≥ 4 the kernel is shared-memory bound (8× activation reads per weight load), so more MLP per warp does not help |
| fewer SWA attention splits (4/6/8/12 vs 21) | attention 1.99 / 1.64 / 1.72 / 1.54 vs **1.48** ms | short kernels need the block count, not more keys per block |
| 256-thread split merge | 0.79 → 0.71 ms | only 0.08 ms; the chain is launch-bound, the new kernel still keeps the win |
| scalar 4-key pipeline as the default | 1.91 ms vs 1.48 for the row-split kernel at 4K | superseded |

### Quality gates (all run on this build)

* 121-token code prompt, plain greedy **128/128 identical** to the pre-change engine
  (`tmp/base_code128_ids.txt` vs `tmp/h_plain_ids.txt` on the shipping build).
* same prompt, `--draft 7`: **128/128 identical to plain greedy** (`tmp/h_spec_ids.txt`).
* 16K prompt logits vs the pre-change engine: same argmax, top-5 5/5, top-50 49/50,
  max |Δ| 0.563 — exactly the delta class documented in §3 (the engine's own oracle noise).
* prefill 4K/8K/16K/32K within noise of the pre-change build (2633/2531/2341/2038).

The single-row attention now shares the verify path's numerics (fp16 `QK^T`, fp32 accumulate —
the same convention as llama.cpp `-fa on`).  `BRUCE_ATTN_M1PATH=0` restores the previous scalar
bit-identical path for oracle runs; it costs 0.9 ms at 4K and 9.4 ms at 128K.

### Switches added

| switch | meaning |
|---|---|
| `BRUCE_ATTN_M1PATH=2\|1\|0` | single-row attention: row-split tensor core (default) / dim-split tensor core / original scalar |
| `BRUCE_VERIFY_SPLITS=N` | split-K count for 1..8 rows (default 21 = one wave) |
| `BRUCE_SWA_NS=N` | separate split count for the 42 sliding-window layers (default = verify splits) |
| `BRUCE_COMB_FAST=0` | use the old single-warp split merge |
| `BRUCE_ATTN_U4=1` | scalar 4-key pipelined attention instead of the row-split kernel |
| `BRUCE_LAZYV=0` | disable the lazy-V tile skip in single-row decode attention |
| `BRUCE_Q6R` / `BRUCE_Q6B` | Q6_K decode row-unroll / grid override (dev sweep; defaults R=2, balanced grid) |
| `BRUCE_Q8_GPU=0` | use the scalar host BF16→Q8_0 logits path instead of `k_bf16_to_q8` |
| `BRUCE_LOAD_PROF=1` | print the weight-load phase breakdown (open/tokenizer/pread/slot-wait/drain) |
| `--probe` | phase-attribution mode, see above |

## 4d. Prefill pass 2026-10-27 (round 3): what the staging stream costs

Round 2 left prefill at **2617 / 2518 / 2345 / 2038 tok/s at 4K / 8K / 16K / 32K** and 954 at
128K.  This round attacked the prompt GEMM (60 % of a prefill pass) and the full-layer
attention split count, with **no change to decode or verification** (measured, see the gates
below).  New numbers, same 512-row chunks, same machine state:

| prompt | 4K | 8K | 16K | 32K | 128K |
|---|---|---|---|---|---|
| before | 2617 | 2518 | 2345 | 2038 | 954* |
| **now** | **2802** | **2695** | **2502** | **2186** | **1101** |
| µs/token | 357 | 371 | 400 | 457 | 908 |

"before" is the pre-change build measured today on the same machine state (§4c quotes
2633 / 2531 / 2341 / 2038 from its own run); \* the 128K baseline is §4's 954 tok/s, the
only depth not re-measured on the old build.  That is +7 % at every depth that matters and
+15 % at 128K.

### What was actually limiting the prompt GEMM

A standalone harness with the shipping 144 B Q5_K kernel and its real shapes
(`tmp/mmq_bench.cu`, K=7168 N=2560 M=512, 1777 MHz) produced this ablation:

| variant | T-MAC/s | read as |
|---|---:|---|
| shipping (row-major weight stage) | 17.1 | baseline |
| **no per-32 scale epilogue** (int accumulate only) | 21.4 | epilogue costs 1.25x |
| **no `cp.async` staging** (re-use stage 0, compute intact) | 27.1 | staging costs 1.58x |
| neither | 43.7 | MMA + LDS ceiling |
| bias-trick epilogue (`0x4B400000`) | 17.2 | still a wash |

Then the decisive one: staging the *same* stage-0 bytes over and over (no new DRAM or L2
traffic) ran at the same speed as the real stream, and a staging-only kernel took 2.1-2.7 µs
per CTA-stage against 2.0 µs of compute.  So the cost is the **`cp.async` copy stream itself
(38 KB per stage at ~18 GB/s per SM), not bandwidth, not latency, not the epilogue**.
Coalescing that stream is what paid:

* **Stage-major Q5_K pre-pack.**  `w.dp` was `[row][stage][144]`, so one CTA stage was 128
  separate 144 B rows (4 segments per warp copy).  The load-time pre-pack now writes
  `[stage][row][144]`, so a stage is a contiguous 18,432 B run and a warp copies 512
  contiguous bytes.  Measured 17.1 → 18.9 T-MAC/s in the harness, 44.7 → 40.9 ms per
  512-row pass in-engine.  The prepacked bytes, the values and the accumulation order are
  unchanged — this only moves bytes — so the result is bit-identical (verified: 121-token
  prompt plain greedy 128/128 and `--draft 7` 128/128 against the pre-change engine).

### What was *not* the answer (all measured, kept out)

| idea | measured | why |
|---|---|---|
| activations read straight from L1 into B-fragment registers (no smem) | 12.98 T-MAC/s vs 17.06 | 8 rows per 4-byte lane load = 8 sectors per instruction; the smem bounce is cheaper |
| 3-deep `cp.async` pipeline, K=64 stages (fits smem) | 17.11 T-MAC/s | staging is throughput-bound, so a deeper pipe only adds stages |
| 2 CTAs/SM (128x64 tile, `__launch_bounds__(256,2)`) | 2.7-11 T-MAC/s | 128-register budget forces 1 KB of spills into the inner loop |
| 16 warps, warp tile 16x64 or 32x32 | 15.5-16.0 T-MAC/s | smaller warp tiles raise LDS per MMA faster than the extra warps help |
| attention KV staged with `cp.async` one tile ahead | +0.2-0.5 % | the second resident block already covers the DRAM latency |
| full attention at 48 rows/block (12 warps, 1 block/SM, half the KV traffic) | 2468 vs 2484 tok/s at 16K | confirms attention is MMA/issue bound, not KV-bandwidth bound |

### The one attention win: fill the last wave

Full layers ran `4 * ceil(rows/24)` blocks, i.e. ~6.3 waves on 28 SMs, so the final wave was
a 20-30 % tail.  Splitting each row-chunk's key range **4 ways instead of 2** (disjoint key
ranges, so exactly the same KV bytes) fills the machine: attention 82.7 → 77.6 ms per 512-row
pass, +1.8 % end-to-end at 32K, +0.5 % at 16K.  The floor is only applied to prefill
(`rows > 8`) full layers; decode/verify keep their tuned split counts.

Because it changes the fp32 reassociation of the split merge, prompt logits move in the
documented near-tie class: 32K prompt, same argmax, top-5 5/5, top-50 48/50, max |Δlogit|
0.86 (the engine's own `--prefill-tc 0` vs shared-KV delta is 0.50, and changing the decode
split count alone already flips ties).  121-token code prompt: plain greedy **128/128** and
`--draft 7` **128/128** identical to the pre-change engine.
`BRUCE_PREFILL_SPLITS_MIN=0` restores the old rule exactly.

### Where a 512-row prefill pass goes now (`--probe --bench-rows 512`, depth 16K)

| phase | before | now | share |
|---|---:|---:|---:|
| qkv gemm (Q5_K fused) | 44.7 | **40.9 ms** | 17 % |
| ffn gate+up (Q5_K fused) | 56.3 | **52.1 ms** | 22 % |
| ffn down (Q5_K + Q6_K) | 34.8 | **32.0 ms** | 14 % |
| wo (Q5_K) | 16.5 | **16.4 ms** | 7 % |
| attention | 82.7 | **77.6 ms** | 33 % |
| silu / resid / combine+gate / prep / norms / embed / logits | 19.5 | **17.4 ms** | 7 % |
| **total** | **254.5 ms** | **236.5 ms** | 462 µs/token |

`--probe --bench-rows N` now accepts `N` up to `MBUF` for the batched path (it used to clamp
to 8), so the same per-phase attribution works for prefill chunks.

The remaining GEMM gap is structural on this card: 38 KB of staged operands per 2.1 M MAC at
the measured ~18 GB/s/SM `cp.async` rate, with the accumulator tile capped at 128 weight x
128 activation rows by 96 KB of shared memory and 64 accumulator registers per lane.  The
epilogue (one I2FP + one FMUL + one FMA per output per 32-value group) is already the
minimum that keeps Q5_K's per-32 scales exact, and the `0x4B400000` bias trick that removes
the I2FP measured neutral twice.

### Switches added this round

| switch | meaning |
|---|---|
| `BRUCE_PREFILL_SPLITS_MIN=N` | floor on full-layer prefill splits (default 4; `0` = old rule) |
| `BRUCE_ATTN_PIPE=0` | prefill attention stages KV with direct `LDG` instead of the cp.async raw tile |
| `BRUCE_ATTN_FULL_NW=N` | full-layer prefill attention warps (4/6/12, default 6) |

## 4e. Decode pass 2026-10-27 (round 3): warp-owned decode attention

Round 2 put single-row decode on the same row-split tensor-core attention as verification,
which left a structural waste: the 16-row `mma` tile is indexed by `(row, head)` pairs, so at
`rows == 1` only 4 of the 16 tile rows are live and **all four warps compute the same QK^T over
the same 16-key tile** while re-reading the whole staged K/V tile from shared memory.  Measured
at 256K that kernel streams the KV cache at 258 GB/s of the card's 381 GB/s achievable read
bandwidth (a 272 B-record read of the same volume in isolation reaches 381 GB/s, so the kernel,
not the layout, was the limit).  It is also latency-bound rather than bandwidth-bound: the
lazy-V tile skip only moves 0.2 ms because a 16-key tile is almost never 14 nats cold.

### What changed

| change | mechanism | code |
|---|---|---|
| **warp-owned single-row attention** | One warp per `(kv head, split)` with its own 16-key tile: 4/8 128-bit KV loads issued before the dequant chain, vectorised `PRMT/HSUB2/HMUL2` dequant into a private 8.4 KB smem tile, only `__syncwarp()` (no block barriers), one `mma.m16n8k16.f16` score pass and one PV pass per tile.  Per decoded key this is 4x less tensor-core work, 4x less LDS traffic and 4x fewer barriers than the row-split kernel.  32-thread blocks with a 100 % shared-memory carveout hint; 56 splits for full layers, 28 for SWA (`4 x 56 = 224` independent KV streams = exactly one wave at 8 blocks/SM). | `k_attn_m1_w` (`BRUCE_ATTN_M1PATH=3`, default) |
| **2..4-row verify attention on the same structure** | For 4 verified rows the 16-row tile is exactly `4 rows x 4 kv heads`, i.e. fully live, so one warp carries all of them.  Same warp-owned tile, same dequant.  Measured 33.2 -> 28.5 ms per 4-row round at 128K and 48.9 -> 39.3 ms at 256K (a 4-row round is now *below* four single-row passes at every depth). | `k_attn_mv_w<2,3,4>` |
| **float4 quarter-warp norms on the M=1 path** | The single-row residual/SiLU/gate norms now use the 160/256-thread register-resident float4 kernels (8 lanes per 32-value q8 group) instead of the 1024-thread scalar forms: 0.45 ms/token at every depth. | `k_resid_vec4`, `k_silu_mul_norm_vec4`, `k_gate_norm_vec4` in `forward_launch` |
| **cp.async prefetch on the 5..8-row fallback** | The row-split kernel's PIPE template (raw q8 K+V tile copied one tile ahead) is now the default for the 5..8-row path: 52.4 -> 50.7 ms per 8-row round at 256K. | `k_attn_tc_rs_opt<4,3,false,true>` |
| **lazy-V threshold made a symbol** | `g_lazy_thr` (nats) instead of a literal, so the skip threshold can be swept without a rebuild. | `--` / `BRUCE_LAZY_THR` |

### Measured (RTX 3060, 1875-1897 MHz, 3 reps, graph replay, `--bench 20 --bench-at ctx`)

| ctx | before ms/tok | **now ms/tok** | before tok/s | **now tok/s** | delta |
|---:|---:|---:|---:|---:|---:|
| 4 096 | 15.427 | **14.61** | 64.8 | **68.4** | −5.3 % |
| 32 768 | 18.819 | **16.91** | 53.1 | **59.2** | −10.1 % |
| 131 072 | 30.506 | **24.84** | 32.8 | **40.3** | −18.6 % |
| 262 144 | 45.885 | **35.35** | 21.8 | **28.3** | −23.0 % |

Verify rounds (`--bench-rows M`, host sync + argmax included):

| ctx \ rows | 2 | 4 | 8 |
|---|---|---|---|
| 4 096 | 17.17 → **17.13** | 17.96 → **18.04** | 21.60 → **21.26** |
| 32 768 | 20.56 → **19.55** | 21.46 → **20.47** | 25.01 → **24.71** |
| 131 072 | 32.42 → **28.60** | 33.22 → **28.52** | 36.95 → **35.96** |
| 262 144 | 48.00 → **40.39** | 48.90 → **39.32** | 52.46 → **50.70** |

End-to-end generation (same prompts as §4):

| workload | before | **now** |
|---|---:|---:|
| 121-token code, plain greedy | 68.5 tok/s | **71.2 tok/s** |
| 121-token code, `--draft 7` | 73.9 tok/s | **75.8 tok/s** |
| 32.7K CUDA corpus, plain (64-token window) | 53.5 tok/s | **59.5 tok/s** |
| 32.7K CUDA corpus, `--draft 7` | 97.4 tok/s | **101.4 tok/s** |

The last row is the first time this engine crosses 100 tok/s on a real 32K prompt without
`--adaptive-loops`.  A 4-row round at 128K costs 28.6 ms, i.e. **113 tok/s at the 3.23
tokens/round measured on long code**, and 53 tok/s at 256K at 2.10 tokens/round.

Prefill is untouched by construction (rows > 8 never enters the new kernels) and measured
**2092 tok/s** on the 121-token prompt and **2179 tok/s** at 32K — the same as the round-3
prefill build (2186), i.e. inside run noise.

### Where the step is now

At 4K the phase budget is q5k/q6k/q8 weight streams **13.0 ms** (322 GB/s of the 381 GB/s
achievable read bandwidth, measured with a trivial 4 GB read kernel on this card, i.e. above
its 360 GB/s spec sheet; 4.20 GB/token), attention **0.7 ms**, resident norms/residual/prep
**1.2 ms** and
~0.4 ms of graph-node overhead: 14.6 ms total against a ~14.0 ms practical floor.  At 256K the
same weights (13.0 ms) plus the 8.0 GB KV stream at 344 GB/s (23.3 ms) gives 35.3 ms against a
34.0 ms DRAM floor — **within 4 %**.  The long-context KV traffic is now irreducible without
quantising the cache (q4 KV would halve it) or skipping tiles, and both lazy-V and loop-0 ->
loop-1 tile forwarding are dead ends on this model: **no 16-key tile is ever 4 nats below the
running maximum**, so every tile is hot (threshold sweep 4/6/8/10/14 nats is time-identical at
128K and 256K).

Two micro-architectural notes from this round: the warp kernels want *registers, not
occupancy* — with `__launch_bounds__(32, 10)` ptxas settled at 168 registers with 40-112 B of
spill and 11 warps/SM; dropping the min-blocks hint lets it use 250 registers, spill nothing and
run with 8 warps/SM, which is worth 0.6 ms/token at 256K and 2.7 ms per 4-row round.  And the
split count should equal the wave: `4 x 56 = 224` warps = exactly one wave at 8 blocks/SM
(42 splits leaves 25 % of the machine idle, 64 starts a partial second wave and costs 20 %).

### Excluded / measured negative results this round

| idea | measured result | why rejected |
|---|---|---|
| tensor-core q6_K verify GEMM (`mma.m16n8k16.s8`, 8- and 16-row A tiles, activations read direct from global, 160 B repack) | 208-242 GB/s at M=4 and 138-144 GB/s at M=8 vs **314 GB/s** for the dp4a kernel | the A-fragment fetch is 4 B per lane over 8 different rows (8 sectors per instruction, 128 B used): the MMA saves math instructions but the load path dominates.  A coalesced 16 B staging pass would need +2.5 KB smem per warp and a barrier per 128-value stage; not worth it against a 314 GB/s kernel |
| M=1 split counts away from the wave (42, 64, 84 at 256K) | 35.91 / 43.00 / 39.18 ms vs **35.49** at 56 | 42 leaves 25 % of the machine idle, 64 starts a partial second wave; the split merge also grows with `ns` |
| lazy-V with a tighter threshold (4/6/8/10 nats) | identical time at 128K and 256K | the model's score distribution is broad; the running max within a split is never far above a whole tile |
| `--adaptive-loops`, tree verification, Jacobi lookahead | not implemented this round | unchanged from §4d; `--adaptive-loops` remains the one (lossy) way past the 87 tok/s single-token floor |

### Quality gates (all run on this build)

* 121-token code prompt, plain greedy **128/128 identical** to the previous shipping build
  (`tmp/fin2_ids.txt` vs `tmp/w_new_ids.txt`), and the same with `--draft 7`.
* Same prompt with the *old* kernels forced (`BRUCE_ATTN_M1PATH=2`, `BRUCE_FAST_NORMS_M1=0`):
  128/128 identical, so the new numerics are inside the engine's own noise class.
* Prose prompt, 48 greedy tokens: character-identical to the old path.
* Short context (3-token prompt): identical to the old path; empty-split partials (a 42-way
  split of a 2-key range) merge to zero without NaN.
* Prefill 4K/32K within noise of the pre-round build.

### Switches added

| switch | meaning |
|---|---|
| `BRUCE_ATTN_M1PATH=3\|2\|1\|0` | single-row attention: warp-owned (default) / row-split TC / dim-split TC / scalar |
| `BRUCE_M1_SPLITS=N` | single-row split count (default 56 full / 28 SWA; `--splits*` overrides it) |
| `BRUCE_MV_SPLITS=N` | 2..4-row verify split count (default 42 full / 21 SWA) |
| `BRUCE_FAST_NORMS_M1=0` | keep the 1024-thread scalar M=1 norms |
| `BRUCE_ATTN_DECPIPE=0` | disable cp.async prefetch on the 5..8-row fallback |
| `BRUCE_LAZY_THR=N` | lazy-V skip threshold in nats (default 14) |

## 4f. Decode pass 2026-10-28 (round 4): loop-1 attention reuse

Round 3 put single-row decode at the DRAM floor: 35.35 ms/token at 256K against a
34.0 ms floor, of which 23.3 ms was the 8 GB KV stream of the 14 full-attention
block instances.  The only way left was to read fewer KV bytes.

Dory 1.0 is a **looped** transformer: the 40 recurrent blocks execute twice per
token, and that includes five *full-attention* blocks (`blk.22, 30, 38, 46, 54`).
Loop 1 queries the **same KV cache** as loop 0 with a query that is a tiny
perturbation of the loop-0 query.  Measured on the activation dumps (plan §3.3):
cosine 0.9826-0.9912 on the block input and 0.9846-0.9990 on the block update.  So
loop 1 recomputes a nearly identical softmax over 5/14 of the full-attention KV
stream — 2.86 GB at 256K.

Tile-level skipping is dead on this model (a 16-key tile is never 4 nats cold), but
the *whole output* can be reused: the loop-0 attention output `aout` is kept for
those five blocks and loop 1 reads it instead of launching the attention kernel and
its split-combine.  Residency cost is **640 KiB** (5 slots x 8 decode rows x 4096
floats), and it applies to decode and verification only: prefill (`rows > 8`) never
enters the path, so prompt numerics and prefill speed are untouched by construction.

### What changed

| change | mechanism | code |
|---|---|---|
| **loop-1 attention reuse** | the 5 recurrent full-attention blocks write their loop-0 attention output into a per-block slot (`M.aout_l0`); loops >= 1 skip the attention kernel *and* the split-K combine and point the gate norm at that slot.  The qkv GEMM and `k_qkv_prep` still run in loop 1, so the KV written per position (last-loop semantics) is unchanged. | `l1reuse()`, `M.l1_slot[]`, `aout_p` in `forward_launch` / `forward_batch` |
| **Q6_K verify rows >= 3 on the raw 210 B layout** | the aligned 160 B `w.dp` decode kernel moves 1.28 GB/token; the raw GGUF layout moves 0.84 GB.  Its 11 narrow unaligned loads cost more per byte at 1-2 rows, but as the verified row count grows the byte saving wins: tie at 2 rows, -0.7 % at 4, -1.2 % at 6, **-10 % at 8** (21.0 -> 18.9 ms/round at 4K). | `launch_gemm_b()` (`BRUCE_Q6GEMM`) |

### Measured (RTX 3060 @ 1890 MHz, 56 C, graph replay, `--bench 20 --bench-at ctx`)

| ctx | before ms/tok | **now ms/tok** | before tok/s | **now tok/s** | delta |
|---:|---:|---:|---:|---:|---:|
| 4 096 | 14.588 | **14.443** | 68.55 | **69.24** | -1.0 % |
| 32 768 | 16.908 | **15.905** | 59.14 | **62.87** | -5.9 % |
| 131 072 | 24.844 | **21.005** | 40.25 | **47.61** | -15.5 % |
| 262 144 | 35.379 | **27.765** | 28.27 | **36.02** | -21.5 % |

The saving tracks the predicted 5/14 of the full-attention KV stream: 0.36 GB at
32K (0.95 ms), 1.43 GB at 128K (3.8 ms), 2.86 GB at 256K (7.5 ms).  The KV floor at
256K drops from 21.0 to **13.5 ms**, so the whole token floor is now ~26.8 ms and
the measured 27.8 ms is within 4 % of it.

Verify rounds (`--bench-rows`, host sync + argmax included):

| ctx \ rows | 2 | 4 | 8 |
|---|---:|---:|---:|
| 4 096 | **16.69** (17.13) | **17.61** (18.04) | **18.82** (21.26) |
| 32 768 | **18.33** (19.55) | **19.02** (20.47) | **20.94** (24.71) |
| 131 072 | **24.17** (28.60) | **24.34** (28.52) | **28.16** (35.96) |
| 262 144 | **31.83** (40.39) | **31.33** (39.32) | **37.82** (50.70) |

At 128K a 4-row round now costs the same as a single-row pass (24.3 vs 21.0 ms) and
at 256K it is *cheaper* than the 2-row round, because the round cost is weights plus
one shared KV sweep.  A 4-row round at 256K is 31.3 ms = **127 tok/s at the
3.23 tokens/round** measured on long code, and the 8-row round is 18.9 ms at 4K
(previous 21.3), which is what makes long-match speculation pay.

End-to-end generation (same prompts as §4e):

| workload | before | **now** |
|---|---:|---:|
| 4K, 121-token code, plain greedy | 71.2 tok/s | **72.0** |
| 4K, 121-token code, `--draft 7` | 75.8 tok/s | **77.9** |
| 32.7K CUDA corpus, plain (64-token window) | 59.5 tok/s | **63.2** |
| 32.7K CUDA corpus, `--draft 7` | 101.4 tok/s | **135.7** (2.62 tok/round) |
| 128K repetitive code, `--draft 7` | - | **253 tok/s** (7.00 tok/round, 8-row rounds) |

Prefill is unchanged (2812 / 2659 / 2184 tok/s at 4K / 8K / 32K, the same as the
round-3 build): the reuse path is guarded by `rows <= 8` and the Q6_K change only
touches `launch_gemm_b`'s `rows <= 8` branch, so the prompt kernels are not even
reachable.  Measured with the flag on and off: 2140.9 vs 2136.6 (4K), 1760.2 vs
1757.1 (32K, prepack off at ctx 262144), i.e. inside run noise.

### Quality gates (all run on this build)

* 121-token code prompt, plain greedy **128/128 identical** to the pre-change build
  (`tmp/final_plain_ids.txt` vs `tmp/d2_code128.txt`), and **128/128** with
  `--draft 7` (`tmp/final_spec_ids.txt` vs `tmp/d2_spec128.txt`).
* Prose prompt, 48 greedy tokens: **48/48 identical**.
* Chat prompt, 48 greedy tokens: first divergence at token 6.  That step was dumped
  and compared: the two candidates were 3039/17984 at 18.06/17.85 nats in the exact
  path and 17.93/17.83 in the reuse path, i.e. a **0.10-0.21 nat near-tie**, with
  centred cosine 0.999711, rms |dlogit| **0.076 nats** and max |d| 0.42.  That is the
  engine's documented noise class (its own per-logit noise is ~0.06 nats and the
  prefill-tc change moves logits by 0.5), so this is a coin flip, not a quality
  change.
* Q6_K verify path (raw 210 B): 128/128 plain and 128/128 `--draft 7`.

### Excluded / measured negative results this round

| idea | measured result | why rejected |
|---|---|---|
| extend the reuse to the 15 recurrent **SWA** blocks as well | 4K 14.17 vs 14.44 ms, 128K 20.76 vs 21.04 (-0.3 ms/token) | code prompt diverges at token 78 and prose at token 5; the 0.3 ms is not worth leaving the 128/128 gate |
| multi-row dp4a Q5_K GEMV (M accumulators on the single-row kernel) | 242 GB/s at M=2, **115 GB/s at M=4**, 74 at M=8 vs 284-310 for the tensor-core verify kernel | M x activation LDS reads plus M x dp4a issue dominate once M > 2 — re-confirmed the plan's original finding |
| 2..4-row split count 56 / 64 (vs 42 full / 21 SWA) | 4K rows4 18.90 / 19.59 vs **18.00**; 128K 25.40 / 28.38 vs **24.79** | the split merge grows faster than the wave fill helps |
| Q6_K raw 210 B layout at 1-2 rows | 4K rows2 16.74 vs 16.75 | below 3 rows the unaligned narrow loads cost exactly the saved bytes |
| verify-GEMM grid remapping (flat tile-stride, 1 block/SM) | qkv 293 vs 284 GB/s, gu 310 vs 302, wo 211 vs 205 | +2-3 % per phase, ~0.15 ms/round, not worth new kernel code yet |
| verify GEMM with coalesced 16 B weight loads staged through smem | qkv 258 vs **284 GB/s**, gu 278 vs **310**, wo 172 vs **205** | the direct 4 B loads already hit L1 and the sectors are reused by the paired instruction; the smem bounce (5.5 LDG.128 + 5.5 STS.128 + 14 LDS) costs more than the coalescing gains |

### What is left (measured estimates, not implemented)

* **q4 V for the KV cache**: the reuse left 9/14 of the full-attention KV on the read
  path, i.e. 5.14 GB/token at 256K of which V is 2.57 GB.  A 4-bit V (per-32 fp16
  scale, 144 B per 256 values per kv head) would cut 1.28 GB = ~3.3 ms at 256K
  (27.8 -> 24.5 ms) and ~1.7 ms at 128K, at the cost of a 0.076-nat-class logit
  perturbation (V errors average out over the softmax; K stays q8_0 so the scores are
  exact).  It touches the KV writers plus `k_attn_m1_w` / `k_attn_mv_w` and, if the
  cache is single-layout, the prompt attention kernels too — the one change that
  would move prefill, so it needs its own prefill gate.
* **Verify GEMM on 16-row tiles with ldmatrix**: worth at most the remaining
  ~2.9 ms/round (qkv 1.1, gu 0.6, wo 0.6, down 0.6 at 4K).  The two cheaper load-path
  experiments above both lost, so this needs a real staged+ldmatrix kernel.
* **Loop-1 reuse for the prompt path**: the same five blocks re-read the KV during
  prefill (36 % of the full-attention prefill reads at 128K+).  Not done because it
  moves prompt logits, and prefill is the quality gate.

### Switches added

| switch | meaning |
|---|---|
| `--loop1-reuse` / `--no-loop1-reuse` (`BRUCE_L1REUSE=0/1`) | reuse the loop-0 attention output in loop 1 (default on; `--no-...` restores the bit-exact decode path) |
| `BRUCE_Q6GEMM=0` | force the aligned 160 B `w.dp` dp4a kernel for verify rows (default: the raw 210 B batched kernel for 3..5 rows, the tensor-core kernel for 6..8) |
| `BRUCE_VERIFYPF=0` | disable the superblock prefetch in the Q5_K / Q6_K verify GEMMs (§4g) |
| `BRUCE_Q6TC=0` | disable the tensor-core Q6_K verify GEMM for 6..8 rows (§4g) |

---

## 4g. Decode pass 2026-10-28 (round 5): the verification round's weight stream

Round 4 left the **single-token** path within 4 % of its DRAM floor at every depth
(14.42 ms/token at 4K of which 13.1 ms is the 4.15 GB weight stream at 324-372 GB/s
per phase, i.e. 93-98 % of this card's measured 381 GB/s read ceiling; 27.76 ms at
256K of which 13.5 ms is the irreducible KV stream).  No kernel change can move that
without quantising the payload.  The remaining kernel-level decode lever is the
**2..8-row verification round**, which is where all speculative decoding time is spent.

Two measured inefficiencies, both in the round's GEMMs:

1. `k_gemm_tc_verify_q5k` streamed weights at **265 GB/s** at 8 rows, against 323-356
   for the M=1 GEMV on the same bytes.  Ablating the MMA + epilogue (keeping the exact
   load pattern) reached **339 GB/s** at 28 blocks, so the loss was *latency*, not
   traffic or coalescing: with 8 warps/SM the `load -> MMA -> scale-epilogue` chain of
   one superblock is not covered by the next iteration's loads.
2. The Q6_K tensors at `rows >= 3` used the raw 210 B GGUF layout, whose 6-bit fields
   can only be read as two 16-bit loads per 4-byte group: **3.59 ms per 8-row round at
   120-143 GB/s**, 1.7 ms more than the M=1 dp4a path spends on the same tensors.

### What changed

| change | mechanism | code |
|---|---|---|
| **superblock prefetch in the Q5_K verify GEMM** | both rows' whole 176 B superblock (header, 5th-bit plane, all eight 16 B nibble groups) is loaded into registers one iteration ahead of the MMA that consumes it; no extra shared memory, no barrier. 28 blocks (one CTA/SM) for large N, one warp per tile (<=20 blocks) for the 160-tile `wo`/`down` shapes. | `k_gemm_tc_verify_q5k<M,8,PF>` |
| **tensor-core Q6_K verify GEMM** | new `mma.m16n8k16.s8` kernel on the aligned 160 B repack: 16 weight rows on A, up to 8 verified rows on B (activations staged once in shared), one instruction per 16-value scale group (half the cycles of k32 on SM86), 4-byte loads, one barrier, same prefetch. Routed for rows >= 6, where it beats the raw layout; 3..5 rows keep the raw path (34 % fewer bytes). | `k_gemm_tc_verify_q6k<M,8,PF>`, `launch_q6_verify_fused()` |

### Measured (RTX 3060, 1875-1897 MHz, 3 reps, graph replay)

Same machine state, baseline = round-4 build:

| verify round (`--bench-rows`) | ctx \ rows | base ms | **now ms** | delta |
|---|---|---:|---:|---:|
| 4 096 | 2 | 16.73 | **16.41** | -1.9 % |
| 4 096 | 4 | 17.68 | **17.27** | -2.3 % |
| 4 096 | 6 | 18.09 | **17.57** | -2.9 % |
| 4 096 | 8 | 18.92 | **17.60** | **-7.0 %** |
| 32 768 | 4 | 19.14 | **18.77** | -1.9 % |
| 32 768 | 8 | 21.08 | **19.62** | **-6.9 %** |
| 131 072 | 4 | 24.49 | **24.00** | -2.0 % |
| 131 072 | 8 | 28.16 | **27.11** | -3.7 % |
| 262 144 | 4 | 31.49 | **31.04** | -1.4 % |

Plain decode is unchanged by construction (the M=1 path never enters these kernels)
and measured 14.42 / 15.90 / 21.01 / 27.76 ms/token at 4K / 32K / 128K / 256K
(69.3 / 62.9 / 47.6 / 36.0 tok/s), i.e. within run noise of round 4.

End-to-end generation (same prompts as §4f):

| workload | base | **now** |
|---|---:|---:|
| 32.7K CUDA corpus, `--draft 7` (64-token window) | 135.5 tok/s (2.62 tok/round) | **139.0** |
| 4K, 121-token code, `--draft 7` | 76.1 tok/s | **76.5** |

The 4K code prompt gains little end-to-end because 85 of its 115 rounds find no n-gram
match at all and run the plain path; the round cost only matters on the 30 speculative
rounds.  Long-context rounds gain less than at 4K because they are KV-bound: an 8-row
round at 128K is 13.5 ms of KV stream plus ~13 ms of weights.

### Where an 8-row round goes now (`--probe` + nsys kernel diff, 4K, M=8)

| component | ms/round | note |
|---|---:|---|
| Q5_K verify GEMM (199 launches) | **10.35** | 278 GB/s of the raw bytes (was 10.86 / 265) |
| Q6_K verify GEMM (50 launches) | **2.72** | was 3.59 (raw 210 B path) |
| verify attention + split combine | 1.92 | KV floor 0.57 ms at 4K; latency-bound |
| logits Q8_0 | 1.03 | 345 GB/s, at the ceiling |
| silu / resid / gate-norm / qkv-prep (280 launches) | 1.34 | 4-8 us each, launch-bound |
| argmax + graph node overhead | ~1.0 | |
| **total** | **17.60** | 16.58 ms of kernel time |

### Quality gates (all run on this build)

* 121-token code prompt, plain greedy **128/128 identical** to the round-4 build
  (`tmp/final_plain_ids.txt` vs `tmp/v3_plain_ids.txt`).
* Same prompt, `--draft 7`: **128/128 identical** to plain greedy.
* 32.7K CUDA corpus, `--draft 7`, 64 tokens: **64/64 identical** to the round-4 build
  and to plain greedy.
* Prose prompt: plain greedy 128/128 identical to the round-4 build; `--draft 7`
  128/128 identical to plain greedy.
* Chat and long prompts: plain greedy 128/128 identical to the round-4 build.  Their
  *speculative* continuations resolve differently (chat 16/128, long 100/128 identical),
  because both prompts sit on near-ties and the new Q6_K kernel sums the two 16-value
  scale groups in a different fp32 order.  That is the class §3 already documents: chat
  is the prompt where the engine's own plain path only agrees with the reference on
  11/128 tokens, and the round-4 loop-1 reuse already made chat diverge at token 6 on a
  0.10-0.21 nat tie.  Spec-vs-plain agreement is 2/128 differences on `long` (was 28/128)
  and 106/128 on `chat` (was 82/128): a single early coin flip on chat re-branches the
  rest of the continuation.  `BRUCE_Q6TC=0` restores the round-4 arithmetic for
  chat/prose A/B.
* New Q6_K kernel vs the shipping dp4a kernel on synthetic weights (K=2560, M=8):
  max |diff| 3.8e-6 over 2048 outputs; structured probes (all-ones, ramp weights,
  per-group scales) reproduce the reference exactly.
* **Prefill unchanged, measured with both binaries back to back**:
  2806/2665/2181/1100 tok/s (base) vs 2800/2667/2181/1100 (now) at 4K/8K/32K/128K;
  repeated runs of either binary span 2800-2820 at 4K, so this is run noise.  The new
  kernels are unreachable for `rows > 8` by construction.
* `tools/final_suite.py` on this build: code plain 71.5 tok/s / spec 76.6 (1.43
  tok/round, 128/128 spec==plain, 128/128 vs the reference), prose 71.4 / 71.4
  (128/128), long 71.4 / 97.9 (3.11 tok/round), 16K prefill 2499 tok/s and a 16K
  4-row round 17.84 ms (was 19.59 in §4b).

### Excluded / measured negative results this round

| idea | measured result | why rejected |
|---|---|---|
| depth-2 prefetch / relaxed `__launch_bounds__(256,1)` in the verify GEMM | 125 vs **276** GB/s (qkv) | 123 regs + 256 B stack frame: the second buffer spills, so every iteration round-trips local memory |
| 16 warps/CTA (28 or 56 blocks) | 145 vs **290** GB/s | halves the tiles per warp and adds a partial second wave; 448 warps for 576 tiles is 1.3 waves |
| two 16-row tiles interleaved per warp (more MLP) | 221 vs **290** GB/s | 168 regs, one CTA/SM, and the extra live fragments evict the L1 sectors the pattern relies on |
| bias-trick epilogue (`0x4B400000`, removes the I2F) | 276 vs 276 GB/s | neutral here: the epilogue is not the limiter |
| coalesced 16 B weight staging through shared memory | (round 4) 242 vs **285** GB/s | re-confirmed: the direct 4 B loads already hit L1 and the bounce costs more than the coalescing |
| more verify splits at 4K (28/42/64 vs 21) | 18.02 / 18.46 / 18.61 vs **17.59** ms | 21 splits is already exactly one wave; more splits multiply the partials read by the combine |
| fewer SWA splits at 8 rows (4/6/8/12 vs 21) | 17.53 / 17.57 / 17.57 / 17.61 vs **17.60** | inside run noise |
| M=1 `--gemv-blocks` 56/84/140/168/224/336 | 14.88 / 14.52 / 14.45 / 14.72 / 15.66 / 14.80 vs **14.43** at 112 | the default is already the optimum; more blocks thrash L1/L2 |

### What is left (measured, not implemented)

* **q4 V for the KV cache** is the one remaining >5 % decode lever at long context:
  5.14 GB/token of KV at 256K of which V is 2.57 GB, so a per-32 fp16-scale int4 V
  (144 B per kv head per position instead of 272) saves ~1.06 GB = **~2.8 ms at 256K
  and ~1.4 ms at 128K** (~11 % / ~7 %).  It needs a second (decode-only) V layout so
  the prompt attention kernel and its logits stay untouched, plus ~1.2 GB of VRAM at
  256K; the quality cost is a 0.07-nat-class logit perturbation (V errors average out
  over the softmax, K stays q8_0 so the scores are exact).
* **The remaining 8-row gap** at 4K: Q5_K verify 10.35 ms vs 9.1 ms if it matched the
  M=1 GEMV rate.  Closing it needs A-fragment-major weights (a second 3.1 GB prepack
  does not fit at low ctx) or a cp.async + `ldmatrix` stage, which the two staging
  experiments above say will not pay for itself.
* **The verification round's elementwise chain** (1.34 ms) is 280 launches of 4-8 us;
  it can only shrink by fusing norms into GEMV prologues, worth at most ~0.3 ms.

---

## 4h. Prefill pass 2026-10-28 (round 5): deep-context attention and the prepack budget

Round 4 published prefill at **2802 / 2695 / 2502 / 2186 tok/s at 4K / 8K / 16K / 32K and
1101 at 128K** (defaults); the same-session re-baseline used in the table below is
2782 / 2640 / 2182 / 1101.  The depth curve was the problem: 72 % of the last 128K
chunk was attention, and the 144 B Q5_K pre-unpack -- worth **+14 %** at every depth
-- was disabled above 32K as a blunt proxy for "the KV cache will not fit".  This
round fixes the budget rule and attacks the attention tile itself.  No decode or
verification kernel was touched.

| prompt | 4K | 8K | 16K | 32K | 128K | 256K |
|---|---|---|---|---|---|---|
| before (defaults, same session) | 2782 | 2640 | 2500 | 2182 | 1101* | - |
| **now** | **2803** | **2667** | **2500** | **2213** | **1343** | **805** |
| us/token | 357 | 375 | 400 | 452 | 744 | 1242 |

Intermediate point: a 65,400-token prompt at `--ctx 65536` measures **1824 tok/s**
(prepack on, 2205 MiB free).  The same path through the HTTP server was re-checked
on this build: `/completion` with the 131,043-token id array reports 1343.2 tok/s
and 97.56 s, identical to the CLI (the server's `forward_batch_tokens` now records
the chunk position so it uses the same wide-block dispatch).

\* the 128K "before" is the shipped auto rule (`--q5k-prepack` skipped above 32K,
1098.8 tok/s measured today); forcing the prepack on gave 1251.4, so this round is
**+22 % vs the default** and **+7.3 % vs the old forced-prepack best**.  The 256K
point is a 262,000-token code prompt on this build; the last published 256K number
(626 tok/s) is from the 2026-10-24 build and a different prompt set, so it is not a
same-build baseline.

### The Q5_K prepack budget (host)

The decision ran before `alloc_buffers()` and only checked the 3.3 GiB of extra
prepack against free VRAM, so at 64K-128K it took the headroom the KV cache was
about to need; at 256K it was skipped.  It now reserves
`9 * n_ctx * 1088 * 2 + 27 * SWA_SLOTS * 1088 * 2` for the KV cache plus 512 MiB:

| n_ctx | kv reserve | prepack | free after everything |
|---:|---:|---|---:|
| 4 096 | 191 MiB | on | 3349 MiB |
| 32 768 | 727 MiB | on | 2817 MiB |
| 131 072 | 2 563 MiB | **on** | 981 MiB |
| 262 144 | 5 011 MiB | off (does not fit) | 2037 MiB |

Forcing it on at 256K is a trap worth documenting: the allocation "succeeds" with
0 MiB free and WDDM silently pages the KV cache, making a 512-row chunk take
**10.98 s instead of 0.70 s**.  The budget check now applies to `--q5k-prepack on`
too.

### The deep-chunk attention tile: 48 rows, 16 threads per key

Two changes, both prefill-only (`rows > 8`), both measured on the 512-row chunk:

1. **48-row CTA (12 warps) for full layers above ~12K depth.**  The row-split
   kernel holds Q (64 regs) and O (64 regs) per lane, so 168 regs caps it at 12
   warps = 48 rows and one CTA/SM.  That is half the KV passes and half the
   per-tile dequant/barrier work per row of the 24-row/6-warp shape, at the same
   12 warps/SM.  Measured attention-only crossover is 8-16K: 3.5K 31.1 (nw6) vs
   34.9 ms (nw12); 16K 77.5 vs 73.7; 32K 134.8 vs 121.7; 128K 482 vs 417.  The
   dispatcher switches at `pos + rows >= 12288` (`BRUCE_ATTN_NW_DEPTH`).
2. **16 threads per KV key in the dequant.**  The raw q8 tile is dequantised
   between the tile's two barriers by 128 threads (8 per key, 32 B each).  At 12
   warps there are 256 threads available, so each key now gets 16 threads with
   16 B of K and V: the same math, bit-identical per element, but half the serial
   chain.  This was worth **8 % at 32K and 9.4 % at 128K** on top of the shape
   change (attention 130.8 -> 120.5 ms at 32K, 460.6 -> 417.3 at 128K).

Attention now runs at **17.3 T-MAC/s at 32K and 18.8 at 128K** of the card's
measured 26.9 T-MAC/s `mma.m16n8k16.f16` peak (64-70 %).  The mixed LDSM+HMMA
probe says those two instructions overlap perfectly at 1:1, so the remaining 30 %
is the score -> softmax -> PV dependency chain and the two barriers per 16-key
tile, not smem bandwidth.

### Where a 512-row chunk goes now (`--probe --bench-rows 512`, last chunk at depth)

| phase (ms per 512-row chunk) | 4K | 32K | 128K |
|---|---:|---:|---:|
| qkv gemm | 40.5 | 40.8 | 41.1 |
| ffn gate+up | 51.3 | 51.5 | 51.7 |
| ffn down (Q5_K + Q6_K) | 31.1 | 31.2 | 31.4 |
| wo | 15.5 | 15.7 | 15.9 |
| attention | 30.8 | 121.4 | 418.2 |
| silu / resid / combine+gate / prep / norms / embed / logits | 21.1 | 22.4 | 23.5 |
| **total** | **190.3** | **282.1** | **578.6** |

The GEMM block is flat at **139-140 ms per chunk (18.9 T-MAC/s, 35 % of the 53.9
T-MAC/s int8 peak)** at every depth; the whole depth curve is attention.  At 128K
the last chunk is 1130 us/token (885 tok/s at depth) while the prompt average is
1343 tok/s, because the average depth seen by a chunk is 64K.

### Excluded / measured negative results this round

| idea | measured result | why rejected |
|---|---|---|
| two-buffer fp16 K/V tile, one barrier per tile (`BRUCE_ATTN_DBUF=1`) | 32K 126.8 vs **120.5 ms**; 128K 426.6 vs **419.6** | the dequant is already off the critical path; issuing it before the tile's MMA delays the 8 dequant warps instead of overlapping with the other 4 |
| 48-row block below 12K | 3.5K 34.9 vs **31.1 ms**, 8K 49.0 vs **47.9** | a block then covers only a few KV tiles and the 48-row Q load dominates |
| split count 0/2/4/8/16 at 128K | 417.1 / 420.3 / 419.6 / 420.4 / 420.3 ms | inside run noise; the default (8) stays |
| forcing the Q5_K prepack at 256K | 10.98 s vs **0.70 s** per 512-row chunk | 0 MiB free: WDDM pages the KV cache |
| 3-stage cp.async pipeline / 128x64 at 2 CTAs (round 4) | 17.1 T-MAC/s / spills | staging is copy-path bound, not latency bound |

### What is left (measured, not implemented)

* **The GEMM is additive, not overlapped.**  Per 128x128 CTA stage at 512 rows the
  measured budget is MMA 2058 + scale epilogue 1360 + LDS 1120 + cp.async staging
  2170 = 6708 cycles against 6250 measured (93 % of the sum).  The 16,384-output
  CTA tile is forced: 64 accumulators/lane x 256 threads is the whole 64K register
  file, and 128x128 minimises `(BM+BN)/(BM*BN)` at that budget.  Overlapping
  staging would need two resident CTAs at <=128 regs, which cannot hold the
  accumulator tile.  The measured no-staging/no-epilogue ceiling is 40 T-MAC/s.
* **Attention's remaining 30 %** is the softmax dependency chain; more independent
  tiles per warp needs registers the kernel does not have (168 of 170 at 384
  threads).  An int8 QK^T was measured slower in round 3 (per-32 scale epilogue),
  and int8 PV would require quantising P (a 0.4 % class perturbation).
* **Q6_K** is 8 % of MACs but ~20 % of GEMM time: its 16-value scale group forces
  two `mma.m16n8k16` per 32 values (2x the MMA per MAC) and its 160 B/128-value
  repack is 1.25 B/value.  A 144 B variant with fp16 scales is a ~2 % GEMM lever.
* **`wo` is the one GEMM outlier:** 15.9 ms/chunk (12.1 T-MAC/s) where per-stage
  scaling from the standalone harness predicts ~10.5 ms; its 4 x 20 = 80-CTA grid
  is 2.86 waves on 28 SMs, so a persistent / stream-K schedule is the candidate
  (worth at most ~1.5 % end-to-end at 4K).

### Quality gates run on this build

* 121-token code prompt, plain greedy: **128/128 identical** to the pre-change
  build (`tmp/fin_code.txt` vs `tmp/v4_plain_ids.txt`).
* 16K prompt, 16 greedy tokens: **16/16 identical** between the 48-row and 24-row
  attention paths; the 32K prompt is 24/24 identical.
* Decode (graph replay): **14.42 / 15.92 / 21.00 ms** at 4K / 32K / 128K vs
  14.44 / 15.95 / 21.00 before; 4-row round **17.31 / 18.79 / 24.07 ms** vs
  17.35 / 18.76 / 24.17; 8-row **17.51 / 19.65** vs 17.65 / 19.67.  All inside
  the +-0.4 % run band, both directions.
* The 48-row shape changes only the fp32 reassociation of the split merge (ns 8 vs
  4), the documented near-tie class.

### Switches added this round

| switch | meaning |
|---|---|
| `BRUCE_ATTN_FULL_NW=N` | warps (4 rows each) per full-layer prompt CTA, default 12 |
| `BRUCE_ATTN_NW_DEPTH=N` | depth above which the wide block is used, default 12288; `0` forces it |
| `BRUCE_ATTN_DBUF=1` | the measured-negative two-buffer one-barrier tile loop |
| `--q5k-prepack auto\|on\|off` | `auto` now means "fits beside the KV cache", not "ctx <= 32K" |

---

## 4i. Multi-slot serving pass 2026-10-29 (round 6)

The single-slot decode graph and every prefill kernel are untouched: this round changes
only the `--slots N` batch path and the server's prompt scheduler.  The gates at the end
show plain decode, speculative verification and prefill are unchanged in a one-slot
process.

### What changed

| lever | mechanism | measured |
|---|---|---|
| **batched attention split count scales with rows** | `attn_ms_splits` used the M=1 split count (56 full / 28 SWA) for every row, so an 8-row step launched 4 kvh x 56 x 8 = 1792 one-warp blocks and every split's `(m, l)` partial fed the combine kernel.  It now picks `ceil(56/rows)` full and `ceil(28/rows)` SWA, keeping the one-warp grid near one wave (~224 blocks); the per-row KV bytes are unchanged because the key ranges are disjoint. | 8-row step 21.92 → **20.40 ms** (eager, 2K/slot) |
| **coalesced double-buffered Q5_K verify tiles** | the 2..8-row kernel's register-direct loads issue 22 four-byte loads per lane per super-block, and each warp instruction touches eight rows x 16 B (2x the 32 B sectors it uses).  For K=2560 it now stages its 16x176 B tile with 16 B-per-lane `cp.async` into two per-warp shared buffers and reads it back from shared; K=7168 (ffn down) keeps the register-direct loop because the 45 KB does not fit beside its activation stage. | −1.5 % at 8 rows, −2 % at 4 rows |
| **prefill burst scheduling** | the server used a 256-row prefill chunk whenever any other session was active, even with eight prompts queued and no decoder running.  Now several queued prompts drain at the 512-row chunk (`chunk_burst`), a single prompt behind live decoders uses 384 rows (a multiple of the 128-row GEMM tile), and solo prefill stays at 512.  This is scheduler-only: the kernels are the same. | queued prefill **897 → 993 tok/s** at 8 slots |

The 256→384 change is worth 6.6 % of prefill throughput (2401 → 2560 tok/s at 4K) and
keeps the chunk near the documented ~150 ms decode gap at 8K depth.  The 512-row burst is
only used when at least one other prompt is already queued, so a lone interactive prompt
behind running decoders keeps the narrow chunk.

### Measured (`serve_concurrency.py`, ctx 32768 = 4096/slot, 512-token prompts, 256 generated)

| slots | rows/step | engine step (before → now) | per-request decode (before → now) | aggregate before → now |
|---:|---:|---:|---|---:|
| 1 | 1.0 | 14.4 → **14.3 ms** | 70.4 → 70.3 tok/s | 67.2 → **67.2** |
| 2 | 2.0 | 16.4 → **15.7 ms** | 61.5 / 58.9 → 61.3 / 63.7 | 111.5 → **117.1** |
| 4 | 3.9 | 17.7 → **16.5 ms** | 50-57 → 54-60 | 190.9 → **206.0** |
| 8 | 7.8 | 19.3 → **17.7 ms** | 41-53 → 45-56 | 306.4 → **341.1** |

The same table on deeper prompts (2048 tokens, 128 generated) moves 67.0 → **73.3**,
89.1 → **100.1** and 107.8 → **124.9** tok/s for 2/4/8 slots; the 8-row step at 2K/slot is
21.9 → **18.9 ms** wall (graph replay + argmax + host sync).

### Quality and single-slot gates

* `--slots 1` decode: **69.30 tok/s / 14.431 ms** at 4K and **62.80 / 15.922 ms** at 32K
  (the pre-change build measured 69.35 / 62.89).
* `--slots 1` speculative: the 121-token code prompt is **128/128 identical** to the
  pre-change build for plain greedy and `--draft 7`; the staged verify kernel is
  byte-identical to the register-direct one (64/64 tokens at both settings).
* `--slots 1` prefill: **2800 / 2701 / 2214 tok/s** at 4K / 8K / 32K (pre-change
  2803 / 2667 / 2213).
* `--ms-test` at 4 and 8 slots: every row's argmax matches the per-slot M=1 path, and the
  attention A/B (verify vs MS, same GEMM) has 0/8 argmax mismatches.

The split count depends on the number of rows in the step, so a batched run's rounding —
like the existing M=1-vs-batched difference — can depend on the load pattern.  That is the
near-tie class documented in §7; `BRUCE_MS_SPLITS=14` pins one split count for every batch
size when reproducibility across load patterns matters more than the last ~1 %.

### Switches added this round

| switch | meaning |
|---|---|
| `BRUCE_MS_SPLITS=0\|1\|N` | multi-slot attention splits: `0` = old M=1 count, `1` = scale with rows (default), `N>=2` = fixed count |
| `BRUCE_VERIFY_STAGE=0` | disable the staged Q5_K verify tiles (register-direct loop) |
| `BRUCE_SERVE_CHUNK_BURST=N` | prompt chunk when several prompts are queued (default 512) |
| `--probe-ms` / `--bench-ms` | dev: attribute / time one multi-slot batched step directly |

---

## 5. Where the time goes, and why 100 tok/s is conditional

Per decoded token the weights that must be streamed are

```
q5k 3.13 GB + q6k 0.48 GB + q8 logits 0.36 GB = 3.97 GB
```

(the recurrent blocks are visited twice).  Measured achievable GEMV bandwidth on this card
is 345 GB/s, so the **sequential-decode floor is 11.5 ms = 87 tok/s** - 100 tok/s needs
more than one token per weight pass, or skipping loop 1.

Current 16.4 ms breakdown (nsys, 4K): q5k 10.2 ms (~310 GB/s, 90 % of the GEMV ceiling),
q6k 2.2 ms, q8 logits 0.95 ms, producer kernels 1.5 ms, attention 0.6 ms, prep 0.4 ms.
The 0.06 ms-per-row tensor-core verify path makes a 4-row round 18.2 ms at 4K and
19.7 ms at 16K (3.1 GB/round, ~85 % of the bandwidth of a 1-row pass) and an 8-row
round 21.8 ms at 4K.

Ways past 87 tok/s, all requiring more than one token per weight pass:

* **Speculative decoding** - a verification of M rows costs almost the same traffic as one
  decode (measured: 4 rows = 1.2x, 8 rows = 1.6x).  With the n-gram drafter this reaches
  72.2 tok/s on the 121-token code prompt (1.43 tokens/round), 95.1 tok/s on the 32K
  CUDA corpus (2.10 tokens/round), and 281 tok/s on an 8K repeated-code prompt
  (7.00 tokens/round).  On free-form prose the drafter still finds only
  1.1-1.4 tokens/round, so speculation is near break-even there.  A trained draft head
  (Medusa/EAGLE) or tree verification would be the next acceptance lever.
* **Confidence-gated loop-1 exit** (`--adaptive-loops`) - the loop-0 state is already at
  the 2-loop fixed point for most tokens; replaying a 1-loop graph on those tokens gives
  ~11.5 ms/token and 93 tok/s plain, 95-99 tok/s with the drafter, and 122 tok/s at 32K.
  This is the one lossy mode: 128/128 token-identical on the code prompt, coherent but
  differently-sampled text on prose/chat.
* **Batched serving** (multiple sequences) reuses each weight pass across rows; that is the
  regime big engines target, and the batched path here already reaches 148 GB/s in prefill.
  Implemented as `--slots N` in the server (§7): 4 concurrent sequences reach **206 tok/s**
  aggregate at 4K (3.1x a single sequence) and 8 reach **341 tok/s**, with single-request
  prefill and decode unchanged.

Long-context decode is KV-bound: 14 full-attention block instances x 4 kv heads x ctx x 544 B
(4.0 GB per token at 131072, 8.0 GB at 262144) against a 345 GB/s ceiling, i.e. 11.6 / 23 ms
of unavoidable traffic.  The aligned KV layout took 262144 from 96.8-103.3 to 64.2 ms/token
(attention 64 -> ~160 GB/s); what is left is the per-key serial online-softmax chain.

Verified speculation at long context is a different story, and the measurements say why:

    at 131072, a 4-row round:  weights ~26 ms
                    + 14 layers x 4 kvh x 131072 keys x 544 B = 4.0 GB of KV
                    measured 34.1 ms/round after the row-split verify + shared SWA pass
                    (KV floor 11.6 ms)

An ideal fp32 kernel would be memory bound there: the fma work for M rows is only
`M x 4 heads x 256 dims x 2` per key, ~2.4 ms per row over 7.34M key-visits, so 4 rows need
~9.5 ms of maths - below the 11.6 ms KV stream.  The fused kernel instead spends ~456 cycles
per key per warp: 128 fma, 160 for the 5-step warp reduction per row (SHFL issues at 1/4
rate), and 128 for two MUFU exps per row that all 32 lanes compute redundantly.  It is
**instruction-issue bound**, so each extra verified row costs ~20 ms instead of ~0, and 8
fused rows cost 7.3x a single row instead of 1x (measured: 5.3 -> 38.9 ms per 32K launch).

The fix is the standard one: tensor cores.  `mma.m16n8k16` folds the K-dimension reduction
into the instruction (removing both the shuffles and the redundant scalar work) and does the
fma at 4-8x the fp32 rate, with K/V dequantised int8->fp16 into shared-memory tiles the way
llama.cpp's `fattn` does (the oracle itself runs `-fa on`, i.e. fp16 KQ).  That brings a 128K
round is now `~21 ms weights + ~13 ms attention` = 34 ms/round at 128K, i.e. ~60 tok/s at
the 2.10 tokens/round measured on the 32K corpus, and past 90-100 tok/s once acceptance
reaches ~3.2 tokens/round on real code.  It costs exactness: fp16 KQ is no longer
bit-identical to the fp32 path (`q5k`/`q6k`/logits stay int8 dp4a and are untouched).

Two cheaper ideas were tried and rejected on measurement: frequency-voting the drafter over
several n-gram occurrences (recency beats frequency here: 3.23 -> 2.48 tokens/round on the
32K corpus) and a one-exp online softmax (the branchy state spilled: 45.2 -> 53.3 ms/round).

---

## 6. Debug / validation tooling

| tool | purpose |
|---|---|
| `tools/tok.py` | tokenise test prompts through the reference server |
| `tools/final_suite.py` | full benchmark + oracle equivalence + sampling run |
| `tools/gen_cmp.py` | engine vs server greedy generation on one prompt |
| `tools/pos_cmp.py` | full-logit comparison at a generation position (`oracle_dump.exe`) |
| `tools/oracle_dump.cpp/.exe` | sequential (batch-of-1) reference dumper: per-block activations + logits |
| `tools/validate.py` | prefill top-5 comparison against the server |
| `tools/gemmbench.cu` | batched GEMM microbenchmark (variants A/B/C/E) |
| `tools/mmq_test.cu` | standalone 128x128 Q5_K MMQ correctness + throughput harness |
| `tools/check_kernels.py` | model-level A/B gate: raw Q5_K vs 144 B pre-unpack at 16K (argmax/top-5/top-50) |
| `tools/gguf_stats.py` | GGUF metadata / tensor byte breakdown |

Server flags used for all comparisons: `-ngl 999
-fa on -c 4096 --cache-type-k q8_0 --cache-type-v q8_0 --override-kv
dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop`.

---

## 7. OpenAI-compatible server

`--serve` exposes the engine through a llama.cpp-style HTTP API. The server runs
`--slots N` sequences concurrently (default 4 when `--ctx >= 16384`, else 1; each
slot owns a contiguous slice of the KV pool, so per-request context is `--ctx / N`).
Requests that arrive together are decoded in one batched weight pass; a request that
is alone keeps the exact single-sequence fast path (CUDA-graph replay of the M=1
kernels).

```bat
engine\bruce.exe --model models\NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf ^
  --serve --host 127.0.0.1 --port 8080 --ctx 32768 --slots 4 --n-batch 512
```

Use `--slots 1` for the historical single-slot server (full `--ctx` for one request,
adaptive loops and speculative decoding always on). At the default `--ctx 262144`
four slots give 65536 tokens each; raise `--slots` only if the VRAM headroom allows
the per-slot SWA rings (+86 MiB per extra slot).

Add `--api-key KEY` to require authentication. The key is accepted as
`Authorization: Bearer KEY` or `X-API-Key: KEY`; `/health`, `/healthz` and CORS `OPTIONS`
requests stay public so browser clients can preflight.

Add `--draft 7 --adaptive-loops` to the server for the fast decode mode
(~95–99 tok/s on the 4K chat/code prompts on this card; loop-1 skipping is lossy on
free-form prose, so leave it off when exact 2-loop sampling matters). In multi-slot
mode `--draft` still applies whenever a request is the only active one, while
`--adaptive-loops` is single-slot only (`--slots 1`).

### Continuous batching (multi-agent serving)

Subagents and parallel tool calls usually arrive together and then decode in
lockstep. The server turns that into one weight pass per step:

* **Slots.** `--slots N` creates N independent sequences, each with its own KV
  partition, prefix cache, sampler RNG and logit row. Positions are local to the
  slot (`0..slot_ctx-1`), so every existing kernel runs unchanged inside a
  partition. Full-attention layers share the `--ctx` pool (N slices of `ctx/N`);
  SWA layers get one 2x1024 ring per slot because the ring index is `pos & 2047`.
* **One worker, one CUDA stream.** Request threads submit a token (or a prompt
  chunk) and block on a condition variable; a single worker owns the stream. All
  decode submits that are queued within `batch_wait_us` (default 2 ms, and it
  usually ends in tens of microseconds because sessions stay in lockstep) are
  fused into one `forward_batch_ms(rows)` pass: the weight stream is shared, each
  row runs the M=1 warp attention kernel with its own slot/position. `rows == 1`
  replays the per-slot M=1 CUDA graph, so a lone request is byte-for-byte the
  old path.
* **Prefill is chunked and interleaved.** A prompt behind live decoders is ingested
  in 384-row chunks (a multiple of the 128-row GEMM tile, ~150 ms at 8K depth),
  while queued bursts of several prompts drain at 512 rows and solo prefill stays at
  512. Queued decode steps run between chunks. `BRUCE_SERVE_CHUNK` overrides the
  decoder-behind chunk, `BRUCE_SERVE_CHUNK_BURST` the burst chunk.
* **Sampling.** Greedy rows read the argmax the batch graph already produced (no
  per-slot sampler launch, no extra sync). Temperature/top-p, penalties and
  logprobs fall back to the existing per-slot sampler under a scratch mutex.

Measured on the RTX 3060 with 512-token code prompts, greedy, `--ctx 32768`,
generation 256 tokens (the prompt depth, not the per-slot window, sets the step
cost here):

| slots | concurrent requests | rows/step | engine step | per-request decode | aggregate |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 14.3 ms | 70.3 tok/s | 67 tok/s |
| 2 | 2 | 2.0 | 15.7 ms | 61-64 tok/s | 117 tok/s |
| 4 | 4 | 3.9 | 16.5 ms | 54-60 tok/s | 206 tok/s |
| 8 | 8 | 7.8 | 17.7 ms | 45-56 tok/s | 341 tok/s |

The marginal cost of a batched row is ~0.35 ms of attention/GEMM work against a
14.3 ms weight stream, so aggregate throughput scales almost linearly until the
KV traffic of the longest sequence dominates. Mixed load behaves the same way:
an 8000-token prefill plus three 96-token decoders finishes in 5.0 s (prefill
2040 tok/s wall, decoders ~20 tok/s each) with the decode steps interleaved
between prompt chunks. Single-request speed is unchanged:
prefill 2800 vs 2803 tok/s and decode 69.3 vs 69.3 tok/s at 4K for slots=4 vs
slots=1, and the CLI `--bench` numbers are identical to the pre-change build.
The aggregate column is wall throughput including the queued prompt time, not
the sum of the per-request decode rates.
`python bench/serve_concurrency.py` reproduces the table (it starts the server,
runs n = 1/2/4/8 and prints the same columns).

Caveats:

* Greedy output can differ from a solo run at near-ties because rows >= 2 use the
  batched (verify) GEMM and the solo path uses the M=1 GEMV kernels, and the
  per-row attention kernel has its own rounding. This is the same class of
  trade-off the speculative verify path already makes: within one load pattern
  the batched output is deterministic, and the first divergence from a serial run
  is a single near-tie flip that then cascades. The batched argmax matched the
  solo argmax on every row in the `--ms-test` A/B, and speculative decoding on
  `--slots 4` is token-identical to `--slots 1`. Since §4i the batched attention
  split count follows the number of rows in the step, so which near-ties resolve
  differently can also depend on how a burst assembles; `BRUCE_MS_SPLITS=14` pins
  one split count for every batch size.
* `--adaptive-loops` is a single-slot mode (the loop-1 decision is per step, not
  per row). `--draft` still works in multi-slot mode while a request is the only
  active one, and is skipped while other slots are decoding.
* `/slots` reports one entry per slot, `/health` and `/props` report
  `n_ctx_per_slot`, and `/metrics` adds `bruce_batch_step_ms`,
  `bruce_batch_wait_ms`, `bruce_batch_rows` and `bruce_batch_steps_total`.
* `/v1/models` carries the llama.cpp `meta` block with `meta.n_ctx` set to the
  per-slot window (`ctx / slots`) plus `n_ctx_per_slot`.  Clients such as
  pi-llama-cpp size their prompts from `meta.n_ctx`; reporting the whole pool
  there makes them send prompts the server rejects with "prompt is longer than
  the configured context".

### Endpoints

| method | path | notes |
|---|---|---|
| `GET` | `/` | small HTML status page |
| `GET` | `/health`, `/healthz` | `status`, `n_ctx`, `n_ctx_per_slot`, `slots_idle`, `slots_processing` |
| `GET` | `/v1/models`, `/v1/models/{id}` | OpenAI model list / detail (`id` = `--alias`; `meta.n_ctx` = per-slot window) |
| `POST` | `/v1/chat/completions` | OpenAI chat, streaming and non-streaming; `reasoning_content`, `tool_calls`; `n` up to 8 |
| `POST` | `/v1/completions` | OpenAI completions; `echo` is supported only when `stream=false` |
| `POST` | `/completion` | native llama.cpp completion; string prompt or token-id array |
| `POST` | `/tokenize`, `/detokenize` | token ids, optional pieces (`with_pieces`) |
| `GET` | `/props` | defaults, context size, chat template |
| `GET` | `/slots` | live slot state and counters |
| `GET` | `/metrics` | Prometheus text counters |

Unknown generation options are ignored. The OpenAI `developer` role is treated as `system`.

### Tool calling

`/v1/chat/completions` supports OpenAI-style tools. Tool schemas are rendered into the system
prompt with the Dory GGUF template's XML layout, and the model's `<tool_call>` blocks are parsed
back into OpenAI `tool_calls`:

- `tools`: array of `{"type":"function","function":{...}}` objects. Nested objects, arrays,
  enums, `required`, and extra schema keys are rendered.
- `tool_choice`: `"auto"` (default), `"none"` (tools are not rendered), `"required"` (advisory:
  without grammar-constrained decoding the model may still answer without a call), or a
  specific `{"type":"function","function":{"name":"..."}}` object (only that tool is rendered).
- Multiple `<tool_call>` blocks in one answer become multiple `tool_calls` entries; no special
  `parallel_tool_calls` switch is needed.
- Non-streaming responses set `finish_reason: "tool_calls"` and return
  `message.tool_calls` with `function.arguments` as a JSON string.
- Streaming responses emit `delta.tool_calls` entries with `index`, `id`, `type`,
  `function.name`, and the complete `function.arguments` JSON. The final chunk uses
  `finish_reason: "tool_calls"`.
- Assistant history with `tool_calls` and `role: "tool"` results is rendered back into the prompt,
  so multi-turn tool loops work.
- Parameter values are converted using the tool schema: strings stay strings, while
  integer/number/boolean/object/array parameters are parsed from their JSON representation.
  Multi-line string values are preserved.

There is no grammar-constrained decoding. If the model does not emit a complete, valid
`<tool_call>` block, the response is returned as normal content. This server has been tested with
the `pi` coding agent, including its `developer` system message and 23 built-in tools.

### Chat template options

The Dory GGUF template is rendered by the server itself. These keys can be sent at the top level
or inside `chat_template_kwargs`:

| key | default | effect |
|---|---|---|
| `enable_thinking` | on | if false, the prompt contains the empty `<think></think>` pair and the model answers directly |
| `medium_effort` | off | appends `\n\n{reasoning effort: efficient}` to the last user message |
| `truncate_history_thinking` | on | strips reasoning from assistant turns before the latest user message |
| `reasoning_effort` | server default | `none`/`off` disables thinking; `low`/`minimal`/`efficient` enables medium effort |

When thinking is on, output before `</think>` is returned as `reasoning_content` and the rest as
`content`. If generation stops before the tag appears, everything is reasoning and `content` is
empty. When thinking is off, a stray `</think>` is still removed instead of leaking into content.

### Generation parameters

Accepted by `/v1/chat/completions`, `/v1/completions` and `/completion` (aliases in parentheses):

| key | default | notes |
|---|---|---|
| `max_tokens` (`max_completion_tokens`, `n_predict`, `max_gen_tokens`, `num_predict`) | 256 | `-1` fills the remaining context; clamped to the context |
| `temperature` (`temp`) | 1.0 | `<= 0` selects greedy sampling |
| `top_p` | 0.95 | |
| `top_k` | 40 | |
| `min_p` | 0.0 | |
| `repeat_penalty` (`repetition_penalty`, `repeat-penalty`) | 1.0 | |
| `repeat_last_n` (`repeat_penalty_last_n`, `penalty_last_n`) | 64 | |
| `frequency_penalty`, `presence_penalty` | 0.0 | range `[-2, 2]` |
| `seed` | random | negative means random |
| `stop` | empty | string or array, at most 8 entries |
| `stop_token_ids` | empty | token ids that stop generation |
| `logit_bias` | empty | token id -> bias, clamped to `[-100, 100]` |
| `logprobs`, `top_logprobs` | off | chat: bool + int; completions also accept a number |
| `n` | 1 | 1..8 choices; prompt tokens are counted once in `usage` |
| `cache_prompt` (`cache`) | on | reuse the prompt prefix from the previous request |
| `n_keep` (`keep`) | 0 | accepted for llama.cpp compatibility; currently no effect |
| `ignore_eos` | false | generate past EOS |
| `min_tokens` | 0 | do not stop before this many tokens |
| `draft`, `draft_ngram`, `draft_nmin` | 0 | prompt-lookup speculation |
| `stream` | false | SSE for the OpenAI and native endpoints |
| `stream_options.include_usage` | false | add a final usage chunk to OpenAI streams |
| `echo` | false | prefix the completion with the prompt (non-stream `/v1/completions` only) |

Non-stream chat and completion responses include `usage` and a `timings` object
(`prompt_n`, `prompt_ms`, `prompt_per_token_ms`, `prompt_per_second`, `predicted_n`,
`predicted_ms`, `predicted_per_token_ms`, `predicted_per_second`, `ttft_ms`, `total_ms`).
OpenAI streams add a final usage chunk when `stream_options.include_usage` is set; native
`/completion` streams end with a timings event.
`usage.prompt_tokens_details.cached_tokens` reports the prefix-cache hit. Native `/completion`
also returns `tokens_evaluated`, `tokens_predicted`, `tokens_cached` and `generation_settings`.

### Examples

Non-streaming chat:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"bruce","messages":[{"role":"user","content":"What is 2+2?"}],
       "temperature":0,"max_tokens":64,
       "chat_template_kwargs":{"enable_thinking":false}}'
```

Streaming chat (`reasoning_content` and `content` arrive in separate deltas):

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"bruce","messages":[{"role":"user","content":"Explain prefix caching."}],
       "stream":true,"stream_options":{"include_usage":true}}'
```

Python with the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
stream = client.chat.completions.create(
    model="bruce",
    messages=[{"role": "user", "content": "Say hello in one word."}],
    temperature=0,
    max_tokens=64,
    stream=True,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
for chunk in stream:
    delta = chunk.choices[0].delta
    reasoning = (delta.model_extra or {}).get("reasoning_content")
    if reasoning:
        print(reasoning, end="", flush=True)
    if delta.content:
        print(delta.content, end="", flush=True)
```

Native llama.cpp completion:

```bash
curl -s http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","n_predict":16,"temperature":0}'
```

### Live output

On a terminal the server keeps one self-overwriting status line with current prefill/decode
progress, tokens per second, context usage and ETA. When stderr is redirected it prints a
throttled line at most every 500 ms. `--no-live` disables both.

```text
[bruce] #1 chat | 21 prompt tokens | cache hit 0 (0%) | batch 512
[bruce] #1 prefill 21 processed (0 cached of 21) in 306.2 ms (69 tok/s)
[bruce] #1 decode 64/128 | 58.4 tok/s | ctx 85/4096 | 1.1s | eta 1.1s
[bruce] #1 done | prompt 21 (0 cached) | gen 94/128 | 58.1 tok/s | finish=stop
```

`Ctrl+C` cancels the active generation, stops accepting connections and waits briefly for the
request thread to exit. A client that resets its connection also cancels a streaming request as
soon as the next SSE write fails.

---

## 8. Notes / gotchas found while building this

* `get_tok()` originally read the sampler's output with a blocking `cudaMemcpy` from the
  null stream while the argmax kernel ran on a non-blocking stream.  The read raced the
  kernel and returned the *previous* token, making every generation loop on a single token.
  Fixed by reading through the same stream and synchronising it.
* `--splits > 32` silently overran the split-K part buffer (`PART_MAX_SPLITS` is now 128
  and clamped at startup).
* Debug taps keyed by physical block index capture the *second* visit of a recurrent block
  under 2 loops; they are keyed by virtual index instead.
* A GGUF left memory-mapped while weights are streamed from it costs 19x on cached reads
  (392 MB/s vs 7.4 GB/s): the cache manager serves `ReadFile` on a mapped file at disk
  speed.  The loader therefore drops the mapping after parsing metadata and reads through
  its own sequential handle.  Keeping a mapping for the data and using pageable
  `cudaMemcpy` was the original 17 s load.
* `cudaMemcpyAsync` from pinned memory on WDDM blocks the calling thread until the copy is
  staged, and its event queries take the same driver lock.  One dedicated copy thread keeps
  the reader free, which is what makes the read and the DMA overlap (2.3 s -> 1.2 s warm).
* `--bench` / `--bench-rows` with `--bench-at n_ctx` wrote KV slots past the end of the
  cache: the bench graph advances the position on every replay, and the 4-row path writes
  `rows` slots from `bench_at`.  That only became visible when the tighter loader layout
  placed the overflow in unmapped VRAM; the bench position is now clamped to leave room for
  the warm-up and the timed steps.
* On this card a trivial kernel costs ~1–1.8 us, and the fused normalise+quantise kernels
  are latency-bound rather than bandwidth-bound; that is the remaining ~2 ms of decode.
