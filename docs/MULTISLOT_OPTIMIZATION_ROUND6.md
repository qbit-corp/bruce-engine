# Multi-slot serving pass — round 6 (2026-10-29)

Scope: `--slots N` decode batching and the server's prompt scheduler.  The
single-slot decode graph and every prefill kernel are untouched; §4i of the README
has the same numbers and is the canonical summary.

## Target card

```text
Hardware:        RTX 3060 12 GB, 28 SMs, measured ~345 GB/s, 50.2 T-MAC/s int8
Model:           Dory 1.0 4B Q5_K_M, 112 virtual blocks, 2 loops, q8_0 KV
Regimes:         decode M=1 (slots=1) | verify 2..8 | prefill >8 | multi-slot MS 2..8
Target:          raise multi-slot aggregate without moving slots=1 numbers
Quality class:   batched near-tie tolerance (documented); slots=1 token-identical
Hard invariants: slots=1 decode/prefill within noise, 128/128 code-prompt identity
```

## Floors (8 rows, 2K tokens per slot)

```text
weights/token            4.20 GB (shared by all rows in the step)
weight floor             4.20 / 345 = 12.2 ms
KV full (14 layers, loop-1 reuse on 5)  51.3 MB/row
KV SWA (42 layers x 1024)               93.6 MB/row
KV floor, 8 rows         1.16 GB / 345 = 3.4 ms
logits + elementwise     ~2.5 ms (logits 1.1, combine/gate 1.1, norms 0.9)
combined floor           ~18.1 ms
measured wall step       18.9 ms (graph + argmax + host sync)
```

The multi-slot step was already close to the bandwidth floor; the wins came from
deleting redundant partial traffic, coalescing the verify weight stream, and not
serialising prompts behind a 256-row chunk.

## Changes (ranked by measured effect)

| change | mechanism | measured |
|---|---|---|
| prefill burst scheduler | queued bursts use 512-row chunks; a lone prompt behind decoders uses 384; solo 512 | queued prefill 897 → 993 tok/s at 8 slots |
| batched attention splits scale with rows | `ceil(56/rows)` full / `ceil(28/rows)` SWA instead of 56/28 per row; grid stays ~one wave, combine partials drop 4-8x | 8-row step 21.92 → 20.40 ms eager |
| staged Q5_K verify tiles | 16x176 B tile per warp, 16 B/lane `cp.async`, double buffered, shared read-back; K=2560 only | −1.5 % at 8 rows, −2 % at 4 rows |

The 384-row chunk is 6.6 % faster than 256 at 4K (2401 → 2560 tok/s) and keeps a
chunk near the documented ~150 ms decode gap at 8K.

## Scorecard (probe-ms, 8 rows, 2K/slot, eager launches)

| phase | ms/pass | note |
|---|---:|---|
| qkv GEMM | 4.07 | fused q/k/g/v, verify kernel |
| ffn gate+up | 4.48 | fused, verify kernel |
| ffn down | 3.93 | q5/q6 verify |
| attention | 3.58 | per-row M=1 warp kernel, near the 3.4 ms floor |
| wo | 1.79 | verify kernel |
| combine+gate | 1.06 | fused register kernel |
| logits | 1.06 | Q8_0 head, shared across rows |
| silu / resid / prep / norms | 1.43 | per-row, launch-light |
| **total** | **20.28** | graph replay 18.9 ms |

## End-to-end

`bench/serve_concurrency.py`, ctx 32768 (4096/slot), 512-token prompt, 256 generated:

| slots | rows/step | step before → now | aggregate before → now |
|---:|---:|---:|---:|
| 1 | 1.0 | 14.4 → 14.3 ms | 67.2 → 67.2 |
| 2 | 2.0 | 16.4 → 15.7 ms | 111.5 → 117.1 |
| 4 | 3.9 | 17.7 → 16.5 ms | 190.9 → 206.0 |
| 8 | 7.8 | 19.3 → 17.7 ms | 306.4 → 341.1 |

2048-token prompts, 128 generated: 67.0 → 73.3, 89.1 → 100.1, 107.8 → 124.9 tok/s
for 2/4/8 slots.

## Slots=1 gates

* decode 4K 69.47 tok/s / 14.395 ms, 32K 62.97 / 15.881 ms (pre-change 69.35 /
  62.89).
* prefill 4K/8K/32K 2817 / 2696 / 2218 tok/s (pre-change 2803 / 2667 / 2213).
* 121-token code prompt, plain greedy and `--draft 7`: **128/128 identical** to the
  pre-change build; the staged verify kernel is byte-identical to the direct one
  (64/64 tokens).
* `--ms-test` at 4 and 8 slots: every row argmax matches the per-slot M=1 path and
  the attention A/B has 0/8 argmax mismatches.

## Excluded / measured negative

| idea | measured result | why rejected |
|---|---|---|
| fixed split count 7 for all batch sizes | 2-row step 19.14 vs 18.21 ms | underfills the grid at small rows; kept as `BRUCE_MS_SPLITS=N` for reproducibility |
| fixed split count 14 for all batch sizes | 8-row step 20.99 vs 20.63 ms | ~2 % slower at full load; row-scaled stays default |
| 384-row burst chunk | 8-slot wall 9.19 s vs 8.32 s for 512 | 512 drains queued bursts faster; 384 kept only for the lone-prompt-behind-decoders case |
| K=7168 staged verify tiles | launch fails at 117 KB | two tiles (45 KB) do not fit beside the K=7168 activation stage; direct loop kept |
| 2 CTAs/SM for verify GEMM | prior round: 15 % lower bandwidth | L2 thrash; 28 one-CTA blocks stay |

## Switches

| switch | meaning |
|---|---|
| `BRUCE_MS_SPLITS=0\|1\|N` | 0 = old M=1 count, 1 = scale with rows (default), N = fixed |
| `BRUCE_VERIFY_STAGE=0` | register-direct verify loop instead of staged tiles |
| `BRUCE_SERVE_CHUNK_BURST=N` | queued-burst prompt chunk (default 512) |
| `BRUCE_SERVE_CHUNK=N` | lone-prompt-behind-decoders chunk (default 384) |
| `--probe-ms` / `--bench-ms` | dev phase attribution / direct step timing |
