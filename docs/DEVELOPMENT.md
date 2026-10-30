# Development notes

Everything here is optional for *running* Bruce. It covers the reference checkout,
the validation loop, and how to reproduce the benchmarks.

## Reference llama.cpp checkout

Bruce's math is pinned to the Dory implementation in
[`brianchmielNV/llama.cpp`](https://github.com/brianchmielNV/llama.cpp). The checkout is
not vendored (it is ~170 MB and has its own history), so clone it next to the engine:

```bat
git clone https://github.com/brianchmielNV/llama.cpp tools\doryref
git -C tools\doryref checkout 98a74c9
```

Build it with CUDA enabled as usual (CMake + `-DGGML_CUDA=ON`, or `build-cuda` in the
LLVM/MSVC toolchain). The head-to-head benchmark numbers in the README were produced
with `llama-server.exe` build 1, commit `57b70d0`; the local `tools/doryref` tree at
`98a74c9` is the newer metadata/oracle snapshot and produces the same tokens for the
Dory graph.

Tools that expect the reference checkout look at `%LLAMA_CPP%` and fall back to
`tools/doryref`.

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `BRUCE_GGUF` | engine | default model path when `--model` is omitted |
| `BRUCE_MODEL` | benchmark scripts | model path (default `models/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf`) |
| `LLAMA_SERVER` | benchmark scripts | path to `llama-server.exe` (default: on `PATH`) |
| `LLAMA_CPP` | corpus builder | llama.cpp source tree for the code corpus |

(Use `$BRUCE_GGUF` etc. on Linux; the `%VAR%` spelling above is Windows.)

## Validation loop

The engine is checked against the reference, not against itself. From weakest to
strongest:

1. **Same tokens, greedy.** Generate with `--gen 128 --greedy` on a fixed token-id
   prompt and compare ids with the reference server (`tools/gen_cmp.py`).
2. **Same logits.** `tools/oracle_dump.exe` (build it with `tools/build_oracle.cmd`) writes
   per-block activations and the full logit vector batch-of-one; `tools/pos_cmp.py`,
   `tools/cmp_logits.py`, and `tools/validate.py` compare argmax / top-5 / top-50 and max |Δlogit|.
3. **Same activations.** `--dump-x`, `--dump-taps`, and `--debug-block` dump the
   residual stream at chosen blocks; `tools/step_cmp.py` and `tools/cmp_dump.py`
   diff them against the oracle.
4. **Kernel-level.** `tools/mmq_test.cu`, `tools/q6test.cu`, and the CPU reference in
   `tools/ref_np.py` check individual GEMMs (Q5_K max |diff| ~7e-5, Q6_K ~1e-6).
5. **Regression gates.** Plain single-token decode must stay bit-identical to the
   pre-optimisation engine; prompt-prefill must keep the same argmax and top-5 5/5.

Some fast paths are intentionally approximate (fp16 flash attention, batched verify
GEMMs), matching what llama.cpp does with `-fa on`. Near-ties can therefore flip; the
engineering log records which path is exact and why.

## Reproducing the benchmarks

Long-context sweep (8K–256K, chart + report + CSV):

```bat
python benchmarks\run_context_bench.py --engine both --fresh
python benchmarks\analyze_context.py
python benchmarks\make_context_chart.py
```

or `benchmarks\run_all.cmd` for all three steps (`./benchmarks/run_all.sh` on Linux).

The older 4K/32K/256K sweep with speculation and multi-slot tests:

```bat
python bench\make_corpus.py
python bench\tokenize_corpora.py
python bench\run_bench.py --engine both --tag plain
python bench\run_bench.py --engine both --tag spec --spec
python bench\analyze.py
python bench\make_chart.py
```

Notes:

- One server process per context point: no prefix cache or warm KV carries over.
- The GPU must be idle; the harness waits for VRAM to be released and cools down
  between engines.
- Generated corpora and token-id arrays are gitignored — `make_corpus.py` and
  `tokenize_corpora.py` rebuild them deterministically.
- `raw/results_*.json` in both directories are the evidence behind the published
  tables; server `logs/` are generated locally by the harness and are gitignored.

## Code map

| File | Responsibility |
|---|---|
| `engine/src/bruce.cu` | GGUF loader, all CUDA kernels, samplers, CUDA graphs, CLI, model orchestration |
| `engine/src/server.h` | OpenAI-compatible HTTP server, slots, prefix cache, batching |
| `engine/src/session.h` | per-sequence state and KV partitioning |
| `engine/src/chat.h` | Dory chat template, reasoning/tool-call handling |
| `engine/src/tokenizer.h` | BPE tokenizer |
| `engine/src/gguf.h` | zero-copy GGUF reader |
| `engine/src/http.h` | minimal HTTP/SSE server |
| `engine/src/json.h` | JSON parser/writer |

The single-file design is deliberate: kernels can be inlined and tuned together, and
there is no ABI boundary between the loader, the graph builder, and the server.

## Profiling

`--probe`, `--occ`, and `--bench` are built-in:

```bat
engine\bruce.exe --bench 20 --bench-at 32768 --probe
engine\bruce.exe --occ
```

For Nsight Systems traces, run the engine under `nsys profile` and inspect the token
loop; `tools/final_suite.py` bundles the A/B and equivalence checks used after each
optimization round.
