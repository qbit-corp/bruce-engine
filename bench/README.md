# Bruce vs llama.cpp benchmark (Nemotron Dory 1.0 4B, 256K context)

> **Long-context sweep (8K–256K, single slot):** see
> [`../benchmarks/README.md`](../benchmarks/README.md).

Final report: [`report.md`](report.md) · chart: [`benchmark.png`](benchmark.png) ·
medians: [`summary.csv`](summary.csv)

## What is measured

Both engines serve the same GGUF through their HTTP APIs:

- Bruce: `engine/bruce.exe --serve --ctx 262144 --loops 2 --n-batch 512 --slots 1 --draft 0`
- llama.cpp: `llama-server.exe` (PR #30306 / `57b70d0`), `-c 262144 -fa on -b 2048 -ub 1024`,
  q8_0 K/V, 2 loops / `last_loop`

The request is `POST /completion` with a **token-id array**, `n_predict=256`,
`temperature=0`, `cache_prompt=false`, `ignore_eos=true`. Both engines therefore process
exactly the same tokens, generate exactly 256 tokens, and do a full prompt pass every time.

**Text prompt:** 16 long English Wikipedia articles followed by an essay instruction.
**Code prompt:** concatenated llama.cpp C++/CUDA sources followed by a ring-buffer task.

Depths: 4,096 (3 reps), 32,768 (2 reps), 256,000 (1 rep), all in a 262,144-token context.
A 4K warmup request precedes the measurements.

The primary table has speculation off on both engines. The secondary table enables
doryFast's prompt-lookup drafter (`--draft 7`, n=4) and llama.cpp's equivalent ngram drafter
(`--spec-type ngram-simple`, n=4, m=7). A 32K control run gives llama.cpp the same 512-token
compute batch as Bruce (`-ub 512`).

Paths are taken from `%BRUCE_MODEL%` (default `models/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf`)
and `%LLAMA_SERVER%` (default `llama-server.exe` on `PATH`).

## Files

| file | contents |
|---|---|
| `report.md` | full report, tables, samples, caveats |
| `benchmark.png` | prefill/decode summary chart |
| `summary.csv` | median prefill/decode tok/s per engine, prompt, depth |
| `results_*_plain.json` | raw plain-decoding requests |
| `results_*_spec.json` | raw ngram-speculation requests |
| `results_llama_ub512.json` | equal-batch control |
| `logs/` | server logs for every session (generated locally, gitignored) |
| `run_bench.py` | server launcher + request loop |
| `serve_concurrency.py` | multi-slot (`--slots`) concurrency regression: solo vs 1/2/4/8 parallel requests |
| `analyze.py` | medians, speedups, report generator |
| `make_chart.py` | chart generator |
| `make_corpus.py`, `fetch_more_text*.py` | corpus construction |
| `tokenize_corpora.py` | one-time tokenization shared by both engines |

## Reproduce

```bat
python bench\make_corpus.py
python bench\tokenize_corpora.py
python bench\run_bench.py --engine both --tag plain
python bench\run_bench.py --engine both --tag spec --spec
python bench\run_bench.py --engine llama --tag ub512 --depths 32768:2 --ubatch 512
python bench\analyze.py
python bench\make_chart.py
```

The scripts are cross-platform and pick `engine/bruce.exe` or `engine/bruce` based
on the OS. On Linux use forward slashes (`python bench/make_corpus.py`).

The GPU must be free; the harness starts one server at a time, waits for VRAM to be
released, and cools down 90 s between engines. Close other GPU-heavy applications first.
