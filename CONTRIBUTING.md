# Contributing

Bruce is a research-grade engine, so the bar for a change is simple: **it must be
measurable and it must not change the tokens.**

## Before you open a PR

1. Build it: `engine\build.cmd`.
2. Run a greedy token-identity check on the 121-token code prompt against the
   reference (`tools/gen_cmp.py`), or explain in the PR why the change is expected to
   alter sampling.
3. Measure it with the built-in harness, not with wall-clock vibes:
   ```bat
   engine\bruce.exe --bench 20 --bench-at 32768
   engine\bruce.exe --bench 20 --bench-at 32768 --bench-rows 4
   ```
   Include before/after numbers and the GPU/clock they were taken at.
4. Check that prefill did not regress when you touch decode, and vice versa.

## Style

- Keep the single-file kernel layout in `engine/src/bruce.cu`; new kernels live next
  to the ones they replace, with a comment that states the constraint they exploit.
- Every fast path needs a fallback selected by a flag or an `auto` rule, so an A/B can
  be rerun later.
- Prefer exactness. If a path is approximate (fp16 attention, batched verify), say so
  in the comment and in the PR, and say which reference path it matches.
- No new dependency unless there is no reasonable way to avoid it. The engine has no
  ggml, no PyTorch, and no BLAS.

## Reporting a benchmark

State the GPU, driver, CUDA version, clock, model file, context depth, batch size,
sampler, and whether speculation/prefix cache were on. The `benchmarks/` harness
records all of this automatically; use it if you can.

## License

By contributing you agree that your changes are licensed under the MIT license in
[`LICENSE`](LICENSE).
