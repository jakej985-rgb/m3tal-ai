# Performance notes

Measured optimizations and remaining bottlenecks (this machine: CPU +
optional GTX 960).

## Implemented

### Parallel `linear32` (`internal/lm/cpu_backend.go`)

All dense projections (Q/K/V/O, FFN gate/up/down, output) route through
`linear32`, which now splits output columns across up to 8 workers when
the matmul exceeds ~262K MACs (`linearWorkThreshold`). Below the
threshold the serial path is kept — goroutine overhead would dominate
for tiny models and single-token KV steps.

Affects: training forward, generation KV steps, and evaluation forwards.

### Chunked evaluation (`Model.EvaluateFast`, `eval-bench --eval-mode`)

`Evaluate` scores each token with a full sliding-window forward —
O(n·context) work, ~15 tok/s measured on the 2.9M model (4,097 tokens in
4m37s). `EvaluateFast` scores non-overlapping context-sized blocks, one
forward per block — **~35× faster** (same corpus in 7.8s).

Semantics differ slightly: tokens early in each block see only their
in-block prefix, not a full sliding window. Chunked loss 4.6303 vs exact
4.6714 on the same corpus — close but not identical, so **baseline
comparisons must use the same `--eval-mode`** (recorded in reports as
`eval_mode`). `train --eval-corpus` still uses exact `Evaluate`.

`eval-bench` defaults to `--eval-mode fast`; pass `--eval-mode exact`
for sliding-window scoring.

## Measured

| Path | Before | After |
|------|--------|-------|
| eval-bench 4K tokens (2.9M, ctx64) | 4m37s | 7.8s (~35×) |
| generate 40 tok (2.9M SFT ckpt) | 7.7s | 4.2s (~1.8×, incl. load) |
| train 250 steps, 2.9M ctx64 CPU | 59 tok/s | ~330 tok/s (~5.6×) |

The `user` time exceeding `real` ~4× confirms cores are engaged.

## Parallel backward (added)

`bwdLinear32` parallelizes every linear-layer backward: the input-gradient
pass is partitioned by rows and the weight-gradient pass by input dim, so
each gradient element keeps the serial accumulation order (bit-identical
per element; cross-layer summation order changes only where multiple
projections feed one tensor, e.g. Q+K+V). Verified by the finite
difference gradcheck suite and a 250-step training run whose loss
trajectory matches the serial baseline (epoch loss identical: 5.1387).

## Remaining bottlenecks (not yet done)
- **KV cache has no eviction** — long-context eval/generation falls back
  to full forwards past `context`. Sliding-window KV would unlock cheap
  long eval.
- **CUDA heuristic** — `auto` picks CUDA whenever a device exists, but
  small models are faster on CPU (transfer overhead). A size-based
  heuristic (e.g., CUDA only above ~5M params or large GEMMs) would
  avoid regressions; benchmarks in docs/SCALING.md.
- **Tokenifier/encode** is fine at current sizes.

## Reproduce

```sh
go build -o m3tal-lm ./cmd/m3tal-lm
./m3tal-lm eval-bench --checkpoint CKPT --corpus data/instructions.txt \
  --max-eval-tokens 4096 --tokens 1 --eval-mode fast
time ./m3tal-lm generate --checkpoint CKPT --chat --tokens 40
```
