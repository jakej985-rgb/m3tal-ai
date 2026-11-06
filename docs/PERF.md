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
| train 250 steps, 2.9M ctx64 CPU | 59 tok/s | ~380 tok/s (~6.4×) |

The `user` time exceeding `real` ~4× confirms cores are engaged.

## Parallel backward (added)

`bwdLinear32` parallelizes every linear-layer backward: the input-gradient
pass is partitioned by rows and the weight-gradient pass by input dim, so
each gradient element keeps the serial accumulation order (bit-identical
per element; cross-layer summation order changes only where multiple
projections feed one tensor, e.g. Q+K+V). Verified by the finite
difference gradcheck suite and a 250-step training run whose loss
trajectory matches the serial baseline (epoch loss identical: 5.1387).

Attention forward/backward is also parallelized by head via `parallelFor`
(disjoint per-head slices, no reduction needed), and RoPE theta tables are
hoisted out of the row loop. Remaining serial work is small:
`rmsNormBackward`, the SwiGLU activation backward, `gradLogits`, and
embedding-gradient scatter (row ids can collide, so it stays serial).

KV-step attention is also head-parallel via `parallelFor`, engaging only
when `cachedTokens*headDim` is large enough — short generations stay on
the serial path.

## Remaining bottlenecks (not yet done)
- **KV cache has no eviction** — long-context eval/generation falls back
  to full forwards past `context`. Sliding-window KV would unlock cheap
  long eval.
- **CUDA auto-selection** — measured KV-cache generation (60 tokens,
  warm, GTX 960) shows CUDA faster than CPU at every tested size, so
  `auto` always prefers CUDA when a device is present; no size heuristic
  is applied (`internal/lm/backend_select_cuda.go`):

  | Params | CPU tok/s | CUDA tok/s |
  |--------|-----------|------------|
  | 0.15M  | 415       | 1316       |
  | 2.9M   | 100       | 498        |
  | 8.4M   | 41        | 277        |
  | 21.5M  | 15.9      | 138        |
  | 48M    | 7.9       | 81.8       |

  Note this contradicts the earlier assumption that small models are
  faster on CPU — that observation was about *training* throughput,
  where the backward pass is CPU-side regardless of the inference
  backend. Explicit `M3TAL_LLM_BACKEND=cpu` still forces CPU.
- **Tokenifier/encode** is fine at current sizes.

## Reproduce

```sh
go build -o m3tal-lm ./cmd/m3tal-lm
./m3tal-lm eval-bench --checkpoint CKPT --corpus data/instructions.txt \
  --max-eval-tokens 4096 --tokens 1 --eval-mode fast
time ./m3tal-lm generate --checkpoint CKPT --chat --tokens 40
```
