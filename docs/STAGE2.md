# Stage 2: model configuration and hardware feasibility

## Target: 124M-class configuration

GPT-2-small reference is 124M parameters (context 1024, width 768,
12 layers, 12 heads, vocab 50257, learned position embeddings, GELU MLP).
Our architecture differs — RoPE instead of position tables, RMSNorm,
SwiGLU-style FFN (gate/up/down), vocab ≤ 32768 — so "124M-class" means a
configuration in the same parameter range, verified analytically:

```
m3tal-lm model-info --context 1024 --width 1024 --layers 12 --heads 16 --vocab 512
```

| Component | Parameters |
|---|---|
| Token embedding | 0.52M |
| 12 transformer blocks | 125.96M |
| Output projection | 0.52M |
| **Total** | **127,013,376 (127.0M)** |

`Config.ParameterCount(vocab)` computes this analytically and
`TestConfigParameterCountMatchesModel` verifies it against a constructed
`Model` for multiple configs plus the known stage0-tiny reference
(149,376). `model-info` also prints memory estimates:

- Inference weights (fp32): **0.47 GiB**
- Training state — weights + gradient + Adam first/second moments
  (4× fp32 per parameter): **1.89 GiB**
- Approximate compute: ~0.76 GFLOP per token (forward+backward, 6N rule)

Width validation was raised from 512 to 2048 so this config can be
constructed; context (≤2048), layers (≤16), and heads (≤32, must divide
width) already accommodate it.

## Measured throughput on this hardware

Benchmarks: `train --corpus data/instructions.txt` (self-owned corpus),
context 64, vocab 384, ~75 steps, GTX 960 + CUDA 12.0 / CPU AVX2.

| Config | Params | CPU tok/s | CUDA tok/s |
|---|---|---|---|
| width 128, 2 layers, 4 heads | 0.43M | ~415 | — |
| width 256, 4 layers, 8 heads | 2.83M | ~56 | ~55 |

CUDA training produces bit-identical loss trajectories to CPU (parity
verified again here) but does **not** accelerate these small models —
per-op host↔device transfer overhead dominates. The CUDA GEMM kernels win
on large dense ops (benchmark: ~2.5 ms vs ~287 ms end-to-end) but the
transformer step issues many small ops.

## Feasibility assessment

Throughput scales roughly inversely with parameter count in this regime
(0.43M→415 tok/s, 2.83M→56 tok/s; attention and optimizer overhead grow
superlinearly). Extrapolating:

- **127M-class config: ~1 tok/s or worse** — a 1B-token run would take
  decades. Not feasible on this machine.
- Practical ceiling for short experimental runs: **~3M params**
  (~55 tok/s → ~200K tokens/hour).
- GPU helps inference on larger models but not training at these sizes;
  a real CUDA training path needs fused kernels or larger models where
  GEMMs dominate.

Recommendation: Stage 2 pretraining (#22) runs the largest *feasible*
configuration — width 256 / 4 layers / 8 heads / context 64–128 /
vocab 384 (~2.8M params) — on the TinyStories shard dataset with full
checkpointing, validation holdout, and resume evidence. The 127M-class
config above is the documented target for future hardware; `model-info`
keeps its parameter/memory accounting reproducible.

## Smoke configuration

```
train --corpus data/instructions.txt --width 256 --layers 4 --heads 8 \
  --context 64 --vocab 384 --max-steps 75
```

→ 2,829,696 parameters, ~55 tok/s CPU and CUDA (measured above).
