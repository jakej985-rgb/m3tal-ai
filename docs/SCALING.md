# Scaling study — how far can this hardware go?

Issue: decide whether larger models are practical using **measured**
compute/memory/data, not parameter count alone. All numbers below come
from `model-info` (analytic, verified against constructed models) or
recorded benchmark runs on this machine (GTX 960 sm_52, CUDA 12, ~4-core
CPU class). Nothing here is extrapolated from marketing FLOPs.

## Measured throughput (this machine)

| Model | Config | Backend | Measured train tok/s | Source |
|-------|--------|---------|---------------------|--------|
| 0.15M  | ctx64 w64 L2 H4 v512 | CPU | ~1,300 (tiny windows) | Stage 0 smoke |
| 2.9M   | ctx64 w256 L4 H8 v512 | CUDA | ~51 | docs/RUN-stage2.md |
| 2.9M   | ctx64 w256 L4 H8 v512 | CPU  | ~55 (SFT windows) | Stage 3 SFT log |
| 21.5M  | ctx128 w512 L8 H8 v512 | CPU | **~4** | this study (timed out >300s for 20 steps) |

Key observation: CUDA does **not** help at these sizes — per-op
host↔device transfer overhead dominates (documented in docs/CUDA.md and
RUN-stage2). The GTX 960's value would appear only at GEMM sizes this
project hasn't reached; even then its 4 GB VRAM caps practical training
size near ~500M params in fp32 (15 GB training state at 1B — infeasible).

## Cost model

Training cost ≈ `6·N·T` FLOPs (fwd+bwd), T = tokens. At measured CPU
tok/s, daily token budgets:

| Params | GFLOP/tok | CPU tok/s (est.) | Tokens/day | 1B-token run |
|--------|-----------|------------------|------------|--------------|
| 2.9M   | 0.02      | ~55              | ~4.8M      | ~8.7 h        |
| 21.5M  | 0.13      | ~4               | ~0.35M     | ~119 days     |
| 127M   | 0.76      | ~0.7 (est.)      | ~60K       | ~4.5 years    |
| 1.0B   | 6.06      | ~0.1 (est.)      | ~9K        | ~30 years     |

CPU tok/s scaling is roughly inverse to params at fixed context (measured
2.9M→55, 21.5M→4; superlinear slowdown from cache effects). GPU could
improve the mid-range ~5-20× on large GEMMs, but 127M still needs ~0.76
GFLOP/tok × throughput — a modern GPU might do ~5-20K tok/s; the GTX 960
cannot hold the 1.89 GiB training state plus activations efficiently and
was already slower than CPU at 2.9M.

## Go / no-go thresholds (this hardware)

- **GO (interactive iteration): ≤ ~10M params.** Trains in minutes-hours;
  CPU and CUDA both viable; checkpoint <100 MB.
- **GO (overnight experiments): ~20-30M params.** ~4 tok/s CPU → a
  meaningful 1-5M-token run takes days; use only with a fixed budget.
- **NO-GO: ≥ ~100M params on this machine.** Even a modest 100M-token run
  is months; the 127M-class config documented in Stage 2 remains a
  reference config, not a runnable one.
- **Data side:** local-only policy means corpus size is the binding
  constraint anyway — 12M tokens exists today; a 127M model would need
  ~10-100× more tokens to be worth training, which we cannot source
  locally.

## Recommendations

1. Keep iterating at ≤10M params; scale **data quality** (more self-owned
   instruction/chat data) before model size — returns are larger.
2. If a bigger model is ever needed, the path is a modern CUDA GPU
   (≥16 GB VRAM) + licensed corpus, not this GTX 960.
3. Preserve config/data provenance for every run (done via dataset
   manifests + experiment conventions in docs/EXPERIMENTS.md).
4. Re-run this study if hardware or the local corpus changes materially.
