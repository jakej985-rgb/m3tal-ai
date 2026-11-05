# Stage 2 run report: base-model pretraining (#22)

Date: 2026-11-05. Hardware: GTX 960 (sm_52, 4 GiB) + CPU, CUDA 12.0,
Go 1.25. Backend: CUDA for training (`M3TAL_LLM_BACKEND=cuda`), CPU for
evaluation.

## Data

- Source: TinyStories-derived text exported from the project's own
  `data/tinystories-ds` train split (CDLA-Sharing-1.0; see
  `data/provenance/tinystories.sources.json`). No external downloads.
- Export: 6 train shards -> 12,000,000 tokens -> 28 MB text -> 10 corpus
  parts (<4 MB each, `LoadCorpus` limits).
- Tokenizer: byte-level BPE vocab 512 trained on a 9 MB corpus sample
  (`fnv64a:4ff3e38a05d53464`).
- Dataset: `/tmp/ts2-ds` — 11 train shards + 1 held-out val shard
  (`/tmp/ts2-val`, 998,522 tokens), built via the Stage 2 pipeline
  (`dataset build` + `dataset split`), `dataset verify` clean
  (CRC32, vocab range, decode spot-checks all pass).
- Val eval text: first 100 KB of the exported val shard
  (`/tmp/ts2-val-eval.txt`) — held out from all training.

## Model

- Config: context 64, width 256, 4 layers, 8 heads, vocab 512
  = **2,895,360 parameters** (the Stage 2 smoke/feasible config from
  docs/STAGE2.md; 127M-class target documented but not feasible here).
- Init: `/tmp/ckpt-s2-init.json`, seed 17, 5 warmup steps.

## Runs

### Run 1 — `/tmp/s2-train-run1.json`

```
train --base-model ckpt-s2-init --dataset ts2-ds --order shuffled \
  --max-steps 1500 --log-every 100 --learning-rate 0.003 \
  --warmup-steps 100 --min-lr-ratio 0.1 --grad-clip 1.0 --seed 17
```

- Backend: CUDA. Sustained ~51 tok/s (~0.8 steps/s at 64-token windows).
- Wall time ~30 min for 1500 steps (~96K tokens seen).
- Loss trajectory (per-100-step means, noisy): 5.98 -> 5.48 -> 5.22 ->
  4.97 -> 4.87 -> 5.09 -> 4.98 -> 4.37 -> 5.15 -> 3.97 -> 4.38 -> 4.01 ->
  3.54 -> 3.95; epoch mean 4.6133.
- Checkpoint saved with full optimizer state (Adam moments + step).

### Run 2 — resume demonstration — `/tmp/ckpt-s2-run2.json`

```
train --base-model ckpt-s2-run1 --dataset ts2-ds --order shuffled \
  --max-steps 1800 --learning-rate 0.001 --seed 17
```

- `--max-steps` is a total-step target: resumed at step 1500 and ran to
  1800 (300 additional steps).
- Loss continued in the trained range (3.47 -> 3.67 -> 3.32; epoch mean
  3.6403) — no restart to ~6.2, confirming weights **and** optimizer
  state resumed correctly.
- Note: `--max-steps` semantics mean "total steps", not "additional";
  documented here for operators.

## Evaluation

- `eval-bench` on the held-out val slice (CPU backend):
  **eval loss 3.5597, perplexity 35.15** over 16,385 tokens
  (`/tmp/s2-eval-report.json`, ~40 min — see limitations).
- For reference, stage0-tiny-stories (149K params, 4000 steps)
  measured 3.646 / 38.3 on the same-style val slice; this 2.9M run at
  1800 steps is already in the same range.
- Generation smoke (`generate`, temp 0.8, seed 7): output is broken but
  story-flavored English — expected for 1800 steps / 2.9M params /
  ~115K tokens; Stage 0's 4000-step run showed the same trajectory.

## Limitations (honest accounting)

- **Scale**: 2.9M params / ~115K tokens is a smoke run, not meaningful
  pretraining. docs/STAGE2.md measures ~51 tok/s at this size; the
  127M-class target would run at ~1 tok/s — infeasible on this hardware.
- **Eval cost**: `Model.Evaluate` does a full forward per token (no KV
  cache). On the CUDA backend each `Forward` call carries host↔device
  transfer overhead, making corpus eval extremely slow (~40 min for a
  100 KB slice was killed). CPU eval-bench with `--max-eval-tokens` is
  the practical path until a KV-cache eval exists.
- **Corpus**: TinyStories is synthetic children's stories — good for
  pipeline validation, weak for general conversational ability.
- **Sharding**: val split is shard-granular; a document may straddle the
  train/val shard boundary (documented in docs/DATA.md).

## Reproduce

All commands above; datasets/checkpoints are local artifacts under
`/tmp` (git-ignored). Pipeline code is in `internal/lm/{provenance,
dataset_builder,dataset_ops}.go`, `cmd/m3tal-lm` dataset/eval-bench
subcommands, docs in `docs/DATA.md`, `docs/EVAL.md`, `docs/STAGE2.md`.
