# Evaluation benchmark

`m3tal-lm eval-bench` is the repeatable evaluation harness (Stage 2).
It produces a JSON report and optionally fails on regression vs a
baseline report.

```sh
m3tal-lm eval-bench \
  --checkpoint models/<run>.json \
  --corpus /path/to/held-out.txt \
  --tokens 60 --temperature 0.8 --seed 19 \
  --report reports/eval-<run>.json \
  --baseline reports/eval-<prev>.json --max-loss-increase 0.05
```

## What it measures

- **Held-out loss / perplexity** — corpus loss capped by
  `--max-eval-tokens` (default 65,536). Two modes via `--eval-mode`
  (recorded in reports as `eval_mode`): `fast` (default) uses
  `Model.EvaluateFast`, which scores non-overlapping context-sized
  blocks in one forward each (~35× faster than exact but slightly
  different semantics — early-block tokens see only their in-block
  prefix); `exact` uses `Model.Evaluate`, a full sliding-window forward
  per token. The token cap exists because exact evaluation is one
  forward pass per token without a KV cache; larger corpora need
  proportionally more time. Truncation is deterministic (first N
  tokens of the corpus in file order). Baseline comparisons must use
  the same `--eval-mode` — see `docs/PERF.md`.
- **Fixed generations** — a built-in prompt set (or `--prompts FILE`,
  one prompt per line, `#` comments) decoded with fixed
  `--tokens`/`--temperature`/`--seed`. Each prompt gets `seed+i` so the
  set is reproducible run-to-run.
- **Backend** — records `kernel.Selected()` so CPU vs CUDA results are
  attributable.

## Baseline and regression

`--baseline REPORT.json` compares `eval_loss` against a previous
`eval-bench` report; the command exits non-zero when the increase exceeds
`--max-loss-increase` (default 0.05 nats). Reports are plain JSON —
checkpoint, backend, corpus path, eval token count, per-prompt
prompt/output pairs, and timestamp — so they can be diffed and archived.

## Held-out corpus provenance

The evaluation corpus must be held out from training. Recommended flow:

1. Build the training dataset with `dataset build --val-frac 0.05`
   (document-level split) or `dataset split` for shard-level holdouts —
   see `docs/DATA.md`.
2. Materialize the held-out text with
   `dataset export --out eval.txt --checkpoint CKPT VAL_DIR`.
3. Record the corpus path and dataset provenance (`provenance.json`)
   alongside the eval report.

Do not evaluate on training corpora — losses will be optimistic and
regression detection meaningless.

## Known limitation

`--eval-mode exact` (`Model.Evaluate`) does a full forward pass per
token (no KV cache), so eval throughput is roughly training-forward
speed; `--eval-mode fast` (`Model.EvaluateFast`) is the default and is
far cheaper but not bit-identical to exact loss. A KV-cache evaluation
path is a candidate future optimization; until then keep
`--max-eval-tokens` modest and consistent across runs being compared.
