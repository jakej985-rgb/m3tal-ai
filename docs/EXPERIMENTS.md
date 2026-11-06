# Experiments and reproducibility conventions

This repository keeps ordinary CI fast and cheap: `go build`, `go vet`,
`gofmt`, `go test ./...`, staticcheck, and govulncheck run on every push.
**Expensive training, evaluation, and benchmark runs are never part of CI** —
they are run explicitly as experiments and their results are recorded in
`docs/`.

## Conventions

- Every experiment records: model `Config` (context/width/layers/heads),
  tokenizer fingerprint or vocab size, dataset provenance (see
  `docs/DATA.md`), seed, learning-rate schedule (`TrainConfig`), backend
  (`cpu`/`cuda`/`auto` via `M3TAL_LLM_BACKEND`), step count, and the
  checkpoint path.
- Run reports live in `docs/RUN-*.md` (e.g. `docs/RUN-stage2.md`,
  `docs/STAGE3.md`) and include exact commands, measured throughput, and
  honest limitations.
- Checkpoints are JSON and self-describing (config, tokenizer, step,
  optimizer state). Treat published/eval'd checkpoints as immutable —
  always write new checkpoints rather than overwriting.
- Eval comparisons use `eval-bench --baseline` and `chat-eval --baseline`
  so regressions are machine-checked, not eyeballed.

## Reproducing the Stage 2/3 runs

```sh
# Stage 2 base pretraining (see docs/RUN-stage2.md for the full recipe)
go run ./cmd/m3tal-lm dataset build ...   # provenance-tracked shards
go run ./cmd/m3tal-lm train --dataset DIR --order shuffled \
  --max-steps 1500 --checkpoint ckpt.json

# Stage 3 SFT + eval (see docs/STAGE3.md)
go run ./cmd/m3tal-lm instructions --chat data/chat-instructions.jsonl
go run ./cmd/m3tal-lm chatdata split --in data/chat-instructions.jsonl \
  --out-dir /tmp/chat-split --val-frac 0.15
go run ./cmd/m3tal-lm train --base-model CKPT --sft /tmp/chat-split/train.jsonl \
  --checkpoint sft.json --max-steps <base-step+N>
go run ./cmd/m3tal-lm chat-eval --checkpoint sft.json \
  --suite data/chat-eval-suite.jsonl --report eval.json
```

Note: `--max-steps` is an absolute step target (resumed runs continue the
counter), not "additional steps".

## CPU vs GPU workflow

- Default build has **no CUDA dependency**: `go build ./...` works
  everywhere. CUDA is opt-in via `-tags cuda` + `CGO_ENABLED=1`; see
  `docs/CUDA.md`.
- `M3TAL_LLM_BACKEND=cpu|cuda|auto` selects the backend at runtime; CUDA
  falls back to CPU on any device error. Tiny models are often faster on
  CPU due to per-op transfer overhead — benchmark before assuming GPU wins.
- GPU parity is enforced by tests (`internal/lm/cuda_backend_test.go`,
  `internal/nn/cuda_parity_test.go`, `internal/tensor/kernel/cuda_parity_test.go`)
  which skip cleanly without a device, so CI stays GPU-free.
