# Documentation index

`m3tal-llm` is a from-scratch Go AI/LLM research repository: scalar
autodiff, tensors, NN layers, a byte-level BPE tokenizer, a causal
transformer LM, dataset/provenance/eval tooling, chat fine-tuning
(SFT/DPO), an optional CUDA backend, and the `cmd/m3tal-lm` CLI. The
M3TAL assistant (m3tal-ai) lives in a different repository —
`m3tal-plugins/m3tal-ai` — and consumes models from here optionally;
see `ECOSYSTEM_INTEGRATION.md`.

## Reading order

1. **Architecture** — what the code is and how it fits together.
   - [`../STAGE0.md`](../STAGE0.md) — the original CPU pipeline: dataset, checkpoints,
     trainer, first TinyStories run.
   - `STAGE2.md` — model configuration math, parameter counts, memory
     estimates, hardware feasibility for larger configs.
   - `STAGE3.md` — chat template (`m3tal.chatml.v1`), SFT data, decoding
     controls, chat-eval, DPO.
   - `RUN-stage2.md` — a recorded pretraining run with exact commands,
     throughput, and honest limitations.
   - `EXPERIMENTS.md` — reproducibility conventions: what every run
     records and how to reproduce Stage 2/3 runs.
   - `SCALING.md` — measured throughput/cost model and go/no-go size
     thresholds for this hardware.
2. **Datasets** — building, provenance, verification.
   - `DATA.md` — licensed corpus sources, `dataset build|report|verify|
     split|annotate|export`, shard format, provenance manifest, chat
     JSONL validation/splitting.
3. **Training** — covered by the architecture docs above plus
   `EXPERIMENTS.md` (train flags live in `cmd/m3tal-lm`; `--dataset`,
   `--sft`, `--dpo`, `--max-steps` semantics).
4. **Evaluation** — `EVAL.md` (`eval-bench`, `--eval-mode`, baseline
   regression) and the chat-eval section of `STAGE3.md`.
5. **Serving** — `SERVE.md` (`m3tal-lm serve` HTTP API, safety policy,
   `internal/llmclient` Go client), `PACKAGING.md` (`package`,
   `model-card`, hardware requirements), `CUDA.md` (optional GPU
   backend), `PERF.md` (measured optimizations and remaining
   bottlenecks).
6. **Ecosystem** — `ECOSYSTEM_INTEGRATION.md`: relationship to
   m3tal-ai, M3tal-core, and the safety boundary. Assistant intent and
   safety-class details: `../INTENTS.md` and `../ROADMAP.md` (kept in
   the repo root; the assistant code itself is in m3tal-plugins/m3tal-ai).

## Quick reference

| Doc | Covers |
|---|---|
| [`../STAGE0.md`](../STAGE0.md) | First end-to-end CPU run |
| `STAGE2.md` | Config/parameter feasibility study |
| `STAGE3.md` | Chat template, SFT, DPO, chat-eval |
| `RUN-stage2.md` | Recorded Stage 2 training run |
| `DATA.md` | Dataset build/verify/provenance pipeline |
| `EVAL.md` | `eval-bench` repeatable evaluation |
| `SERVE.md` | HTTP inference server + Go client |
| `PACKAGING.md` | Model cards, bundles, hardware needs |
| `CUDA.md` | Optional CUDA backend (build tag `cuda`) |
| `PERF.md` | Performance measurements/bottlenecks |
| `SCALING.md` | Scaling limits on this hardware |
| `EXPERIMENTS.md` | Reproducibility conventions |
| `ECOSYSTEM_INTEGRATION.md` | m3tal-ai / M3tal-core relationship |
