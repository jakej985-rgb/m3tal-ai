# Ecosystem integration

This repository (`m3tal-llm`) is the **research, training, and serving**
side of the M3TAL local-AI experiments. It is not the assistant itself.

## Repository boundaries

| Repo | Role |
|---|---|
| `m3tal-llm` (this repo) | From-scratch Go autodiff/tensors/NN, byte-level BPE, transformer LM, dataset + provenance + eval pipeline, chat SFT/DPO, `cmd/m3tal-lm` CLI, optional CUDA backend, HTTP inference server (`serve`) and Go client (`internal/llmclient`). |
| `m3tal-plugins/m3tal-ai` | The M3TAL assistant/plugin (module `m3tal-plugin-ai`). Deterministic intent routing, local documentation search, read-only M3TAL Core queries, confirmed Core mutations, and an **optional** language model consumed through a `Generator` interface. It **moved out of this repo** — no assistant source lives here anymore. |
| M3tal-core | The dashboard/plugin host. Owns the live allowlists of stacks, containers, and services that mutations may target. |

`INTENTS.md` and `ROADMAP.md` remain at this repo's root as the canonical
intent/safety-class and product-plan references for the assistant; the
code they describe is maintained in `m3tal-plugins/m3tal-ai`.

## How the pieces connect

- **Optional model, never required.** m3tal-ai works fully without a
  language model: deterministic routing handles clear phrases, and a
  small trained classifier plus retrieval cover the rest.
- **Current state: no shipped wiring.** As of this writing, m3tal-ai's
  `cmd/m3tal-ai` exposes no `--model` flag or `M3TAL_MODEL_PATH` env
  var, and nothing outside its tests calls `Assistant.SetLanguageModel`.
  There is no in-process checkpoint loader — m3tal-ai does not depend on
  this repo and cannot parse its checkpoint format.
- **The intended path is HTTP.** `m3tal-lm serve` exposes `/v1/generate`
  and `/v1/chat`; `internal/llmclient`'s `Generator` adapter matches the
  assistant's `Generator` interface signature exactly, so wiring is a
  few lines in m3tal-ai (see `docs/SERVE.md`):

  ```go
  c := llmclient.New("http://127.0.0.1:8080")
  assistant.SetLanguageModel(llmclient.NewGenerator(c))
  ```

  Until that glue lands in m3tal-ai, the model is consumed only by this
  repo's own CLI/server and by anything else that speaks the HTTP API.
- **Model output is never auto-trusted.** Assistant plans must parse
  into the fixed tool schema; invalid output runs no tools. Every
  mutating plan step is checked against M3TAL Core's **live** allowlist
  and requires the exact per-step confirmation (`confirm <action>
  <target-type> <name>`); the target is re-checked immediately before
  the mutation, and declining a step stops the rest of the plan.
- **Core mutations stay deterministic.** Routing, allowlist checks, and
  confirmation are implemented in m3tal-ai/M3tal-core code — not in the
  model. The model can only *propose*; it cannot execute.
- **Fallback is safe by construction.** When no model is loaded, the
  server is unreachable, or generation fails, the assistant falls back
  to deterministic behavior. Model unavailability is an ordinary error,
  not a crash.

## What this repo is / is not

**Is:** experimental training infrastructure — datasets with provenance,
checkpoints, eval harnesses, a serving API, packaging — plus the honesty
conventions (self-owned/licensed data only, recorded run evidence,
regression-checked evals) that any model handed to the ecosystem must
meet.

**Is not:** the assistant, a plugin host, or an action executor. It
contains no Core mutation logic and no user-facing assistant UI. Model
checkpoints produced here are educational prototypes; plain next-token
training does not make them instruction-following, and their proposals
should be expected to be rejected by the assistant's schema/allowlist/
confirmation gates until suitable instruction-tuned data exists.
