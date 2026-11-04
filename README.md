# m3tal-ai

A lightweight AI model and M3TAL ecosystem assistant built from scratch in Go.
It includes scalar automatic differentiation, tensor operations, neural-network
layers and training, and a local intent model that can answer M3TAL questions
and safely operate M3TAL Core.

Requires Go 1.23 or newer. See [INTENTS.md](INTENTS.md) for the assistant's
available intents and their safety classes. This project is licensed under the
[MIT License](LICENSE).

## Run the XOR demo

```sh
go run ./cmd/xor
```

The scalar math engine lives in `internal/autograd`. Dense matrix operations
live in `internal/tensor`, and reusable layers live in `internal/nn`. The XOR
demo trains a two-layer network with matrix-based forward and backward passes,
using a reusable sequential model and trainer. The trainer supports full or
mini-batches, deterministic shuffling, SGD or Adam, mean squared error, and
softmax cross-entropy. Available activations include tanh and ReLU.

## M3TAL specialist prototype

The local assistant trains a small intent classifier in Go at startup, searches
documentation and source files across M3TAL Core, M3TAL-Hub, M3TAL Plugins, and
projects registered in the Hub manifest. It can query M3TAL Core for stats,
health, services, and container logs. Stack start, stop, and restart requests
ask for an exact confirmation phrase, and start/stop/restart actions are
limited to stack, container, and system-service names returned by Core.
Documentation matches include freshness flags for old
listed dates and undated setup or version details; those flags identify review
needs and do not prove whether a claim is true.

```sh
go run ./cmd/m3tal-ai "how do I install M3TAL?"
go run ./cmd/m3tal-ai "show system stats"
go run ./cmd/m3tal-ai "list running containers"
go run ./cmd/m3tal-ai "show logs for jellyfin"
go run ./cmd/m3tal-ai "restart stack media"
```

The Core API defaults to `http://localhost:5050`. Set `M3TAL_API_URL` and
`M3TAL_API_TOKEN` to override the connection. Documentation and source are
discovered from local M3TAL repositories registered in the Hub manifest. README
files rank below specific docs and source matches. Set `M3TAL_DOCS_PATHS` or pass
`--docs PATH` to add other roots.

## Experimental from-scratch language model

`internal/lm` contains a small pure-Go byte-level BPE tokenizer and one-block
causal self-attention model. It trains from UTF-8 text and saves local JSON
checkpoints; it does not download weights or call a model service. This is an
educational prototype and is not a production-quality general language model.
Training uses explicit backpropagation on the same contiguous `float32` CPU
weights as generation and evaluation. Byte-level sampling rejects malformed
UTF-8. The CPU backend is the intended boundary for a future GPU
implementation; no GPU acceleration is included yet. Use only text
you have the right to train on, and record its source and license yourself.

```sh
go run ./cmd/m3tal-lm train --corpus "$HOME/apps/M3tal-core:$HOME/apps/M3tal-Hub:/path/to/public-domain-text" --eval-corpus /path/to/held-out-text --checkpoint models/m3tal-small.json
go run ./cmd/m3tal-lm generate --checkpoint models/m3tal-small.json --prompt "M3TAL is"
go run ./cmd/m3tal-ai --model models/m3tal-small.json "how do I install M3TAL?"
go run ./cmd/m3tal-ai --interactive --model models/m3tal-small.json
```

`--corpus` accepts a path-list of files or directories containing Markdown or
plain text. A limited amount of text is indexed per run
to keep the prototype bounded. The assistant keeps deterministic Core routing;
the local model may draft documentation answers and propose short multi-step
plans. Model proposals are rejected unless they parse into the fixed tool
schema, targets are present in Core's live allowlist, and each mutation is
confirmed individually. If a plan cannot be parsed, no action runs.
Plain next-token training does not make this tiny model instruction-following;
expect most proposed plans to be rejected until suitable instruction examples
are included in a separately licensed training corpus.
