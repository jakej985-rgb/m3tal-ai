# m3tal-ai

A small AI model built from scratch in Go. The first milestone is a scalar
automatic-differentiation engine and a tiny neural network that learns XOR.

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
ask for confirmation. Documentation matches include freshness flags for old
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
