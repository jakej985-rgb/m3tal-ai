# m3tal-llm

A from-scratch AI research repository in Go: scalar automatic differentiation,
tensor operations, neural-network layers and training, and a small experimental
language model. The M3TAL local assistant that used to live here has moved to
the M3TAL Plugins repository at `m3tal-plugins/m3tal-ai` (module
`m3tal-plugin-ai`).

Requires Go 1.23 or newer. This project is licensed under the
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

## Experimental from-scratch language model

`internal/lm` contains a small pure-Go byte-level BPE tokenizer and a causal
transformer with RMSNorm and RoPE position encoding. It trains from UTF-8 text
and saves local JSON checkpoints; it does not download weights or call a model
service. This is an educational prototype and is not a production-quality
general language model. Training uses explicit backpropagation on the same
contiguous `float32` CPU weights as generation and evaluation. Byte-level
sampling rejects malformed UTF-8. An optional CUDA backend (build tag `cuda`)
can accelerate tensor kernels on NVIDIA GPUs; see `docs/CUDA.md`. Use only
text you have the right to train on, and record its source and license
yourself.

```sh
go run ./cmd/m3tal-lm train --corpus "$HOME/apps/M3tal-core:$HOME/apps/M3tal-Hub:/path/to/public-domain-text" --eval-corpus /path/to/held-out-text --checkpoint models/m3tal-small.json
go run ./cmd/m3tal-lm generate --checkpoint models/m3tal-small.json --prompt "M3TAL is"
```

`--corpus` accepts a path-list of files or directories containing Markdown or
plain text. A limited amount of text is indexed per run to keep the prototype
bounded. `go run ./cmd/regen-instructions` regenerates `data/instructions.txt`
from the built-in instruction set.

The assistant in `m3tal-plugins/m3tal-ai` keeps deterministic Core routing and
accepts an optional language model through its `Generator` interface. Model
proposals are rejected unless they parse into the fixed tool schema, targets
are present in Core's live allowlist, and each mutation is confirmed
individually. Plain next-token training does not make this tiny model
instruction-following; expect most proposed plans to be rejected until
suitable instruction examples are included in a separately licensed training
corpus.
