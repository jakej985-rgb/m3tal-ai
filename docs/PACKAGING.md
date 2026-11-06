# Model packaging

`m3tal-lm package` produces a reproducible, checksummed artifact bundle
for a checkpoint: the checkpoint JSON, a generated model card, a manifest,
and a SHA-256 checksum file.

```sh
go run ./cmd/m3tal-lm package --checkpoint models/stage0-tiny-stories.json \
  --out-dir dist --name stage0-tiny-stories \
  --data-notes "self-owned tinystories + instruction corpus"
```

Output in `dist/`:

- `<name>.tar.gz` — checkpoint + `<name>-MODEL-CARD.md` + `<name>-manifest.json`
- `<name>.tar.gz.sha256` — checksum for the archive
- `<name>-MODEL-CARD.md`, `<name>-manifest.json` — loose copies

The manifest records config, parameter count, vocab size, tokenizer
fingerprint, training step, and the checkpoint's SHA-256 so artifacts
can be verified after transfer:

```sh
sha256sum -c dist/<name>.tar.gz.sha256
```

## Model cards

`m3tal-lm model-card --checkpoint CKPT --out CARD.md [--chat]
[--data-notes "..."]` renders a Markdown card covering intended use,
limitations, training-data provenance, evaluation results, and hardware
notes (`internal/lm/modelcard.go`). Cards are generated artifacts —
regenerate them when eval results or provenance notes change.

## Hardware requirements

| Checkpoint class | Params  | Inference RAM (fp32) | Training RAM | Notes                          |
|------------------|---------|----------------------|--------------|--------------------------------|
| stage0-tiny      | 0.15M   | <0.01 GiB            | <0.1 GiB     | any CPU                        |
| smoke (ctx64)    | 2.9M    | ~0.01 GiB            | ~0.04 GiB    | CPU interactive; CUDA optional |
| mid (ctx128)     | 21.5M   | ~0.08 GiB            | ~0.32 GiB    | CPU slow (~4 tok/s measured)   |
| 127M-class       | 127M    | ~0.47 GiB            | ~1.89 GiB    | needs GPU or patience          |
| ~1B-class        | 1.0B    | ~3.76 GiB            | ~15 GiB      | workstation GPU recommended    |

Memory figures are fp32 weights (+grad+2 Adam moments for training);
see `model-info` for exact per-config accounting. There is no model
quantization yet — fp32 only.

## Reproducibility

Checkpoints are self-describing JSON (config, tokenizer, step, optimizer
state). Pin the checkpoint SHA-256 (in the manifest) when reproducing
results; see `docs/EXPERIMENTS.md` for run conventions.
