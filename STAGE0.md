# Stage 0 — CPU LLM pipeline: dataset, checkpoints, trainer, TinyStories run

This document records the Stage 0 deliverables (issues #12–#15) and the
TinyStories-scale CPU experiment used to validate them.

## What was built

### Sharded token dataset (#12) — `internal/lm/dataset.go`

- Binary shard format `M3TDS001`: magic, header length, JSON header
  (format version, `u32` dtype, token count, shard index, dataset ID,
  tokenizer fingerprint, creation time), little-endian `uint32` token
  payload, trailing CRC32 (IEEE).
- `dataset.json` manifest records schema/version, dtype, tokenizer
  fingerprint, shard count, and total tokens.
- `BuildDataset` consumes a `TokenFeed` and writes at most one shard in
  memory; `OpenDataset`/`Shard` validate magic, header fields, sizes, and
  CRC before use. Truncated or corrupt shards fail with an error.
- `TokenStream` yields `context+1` token windows with one resident shard.
  `Sequential` order walks shards in index order; `ShuffledSeeded`
  permutes shard visitation with a seeded RNG (sequential I/O within each
  shard). Tests cover sequential access, shuffled access, truncation,
  bit-rot CRC failures, and tokenizer fingerprint mismatch.

CLI: `m3tal-lm dataset build --corpus PATHS --checkpoint CKPT --out DIR
[--shard-tokens N]`.

### Checkpointing and resume (#13) — `internal/lm/model.go`, `safetensors.go`

- JSON checkpoint format v2 adds `step`, `seed`, `tokenizer_hash`, and
  optional Adam `first`/`second` moments alongside weights and tokenizer.
  Older checkpoints still load (zeroed optimizer state).
- `LoadCheckpoint` verifies parameter count, tokenizer hash, and optimizer
  moment lengths; corrupt or incompatible files fail safely.
- `SaveSafetensors`/`LoadSafetensors` implement the safetensors container
  for weights-only export with named F32 tensors and `__metadata__`
  (config, tokenizer hash, parameter count).
- Resume is deterministic: restoring step + optimizer moments continues
  the exact Adam trajectory (verified by `TestCheckpointResumeIsDeterministic`).

### Training loop (#14) — `internal/lm/train.go`

- `TrainStream` consumes any `WindowSource` (`TokenStream` for datasets,
  `SliceSource` for in-memory corpora) and performs causal LM training:
  inputs `window[:context]`, targets `window[1:]`, loss only on valid
  target positions (mask support retained for instruction tuning).
- AdamW-style update with global gradient-norm clipping, linear warmup +
  cosine decay (`MinLRRatio` floor), and per-step metrics: step, loss,
  learning rate, tokens/sec, grad norm, tokens seen.
- CLI `train` gains `--dataset`, `--order`, `--max-steps`, `--log-every`,
  `--warmup-steps`, `--min-lr-ratio`, `--grad-clip`. `--base-model`
  checkpoints resume step and optimizer state.

## TinyStories-scale CPU experiment (#15)

### Corpus

- Source: `roneneldan/TinyStories` on Hugging Face
  (`TinyStories-train.txt` prefix, 60 MB ≈ first 18 split parts;
  `TinyStories-valid.txt`, 19 MB, held out).
- License: CDLA-Sharing-1.0 (share-alike; record attribution when
  redistributing derivatives).
- Preprocessing: corpus split into <4 MB UTF-8 parts for `LoadCorpus`
  limits; `dataset build` BPE-encodes with the checkpoint tokenizer and
  writes `data/tinystories-ds` (git-ignored): **7 shards, 13,613,312
  tokens**, shard size 2M tokens.

### Model and run

- Tokenizer: byte-level BPE, vocab 512, trained on a 7 MB corpus sample.
- Config: context 64, width 64, 2 layers, 4 heads — **149,376 parameters**.
- Seed 17; AdamW-style loop, LR 0.003, warmup 100, cosine floor 0.1,
  grad clip 1.0, shuffled shard order.
- Command:

```
m3tal-lm train --base-model models/stage0-tiny.json \
  --dataset data/tinystories-ds --order shuffled \
  --max-steps 4000 --log-every 200 --warmup-steps 100 \
  --learning-rate 0.003 --checkpoint models/stage0-tiny-stories.json \
  --eval-corpus /tmp/ts-eval-small.txt --seed 17
```

### Results

- Baseline held-out loss (1 optimizer step): **6.2418** (ppl ≈ 514).
- Post-training held-out loss (4,000 steps, ~260K tokens seen):
  **3.2757** (ppl ≈ 26.5) on the same ~150 KB valid-split slice —
  a clear improvement over baseline.
- Training throughput: ~1,310 tokens/sec sustained on this CPU
  (~65-token windows, ~20 steps/sec effective).
- Final mean window loss over the run: 3.55 (per-step losses are noisy;
  see `/tmp/stage0-train.log` for the full log).

### Generated samples (temp 0.8, 120 tokens)

Prompt `Once upon a time` (seed 7):

> Once upon a time.
> Ben were not adress to play with the wrettuat and seased. They felt to
> help their mom. He sayelned her mom. She said, "Lind you have the
> park. Tom is not shove oning!" ...

Prompt `A little girl named Lily` (seed 11):

> A little girl named Lily. You are fun. They anyhe rones tail, but she
> saw a sordy toy. She is glad and friends the badn the noise. ...

Assessment: output is orthographically close to children's-story English
(correct function words, dialogue punctuation, sentence-ish rhythm) but
semantically incoherent — expected for 149K parameters trained on ~260K
tokens. This is a pipeline validation run, not a quality benchmark; no
parity claims are made.

### Resource profile

- Training is single-threaded Go on CPU; memory is bounded to one dataset
  shard (~8 MB) plus model state (<10 MB).
- Dataset build tokenizes ~63 MB of text in ~73 s; shard write is atomic
  (tmp + rename).

## Known limitations

- `SliceSource` windows are non-overlapping; `TokenStream` windows cross
  shard boundaries seamlessly.
- Resume determinism holds for the LR schedule horizon (`MaxSteps`) and
  data order; mid-shard stream position is not checkpointed (resume
  restarts the stream — documented limit).
- `internal/transformer` primitives remain unused by `internal/lm`; the
  inline implementation is authoritative for Stage 0.
