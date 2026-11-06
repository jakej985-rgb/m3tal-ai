# Corpus ingestion, provenance, and datasets

Stage 2 data pipeline. Everything is built from sources already present in
this repository — no external downloads. Every dataset records where its
text came from, under which license, and which processing was applied.

## Sources and license review

Corpus sources are declared in JSON files (examples in
`data/provenance/`):

| Source | Kind | License | Review |
|---|---|---|---|
| `tinystories` | tokendataset (`data/tinystories-ds`) | CDLA-Sharing-1.0 | TinyStories (Eldan & Li, 2023); share-alike — attribution required in model cards |
| `m3tal-instructions` | text (`data/instructions.txt`) | self-owned | Curated in-repo examples; no third-party text |

`license` and `license_review` are **required** — `dataset build` and
`dataset annotate` refuse sources that omit them. A source is never
treated as reusable merely because it is publicly accessible.

Source spec fields: `name`, `kind` (`text` or `tokendataset`), `paths`,
`license`, `license_review`, optional `url`, `retrieved_unix`, `version`,
`filters`, `dedup`. At build time each source is hashed (SHA-256 over the
raw file bytes in sorted-path order) and the digest is recorded in
`provenance.json` inside the dataset directory.

## Building a dataset

```sh
m3tal-lm dataset build \
  --sources data/provenance/instructions.sources.json \
  --checkpoint models/stage0-tiny.json \
  --out data/my-ds --val-frac 0.05 --dedup exact --shard-tokens 1000000
```

- `--sources FILE` — licensed source list (preferred). `--corpus PATHS`
  still works for local experiments but records the source as
  `unreviewed-local` with an explicit warning in the manifest.
- `--val-frac F` — document-level validation split: each document is
  assigned to `<out>-val` by a deterministic SHA-256 hash, so the same
  document can never appear in both splits. Requires text sources.
- `--dedup exact` — SHA-256 per-document dedup across all sources;
  duplicate and kept counts are recorded in the manifest.
- `--resume` — continue an interrupted build. Shards are written
  atomically and `dataset.json` is updated after every shard, so a crash
  leaves a consistent partial dataset. Resume requires identical
  sources/flags; already-written tokens are skipped to realign shard
  boundaries. Appending to a finalized dataset without `--resume` is an
  error.
- `--shard-tokens N` — tokens per shard (default 1M).

`kind: "tokendataset"` sources merge an existing shard dataset into the
output (re-packed under the new manifest; tokenizer fingerprint must
match). Use `dataset split` for shard-level holdouts instead.

## Inspecting and verifying

```sh
m3tal-lm dataset report [--json] DIR      # manifest + per-shard headers + provenance
m3tal-lm dataset verify [--checkpoint CKPT] [--samples N] DIR
m3tal-lm dataset split --val-shards K --val-out DIR SRCDIR
m3tal-lm dataset annotate --sources FILE [--split DESC] DIR   # provenance for pre-existing datasets
```

`verify` checks every shard's magic, header, size, and CRC32; with
`--checkpoint` it additionally confirms all token IDs are within vocabulary
and decodes sample windows to measure the printable-character ratio (a
corruption spot check).

`split` moves trailing shards into a separate validation dataset,
rewriting shard headers for their new indices. The split is at shard
granularity — neighbouring shards may share a document boundary; for
text corpora prefer `--val-frac` at build time.

## Shard format and packing

Shards are `M3TDS001` binary files: JSON header (format version, dtype
`u32`, token count, shard index, dataset ID, tokenizer fingerprint) +
raw uint32 LE tokens + CRC32. Token streams are packed contiguously —
training windows slice `context+1` tokens directly from the stream, so
sequence packing is implicit and lossless. The trainer consumes shards
via `TokenStream` (sequential or seeded-shuffled shard order), one shard
resident in memory at a time.

## Provenance manifest

`provenance.json` records: pipeline name + git hash, creation time,
manifest ID, per-source license/review/URL/version/filters/dedup/SHA-256,
and ingest stats (documents in/kept/duplicate, train/val token counts).
`dataset annotate` backfills manifests for datasets built before this
pipeline existed (used for `data/tinystories-ds`).

## Custom self-owned chat data

`data/chat-custom.jsonl` is a starter set of self-owned chat examples
(ops Q&A, general QA, math, translation, multi-turn, formatting,
tech explanations, how-tos) for SFT experiments. It is licensed
self-owned — edit, extend, or replace freely. Validate and split with:

```sh
./m3tal-lm chatdata validate data/chat-custom.jsonl
./m3tal-lm chatdata split --in data/chat-custom.jsonl --out-dir /tmp/custom-split
./m3tal-lm train --base-model CKPT --sft /tmp/custom-split/train.jsonl --checkpoint out.json
```
