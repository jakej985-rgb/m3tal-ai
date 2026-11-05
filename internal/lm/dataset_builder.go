package lm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// This file adds a resumable dataset builder on top of the shard format in
// dataset.go (issue: Stage 2 tokenization/packing). A builder owns one
// output directory, appends tokens into fixed-size shards, and rewrites
// dataset.json after every flushed shard so an interrupted build leaves a
// consistent, openable partial dataset that can be resumed.

// DatasetBuilder incrementally writes shards for one dataset directory.
// It is safe to create, feed tokens, close, and re-open for resume.
type DatasetBuilder struct {
	dir       string
	meta      *DatasetMeta
	buf       []uint32
	finalized bool
}

// NewDatasetBuilder opens dir for writing. If a manifest already exists
// the builder resumes it: the caller must skip meta.TotalTokens tokens of
// the feed (PendingSkip reports how many) so shard boundaries realign.
// Resume requires the same tokenizer fingerprint and shard size.
//
// If the existing dataset is finalized and resume is false, an error is
// returned so a completed dataset is never silently appended to.
func NewDatasetBuilder(dir string, tokenizer *Tokenizer, shardTokens int, resume bool) (*DatasetBuilder, error) {
	if shardTokens <= 0 || shardTokens > maxShardTokens {
		return nil, fmt.Errorf("shardTokens must be in (0, %d]", maxShardTokens)
	}
	fp := TokenizerFingerprint(tokenizer)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	manifestPath := filepath.Join(dir, "dataset.json")
	if _, err := os.Stat(manifestPath); err == nil {
		d, err := OpenDataset(dir)
		if err != nil {
			return nil, fmt.Errorf("existing dataset manifest is invalid: %w", err)
		}
		if d.Meta.Tokenizer != fp {
			return nil, fmt.Errorf("existing dataset tokenizer %q does not match %q", d.Meta.Tokenizer, fp)
		}
		if d.Meta.ShardTokens != shardTokens {
			return nil, fmt.Errorf("existing dataset shard_tokens %d does not match %d", d.Meta.ShardTokens, shardTokens)
		}
		if d.Meta.Finalized && !resume {
			return nil, fmt.Errorf("dataset %s is finalized; pass --resume to append or choose a new --out", dir)
		}
		meta := d.Meta
		return &DatasetBuilder{dir: dir, meta: &meta, buf: make([]uint32, 0, shardTokens)}, nil
	}
	meta := &DatasetMeta{
		Format:      "m3tal.tokendataset",
		Version:     datasetFormatVer,
		DType:       tokenDType,
		Tokenizer:   fp,
		ShardTokens: shardTokens,
		CreatedUnix: time.Now().Unix(),
		DatasetID:   fmt.Sprintf("ds-%d", time.Now().UnixNano()),
	}
	return &DatasetBuilder{dir: dir, meta: meta, buf: make([]uint32, 0, shardTokens)}, nil
}

// PendingSkip returns how many feed tokens must be discarded before new
// tokens are appended, so resumed builds realign with existing shards.
func (b *DatasetBuilder) PendingSkip() int64 { return b.meta.TotalTokens }

// Meta returns the current (possibly partial) manifest.
func (b *DatasetBuilder) Meta() *DatasetMeta { return b.meta }

// SkipFeed discards n tokens from feed; used to realign on resume.
func SkipFeed(feed TokenFeed, n int64) error {
	var skipped int64
	for n > 0 {
		want := n
		if want > 1<<22 {
			want = 1 << 22
		}
		chunk, ok := feed.Next(int(want))
		if !ok {
			if err := feed.Err(); err != nil {
				return err
			}
			return fmt.Errorf("feed ended after %d skipped tokens, need %d", skipped, n)
		}
		skipped += int64(len(chunk))
		n -= int64(len(chunk))
	}
	return feed.Err()
}

// Consume reads tokens from feed until exhaustion, flushing full shards.
// It may be called multiple times to append several sources.
func (b *DatasetBuilder) Consume(feed TokenFeed) error {
	if b.finalized {
		return fmt.Errorf("dataset builder for %s is finalized", b.dir)
	}
	shardTokens := b.meta.ShardTokens
	for {
		chunk, ok := feed.Next(shardTokens - len(b.buf))
		if !ok {
			break
		}
		b.buf = append(b.buf, chunk...)
		if len(b.buf) == shardTokens {
			if err := b.flush(); err != nil {
				return err
			}
		}
	}
	if err := feed.Err(); err != nil {
		return fmt.Errorf("token feed: %w", err)
	}
	return nil
}

func (b *DatasetBuilder) flush() error {
	if len(b.buf) == 0 {
		return nil
	}
	if err := writeShard(filepath.Join(b.dir, shardName(b.meta.ShardCount)), b.buf, b.meta, b.meta.ShardCount); err != nil {
		return fmt.Errorf("write shard %d: %w", b.meta.ShardCount, err)
	}
	b.meta.TotalTokens += int64(len(b.buf))
	b.meta.ShardCount++
	b.buf = b.buf[:0]
	return b.writeManifest()
}

// Close flushes any partial shard and marks the dataset finalized.
func (b *DatasetBuilder) Close() (*DatasetMeta, error) {
	if err := b.flush(); err != nil {
		return nil, err
	}
	b.meta.Finalized = true
	if err := b.writeManifest(); err != nil {
		return nil, err
	}
	b.finalized = true
	return b.meta, nil
}

// Abort discards the in-memory partial shard without writing the
// manifest; already-flushed shards remain resumable.
func (b *DatasetBuilder) Abort() { b.buf = b.buf[:0] }

func (b *DatasetBuilder) writeManifest() error {
	data, err := json.MarshalIndent(b.meta, "", "  ")
	if err != nil {
		return err
	}
	tmp := filepath.Join(b.dir, "dataset.json.tmp")
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(b.dir, "dataset.json"))
}
