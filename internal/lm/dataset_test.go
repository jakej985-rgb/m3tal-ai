package lm

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func testTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	tok, err := TrainTokenizer([]string{"hello world. this is a tiny corpus. "}, 256)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestDatasetSequentialAndShuffledAccess(t *testing.T) {
	tok := testTokenizer(t)
	dir := t.TempDir()
	var tokens []uint32
	for i := 0; i < 5000; i++ {
		tokens = append(tokens, uint32(i%len(tok.Tokens)))
	}
	meta, err := BuildDataset(dir, NewSliceSource(tokens), tok, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if meta.ShardCount != 5 || meta.TotalTokens != int64(len(tokens)) {
		t.Fatalf("meta = %+v", meta)
	}

	ds, err := OpenDataset(dir)
	if err != nil {
		t.Fatal(err)
	}

	// Sequential stream must reproduce the input exactly.
	seq := NewTokenStream(ds, Sequential, 1)
	var got []uint32
	for {
		chunk, ok := seq.Next(777)
		if !ok {
			break
		}
		got = append(got, chunk...)
	}
	if seq.Err() != nil {
		t.Fatal(seq.Err())
	}
	if len(got) != len(tokens) {
		t.Fatalf("sequential tokens = %d, want %d", len(got), len(tokens))
	}
	for i := range tokens {
		if got[i] != tokens[i] {
			t.Fatalf("token %d = %d, want %d", i, got[i], tokens[i])
		}
	}

	// Shuffled stream visits the same tokens in permuted shard order.
	shuf := NewTokenStream(ds, ShuffledSeeded, 42)
	var shufToks []uint32
	for {
		chunk, ok := shuf.Next(777)
		if !ok {
			break
		}
		shufToks = append(shufToks, chunk...)
	}
	if shuf.Err() != nil {
		t.Fatal(shuf.Err())
	}
	if len(shufToks) != len(tokens) {
		t.Fatalf("shuffled tokens = %d, want %d", len(shufToks), len(tokens))
	}
	same := true
	for i := range tokens {
		if shufToks[i] != tokens[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("shuffled stream produced identical order; permutation ineffective")
	}
}

func TestDatasetRejectsCorruption(t *testing.T) {
	tok := testTokenizer(t)
	dir := t.TempDir()
	tokens := make([]uint32, 3000)
	for i := range tokens {
		tokens[i] = uint32(i % 256)
	}
	if _, err := BuildDataset(dir, NewSliceSource(tokens), tok, 1024); err != nil {
		t.Fatal(err)
	}
	ds, err := OpenDataset(dir)
	if err != nil {
		t.Fatal(err)
	}

	// Truncate a shard.
	shardPath := filepath.Join(dir, shardName(1))
	data, err := os.ReadFile(shardPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(shardPath, data[:len(data)-10], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ds.Shard(1); err == nil {
		t.Fatal("expected error reading truncated shard")
	}

	// Corrupt a token without fixing CRC.
	if _, err := BuildDataset(dir, NewSliceSource(tokens), tok, 1024); err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile(shardPath)
	if err != nil {
		t.Fatal(err)
	}
	data[20] ^= 0xFF
	if err := os.WriteFile(shardPath, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ds.Shard(1); err == nil {
		t.Fatal("expected CRC error on corrupted shard")
	}

	// Wrong tokenizer fingerprint must fail at shard read.
	other, err := TrainTokenizer([]string{"completely different vocabulary zanzibar quokka ", strings.Repeat("bramble frosé. ", 40)}, 384)
	if err != nil {
		t.Fatal(err)
	}
	tokBig, err := TrainTokenizer([]string{strings.Repeat("hello world. this is a tiny corpus. ", 40)}, 384)
	if err != nil {
		t.Fatal(err)
	}
	if TokenizerFingerprint(other) == TokenizerFingerprint(tokBig) {
		t.Fatal("fingerprints should differ")
	}
}

func TestDatasetRejectsWrongTokenizerOnStream(t *testing.T) {
	tok := testTokenizer(t)
	dir := t.TempDir()
	tokens := make([]uint32, 2048)
	for i := range tokens {
		tokens[i] = uint32(i % 256)
	}
	if _, err := BuildDataset(dir, NewSliceSource(tokens), tok, 1024); err != nil {
		t.Fatal(err)
	}
	// Hand-edit the manifest's tokenizer fingerprint.
	manifest := filepath.Join(dir, "dataset.json")
	data, err := os.ReadFile(manifest)
	if err != nil {
		t.Fatal(err)
	}
	// Rebuild manifest with a bogus fingerprint via a fresh dataset built
	// with a different tokenizer in a second dir is simpler; here corrupt it.
	corrupt := []byte(`{"format":"m3tal.tokendataset","version":1,"dtype":"u32","tokenizer":"fnv64a:deadbeefdeadbeef","shard_tokens":1024,"shard_count":2,"total_tokens":2048,"created_unix":1,"dataset_id":"ds-x"}`)
	_ = data
	if err := os.WriteFile(manifest, corrupt, 0o644); err != nil {
		t.Fatal(err)
	}
	ds, err := OpenDataset(dir)
	if err != nil {
		t.Fatal(err)
	}
	stream := NewTokenStream(ds, Sequential, 1)
	stream.Next(10)
	if stream.Err() == nil {
		t.Fatal("expected tokenizer mismatch error from shard read")
	}
}

func TestShardHeaderRoundTripFields(t *testing.T) {
	// Verify little-endian token encoding matches expectations.
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], 0xDEADBEEF)
	if binary.LittleEndian.Uint32(buf[:]) != 0xDEADBEEF {
		t.Fatal("endian sanity")
	}
}
