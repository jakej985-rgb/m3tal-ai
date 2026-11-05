package lm

import (
	"os"
	"path/filepath"
	"testing"
)

func seqTokens(n int, vocab int) []uint32 {
	out := make([]uint32, n)
	for i := range out {
		out[i] = uint32(i % vocab)
	}
	return out
}

func TestCorpusSourceValidation(t *testing.T) {
	good := CorpusSource{
		Name:          "s",
		Paths:         []string{"/tmp"},
		License:       "MIT",
		LicenseReview: "reviewed",
	}
	if err := good.Validate(); err != nil {
		t.Fatalf("valid source rejected: %v", err)
	}
	for _, mutate := range []func(*CorpusSource){
		func(s *CorpusSource) { s.License = "" },
		func(s *CorpusSource) { s.LicenseReview = "" },
		func(s *CorpusSource) { s.Name = "" },
		func(s *CorpusSource) { s.Paths = nil },
		func(s *CorpusSource) { s.Kind = "bogus" },
	} {
		bad := good
		mutate(&bad)
		if err := bad.Validate(); err == nil {
			t.Fatalf("invalid source %+v accepted", bad)
		}
	}
}

func TestHashSourceDeterministic(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("hello corpus"), 0o644); err != nil {
		t.Fatal(err)
	}
	s := &CorpusSource{
		Name:          "s",
		Kind:          "text",
		Paths:         []string{dir},
		License:       "MIT",
		LicenseReview: "r",
	}
	if err := HashSource(s); err != nil {
		t.Fatal(err)
	}
	first := s.SHA256
	if len(first) != 64 || s.Bytes == 0 {
		t.Fatalf("bad hash result: %q bytes %d", first, s.Bytes)
	}
	s2 := *s
	if err := HashSource(&s2); err != nil {
		t.Fatal(err)
	}
	if s2.SHA256 != first {
		t.Fatal("hash not deterministic")
	}
}

func TestDatasetBuilderResume(t *testing.T) {
	tok := testTokenizer(t)
	dir := t.TempDir()
	all := seqTokens(10000, len(tok.Tokens))

	// Simulate an interrupted build: consume 2 full shards + partial,
	// write manifest via flush but never Close (unfinalized).
	b, err := NewDatasetBuilder(dir, tok, 1024, false)
	if err != nil {
		t.Fatal(err)
	}
	// Feed only enough for ~2.5 shards by wrapping a limited source.
	limited := NewSliceSource(all[:2560])
	if err := b.Consume(limited); err != nil {
		t.Fatal(err)
	}
	// Force two shard flushes by consuming exactly 2048 more via a second feed.
	rest := NewSliceSource(all[2560:])
	if err := b.Consume(rest); err != nil {
		t.Fatal(err)
	}
	// Do NOT Close: leave build "crashed" with flushed shards + manifest.

	meta := b.Meta()
	if meta.Finalized {
		t.Fatal("builder should not be finalized before Close")
	}
	wantShards := meta.ShardCount
	wantTokens := meta.TotalTokens
	if wantShards == 0 {
		t.Fatal("expected flushed shards before resume")
	}

	// Resume: new builder must skip already-written tokens.
	b2, err := NewDatasetBuilder(dir, tok, 1024, true)
	if err != nil {
		t.Fatal(err)
	}
	if b2.PendingSkip() != wantTokens {
		t.Fatalf("PendingSkip = %d, want %d", b2.PendingSkip(), wantTokens)
	}
	feed := NewSliceSource(all)
	if err := SkipFeed(feed, b2.PendingSkip()); err != nil {
		t.Fatal(err)
	}
	if err := b2.Consume(feed); err != nil {
		t.Fatal(err)
	}
	final, err := b2.Close()
	if err != nil {
		t.Fatal(err)
	}
	if !final.Finalized {
		t.Fatal("final meta not finalized")
	}
	if final.TotalTokens != int64(len(all)) {
		t.Fatalf("TotalTokens = %d, want %d (resume duplicated or lost tokens)", final.TotalTokens, len(all))
	}

	// The resumed dataset must reproduce the original token stream.
	ds, err := OpenDataset(dir)
	if err != nil {
		t.Fatal(err)
	}
	stream := NewTokenStream(ds, Sequential, 0)
	var got []uint32
	for {
		chunk, ok := stream.Next(4096)
		if !ok {
			break
		}
		got = append(got, chunk...)
	}
	if len(got) != len(all) {
		t.Fatalf("streamed %d tokens, want %d", len(got), len(all))
	}
	for i := range all {
		if got[i] != all[i] {
			t.Fatalf("token %d: got %d want %d", i, got[i], all[i])
		}
	}
}

func TestDatasetBuilderRejectsFinalizedAppend(t *testing.T) {
	tok := testTokenizer(t)
	dir := t.TempDir()
	b, err := NewDatasetBuilder(dir, tok, 512, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Consume(NewSliceSource(seqTokens(600, len(tok.Tokens)))); err != nil {
		t.Fatal(err)
	}
	if _, err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := NewDatasetBuilder(dir, tok, 512, false); err == nil {
		t.Fatal("expected error appending to finalized dataset without --resume")
	}
	if _, err := NewDatasetBuilder(dir, tok, 512, true); err != nil {
		t.Fatalf("resume of finalized dataset should be allowed: %v", err)
	}
}

func TestSplitDataset(t *testing.T) {
	tok := testTokenizer(t)
	dir := t.TempDir()
	valDir := filepath.Join(t.TempDir(), "val")
	all := seqTokens(5000, len(tok.Tokens))
	if _, err := BuildDataset(dir, NewSliceSource(all), tok, 1000); err != nil {
		t.Fatal(err)
	}
	res, err := SplitDataset(dir, valDir, 2)
	if err != nil {
		t.Fatal(err)
	}
	if res.TrainShard != 3 || res.ValShard != 2 {
		t.Fatalf("split = %+v", res)
	}
	train, err := OpenDataset(dir)
	if err != nil {
		t.Fatal(err)
	}
	val, err := OpenDataset(valDir)
	if err != nil {
		t.Fatal(err)
	}
	if train.Meta.ShardCount != 3 || val.Meta.ShardCount != 2 {
		t.Fatalf("train %d val %d shards", train.Meta.ShardCount, val.Meta.ShardCount)
	}
	if train.Meta.TotalTokens+val.Meta.TotalTokens != int64(len(all)) {
		t.Fatal("token totals do not add up")
	}
	// Val shards must validate (renumbered headers) and keep order.
	var valTokens []uint32
	for i := 0; i < val.Meta.ShardCount; i++ {
		s, err := val.Shard(i)
		if err != nil {
			t.Fatal(err)
		}
		valTokens = append(valTokens, s.Tokens...)
	}
	// Val holds the LAST tokens of the original stream.
	want := all[len(all)-len(valTokens):]
	for i := range want {
		if valTokens[i] != want[i] {
			t.Fatalf("val token %d mismatch", i)
		}
	}
	// Provenance is not required for split, but manifests must agree on id.
	if train.Meta.DatasetID != val.Meta.DatasetID {
		t.Fatal("split halves must share dataset ID")
	}
}

func TestVerifyDatasetDetectsCorruption(t *testing.T) {
	tok := testTokenizer(t)
	dir := t.TempDir()
	if _, err := BuildDataset(dir, NewSliceSource(seqTokens(3000, len(tok.Tokens))), tok, 1000); err != nil {
		t.Fatal(err)
	}
	rep, err := VerifyDataset(dir, tok, 10)
	if err != nil {
		t.Fatal(err)
	}
	if rep.ShardsOK != 3 || rep.BadTokens != 0 || rep.Samples == 0 {
		t.Fatalf("clean verify = %+v", rep)
	}
	// Corrupt one byte inside shard-000001 payload.
	path := filepath.Join(dir, "shard-000001.bin")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)/2] ^= 0xFF
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyDataset(dir, tok, 0); err == nil {
		t.Fatal("verify should fail on corrupted shard")
	}
}

func TestProvenanceRoundTrip(t *testing.T) {
	dir := t.TempDir()
	sources := []CorpusSource{{
		Name: "s", Kind: "text", Paths: []string{"/tmp"},
		License: "MIT", LicenseReview: "r", Dedup: "exact",
	}}
	m := NewProvenance(sources, "abc123", "none", IngestStats{DocumentsIn: 2, DocumentsKept: 2, TokensTrain: 10})
	if err := WriteProvenance(dir, m); err != nil {
		t.Fatal(err)
	}
	got, err := ReadProvenance(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got.ManifestID != m.ManifestID || got.Sources[0].License != "MIT" || got.IngestStats.TokensTrain != 10 {
		t.Fatalf("round trip = %+v", got)
	}
}
