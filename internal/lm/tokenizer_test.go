package lm

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestTokenizerDeterministicTraining(t *testing.T) {
	corpus := []string{
		"The quick brown fox jumps over the lazy dog.",
		"Café crème and UTF-8 characters: 🤖 日本語.",
		"BPE tokenization learns frequent character and subword pairs deterministically.",
	}
	cfg := TokenizerConfig{
		VocabSize: 320,
		SpecialTokens: map[string]int{
			DefaultEndOfTextToken: 256,
			DefaultPadToken:       257,
			DefaultUnkToken:       258,
		},
	}

	tok1, err := TrainTokenizerWithOptions(corpus, cfg)
	if err != nil {
		t.Fatalf("TrainTokenizerWithOptions 1 failed: %v", err)
	}
	tok2, err := TrainTokenizerWithOptions(corpus, cfg)
	if err != nil {
		t.Fatalf("TrainTokenizerWithOptions 2 failed: %v", err)
	}

	if !reflect.DeepEqual(tok1.Tokens, tok2.Tokens) {
		t.Fatalf("tokens not deterministic: %v vs %v", tok1.Tokens, tok2.Tokens)
	}
	if !reflect.DeepEqual(tok1.Merges, tok2.Merges) {
		t.Fatalf("merges not deterministic: %v vs %v", tok1.Merges, tok2.Merges)
	}
}

func TestTokenizerSpecialAndUnknownTokens(t *testing.T) {
	corpus := []string{"hello world, testing special tokens"}
	tok, err := TrainTokenizerWithOptions(corpus, TokenizerConfig{
		VocabSize: 280,
		SpecialTokens: map[string]int{
			"<|endoftext|>": 256,
			"<|pad|>":       257,
			"<|unk|>":       258,
		},
	})
	if err != nil {
		t.Fatalf("TrainTokenizerWithOptions failed: %v", err)
	}

	// Verify special token IDs
	if id, ok := tok.SpecialTokenID("<|endoftext|>"); !ok || id != 256 {
		t.Fatalf("expected <|endoftext|> id 256, got %d (ok=%v)", id, ok)
	}
	if !tok.IsSpecialToken("<|endoftext|>") {
		t.Fatalf("expected IsSpecialToken to be true")
	}

	// Test encode with special tokens
	input := "hello <|endoftext|> world <|pad|>"
	ids := tok.Encode(input)
	decoded := tok.Decode(ids)
	if decoded != input {
		t.Fatalf("special token round trip mismatch: got %q, want %q", decoded, input)
	}

	// Verify that the special token IDs appear in the encoded output
	hasEOT, hasPad := false, false
	for _, id := range ids {
		if id == 256 {
			hasEOT = true
		}
		if id == 257 {
			hasPad = true
		}
	}
	if !hasEOT || !hasPad {
		t.Fatalf("expected ids to contain 256 and 257, got %v", ids)
	}

	// Unknown token fallback test
	if err := tok.SetUnknownToken("<|unk|>"); err != nil {
		t.Fatalf("SetUnknownToken failed: %v", err)
	}
	if tok.UnknownTokenID() != 258 {
		t.Fatalf("expected unknown token ID 258, got %d", tok.UnknownTokenID())
	}
	if err := tok.SetUnknownToken("<|nonexistent|>"); err == nil {
		t.Fatalf("expected error setting non-existent special token as unknown token")
	}
}

func TestTokenizerArtifactSaveLoadMetadata(t *testing.T) {
	corpus := []string{"save and load tokenizer artifact with version and config metadata"}
	cfg := TokenizerConfig{
		VocabSize: 300,
		SpecialTokens: map[string]int{
			"<|endoftext|>": 256,
		},
	}
	tok, err := TrainTokenizerWithOptions(corpus, cfg)
	if err != nil {
		t.Fatalf("TrainTokenizerWithOptions failed: %v", err)
	}

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "tokenizer.json")

	if err := tok.Save(path); err != nil {
		t.Fatalf("tok.Save failed: %v", err)
	}

	// Verify file contents contains version and metadata
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}
	var artifact TokenizerArtifact
	if err := json.Unmarshal(raw, &artifact); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}
	if artifact.Metadata.Version != TokenizerArtifactVersion {
		t.Fatalf("expected version %s, got %s", TokenizerArtifactVersion, artifact.Metadata.Version)
	}
	if artifact.Metadata.VocabSize != len(tok.Tokens) {
		t.Fatalf("metadata vocab_size mismatch: got %d, want %d", artifact.Metadata.VocabSize, len(tok.Tokens))
	}

	loaded, err := LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer failed: %v", err)
	}

	if !reflect.DeepEqual(tok.Tokens, loaded.Tokens) {
		t.Fatalf("loaded tokens mismatch")
	}
	if !reflect.DeepEqual(tok.Merges, loaded.Merges) {
		t.Fatalf("loaded merges mismatch")
	}
	if !reflect.DeepEqual(tok.SpecialTokens, loaded.SpecialTokens) {
		t.Fatalf("loaded special tokens mismatch: %v vs %v", tok.SpecialTokens, loaded.SpecialTokens)
	}

	// Check round-trip encode/decode on loaded tokenizer
	sample := "testing tokenizer artifact load and decode <|endoftext|>"
	if loaded.Decode(loaded.Encode(sample)) != sample {
		t.Fatalf("encode/decode on loaded tokenizer failed")
	}
}

func TestTokenizerScalesUpTo32k(t *testing.T) {
	// Baseline configurable 32k vocabulary
	var sb strings.Builder
	for i := 0; i < 50; i++ {
		sb.WriteString("the quick brown fox jumps over the lazy dog. ")
	}
	corpus := []string{sb.String()}

	// Verify targetSize up to 32768 is valid and accepted
	tok, err := TrainTokenizer(corpus, 32768)
	if err != nil {
		t.Fatalf("TrainTokenizer(32768) failed: %v", err)
	}
	if len(tok.Tokens) < 256 || len(tok.Tokens) > 32768 {
		t.Fatalf("unexpected token count: %d", len(tok.Tokens))
	}

	// Verify vocabulary size exceeding 32768 returns actionable error
	if _, err := TrainTokenizer(corpus, 32769); err == nil {
		t.Fatalf("expected error for vocab size > 32768")
	}
	// Verify vocabulary size below 256 returns actionable error
	if _, err := TrainTokenizer(corpus, 255); err == nil {
		t.Fatalf("expected error for vocab size < 256")
	}
	// Verify empty corpus returns error
	if _, err := TrainTokenizer(nil, 512); err == nil {
		t.Fatalf("expected error for empty corpus")
	}
}

func BenchmarkTokenizerTrain(b *testing.B) {
	text := strings.Repeat("BPE benchmark tokenizer training performance on multi-word multi-sentence corpus. ", 10)
	corpus := []string{text}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := TrainTokenizer(corpus, 320)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTokenizerEncodeDecode(b *testing.B) {
	text := strings.Repeat("benchmark tokenizer encode and decode performance across various paragraphs. ", 10)
	tok, err := TrainTokenizer([]string{text}, 320)
	if err != nil {
		b.Fatal(err)
	}
	sample := "benchmark tokenizer encode and decode performance"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ids := tok.Encode(sample)
		_ = tok.Decode(ids)
	}
}
