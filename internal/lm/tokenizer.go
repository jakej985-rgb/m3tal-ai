// Package lm implements a small language model and tokenizer in pure Go.
//
// The tokenizer implements byte-level Byte-Pair Encoding (BPE) with support
// for configurable vocabulary sizes from 256 up to 32,768 (32k baseline),
// deterministic training, special token handling (<|endoftext|>, <|pad|>, <|unk|>),
// and versioned serialization with metadata.
//
// Vocabulary sizes can be adjusted based on the training corpus and hardware
// constraints (e.g. smaller vocabularies such as 256–2048 for tiny toy datasets
// or specialized domains, and up to 32,768 for general conversational pretraining).
package lm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	// MinVocabSize is the minimum allowable vocabulary size (256 raw bytes).
	MinVocabSize = 256
	// MaxVocabSize is the maximum allowable vocabulary size (32k baseline).
	MaxVocabSize = 32768
	// TokenizerArtifactVersion is the current version string for saved tokenizer artifacts.
	TokenizerArtifactVersion = "1.0.0"

	// Default special token string constants.
	DefaultEndOfTextToken = "<|endoftext|>"
	DefaultPadToken       = "<|pad|>"
	DefaultUnkToken       = "<|unk|>"
)

// Tokenizer is a byte-level BPE tokenizer. IDs 0..255 represent raw bytes,
// followed by optional special tokens, followed by learned merges appended in training order.
type Tokenizer struct {
	Tokens        []string       `json:"tokens"`
	Merges        []Merge        `json:"merges"`
	SpecialTokens map[string]int `json:"special_tokens,omitempty"`
	unkTokenID    int
	index         map[string]int
}

// Merge records one ordered byte-pair merge.
type Merge struct {
	Left  string `json:"left"`
	Right string `json:"right"`
	Token string `json:"token"`
}

// TokenizerConfig configures tokenizer training and special token assignments.
// Target vocabulary size can be adjusted based on the target corpus size.
type TokenizerConfig struct {
	VocabSize     int            `json:"vocab_size"`
	SpecialTokens map[string]int `json:"special_tokens,omitempty"`
	UnknownToken  string         `json:"unknown_token,omitempty"`
}

// TokenizerMetadata stores versioning and configuration metadata with saved tokenizer artifacts.
type TokenizerMetadata struct {
	Version   string          `json:"version"`
	VocabSize int             `json:"vocab_size"`
	Config    TokenizerConfig `json:"config"`
}

// TokenizerArtifact is the complete serialized representation of a trained tokenizer.
// Tokens and merges use base64-encoded byte slices for robust byte-level serialization.
type TokenizerArtifact struct {
	Metadata      TokenizerMetadata `json:"metadata"`
	Tokens        [][]byte          `json:"tokens"`
	Merges        []artifactMerge   `json:"merges"`
	SpecialTokens map[string]int    `json:"special_tokens,omitempty"`
}

type artifactMerge struct {
	Left  []byte `json:"left"`
	Right []byte `json:"right"`
	Token []byte `json:"token"`
}

// TrainTokenizer learns at most targetSize tokens from the supplied corpus.
// It supports configurable vocabulary sizes up to MaxVocabSize (32,768).
func TrainTokenizer(corpus []string, targetSize int) (*Tokenizer, error) {
	return TrainTokenizerWithOptions(corpus, TokenizerConfig{
		VocabSize: targetSize,
	})
}

// TrainTokenizerWithOptions trains a byte-level BPE tokenizer with specific configuration options,
// including special tokens and configurable target vocabulary sizes up to 32k.
func TrainTokenizerWithOptions(corpus []string, cfg TokenizerConfig) (*Tokenizer, error) {
	if len(corpus) == 0 {
		return nil, fmt.Errorf("tokenizer corpus is empty")
	}
	targetSize := cfg.VocabSize
	if targetSize < MinVocabSize {
		return nil, fmt.Errorf("tokenizer target size must be at least %d", MinVocabSize)
	}
	if targetSize > MaxVocabSize {
		return nil, fmt.Errorf("tokenizer target size must not exceed %d", MaxVocabSize)
	}

	t := &Tokenizer{
		Tokens:        make([]string, 0, targetSize),
		SpecialTokens: make(map[string]int),
		unkTokenID:    -1,
	}

	known := make(map[string]bool, targetSize)

	// 1. Initial 256 byte tokens: 0..255
	for i := 0; i < 256; i++ {
		bTok := string([]byte{byte(i)})
		t.Tokens = append(t.Tokens, bTok)
		known[bTok] = true
	}

	// 2. Register special tokens if provided
	if len(cfg.SpecialTokens) > 0 {
		// Sort special tokens for deterministic assignment
		stNames := make([]string, 0, len(cfg.SpecialTokens))
		for st := range cfg.SpecialTokens {
			stNames = append(stNames, st)
		}
		sort.Strings(stNames)
		for _, st := range stNames {
			if !known[st] {
				id := len(t.Tokens)
				t.Tokens = append(t.Tokens, st)
				known[st] = true
				t.SpecialTokens[st] = id
			} else {
				t.SpecialTokens[st] = t.indexOf(st)
			}
		}
	}

	if cfg.UnknownToken != "" {
		if id, ok := t.SpecialTokens[cfg.UnknownToken]; ok {
			t.unkTokenID = id
		}
	}

	// 3. Prepare sequences for BPE merge training.
	// Sequences are split so special tokens do not get merged with regular text.
	sequences := make([][]string, 0, len(corpus))
	for _, text := range corpus {
		subTexts := t.splitSpecialTokens(text)
		for _, chunk := range subTexts {
			if !chunk.isSpecial && len(chunk.text) > 0 {
				sequences = append(sequences, byteSymbols(chunk.text))
			}
		}
	}

	// 4. Learn BPE merges up to targetSize
	for len(t.Tokens) < targetSize {
		counts := make(map[pair]int)
		for _, seq := range sequences {
			for i := 0; i+1 < len(seq); i++ {
				candidate := pair{seq[i], seq[i+1]}
				if !known[candidate.left+candidate.right] {
					counts[candidate]++
				}
			}
		}
		best := pair{}
		keys := make([]pair, 0, len(counts))
		for candidate, count := range counts {
			if count > 1 {
				keys = append(keys, candidate)
			}
		}
		if len(keys) == 0 {
			break
		}
		sort.Slice(keys, func(i, j int) bool {
			if keys[i].left == keys[j].left {
				return keys[i].right < keys[j].right
			}
			return keys[i].left < keys[j].left
		})
		// The sorted tie-break is stable and independent of map iteration order.
		maxCount := 0
		for _, candidate := range keys {
			if counts[candidate] > maxCount {
				maxCount = counts[candidate]
			}
		}
		for _, candidate := range keys {
			if counts[candidate] == maxCount {
				best = candidate
				break
			}
		}
		merged := best.left + best.right
		known[merged] = true
		t.Merges = append(t.Merges, Merge{Left: best.left, Right: best.right, Token: merged})
		t.Tokens = append(t.Tokens, merged)
		for i, seq := range sequences {
			sequences[i] = applyMerge(seq, best.left, best.right, merged)
		}
	}

	t.reindex()
	return t, nil
}

type pair struct{ left, right string }

func byteSymbols(text string) []string {
	out := make([]string, len([]byte(text)))
	for i, b := range []byte(text) {
		out[i] = string([]byte{b})
	}
	return out
}

func applyMerge(seq []string, left, right, merged string) []string {
	out := make([]string, 0, len(seq))
	for i := 0; i < len(seq); {
		if i+1 < len(seq) && seq[i] == left && seq[i+1] == right {
			out = append(out, merged)
			i += 2
		} else {
			out = append(out, seq[i])
			i++
		}
	}
	return out
}

func (t *Tokenizer) indexOf(tok string) int {
	if t.index != nil {
		if id, ok := t.index[tok]; ok {
			return id
		}
	}
	for id, s := range t.Tokens {
		if s == tok {
			return id
		}
	}
	return -1
}

func (t *Tokenizer) reindex() {
	t.index = make(map[string]int, len(t.Tokens))
	for id, token := range t.Tokens {
		t.index[token] = id
	}
}

// SpecialTokenID returns the token ID associated with a special token, if present.
func (t *Tokenizer) SpecialTokenID(token string) (int, bool) {
	if t.SpecialTokens == nil {
		return 0, false
	}
	id, ok := t.SpecialTokens[token]
	return id, ok
}

// IsSpecialToken returns whether the given token string is registered as a special token.
func (t *Tokenizer) IsSpecialToken(token string) bool {
	if t.SpecialTokens == nil {
		return false
	}
	_, ok := t.SpecialTokens[token]
	return ok
}

// SetUnknownToken designates a special token as the fallback for unknown or out-of-vocab characters.
func (t *Tokenizer) SetUnknownToken(token string) error {
	id, ok := t.SpecialTokenID(token)
	if !ok {
		return fmt.Errorf("token %q is not a registered special token", token)
	}
	t.unkTokenID = id
	return nil
}

// UnknownTokenID returns the ID of the designated unknown token, or -1 if not set.
func (t *Tokenizer) UnknownTokenID() int {
	return t.unkTokenID
}

type textChunk struct {
	text      string
	isSpecial bool
	id        int
}

func (t *Tokenizer) splitSpecialTokens(text string) []textChunk {
	if len(t.SpecialTokens) == 0 {
		return []textChunk{{text: text, isSpecial: false}}
	}
	var res []textChunk
	curr := text
	for len(curr) > 0 {
		bestPos := -1
		bestLen := 0
		bestTok := ""
		bestID := -1
		for st, id := range t.SpecialTokens {
			pos := strings.Index(curr, st)
			if pos != -1 {
				if bestPos == -1 || pos < bestPos || (pos == bestPos && len(st) > bestLen) {
					bestPos = pos
					bestLen = len(st)
					bestTok = st
					bestID = id
				}
			}
		}
		if bestPos == -1 {
			res = append(res, textChunk{text: curr, isSpecial: false})
			break
		}
		if bestPos > 0 {
			res = append(res, textChunk{text: curr[:bestPos], isSpecial: false})
		}
		res = append(res, textChunk{text: bestTok, isSpecial: true, id: bestID})
		curr = curr[bestPos+bestLen:]
	}
	return res
}

// Encode turns text into token IDs by replaying learned BPE merges,
// preserving any registered special tokens.
func (t *Tokenizer) Encode(text string) []int {
	if t.index == nil {
		t.reindex()
	}

	chunks := t.splitSpecialTokens(text)
	var ids []int

	for _, chunk := range chunks {
		if chunk.isSpecial {
			ids = append(ids, chunk.id)
			continue
		}

		seq := byteSymbols(chunk.text)
		for _, merge := range t.Merges {
			seq = applyMerge(seq, merge.Left, merge.Right, merge.Token)
		}
		for _, token := range seq {
			if id, ok := t.index[token]; ok {
				ids = append(ids, id)
			} else if t.unkTokenID >= 0 {
				ids = append(ids, t.unkTokenID)
			}
		}
	}
	return ids
}

// Decode concatenates token bytes, omitting invalid IDs.
func (t *Tokenizer) Decode(ids []int) string {
	var out strings.Builder
	for _, id := range ids {
		if id >= 0 && id < len(t.Tokens) {
			out.WriteString(t.Tokens[id])
		}
	}
	return out.String()
}

// Save serializes the tokenizer, merges, special tokens, and metadata to a JSON artifact.
func (t *Tokenizer) Save(path string) error {
	tokensBytes := make([][]byte, len(t.Tokens))
	for i, tok := range t.Tokens {
		tokensBytes[i] = []byte(tok)
	}
	mergesBytes := make([]artifactMerge, len(t.Merges))
	for i, m := range t.Merges {
		mergesBytes[i] = artifactMerge{
			Left:  []byte(m.Left),
			Right: []byte(m.Right),
			Token: []byte(m.Token),
		}
	}

	artifact := TokenizerArtifact{
		Metadata: TokenizerMetadata{
			Version:   TokenizerArtifactVersion,
			VocabSize: len(t.Tokens),
			Config: TokenizerConfig{
				VocabSize:     len(t.Tokens),
				SpecialTokens: t.SpecialTokens,
			},
		},
		Tokens:        tokensBytes,
		Merges:        mergesBytes,
		SpecialTokens: t.SpecialTokens,
	}
	data, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		return fmt.Errorf("encode tokenizer artifact: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// LoadTokenizer restores a Tokenizer from a saved JSON artifact.
func LoadTokenizer(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var artifact TokenizerArtifact
	if err := json.Unmarshal(data, &artifact); err != nil {
		return nil, fmt.Errorf("decode tokenizer artifact: %w", err)
	}

	toks := make([]string, len(artifact.Tokens))
	for i, b := range artifact.Tokens {
		toks[i] = string(b)
	}
	merges := make([]Merge, len(artifact.Merges))
	for i, m := range artifact.Merges {
		merges[i] = Merge{
			Left:  string(m.Left),
			Right: string(m.Right),
			Token: string(m.Token),
		}
	}

	tok := &Tokenizer{
		Tokens:        toks,
		Merges:        merges,
		SpecialTokens: artifact.SpecialTokens,
		unkTokenID:    -1,
	}
	if tok.SpecialTokens == nil {
		tok.SpecialTokens = make(map[string]int)
	}
	tok.reindex()
	return tok, nil
}
