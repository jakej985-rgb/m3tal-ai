// Package lm implements a small language model and tokenizer in pure Go.
package lm

import (
	"fmt"
	"sort"
	"strings"
)

// Tokenizer is a byte-level BPE tokenizer. IDs 0..255 represent raw bytes and
// each learned merge is appended in training order.
type Tokenizer struct {
	Tokens []string `json:"tokens"`
	Merges []Merge  `json:"merges"`
	index  map[string]int
}

// Merge records one ordered byte-pair merge.
type Merge struct {
	Left  string `json:"left"`
	Right string `json:"right"`
	Token string `json:"token"`
}

// TrainTokenizer learns at most targetSize tokens from the supplied corpus.
func TrainTokenizer(corpus []string, targetSize int) (*Tokenizer, error) {
	if len(corpus) == 0 {
		return nil, fmt.Errorf("tokenizer corpus is empty")
	}
	if targetSize < 256 {
		return nil, fmt.Errorf("tokenizer target size must be at least 256")
	}
	if targetSize > 2048 {
		return nil, fmt.Errorf("tokenizer target size must not exceed 2048")
	}
	t := &Tokenizer{Tokens: make([]string, 256)}
	known := make(map[string]bool, targetSize)
	for i := 0; i < 256; i++ {
		t.Tokens[i] = string([]byte{byte(i)})
		known[t.Tokens[i]] = true
	}
	sequences := make([][]string, len(corpus))
	for i, text := range corpus {
		sequences[i] = byteSymbols(text)
	}
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

func (t *Tokenizer) reindex() {
	t.index = make(map[string]int, len(t.Tokens))
	for id, token := range t.Tokens {
		t.index[token] = id
	}
}

// Encode turns text into token IDs by replaying learned BPE merges.
func (t *Tokenizer) Encode(text string) []int {
	if t.index == nil {
		t.reindex()
	}
	seq := byteSymbols(text)
	for _, merge := range t.Merges {
		seq = applyMerge(seq, merge.Left, merge.Right, merge.Token)
	}
	ids := make([]int, len(seq))
	for i, token := range seq {
		ids[i] = t.index[token]
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
