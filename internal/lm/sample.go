package lm

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"unicode/utf8"
)

// This file implements generation sampling controls (issue: Stage 3
// inference sampling/stopping/streaming):
//
//   - Temperature: 0 selects greedy argmax; >0 scales logits.
//   - TopK: keep only the k highest-probability tokens (0 = disabled).
//   - TopP: nucleus sampling — smallest set whose cumulative probability
//     reaches p (0 = disabled, must be in (0,1] when set).
//   - Stop: decoded-text markers that halt generation and are truncated
//     from the output (e.g. chat "<|im_end|>").
//   - RepetitionPenalty: >1.0 down-weights already-generated tokens.
//   - GenerateStream emits decoded text deltas per token; an emit error
//     cancels generation and is propagated to the caller.
//
// Generation uses the KV cache (O(1) per token) with full-forward fallback
// once the context window is exhausted. Fixed Seed + settings reproduce
// output exactly on the same backend.

// GenerateOptions configures text generation.
type GenerateOptions struct {
	MaxTokens         int
	Temperature       float64 // 0 = greedy argmax; >0 = softmax temperature
	TopK              int     // 0 = disabled
	TopP              float64 // 0 = disabled; otherwise in (0,1]
	Seed              int64
	RepetitionPenalty float64  // > 1.0 penalizes already generated tokens (e.g. 1.15)
	Stop              []string // decoded-text stop markers; truncated from output
}

// Validate checks option ranges.
func (o GenerateOptions) Validate() error {
	if o.MaxTokens < 0 {
		return fmt.Errorf("max tokens must be non-negative")
	}
	if o.Temperature < 0 || math.IsNaN(o.Temperature) || math.IsInf(o.Temperature, 0) {
		return fmt.Errorf("temperature must be finite and >= 0")
	}
	if o.TopK < 0 {
		return fmt.Errorf("top-k must be >= 0")
	}
	if o.TopP != 0 && (o.TopP <= 0 || o.TopP > 1 || math.IsNaN(o.TopP) || math.IsInf(o.TopP, 0)) {
		return fmt.Errorf("top-p must be 0 (disabled) or in (0,1]")
	}
	if math.IsNaN(o.RepetitionPenalty) || math.IsInf(o.RepetitionPenalty, 0) {
		return fmt.Errorf("repetition penalty must be finite")
	}
	return nil
}

// sampleCandidate is a token id with its selection probability.
type sampleCandidate struct {
	id int
	p  float64
}

// filterCandidates applies top-k then top-p nucleus filtering to a
// probability map and returns candidates sorted by descending probability.
func filterCandidates(probs map[int]float64, topK int, topP float64) []sampleCandidate {
	cands := make([]sampleCandidate, 0, len(probs))
	for id, p := range probs {
		if p > 0 {
			cands = append(cands, sampleCandidate{id, p})
		}
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].p > cands[j].p })
	if topK > 0 && len(cands) > topK {
		cands = cands[:topK]
	}
	if topP > 0 {
		total, keep := 0.0, 0
		for keep < len(cands) {
			total += cands[keep].p
			keep++
			if total >= topP {
				break
			}
		}
		cands = cands[:keep]
	}
	return cands
}

// pickCandidate selects one candidate: argmax for greedy (temperature 0),
// seeded categorical sampling otherwise. probs must be non-empty.
func pickCandidate(cands []sampleCandidate, temperature float64, rng *rand.Rand) int {
	if temperature == 0 {
		return cands[0].id // sorted desc → highest probability
	}
	total := 0.0
	for _, c := range cands {
		total += c.p
	}
	draw := rng.Float64() * total
	for _, c := range cands {
		draw -= c.p
		if draw <= 0 {
			return c.id
		}
	}
	return cands[len(cands)-1].id
}

// generateCore runs autoregressive generation, invoking emit with decoded
// text deltas when non-nil. Stop markers halt generation and are removed
// from the returned text. An emit error cancels generation; the partial
// output and the error are both returned.
func (m *Model) generateCore(prompt string, opts GenerateOptions, emit func(string) error) (string, bool, error) {
	if err := opts.Validate(); err != nil {
		return "", false, err
	}
	penalty := float32(opts.RepetitionPenalty)
	if penalty <= 0 {
		penalty = 1.0
	}
	ids := m.Tokenizer.Encode(prompt)
	if len(ids) == 0 {
		ids = m.Tokenizer.Encode(" ")
	}
	prefix := []byte(m.Tokenizer.Decode(ids))
	genOffset := len(prefix) // byte offset where generated text begins
	backend, rng := m.inferenceBackend(), rand.New(rand.NewSource(opts.Seed))

	generatedCounts := make(map[int]int)
	genIDs := make([]int, 0, opts.MaxTokens)

	// KV-cache prefill.
	m.Config.Normalize()
	kvCache := backend.NewKVCache()
	defer kvCache.Free()
	var rawLogits []float32
	for i, id := range ids {
		rawLogits = backend.ForwardStepKV(id, i, kvCache)
	}
	pos := len(ids)

	emitted := 0 // bytes of generated text already streamed

	for n := 0; n < opts.MaxTokens; n++ {
		logits := append([]float32(nil), rawLogits...)

		if penalty > 1.0 {
			for id, count := range generatedCounts {
				if id >= 0 && id < len(logits) && count > 0 {
					tok := m.Tokenizer.Tokens[id]
					if tok == "{" || tok == "}" || tok == "[" || tok == "]" || tok == ":" || tok == "\"" || tok == "," || tok == " " || tok == "\n" {
						continue
					}
					if logits[id] > 0 {
						logits[id] /= penalty
					} else {
						logits[id] *= penalty
					}
				}
			}
		}

		// Greedy decoding: argmax over tokens that keep the output valid UTF-8.
		var choice int
		if opts.Temperature == 0 {
			best := float32(math.Inf(-1))
			choice = -1
			for id, value := range logits {
				if id >= len(m.Tokenizer.Tokens) || !validUTF8Append(prefix, m.Tokenizer.Tokens[id]) {
					continue
				}
				if value > best {
					best, choice = value, id
				}
			}
			if choice < 0 {
				break
			}
		} else {
			// Build probability candidates over tokens that keep the
			// output valid UTF-8.
			maximum := float32(math.Inf(-1))
			for _, value := range logits {
				if value > maximum {
					maximum = value
				}
			}
			probs := make(map[int]float64, len(logits))
			var sum float64
			for id, value := range logits {
				if id >= len(m.Tokenizer.Tokens) || !validUTF8Append(prefix, m.Tokenizer.Tokens[id]) {
					continue
				}
				p := math.Exp(float64(value-maximum) / opts.Temperature)
				probs[id] = p
				sum += p
			}
			if sum == 0 || len(probs) == 0 {
				// Fallback: first token that keeps UTF-8 valid.
				for id, token := range m.Tokenizer.Tokens {
					if validUTF8Append(prefix, token) {
						probs[id], sum = 1, 1
						break
					}
				}
			}
			if len(probs) == 0 {
				break
			}
			for id := range probs {
				probs[id] /= sum
			}
			cands := filterCandidates(probs, opts.TopK, opts.TopP)
			if len(cands) == 0 {
				break
			}
			choice = pickCandidate(cands, opts.Temperature, rng)
		}

		ids = append(ids, choice)
		genIDs = append(genIDs, choice)
		generatedCounts[choice]++
		prefix = appendString(prefix, m.Tokenizer.Tokens[choice])

		// Stream decoded delta (UTF-8-safe boundary).
		if emit != nil {
			full := m.Tokenizer.Decode(genIDs)
			delta := full[emitted:]
			for len(delta) > 0 && !utf8.ValidString(delta) {
				delta = delta[:len(delta)-1]
			}
			if len(delta) > 0 {
				if err := emit(delta); err != nil {
					return string(prefix[genOffset:]), false, fmt.Errorf("stream cancelled: %w", err)
				}
				emitted += len(delta)
			}
		}

		// Stop-marker check on decoded generated text.
		genText := string(prefix[genOffset:])
		stopAt := -1
		for _, marker := range opts.Stop {
			if marker == "" {
				continue
			}
			if idx := strings.Index(genText, marker); idx >= 0 && (stopAt < 0 || idx < stopAt) {
				stopAt = idx
			}
		}
		if stopAt >= 0 {
			prefix = prefix[:genOffset+stopAt]
			return string(prefix[genOffset:]), true, nil
		}

		// Next-token logits: KV step or full-forward fallback.
		if pos < m.Config.Context {
			rawLogits = backend.ForwardStepKV(choice, pos, kvCache)
			pos++
		} else {
			rawLogits = backend.Forward(ids)
		}
	}
	return string(prefix[genOffset:]), false, nil
}

// GenerateWithOptions generates text with the given sampling options.
func (m *Model) GenerateWithOptions(prompt string, opts GenerateOptions) (string, error) {
	out, _, err := m.generateCore(prompt, opts, nil)
	return out, err
}

// GenerateStream generates text like GenerateWithOptions but invokes emit
// with each decoded text delta as tokens are produced. If emit returns an
// error, generation stops and the partial output is returned with the
// error wrapped as "stream cancelled".
func (m *Model) GenerateStream(prompt string, opts GenerateOptions, emit func(string) error) (string, error) {
	out, _, err := m.generateCore(prompt, opts, emit)
	return out, err
}

// Generate samples tokens at the requested temperature using deterministic
// seeded sampling with a 1.15 repetition penalty.
func (m *Model) Generate(prompt string, maxTokens int, temperature float64, seed int64) (string, error) {
	return m.GenerateWithOptions(prompt, GenerateOptions{
		MaxTokens:         maxTokens,
		Temperature:       temperature,
		Seed:              seed,
		RepetitionPenalty: 1.15,
	})
}
