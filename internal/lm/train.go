package lm

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// This file implements the Stage 0 CPU training loop: causal language-model
// loss over dataset windows, AdamW-style updates with gradient clipping,
// learning-rate warmup + cosine decay, and step-level metrics logging.
//
// Loss is computed only on valid target tokens: each window of Context+1
// tokens produces Context predictions; every position is a valid next-token
// target for plain LM training (mask support remains for instruction data).

// TrainConfig controls a TrainStream run.
type TrainConfig struct {
	MaxSteps     int64   // total optimizer steps (0 = until stream exhausted)
	LearningRate float64 // peak learning rate
	WarmupSteps  int64   // linear warmup steps (0 = none)
	MinLRRatio   float64 // cosine decay floor as fraction of peak (default 0.1; 0 = decay to 0)
	GradClip     float64 // global grad-norm clip threshold (default 1.0)
	LogEvery     int64   // log metrics every N steps (0 = silent)
	Seed         int64   // window-order seed (only used with ShuffledSeeded streams)
	// EvalEvery, if > 0, is reserved for periodic evaluation hooks.
	EvalEvery int64
}

// Normalize fills defaults.
func (c *TrainConfig) Normalize() {
	if c.MinLRRatio <= 0 {
		c.MinLRRatio = 0.1
	}
	if c.GradClip <= 0 {
		c.GradClip = 1.0
	}
}

// TrainMetrics is one logged training step.
type TrainMetrics struct {
	Step         int64   `json:"step"`
	Loss         float64 `json:"loss"`
	LearningRate float64 `json:"learning_rate"`
	TokensPerSec float64 `json:"tokens_per_sec"`
	GradNorm     float64 `json:"grad_norm"`
	TokensSeen   int64   `json:"tokens_seen"`
}

// learningRateAt returns the LR for a step under linear warmup followed by
// cosine decay to MinLRRatio of the peak.
func (c TrainConfig) learningRateAt(step int64) float64 {
	if c.WarmupSteps > 0 && step <= c.WarmupSteps {
		return c.LearningRate * float64(step) / float64(c.WarmupSteps)
	}
	if c.MaxSteps <= c.WarmupSteps {
		return c.LearningRate
	}
	progress := float64(step-c.WarmupSteps) / float64(c.MaxSteps-c.WarmupSteps)
	if progress > 1 {
		progress = 1
	}
	cos := 0.5 * (1 + math.Cos(math.Pi*progress))
	return c.LearningRate * (c.MinLRRatio + (1-c.MinLRRatio)*cos)
}

// WindowSource yields training windows of token IDs. TokenStream (sharded
// datasets) and SliceSource (in-memory corpora) both implement it.
type WindowSource interface {
	// Window returns up to context+1 consecutive token IDs, or nil,false at
	// end of stream.
	Window(context int) ([]uint32, bool)
	// Err returns the first stream error encountered, if any.
	Err() error
}

// SliceSource adapts an in-memory token slice to WindowSource. Windows are
// non-overlapping chunks of context+1 tokens.
type SliceSource struct {
	tokens []uint32
	pos    int
}

// NewSliceSource wraps tokens for sequential windowed training.
func NewSliceSource(tokens []uint32) *SliceSource { return &SliceSource{tokens: tokens} }

func (s *SliceSource) Window(context int) ([]uint32, bool) {
	need := context + 1
	if s.pos+need > len(s.tokens) {
		// Allow a final short window if it still has >= 2 tokens.
		if len(s.tokens)-s.pos >= 2 {
			w := s.tokens[s.pos:]
			s.pos = len(s.tokens)
			return w, true
		}
		return nil, false
	}
	w := s.tokens[s.pos : s.pos+need]
	s.pos += need
	return w, true
}

// Next implements TokenFeed: returns up to n consecutive tokens.
func (s *SliceSource) Next(n int) ([]uint32, bool) {
	if s.pos >= len(s.tokens) {
		return nil, false
	}
	end := s.pos + n
	if end > len(s.tokens) {
		end = len(s.tokens)
	}
	out := s.tokens[s.pos:end]
	s.pos = end
	return out, len(out) > 0
}

func (s *SliceSource) Err() error { return nil }

// TrainStream runs the training loop over a token stream until MaxSteps is
// reached or the stream ends. Windows of Config.Context+1 tokens are consumed
// sequentially; each window's last token predicts nothing (inputs are
// window[:context], targets window[1:context+1]).
//
// logf, if non-nil, is invoked every cfg.LogEvery steps and once at the end.
// Returns the mean loss over the final window batch for convenience.
func (m *Model) TrainStream(ts WindowSource, cfg TrainConfig, logf func(TrainMetrics)) (float64, error) {
	cfg.Normalize()
	if cfg.MaxSteps < 0 || cfg.LearningRate <= 0 || math.IsNaN(cfg.LearningRate) || math.IsInf(cfg.LearningRate, 0) {
		return 0, fmt.Errorf("max steps must be >= 0 and learning rate finite and positive")
	}
	if ts == nil {
		return 0, fmt.Errorf("token stream is required")
	}
	m.backend = nil
	context := m.Config.Context
	var totalLoss float64
	var windows, tokensSeen int64
	start := time.Now()
	lastLog := m.step

	for cfg.MaxSteps <= 0 || m.step < cfg.MaxSteps {
		window, ok := ts.Window(context)
		if !ok {
			break
		}
		if len(window) < 2 {
			continue
		}
		input := make([]int, 0, len(window)-1)
		targets := make([]int, 0, len(window)-1)
		for _, tok := range window[:len(window)-1] {
			input = append(input, int(tok))
		}
		for _, tok := range window[1:] {
			targets = append(targets, int(tok))
		}
		cache := m.forward(input)
		loss := meanWindowLoss(cache, targets, nil)
		totalLoss += loss
		windows++
		tokensSeen += int64(len(window))

		zeroGradients(m.parameters)
		m.backward(cache, targets, nil)
		lr := cfg.learningRateAt(m.step + 1)
		gradNorm := m.adamStepLR(float32(lr), cfg.GradClip)

		if logf != nil && cfg.LogEvery > 0 && (m.step-lastLog) >= cfg.LogEvery {
			elapsed := time.Since(start).Seconds()
			logf(TrainMetrics{
				Step:         m.step,
				Loss:         loss,
				LearningRate: lr,
				TokensPerSec: float64(tokensSeen) / math.Max(elapsed, 1e-9),
				GradNorm:     gradNorm,
				TokensSeen:   tokensSeen,
			})
			lastLog = m.step
		}
	}
	if err := ts.Err(); err != nil {
		return 0, fmt.Errorf("token stream: %w", err)
	}
	if windows == 0 {
		return 0, fmt.Errorf("no complete training windows in stream")
	}
	return totalLoss / float64(windows), nil
}

// ShuffleTokenIDs is a helper for callers that want a fully shuffled token
// slice (used by tests and small in-memory corpora).
func ShuffleTokenIDs(ids []uint32, seed int64) {
	r := rand.New(rand.NewSource(seed))
	r.Shuffle(len(ids), func(i, j int) { ids[i], ids[j] = ids[j], ids[i] })
}
