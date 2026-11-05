package lm

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// This file implements the supervised fine-tuning pipeline over chat
// conversations (issue: Stage 3 SFT). Loss is applied only to assistant
// tokens via the chat template mask; the learning-rate schedule reuses
// TrainConfig (warmup + cosine decay + grad clip). Optimizer state and
// step count persist through SaveCheckpoint/LoadCheckpoint, so SFT runs
// resume exactly like pretraining.

// SFTMetrics is one logged SFT step.
type SFTMetrics struct {
	Step         int64   `json:"step"`
	Loss         float64 `json:"loss"`
	LearningRate float64 `json:"learning_rate"`
	TokensPerSec float64 `json:"tokens_per_sec"`
	GradNorm     float64 `json:"grad_norm"`
	TokensSeen   int64   `json:"tokens_seen"`
	MaskedTokens int64   `json:"masked_tokens_seen"` // assistant (loss) tokens
	WindowTokens int64   `json:"window_tokens_seen"` // total window tokens
	ExamplesSeen int64   `json:"examples_seen"`
}

// sftWindow is a prepared masked training window.
type sftWindow struct {
	input   []int
	targets []int
	mask    []bool // per-target mask: true = assistant token (loss applies)
}

// PrepareChatExamples serializes chat examples into masked windows of at
// most context+1 tokens, dropping overflow from the front so the most
// recent turns are kept.
func (m *Model) PrepareChatExamples(examples []ChatExample) ([]sftWindow, int64, error) {
	context := m.Config.Context
	var windows []sftWindow
	var maskedTotal int64
	for _, ex := range examples {
		ids, idMask := ex.SerializedParts(m.Tokenizer)
		if len(ids) < 2 {
			continue
		}
		if len(ids) > context+1 {
			ids = ids[len(ids)-(context+1):]
			idMask = idMask[len(idMask)-(context+1):]
		}
		input := make([]int, len(ids)-1)
		targets := make([]int, len(ids)-1)
		mask := make([]bool, len(ids)-1)
		hasActive := false
		for i := range input {
			input[i] = ids[i]
			targets[i] = ids[i+1]
			mask[i] = idMask[i+1]
			if mask[i] {
				hasActive = true
				maskedTotal++
			}
		}
		if hasActive {
			windows = append(windows, sftWindow{input, targets, mask})
		}
	}
	if len(windows) == 0 {
		return nil, 0, fmt.Errorf("no chat examples produce assistant-masked windows within context %d", context)
	}
	return windows, maskedTotal, nil
}

// TrainChat runs SFT over chat examples until cfg.MaxSteps is reached.
// Each step samples one prepared window uniformly at random (seeded by
// cfg.Seed for reproducibility). logf is invoked every cfg.LogEvery
// steps and once at the end. Returns the mean loss over all steps.
func (m *Model) TrainChat(examples []ChatExample, cfg TrainConfig, logf func(SFTMetrics)) (float64, error) {
	cfg.Normalize()
	if cfg.MaxSteps < 0 || cfg.LearningRate <= 0 || math.IsNaN(cfg.LearningRate) || math.IsInf(cfg.LearningRate, 0) {
		return 0, fmt.Errorf("max steps must be >= 0 and learning rate finite and positive")
	}
	if len(examples) == 0 {
		return 0, fmt.Errorf("chat examples required")
	}
	windows, maskedTotal, err := m.PrepareChatExamples(examples)
	if err != nil {
		return 0, err
	}
	m.backend = nil

	rng := rand.New(rand.NewSource(cfg.Seed))
	var totalLoss float64
	var steps, tokensSeen, maskedSeen, examplesSeen int64
	start := time.Now()
	lastLog := m.step

	for cfg.MaxSteps <= 0 || m.step < cfg.MaxSteps {
		w := windows[rng.Intn(len(windows))]
		cache := m.forward(w.input)
		loss := meanWindowLoss(cache, w.targets, w.mask)
		totalLoss += loss
		steps++
		tokensSeen += int64(len(w.input))
		for _, mk := range w.mask {
			if mk {
				maskedSeen++
			}
		}
		examplesSeen++

		zeroGradients(m.parameters)
		m.backward(cache, w.targets, w.mask)
		lr := cfg.learningRateAt(m.step + 1)
		gradNorm := m.adamStepLR(float32(lr), cfg.GradClip)

		if logf != nil && cfg.LogEvery > 0 && (m.step-lastLog) >= cfg.LogEvery {
			elapsed := time.Since(start).Seconds()
			logf(SFTMetrics{
				Step:         m.step,
				Loss:         loss,
				LearningRate: lr,
				TokensPerSec: float64(tokensSeen) / math.Max(elapsed, 1e-9),
				GradNorm:     gradNorm,
				TokensSeen:   tokensSeen,
				MaskedTokens: maskedSeen,
				WindowTokens: tokensSeen,
				ExamplesSeen: examplesSeen,
			})
			lastLog = m.step
		}
	}
	if steps == 0 {
		return 0, fmt.Errorf("no training steps completed")
	}
	if logf != nil {
		logf(SFTMetrics{
			Step:         m.step,
			Loss:         totalLoss / float64(steps),
			LearningRate: cfg.learningRateAt(m.step),
			TokensPerSec: float64(tokensSeen) / math.Max(time.Since(start).Seconds(), 1e-9),
			TokensSeen:   tokensSeen,
			MaskedTokens: maskedSeen,
			WindowTokens: tokensSeen,
			ExamplesSeen: examplesSeen,
		})
	}
	_ = maskedTotal // reserved for reports; per-step counts are authoritative
	return totalLoss / float64(steps), nil
}

// SFTSummary describes a completed SFT run for reports/checkpoints.
type SFTSummary struct {
	TemplateVersion string  `json:"template_version"`
	TokenizerHash   string  `json:"tokenizer_hash"`
	Examples        int     `json:"examples"`
	Windows         int     `json:"windows"`
	MaskedTokens    int64   `json:"masked_tokens"`
	FinalLoss       float64 `json:"final_loss"`
	Steps           int64   `json:"steps"`
}

// SummarizeSFT prepares examples (dry-run) and reports dataset statistics
// without training — used by the CLI to log token counts before a run.
func (m *Model) SummarizeSFT(examples []ChatExample) (SFTSummary, error) {
	windows, maskedTotal, err := m.PrepareChatExamples(examples)
	if err != nil {
		return SFTSummary{}, err
	}
	return SFTSummary{
		TemplateVersion: ChatTemplateVersion,
		TokenizerHash:   TokenizerFingerprint(m.Tokenizer),
		Examples:        len(examples),
		Windows:         len(windows),
		MaskedTokens:    maskedTotal,
		Steps:           m.step,
	}, nil
}
