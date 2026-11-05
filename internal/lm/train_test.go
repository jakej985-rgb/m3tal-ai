package lm

import (
	"math"
	"strings"
	"testing"
)

func TestTrainStreamLowersLossOnRepeatedCorpus(t *testing.T) {
	text := strings.Repeat("the little model learns to predict tokens. ", 40)
	tokenizer, err := TrainTokenizer([]string{text}, 256)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(Config{Context: 8, Width: 8, Layers: 1, Heads: 2}, tokenizer, 31)
	if err != nil {
		t.Fatal(err)
	}
	tokens := EncodeCorpusTokenizes(tokenizer, []string{text})

	before, err := model.Evaluate([]string{text})
	if err != nil {
		t.Fatal(err)
	}
	var logged []TrainMetrics
	loss, err := model.TrainStream(NewSliceSource(tokens), TrainConfig{
		MaxSteps:     30,
		LearningRate: 0.01,
		WarmupSteps:  3,
		GradClip:     1.0,
		LogEvery:     10,
	}, func(m TrainMetrics) { logged = append(logged, m) })
	if err != nil {
		t.Fatal(err)
	}
	if loss <= 0 || math.IsNaN(loss) || math.IsInf(loss, 0) {
		t.Fatalf("invalid final loss %v", loss)
	}
	after, err := model.Evaluate([]string{text})
	if err != nil {
		t.Fatal(err)
	}
	if !(after < before) {
		t.Fatalf("training did not lower loss: before %.4f after %.4f", before, after)
	}
	if len(logged) == 0 {
		t.Fatal("expected metrics to be logged")
	}
	last := logged[len(logged)-1]
	if last.Step <= 0 || last.TokensSeen <= 0 || last.LearningRate <= 0 {
		t.Fatalf("bad metrics: %+v", last)
	}
	if last.TokensPerSec <= 0 {
		t.Fatalf("tokens/sec not measured: %+v", last)
	}
}

func TestLearningRateSchedule(t *testing.T) {
	cfg := TrainConfig{MaxSteps: 100, LearningRate: 0.01, WarmupSteps: 10, MinLRRatio: 0.1}
	if got := cfg.learningRateAt(5); math.Abs(got-0.005) > 1e-9 {
		t.Fatalf("warmup lr = %v, want 0.005", got)
	}
	peak := cfg.learningRateAt(10)
	if math.Abs(peak-0.01) > 1e-9 {
		t.Fatalf("peak lr = %v, want 0.01", peak)
	}
	mid := cfg.learningRateAt(55)
	if mid >= peak || mid <= 0 {
		t.Fatalf("decay lr = %v out of range", mid)
	}
	end := cfg.learningRateAt(100)
	if math.Abs(end-0.001) > 1e-6 {
		t.Fatalf("final lr = %v, want ~0.001", end)
	}
}

func TestAdamWStyleUpdateMovesWeights(t *testing.T) {
	model, _ := smallModel(t)
	// Inject a known gradient and verify a single clipped step updates values.
	for _, p := range model.parameters {
		for i := range p.gradient {
			p.gradient[i] = 0.5
		}
	}
	before := parameterValues(model)
	norm := model.adamStepLR(0.01, 1.0)
	if norm <= 0 {
		t.Fatal("expected positive grad norm")
	}
	after := parameterValues(model)
	changed := false
	for i := range before {
		if after[i] != before[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("adam step did not update weights")
	}
	if model.step != 1 {
		t.Fatalf("step = %d, want 1", model.step)
	}
}

func TestTrainStreamStopsAtMaxSteps(t *testing.T) {
	model, _ := smallModel(t)
	tokens := make([]uint32, 100000)
	for i := range tokens {
		tokens[i] = uint32(i % 256)
	}
	_, err := model.TrainStream(NewSliceSource(tokens), TrainConfig{
		MaxSteps:     5,
		LearningRate: 0.001,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if model.step != 5 {
		t.Fatalf("step = %d, want 5", model.step)
	}
}
