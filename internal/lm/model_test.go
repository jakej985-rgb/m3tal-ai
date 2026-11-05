package lm

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"
	"unicode/utf8"

	"m3tal-ai/internal/autograd"
)

func TestTokenizerRoundTripsUTF8(t *testing.T) {
	text := "M3TAL café — 東京 🤖\n"
	tokenizer, err := TrainTokenizer([]string{text + text}, 280)
	if err != nil {
		t.Fatal(err)
	}
	got := tokenizer.Decode(tokenizer.Encode(text))
	if got != text {
		t.Fatalf("round trip = %q, want %q", got, text)
	}
}

func TestCheckpointRoundTripPreservesGeneration(t *testing.T) {
	text := "the quick brown fox jumps over the lazy dog. café 東京 🤖 "
	tokenizer, err := TrainTokenizer([]string{text + text}, 280)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(Config{Context: 8, Width: 4}, tokenizer, 13)
	if err != nil {
		t.Fatal(err)
	}
	before, err := model.Generate("the quick", 12, 0.8, 7)
	if err != nil {
		t.Fatal(err)
	}
	path := t.TempDir() + "/model.json"
	if err := model.SaveCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	after, err := loaded.Generate("the quick", 12, 0.8, 7)
	if err != nil {
		t.Fatal(err)
	}
	if before != after {
		t.Fatalf("generation changed after checkpoint round trip: %q != %q", before, after)
	}
}

func TestTrainingLowersHeldOutLossOnRepeatedTinyCorpus(t *testing.T) {
	text := strings.Repeat("hello world. cpu model learns text. ", 12)
	tokenizer, err := TrainTokenizer([]string{text}, 256)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(Config{Context: 6, Width: 4}, tokenizer, 23)
	if err != nil {
		t.Fatal(err)
	}
	before, err := model.Evaluate([]string{text})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Train([]string{text}, 8, 20, 0.003); err != nil {
		t.Fatal(err)
	}
	after, err := model.Evaluate([]string{text})
	if err != nil {
		t.Fatal(err)
	}
	if !(after < before) {
		t.Fatalf("training did not lower loss: before %.4f after %.4f", before, after)
	}
}

func TestGenerationAlwaysReturnsValidUTF8(t *testing.T) {
	tokenizer, err := TrainTokenizer([]string{"small test corpus with non-ascii café 🤖"}, 256)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(Config{Context: 8, Width: 4}, tokenizer, 41)
	if err != nil {
		t.Fatal(err)
	}
	generated, err := model.Generate("café", 80, 1, 5)
	if err != nil {
		t.Fatal(err)
	}
	if !utf8.ValidString(generated) {
		t.Fatalf("generation is not valid UTF-8: %q", generated)
	}
	if strings.ContainsRune(generated, 0) {
		t.Fatalf("generation contains a NUL control character: %q", generated)
	}
}

func TestExplicitGradientsMatchFiniteDifferences(t *testing.T) {
	tokenizer, err := TrainTokenizer([]string{"cpu gradient check café"}, 256)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(Config{Context: 6, Width: 4}, tokenizer, 29)
	if err != nil {
		t.Fatal(err)
	}
	ids := tokenizer.Encode("grad")
	if len(ids) < 3 {
		t.Fatal("expected prompt to encode to at least three tokens")
	}
	input := ids[:len(ids)-1]
	targets := ids[1:]
	zeroGradients(model.parameters)
	model.backward(model.forward(input), targets, nil)

	const epsilon float32 = 1e-3
	for parameterIndex, parameter := range model.parameters {
		for i, analytic := range parameter.gradient {
			original := parameter.values[i]
			parameter.values[i] = original + epsilon
			plus := meanWindowLoss(model.forward(input), targets, nil)
			parameter.values[i] = original - epsilon
			minus := meanWindowLoss(model.forward(input), targets, nil)
			parameter.values[i] = original
			numerical := (plus - minus) / (2 * float64(epsilon))
			difference := math.Abs(float64(analytic) - numerical)
			scale := math.Max(1, math.Max(math.Abs(float64(analytic)), math.Abs(numerical)))
			if difference > 5e-2 && difference/scale > 5e-2 {
				t.Fatalf("parameter %d[%d] analytic=%g finite-difference=%g diff=%g", parameterIndex, i, analytic, numerical, difference)
			}
		}
	}
}

func TestMaskedGradientsMatchFiniteDifferences(t *testing.T) {
	tokenizer, err := TrainTokenizer([]string{"masked prompt completion test café"}, 256)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(Config{Context: 6, Width: 4}, tokenizer, 31)
	if err != nil {
		t.Fatal(err)
	}
	ids := tokenizer.Encode("masked")
	if len(ids) < 4 {
		t.Fatal("expected prompt to encode to at least four tokens")
	}
	input := ids[:len(ids)-1]
	targets := ids[1:]
	// Mask: first token masked out (false), remaining active (true)
	mask := make([]bool, len(targets))
	for i := 1; i < len(mask); i++ {
		mask[i] = true
	}

	zeroGradients(model.parameters)
	model.backward(model.forward(input), targets, mask)

	const epsilon float32 = 1e-3
	for parameterIndex, parameter := range model.parameters {
		for i, analytic := range parameter.gradient {
			original := parameter.values[i]
			parameter.values[i] = original + epsilon
			plus := meanWindowLoss(model.forward(input), targets, mask)
			parameter.values[i] = original - epsilon
			minus := meanWindowLoss(model.forward(input), targets, mask)
			parameter.values[i] = original
			numerical := (plus - minus) / (2 * float64(epsilon))
			difference := math.Abs(float64(analytic) - numerical)
			scale := math.Max(1, math.Max(math.Abs(float64(analytic)), math.Abs(numerical)))
			if difference > 5e-2 && difference/scale > 5e-2 {
				t.Fatalf("masked parameter %d[%d] analytic=%g finite-difference=%g diff=%g", parameterIndex, i, analytic, numerical, difference)
			}
		}
	}
}

func TestLoadCheckpointAcceptsLegacyFloat64Values(t *testing.T) {
	text := "the quick brown fox jumps over the lazy dog. café 東京 🤖 "
	tokenizer, err := TrainTokenizer([]string{text + text}, 280)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(Config{Context: 8, Width: 4}, tokenizer, 13)
	if err != nil {
		t.Fatal(err)
	}
	before, err := model.Generate("the quick", 12, 0.8, 7)
	if err != nil {
		t.Fatal(err)
	}
	path := t.TempDir() + "/legacy.json"
	if err := model.SaveCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var state map[string]json.RawMessage
	if err := json.Unmarshal(data, &state); err != nil {
		t.Fatal(err)
	}
	var values []float64
	if err := json.Unmarshal(state["values"], &values); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(values)
	if err != nil {
		t.Fatal(err)
	}
	state["values"] = encoded
	rewritten, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, rewritten, 0o600); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	after, err := loaded.Generate("the quick", 12, 0.8, 7)
	if err != nil {
		t.Fatal(err)
	}
	if before != after {
		t.Fatalf("generation changed after loading float64 checkpoint values: %q != %q", before, after)
	}
}

func TestLoadCorpusSkipsSymlinksAndSensitiveFiles(t *testing.T) {
	root, outside := t.TempDir(), t.TempDir()
	if err := os.WriteFile(root+"/guide.md", []byte("safe corpus text"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(root+"/secrets.md", []byte("secret text"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(outside+"/outside.md", []byte("outside text"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside+"/outside.md", root+"/linked.md"); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	texts, err := LoadCorpus([]string{root})
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(texts, " ")
	if !strings.Contains(joined, "safe corpus text") || strings.Contains(joined, "secret text") || strings.Contains(joined, "outside text") {
		t.Fatalf("unexpected corpus contents: %q", joined)
	}
}

func TestMultiLayerMultiHeadGradientsMatchFiniteDifferences(t *testing.T) {
	tokenizer, err := TrainTokenizer([]string{"multi layer multi head gradient check café"}, 256)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(Config{Context: 6, Width: 4, Layers: 2, Heads: 2}, tokenizer, 43)
	if err != nil {
		t.Fatal(err)
	}
	ids := tokenizer.Encode("grad")
	if len(ids) < 3 {
		t.Fatal("expected prompt to encode to at least three tokens")
	}
	input := ids[:len(ids)-1]
	targets := ids[1:]
	zeroGradients(model.parameters)
	model.backward(model.forward(input), targets, nil)

	const epsilon float32 = 1e-3
	for parameterIndex, parameter := range model.parameters {
		for i, analytic := range parameter.gradient {
			original := parameter.values[i]
			parameter.values[i] = original + epsilon
			plus := meanWindowLoss(model.forward(input), targets, nil)
			parameter.values[i] = original - epsilon
			minus := meanWindowLoss(model.forward(input), targets, nil)
			parameter.values[i] = original
			numerical := (plus - minus) / (2 * float64(epsilon))
			difference := math.Abs(float64(analytic) - numerical)
			scale := math.Max(1, math.Max(math.Abs(float64(analytic)), math.Abs(numerical)))
			if difference > 5e-2 && difference/scale > 5e-2 {
				t.Fatalf("layer parameter %d[%d] analytic=%g finite-difference=%g diff=%g", parameterIndex, i, analytic, numerical, difference)
			}
		}
	}
}

func TestMultiLayerMultiHeadCheckpointRoundTrip(t *testing.T) {
	text := "the quick brown fox jumps over the lazy dog. café 東京 🤖 "
	tokenizer, err := TrainTokenizer([]string{text + text}, 280)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{Context: 8, Width: 4, Layers: 2, Heads: 2}
	model, err := NewModel(cfg, tokenizer, 13)
	if err != nil {
		t.Fatal(err)
	}
	before, err := model.Generate("the quick", 12, 0.8, 7)
	if err != nil {
		t.Fatal(err)
	}
	path := t.TempDir() + "/model_multi.json"
	if err := model.SaveCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Config.Layers != 2 || loaded.Config.Heads != 2 {
		t.Fatalf("loaded config mismatch: layers=%d heads=%d", loaded.Config.Layers, loaded.Config.Heads)
	}
	after, err := loaded.Generate("the quick", 12, 0.8, 7)
	if err != nil {
		t.Fatal(err)
	}
	if before != after {
		t.Fatalf("generation changed after checkpoint round trip: %q != %q", before, after)
	}
}

func BenchmarkGenerate10Tokens(b *testing.B) {
	tokenizer, err := TrainTokenizer([]string{"benchmark corpus text repeated benchmark corpus text"}, 256)
	if err != nil {
		b.Fatal(err)
	}
	model, err := NewModel(Config{Context: 8, Width: 8}, tokenizer, 3)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.Generate("benchmark", 10, 0.8, int64(i)); err != nil {
			b.Fatal(err)
		}
	}
}

func TestModelConfigAndInitialization(t *testing.T) {
	// Test validation of invalid configurations
	invalidConfigs := []struct {
		name string
		cfg  Config
		seed int64
	}{
		{"context < 2", Config{Context: 1, Width: 16, Layers: 2, Heads: 2, VocabSize: 16}, 1},
		{"width < 2", Config{Context: 8, Width: 1, Layers: 2, Heads: 1, VocabSize: 16}, 1},
		{"layers < 0", Config{Context: 8, Width: 16, Layers: -1, Heads: 2, VocabSize: 16}, 1},
		{"heads < 0", Config{Context: 8, Width: 16, Layers: 2, Heads: -1, VocabSize: 16}, 1},
		{"width not divisible by heads", Config{Context: 8, Width: 15, Layers: 2, Heads: 4, VocabSize: 16}, 1},
		{"vocab < 2", Config{Context: 8, Width: 16, Layers: 2, Heads: 2, VocabSize: 1}, 1},
	}

	for _, tc := range invalidConfigs {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewModel(tc.cfg, nil, tc.seed)
			if err == nil {
				t.Fatalf("expected error for %s, got nil", tc.name)
			}
		})
	}

	// Test deterministic initialization: same seed produces identical weights
	cfg := Config{Context: 16, Width: 16, Layers: 2, Heads: 2, VocabSize: 32}
	m1, err := NewModel(cfg, nil, 12345)
	if err != nil {
		t.Fatal(err)
	}
	m2, err := NewModel(cfg, nil, 12345)
	if err != nil {
		t.Fatal(err)
	}
	m3, err := NewModel(cfg, nil, 54321)
	if err != nil {
		t.Fatal(err)
	}

	p1 := m1.ParametersInfo()
	p2 := m2.ParametersInfo()
	p3 := m3.ParametersInfo()

	if len(p1) != len(p2) || len(p1) != len(p3) {
		t.Fatalf("parameter slice count mismatch: %d vs %d vs %d", len(p1), len(p2), len(p3))
	}

	for i := range p1 {
		for j := range p1[i].Values {
			if p1[i].Values[j] != p2[i].Values[j] {
				t.Fatalf("m1 and m2 weights differ at parameter %s[%d]: %f != %f", p1[i].Name, j, p1[i].Values[j], p2[i].Values[j])
			}
		}
	}

	// m1 and m3 should have different weights
	diffCount := 0
	for i := range p1 {
		for j := range p1[i].Values {
			if p1[i].Values[j] != p3[i].Values[j] {
				diffCount++
			}
		}
	}
	if diffCount == 0 {
		t.Fatal("expected models with different seeds to have differing initial weights")
	}

	// Verify no arbitrary caps: e.g. Context > 2048, Width > 512, Layers > 16 works if specified
	largeCfg := Config{Context: 4096, Width: 64, Layers: 1, Heads: 4, VocabSize: 32}
	if _, err := NewModel(largeCfg, nil, 1); err != nil {
		t.Fatalf("NewModel rejected large context (>2048): %v", err)
	}
}

func TestModelParameterCounts(t *testing.T) {
	testCases := []struct {
		cfg   Config
		vocab int
	}{
		{Config{Context: 8, Width: 4, Layers: 1, Heads: 1}, 16},
		{Config{Context: 16, Width: 8, Layers: 2, Heads: 2}, 32},
		{Config{Context: 64, Width: 16, Layers: 3, Heads: 4}, 64},
		{Config{Context: 256, Width: 64, Layers: 4, Heads: 4}, 256},
		{Config{Context: 128, Width: 32, Layers: 6, Heads: 8, VocabSize: 128}, 128},
	}

	for _, tc := range testCases {
		tc.cfg.VocabSize = tc.vocab
		model, err := NewModel(tc.cfg, nil, 42)
		if err != nil {
			t.Fatalf("NewModel failed for %+v: %v", tc.cfg, err)
		}

		expected := ExpectedParameterCount(tc.cfg, tc.vocab)
		actual := model.ParameterCount()
		if actual != expected {
			t.Fatalf("Parameter count mismatch for %+v: actual %d != expected %d", tc.cfg, actual, expected)
		}
	}
}

func TestModelForwardLogitsShapeAndFiniteness(t *testing.T) {
	cfg := Config{Context: 16, Width: 16, Layers: 2, Heads: 4, VocabSize: 32}
	model, err := NewModel(cfg, nil, 99)
	if err != nil {
		t.Fatal(err)
	}

	batch := [][]int{
		{1, 2, 3, 4, 5},
		{6, 7, 8, 9, 10},
		{11, 12, 13, 14, 15},
	}

	logitsTensor, err := model.ForwardLogits(batch)
	if err != nil {
		t.Fatalf("ForwardLogits failed: %v", err)
	}

	expectedShape := []int{3, 5, 32}
	if len(logitsTensor.Shape()) != 3 {
		t.Fatalf("expected 3D tensor, got shape %v", logitsTensor.Shape())
	}
	for i := range expectedShape {
		if logitsTensor.Shape()[i] != expectedShape[i] {
			t.Fatalf("expected shape %v, got %v", expectedShape, logitsTensor.Shape())
		}
	}

	// Check finiteness of all elements
	flatLogits := logitsTensor.Float32Slice()
	for i, val := range flatLogits {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			t.Fatalf("non-finite logit at index %d: %f", i, val)
		}
	}

	// Single sequence forward
	seqLogits, err := model.ForwardSequence([]int{1, 2, 3, 4, 5})
	if err != nil {
		t.Fatalf("ForwardSequence failed: %v", err)
	}
	if len(seqLogits.Shape()) != 2 || seqLogits.Shape()[0] != 5 || seqLogits.Shape()[1] != 32 {
		t.Fatalf("expected shape [5, 32], got %v", seqLogits.Shape())
	}

	// Error handling: mismatched sequence length
	badBatch := [][]int{
		{1, 2, 3},
		{1, 2},
	}
	if _, err := model.ForwardLogits(badBatch); err == nil {
		t.Fatal("expected error for mismatched sequence lengths in batch")
	}

	// Error handling: token out of bounds
	oobBatch := [][]int{
		{1, 2, 999},
	}
	if _, err := model.ForwardLogits(oobBatch); err == nil {
		t.Fatal("expected error for token out of vocabulary bounds")
	}
}

func TestCausalMaskingPreventsFutureLeakage(t *testing.T) {
	cfg := Config{Context: 16, Width: 16, Layers: 2, Heads: 2, VocabSize: 32}
	model, err := NewModel(cfg, nil, 123)
	if err != nil {
		t.Fatal(err)
	}

	// Baseline input: [5, 10, 15, 20]
	input1 := []int{5, 10, 15, 20}
	t1, err := model.ForwardSequence(input1)
	if err != nil {
		t.Fatal(err)
	}

	// Input 2: change future token at index 3: [5, 10, 15, 25]
	input2 := []int{5, 10, 15, 25}
	t2, err := model.ForwardSequence(input2)
	if err != nil {
		t.Fatal(err)
	}

	vocab := cfg.VocabSize
	slice1 := t1.Float32Slice()
	slice2 := t2.Float32Slice()

	// Check position 0, 1, 2 logits: MUST BE IDENTICAL
	for pos := 0; pos < 3; pos++ {
		for v := 0; v < vocab; v++ {
			idx := pos*vocab + v
			if slice1[idx] != slice2[idx] {
				t.Fatalf("causal leakage! position %d logit for token %d changed when modifying future token at index 3: %f vs %f", pos, v, slice1[idx], slice2[idx])
			}
		}
	}

	// Position 3 logits CAN and SHOULD differ because the token at position 3 is different
	diffAt3 := false
	for v := 0; v < vocab; v++ {
		idx := 3*vocab + v
		if slice1[idx] != slice2[idx] {
			diffAt3 = true
			break
		}
	}
	if !diffAt3 {
		t.Fatal("expected position 3 logits to differ when position 3 token changed")
	}

	// Input 3: change past token at index 1: [5, 9, 15, 20]
	input3 := []int{5, 9, 15, 20}
	t3, err := model.ForwardSequence(input3)
	if err != nil {
		t.Fatal(err)
	}
	slice3 := t3.Float32Slice()

	// Position 0 logits must remain identical (only depends on token at pos 0)
	for v := 0; v < vocab; v++ {
		if slice1[v] != slice3[v] {
			t.Fatalf("position 0 logit changed when modifying token at pos 1: %f vs %f", slice1[v], slice3[v])
		}
	}

	// Position 1 logits should differ (past token 1 changed)
	diffAt1 := false
	for v := 0; v < vocab; v++ {
		idx := 1*vocab + v
		if slice1[idx] != slice3[idx] {
			diffAt1 = true
			break
		}
	}
	if !diffAt1 {
		t.Fatal("expected position 1 logits to change when position 1 token changed")
	}
}

func TestForwardAutogradGraph(t *testing.T) {
	cfg := Config{Context: 8, Width: 8, Layers: 1, Heads: 2, VocabSize: 8}
	model, err := NewModel(cfg, nil, 42)
	if err != nil {
		t.Fatal(err)
	}

	tokenIDs := []int{1, 2, 3, 4}
	logitsVar, err := model.ForwardAutograd(tokenIDs)
	if err != nil {
		t.Fatalf("ForwardAutograd failed: %v", err)
	}

	if logitsVar.Shape()[0] != 4 || logitsVar.Shape()[1] != 8 {
		t.Fatalf("expected shape [4, 8], got %v", logitsVar.Shape())
	}

	// Verify Variables accessor
	vars := model.Variables()
	if len(vars) != len(model.parameters) {
		t.Fatalf("expected %d variables, got %d", len(model.parameters), len(vars))
	}
	for i, v := range vars {
		if v == nil || v.Data == nil {
			t.Fatalf("nil variable or data at index %d", i)
		}
	}

	// Verify reverse-mode automatic differentiation through the autograd computation graph
	lossVar, err := autograd.Sum(logitsVar)
	if err != nil {
		t.Fatalf("Sum failed: %v", err)
	}
	if err := lossVar.Backward(); err != nil {
		t.Fatalf("Backward on autograd graph failed: %v", err)
	}

	// Output weights must receive gradients via backward
	if model.output.weights.variable.Grad == nil {
		t.Fatal("expected output weights variable to have non-nil gradient after Backward")
	}
	hasNonZeroGrad := false
	for _, g := range model.output.weights.variable.Grad.Float32Slice() {
		if g != 0 {
			hasNonZeroGrad = true
			break
		}
	}
	if !hasNonZeroGrad {
		t.Fatal("expected non-zero gradient in output weights after backward pass")
	}
}

func TestTinyModelOverfitsSyntheticBatch(t *testing.T) {
	cfg := Config{Context: 8, Width: 16, Layers: 1, Heads: 2, VocabSize: 8}
	model, err := NewModel(cfg, nil, 777)
	if err != nil {
		t.Fatal(err)
	}

	// Synthetic batch of 2 sequences
	batchInputs := [][]int{
		{1, 2, 3, 4},
		{2, 3, 4, 5},
	}
	batchTargets := [][]int{
		{2, 3, 4, 1},
		{3, 4, 5, 2},
	}

	// Initial loss
	initialLoss, err := model.TrainBatchStep(batchInputs, batchTargets, 0.05)
	if err != nil {
		t.Fatal(err)
	}

	// Train for 80 steps
	var finalLoss float64
	for step := 0; step < 80; step++ {
		loss, err := model.TrainBatchStep(batchInputs, batchTargets, 0.05)
		if err != nil {
			t.Fatalf("step %d failed: %v", step, err)
		}
		finalLoss = loss
	}

	if finalLoss >= initialLoss*0.1 {
		t.Fatalf("expected overfit loss to drop below 10%% of initial loss (from %f to < %f), got %f", initialLoss, initialLoss*0.1, finalLoss)
	}

	// Verify next-token prediction accuracy on the synthetic batch is 100%
	for b := range batchInputs {
		tLogits, err := model.ForwardSequence(batchInputs[b])
		if err != nil {
			t.Fatal(err)
		}
		flat := tLogits.Float32Slice()
		vocab := cfg.VocabSize
		for pos := 0; pos < len(batchTargets[b]); pos++ {
			bestTok := 0
			bestLogit := float32(math.Inf(-1))
			for v := 0; v < vocab; v++ {
				logit := flat[pos*vocab+v]
				if logit > bestLogit {
					bestLogit = logit
					bestTok = v
				}
			}
			expectedTok := batchTargets[b][pos]
			if bestTok != expectedTok {
				t.Fatalf("sample %d pos %d: predicted token %d != target %d (final loss %f)", b, pos, bestTok, expectedTok, finalLoss)
			}
		}
	}
}

func BenchmarkTrainOneStep(b *testing.B) {
	text := strings.Repeat("benchmark text for the from scratch model. ", 8)
	tokenizer, err := TrainTokenizer([]string{text}, 256)
	if err != nil {
		b.Fatal(err)
	}
	model, err := NewModel(Config{Context: 8, Width: 8}, tokenizer, 5)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.Train([]string{text}, 1, 1, 0.001); err != nil {
			b.Fatal(err)
		}
	}
}

// TestKVCacheMatchesFullForward verifies that the KV-cache prefill+step path
// produces numerically identical logits to the full forward pass at every
// generated position. This ensures the O(1) generation path is a correct
// implementation of the full O(N²) causal attention forward pass.
func TestKVCacheMatchesFullForward(t *testing.T) {
	text := "kv cache equivalence check for all tokens"
	tokenizer, err := TrainTokenizer([]string{text + text}, 256)
	if err != nil {
		t.Fatal(err)
	}
	// Use context=16 so we stay comfortably within the window.
	model, err := NewModel(Config{Context: 16, Width: 8, Layers: 2, Heads: 2}, tokenizer, 77)
	if err != nil {
		t.Fatal(err)
	}

	promptIDs := tokenizer.Encode("kv cache")
	if len(promptIDs) == 0 {
		t.Fatal("prompt encoded to zero tokens")
	}

	backend := model.inferenceBackend()
	const genSteps = 4

	// ── Full forward reference ─────────────────────────────────────────────
	// At each step i, run the full forward on promptIDs[0..i] to get the
	// next-token prediction. This is the ground-truth O(N²) path.
	refLogits := make([][]float32, len(promptIDs)+genSteps)
	ids := append([]int(nil), promptIDs...)
	for i := 0; i < len(promptIDs)+genSteps; i++ {
		logits := backend.Forward(ids[:i+1])
		refLogits[i] = append([]float32(nil), logits...)
		if i >= len(promptIDs)-1 {
			// Extend with the argmax token to keep the sequence deterministic.
			best, bestLogit := 0, float32(math.Inf(-1))
			for id, v := range logits {
				if v > bestLogit {
					bestLogit = v
					best = id
				}
			}
			if i < len(promptIDs)+genSteps-1 {
				ids = append(ids, best)
			}
		}
	}

	// ── KV-cache path ──────────────────────────────────────────────────────
	// Prefill all prompt tokens, then step one token at a time, using the
	// same argmax choices as above to keep sequences aligned.
	kvCache := backend.NewKVCache()
	ids2 := append([]int(nil), promptIDs...)
	for i, id := range promptIDs {
		kvLogits := backend.ForwardStepKV(id, i, kvCache)

		// Compare logits at every prompt position.
		ref := refLogits[i]
		if len(kvLogits) != len(ref) {
			t.Fatalf("position %d: logit length mismatch %d vs %d", i, len(kvLogits), len(ref))
		}
		for j, v := range kvLogits {
			diff := math.Abs(float64(v - ref[j]))
			if diff > 1e-4 {
				t.Fatalf("position %d logit[%d]: KV=%.6f full=%.6f diff=%.6f", i, j, v, ref[j], diff)
			}
		}
	}
	pos := len(promptIDs)
	for step := 0; step < genSteps; step++ {
		refIdx := len(promptIDs) - 1 + step
		// Use the same argmax token as the reference sequence.
		best, bestLogit := 0, float32(math.Inf(-1))
		for id, v := range refLogits[refIdx] {
			if v > bestLogit {
				bestLogit = v
				best = id
			}
		}
		kvLogits := backend.ForwardStepKV(best, pos, kvCache)
		ids2 = append(ids2, best)
		pos++

		refIdx2 := len(promptIDs) + step
		ref := refLogits[refIdx2]
		if len(kvLogits) != len(ref) {
			t.Fatalf("gen step %d: logit length mismatch %d vs %d", step, len(kvLogits), len(ref))
		}
		for j, v := range kvLogits {
			diff := math.Abs(float64(v - ref[j]))
			if diff > 1e-4 {
				t.Fatalf("gen step %d logit[%d]: KV=%.6f full=%.6f diff=%.6f", step, j, v, ref[j], diff)
			}
		}
	}
	_ = ids2
}

