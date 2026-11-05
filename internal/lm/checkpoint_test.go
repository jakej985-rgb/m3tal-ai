package lm

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func smallModel(t *testing.T) (*Model, *Tokenizer) {
	t.Helper()
	tok, err := TrainTokenizer([]string{"alpha beta gamma delta. " + strings.Repeat("epsilon zeta. ", 20)}, 256)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(Config{Context: 8, Width: 8, Layers: 1, Heads: 2}, tok, 7)
	if err != nil {
		t.Fatal(err)
	}
	return model, tok
}

func parameterValues(m *Model) []float32 {
	var out []float32
	for _, p := range m.parameters {
		out = append(out, p.values...)
	}
	return out
}

func TestCheckpointRoundTripPreservesWeightsAndState(t *testing.T) {
	model, _ := smallModel(t)
	// Perturb weights and optimizer state so the round trip is meaningful.
	for i, p := range model.parameters {
		for j := range p.values {
			p.values[j] += float32(i*1000+j) * 1e-4
			p.first[j] = float32(j%7) * 0.01
			p.second[j] = float32(j%5) * 0.02
		}
	}
	model.step = 1234
	model.seed = 99

	path := filepath.Join(t.TempDir(), "ckpt.json")
	if err := model.SaveCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	got := parameterValues(loaded)
	want := parameterValues(model)
	if len(got) != len(want) {
		t.Fatalf("parameter count %d != %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("weight %d: %v != %v", i, got[i], want[i])
		}
	}
	if loaded.step != 1234 || loaded.seed != 99 {
		t.Fatalf("state not restored: step=%d seed=%d", loaded.step, loaded.seed)
	}
	// Optimizer moments must round-trip exactly.
	for i, p := range model.parameters {
		lp := loaded.parameters[i]
		for j := range p.first {
			if lp.first[j] != p.first[j] || lp.second[j] != p.second[j] {
				t.Fatalf("optimizer state mismatch at param %d idx %d", i, j)
			}
		}
	}
}

func TestCheckpointResumeIsDeterministic(t *testing.T) {
	model, _ := smallModel(t)
	texts := []string{strings.Repeat("the cat sat on the mat. ", 60)}
	tokens := EncodeCorpusTokenizes(model.Tokenizer, texts)
	// Context=8 → each window consumes 9 tokens; 12 windows = 108 tokens.
	const windowSize = 9
	const firstLegWindows = 12
	cfg := TrainConfig{MaxSteps: 18, LearningRate: 0.005, WarmupSteps: 2, GradClip: 1.0}
	// First leg sees only enough tokens for 12 windows, simulating an
	// interrupted run; the schedule horizon stays at 18 steps.
	if _, err := model.TrainStream(NewSliceSource(tokens[:firstLegWindows*windowSize]), cfg, nil); err != nil {
		t.Fatal(err)
	}
	if model.step != firstLegWindows {
		t.Fatalf("step = %d, want %d", model.step, firstLegWindows)
	}

	// Save, then resume on the remaining tokens with the same horizon.
	dir := t.TempDir()
	path := filepath.Join(dir, "ckpt.json")
	if err := model.SaveCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	resumed, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := resumed.TrainStream(NewSliceSource(tokens[firstLegWindows*windowSize:]), cfg, nil); err != nil {
		t.Fatal(err)
	}

	// Uninterrupted reference: same seed, 18 steps in one run.
	reference, _ := smallModel(t)
	if _, err := reference.TrainStream(NewSliceSource(tokens), cfg, nil); err != nil {
		t.Fatal(err)
	}

	got, want := parameterValues(resumed), parameterValues(reference)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("resume diverged at weight %d: %v vs %v", i, got[i], want[i])
		}
	}
}

func TestCheckpointRejectsCorruption(t *testing.T) {
	model, _ := smallModel(t)
	dir := t.TempDir()

	// Invalid JSON.
	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCheckpoint(bad); err == nil {
		t.Fatal("expected error for malformed checkpoint")
	}

	// Truncated weights.
	path := filepath.Join(dir, "ckpt.json")
	if err := model.SaveCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var state map[string]interface{}
	if err := json.Unmarshal(data, &state); err != nil {
		t.Fatal(err)
	}
	state["values"] = state["values"].([]interface{})[:10]
	trunc, _ := json.Marshal(state)
	if err := os.WriteFile(path, trunc, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCheckpoint(path); err == nil {
		t.Fatal("expected error for truncated weights")
	}

	// Mismatched optimizer state length.
	if err := model.SaveCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &state); err != nil {
		t.Fatal(err)
	}
	state["optimizer"] = map[string]interface{}{"first": []float64{1, 2, 3}, "second": []float64{1, 2, 3}}
	badOpt, _ := json.Marshal(state)
	if err := os.WriteFile(path, badOpt, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCheckpoint(path); err == nil {
		t.Fatal("expected error for mismatched optimizer state")
	}
}

func TestSafetensorsRoundTrip(t *testing.T) {
	model, _ := smallModel(t)
	for i, p := range model.parameters {
		for j := range p.values {
			p.values[j] = float32(math.Sin(float64(i*7919 + j)))
		}
	}
	path := filepath.Join(t.TempDir(), "weights.safetensors")
	if err := model.SaveSafetensors(path); err != nil {
		t.Fatal(err)
	}

	fresh, _ := smallModel(t)
	if err := fresh.LoadSafetensors(path); err != nil {
		t.Fatal(err)
	}
	got, want := parameterValues(fresh), parameterValues(model)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("safetensors weight %d: %v != %v", i, got[i], want[i])
		}
	}

	// Truncated file must fail safely.
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data[:len(data)-7], 0o600); err != nil {
		t.Fatal(err)
	}
	if err := fresh.LoadSafetensors(path); err == nil {
		t.Fatal("expected error for truncated safetensors file")
	}
}
