package lm

import (
	"errors"
	"strings"
	"testing"
)

func sampleTestModel(t *testing.T) *Model {
	t.Helper()
	tok := testChatTokenizer(t)
	cfg := Config{Context: 32, Width: 32, Layers: 2, Heads: 4}
	m, err := NewModel(cfg, tok, 1)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestGenerateGreedyDeterministic(t *testing.T) {
	m := sampleTestModel(t)
	prompt := "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
	opts := GenerateOptions{MaxTokens: 8, Temperature: 0, Seed: 1}
	a, err := m.GenerateWithOptions(prompt, opts)
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.GenerateWithOptions(prompt, opts)
	if err != nil {
		t.Fatal(err)
	}
	if a != b {
		t.Fatalf("greedy not deterministic:\n%q\n%q", a, b)
	}
}

func TestGenerateSeededDeterministic(t *testing.T) {
	m := sampleTestModel(t)
	prompt := "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
	opts := GenerateOptions{MaxTokens: 8, Temperature: 0.8, TopP: 0.9, Seed: 42}
	a, _ := m.GenerateWithOptions(prompt, opts)
	b, _ := m.GenerateWithOptions(prompt, opts)
	if a != b {
		t.Fatalf("seeded sampling not deterministic:\n%q\n%q", a, b)
	}
}

func TestGenerateTopKTopP(t *testing.T) {
	m := sampleTestModel(t)
	prompt := "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
	for _, opts := range []GenerateOptions{
		{MaxTokens: 6, Temperature: 1.0, TopK: 1, Seed: 5},
		{MaxTokens: 6, Temperature: 1.0, TopP: 0.5, Seed: 5},
		{MaxTokens: 6, Temperature: 1.0, TopK: 3, TopP: 0.95, Seed: 5},
	} {
		out, err := m.GenerateWithOptions(prompt, opts)
		if err != nil {
			t.Fatalf("opts %+v: %v", opts, err)
		}
		if strings.Contains(out, "\x00") {
			t.Fatalf("control char in output %q", out)
		}
	}
}

func TestGenerateStopMarker(t *testing.T) {
	m := sampleTestModel(t)
	prompt := "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
	out, err := m.GenerateWithOptions(prompt, GenerateOptions{
		MaxTokens: 50, Temperature: 0.7, Seed: 9, Stop: StopMarkers(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out, "<|im_end|>") {
		t.Fatalf("stop marker leaked into output: %q", out)
	}
	// Without stop markers the raw generation may contain the marker text;
	// with markers it must terminate early. Just verify no marker present.
}

func TestGenerateStream(t *testing.T) {
	m := sampleTestModel(t)
	prompt := "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
	var streamed strings.Builder
	full, err := m.GenerateStream(prompt, GenerateOptions{
		MaxTokens: 6, Temperature: 0.5, Seed: 3,
	}, func(delta string) error {
		streamed.WriteString(delta)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if streamed.String() != full {
		t.Fatalf("streamed %q != final %q", streamed.String(), full)
	}
}

func TestGenerateStreamCancel(t *testing.T) {
	m := sampleTestModel(t)
	prompt := "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
	sentinel := errors.New("cancel")
	calls := 0
	out, err := m.GenerateStream(prompt, GenerateOptions{
		MaxTokens: 20, Temperature: 0.8, Seed: 3,
	}, func(delta string) error {
		calls++
		if calls >= 2 {
			return sentinel
		}
		return nil
	})
	if err == nil || !strings.Contains(err.Error(), "stream cancelled") {
		t.Fatalf("expected cancellation error, got %v", err)
	}
	if !errors.Is(err, sentinel) {
		t.Fatalf("sentinel not wrapped: %v", err)
	}
	_ = out // partial output returned alongside error
}

func TestGenerateOptionValidation(t *testing.T) {
	m := sampleTestModel(t)
	for _, opts := range []GenerateOptions{
		{MaxTokens: -1},
		{MaxTokens: 1, Temperature: -0.5},
		{MaxTokens: 1, Temperature: 1, TopK: -1},
		{MaxTokens: 1, Temperature: 1, TopP: 1.5},
		{MaxTokens: 1, Temperature: 1, TopP: -0.1},
	} {
		if _, err := m.GenerateWithOptions("hi", opts); err == nil {
			t.Fatalf("invalid opts accepted: %+v", opts)
		}
	}
}

func TestGenerateContextLimit(t *testing.T) {
	// Context exhaustion path: prompt longer than context must still
	// generate (full-forward fallback) without error.
	m := sampleTestModel(t)
	longPrompt := strings.Repeat("word ", 200) // >> 32-token context
	out, err := m.GenerateWithOptions(longPrompt, GenerateOptions{
		MaxTokens: 4, Temperature: 0, Seed: 1,
	})
	if err != nil {
		t.Fatalf("context-limit generation failed: %v", err)
	}
	_ = out
}
