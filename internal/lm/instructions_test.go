package lm

import (
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestDefaultInstructionSetValid(t *testing.T) {
	examples := DefaultInstructionSet()
	if len(examples) == 0 {
		t.Fatal("expected non-empty instruction set")
	}

	planCount := 0
	ragCount := 0

	for i, ex := range examples {
		if !utf8.ValidString(ex.Prompt) || !utf8.ValidString(ex.Completion) {
			t.Fatalf("example %d has invalid UTF-8", i)
		}
		if strings.TrimSpace(ex.Prompt) == "" || strings.TrimSpace(ex.Completion) == "" {
			t.Fatalf("example %d has empty prompt or completion", i)
		}

		switch ex.Category {
		case "plan":
			planCount++
			if err := ValidatePlanJSON(ex.Completion); err != nil {
				t.Fatalf("example %d has invalid plan JSON: %v\nCompletion: %s", i, err, ex.Completion)
			}
			if !strings.HasPrefix(ex.Prompt, "Plan the user's request") {
				t.Errorf("example %d prompt does not start with PlanPrompt prefix", i)
			}
		case "rag":
			ragCount++
			if !strings.HasPrefix(ex.Prompt, "Question: ") {
				t.Errorf("example %d prompt does not start with Question: prefix", i)
			}
			// Must either cite [1] or state uncertainty
			citesRef := strings.Contains(ex.Completion, "[1]")
			uncertain := strings.Contains(strings.ToLower(ex.Completion), "don't know")
			if !citesRef && !uncertain {
				t.Errorf("example %d RAG completion neither cites [1] nor states uncertainty: %q", i, ex.Completion)
			}
		default:
			t.Fatalf("example %d has unknown category %q", i, ex.Category)
		}
	}

	if planCount == 0 {
		t.Error("expected at least one plan example")
	}
	if ragCount == 0 {
		t.Error("expected at least one RAG example")
	}
}

func TestWriteAndLoadInstructionCorpus(t *testing.T) {
	tempDir := t.TempDir()
	path := filepath.Join(tempDir, "sub", "instructions.txt")

	examples := DefaultInstructionSet()
	if err := WriteInstructionCorpusFile(path, examples); err != nil {
		t.Fatal(err)
	}

	texts, err := LoadCorpus([]string{path})
	if err != nil {
		t.Fatal(err)
	}
	if len(texts) == 0 {
		t.Fatal("expected texts loaded from instruction corpus")
	}
	joined := strings.Join(texts, " ")
	if !strings.Contains(joined, "Plan the user's request") || !strings.Contains(joined, "Question: ") {
		t.Fatal("instruction corpus missing expected content")
	}
}

func TestTrainInstructionsLowersLoss(t *testing.T) {
	examples := DefaultInstructionSet()
	passages := FormatInstructionCorpus(examples)
	tokenizer, err := TrainTokenizer(passages, 280)
	if err != nil {
		t.Fatal(err)
	}

	model, err := NewModel(Config{Context: 128, Width: 16}, tokenizer, 42)
	if err != nil {
		t.Fatal(err)
	}

	losses, err := model.TrainInstructions(examples, 3, 20, 0.005)
	if err != nil {
		t.Fatal(err)
	}
	if len(losses) != 3 {
		t.Fatalf("expected 3 losses, got %d", len(losses))
	}
	if losses[2] >= losses[0] {
		t.Fatalf("instruction training did not lower loss: epoch 1 = %.4f, epoch 3 = %.4f", losses[0], losses[2])
	}
}

func TestTrainInstructionsGeneratesValidPlan(t *testing.T) {
	examples := DefaultInstructionSet()
	passages := FormatInstructionCorpus(examples)
	tokenizer, err := TrainTokenizer(passages, 320)
	if err != nil {
		t.Fatal(err)
	}

	model, err := NewModel(Config{Context: 256, Width: 16, Layers: 2, Heads: 2}, tokenizer, 42)
	if err != nil {
		t.Fatal(err)
	}

	prompt := PlanPrompt("show stats and health")
	promptIDs := tokenizer.Encode(prompt)
	if len(promptIDs) > model.Config.Context {
		t.Fatalf("prompt has %d tokens, exceeding context %d", len(promptIDs), model.Config.Context)
	}

	epochs, steps := 2, 10
	if !testing.Short() {
		epochs, steps = 3, 20
	}
	losses, err := model.TrainInstructions(examples, epochs, steps, 0.005)
	if err != nil {
		t.Fatal(err)
	}
	if losses[len(losses)-1] >= losses[0] {
		t.Fatalf("losses did not decrease: %v", losses)
	}

	out, err := model.Generate(prompt, 20, 0.2, 42)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) == 0 {
		t.Fatal("expected non-empty generated output")
	}
}
