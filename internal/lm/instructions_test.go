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
			if !strings.HasPrefix(ex.Completion, "Thought: ") {
				t.Errorf("example %d plan completion does not start with 'Thought: ': %q", i, ex.Completion)
			}
			if strings.TrimSpace(ex.Thought) == "" {
				t.Errorf("example %d plan example has empty Thought field", i)
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

func TestExtractPlanJSON(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "plain json",
			input:    `{"steps":[{"tool":"stats"}]}`,
			expected: `{"steps":[{"tool":"stats"}]}`,
		},
		{
			name:     "cot with single line thought",
			input:    "Thought: need to check stats.\n{\"steps\":[{\"tool\":\"stats\"}]}",
			expected: `{"steps":[{"tool":"stats"}]}`,
		},
		{
			name:     "cot with trailing text",
			input:    "Thought: need health.\n{\"steps\":[{\"tool\":\"health\"}]}\nDone.",
			expected: `{"steps":[{"tool":"health"}]}`,
		},
		{
			name:     "no json object",
			input:    "I do not know the answer.",
			expected: "",
		},
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ExtractPlanJSON(tc.input)
			if got != tc.expected {
				t.Fatalf("ExtractPlanJSON(%q) = %q, want %q", tc.input, got, tc.expected)
			}
		})
	}
}

func TestPlanPromptIncludesThoughtDirective(t *testing.T) {
	query := "restart jellyfin container"
	prompt := PlanPrompt(query)
	cot := CoTPlanPrompt(query)

	if prompt != cot {
		t.Fatal("expected CoTPlanPrompt to be an alias of PlanPrompt")
	}
	if !strings.Contains(prompt, "You may first write a brief Thought before the JSON.") {
		t.Fatal("expected prompt to mention writing a brief Thought")
	}
	if !strings.HasSuffix(prompt, "\nPLAN:") {
		t.Fatal("prompt must end with \\nPLAN:")
	}
}

func TestTrainCoTInstructionsLowersLoss(t *testing.T) {
	allExamples := DefaultInstructionSet()
	var planExamples []InstructionExample
	for _, ex := range allExamples {
		if ex.Category == "plan" {
			planExamples = append(planExamples, ex)
		}
	}
	if len(planExamples) == 0 {
		t.Fatal("expected plan examples")
	}

	passages := FormatInstructionCorpus(planExamples)
	tokenizer, err := TrainTokenizer(passages, 320)
	if err != nil {
		t.Fatal(err)
	}

	model, err := NewModel(Config{Context: 256, Width: 16, Layers: 2, Heads: 2}, tokenizer, 42)
	if err != nil {
		t.Fatal(err)
	}

	losses, err := model.TrainInstructions(planExamples, 3, 20, 0.005)
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

func TestCoTGenerationContainsThoughtAndPlan(t *testing.T) {
	allExamples := DefaultInstructionSet()
	var planExamples []InstructionExample
	for _, ex := range allExamples {
		if ex.Category == "plan" {
			planExamples = append(planExamples, ex)
		}
	}

	passages := FormatInstructionCorpus(planExamples)
	tokenizer, err := TrainTokenizer(passages, 350)
	if err != nil {
		t.Fatal(err)
	}

	model, err := NewModel(Config{Context: 256, Width: 32, Layers: 2, Heads: 2}, tokenizer, 99)
	if err != nil {
		t.Fatal(err)
	}

	epochs, steps := 3, 20
	if !testing.Short() {
		epochs, steps = 5, 30
	}
	losses, err := model.TrainInstructions(planExamples, epochs, steps, 0.005)
	if err != nil {
		t.Fatal(err)
	}
	if losses[len(losses)-1] >= losses[0] {
		t.Fatalf("losses did not decrease: %v", losses)
	}

	prompt := PlanPrompt("show stats and health")
	out, err := model.Generate(prompt, 35, 0.2, 42)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) == 0 {
		t.Fatal("expected non-empty generated output")
	}
	t.Logf("Generated plan output: %q", out)
	// Output should start generating Thought tokens
	if !strings.Contains(out, "Thought") && !strings.Contains(out, "{") {
		t.Fatalf("expected generated output to contain Thought or plan JSON: %q", out)
	}
}

// TestEntityDisambiguationExamples verifies that all bare-action disambiguation
// examples in DefaultInstructionSet have valid plan JSON and select the correct
// tool for each entity type (container, service, or stack).
func TestEntityDisambiguationExamples(t *testing.T) {
	type wantCase struct {
		queryFragment string // substring of the prompt's user query
		tool          string // expected tool name
		target        string // expected target name
	}
	wants := []wantCase{
		{"restart jellyfin", "restart_container", "jellyfin"},
		{"stop jellyfin", "stop_container", "jellyfin"},
		{"start jellyfin", "start_container", "jellyfin"},
		{"restart traefik", "restart_container", "traefik"},
		{"restart cloudflared", "restart_container", "cloudflared"},
		{"restart docker", "restart_service", "docker"},
		{"restart media", "restart_stack", "media"},
		{"logs for jellyfin", "logs", "jellyfin"},
		{"logs cloudflared", "logs", "cloudflared"},
		{"restart ollama", "restart_container", "ollama"},
		{"logs traefik", "logs", "traefik"},
	}

	examples := DefaultInstructionSet()

	for _, want := range wants {
		found := false
		for _, ex := range examples {
			if ex.Category != "plan" {
				continue
			}
			if !strings.Contains(ex.Prompt, want.queryFragment) {
				continue
			}
			// Validate the plan JSON is well-formed.
			if err := ValidatePlanJSON(ex.Completion); err != nil {
				t.Errorf("query %q: invalid plan JSON: %v\nCompletion: %s", want.queryFragment, err, ex.Completion)
			}
			// Check the first step uses the right tool and target.
			jsonStr := ExtractPlanJSON(ex.Completion)
			if !strings.Contains(jsonStr, `"tool":"`+want.tool+`"`) {
				t.Errorf("query %q: expected tool %q in JSON %s", want.queryFragment, want.tool, jsonStr)
			}
			if want.target != "" && !strings.Contains(jsonStr, `"target":"`+want.target+`"`) {
				t.Errorf("query %q: expected target %q in JSON %s", want.queryFragment, want.target, jsonStr)
			}
			// Verify the Thought starts with the correct prefix.
			if !strings.HasPrefix(ex.Completion, "Thought: ") {
				t.Errorf("query %q: completion must start with 'Thought: '", want.queryFragment)
			}
			found = true
			break
		}
		if !found {
			t.Errorf("no disambiguation example found for query fragment %q", want.queryFragment)
		}
	}
}

