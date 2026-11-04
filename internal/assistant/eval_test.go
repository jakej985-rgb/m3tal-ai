package assistant

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunEvalWithoutLM(t *testing.T) {
	tempDir := t.TempDir()
	docPath := filepath.Join(tempDir, "M3tal-Hub-install.md")
	content := "# Install\n\nRun the install command to install M3TAL Hub.\n"
	if err := os.WriteFile(docPath, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	agent, err := New([]string{tempDir}, nil)
	if err != nil {
		t.Fatal(err)
	}

	metrics := agent.RunEval()
	if metrics.TotalQuestions != len(EvalSet) {
		t.Fatalf("total questions = %d, want %d", metrics.TotalQuestions, len(EvalSet))
	}
	if metrics.RetrievalRate <= 0 {
		t.Fatalf("retrieval rate = %f, want > 0", metrics.RetrievalRate)
	}
	if metrics.RAGRate != 0 {
		t.Fatalf("RAG rate without LM = %f, want 0", metrics.RAGRate)
	}

	report := PrintEvalReport(metrics)
	if !strings.Contains(report, "Eval Report") || !strings.Contains(report, "Retrieval rate:") {
		t.Fatalf("unexpected eval report content:\n%s", report)
	}
}

func TestRunEvalWithLM(t *testing.T) {
	tempDir := t.TempDir()
	docPath := filepath.Join(tempDir, "M3tal-Hub-install.md")
	content := "# Install\n\nRun the install command to install M3TAL Hub.\n"
	if err := os.WriteFile(docPath, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	agent, err := New([]string{tempDir}, nil)
	if err != nil {
		t.Fatal(err)
	}

	agent.SetLanguageModel(stubGenerator{text: "According to Install, run the installer."})
	if !agent.HasLanguageModel() {
		t.Fatal("expected HasLanguageModel() to be true")
	}

	metrics := agent.RunEval()
	if metrics.RAGRate <= 0 {
		t.Fatalf("expected RAGRate > 0 when LM is set, got %f", metrics.RAGRate)
	}
	if metrics.RAGValidUTF8Rate != 1.0 {
		t.Fatalf("expected RAGValidUTF8Rate == 1.0, got %f", metrics.RAGValidUTF8Rate)
	}
	if metrics.RAGCitationRate <= 0 {
		t.Fatalf("expected RAGCitationRate > 0, got %f", metrics.RAGCitationRate)
	}
}

func TestRunPlanEval(t *testing.T) {
	agent, err := New(nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	// No LM: valid plan rate should be 0
	metrics := agent.RunPlanEval()
	if metrics.TotalPrompts != len(MultiStepPrompts) {
		t.Fatalf("total prompts = %d, want %d", metrics.TotalPrompts, len(MultiStepPrompts))
	}
	if metrics.ValidPlanRate != 0 {
		t.Fatalf("valid plan rate without LM = %f, want 0", metrics.ValidPlanRate)
	}

	// LM proposing valid plans
	agent.SetLanguageModel(stubGenerator{text: `PLAN:{"steps":[{"tool":"stats"},{"tool":"health"}]}`})
	validMetrics := agent.RunPlanEval()
	if validMetrics.ValidPlanRate != 1.0 {
		t.Fatalf("valid plan rate with valid stub = %f, want 1.0", validMetrics.ValidPlanRate)
	}

	report := PrintPlanEvalReport(validMetrics)
	if !strings.Contains(report, "Plan Eval Report") || !strings.Contains(report, "Valid plan rate: 100.0%") {
		t.Fatalf("unexpected plan report:\n%s", report)
	}

	// LM proposing invalid JSON
	agent.SetLanguageModel(stubGenerator{text: `PLAN:not-valid-json`})
	invalidMetrics := agent.RunPlanEval()
	if invalidMetrics.ValidPlanRate != 0.0 {
		t.Fatalf("valid plan rate with invalid stub = %f, want 0", invalidMetrics.ValidPlanRate)
	}
	for _, res := range invalidMetrics.Results {
		if res.ValidPlan {
			t.Errorf("expected plan for %q to be invalid", res.Prompt)
		}
		if res.ParseError == "" {
			t.Errorf("expected ParseError for %q", res.Prompt)
		}
	}
}
