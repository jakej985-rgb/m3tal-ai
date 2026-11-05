package lm

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadChatEvalSuite(t *testing.T) {
	p := filepath.Join(t.TempDir(), "suite.jsonl")
	content := `{"id":"s1","messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}
{"id":"p1","messages":[{"role":"user","content":"probe?"}],"max_tokens":4}
# comment
{"id":"p2","messages":[{"role":"system","content":"s"},{"role":"user","content":"u"}]}
`
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	entries, err := LoadChatEvalSuite(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 3 {
		t.Fatalf("entries=%d", len(entries))
	}
	if entries[0].IsProbe() || !entries[1].IsProbe() || !entries[2].IsProbe() {
		t.Fatal("probe classification wrong")
	}
}

func TestRunChatEval(t *testing.T) {
	m := sftTestModel(t)
	entries := []ChatEvalEntry{
		{ID: "scored", Messages: []ChatMessage{
			{Role: RoleUser, Content: "What is 2+2?"},
			{Role: RoleAssistant, Content: "4"},
		}},
		{ID: "probe", Messages: []ChatMessage{
			{Role: RoleUser, Content: "Say hi"},
		}, MaxTokens: 4},
	}
	report, err := m.RunChatEval(entries, "test-ckpt")
	if err != nil {
		t.Fatal(err)
	}
	if report.ScoredExamples != 1 || report.MaskedTokens <= 0 {
		t.Fatalf("scored=%d masked=%d", report.ScoredExamples, report.MaskedTokens)
	}
	if report.AssistantLoss <= 0 {
		t.Fatalf("loss=%v", report.AssistantLoss)
	}
	if report.TemplateVersion != ChatTemplateVersion {
		t.Fatal("template version not recorded")
	}
	if len(report.Probes) != 1 || report.ProbeCount != 1 {
		t.Fatalf("probes=%d", report.ProbeCount)
	}
	if !report.Probes[0].AutomatedOnly {
		t.Fatal("probe not marked automated-only")
	}
	if report.Note == "" {
		t.Fatal("human-review note missing")
	}
}

func TestEvaluateChatMaskedOnly(t *testing.T) {
	m := sftTestModel(t)
	examples := []ChatExample{{
		Conversation: Conversation{Messages: []ChatMessage{
			{Role: RoleUser, Content: "q"},
			{Role: RoleAssistant, Content: "a"},
		}},
	}}
	loss, masked, err := m.EvaluateChat(examples)
	if err != nil {
		t.Fatal(err)
	}
	if masked <= 0 || loss <= 0 {
		t.Fatalf("loss=%v masked=%d", loss, masked)
	}
	// Masked count must equal number of assistant content+im_end tokens.
	_, idMask := examples[0].SerializedParts(m.Tokenizer)
	want := 0
	for _, mk := range idMask {
		if mk {
			want++
		}
	}
	// targets exclude first token; masked targets ≈ masked ids minus edge
	if masked > int64(want) {
		t.Fatalf("masked=%d exceeds serialized masked ids %d", masked, want)
	}
}

func TestChatEvalSuiteRejectsBad(t *testing.T) {
	p := filepath.Join(t.TempDir(), "bad.jsonl")
	if err := os.WriteFile(p, []byte(`{"messages":[{"role":"user","content":"q"}]}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadChatEvalSuite(p); err == nil || !strings.Contains(err.Error(), "missing id") {
		t.Fatalf("expected missing-id error, got %v", err)
	}
}
