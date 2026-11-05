package lm

import (
	"os"
	"path/filepath"
	"testing"
)

func writeChatTestFile(t *testing.T, lines []string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "data.jsonl")
	content := ""
	for _, l := range lines {
		content += l + "\n"
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

const validChatLine = `{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}`

func TestLoadChatJSONL(t *testing.T) {
	p := writeChatTestFile(t, []string{
		validChatLine,
		validChatLine, // exact duplicate
		`{"messages":[{"role":"user","content":"   "},{"role":"assistant","content":"a"}]}`, // empty content
		`not json`,
		`{"messages":[{"role":"user","content":"q"}]}`, // no assistant
		``,
		`# comment`,
	})
	examples, stats, err := LoadChatJSONL(p)
	if err != nil {
		t.Fatal(err)
	}
	if stats.RecordsIn != 5 {
		t.Fatalf("records_in=%d want 5", stats.RecordsIn)
	}
	if stats.Valid != 1 || stats.Duplicates != 1 || stats.Invalid != 3 {
		t.Fatalf("stats=%+v", stats)
	}
	if len(examples) != 1 {
		t.Fatalf("examples=%d", len(examples))
	}
}

func TestSplitChatExamplesDeterministic(t *testing.T) {
	var exs []ChatExample
	for _, q := range []string{"a", "b", "c", "d", "e", "f", "g", "h"} {
		exs = append(exs, ChatExample{Conversation: Conversation{Messages: []ChatMessage{
			{Role: RoleUser, Content: q}, {Role: RoleAssistant, Content: "ans-" + q},
		}}})
	}
	tr1, va1 := SplitChatExamples(exs, 0.25)
	tr2, va2 := SplitChatExamples(exs, 0.25)
	if len(tr1)+len(va1) != len(exs) || len(tr2)+len(va2) != len(exs) {
		t.Fatal("split lost examples")
	}
	for i := range tr1 {
		if tr1[i].Serialize() != tr2[i].Serialize() {
			t.Fatal("split not deterministic")
		}
	}
	// No overlap.
	valKeys := map[string]bool{}
	for _, v := range va1 {
		valKeys[chatDedupKey(v.Conversation)] = true
	}
	for _, tr := range tr1 {
		if valKeys[chatDedupKey(tr.Conversation)] {
			t.Fatal("train/val overlap")
		}
	}
}

func TestWriteChatJSONLRoundTrip(t *testing.T) {
	exs := []ChatExample{{
		Conversation: Conversation{Messages: []ChatMessage{
			{Role: RoleUser, Content: "q"}, {Role: RoleAssistant, Content: "a"},
		}},
		Category: "test",
	}}
	p := filepath.Join(t.TempDir(), "out.jsonl")
	if err := WriteChatJSONL(p, exs); err != nil {
		t.Fatal(err)
	}
	back, stats, err := LoadChatJSONL(p)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Valid != 1 || len(back) != 1 || back[0].Category != "test" {
		t.Fatalf("round trip failed: %+v", back)
	}
}
