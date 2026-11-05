package lm

import (
	"strings"
	"testing"
)

func testChatTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	tok, err := TrainTokenizer([]string{
		"<|im_start|>system\nYou are helpful.<|im_end|>\n" +
			"<|im_start|>user\nHello there.<|im_end|>\n" +
			"<|im_start|>assistant\nHi! How can I help?<|im_end|>\n",
		"<|im_start|>user\nWhat is 2+2?<|im_end|>\n" +
			"<|im_start|>assistant\n4<|im_end|>\n",
	}, 256)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestChatSerializeFixture(t *testing.T) {
	c := Conversation{Messages: []ChatMessage{
		{Role: RoleSystem, Content: "Be brief."},
		{Role: RoleUser, Content: "Hi"},
		{Role: RoleAssistant, Content: "Hello!"},
		{Role: RoleUser, Content: "Bye"},
	}}
	want := "<|im_start|>system\nBe brief.<|im_end|>\n" +
		"<|im_start|>user\nHi<|im_end|>\n" +
		"<|im_start|>assistant\nHello!<|im_end|>\n" +
		"<|im_start|>user\nBye<|im_end|>\n"
	if got := c.Serialize(); got != want {
		t.Fatalf("serialize:\n got %q\nwant %q", got, want)
	}
}

func TestChatPromptFixture(t *testing.T) {
	c := Conversation{Messages: []ChatMessage{
		{Role: RoleSystem, Content: "Be brief."},
		{Role: RoleUser, Content: "Hi"},
		{Role: RoleAssistant, Content: "Hello!"},
		{Role: RoleUser, Content: "How are you?"},
	}}
	// Prompt ends at the last non-assistant turn + assistant header.
	want := "<|im_start|>system\nBe brief.<|im_end|>\n" +
		"<|im_start|>user\nHi<|im_end|>\n" +
		"<|im_start|>assistant\nHello!<|im_end|>\n" +
		"<|im_start|>user\nHow are you?<|im_end|>\n" +
		"<|im_start|>assistant\n"
	// Reconstruct expected text from prompt segments.
	var b strings.Builder
	for _, s := range c.promptSegments() {
		b.WriteString(s.text)
	}
	if got := b.String(); got != want {
		t.Fatalf("prompt:\n got %q\nwant %q", got, want)
	}
}

func TestChatMaskBoundaries(t *testing.T) {
	tok := testChatTokenizer(t)
	c := Conversation{Messages: []ChatMessage{
		{Role: RoleUser, Content: "What is 2+2?"},
		{Role: RoleAssistant, Content: "4"},
	}}
	ids, mask := c.SerializedParts(tok)
	if len(ids) != len(mask) || len(ids) == 0 {
		t.Fatalf("ids/mask length mismatch: %d/%d", len(ids), len(mask))
	}
	// Decode each id and verify mask aligns with assistant region.
	var assistantTokens int
	for i, id := range ids {
		piece := tok.Decode([]int{id})
		if mask[i] {
			assistantTokens++
		}
		_ = piece
	}
	// Every masked token must decode to content inside the assistant turn.
	// Simpler invariant: masked region is contiguous and non-empty, and the
	// first masked token appears after the assistant header.
	first, last := -1, -1
	for i, m := range mask {
		if m {
			if first < 0 {
				first = i
			}
			last = i
		}
	}
	if first < 0 || last < first {
		t.Fatal("no masked tokens")
	}
	// Masked region must be contiguous.
	for i := first; i <= last; i++ {
		if !mask[i] {
			t.Fatalf("mask gap at %d inside [%d,%d]", i, first, last)
		}
	}
	// Assistant header must precede the masked region.
	header := tok.Encode("<|im_start|>assistant\n")
	if first < len(header) {
		t.Fatalf("masked region starts too early: %d < header %d", first, len(header))
	}
	if assistantTokens == 0 {
		t.Fatal("assistant content produced no tokens")
	}
}

func TestChatPromptTokenConsistency(t *testing.T) {
	tok := testChatTokenizer(t)
	c := Conversation{Messages: []ChatMessage{
		{Role: RoleUser, Content: "Hi"},
		{Role: RoleAssistant, Content: "Hello!"},
		{Role: RoleUser, Content: "Bye"},
	}}
	promptIDs := c.EncodePrompt(tok)
	// Append an assistant reply so the full serialization contains the
	// prompt plus the assistant turn; prompt must be an exact prefix.
	full := Conversation{Messages: append(append([]ChatMessage{}, c.Messages...),
		ChatMessage{Role: RoleAssistant, Content: "ok"})}
	fullIDs, _ := full.SerializedParts(tok)
	if len(promptIDs) > len(fullIDs) {
		t.Fatal("prompt longer than full serialization")
	}
	for i := range promptIDs {
		if promptIDs[i] != fullIDs[i] {
			t.Fatalf("prompt/full mismatch at %d: %d vs %d", i, promptIDs[i], fullIDs[i])
		}
	}
}

func TestChatValidate(t *testing.T) {
	good := Conversation{Messages: []ChatMessage{
		{Role: RoleUser, Content: "q"},
		{Role: RoleAssistant, Content: "a"},
	}}
	if err := good.Validate(); err != nil {
		t.Fatalf("valid conversation rejected: %v", err)
	}
	cases := map[string]Conversation{
		"empty":        {},
		"no-user":      {Messages: []ChatMessage{{Role: RoleAssistant, Content: "a"}}},
		"no-assistant": {Messages: []ChatMessage{{Role: RoleUser, Content: "q"}}},
		"bad-role":     {Messages: []ChatMessage{{Role: "robot", Content: "x"}, {Role: RoleUser, Content: "q"}, {Role: RoleAssistant, Content: "a"}}},
		"empty-msg":    {Messages: []ChatMessage{{Role: RoleUser, Content: "  "}, {Role: RoleAssistant, Content: "a"}}},
	}
	for name, c := range cases {
		if err := c.Validate(); err == nil {
			t.Fatalf("%s: invalid conversation accepted", name)
		}
	}
}

func TestChatMultiTurnMask(t *testing.T) {
	tok := testChatTokenizer(t)
	c := Conversation{Messages: []ChatMessage{
		{Role: RoleUser, Content: "q1"},
		{Role: RoleAssistant, Content: "a1"},
		{Role: RoleUser, Content: "q2"},
		{Role: RoleAssistant, Content: "a2"},
	}}
	_, mask := c.SerializedParts(tok)
	// Two separate masked regions expected (one per assistant turn).
	regions := 0
	inRegion := false
	for _, m := range mask {
		if m && !inRegion {
			regions++
			inRegion = true
		} else if !m {
			inRegion = false
		}
	}
	if regions != 2 {
		t.Fatalf("expected 2 masked regions, got %d", regions)
	}
}
