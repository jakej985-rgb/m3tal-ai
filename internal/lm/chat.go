package lm

import (
	"fmt"
	"strings"
)

// This file implements the chat conversation format and serialization
// (issue: Stage 3 chat template + assistant-only loss masks).
//
// Template "m3tal.chatml.v1" serializes a conversation as a sequence of
// segments, each tokenized independently so mask boundaries are exact:
//
//	"<|im_start|>" + role + "\n"          (prompt segment, masked)
//	content                             (masked unless role=assistant)
//	"<|im_end|>\n"                      (masked unless role=assistant)
//
// Loss masking marks only assistant message content tokens (plus the
// assistant turn's <|im_end|>) as trainable; all template markers and
// non-assistant content are masked out. Generation prompts serialize all
// turns through the last non-assistant message plus the assistant header
// "<|im_start|>assistant\n", and stop when the <|im_end|> marker is
// produced. Segment-wise encoding is used for BOTH training and prompts
// so the token streams are identical at the boundary.

const (
	// ChatTemplateVersion identifies the serialization format recorded
	// in checkpoints and eval reports.
	ChatTemplateVersion = "m3tal.chatml.v1"

	imStart = "<|im_start|>"
	imEnd   = "<|im_end|>"
)

// ChatRole is a conversation participant role.
type ChatRole string

const (
	RoleSystem    ChatRole = "system"
	RoleUser      ChatRole = "user"
	RoleAssistant ChatRole = "assistant"
	RoleTool      ChatRole = "tool"
)

// ChatMessage is one turn in a conversation.
type ChatMessage struct {
	Role    ChatRole `json:"role"`
	Content string   `json:"content"`
}

// Conversation is an ordered list of chat messages.
type Conversation struct {
	Messages []ChatMessage `json:"messages"`
}

// Validate checks structural rules: known roles, non-empty content, at
// least one user and one assistant turn.
func (c Conversation) Validate() error {
	if len(c.Messages) == 0 {
		return fmt.Errorf("conversation has no messages")
	}
	users, assistants := 0, 0
	for i, m := range c.Messages {
		switch m.Role {
		case RoleSystem, RoleUser, RoleAssistant, RoleTool:
		default:
			return fmt.Errorf("message %d: unknown role %q", i, m.Role)
		}
		if strings.TrimSpace(m.Content) == "" {
			return fmt.Errorf("message %d (%s): empty content", i, m.Role)
		}
		switch m.Role {
		case RoleUser:
			users++
		case RoleAssistant:
			assistants++
		}
	}
	if users == 0 {
		return fmt.Errorf("conversation has no user message")
	}
	if assistants == 0 {
		return fmt.Errorf("conversation has no assistant message")
	}
	return nil
}

// chatSegment is one tokenized piece of the serialized conversation.
type chatSegment struct {
	text string
	loss bool // true when this segment contributes to SFT loss
}

// segments returns the tokenization segments for the full conversation.
func (c Conversation) segments() []chatSegment {
	var segs []chatSegment
	for _, m := range c.Messages {
		segs = append(segs,
			chatSegment{imStart + string(m.Role) + "\n", false},
			chatSegment{m.Content, m.Role == RoleAssistant},
			chatSegment{imEnd + "\n", m.Role == RoleAssistant},
		)
	}
	return segs
}

// promptSegments returns segments for a generation prompt: all turns
// through the last non-assistant message, then the assistant header.
func (c Conversation) promptSegments() []chatSegment {
	last := -1
	for i, m := range c.Messages {
		if m.Role != RoleAssistant {
			last = i
		}
	}
	if last < 0 {
		last = len(c.Messages) - 1
	}
	var segs []chatSegment
	for _, m := range c.Messages[:last+1] {
		segs = append(segs,
			chatSegment{imStart + string(m.Role) + "\n", false},
			chatSegment{m.Content, false},
			chatSegment{imEnd + "\n", false},
		)
	}
	segs = append(segs, chatSegment{imStart + string(RoleAssistant) + "\n", false})
	return segs
}

// encodeSegments tokenizes each segment independently and concatenates,
// returning ids plus a per-id loss mask.
func encodeSegments(t *Tokenizer, segs []chatSegment) (ids []int, mask []bool) {
	for _, s := range segs {
		enc := t.Encode(s.text)
		ids = append(ids, enc...)
		for range enc {
			mask = append(mask, s.loss)
		}
	}
	return ids, mask
}

// SerializedParts returns token ids for the full conversation plus a
// per-id mask that is true only on assistant content and the assistant
// turn's <|im_end|> — the positions where SFT loss applies.
func (c Conversation) SerializedParts(t *Tokenizer) (ids []int, mask []bool) {
	return encodeSegments(t, c.segments())
}

// EncodePrompt returns token ids for the generation prompt (through the
// assistant header), using the same segment-wise encoding as training.
func (c Conversation) EncodePrompt(t *Tokenizer) []int {
	ids, _ := encodeSegments(t, c.promptSegments())
	return ids
}

// Serialize renders the full conversation as text (debug/fixture use).
func (c Conversation) Serialize() string {
	var b strings.Builder
	for _, s := range c.segments() {
		b.WriteString(s.text)
	}
	return b.String()
}

// StopMarkers returns the marker strings that terminate generation.
// Detection runs on decoded text.
func StopMarkers() []string { return []string{imEnd} }
