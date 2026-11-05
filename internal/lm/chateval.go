package lm

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
)

// This file implements the held-out conversational evaluation suite
// (issue: Stage 3 conversational eval). A suite is JSONL; each line is:
//
//	{"id":"...","category":"...","messages":[...],"max_tokens":32,"temperature":0.0}
//
// Entries ending with an assistant message are scored examples: loss is
// computed only over assistant tokens (same mask as SFT). Entries ending
// with a user/tool message are generation probes: the model generates a
// reply under the recorded decoding settings and automated checks are
// applied (empty output, stop-marker leak, max-token exhaustion).
//
// Automated metrics are heuristics only — the report explicitly marks
// them as such; relevance/coherence/adherence quality requires human
// review of the recorded generations.

// ChatEvalEntry is one suite record.
type ChatEvalEntry struct {
	ID          string        `json:"id"`
	Category    string        `json:"category,omitempty"`
	Messages    []ChatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
}

// IsProbe reports whether this entry is a generation probe (no final
// assistant turn to score).
func (e ChatEvalEntry) IsProbe() bool {
	if len(e.Messages) == 0 {
		return true
	}
	return e.Messages[len(e.Messages)-1].Role != RoleAssistant
}

// Conversation returns the entry as a Conversation.
func (e ChatEvalEntry) Conversation() Conversation {
	return Conversation{Messages: e.Messages}
}

// LoadChatEvalSuite reads a JSONL eval suite. Every line must be valid
// JSON with a non-empty id and at least one message; entries are NOT
// required to end with an assistant turn (probes end with user/tool).
func LoadChatEvalSuite(path string) ([]ChatEvalEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []ChatEvalEntry
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 16<<20)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		var e ChatEvalEntry
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		}
		if e.ID == "" {
			return nil, fmt.Errorf("%s:%d: missing id", path, lineNo)
		}
		if len(e.Messages) == 0 {
			return nil, fmt.Errorf("%s:%d: no messages", path, lineNo)
		}
		out = append(out, e)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s: empty eval suite", path)
	}
	return out, nil
}

// EvaluateChat returns mean next-token cross-entropy over assistant
// tokens only, across all examples, plus the masked-token count. Windows
// longer than context are truncated from the front (same rule as SFT).
func (m *Model) EvaluateChat(examples []ChatExample) (meanLoss float64, maskedTokens int64, err error) {
	if len(examples) == 0 {
		return 0, 0, fmt.Errorf("no chat examples")
	}
	context := m.Config.Context
	var totalLoss float64
	for _, ex := range examples {
		ids, idMask := ex.SerializedParts(m.Tokenizer)
		if len(ids) > context+1 {
			ids = ids[len(ids)-(context+1):]
			idMask = idMask[len(idMask)-(context+1):]
		}
		// Positions 1..len-1 are predictable targets; target t is scored
		// when idMask[t] is set.
		input := ids[:len(ids)-1]
		targets := ids[1:]
		mask := make([]bool, len(targets))
		any := false
		for i := range targets {
			mask[i] = idMask[i+1]
			if mask[i] {
				any = true
			}
		}
		if !any || len(input) < 1 {
			continue
		}
		cache := m.forward(input)
		vocab := len(cache.logits) / len(cache.ids)
		var exLoss float64
		var exCount int64
		for i, tgt := range targets {
			if !mask[i] {
				continue
			}
			exLoss += tokenLoss32(cache.logits[i*vocab:(i+1)*vocab], tgt)
			exCount++
		}
		if exCount > 0 {
			totalLoss += exLoss
			maskedTokens += exCount
		}
	}
	if maskedTokens == 0 {
		return 0, 0, fmt.Errorf("no assistant tokens to evaluate")
	}
	return totalLoss / float64(maskedTokens), maskedTokens, nil
}

// ChatProbeResult records one generation probe outcome.
type ChatProbeResult struct {
	ID            string `json:"id"`
	Category      string `json:"category,omitempty"`
	Prompt        string `json:"prompt"`   // serialized generation prompt
	Output        string `json:"output"`   // generated text (stop markers stripped)
	StopHit       bool   `json:"stop_hit"` // terminated via stop marker, not max tokens
	Empty         bool   `json:"empty"`
	StopLeaked    bool   `json:"stop_leaked"`
	AutomatedOnly bool   `json:"automated_only"` // true: heuristics only, human review needed
}

// ChatEvalReport is the machine-readable eval output.
type ChatEvalReport struct {
	TemplateVersion   string            `json:"template_version"`
	TokenizerHash     string            `json:"tokenizer_hash"`
	Checkpoint        string            `json:"checkpoint,omitempty"`
	AssistantLoss     float64           `json:"assistant_loss"`
	AssistantPPL      float64           `json:"assistant_perplexity"`
	MaskedTokens      int64             `json:"masked_tokens"`
	ScoredExamples    int               `json:"scored_examples"`
	Probes            []ChatProbeResult `json:"probes"`
	ProbeCount        int               `json:"probe_count"`
	EmptyOutputs      int               `json:"empty_outputs"`
	MaxTokenExhausted int               `json:"max_token_exhausted"`
	StopLeaks         int               `json:"stop_leaks"`
	DecodingSettings  string            `json:"decoding_settings"`
	Note              string            `json:"note"`
}

// RunChatEval executes the suite: scored examples contribute to masked
// assistant-token loss; probes are generated with the entry's decoding
// settings (defaults: greedy, 32 tokens, chat stop markers).
func (m *Model) RunChatEval(entries []ChatEvalEntry, checkpointPath string) (*ChatEvalReport, error) {
	var scored []ChatExample
	var probes []ChatEvalEntry
	for _, e := range entries {
		if e.IsProbe() {
			probes = append(probes, e)
		} else {
			scored = append(scored, ChatExample{Conversation: e.Conversation(), Category: e.Category})
		}
	}

	report := &ChatEvalReport{
		TemplateVersion: ChatTemplateVersion,
		TokenizerHash:   TokenizerFingerprint(m.Tokenizer),
		Checkpoint:      checkpointPath,
		Probes:          []ChatProbeResult{},
		Note:            "automated metrics are heuristics only; relevance/coherence/adherence require human review of recorded outputs",
	}

	if len(scored) > 0 {
		loss, masked, err := m.EvaluateChat(scored)
		if err != nil {
			return nil, err
		}
		report.AssistantLoss = loss
		report.AssistantPPL = expLoss(loss)
		report.MaskedTokens = masked
		report.ScoredExamples = len(scored)
	}

	for _, e := range probes {
		conv := e.Conversation()
		maxTok := e.MaxTokens
		if maxTok <= 0 {
			maxTok = 32
		}
		temp := e.Temperature // 0 = greedy default
		promptIDs := conv.EncodePrompt(m.Tokenizer)
		promptText := m.Tokenizer.Decode(promptIDs)
		opts := GenerateOptions{
			MaxTokens:   maxTok,
			Temperature: temp,
			Seed:        17,
			Stop:        StopMarkers(),
		}
		out, stopHit, err := m.generateCore(promptText, opts, nil)
		if err != nil {
			return nil, fmt.Errorf("probe %s: %w", e.ID, err)
		}
		res := ChatProbeResult{
			ID:            e.ID,
			Category:      e.Category,
			Prompt:        promptText,
			Output:        out,
			Empty:         strings.TrimSpace(out) == "",
			StopLeaked:    strings.Contains(out, imEnd),
			StopHit:       stopHit,
			AutomatedOnly: true,
		}
		if res.Empty {
			report.EmptyOutputs++
		}
		if res.StopLeaked {
			report.StopLeaks++
		}
		if !res.StopHit && !res.Empty {
			report.MaxTokenExhausted++
		}
		report.Probes = append(report.Probes, res)
	}
	report.ProbeCount = len(probes)
	return report, nil
}

// expLoss returns e^loss for perplexity reporting.
func expLoss(loss float64) float64 {
	if loss > 50 {
		return 0
	}
	return math.Exp(loss)
}
