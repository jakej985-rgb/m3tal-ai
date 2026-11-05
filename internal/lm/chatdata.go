package lm

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// This file implements the chat instruction dataset format (issue: Stage 3
// instruction dataset prep). A chat dataset is JSONL — one Conversation
// per line, optionally with a "category" field:
//
//	{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"category":"plan"}
//
// Loading validates every record's schema, deduplicates exact-duplicate
// conversations, and reports reproducible stats. Splitting is
// deterministic (SHA-256 of canonical serialization) so train/validation
// partitions never overlap.

// ChatExample is a conversation plus optional metadata.
type ChatExample struct {
	Conversation
	Category string `json:"category,omitempty"`
}

// ChatDataStats summarizes a load/split for reports.
type ChatDataStats struct {
	RecordsIn    int64 `json:"records_in"`
	Valid        int64 `json:"valid"`
	Invalid      int64 `json:"invalid"`
	Duplicates   int64 `json:"duplicates"`
	Train        int64 `json:"train"`
	Validation   int64 `json:"validation"`
	AssistantTok int64 `json:"assistant_tokens"` // filled when tokenizer provided
}

// LoadChatJSONL reads and validates a JSONL chat dataset. malformed
// lines are counted and skipped (never silently accepted); exact
// duplicate conversations (canonical serialization hash) are dropped.
// Returns examples in file order.
func LoadChatJSONL(path string) ([]ChatExample, ChatDataStats, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, ChatDataStats{}, err
	}
	defer f.Close()

	var stats ChatDataStats
	var out []ChatExample
	seen := make(map[string]struct{})
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 16<<20)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		stats.RecordsIn++
		var ex ChatExample
		if err := json.Unmarshal([]byte(line), &ex); err != nil {
			stats.Invalid++
			continue
		}
		if err := ex.Conversation.Validate(); err != nil {
			stats.Invalid++
			continue
		}
		key := chatDedupKey(ex.Conversation)
		if _, dup := seen[key]; dup {
			stats.Duplicates++
			continue
		}
		seen[key] = struct{}{}
		stats.Valid++
		out = append(out, ex)
	}
	if err := scanner.Err(); err != nil {
		return nil, stats, fmt.Errorf("read %q: %w", path, err)
	}
	return out, stats, nil
}

// chatDedupKey hashes the canonical serialization of a conversation.
func chatDedupKey(c Conversation) string {
	h := sha256.New()
	for _, m := range c.Messages {
		h.Write([]byte(m.Role))
		h.Write([]byte{0})
		h.Write([]byte(m.Content))
		h.Write([]byte{0xff})
	}
	return hex.EncodeToString(h.Sum(nil))
}

// SplitChatExamples partitions examples deterministically: an example
// lands in validation when sha256(canonical)%1000 < frac*1000. Order is
// preserved within each partition.
func SplitChatExamples(examples []ChatExample, valFrac float64) (train, val []ChatExample) {
	for _, ex := range examples {
		h := sha256.Sum256([]byte(chatDedupKey(ex.Conversation)))
		u := uint64(h[0])<<56 | uint64(h[1])<<48 | uint64(h[2])<<40 | uint64(h[3])<<32 |
			uint64(h[4])<<24 | uint64(h[5])<<16 | uint64(h[6])<<8 | uint64(h[7])
		if float64(u)/float64(^uint64(0)) < valFrac {
			val = append(val, ex)
		} else {
			train = append(train, ex)
		}
	}
	return train, val
}

// WriteChatJSONL writes examples as JSONL (canonical field order via
// struct marshaling).
func WriteChatJSONL(path string, examples []ChatExample) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	for _, ex := range examples {
		data, err := json.Marshal(ex)
		if err != nil {
			return err
		}
		if _, err := w.Write(data); err != nil {
			return err
		}
		if err := w.WriteByte('\n'); err != nil {
			return err
		}
	}
	return w.Flush()
}

// ChatCategoryCounts returns per-category example counts for reports.
func ChatCategoryCounts(examples []ChatExample) map[string]int {
	counts := make(map[string]int)
	for _, ex := range examples {
		cat := ex.Category
		if cat == "" {
			cat = "(uncategorized)"
		}
		counts[cat]++
	}
	return counts
}

// SortedCategories returns category names in sorted order (report helper).
func SortedCategories(counts map[string]int) []string {
	out := make([]string, 0, len(counts))
	for k := range counts {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
