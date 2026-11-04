package assistant

import (
	"fmt"
	"io"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

var numCitationRegex = regexp.MustCompile(`\[(?:source\s*)?(\d+)\]|\((?:source\s*)?(\d+)\)`)

// draftStatesUncertainty checks if the draft indicates the answer wasn't found in reference notes.
func draftStatesUncertainty(draft string) bool {
	lower := strings.ToLower(draft)
	patterns := []string{
		"don't know",
		"do not know",
		"not contain",
		"no mention",
		"not mentioned",
		"no information",
		"cannot find",
		"could not find",
	}
	for _, p := range patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

// extractCitations finds all valid knowledge results cited in the draft text.
func extractCitations(draft string, results []KnowledgeResult) []KnowledgeResult {
	if len(results) == 0 || strings.TrimSpace(draft) == "" {
		return nil
	}

	lower := strings.ToLower(draft)
	citedMap := make(map[int]bool)

	// Check for numerical citations like [1], [2], (1)
	matches := numCitationRegex.FindAllStringSubmatch(lower, -1)
	for _, m := range matches {
		numStr := m[1]
		if numStr == "" {
			numStr = m[2]
		}
		idx, err := strconv.Atoi(numStr)
		if err == nil && idx >= 1 && idx <= len(results) {
			citedMap[idx-1] = true
		}
	}

	// Check for title and filename citations
	for i, r := range results {
		if citedMap[i] {
			continue
		}
		title := strings.TrimSpace(r.Title)
		if len(title) >= 3 && strings.Contains(lower, strings.ToLower(title)) {
			citedMap[i] = true
			continue
		}
		base := strings.TrimSpace(filepath.Base(r.Source))
		if len(base) >= 3 && strings.Contains(lower, strings.ToLower(base)) {
			citedMap[i] = true
			continue
		}
	}

	var cited []KnowledgeResult
	for i, r := range results {
		if citedMap[i] {
			cited = append(cited, r)
		}
	}
	return cited
}

// hasInvalidCitation detects citations to numbers out of range (e.g. [5] when only 2 sources provided).
func hasInvalidCitation(draft string, resultCount int) bool {
	lower := strings.ToLower(draft)
	matches := numCitationRegex.FindAllStringSubmatch(lower, -1)
	for _, m := range matches {
		numStr := m[1]
		if numStr == "" {
			numStr = m[2]
		}
		idx, err := strconv.Atoi(numStr)
		if err == nil && (idx < 1 || idx > resultCount) {
			return true
		}
	}
	return false
}

// draftCitesSources returns true if the draft references at least one retrieved source and has no invalid citations.
func draftCitesSources(draft string, results []KnowledgeResult) bool {
	if hasInvalidCitation(draft, len(results)) {
		return false
	}
	cited := extractCitations(draft, results)
	return len(cited) > 0
}

// printGroundedAnswer prints the model draft and cited/uncited sources clearly.
func printGroundedAnswer(draft string, results []KnowledgeResult, output io.Writer) {
	fmt.Fprintf(output, "Local model draft (check against sources):\n%s\n\n", strings.TrimSpace(draft))
	cited := extractCitations(draft, results)
	if len(cited) > 0 {
		fmt.Fprintln(output, "Sources:")
		for _, c := range cited {
			if c.Title != "" {
				fmt.Fprintf(output, "- %s — %s\n", c.Title, c.Source)
			} else {
				fmt.Fprintf(output, "- %s\n", c.Source)
			}
		}
	}
	// Also list other retrieved sources if any were not directly cited
	var uncited []KnowledgeResult
	for _, r := range results {
		isCited := false
		for _, c := range cited {
			if c.Source == r.Source && c.Title == r.Title {
				isCited = true
				break
			}
		}
		if !isCited {
			uncited = append(uncited, r)
		}
	}
	if len(uncited) > 0 {
		fmt.Fprintln(output, "\nOther retrieved sources:")
		for _, u := range uncited {
			if u.Title != "" {
				fmt.Fprintf(output, "- %s — %s\n", u.Title, u.Source)
			} else {
				fmt.Fprintf(output, "- %s\n", u.Source)
			}
		}
	}
}
