package assistant

import (
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode"
)

type document struct {
	source string
	title  string
	text   string
	terms  map[string]int
}

// KnowledgeBase indexes local M3TAL documentation for lightweight retrieval.
type KnowledgeBase struct {
	documents    []document
	documentFreq map[string]int
	apps         []HubApp
}

// HubApp is an application registered in the M3TAL-Hub manifest.
type HubApp struct {
	ID          string
	Name        string
	Repo        string
	Description string
	Status      string
}

// LoadKnowledgeBase indexes supported text documents found at files or roots.
func LoadKnowledgeBase(roots []string) (*KnowledgeBase, error) {
	base := &KnowledgeBase{}
	for _, root := range roots {
		info, err := os.Stat(root)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, fmt.Errorf("read knowledge path %q: %w", root, err)
		}
		if !info.IsDir() {
			if supportedText(root) {
				if err := base.addFile(root); err != nil {
					return nil, err
				}
			}
			continue
		}
		err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if entry.IsDir() && path != root && ignoredDirectory(entry.Name()) {
				return filepath.SkipDir
			}
			if entry.IsDir() || !supportedText(path) {
				return nil
			}
			info, err := entry.Info()
			if err != nil {
				return err
			}
			if info.Size() > 1024*1024 {
				return nil
			}
			return base.addFile(path)
		})
		if err != nil {
			return nil, fmt.Errorf("index knowledge path %q: %w", root, err)
		}
	}
	base.documentFreq = make(map[string]int)
	for index := range base.documents {
		base.documents[index].terms = termCounts(base.documents[index].text)
		for term := range base.documents[index].terms {
			base.documentFreq[term]++
		}
	}
	return base, nil
}

// Search returns the top matching document excerpts and their relative scores.
func (base *KnowledgeBase) Search(query string, limit int) []KnowledgeResult {
	if base == nil || limit <= 0 {
		return nil
	}
	queryTerms := uniqueTerms(query)
	if len(queryTerms) == 0 {
		return nil
	}
	type scored struct {
		result KnowledgeResult
	}
	sourceResults := make(map[string]KnowledgeResult)
	for _, doc := range base.documents {
		score := 0.0
		for _, term := range queryTerms {
			frequency := doc.terms[term]
			if frequency == 0 {
				continue
			}
			documentFrequency := base.documentFreq[term]
			idf := math.Log(1 + float64(len(base.documents)-documentFrequency+1)/float64(documentFrequency+1))
			score += idf * float64(frequency) / (float64(frequency) + 1.2)
		}
		if strings.EqualFold(filepath.Base(doc.source), "README.md") {
			score *= 0.55
		}
		if score > 0 {
			kind := "Documentation"
			if sourceFile(doc.source) {
				kind = "Source"
			}
			if previous, found := sourceResults[doc.source]; !found || score > previous.Score {
				freshness := FreshnessAssessment{}
				if kind == "Documentation" {
					freshness = AssessFreshness(doc.text, time.Now())
				}
				sourceResults[doc.source] = KnowledgeResult{Source: doc.source, Title: doc.title, Excerpt: excerpt(doc.text, queryTerms), Score: score, Kind: kind, Freshness: freshness}
			}
		}
	}
	scores := make([]scored, 0, len(sourceResults))
	for _, result := range sourceResults {
		scores = append(scores, scored{result: result})
	}
	sort.SliceStable(scores, func(i, j int) bool { return scores[i].result.Score > scores[j].result.Score })
	if len(scores) > limit {
		scores = scores[:limit]
	}
	results := make([]KnowledgeResult, len(scores))
	for i, match := range scores {
		results[i] = match.result
	}
	return results
}

// Apps returns the applications parsed from the Hub manifest.
func (base *KnowledgeBase) Apps() []HubApp {
	if base == nil {
		return nil
	}
	return append([]HubApp(nil), base.apps...)
}

// KnowledgeResult is a retrieved documentation excerpt.
type KnowledgeResult struct {
	Source    string
	Title     string
	Excerpt   string
	Score     float64
	Kind      string
	Freshness FreshnessAssessment
}

func (base *KnowledgeBase) addFile(path string) error {
	contents, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read document %q: %w", path, err)
	}
	if filepath.Base(path) == "manifest.yml" || filepath.Base(path) == "manifest.yaml" {
		base.apps = append(base.apps, parseHubManifest(string(contents))...)
	}
	for _, chunk := range splitDocument(string(contents)) {
		if strings.TrimSpace(chunk.text) == "" {
			continue
		}
		base.documents = append(base.documents, document{source: filepath.Clean(path), title: chunk.title, text: chunk.text})
	}
	return nil
}

func parseHubManifest(contents string) []HubApp {
	var applications []HubApp
	var current *HubApp
	flush := func() {
		if current != nil && current.ID != "" {
			if current.Name == "" {
				current.Name = current.ID
			}
			applications = append(applications, *current)
		}
	}
	for _, line := range strings.Split(contents, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "- id:") {
			flush()
			current = &HubApp{ID: unquote(strings.TrimSpace(strings.TrimPrefix(trimmed, "- id:")))}
			continue
		}
		if current == nil {
			continue
		}
		key, value, found := strings.Cut(trimmed, ":")
		if !found {
			continue
		}
		value = unquote(strings.TrimSpace(value))
		switch key {
		case "name":
			current.Name = value
		case "repo":
			current.Repo = value
		case "description":
			current.Description = value
		case "status":
			current.Status = value
		}
	}
	flush()
	return applications
}

func unquote(value string) string {
	return strings.Trim(value, "\"'")
}

type documentChunk struct{ title, text string }

func splitDocument(content string) []documentChunk {
	lines := strings.Split(content, "\n")
	var chunks []documentChunk
	var current []string
	title := ""
	flush := func() {
		text := strings.TrimSpace(strings.Join(current, "\n"))
		if text != "" {
			chunks = append(chunks, documentChunk{title: title, text: text})
		}
		current = nil
	}
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		isHeading := strings.HasPrefix(trimmed, "#")
		isRegistryEntry := strings.HasPrefix(trimmed, "- id:")
		if (isHeading || isRegistryEntry) && len(current) > 0 {
			flush()
		}
		if isHeading {
			title = strings.TrimSpace(strings.TrimLeft(trimmed, "#"))
		}
		if isRegistryEntry {
			title = strings.TrimSpace(strings.TrimPrefix(trimmed, "- id:"))
		}
		current = append(current, line)
		if len(current) >= 80 {
			flush()
		}
	}
	flush()
	return chunks
}

func supportedText(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".md", ".txt", ".yml", ".yaml", ".json", ".go", ".dart", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".py", ".sh", ".toml", ".xml":
		return true
	default:
		return false
	}
}

func sourceFile(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".go", ".dart", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".py", ".sh":
		return true
	default:
		return false
	}
}

func ignoredDirectory(name string) bool {
	if strings.HasPrefix(name, ".") {
		return true
	}
	switch name {
	case "node_modules", "vendor", "dist", "build", "target", "coverage", ".dart_tool", "out", "release", "tmp", "_site", ".next", ".venv", "venv", "env", "__pycache__", "site-packages", "Pods":
		return true
	default:
		return false
	}
}

func uniqueTerms(text string) []string {
	counts := termCounts(text)
	terms := make([]string, 0, len(counts))
	for term := range counts {
		if !stopWords[term] {
			terms = append(terms, term)
		}
	}
	return terms
}

var stopWords = map[string]bool{
	"a": true, "an": true, "and": true, "are": true, "do": true, "does": true,
	"for": true, "how": true, "i": true, "in": true, "is": true, "it": true,
	"me": true, "of": true, "on": true, "the": true, "to": true, "what": true,
	"where": true, "which": true, "with": true, "you": true,
}

func termCounts(text string) map[string]int {
	counts := make(map[string]int)
	for _, term := range features(splitIdentifiers(text)) {
		if len(term) > 1 {
			counts[term]++
		}
	}
	return counts
}

func splitIdentifiers(text string) string {
	var expanded strings.Builder
	runes := []rune(text)
	for index, current := range runes {
		if index > 0 && unicode.IsUpper(current) && (unicode.IsLower(runes[index-1]) || unicode.IsNumber(runes[index-1])) {
			expanded.WriteByte(' ')
		}
		expanded.WriteRune(current)
	}
	return expanded.String()
}

func excerpt(text string, queryTerms []string) string {
	text = strings.TrimSpace(text)
	if len(text) <= 900 {
		return text
	}
	lower := strings.ToLower(text)
	position := -1
	for _, term := range queryTerms {
		if found := strings.Index(lower, term); found >= 0 && (position < 0 || found < position) {
			position = found
		}
	}
	if position < 0 {
		position = 0
	}
	start := position - 220
	if start < 0 {
		start = 0
	}
	end := start + 900
	if end > len(text) {
		end = len(text)
	}
	return strings.TrimSpace(text[start:end])
}
