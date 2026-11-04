package assistant

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExtractCitations(t *testing.T) {
	results := []KnowledgeResult{
		{Title: "Installation Guide", Source: "/docs/install.md", Excerpt: "Install steps"},
		{Title: "API Reference", Source: "/docs/api.go", Excerpt: "API endpoints"},
		{Title: "Traefik Setup", Source: "/docs/traefik.md", Excerpt: "Traefik proxy"},
	}

	tests := []struct {
		name       string
		draft      string
		wantCount  int
		wantTitles []string
		wantValid  bool
	}{
		{
			name:       "numeric citation [1]",
			draft:      "As shown in [1], you should run the install script.",
			wantCount:  1,
			wantTitles: []string{"Installation Guide"},
			wantValid:  true,
		},
		{
			name:       "multiple numeric citations [1] and [3]",
			draft:      "Refer to [1] for setup and [3] for reverse proxy.",
			wantCount:  2,
			wantTitles: []string{"Installation Guide", "Traefik Setup"},
			wantValid:  true,
		},
		{
			name:       "paren numeric citation (2)",
			draft:      "Check the endpoints in (2).",
			wantCount:  1,
			wantTitles: []string{"API Reference"},
			wantValid:  true,
		},
		{
			name:       "title citation",
			draft:      "According to the Installation Guide, run the script.",
			wantCount:  1,
			wantTitles: []string{"Installation Guide"},
			wantValid:  true,
		},
		{
			name:       "filename citation",
			draft:      "Details can be found in traefik.md.",
			wantCount:  1,
			wantTitles: []string{"Traefik Setup"},
			wantValid:  true,
		},
		{
			name:       "invalid citation [5] out of bounds",
			draft:      "Refer to [5] for instructions.",
			wantCount:  0,
			wantTitles: nil,
			wantValid:  false,
		},
		{
			name:       "no citations",
			draft:      "You should do this and that.",
			wantCount:  0,
			wantTitles: nil,
			wantValid:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cited := extractCitations(tt.draft, results)
			if len(cited) != tt.wantCount {
				t.Fatalf("extractCitations count = %d, want %d", len(cited), tt.wantCount)
			}
			for i, title := range tt.wantTitles {
				if cited[i].Title != title {
					t.Errorf("cited[%d].Title = %q, want %q", i, cited[i].Title, title)
				}
			}
			valid := draftCitesSources(tt.draft, results)
			if valid != tt.wantValid {
				t.Errorf("draftCitesSources = %v, want %v", valid, tt.wantValid)
			}
		})
	}
}

func TestDraftStatesUncertainty(t *testing.T) {
	uncertain := []string{
		"I don't know based on the provided reference notes.",
		"The notes do not contain this information.",
		"There is no mention of backups in the reference notes.",
		"Could not find information regarding database migration.",
	}
	for _, text := range uncertain {
		if !draftStatesUncertainty(text) {
			t.Errorf("expected draftStatesUncertainty(%q) to be true", text)
		}
	}

	certain := []string{
		"Run the installation script.",
		"Traefik is configured on port 80 and 443.",
	}
	for _, text := range certain {
		if draftStatesUncertainty(text) {
			t.Errorf("expected draftStatesUncertainty(%q) to be false", text)
		}
	}
}

func TestGroundedAnswerDistinguishesCitedAndUncited(t *testing.T) {
	results := []KnowledgeResult{
		{Title: "Install Guide", Source: "/docs/install.md"},
		{Title: "Proxy Guide", Source: "/docs/proxy.md"},
	}

	var buf bytes.Buffer
	printGroundedAnswer("According to [1], run the installer.", results, &buf)
	output := buf.String()

	if !strings.Contains(output, "Sources:") || !strings.Contains(output, "Install Guide") {
		t.Fatalf("missing cited source in output: %s", output)
	}
	if !strings.Contains(output, "Other retrieved sources:") || !strings.Contains(output, "Proxy Guide") {
		t.Fatalf("missing uncited source in output: %s", output)
	}
}

func TestAskDocsWithUncertaintyDraft(t *testing.T) {
	tempDir := t.TempDir()
	docPath := filepath.Join(tempDir, "install.md")
	if err := os.WriteFile(docPath, []byte("# Install\nRun install.\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	agent, err := New([]string{tempDir}, nil)
	if err != nil {
		t.Fatal(err)
	}

	agent.SetLanguageModel(stubGenerator{text: "I don't know based on the provided reference notes."})
	var buf bytes.Buffer
	if err := agent.Ask(context.Background(), "how do I install?", nil, &buf); err != nil {
		t.Fatal(err)
	}

	output := buf.String()
	if !strings.Contains(output, "found no answer in retrieved sources") {
		t.Fatalf("expected uncertainty message, got %q", output)
	}
	if !strings.Contains(output, "M3TAL knowledge matches:") {
		t.Fatalf("expected fallback excerpts for reference, got %q", output)
	}
}

func TestAskDocsWithNumericalCitation(t *testing.T) {
	tempDir := t.TempDir()
	docPath := filepath.Join(tempDir, "install.md")
	if err := os.WriteFile(docPath, []byte("# Install\nRun install.\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	agent, err := New([]string{tempDir}, nil)
	if err != nil {
		t.Fatal(err)
	}

	agent.SetLanguageModel(stubGenerator{text: "According to [1], run the install command."})
	var buf bytes.Buffer
	if err := agent.Ask(context.Background(), "how do I install?", nil, &buf); err != nil {
		t.Fatal(err)
	}

	output := buf.String()
	if !strings.Contains(output, "Local model draft") || !strings.Contains(output, "Sources:") || !strings.Contains(output, "install.md") {
		t.Fatalf("expected grounded draft with citations, got:\n%s", output)
	}
}
