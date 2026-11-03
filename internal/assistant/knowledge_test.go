package assistant

import (
	"os"
	"path/filepath"
	"testing"
)

func TestKnowledgeBaseSearchesMarkdown(t *testing.T) {
	directory := t.TempDir()
	path := filepath.Join(directory, "guide.md")
	contents := "# Install\n\nUse the m3tal config wizard to install and configure the system.\n\n# Containers\n\nUse m3tal ps to list containers.\n"
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatal(err)
	}
	base, err := LoadKnowledgeBase([]string{directory})
	if err != nil {
		t.Fatal(err)
	}
	results := base.Search("how do I list containers", 2)
	if len(results) == 0 {
		t.Fatal("expected documentation match")
	}
	if results[0].Title != "Containers" {
		t.Fatalf("top result title = %q, want Containers", results[0].Title)
	}
	if results[0].Source != path {
		t.Fatalf("source = %q, want %q", results[0].Source, path)
	}
}

func TestKnowledgeBaseCanPreferRelevantSourceOverReadme(t *testing.T) {
	directory := t.TempDir()
	readme := filepath.Join(directory, "README.md")
	source := filepath.Join(directory, "health.go")
	if err := os.WriteFile(readme, []byte("# Overview\n\nContainer health and service monitoring are part of this project.\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(source, []byte("func GetContainerHealth() {\n\tmonitorSystemServices()\n}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	base, err := LoadKnowledgeBase([]string{directory})
	if err != nil {
		t.Fatal(err)
	}
	results := base.Search("container health", 2)
	if len(results) == 0 || results[0].Source != source || results[0].Kind != "Source" {
		t.Fatalf("top result = %#v, want source file %q", results, source)
	}
}

func TestKnowledgeBaseParsesHubAppRegistry(t *testing.T) {
	directory := t.TempDir()
	manifest := filepath.Join(directory, "manifest.yml")
	contents := "apps:\n  - id: monster-lab\n    name: Monster Lab\n    repo: Monster-Lab\n    status: ready\n    description: Robotics learning application.\n  - id: media-tools\n    name: Media Tools\n    status: review\n"
	if err := os.WriteFile(manifest, []byte(contents), 0o600); err != nil {
		t.Fatal(err)
	}
	base, err := LoadKnowledgeBase([]string{manifest})
	if err != nil {
		t.Fatal(err)
	}
	apps := base.Apps()
	if len(apps) != 2 || apps[0].Name != "Monster Lab" || apps[1].ID != "media-tools" {
		t.Fatalf("parsed applications = %#v", apps)
	}
}
