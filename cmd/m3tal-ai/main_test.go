package main

import (
	"testing"

	"m3tal-ai/internal/assistant"
)

func TestVersionFlag(t *testing.T) {
	if err := run([]string{"--version"}); err != nil {
		t.Fatal(err)
	}
	if assistant.Version == "" {
		t.Fatal("assistant version is empty")
	}
}

func TestEvalFlag(t *testing.T) {
	tempDir := t.TempDir()
	t.Setenv("M3TAL_DOCS_PATHS", tempDir)
	if err := run([]string{"--eval"}); err != nil {
		t.Fatalf("run --eval failed: %v", err)
	}
	if err := run([]string{"--eval", "--json"}); err != nil {
		t.Fatalf("run --eval --json failed: %v", err)
	}
}

func TestConfirmationMustMatchExactActionAndTarget(t *testing.T) {
	expected := "confirm restart stack media"
	for _, test := range []struct {
		actual string
		want   bool
	}{
		{expected, true},
		{" confirm restart stack media \n", true},
		{"confirm stop stack media", false},
		{"confirm restart stack other", false},
		{"CONFIRM RESTART STACK MEDIA", false},
		{"y", false},
	} {
		if got := confirmationMatches(expected, test.actual); got != test.want {
			t.Errorf("confirmationMatches(%q) = %v, want %v", test.actual, got, test.want)
		}
	}
}
