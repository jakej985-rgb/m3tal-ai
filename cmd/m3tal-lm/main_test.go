package main

import (
	"path/filepath"
	"testing"
)

func TestLMCommands(t *testing.T) {
	tempDir := t.TempDir()
	instructionsPath := filepath.Join(tempDir, "instructions.txt")
	checkpointPath := filepath.Join(tempDir, "model.json")

	// 1. Generate instructions
	if err := run([]string{"instructions", "--output", instructionsPath}); err != nil {
		t.Fatalf("instructions command failed: %v", err)
	}

	// 2. Train with instructions included
	trainArgs := []string{
		"train",
		"--corpus", instructionsPath,
		"--checkpoint", checkpointPath,
		"--epochs", "1",
		"--steps", "5",
		"--context", "16",
		"--width", "8",
		"--vocab", "260",
		"--include-instructions",
	}
	if err := run(trainArgs); err != nil {
		t.Fatalf("train command failed: %v", err)
	}

	// 3. Eval checkpoint on instructions corpus
	evalArgs := []string{
		"eval",
		"--checkpoint", checkpointPath,
		"--corpus", instructionsPath,
	}
	if err := run(evalArgs); err != nil {
		t.Fatalf("eval command failed: %v", err)
	}

	// 4. Generate with checkpoint
	genArgs := []string{
		"generate",
		"--checkpoint", checkpointPath,
		"--prompt", "show stats",
		"--tokens", "10",
	}
	if err := run(genArgs); err != nil {
		t.Fatalf("generate command failed: %v", err)
	}
}

func TestLMCommandErrors(t *testing.T) {
	if err := run([]string{}); err == nil {
		t.Error("expected error for empty args")
	}
	if err := run([]string{"unknown"}); err == nil {
		t.Error("expected error for unknown command")
	}
	if err := run([]string{"train"}); err == nil {
		t.Error("expected error for train without --corpus")
	}
	if err := run([]string{"eval"}); err == nil {
		t.Error("expected error for eval without --corpus")
	}
}
