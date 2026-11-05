// regen-instructions writes the DefaultInstructionSet to data/instructions.txt.
// Usage: go run ./cmd/regen-instructions
package main

import (
	"log"
	"m3tal-llm/internal/lm"
)

func main() {
	examples := lm.DefaultInstructionSet()
	if err := lm.WriteInstructionCorpusFile("data/instructions.txt", examples); err != nil {
		log.Fatalf("write instructions: %v", err)
	}
	log.Printf("wrote %d examples to data/instructions.txt", len(examples))
}
