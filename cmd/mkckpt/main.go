// Command mkckpt writes an untrained checkpoint for backend benchmarking.
package main

import (
	"flag"
	"fmt"
	"os"

	"m3tal-llm/internal/lm"
)

func main() {
	out := flag.String("out", "", "output checkpoint path")
	context := flag.Int("context", 64, "context tokens")
	width := flag.Int("width", 256, "model width")
	layers := flag.Int("layers", 4, "transformer layers")
	heads := flag.Int("heads", 8, "attention heads")
	vocab := flag.Int("vocab", 512, "tokenizer vocab size")
	seed := flag.Int64("seed", 17, "init seed")
	flag.Parse()
	if *out == "" {
		fmt.Fprintln(os.Stderr, "-out required")
		os.Exit(2)
	}
	data, err := os.ReadFile("data/instructions.txt")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	tok, err := lm.TrainTokenizer([]string{string(data)}, *vocab)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	m, err := lm.NewModel(lm.Config{Context: *context, Width: *width, Layers: *layers, Heads: *heads}, tok, *seed)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := m.SaveCheckpoint(*out); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote", *out)
}
