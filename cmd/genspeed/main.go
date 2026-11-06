// Command genspeed measures KV-cache generation throughput for a checkpoint.
package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"m3tal-llm/internal/lm"
)

func main() {
	ckpt := flag.String("checkpoint", "", "checkpoint path")
	tokens := flag.Int("tokens", 100, "tokens to generate")
	prompt := flag.String("prompt", "Once upon a time", "prompt")
	flag.Parse()
	if *ckpt == "" {
		fmt.Fprintln(os.Stderr, "-checkpoint required")
		os.Exit(2)
	}
	m, err := lm.LoadCheckpoint(*ckpt)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	// warmup: one short generation to build caches/JIT paths
	_, _ = m.Generate(*prompt, 4, 0.8, 1)
	start := time.Now()
	out, err := m.Generate(*prompt, *tokens, 0.8, 1)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	el := time.Since(start)
	fmt.Printf("%d tokens in %v -> %.2f tok/s (%d chars out)\n", *tokens, el, float64(*tokens)/el.Seconds(), len(out))
}
