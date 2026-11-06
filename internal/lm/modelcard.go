package lm

import (
	"fmt"
	"math"
	"strings"
)

// ModelCardData is the structured content of a generated model card.
type ModelCardData struct {
	Name            string
	CheckpointPath  string
	Config          Config
	Parameters      int
	VocabSize       int
	TokenizerHash   string
	Step            int64
	TemplateVersion string // chat template, "" for base models
	EvalLoss        float64
	EvalPPL         float64
	EvalMaskedTok   int64
	DataProvenance  string
	Limitations     []string
}

// RenderModelCard produces a Markdown model card documenting intended
// use, limitations, data provenance, and evaluation (Stage 4).
func RenderModelCard(d ModelCardData) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Model Card: %s\n\n", d.Name)
	b.WriteString("## Summary\n\n")
	fmt.Fprintf(&b, "- Checkpoint: `%s`\n", d.CheckpointPath)
	fmt.Fprintf(&b, "- Parameters: %s\n", formatCount(d.Parameters))
	fmt.Fprintf(&b, "- Vocab size: %d (tokenizer %s)\n", d.VocabSize, d.TokenizerHash)
	fmt.Fprintf(&b, "- Context: %d, width: %d, layers: %d, heads: %d\n",
		d.Config.Context, d.Config.Width, d.Config.Layers, d.Config.Heads)
	fmt.Fprintf(&b, "- Training step: %d\n", d.Step)
	if d.TemplateVersion != "" {
		fmt.Fprintf(&b, "- Chat template: `%s`\n", d.TemplateVersion)
	} else {
		b.WriteString("- Chat template: none (base model)\n")
	}
	b.WriteString("\n## Intended use\n\n")
	b.WriteString("Local, CPU-first conversational language model experimentation.\n")
	b.WriteString("Intended for research, evaluation, and M3TAL integration testing.\n")
	b.WriteString("Not intended for production decision-making, medical/legal advice,\n")
	b.WriteString("or deployment to untrusted multi-user environments without further\n")
	b.WriteString("safety review.\n")
	b.WriteString("\n## Limitations\n\n")
	for _, l := range d.Limitations {
		fmt.Fprintf(&b, "- %s\n", l)
	}
	b.WriteString("\n## Training data and provenance\n\n")
	b.WriteString(d.DataProvenance)
	b.WriteString("\n\n## Evaluation\n\n")
	if d.EvalMaskedTok > 0 {
		fmt.Fprintf(&b, "- Held-out assistant-token loss: %.4f (perplexity %.2f, %d masked tokens)\n",
			d.EvalLoss, math.Exp(d.EvalLoss), d.EvalMaskedTok)
	} else {
		b.WriteString("- No held-out conversational evaluation recorded.\n")
	}
	b.WriteString("- Automated metrics are heuristics; output quality requires human review.\n")
	b.WriteString("\n## Hardware and runtime\n\n")
	b.WriteString("CPU-first pure-Go inference; optional CUDA backend behind the `cuda`\n")
	b.WriteString("build tag. See docs/CUDA.md and docs/PACKAGING.md.\n")
	return b.String()
}

func formatCount(n int) string {
	if n >= 1_000_000 {
		return fmt.Sprintf("%.2fM", float64(n)/1e6)
	}
	if n >= 1_000 {
		return fmt.Sprintf("%.1fK", float64(n)/1e3)
	}
	return fmt.Sprintf("%d", n)
}
