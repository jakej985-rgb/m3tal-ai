//go:build cuda

package lm

import "m3tal-llm/internal/tensor/kernel"

// cudaAutoMinParams is the estimated parameter count above which "auto"
// backend selection may choose CUDA for inference. Below this, per-op
// host<->device transfer and kernel-launch overhead dominate on the
// measured hardware (GTX 960), so CPU is faster. Explicit
// M3TAL_LLM_BACKEND=cuda always wins regardless of model size.
const cudaAutoMinParams = 8 << 20 // ~8M parameters

// estimateParams approximates the parameter count from the model config:
// token embedding + output projection (vocab*width each) plus ~10*width^2
// per transformer block (Q/K/V/O + SwiGLU FFN, hidden = 2*width).
func estimateParams(c Config, vocab int) int64 {
	width := int64(c.Width)
	v := int64(vocab)
	perBlock := 10 * width * width
	return 2*v*width + int64(c.Layers)*perBlock
}

// newInferenceBackend selects the inference backend. The CUDA backend is
// used when the kernel backend selection resolved to "cuda" — either
// explicitly (always honored) or via "auto" for models at or above
// cudaAutoMinParams. Any construction failure falls back to CPU so
// inference always works.
func newInferenceBackend(m *Model) inferenceBackend {
	if kernel.Selected() == "cuda" {
		if kernel.AutoSelected() && estimateParams(m.Config, len(m.Tokenizer.Tokens)) < cudaAutoMinParams {
			return newCPUInferenceBackend(m)
		}
		if b, err := newCUDAInferenceBackend(m); err == nil {
			return b
		}
	}
	return newCPUInferenceBackend(m)
}
