//go:build cuda

package lm

import "m3tal-llm/internal/tensor/kernel"

// newInferenceBackend selects the inference backend. The CUDA backend is
// used only when the kernel backend selection resolved to "cuda"
// (M3TAL_LLM_BACKEND=cuda|auto or kernel.Select); any construction failure
// falls back to CPU so inference always works.
func newInferenceBackend(m *Model) inferenceBackend {
	if kernel.Selected() == "cuda" {
		if b, err := newCUDAInferenceBackend(m); err == nil {
			return b
		}
	}
	return newCPUInferenceBackend(m)
}
