//go:build cuda

package lm

import "m3tal-llm/internal/tensor/kernel"

// newInferenceBackend selects the inference backend. The CUDA backend is
// used whenever the kernel backend selection resolved to "cuda" — both
// for explicit M3TAL_LLM_BACKEND=cuda and for "auto" when a device is
// present. Measured on the GTX 960 (docs/PERF.md), CUDA generation is
// faster than CPU at every tested size, including a 147K-param model
// (~1300 vs ~415 tok/s), so no size heuristic is applied. Any
// construction failure falls back to CPU so inference always works.
func newInferenceBackend(m *Model) inferenceBackend {
	if kernel.Selected() == "cuda" {
		if b, err := newCUDAInferenceBackend(m); err == nil {
			return b
		}
	}
	return newCPUInferenceBackend(m)
}
