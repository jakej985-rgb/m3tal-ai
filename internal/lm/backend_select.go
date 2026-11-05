//go:build !cuda

package lm

// newInferenceBackend selects the inference backend. Without the cuda
// build tag only the CPU backend exists.
func newInferenceBackend(m *Model) inferenceBackend {
	return newCPUInferenceBackend(m)
}
