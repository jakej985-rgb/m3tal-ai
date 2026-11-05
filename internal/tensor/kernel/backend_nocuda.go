//go:build !cuda

package kernel

// cudaCompiledFlag is false in builds without the cuda tag.
const cudaCompiledFlag = false

// cudaProbe reports CUDA as not compiled in. The real probe lives in
// backend_cuda.go behind the `cuda` build tag and cgo.
func cudaProbe() []BackendInfo {
	return []BackendInfo{{
		Name:       "cuda",
		Device:     "cuda",
		CompiledIn: false,
		Available:  false,
		Reason: "not compiled with CUDA support: rebuild with " +
			"CGO_ENABLED=1 go build -tags cuda (requires the CUDA toolkit and a " +
			"compatible NVIDIA driver; see docs/CUDA.md)",
	}}
}
