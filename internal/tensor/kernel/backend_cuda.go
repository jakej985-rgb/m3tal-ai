//go:build cuda

package kernel

import (
	"fmt"

	"m3tal-llm/internal/gpu"
)

// cudaCompiledFlag is true in builds with the cuda tag.
const cudaCompiledFlag = true

// cudaProbe enumerates CUDA devices through the gpu package's runtime
// bindings. Device details come from gpu.ProbeDevice (CUDA runtime API).
func cudaProbe() []BackendInfo {
	info, err := gpu.ProbeDevice()
	if err != nil {
		return []BackendInfo{{
			Name:       "cuda",
			Device:     "cuda",
			CompiledIn: true,
			Available:  false,
			Reason:     err.Error(),
		}}
	}
	return []BackendInfo{{
		Name:       "cuda",
		Device:     "cuda:0",
		CompiledIn: true,
		Available:  true,
		Reason: fmt.Sprintf("%d device(s); cuda:0 = %q (%s)",
			info.Count, info.Name, info.DeviceArch),
	}}
}
