package cpu

// CPUFeatures stores detected CPU vector and acceleration capabilities.
type CPUFeatures struct {
	HasAVX2 bool
	HasFMA  bool
	HasNEON bool
}

var (
	detectedFeatures CPUFeatures
	forceFallback    bool
)

func init() {
	detectedFeatures = detectFeatures()
}

// Features returns the detected CPU capabilities on the host system.
func Features() CPUFeatures {
	return detectedFeatures
}

// SetForceFallback forces CPU kernels to bypass SIMD/vector routines
// and use the generic reference implementation. Useful for testing and benchmarks.
func SetForceFallback(force bool) {
	forceFallback = force
}

// IsFallbackForced reports whether the CPU kernel dispatch is currently forced to generic fallback.
func IsFallbackForced() bool {
	return forceFallback
}

// UseAVX2 reports whether AVX2 + FMA kernels should be used on this system.
func UseAVX2() bool {
	if forceFallback {
		return false
	}
	return detectedFeatures.HasAVX2 && detectedFeatures.HasFMA
}

// UseNEON reports whether NEON kernels should be used on this system.
func UseNEON() bool {
	if forceFallback {
		return false
	}
	return detectedFeatures.HasNEON
}
