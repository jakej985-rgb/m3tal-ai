//go:build arm64

package cpu

func detectFeatures() CPUFeatures {
	// ARM64 architecture guarantees NEON (ASIMD) support as part of the base profile
	return CPUFeatures{
		HasNEON: true,
	}
}
