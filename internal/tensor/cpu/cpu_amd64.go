//go:build amd64

package cpu

func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)

func detectFeatures() CPUFeatures {
	var feat CPUFeatures
	maxEax, _, _, _ := cpuid(0, 0)
	if maxEax >= 1 {
		_, _, ecx1, _ := cpuid(1, 0)
		// Check FMA (bit 12) and OSXSAVE (bit 27) and AVX (bit 28)
		hasOSXSAVE := (ecx1 & (1 << 27)) != 0
		hasFMA := (ecx1 & (1 << 12)) != 0
		hasAVX := (ecx1 & (1 << 28)) != 0

		if hasOSXSAVE && hasAVX && maxEax >= 7 {
			_, ebx7, _, _ := cpuid(7, 0)
			// Check AVX2 (bit 5 of EBX)
			hasAVX2 := (ebx7 & (1 << 5)) != 0
			feat.HasAVX2 = hasAVX2
			feat.HasFMA = hasFMA
		}
	}
	return feat
}
