package cpu

import (
	"runtime"
	"testing"
)

func TestFeatureDetection(t *testing.T) {
	feat := Features()
	t.Logf("Detected CPU features on %s/%s: AVX2=%v, FMA=%v, NEON=%v",
		runtime.GOOS, runtime.GOARCH, feat.HasAVX2, feat.HasFMA, feat.HasNEON)

	if runtime.GOARCH == "amd64" {
		t.Logf("On amd64, UseAVX2() = %v", UseAVX2())
	} else if runtime.GOARCH == "arm64" {
		if !feat.HasNEON {
			t.Errorf("Expected arm64 to have NEON support")
		}
	}

	SetForceFallback(true)
	if UseAVX2() || UseNEON() {
		t.Errorf("Expected UseAVX2 and UseNEON to be false when fallback is forced")
	}
	if !IsFallbackForced() {
		t.Errorf("Expected IsFallbackForced to be true")
	}

	SetForceFallback(false)
	if IsFallbackForced() {
		t.Errorf("Expected IsFallbackForced to be false")
	}
}
