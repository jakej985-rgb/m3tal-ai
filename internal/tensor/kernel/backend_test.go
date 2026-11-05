package kernel

import (
	"strings"
	"testing"
)

func TestProbeAlwaysIncludesCPU(t *testing.T) {
	backends := Probe()
	if len(backends) < 1 {
		t.Fatal("Probe returned no backends")
	}
	cpu := backends[0]
	if cpu.Name != "cpu" || !cpu.Available || !cpu.CompiledIn {
		t.Fatalf("CPU backend must always be available, got %+v", cpu)
	}
	if !cpu.Capabilities.MatMulFloat32 || !cpu.Capabilities.ElementwiseFloat32 {
		t.Fatalf("CPU backend missing core capabilities: %+v", cpu.Capabilities)
	}
}

func TestProbeIncludesCUDAEntry(t *testing.T) {
	cuda, ok := lookupBackend("cuda")
	if !ok {
		t.Fatal("Probe did not report a cuda backend entry")
	}
	// Without the cuda build tag CompiledIn must be false; with the tag it
	// depends on the runtime probe. Either way the entry must exist.
	if cuda.Name != "cuda" {
		t.Fatalf("unexpected backend name %q", cuda.Name)
	}
}

func TestSelectCPU(t *testing.T) {
	if err := Select("cpu"); err != nil {
		t.Fatalf("Select(\"cpu\") failed: %v", err)
	}
	if Selected() != "cpu" {
		t.Fatalf("Selected() = %q, want cpu", Selected())
	}
}

func TestSelectUnknownBackend(t *testing.T) {
	err := Select("tpu")
	if err == nil {
		t.Fatal("Select(\"tpu\") should fail")
	}
	if !strings.Contains(err.Error(), "unknown backend") {
		t.Fatalf("error should name the problem, got: %v", err)
	}
}

func TestSelectCUDAWithoutBuildTag(t *testing.T) {
	// This test runs in the default (non-cuda) CI build, where the cuda
	// backend must report a clear rebuild instruction.
	info, ok := lookupBackend("cuda")
	if !ok {
		t.Skip("no cuda backend entry")
	}
	if info.CompiledIn {
		t.Skip("cuda backend compiled in; skipping not-compiled-in assertions")
	}
	err := Select("cuda")
	if err == nil {
		t.Fatal("Select(\"cuda\") should fail when not compiled in")
	}
	for _, want := range []string{"not compiled", "-tags cuda"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error should mention %q, got: %v", want, err)
		}
	}
}

func TestSelectAutoFallsBackToCPU(t *testing.T) {
	if err := Select("auto"); err != nil {
		t.Fatalf("Select(\"auto\") failed: %v", err)
	}
	if got := Selected(); got != "cpu" && got != "cuda" {
		t.Fatalf("Selected() = %q, want cpu or cuda", got)
	}
	// Restore deterministic default for other tests.
	_ = Select("cpu")
}

func TestReportNonEmpty(t *testing.T) {
	if r := Report(); r == "" {
		t.Fatal("Report() returned empty string")
	}
}
