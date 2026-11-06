//go:build cuda

package lm

import (
	"math"
	"testing"

	"m3tal-llm/internal/tensor/kernel"
)

// newTestModel builds a tiny deterministic model for backend parity tests.
func newTestModel(t *testing.T) *Model {
	t.Helper()
	tokenizer, err := TrainTokenizer([]string{
		"the quick brown fox jumps over the lazy dog. m3tal gpu parity test corpus café",
		"another sentence for tokenizer coverage with numbers 12345 and symbols !@#",
	}, 256)
	if err != nil {
		t.Fatalf("TrainTokenizer: %v", err)
	}
	model, err := NewModel(Config{Context: 32, Width: 32, Layers: 2, Heads: 4}, tokenizer, 7)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	return model
}

// newCUDAOrSkip returns a CUDA backend or skips when no device is usable.
func newCUDAOrSkip(t *testing.T, model *Model) *cudaInferenceBackend {
	t.Helper()
	b, err := newCUDAInferenceBackend(model)
	if err != nil {
		t.Skipf("CUDA backend unavailable: %v", err)
	}
	return b.(*cudaInferenceBackend)
}

func logitsNear(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("logits length mismatch: %d vs %d", len(got), len(want))
	}
	for i := range got {
		diff := math.Abs(float64(got[i] - want[i]))
		scale := math.Max(1, math.Max(math.Abs(float64(want[i])), math.Abs(float64(got[i]))))
		if diff > tol*scale {
			t.Fatalf("logit %d: got %v want %v (diff %g)", i, got[i], want[i], diff)
		}
	}
}

func TestCUDAForwardMatchesCPU(t *testing.T) {
	model := newTestModel(t)
	gpuB := newCUDAOrSkip(t, model)
	cpuB := newCPUInferenceBackend(model)

	ids := model.Tokenizer.Encode("the quick brown fox")
	got := gpuB.Forward(ids)
	want := cpuB.Forward(ids)
	logitsNear(t, got, want, 2e-4)
}

func TestCUDAForwardStepKVMatchesCPU(t *testing.T) {
	model := newTestModel(t)
	gpuB := newCUDAOrSkip(t, model)
	cpuB := newCPUInferenceBackend(model)

	ids := model.Tokenizer.Encode("m3tal gpu parity")
	gpuCache := gpuB.NewKVCache()
	defer gpuCache.Free()
	cpuCache := cpuB.NewKVCache()
	defer cpuCache.Free()

	for i, id := range ids {
		got := gpuB.ForwardStepKV(id, i, gpuCache)
		want := cpuB.ForwardStepKV(id, i, cpuCache)
		logitsNear(t, got, want, 2e-4)
	}
}

func TestCUDAForwardStepKVLongerThanPrefill(t *testing.T) {
	model := newTestModel(t)
	gpuB := newCUDAOrSkip(t, model)
	cpuB := newCPUInferenceBackend(model)

	// Exercise cache growth beyond the initial allocation.
	ids := model.Tokenizer.Encode("one two three four five six seven eight nine ten eleven twelve")
	gpuCache := gpuB.NewKVCache()
	defer gpuCache.Free()
	cpuCache := cpuB.NewKVCache()
	defer cpuCache.Free()

	for i, id := range ids {
		got := gpuB.ForwardStepKV(id, i, gpuCache)
		want := cpuB.ForwardStepKV(id, i, cpuCache)
		if i == len(ids)-1 {
			logitsNear(t, got, want, 2e-4)
		}
	}
}

func TestCUDABackendSelectedViaKernel(t *testing.T) {
	// newInferenceBackend must pick CUDA when the kernel backend is cuda.
	if err := kernel.Select("cuda"); err != nil {
		t.Skipf("cuda backend unavailable: %v", err)
	}
	defer func() { _ = kernel.Select("cpu") }()
	model := newTestModel(t)
	b := newInferenceBackend(model)
	if _, ok := b.(*cudaInferenceBackend); !ok {
		t.Fatalf("newInferenceBackend returned %T, want *cudaInferenceBackend", b)
	}
}

func TestCUDAAutoSelectPrefersCPUForSmallModels(t *testing.T) {
	// "auto" on a small model must resolve to the CPU inference backend;
	// explicit "cuda" selection must still be honored.
	if _, err := newCUDAInferenceBackend(newTestModel(t)); err != nil {
		t.Skipf("CUDA backend unavailable: %v", err)
	}
	model := newTestModel(t)

	if err := kernel.Select("auto"); err != nil {
		t.Fatalf("Select(auto): %v", err)
	}
	if kernel.Selected() != "cuda" || !kernel.AutoSelected() {
		t.Skip("auto did not resolve to cuda on this machine")
	}
	if b := newInferenceBackend(model); b == nil {
		t.Fatal("newInferenceBackend returned nil")
	} else if _, ok := b.(*cudaInferenceBackend); ok {
		t.Fatal("auto selected CUDA for a small model; want CPU")
	}

	if err := kernel.Select("cuda"); err != nil {
		t.Fatalf("Select(cuda): %v", err)
	}
	if b := newInferenceBackend(model); b == nil {
		t.Fatal("newInferenceBackend returned nil")
	} else if _, ok := b.(*cudaInferenceBackend); !ok {
		t.Fatalf("explicit cuda returned %T, want *cudaInferenceBackend", b)
	}
	_ = kernel.Select("cpu")
}

func TestEstimateParams(t *testing.T) {
	// 2*vocab*width + layers*10*width^2
	got := estimateParams(Config{Width: 256, Layers: 4}, 512)
	want := int64(2*512*256 + 4*10*256*256)
	if got != want {
		t.Fatalf("estimateParams = %d, want %d", got, want)
	}
}
