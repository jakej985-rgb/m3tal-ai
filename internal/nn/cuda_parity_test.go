//go:build cuda

package nn

import (
	"math"
	"testing"

	"m3tal-llm/internal/tensor"
	"m3tal-llm/internal/tensor/kernel"
)

// TestCUDADeviceOpsRun verifies that CUDA-dispatched tensor ops invoked by
// the nn package's tensor usage produce CPU-parity results.
func TestCUDADeviceOpsRun(t *testing.T) {
	if err := kernel.Select("cpu"); err != nil {
		t.Fatal(err)
	}
	a, err := tensor.FromFloat64([]int{4, 4}, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
	if err != nil {
		t.Fatal(err)
	}
	b, err := tensor.FromFloat64([]int{4, 4}, []float64{1, 0, 0, 1, 2, 0, 0, 2, 3, 0, 0, 3, 4, 0, 0, 4})
	if err != nil {
		t.Fatal(err)
	}
	cpuC, err := tensor.MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	cpuVals := cpuC.Float64Slice()

	if err := kernel.Select("cuda"); err != nil {
		t.Skipf("cuda unavailable: %v", err)
	}
	defer func() { _ = kernel.Select("cpu") }()

	gpuC, err := tensor.MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	gpuVals := gpuC.Float64Slice()
	for i := range gpuVals {
		if math.Abs(gpuVals[i]-cpuVals[i]) > 1e-9 {
			t.Fatalf("matmul elem %d: cpu %v cuda %v", i, cpuVals[i], gpuVals[i])
		}
	}
}
