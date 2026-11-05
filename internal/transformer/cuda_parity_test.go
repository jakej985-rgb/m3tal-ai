//go:build cuda

package transformer

import (
	"math"
	"testing"

	"m3tal-llm/internal/autograd"
	"m3tal-llm/internal/tensor"
	"m3tal-llm/internal/tensor/kernel"
)

// stepModel runs one forward+backward of a tiny two-layer model:
//
//	h = ReLU(X·W1) ; y = h·W2 ; loss = Mean(y)
//
// and returns the scalar loss plus W1/W2 gradients.
func stepModel(t *testing.T, xData, w1Data, w2Data []float32) (float32, []float32, []float32) {
	t.Helper()
	newVar := func(shape []int, data []float32, req bool) *autograd.Variable {
		tens, err := tensor.FromFloat32(shape, data)
		if err != nil {
			t.Fatalf("tensor: %v", err)
		}
		v, err := autograd.NewVariable(tens, req)
		if err != nil {
			t.Fatalf("variable: %v", err)
		}
		return v
	}

	x := newVar([]int{4, 3}, xData, false)
	w1 := newVar([]int{3, 8}, w1Data, true)
	w2 := newVar([]int{8, 2}, w2Data, true)

	h, err := autograd.MatMul(x, w1)
	if err != nil {
		t.Fatal(err)
	}
	h, err = autograd.ReLU(h)
	if err != nil {
		t.Fatal(err)
	}
	y, err := autograd.MatMul(h, w2)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := autograd.Mean(y)
	if err != nil {
		t.Fatal(err)
	}
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	lossVal := loss.Data.Float32Slice()[0]
	return lossVal, w1.Grad.Float32Slice(), w2.Grad.Float32Slice()
}

func TestCUDAForwardBackwardGradientParity(t *testing.T) {
	if err := kernel.Select("cpu"); err != nil {
		t.Fatal(err)
	}
	xData := []float32{0.5, -1, 0.25, 1, 0, -0.5, -0.75, 0.9, 0.1, 0.3, -0.2, 0.8}
	w1Data := []float32{
		0.1, -0.2, 0.3, 0.05, -0.15, 0.25, -0.05, 0.2,
		0.3, 0.1, -0.25, 0.15, -0.1, 0.05, 0.2, -0.3,
		-0.2, 0.4, 0.1, -0.05, 0.3, -0.15, 0.25, 0.1,
	}
	w2Data := []float32{
		0.2, -0.1, 0.3, 0.15, -0.25, 0.05, 0.1, -0.2,
		-0.3, 0.2, 0.1, 0.25, 0.05, -0.15, 0.3, 0.1,
	}

	cpuLoss, cpuG1, cpuG2 := stepModel(t, xData, w1Data, w2Data)

	if err := kernel.Select("cuda"); err != nil {
		t.Skipf("cuda unavailable: %v", err)
	}
	defer func() { _ = kernel.Select("cpu") }()

	gpuLoss, gpuG1, gpuG2 := stepModel(t, xData, w1Data, w2Data)

	if math.Abs(float64(gpuLoss-cpuLoss)) > 1e-5 {
		t.Fatalf("loss: cpu %v cuda %v", cpuLoss, gpuLoss)
	}
	maxDiff := func(a, b []float32) float64 {
		m := 0.0
		for i := range a {
			if d := math.Abs(float64(a[i] - b[i])); d > m {
				m = d
			}
		}
		return m
	}
	if d := maxDiff(cpuG1, gpuG1); d > 1e-4 {
		t.Fatalf("W1 grad max diff %g", d)
	}
	if d := maxDiff(cpuG2, gpuG2); d > 1e-4 {
		t.Fatalf("W2 grad max diff %g", d)
	}
}

// TestCUDATrainingTrajectoryParity runs a fixed-config short training loop
// (forward -> backward -> SGD step) on both backends and compares the loss
// trajectory within tolerance.
func TestCUDATrainingTrajectoryParity(t *testing.T) {
	if err := kernel.Select("cpu"); err != nil {
		t.Fatal(err)
	}

	xData := []float32{0.5, -1, 0.25, 1, 0, -0.5, -0.75, 0.9, 0.1, 0.3, -0.2, 0.8}
	initW1 := []float32{
		0.1, -0.2, 0.3, 0.05, -0.15, 0.25, -0.05, 0.2,
		0.3, 0.1, -0.25, 0.15, -0.1, 0.05, 0.2, -0.3,
		-0.2, 0.4, 0.1, -0.05, 0.3, -0.15, 0.25, 0.1,
	}
	initW2 := []float32{
		0.2, -0.1, 0.3, 0.15, -0.25, 0.05, 0.1, -0.2,
		-0.3, 0.2, 0.1, 0.25, 0.05, -0.15, 0.3, 0.1,
	}

	run := func(t *testing.T) []float32 {
		w1 := append([]float32(nil), initW1...)
		w2 := append([]float32(nil), initW2...)
		losses := make([]float32, 0, 6)
		for step := 0; step < 6; step++ {
			loss, g1, g2 := stepModel(t, xData, w1, w2)
			losses = append(losses, loss)
			for i := range w1 {
				w1[i] -= 0.01 * g1[i]
			}
			for i := range w2 {
				w2[i] -= 0.01 * g2[i]
			}
		}
		return losses
	}

	cpuLosses := run(t)

	if err := kernel.Select("cuda"); err != nil {
		t.Skipf("cuda unavailable: %v", err)
	}
	defer func() { _ = kernel.Select("cpu") }()

	gpuLosses := run(t)

	for i := range cpuLosses {
		diff := math.Abs(float64(gpuLosses[i] - cpuLosses[i]))
		scale := math.Max(1, math.Abs(float64(cpuLosses[i])))
		if diff > 1e-4*scale {
			t.Fatalf("step %d loss: cpu %v cuda %v", i, cpuLosses[i], gpuLosses[i])
		}
	}
}
