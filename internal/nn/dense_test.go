package nn

import (
	"math"
	"math/rand"
	"testing"

	"m3tal-ai/internal/tensor"
)

func TestDenseForwardAndBackward(t *testing.T) {
	layer, err := NewDense(2, 1, rand.New(rand.NewSource(4)))
	if err != nil {
		t.Fatal(err)
	}
	input, err := tensor.NewMatrix(2, 2, []float64{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	if output.Rows() != 2 || output.Cols() != 1 {
		t.Fatalf("output shape = %dx%d, want 2x1", output.Rows(), output.Cols())
	}

	outputGrad, _ := tensor.NewMatrix(2, 1, []float64{1, -2})
	inputGrad, err := layer.Backward(outputGrad)
	if err != nil {
		t.Fatal(err)
	}
	if inputGrad.Rows() != 2 || inputGrad.Cols() != 2 {
		t.Fatalf("input gradient shape = %dx%d, want 2x2", inputGrad.Rows(), inputGrad.Cols())
	}
	weightGrad, biasGrad := layer.Gradients()
	for index, expected := range []float64{-5, -6} {
		if math.Abs(weightGrad[index]-expected) > 1e-12 {
			t.Errorf("weight gradient %d = %v, want %v", index, weightGrad[index], expected)
		}
	}
	if biasGrad[0] != -1 {
		t.Errorf("bias gradient = %v, want -1", biasGrad[0])
	}
}

func TestDenseReportsInvalidCallOrderAndShape(t *testing.T) {
	layer, err := NewDense(2, 1, rand.New(rand.NewSource(4)))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := layer.Backward(tensor.Matrix{}); err == nil {
		t.Fatal("expected backward-before-forward error")
	}
	wrongInput, _ := tensor.NewMatrix(1, 3, []float64{1, 2, 3})
	if _, err := layer.Forward(wrongInput); err == nil {
		t.Fatal("expected input shape error")
	}
}
