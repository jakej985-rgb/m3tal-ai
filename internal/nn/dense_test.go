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

func TestDenseGradientsMatchFiniteDifferences(t *testing.T) {
	layer, err := NewDense(2, 2, rand.New(rand.NewSource(2)))
	if err != nil {
		t.Fatal(err)
	}
	layer.weights.values = []float64{0.2, -0.3, 0.4, 0.1}
	layer.bias.values = []float64{0.05, -0.07}
	input, _ := tensor.NewMatrix(1, 2, []float64{0.6, -0.8})
	upstream, _ := tensor.NewMatrix(1, 2, []float64{0.7, -1.2})
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	inputGradient, err := layer.Backward(upstream)
	if err != nil {
		t.Fatal(err)
	}
	weightGradient, biasGradient := layer.Gradients()
	const epsilon = 1e-6
	objective := func(input tensor.Matrix) float64 {
		prediction, forwardErr := layer.Forward(input)
		if forwardErr != nil {
			t.Fatal(forwardErr)
		}
		return prediction.At(0, 0)*upstream.At(0, 0) + prediction.At(0, 1)*upstream.At(0, 1)
	}
	_ = output
	for index := range layer.weights.values {
		original := layer.weights.values[index]
		layer.weights.values[index] = original + epsilon
		plus := objective(input)
		layer.weights.values[index] = original - epsilon
		minus := objective(input)
		layer.weights.values[index] = original
		numerical := (plus - minus) / (2 * epsilon)
		if math.Abs(weightGradient[index]-numerical) > 1e-6 {
			t.Errorf("weight grad %d = %.8f; numerical = %.8f", index, weightGradient[index], numerical)
		}
	}
	for index := range layer.bias.values {
		original := layer.bias.values[index]
		layer.bias.values[index] = original + epsilon
		plus := objective(input)
		layer.bias.values[index] = original - epsilon
		minus := objective(input)
		layer.bias.values[index] = original
		numerical := (plus - minus) / (2 * epsilon)
		if math.Abs(biasGradient[index]-numerical) > 1e-6 {
			t.Errorf("bias grad %d = %.8f; numerical = %.8f", index, biasGradient[index], numerical)
		}
	}
	for index := 0; index < 2; index++ {
		valuesPlus := []float64{0.6, -0.8}
		valuesPlus[index] += epsilon
		valuesMinus := []float64{0.6, -0.8}
		valuesMinus[index] -= epsilon
		plus, _ := tensor.NewMatrix(1, 2, valuesPlus)
		minus, _ := tensor.NewMatrix(1, 2, valuesMinus)
		numerical := (objective(plus) - objective(minus)) / (2 * epsilon)
		if math.Abs(inputGradient.At(0, index)-numerical) > 1e-6 {
			t.Errorf("input grad %d = %.8f; numerical = %.8f", index, inputGradient.At(0, index), numerical)
		}
	}
}
