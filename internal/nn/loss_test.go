package nn

import (
	"math"
	"testing"

	"m3tal-ai/internal/tensor"
)

func TestSoftmaxCrossEntropyGradientMatchesFiniteDifference(t *testing.T) {
	logits, _ := tensor.NewMatrix(2, 3, []float64{0.2, -0.4, 1.1, 2, 0.5, -1})
	targets, _ := tensor.NewMatrix(2, 3, []float64{0, 0, 1, 1, 0, 0})
	objective := SoftmaxCrossEntropy{}
	gradient, err := objective.Gradient(logits, targets)
	if err != nil {
		t.Fatal(err)
	}

	const epsilon = 1e-6
	for row := 0; row < logits.Rows(); row++ {
		for col := 0; col < logits.Cols(); col++ {
			values := []float64{0.2, -0.4, 1.1, 2, 0.5, -1}
			index := row*logits.Cols() + col
			values[index] += epsilon
			plus, _ := tensor.NewMatrix(2, 3, values)
			plusLoss, err := objective.Value(plus, targets)
			if err != nil {
				t.Fatal(err)
			}
			values[index] -= 2 * epsilon
			minus, _ := tensor.NewMatrix(2, 3, values)
			minusLoss, err := objective.Value(minus, targets)
			if err != nil {
				t.Fatal(err)
			}
			numerical := (plusLoss - minusLoss) / (2 * epsilon)
			if math.Abs(gradient.At(row, col)-numerical) > 1e-6 {
				t.Errorf("gradient (%d,%d) = %.9f; finite difference = %.9f", row, col, gradient.At(row, col), numerical)
			}
		}
	}
}
