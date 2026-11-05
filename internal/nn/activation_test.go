package nn

import (
	"math"
	"testing"

	"m3tal-llm/internal/tensor"
)

func TestTanhBackwardMatchesFiniteDifference(t *testing.T) {
	input, err := tensor.NewMatrix(1, 2, []float64{-0.4, 0.7})
	if err != nil {
		t.Fatal(err)
	}
	output, err := Tanh(input)
	if err != nil {
		t.Fatal(err)
	}
	upstream, err := tensor.NewMatrix(1, 2, []float64{1.3, -0.6})
	if err != nil {
		t.Fatal(err)
	}
	gradient, err := TanhBackward(output, upstream)
	if err != nil {
		t.Fatal(err)
	}

	const epsilon = 1e-6
	for col := 0; col < input.Cols(); col++ {
		value := input.At(0, col)
		numerical := upstream.At(0, col) * (math.Tanh(value+epsilon) - math.Tanh(value-epsilon)) / (2 * epsilon)
		if math.Abs(gradient.At(0, col)-numerical) > 1e-6 {
			t.Errorf("gradient at column %d = %.9f; finite difference = %.9f", col, gradient.At(0, col), numerical)
		}
	}
}
