package autograd

import (
	"math"
	"testing"
)

func TestBackwardMatchesFiniteDifferences(t *testing.T) {
	x := New(0.37)
	y := New(-0.82)
	f := x.Mul(y).Add(x.Tanh())
	f.Backward()

	check := func(name string, value *Value, evaluate func(float64) float64) {
		t.Helper()
		const epsilon = 1e-6
		plus := evaluate(value.Data + epsilon)
		minus := evaluate(value.Data - epsilon)
		numerical := (plus - minus) / (2 * epsilon)
		if math.Abs(value.Grad-numerical) > 1e-6 {
			t.Errorf("%s gradient = %.9f; finite difference = %.9f", name, value.Grad, numerical)
		}
	}

	check("x", x, func(xValue float64) float64 {
		return xValue*y.Data + math.Tanh(xValue)
	})
	check("y", y, func(yValue float64) float64 {
		return x.Data*yValue + math.Tanh(x.Data)
	})
}

func TestBackwardAccumulatesSharedPathGradients(t *testing.T) {
	x := New(3)
	f := x.Mul(x).Add(x)
	f.Backward()

	if f.Data != 12 {
		t.Fatalf("output = %v, want 12", f.Data)
	}
	if x.Grad != 7 {
		t.Fatalf("gradient = %v, want 7", x.Grad)
	}
}
