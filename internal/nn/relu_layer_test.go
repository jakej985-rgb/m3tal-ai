package nn

import (
	"reflect"
	"testing"

	"m3tal-llm/internal/tensor"
)

func TestReLUForwardAndBackward(t *testing.T) {
	layer := &ReLULayer{}
	input, _ := tensor.NewMatrix(1, 3, []float64{-2, 0, 3})
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	if got := []float64{output.At(0, 0), output.At(0, 1), output.At(0, 2)}; !reflect.DeepEqual(got, []float64{0, 0, 3}) {
		t.Fatalf("ReLU output = %v, want [0 0 3]", got)
	}
	upstream, _ := tensor.NewMatrix(1, 3, []float64{4, 5, 6})
	gradient, err := layer.Backward(upstream)
	if err != nil {
		t.Fatal(err)
	}
	if got := []float64{gradient.At(0, 0), gradient.At(0, 1), gradient.At(0, 2)}; !reflect.DeepEqual(got, []float64{0, 0, 6}) {
		t.Fatalf("ReLU gradient = %v, want [0 0 6]", got)
	}
}
