package nn

import (
	"math"
	"testing"
)

func TestSGDAndAdamUpdateParameters(t *testing.T) {
	parameter := &Parameter{values: []float64{2}, grads: []float64{4}}
	SGD{LearningRate: 0.25}.Update([]*Parameter{parameter})
	if parameter.values[0] != 1 || parameter.grads[0] != 0 {
		t.Fatalf("SGD result = value %v, grad %v; want value 1, grad 0", parameter.values[0], parameter.grads[0])
	}

	parameter.grads[0] = 2
	adam := NewAdam(0.1)
	adam.Update([]*Parameter{parameter})
	if math.Abs(parameter.values[0]-0.9) > 1e-7 {
		t.Fatalf("Adam value = %.9f, want about 0.9", parameter.values[0])
	}
	if parameter.grads[0] != 0 {
		t.Fatalf("Adam did not clear gradient: %v", parameter.grads[0])
	}
}
