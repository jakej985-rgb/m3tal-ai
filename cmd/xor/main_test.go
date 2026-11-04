package main

import (
	"math/rand"
	"testing"

	"m3tal-ai/internal/nn"
	"m3tal-ai/internal/tensor"
)

func TestXORConvergesWithFixedSeed(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	hidden, err := nn.NewDense(2, 4, rng)
	if err != nil {
		t.Fatal(err)
	}
	output, err := nn.NewDense(4, 1, rng)
	if err != nil {
		t.Fatal(err)
	}
	model := nn.NewSequential(hidden, &nn.TanhLayer{}, output, &nn.TanhLayer{})
	inputs, _ := tensor.NewMatrix(4, 2, []float64{0, 0, 0, 1, 1, 0, 1, 1})
	targets, _ := tensor.NewMatrix(4, 1, []float64{-1, 1, 1, -1})
	if _, err := (nn.Trainer{Epochs: 3000, LearningRate: 0.1}).Fit(model, inputs, targets); err != nil {
		t.Fatal(err)
	}
	predictions, err := model.Forward(inputs)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{0, 1, 1, 0}
	for row, class := range want {
		value := predictions.At(row, 0)
		if (value >= 0) != (class == 1) {
			t.Errorf("XOR row %d prediction = %.6f; expected class %d", row, value, class)
		}
	}
}
