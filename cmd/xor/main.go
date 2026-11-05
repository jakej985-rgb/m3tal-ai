package main

import (
	"fmt"
	"math/rand"

	"m3tal-llm/internal/nn"
	"m3tal-llm/internal/tensor"
)

func main() {
	if err := run(); err != nil {
		panic(err)
	}
}

func run() error {
	rng := rand.New(rand.NewSource(7))
	hidden, err := nn.NewDense(2, 4, rng)
	if err != nil {
		return err
	}
	output, err := nn.NewDense(4, 1, rng)
	if err != nil {
		return err
	}
	model := nn.NewSequential(hidden, &nn.TanhLayer{}, output, &nn.TanhLayer{})

	inputs, err := tensor.NewMatrix(4, 2, []float64{0, 0, 0, 1, 1, 0, 1, 1})
	if err != nil {
		return err
	}
	targets, err := tensor.NewMatrix(4, 1, []float64{-1, 1, 1, -1})
	if err != nil {
		return err
	}

	trainer := nn.Trainer{Epochs: 3000, LearningRate: 0.1, ReportEvery: 500}
	losses, err := trainer.Fit(model, inputs, targets)
	if err != nil {
		return err
	}
	for _, report := range losses {
		label := "epoch"
		if report.Epoch == trainer.Epochs {
			label = "final"
		}
		fmt.Printf("%-5s %4d | loss %.6f\n", label, report.Epoch, report.Loss)
	}

	fmt.Println("XOR predictions (sequential model):")
	pairs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	for _, pair := range pairs {
		input, err := tensor.NewMatrix(1, 2, pair)
		if err != nil {
			return err
		}
		prediction, err := model.Forward(input)
		if err != nil {
			return err
		}
		value := prediction.At(0, 0)
		fmt.Printf("%v -> %.3f (class %d)\n", pair, value, class(value))
	}
	return nil
}

func class(value float64) int {
	if value >= 0 {
		return 1
	}
	return 0
}
