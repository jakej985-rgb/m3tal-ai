package nn

import (
	"fmt"
	"math"
)

// Optimizer updates trainable parameters from their accumulated gradients.
type Optimizer interface {
	Validate() error
	Update(parameters []*Parameter)
}

// SGD performs standard stochastic gradient descent updates.
type SGD struct {
	LearningRate float64
}

// Validate checks optimizer settings.
func (optimizer SGD) Validate() error {
	if optimizer.LearningRate <= 0 {
		return fmt.Errorf("SGD learning rate must be positive")
	}
	return nil
}

// Update applies one SGD step and clears each parameter's gradient.
func (optimizer SGD) Update(parameters []*Parameter) {
	for _, parameter := range parameters {
		for i, gradient := range parameter.grads {
			parameter.values[i] -= optimizer.LearningRate * gradient
			parameter.grads[i] = 0
		}
	}
}

// Adam adapts the learning rate for each parameter using moving averages of
// gradients and squared gradients.
type Adam struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64

	step   uint64
	first  map[*Parameter][]float64
	second map[*Parameter][]float64
}

// NewAdam returns Adam with the conventional beta and epsilon defaults.
func NewAdam(learningRate float64) *Adam {
	return &Adam{
		LearningRate: learningRate,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		first:        make(map[*Parameter][]float64),
		second:       make(map[*Parameter][]float64),
	}
}

// Validate checks Adam settings.
func (optimizer *Adam) Validate() error {
	if optimizer == nil || optimizer.LearningRate <= 0 {
		return fmt.Errorf("Adam learning rate must be positive")
	}
	if optimizer.Beta1 < 0 || optimizer.Beta1 >= 1 || optimizer.Beta2 < 0 || optimizer.Beta2 >= 1 {
		return fmt.Errorf("Adam beta values must be in [0, 1)")
	}
	if optimizer.Epsilon <= 0 {
		return fmt.Errorf("Adam epsilon must be positive")
	}
	return nil
}

// Update applies one bias-corrected Adam step and clears accumulated gradients.
func (optimizer *Adam) Update(parameters []*Parameter) {
	optimizer.step++
	correction1 := 1 - math.Pow(optimizer.Beta1, float64(optimizer.step))
	correction2 := 1 - math.Pow(optimizer.Beta2, float64(optimizer.step))
	for _, parameter := range parameters {
		first := optimizer.moment(optimizer.first, parameter)
		second := optimizer.moment(optimizer.second, parameter)
		for i, gradient := range parameter.grads {
			first[i] = optimizer.Beta1*first[i] + (1-optimizer.Beta1)*gradient
			second[i] = optimizer.Beta2*second[i] + (1-optimizer.Beta2)*gradient*gradient
			firstHat := first[i] / correction1
			secondHat := second[i] / correction2
			parameter.values[i] -= optimizer.LearningRate * firstHat / (math.Sqrt(secondHat) + optimizer.Epsilon)
			parameter.grads[i] = 0
		}
	}
}

func (optimizer *Adam) moment(store map[*Parameter][]float64, parameter *Parameter) []float64 {
	values := store[parameter]
	if values == nil {
		values = make([]float64, len(parameter.values))
		store[parameter] = values
	}
	return values
}
