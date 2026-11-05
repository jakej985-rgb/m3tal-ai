package nn

import (
	"fmt"

	"m3tal-llm/internal/tensor"
)

// Layer is a trainable or stateless matrix transformation used by Sequential.
type Layer interface {
	Forward(input tensor.Matrix) (tensor.Matrix, error)
	Backward(outputGrad tensor.Matrix) (tensor.Matrix, error)
	ZeroGrad()
}

type parameterProvider interface {
	Parameters() []*Parameter
}

// Sequential applies layers in order and propagates gradients in reverse order.
type Sequential struct {
	layers []Layer
}

// NewSequential creates a model from its layers in forward order.
func NewSequential(layers ...Layer) *Sequential {
	return &Sequential{layers: append([]Layer(nil), layers...)}
}

// Add appends a layer to the model.
func (s *Sequential) Add(layer Layer) {
	s.layers = append(s.layers, layer)
}

// Forward runs input through every layer.
func (s *Sequential) Forward(input tensor.Matrix) (tensor.Matrix, error) {
	if len(s.layers) == 0 {
		return tensor.Matrix{}, fmt.Errorf("sequential model has no layers")
	}
	output := input
	var err error
	for _, layer := range s.layers {
		output, err = layer.Forward(output)
		if err != nil {
			return tensor.Matrix{}, err
		}
	}
	return output, nil
}

// Backward propagates a gradient through layers in reverse order.
func (s *Sequential) Backward(outputGrad tensor.Matrix) (tensor.Matrix, error) {
	gradient := outputGrad
	var err error
	for i := len(s.layers) - 1; i >= 0; i-- {
		gradient, err = s.layers[i].Backward(gradient)
		if err != nil {
			return tensor.Matrix{}, err
		}
	}
	return gradient, nil
}

// ZeroGrad clears accumulated parameter gradients in every layer.
func (s *Sequential) ZeroGrad() {
	for _, layer := range s.layers {
		layer.ZeroGrad()
	}
}

// Parameters returns the trainable parameters from all layers in order.
func (s *Sequential) Parameters() []*Parameter {
	var parameters []*Parameter
	for _, layer := range s.layers {
		if provider, ok := layer.(parameterProvider); ok {
			parameters = append(parameters, provider.Parameters()...)
		}
	}
	return parameters
}
