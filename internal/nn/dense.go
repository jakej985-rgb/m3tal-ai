// Package nn contains reusable neural-network layers built on the tensor math.
package nn

import (
	"fmt"
	"math"
	"math/rand"

	"m3tal-ai/internal/tensor"
)

// Parameter stores trainable values and their accumulated gradients.
type Parameter struct {
	values []float64
	grads  []float64
}

// Values returns a copy of the parameter values.
func (p *Parameter) Values() []float64 { return append([]float64(nil), p.values...) }

// Gradients returns a copy of the accumulated parameter gradients.
func (p *Parameter) Gradients() []float64 { return append([]float64(nil), p.grads...) }

// Dense is a trainable affine layer. It accepts a batch whose rows are samples
// and computes X*W+b.
type Dense struct {
	inputSize  int
	outputSize int
	weights    *Parameter
	bias       *Parameter
	lastInput  tensor.Matrix
	hasInput   bool
}

// NewDense creates a layer with weights initialized uniformly in
// [-1/sqrt(inputSize), 1/sqrt(inputSize)] and zero biases.
func NewDense(inputSize, outputSize int, rng *rand.Rand) (*Dense, error) {
	if inputSize <= 0 || outputSize <= 0 {
		return nil, fmt.Errorf("layer sizes must be positive: input=%d output=%d", inputSize, outputSize)
	}
	if rng == nil {
		return nil, fmt.Errorf("random source must not be nil")
	}

	limit := 1 / math.Sqrt(float64(inputSize))
	weights := make([]float64, inputSize*outputSize)
	for i := range weights {
		weights[i] = (rng.Float64()*2 - 1) * limit
	}
	return &Dense{
		inputSize:  inputSize,
		outputSize: outputSize,
		weights:    &Parameter{values: weights, grads: make([]float64, len(weights))},
		bias:       &Parameter{values: make([]float64, outputSize), grads: make([]float64, outputSize)},
	}, nil
}

// Forward computes X*W+b for each row in input and caches input for Backward.
func (d *Dense) Forward(input tensor.Matrix) (tensor.Matrix, error) {
	if input.Cols() != d.inputSize {
		return tensor.Matrix{}, fmt.Errorf("dense layer expected %d input features, got %d", d.inputSize, input.Cols())
	}
	weights, err := tensor.NewMatrix(d.inputSize, d.outputSize, d.weights.values)
	if err != nil {
		return tensor.Matrix{}, err
	}
	product, err := input.MatMul(weights)
	if err != nil {
		return tensor.Matrix{}, err
	}
	values := make([]float64, product.Rows()*product.Cols())
	for row := 0; row < product.Rows(); row++ {
		for col := 0; col < product.Cols(); col++ {
			values[row*product.Cols()+col] = product.At(row, col) + d.bias.values[col]
		}
	}
	output, err := tensor.NewMatrix(product.Rows(), product.Cols(), values)
	if err != nil {
		return tensor.Matrix{}, err
	}
	d.lastInput = input
	d.hasInput = true
	return output, nil
}

// Backward propagates outputGrad through the most recent Forward call,
// accumulating parameter gradients and returning the gradient with respect to
// the input.
func (d *Dense) Backward(outputGrad tensor.Matrix) (tensor.Matrix, error) {
	if !d.hasInput {
		return tensor.Matrix{}, fmt.Errorf("dense backward called before forward")
	}
	if outputGrad.Rows() != d.lastInput.Rows() || outputGrad.Cols() != d.outputSize {
		return tensor.Matrix{}, fmt.Errorf("dense gradient shape mismatch: got %dx%d, want %dx%d", outputGrad.Rows(), outputGrad.Cols(), d.lastInput.Rows(), d.outputSize)
	}
	weightGrad, err := d.lastInput.Transpose().MatMul(outputGrad)
	if err != nil {
		return tensor.Matrix{}, err
	}
	for row := 0; row < d.inputSize; row++ {
		for col := 0; col < d.outputSize; col++ {
			index := row*d.outputSize + col
			d.weights.grads[index] += weightGrad.At(row, col)
		}
	}
	for row := 0; row < outputGrad.Rows(); row++ {
		for col := 0; col < d.outputSize; col++ {
			d.bias.grads[col] += outputGrad.At(row, col)
		}
	}
	weights, err := tensor.NewMatrix(d.inputSize, d.outputSize, d.weights.values)
	if err != nil {
		return tensor.Matrix{}, err
	}
	return outputGrad.MatMul(weights.Transpose())
}

// ZeroGrad clears the layer's accumulated parameter gradients.
func (d *Dense) ZeroGrad() {
	clear(d.weights.grads)
	clear(d.bias.grads)
}

// Parameters returns this layer's trainable weights and biases.
func (d *Dense) Parameters() []*Parameter { return []*Parameter{d.weights, d.bias} }

// Gradients returns copies of weight and bias gradients in their storage order.
func (d *Dense) Gradients() ([]float64, []float64) {
	return d.weights.Gradients(), d.bias.Gradients()
}

// Weights returns the current weight matrix as a copy.
func (d *Dense) Weights() (tensor.Matrix, error) {
	return tensor.NewMatrix(d.inputSize, d.outputSize, d.weights.values)
}

// Bias returns a copy of the current bias vector.
func (d *Dense) Bias() tensor.Vector { return append(tensor.Vector(nil), d.bias.values...) }
