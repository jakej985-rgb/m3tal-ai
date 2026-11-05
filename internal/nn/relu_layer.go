package nn

import (
	"fmt"

	"m3tal-llm/internal/tensor"
)

// ReLULayer applies max(0, x) element-wise and caches its output for
// backpropagation.
type ReLULayer struct {
	lastOutput tensor.Matrix
	hasOutput  bool
}

// Forward applies ReLU and saves the result for Backward.
func (layer *ReLULayer) Forward(input tensor.Matrix) (tensor.Matrix, error) {
	values := make([]float64, input.Rows()*input.Cols())
	for row := 0; row < input.Rows(); row++ {
		for col := 0; col < input.Cols(); col++ {
			if value := input.At(row, col); value > 0 {
				values[row*input.Cols()+col] = value
			}
		}
	}
	output, err := tensor.NewMatrix(input.Rows(), input.Cols(), values)
	if err != nil {
		return tensor.Matrix{}, err
	}
	layer.lastOutput = output
	layer.hasOutput = true
	return output, nil
}

// Backward zeros gradients where the corresponding forward activation was 0.
func (layer *ReLULayer) Backward(outputGrad tensor.Matrix) (tensor.Matrix, error) {
	if !layer.hasOutput {
		return tensor.Matrix{}, fmt.Errorf("ReLU backward called before forward")
	}
	if outputGrad.Rows() != layer.lastOutput.Rows() || outputGrad.Cols() != layer.lastOutput.Cols() {
		return tensor.Matrix{}, fmt.Errorf("ReLU gradient shape mismatch")
	}
	values := make([]float64, outputGrad.Rows()*outputGrad.Cols())
	for row := 0; row < outputGrad.Rows(); row++ {
		for col := 0; col < outputGrad.Cols(); col++ {
			if layer.lastOutput.At(row, col) > 0 {
				values[row*outputGrad.Cols()+col] = outputGrad.At(row, col)
			}
		}
	}
	return tensor.NewMatrix(outputGrad.Rows(), outputGrad.Cols(), values)
}

// ZeroGrad is a no-op because ReLU has no trainable parameters.
func (layer *ReLULayer) ZeroGrad() {}
