package nn

import (
	"fmt"

	"m3tal-llm/internal/tensor"
)

// TanhLayer applies the tanh activation element-wise and caches its output for
// the matching backward pass.
type TanhLayer struct {
	lastOutput tensor.Matrix
	hasOutput  bool
}

// Forward applies tanh and saves the result for Backward.
func (layer *TanhLayer) Forward(input tensor.Matrix) (tensor.Matrix, error) {
	output, err := Tanh(input)
	if err != nil {
		return tensor.Matrix{}, err
	}
	layer.lastOutput = output
	layer.hasOutput = true
	return output, nil
}

// Backward applies the derivative of tanh to upstream gradients.
func (layer *TanhLayer) Backward(outputGrad tensor.Matrix) (tensor.Matrix, error) {
	if !layer.hasOutput {
		return tensor.Matrix{}, fmt.Errorf("tanh backward called before forward")
	}
	return TanhBackward(layer.lastOutput, outputGrad)
}

// ZeroGrad is a no-op because tanh has no trainable parameters.
func (layer *TanhLayer) ZeroGrad() {}
