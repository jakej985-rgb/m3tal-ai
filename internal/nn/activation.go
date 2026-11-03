package nn

import (
	"fmt"
	"math"

	"m3tal-ai/internal/tensor"
)

// Tanh applies tanh element-wise to a matrix.
func Tanh(input tensor.Matrix) (tensor.Matrix, error) {
	values := make([]float64, input.Rows()*input.Cols())
	for row := 0; row < input.Rows(); row++ {
		for col := 0; col < input.Cols(); col++ {
			values[row*input.Cols()+col] = math.Tanh(input.At(row, col))
		}
	}
	return tensor.NewMatrix(input.Rows(), input.Cols(), values)
}

// TanhBackward multiplies upstream gradients by tanh's derivative, using the
// already-computed tanh output: d/dx tanh(x) = 1 - tanh(x)^2.
func TanhBackward(output, upstream tensor.Matrix) (tensor.Matrix, error) {
	if output.Rows() != upstream.Rows() || output.Cols() != upstream.Cols() {
		return tensor.Matrix{}, fmt.Errorf("tanh gradient shape mismatch: output is %dx%d, gradient is %dx%d", output.Rows(), output.Cols(), upstream.Rows(), upstream.Cols())
	}
	values := make([]float64, output.Rows()*output.Cols())
	for row := 0; row < output.Rows(); row++ {
		for col := 0; col < output.Cols(); col++ {
			value := output.At(row, col)
			values[row*output.Cols()+col] = upstream.At(row, col) * (1 - value*value)
		}
	}
	return tensor.NewMatrix(output.Rows(), output.Cols(), values)
}
