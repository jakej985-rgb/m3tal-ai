package nn

import (
	"fmt"
	"math"

	"m3tal-llm/internal/tensor"
)

// Loss computes a scalar objective and its gradient with respect to predictions.
type Loss interface {
	Value(prediction, targets tensor.Matrix) (float64, error)
	Gradient(prediction, targets tensor.Matrix) (tensor.Matrix, error)
}

// MSE is the mean squared error objective.
type MSE struct{}

// Value returns mean squared error.
func (MSE) Value(prediction, targets tensor.Matrix) (float64, error) {
	return MeanSquaredError(prediction, targets)
}

// Gradient returns the derivative of mean squared error with respect to predictions.
func (MSE) Gradient(prediction, targets tensor.Matrix) (tensor.Matrix, error) {
	return MeanSquaredErrorGradient(prediction, targets)
}

// SoftmaxCrossEntropy computes categorical cross-entropy directly from logits.
// Targets must contain one-hot or probability labels with each row summing to 1.
type SoftmaxCrossEntropy struct{}

// Value returns the batch-mean categorical cross-entropy.
func (SoftmaxCrossEntropy) Value(logits, targets tensor.Matrix) (float64, error) {
	if err := validateClassificationShapes(logits, targets); err != nil {
		return 0, err
	}
	var loss float64
	for row := 0; row < logits.Rows(); row++ {
		maxLogit := rowMax(logits, row)
		var expSum float64
		for col := 0; col < logits.Cols(); col++ {
			expSum += math.Exp(logits.At(row, col) - maxLogit)
		}
		logNormalizer := maxLogit + math.Log(expSum)
		for col := 0; col < logits.Cols(); col++ {
			loss -= targets.At(row, col) * (logits.At(row, col) - logNormalizer)
		}
	}
	return loss / float64(logits.Rows()), nil
}

// Gradient returns the softmax-cross-entropy derivative with respect to logits.
func (SoftmaxCrossEntropy) Gradient(logits, targets tensor.Matrix) (tensor.Matrix, error) {
	if err := validateClassificationShapes(logits, targets); err != nil {
		return tensor.Matrix{}, err
	}
	values := make([]float64, logits.Rows()*logits.Cols())
	for row := 0; row < logits.Rows(); row++ {
		maxLogit := rowMax(logits, row)
		var expSum float64
		for col := 0; col < logits.Cols(); col++ {
			expSum += math.Exp(logits.At(row, col) - maxLogit)
		}
		for col := 0; col < logits.Cols(); col++ {
			probability := math.Exp(logits.At(row, col)-maxLogit) / expSum
			values[row*logits.Cols()+col] = (probability - targets.At(row, col)) / float64(logits.Rows())
		}
	}
	return tensor.NewMatrix(logits.Rows(), logits.Cols(), values)
}

func validateClassificationShapes(logits, targets tensor.Matrix) error {
	if err := sameShape("softmax cross-entropy", logits, targets); err != nil {
		return err
	}
	if logits.Rows() == 0 || logits.Cols() == 0 {
		return fmt.Errorf("softmax cross-entropy requires a non-empty matrix")
	}
	for row := 0; row < targets.Rows(); row++ {
		var total float64
		for col := 0; col < targets.Cols(); col++ {
			value := targets.At(row, col)
			if value < 0 {
				return fmt.Errorf("classification target at (%d,%d) is negative", row, col)
			}
			total += value
		}
		if math.Abs(total-1) > 1e-6 {
			return fmt.Errorf("classification target row %d sums to %g, want 1", row, total)
		}
	}
	return nil
}

func rowMax(matrix tensor.Matrix, row int) float64 {
	maximum := matrix.At(row, 0)
	for col := 1; col < matrix.Cols(); col++ {
		if matrix.At(row, col) > maximum {
			maximum = matrix.At(row, col)
		}
	}
	return maximum
}

// MeanSquaredError returns the mean squared difference between two matrices.
func MeanSquaredError(prediction, targets tensor.Matrix) (float64, error) {
	if err := sameShape("loss", prediction, targets); err != nil {
		return 0, err
	}
	count := prediction.Rows() * prediction.Cols()
	if count == 0 {
		return 0, fmt.Errorf("mean squared error requires at least one value")
	}
	var loss float64
	for row := 0; row < prediction.Rows(); row++ {
		for col := 0; col < prediction.Cols(); col++ {
			difference := prediction.At(row, col) - targets.At(row, col)
			loss += difference * difference
		}
	}
	return loss / float64(count), nil
}

// MeanSquaredErrorGradient returns d(loss)/d(prediction).
func MeanSquaredErrorGradient(prediction, targets tensor.Matrix) (tensor.Matrix, error) {
	if err := sameShape("loss gradient", prediction, targets); err != nil {
		return tensor.Matrix{}, err
	}
	count := prediction.Rows() * prediction.Cols()
	if count == 0 {
		return tensor.Matrix{}, fmt.Errorf("mean squared error requires at least one value")
	}
	values := make([]float64, count)
	for row := 0; row < prediction.Rows(); row++ {
		for col := 0; col < prediction.Cols(); col++ {
			index := row*prediction.Cols() + col
			values[index] = 2 * (prediction.At(row, col) - targets.At(row, col)) / float64(count)
		}
	}
	return tensor.NewMatrix(prediction.Rows(), prediction.Cols(), values)
}

func sameShape(operation string, left, right tensor.Matrix) error {
	if left.Rows() != right.Rows() || left.Cols() != right.Cols() {
		return fmt.Errorf("%s shape mismatch: %dx%d and %dx%d", operation, left.Rows(), left.Cols(), right.Rows(), right.Cols())
	}
	return nil
}
