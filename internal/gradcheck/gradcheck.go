// Package gradcheck provides a finite-difference gradient checking harness
// comparing automatic differentiation or analytical gradients with numerical approximations.
package gradcheck

import (
	"fmt"
	"math"

	"m3tal-ai/internal/tensor"
)

// Config configures finite-difference perturbation parameters and tolerances.
type Config struct {
	// Epsilon is the perturbation step size for two-sided finite difference.
	Epsilon float64
	// AbsoluteTolerance is the maximum allowable absolute difference between analytic and numerical gradients.
	AbsoluteTolerance float64
	// RelativeTolerance is the maximum allowable relative error: |analytic - num| / max(1, |analytic|, |num|).
	RelativeTolerance float64
}

// DefaultConfig returns default finite-difference settings suitable for float64 computations.
func DefaultConfig() Config {
	return Config{
		Epsilon:           1e-6,
		AbsoluteTolerance: 1e-5,
		RelativeTolerance: 1e-4,
	}
}

// Float32Config returns finite-difference settings suitable for float32 computations.
func Float32Config() Config {
	return Config{
		Epsilon:           1e-3,
		AbsoluteTolerance: 5e-2,
		RelativeTolerance: 5e-2,
	}
}

// GradientMismatchError records a failure where analytic and numerical gradients disagree.
type GradientMismatchError struct {
	OpName            string
	ParamName         string
	Index             string
	Analytic          float64
	Numerical         float64
	AbsoluteDiff      float64
	RelativeDiff      float64
	AbsoluteTolerance float64
	RelativeTolerance float64
}

// Error formats the failure details including operation, param, index, analytic vs numerical, and diffs.
func (e *GradientMismatchError) Error() string {
	return fmt.Sprintf("gradient check failed for op %q (param %q at index %s): analytic=%g, numerical=%g, abs_diff=%g (tol=%g), rel_diff=%g (tol=%g)",
		e.OpName, e.ParamName, e.Index, e.Analytic, e.Numerical, e.AbsoluteDiff, e.AbsoluteTolerance, e.RelativeDiff, e.RelativeTolerance)
}

// checkTolerance verifies if analytic and numerical values are within tolerance.
func checkTolerance(opName, paramName, index string, analytic, numerical float64, cfg Config) error {
	if math.IsNaN(analytic) || math.IsNaN(numerical) {
		return fmt.Errorf("gradient check failed for op %q (param %q at index %s): NaN encountered (analytic=%g, numerical=%g)",
			opName, paramName, index, analytic, numerical)
	}
	if math.IsInf(analytic, 0) || math.IsInf(numerical, 0) {
		return fmt.Errorf("gradient check failed for op %q (param %q at index %s): Inf encountered (analytic=%g, numerical=%g)",
			opName, paramName, index, analytic, numerical)
	}

	absDiff := math.Abs(analytic - numerical)
	scale := math.Max(1.0, math.Max(math.Abs(analytic), math.Abs(numerical)))
	relDiff := absDiff / scale

	if absDiff > cfg.AbsoluteTolerance && relDiff > cfg.RelativeTolerance {
		return &GradientMismatchError{
			OpName:            opName,
			ParamName:         paramName,
			Index:             index,
			Analytic:          analytic,
			Numerical:         numerical,
			AbsoluteDiff:      absDiff,
			RelativeDiff:      relDiff,
			AbsoluteTolerance: cfg.AbsoluteTolerance,
			RelativeTolerance: cfg.RelativeTolerance,
		}
	}
	return nil
}

// CheckScalar checks the gradient of a scalar function f(x) with respect to scalar parameter x.
// evalFn evaluates the scalar objective given x.
func CheckScalar(opName, paramName string, x, analyticGrad float64, evalFn func(x float64) float64, cfg Config) error {
	eps := cfg.Epsilon
	if eps <= 0 {
		eps = 1e-6
	}
	fPlus := evalFn(x + eps)
	fMinus := evalFn(x - eps)
	numGrad := (fPlus - fMinus) / (2 * eps)
	return checkTolerance(opName, paramName, "[0]", analyticGrad, numGrad, cfg)
}

// CheckSlice checks gradients for a 1D slice of parameters.
// evalFn computes the scalar loss given values slice.
func CheckSlice(opName, paramName string, values, analyticGrads []float64, evalFn func(v []float64) float64, cfg Config) error {
	if len(values) != len(analyticGrads) {
		return fmt.Errorf("gradient check %q: length mismatch: values len %d vs analyticGrads len %d", opName, len(values), len(analyticGrads))
	}
	eps := cfg.Epsilon
	if eps <= 0 {
		eps = 1e-6
	}

	vCopy := append([]float64(nil), values...)
	for i := range vCopy {
		orig := vCopy[i]
		vCopy[i] = orig + eps
		fPlus := evalFn(vCopy)

		vCopy[i] = orig - eps
		fMinus := evalFn(vCopy)

		vCopy[i] = orig
		numGrad := (fPlus - fMinus) / (2 * eps)

		if err := checkTolerance(opName, paramName, fmt.Sprintf("[%d]", i), analyticGrads[i], numGrad, cfg); err != nil {
			return err
		}
	}
	return nil
}

// CheckSliceFloat32 checks gradients for a 1D slice of float32 parameters.
func CheckSliceFloat32(opName, paramName string, values, analyticGrads []float32, evalFn func(v []float32) float64, cfg Config) error {
	if len(values) != len(analyticGrads) {
		return fmt.Errorf("gradient check %q: length mismatch: values len %d vs analyticGrads len %d", opName, len(values), len(analyticGrads))
	}
	eps := float32(cfg.Epsilon)
	if eps <= 0 {
		eps = 1e-3
	}

	vCopy := append([]float32(nil), values...)
	for i := range vCopy {
		orig := vCopy[i]
		vCopy[i] = orig + eps
		fPlus := evalFn(vCopy)

		vCopy[i] = orig - eps
		fMinus := evalFn(vCopy)

		vCopy[i] = orig
		numGrad := (fPlus - fMinus) / (2 * float64(eps))

		if err := checkTolerance(opName, paramName, fmt.Sprintf("[%d]", i), float64(analyticGrads[i]), numGrad, cfg); err != nil {
			return err
		}
	}
	return nil
}

// CheckMatrix checks gradients for a 2D matrix of parameters.
// evalFn computes the scalar loss given matrix mat.
func CheckMatrix(opName, paramName string, mat, analyticGrads tensor.Matrix, evalFn func(m tensor.Matrix) (float64, error), cfg Config) error {
	if mat.Rows() != analyticGrads.Rows() || mat.Cols() != analyticGrads.Cols() {
		return fmt.Errorf("gradient check %q: matrix shape mismatch: mat %dx%d vs analyticGrads %dx%d",
			opName, mat.Rows(), mat.Cols(), analyticGrads.Rows(), analyticGrads.Cols())
	}
	eps := cfg.Epsilon
	if eps <= 0 {
		eps = 1e-6
	}

	rows, cols := mat.Rows(), mat.Cols()
	flatValues := make([]float64, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			flatValues[r*cols+c] = mat.At(r, c)
		}
	}

	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			idx := r*cols + c
			orig := flatValues[idx]

			flatValues[idx] = orig + eps
			mPlus, err := tensor.NewMatrix(rows, cols, flatValues)
			if err != nil {
				return err
			}
			fPlus, err := evalFn(mPlus)
			if err != nil {
				return err
			}

			flatValues[idx] = orig - eps
			mMinus, err := tensor.NewMatrix(rows, cols, flatValues)
			if err != nil {
				return err
			}
			fMinus, err := evalFn(mMinus)
			if err != nil {
				return err
			}

			flatValues[idx] = orig
			numGrad := (fPlus - fMinus) / (2 * eps)
			analytic := analyticGrads.At(r, c)

			if err := checkTolerance(opName, paramName, fmt.Sprintf("[%d,%d]", r, c), analytic, numGrad, cfg); err != nil {
				return err
			}
		}
	}
	return nil
}
