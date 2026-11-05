package autograd

import (
	"fmt"
	"math"

	"m3tal-llm/internal/tensor"
)

// Tanh applies element-wise hyperbolic tangent: tanh(x).
// dL/dx = dL/dout * (1 - tanh(x)^2).
func Tanh(a *Variable) (*Variable, error) {
	if a == nil || a.Data == nil {
		return nil, ErrNilTensor
	}
	outTensor, err := tensor.New(a.Shape(), a.DType())
	if err != nil {
		return nil, err
	}
	numel := outTensor.Numel()
	switch a.DType() {
	case tensor.Float32:
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, a.Shape())
			if err != nil {
				return nil, err
			}
			v, err := a.Data.Float32At(idx...)
			if err != nil {
				return nil, err
			}
			res := float32(math.Tanh(float64(v)))
			if err := outTensor.SetFloat32At(res, idx...); err != nil {
				return nil, err
			}
		}
	case tensor.Float64:
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, a.Shape())
			if err != nil {
				return nil, err
			}
			v, err := a.Data.Float64At(idx...)
			if err != nil {
				return nil, err
			}
			res := math.Tanh(v)
			if err := outTensor.SetFloat64At(res, idx...); err != nil {
				return nil, err
			}
		}
	default:
		return nil, fmt.Errorf("%w: Tanh for %s", ErrUnsupportedOp, a.DType())
	}

	out := &Variable{
		Data:         outTensor,
		RequiresGrad: a.RequiresGrad,
	}
	if a.RequiresGrad {
		out.parents = []*Variable{a}
		out.backward = func() error {
			if out.Grad == nil {
				return nil
			}
			gradIn, err := tensor.New(a.Shape(), a.DType())
			if err != nil {
				return err
			}
			switch a.DType() {
			case tensor.Float32:
				for i := 0; i < numel; i++ {
					idx, err := flatToIndices(i, a.Shape())
					if err != nil {
						return err
					}
					y, err := outTensor.Float32At(idx...)
					if err != nil {
						return err
					}
					g, err := out.Grad.Float32At(idx...)
					if err != nil {
						return err
					}
					dx := g * (1.0 - y*y)
					if err := gradIn.SetFloat32At(dx, idx...); err != nil {
						return err
					}
				}
			case tensor.Float64:
				for i := 0; i < numel; i++ {
					idx, err := flatToIndices(i, a.Shape())
					if err != nil {
						return err
					}
					y, err := outTensor.Float64At(idx...)
					if err != nil {
						return err
					}
					g, err := out.Grad.Float64At(idx...)
					if err != nil {
						return err
					}
					dx := g * (1.0 - y*y)
					if err := gradIn.SetFloat64At(dx, idx...); err != nil {
						return err
					}
				}
			}
			return a.AccGrad(gradIn)
		}
	}
	return out, nil
}

// ReLU applies element-wise rectified linear unit: max(0, x).
// dL/dx = dL/dout if x > 0 else 0.
func ReLU(a *Variable) (*Variable, error) {
	if a == nil || a.Data == nil {
		return nil, ErrNilTensor
	}
	outTensor, err := tensor.New(a.Shape(), a.DType())
	if err != nil {
		return nil, err
	}
	numel := outTensor.Numel()
	switch a.DType() {
	case tensor.Float32:
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, a.Shape())
			if err != nil {
				return nil, err
			}
			v, err := a.Data.Float32At(idx...)
			if err != nil {
				return nil, err
			}
			res := v
			if res < 0 {
				res = 0
			}
			if err := outTensor.SetFloat32At(res, idx...); err != nil {
				return nil, err
			}
		}
	case tensor.Float64:
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, a.Shape())
			if err != nil {
				return nil, err
			}
			v, err := a.Data.Float64At(idx...)
			if err != nil {
				return nil, err
			}
			res := v
			if res < 0 {
				res = 0
			}
			if err := outTensor.SetFloat64At(res, idx...); err != nil {
				return nil, err
			}
		}
	default:
		return nil, fmt.Errorf("%w: ReLU for %s", ErrUnsupportedOp, a.DType())
	}

	out := &Variable{
		Data:         outTensor,
		RequiresGrad: a.RequiresGrad,
	}
	if a.RequiresGrad {
		out.parents = []*Variable{a}
		out.backward = func() error {
			if out.Grad == nil {
				return nil
			}
			gradIn, err := tensor.New(a.Shape(), a.DType())
			if err != nil {
				return err
			}
			switch a.DType() {
			case tensor.Float32:
				for i := 0; i < numel; i++ {
					idx, err := flatToIndices(i, a.Shape())
					if err != nil {
						return err
					}
					x, err := a.Data.Float32At(idx...)
					if err != nil {
						return err
					}
					g, err := out.Grad.Float32At(idx...)
					if err != nil {
						return err
					}
					var dx float32
					if x > 0 {
						dx = g
					}
					if err := gradIn.SetFloat32At(dx, idx...); err != nil {
						return err
					}
				}
			case tensor.Float64:
				for i := 0; i < numel; i++ {
					idx, err := flatToIndices(i, a.Shape())
					if err != nil {
						return err
					}
					x, err := a.Data.Float64At(idx...)
					if err != nil {
						return err
					}
					g, err := out.Grad.Float64At(idx...)
					if err != nil {
						return err
					}
					var dx float64
					if x > 0 {
						dx = g
					}
					if err := gradIn.SetFloat64At(dx, idx...); err != nil {
						return err
					}
				}
			}
			return a.AccGrad(gradIn)
		}
	}
	return out, nil
}

// Exp computes element-wise exponential: e^x.
// dL/dx = dL/dout * e^x.
func Exp(a *Variable) (*Variable, error) {
	if a == nil || a.Data == nil {
		return nil, ErrNilTensor
	}
	outTensor, err := tensor.New(a.Shape(), a.DType())
	if err != nil {
		return nil, err
	}
	numel := outTensor.Numel()
	switch a.DType() {
	case tensor.Float32:
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, a.Shape())
			if err != nil {
				return nil, err
			}
			v, err := a.Data.Float32At(idx...)
			if err != nil {
				return nil, err
			}
			res := float32(math.Exp(float64(v)))
			if err := outTensor.SetFloat32At(res, idx...); err != nil {
				return nil, err
			}
		}
	case tensor.Float64:
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, a.Shape())
			if err != nil {
				return nil, err
			}
			v, err := a.Data.Float64At(idx...)
			if err != nil {
				return nil, err
			}
			res := math.Exp(v)
			if err := outTensor.SetFloat64At(res, idx...); err != nil {
				return nil, err
			}
		}
	default:
		return nil, fmt.Errorf("%w: Exp for %s", ErrUnsupportedOp, a.DType())
	}

	out := &Variable{
		Data:         outTensor,
		RequiresGrad: a.RequiresGrad,
	}
	if a.RequiresGrad {
		out.parents = []*Variable{a}
		out.backward = func() error {
			if out.Grad == nil {
				return nil
			}
			gradIn, err := elemwiseMul(out.Grad, outTensor)
			if err != nil {
				return err
			}
			return a.AccGrad(gradIn)
		}
	}
	return out, nil
}

// Log computes element-wise natural logarithm: ln(x). Requires x > 0.
// dL/dx = dL/dout / x.
func Log(a *Variable) (*Variable, error) {
	if a == nil || a.Data == nil {
		return nil, ErrNilTensor
	}
	outTensor, err := tensor.New(a.Shape(), a.DType())
	if err != nil {
		return nil, err
	}
	numel := outTensor.Numel()
	switch a.DType() {
	case tensor.Float32:
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, a.Shape())
			if err != nil {
				return nil, err
			}
			v, err := a.Data.Float32At(idx...)
			if err != nil {
				return nil, err
			}
			if v <= 0 {
				return nil, fmt.Errorf("%w: log domain error: value %f <= 0", ErrUnsupportedOp, v)
			}
			res := float32(math.Log(float64(v)))
			if err := outTensor.SetFloat32At(res, idx...); err != nil {
				return nil, err
			}
		}
	case tensor.Float64:
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, a.Shape())
			if err != nil {
				return nil, err
			}
			v, err := a.Data.Float64At(idx...)
			if err != nil {
				return nil, err
			}
			if v <= 0 {
				return nil, fmt.Errorf("%w: log domain error: value %f <= 0", ErrUnsupportedOp, v)
			}
			res := math.Log(v)
			if err := outTensor.SetFloat64At(res, idx...); err != nil {
				return nil, err
			}
		}
	default:
		return nil, fmt.Errorf("%w: Log for %s", ErrUnsupportedOp, a.DType())
	}

	out := &Variable{
		Data:         outTensor,
		RequiresGrad: a.RequiresGrad,
	}
	if a.RequiresGrad {
		out.parents = []*Variable{a}
		out.backward = func() error {
			if out.Grad == nil {
				return nil
			}
			gradIn, err := elemwiseDiv(out.Grad, a.Data)
			if err != nil {
				return err
			}
			return a.AccGrad(gradIn)
		}
	}
	return out, nil
}

// Neg computes element-wise negation: -x.
// dL/dx = -dL/dout.
func Neg(a *Variable) (*Variable, error) {
	return Scale(a, -1.0)
}

// Pow computes element-wise power: x^p for constant scalar p.
// dL/dx = dL/dout * p * x^(p-1).
func Pow(a *Variable, p float64) (*Variable, error) {
	if a == nil || a.Data == nil {
		return nil, ErrNilTensor
	}
	outTensor, err := tensor.New(a.Shape(), a.DType())
	if err != nil {
		return nil, err
	}
	numel := outTensor.Numel()
	switch a.DType() {
	case tensor.Float32:
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, a.Shape())
			if err != nil {
				return nil, err
			}
			v, err := a.Data.Float32At(idx...)
			if err != nil {
				return nil, err
			}
			res := float32(math.Pow(float64(v), p))
			if err := outTensor.SetFloat32At(res, idx...); err != nil {
				return nil, err
			}
		}
	case tensor.Float64:
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, a.Shape())
			if err != nil {
				return nil, err
			}
			v, err := a.Data.Float64At(idx...)
			if err != nil {
				return nil, err
			}
			res := math.Pow(v, p)
			if err := outTensor.SetFloat64At(res, idx...); err != nil {
				return nil, err
			}
		}
	default:
		return nil, fmt.Errorf("%w: Pow for %s", ErrUnsupportedOp, a.DType())
	}

	out := &Variable{
		Data:         outTensor,
		RequiresGrad: a.RequiresGrad,
	}
	if a.RequiresGrad {
		out.parents = []*Variable{a}
		out.backward = func() error {
			if out.Grad == nil {
				return nil
			}
			gradIn, err := tensor.New(a.Shape(), a.DType())
			if err != nil {
				return err
			}
			switch a.DType() {
			case tensor.Float32:
				p32 := float32(p)
				for i := 0; i < numel; i++ {
					idx, err := flatToIndices(i, a.Shape())
					if err != nil {
						return err
					}
					x, err := a.Data.Float32At(idx...)
					if err != nil {
						return err
					}
					g, err := out.Grad.Float32At(idx...)
					if err != nil {
						return err
					}
					dx := g * p32 * float32(math.Pow(float64(x), p-1.0))
					if err := gradIn.SetFloat32At(dx, idx...); err != nil {
						return err
					}
				}
			case tensor.Float64:
				for i := 0; i < numel; i++ {
					idx, err := flatToIndices(i, a.Shape())
					if err != nil {
						return err
					}
					x, err := a.Data.Float64At(idx...)
					if err != nil {
						return err
					}
					g, err := out.Grad.Float64At(idx...)
					if err != nil {
						return err
					}
					dx := g * p * math.Pow(x, p-1.0)
					if err := gradIn.SetFloat64At(dx, idx...); err != nil {
						return err
					}
				}
			}
			return a.AccGrad(gradIn)
		}
	}
	return out, nil
}

// Sum computes the sum of all elements in a (or across all dimensions), returning a scalar Tensor.
func Sum(a *Variable) (*Variable, error) {
	if a == nil || a.Data == nil {
		return nil, ErrNilTensor
	}
	outTensor, err := tensor.New([]int{1}, a.DType())
	if err != nil {
		return nil, err
	}
	numel := a.Data.Numel()
	switch a.DType() {
	case tensor.Float32:
		var sum float32
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, a.Shape())
			if err != nil {
				return nil, err
			}
			v, err := a.Data.Float32At(idx...)
			if err != nil {
				return nil, err
			}
			sum += v
		}
		if err := outTensor.SetFloat32At(sum, 0); err != nil {
			return nil, err
		}
	case tensor.Float64:
		var sum float64
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, a.Shape())
			if err != nil {
				return nil, err
			}
			v, err := a.Data.Float64At(idx...)
			if err != nil {
				return nil, err
			}
			sum += v
		}
		if err := outTensor.SetFloat64At(sum, 0); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("%w: Sum for %s", ErrUnsupportedOp, a.DType())
	}

	out := &Variable{
		Data:         outTensor,
		RequiresGrad: a.RequiresGrad,
	}
	if a.RequiresGrad {
		out.parents = []*Variable{a}
		out.backward = func() error {
			if out.Grad == nil {
				return nil
			}
			// Broadcast scalar grad to full input shape
			gradIn, err := tensor.New(a.Shape(), a.DType())
			if err != nil {
				return err
			}
			switch a.DType() {
			case tensor.Float32:
				g, err := out.Grad.Float32At(0)
				if err != nil {
					return err
				}
				for i := 0; i < numel; i++ {
					idx, err := flatToIndices(i, a.Shape())
					if err != nil {
						return err
					}
					if err := gradIn.SetFloat32At(g, idx...); err != nil {
						return err
					}
				}
			case tensor.Float64:
				g, err := out.Grad.Float64At(0)
				if err != nil {
					return err
				}
				for i := 0; i < numel; i++ {
					idx, err := flatToIndices(i, a.Shape())
					if err != nil {
						return err
					}
					if err := gradIn.SetFloat64At(g, idx...); err != nil {
						return err
					}
				}
			}
			return a.AccGrad(gradIn)
		}
	}
	return out, nil
}

// Mean computes the average across all elements in a, returning a scalar Tensor.
func Mean(a *Variable) (*Variable, error) {
	if a == nil || a.Data == nil {
		return nil, ErrNilTensor
	}
	numel := a.Data.Numel()
	if numel == 0 {
		return nil, fmt.Errorf("%w: Mean on empty tensor", ErrUnsupportedOp)
	}
	sumVar, err := Sum(a)
	if err != nil {
		return nil, err
	}
	return Scale(sumVar, 1.0/float64(numel))
}

// Reshape changes the shape of a Variable.
func Reshape(a *Variable, newShape []int) (*Variable, error) {
	if a == nil || a.Data == nil {
		return nil, ErrNilTensor
	}
	// Verify contiguity or clone to ensure reshape succeeds
	var dataToReshape *tensor.Tensor
	if a.Data.IsContiguous() {
		dataToReshape = a.Data
	} else {
		var err error
		dataToReshape, err = a.Data.Clone()
		if err != nil {
			return nil, err
		}
	}
	reshapedData, err := dataToReshape.Reshape(newShape)
	if err != nil {
		return nil, err
	}

	out := &Variable{
		Data:         reshapedData,
		RequiresGrad: a.RequiresGrad,
	}
	if a.RequiresGrad {
		out.parents = []*Variable{a}
		out.backward = func() error {
			if out.Grad == nil {
				return nil
			}
			// Un-reshape grad back to a.Shape()
			var gradContig *tensor.Tensor
			if out.Grad.IsContiguous() {
				gradContig = out.Grad
			} else {
				var err error
				gradContig, err = out.Grad.Clone()
				if err != nil {
					return err
				}
			}
			gradIn, err := gradContig.Reshape(a.Shape())
			if err != nil {
				return err
			}
			return a.AccGrad(gradIn)
		}
	}
	return out, nil
}

// Transpose swaps two dimensions in 2D or N-D tensors (specifically matrix transpose over last two dimensions).
func Transpose(a *Variable) (*Variable, error) {
	if a == nil || a.Data == nil {
		return nil, ErrNilTensor
	}
	if a.Data.Rank() < 2 {
		return nil, fmt.Errorf("%w: Transpose requires rank >= 2, got rank %d", ErrUnsupportedOp, a.Data.Rank())
	}
	transData, err := transposeMatrixAxes(a.Data)
	if err != nil {
		return nil, err
	}
	out := &Variable{
		Data:         transData,
		RequiresGrad: a.RequiresGrad,
	}
	if a.RequiresGrad {
		out.parents = []*Variable{a}
		out.backward = func() error {
			if out.Grad == nil {
				return nil
			}
			gradIn, err := transposeMatrixAxes(out.Grad)
			if err != nil {
				return err
			}
			return a.AccGrad(gradIn)
		}
	}
	return out, nil
}
