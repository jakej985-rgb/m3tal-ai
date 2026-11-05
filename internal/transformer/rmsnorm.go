// Package transformer implements foundational transformer architecture primitives:
// RMSNorm, Rotary Position Embedding (RoPE), and SwiGLU activations.
package transformer

import (
	"errors"
	"fmt"
	"math"

	"m3tal-ai/internal/autograd"
	"m3tal-ai/internal/tensor"
)

var (
	// ErrNilInput is returned when a nil variable or tensor is passed to a transformer primitive.
	ErrNilInput = errors.New("transformer: nil input")
	// ErrInvalidDim is returned when dimension parameters are non-positive or incompatible.
	ErrInvalidDim = errors.New("transformer: invalid dimension")
	// ErrShapeMismatch is returned when input tensor shapes do not match expected shapes.
	ErrShapeMismatch = errors.New("transformer: shape mismatch")
)

// RMSNormLayer applies Root Mean Square Layer Normalization with learnable scale parameter weight.
type RMSNormLayer struct {
	Dim    int
	Eps    float64
	Weight *autograd.Variable
}

// NewRMSNorm initializes an RMSNormLayer for hidden size dim.
// The weight parameter is initialized to all ones (1.0).
func NewRMSNorm(dim int, eps float64) (*RMSNormLayer, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("%w: dim must be positive, got %d", ErrInvalidDim, dim)
	}
	if eps <= 0 {
		eps = 1e-5
	}
	weightTensor, err := tensor.New([]int{dim}, tensor.Float32)
	if err != nil {
		return nil, err
	}
	for i := 0; i < dim; i++ {
		if err := weightTensor.SetFloat32At(1.0, i); err != nil {
			return nil, err
		}
	}
	weightVar, err := autograd.NewVariable(weightTensor, true)
	if err != nil {
		return nil, err
	}
	return &RMSNormLayer{
		Dim:    dim,
		Eps:    eps,
		Weight: weightVar,
	}, nil
}

// Forward applies RMSNorm to input x using the layer's learnable weight.
func (l *RMSNormLayer) Forward(x *autograd.Variable) (*autograd.Variable, error) {
	return RMSNorm(x, l.Weight, l.Eps)
}

// RMSNorm computes Root Mean Square Normalization over the last dimension of x:
//
//	rms(x) = sqrt( mean(x^2) + eps )
//	y = (x / rms(x)) * weight
//
// Supports tensors of any rank >= 1 (e.g. [dim], [seqLen, dim], [batch, seqLen, dim]).
// The weight parameter can either have shape [dim] or matching broadcastable trailing dimensions.
func RMSNorm(x, weight *autograd.Variable, eps float64) (*autograd.Variable, error) {
	if x == nil || x.Data == nil {
		return nil, ErrNilInput
	}
	if eps <= 0 {
		eps = 1e-5
	}

	xShape := x.Shape()
	rank := len(xShape)
	if rank < 1 {
		return nil, fmt.Errorf("%w: input must have rank >= 1", ErrShapeMismatch)
	}
	dim := xShape[rank-1]
	if dim <= 0 {
		return nil, fmt.Errorf("%w: last dimension must be positive, got %d", ErrInvalidDim, dim)
	}

	// Validate weight shape
	if weight != nil && weight.Data != nil {
		wShape := weight.Shape()
		wRank := len(wShape)
		if wRank < 1 || wShape[wRank-1] != dim {
			return nil, fmt.Errorf("%w: weight last dimension %v does not match input last dimension %d", ErrShapeMismatch, wShape, dim)
		}
		if weight.DType() != x.DType() {
			return nil, fmt.Errorf("%w: weight dtype %s does not match input dtype %s", ErrShapeMismatch, weight.DType(), x.DType())
		}
	}

	dtype := x.DType()
	outTensor, err := tensor.New(xShape, dtype)
	if err != nil {
		return nil, err
	}

	numel := x.Data.Numel()
	numRows := numel / dim

	// Intermediate storage for backward: invRMS for each row
	invRMSList := make([]float64, numRows)

	switch dtype {
	case tensor.Float32:
		eps32 := float32(eps)
		for r := 0; r < numRows; r++ {
			rowBase := r * dim
			var sumSq float32
			for c := 0; c < dim; c++ {
				idx, err := flatToIndices(rowBase+c, xShape)
				if err != nil {
					return nil, err
				}
				v, err := x.Data.Float32At(idx...)
				if err != nil {
					return nil, err
				}
				sumSq += v * v
			}
			meanSq := sumSq / float32(dim)
			invRMS := float32(1.0 / math.Sqrt(float64(meanSq+eps32)))
			invRMSList[r] = float64(invRMS)

			for c := 0; c < dim; c++ {
				idx, err := flatToIndices(rowBase+c, xShape)
				if err != nil {
					return nil, err
				}
				v, err := x.Data.Float32At(idx...)
				if err != nil {
					return nil, err
				}
				normed := v * invRMS
				if weight != nil && weight.Data != nil {
					wIdx := broadcastWeightIndex(idx, weight.Shape())
					wVal, err := weight.Data.Float32At(wIdx...)
					if err != nil {
						return nil, err
					}
					normed *= wVal
				}
				if err := outTensor.SetFloat32At(normed, idx...); err != nil {
					return nil, err
				}
			}
		}

	case tensor.Float64:
		for r := 0; r < numRows; r++ {
			rowBase := r * dim
			var sumSq float64
			for c := 0; c < dim; c++ {
				idx, err := flatToIndices(rowBase+c, xShape)
				if err != nil {
					return nil, err
				}
				v, err := x.Data.Float64At(idx...)
				if err != nil {
					return nil, err
				}
				sumSq += v * v
			}
			meanSq := sumSq / float64(dim)
			invRMS := 1.0 / math.Sqrt(meanSq+eps)
			invRMSList[r] = invRMS

			for c := 0; c < dim; c++ {
				idx, err := flatToIndices(rowBase+c, xShape)
				if err != nil {
					return nil, err
				}
				v, err := x.Data.Float64At(idx...)
				if err != nil {
					return nil, err
				}
				normed := v * invRMS
				if weight != nil && weight.Data != nil {
					wIdx := broadcastWeightIndex(idx, weight.Shape())
					wVal, err := weight.Data.Float64At(wIdx...)
					if err != nil {
						return nil, err
					}
					normed *= wVal
				}
				if err := outTensor.SetFloat64At(normed, idx...); err != nil {
					return nil, err
				}
			}
		}

	default:
		return nil, fmt.Errorf("%w: RMSNorm not implemented for dtype %s", autograd.ErrUnsupportedOp, dtype)
	}

	requiresGrad := x.RequiresGrad || (weight != nil && weight.RequiresGrad)
	parents := make([]*autograd.Variable, 0, 2)
	if x.RequiresGrad {
		parents = append(parents, x)
	}
	if weight != nil && weight.RequiresGrad {
		parents = append(parents, weight)
	}

	out, err := autograd.NewVariable(outTensor, requiresGrad)
	if err != nil {
		return nil, err
	}

	if requiresGrad {
		out.SetBackward(parents, func() error {
			if out.Grad == nil {
				return nil
			}

			// 1. Compute weight gradient: dL/dw_c = sum_{r} ( dL/dy_{r,c} * x_{r,c} * invRMS_r )
			if weight != nil && weight.RequiresGrad {
				wGrad, err := tensor.New(weight.Shape(), dtype)
				if err != nil {
					return err
				}
				switch dtype {
				case tensor.Float32:
					for r := 0; r < numRows; r++ {
						invRMS := float32(invRMSList[r])
						rowBase := r * dim
						for c := 0; c < dim; c++ {
							idx, err := flatToIndices(rowBase+c, xShape)
							if err != nil {
								return err
							}
							xVal, err := x.Data.Float32At(idx...)
							if err != nil {
								return err
							}
							dy, err := out.Grad.Float32At(idx...)
							if err != nil {
								return err
							}
							dw := dy * (xVal * invRMS)
							wIdx := broadcastWeightIndex(idx, weight.Shape())
							currW, err := wGrad.Float32At(wIdx...)
							if err != nil {
								return err
							}
							if err := wGrad.SetFloat32At(currW+dw, wIdx...); err != nil {
								return err
							}
						}
					}
				case tensor.Float64:
					for r := 0; r < numRows; r++ {
						invRMS := invRMSList[r]
						rowBase := r * dim
						for c := 0; c < dim; c++ {
							idx, err := flatToIndices(rowBase+c, xShape)
							if err != nil {
								return err
							}
							xVal, err := x.Data.Float64At(idx...)
							if err != nil {
								return err
							}
							dy, err := out.Grad.Float64At(idx...)
							if err != nil {
								return err
							}
							dw := dy * (xVal * invRMS)
							wIdx := broadcastWeightIndex(idx, weight.Shape())
							currW, err := wGrad.Float64At(wIdx...)
							if err != nil {
								return err
							}
							if err := wGrad.SetFloat64At(currW+dw, wIdx...); err != nil {
								return err
							}
						}
					}
				}
				if err := weight.AccGrad(wGrad); err != nil {
					return err
				}
			}

			// 2. Compute input gradient:
			// Let g_i = dy_i * w_i
			// dL/dx_i = invRMS * ( g_i - (x_i * invRMS^2 / dim) * sum_j(g_j * x_j) )
			if x.RequiresGrad {
				xGrad, err := tensor.New(xShape, dtype)
				if err != nil {
					return err
				}
				dimFloat := float64(dim)

				switch dtype {
				case tensor.Float32:
					dimF32 := float32(dimFloat)
					for r := 0; r < numRows; r++ {
						invRMS := float32(invRMSList[r])
						invRMS3 := invRMS * invRMS * invRMS
						rowBase := r * dim

						// Calculate dot product sum_j (g_j * x_j)
						var dot float32
						for c := 0; c < dim; c++ {
							idx, err := flatToIndices(rowBase+c, xShape)
							if err != nil {
								return err
							}
							xVal, err := x.Data.Float32At(idx...)
							if err != nil {
								return err
							}
							dy, err := out.Grad.Float32At(idx...)
							if err != nil {
								return err
							}
							wVal := float32(1.0)
							if weight != nil && weight.Data != nil {
								wIdx := broadcastWeightIndex(idx, weight.Shape())
								wVal, err = weight.Data.Float32At(wIdx...)
								if err != nil {
									return err
								}
							}
							g := dy * wVal
							dot += g * xVal
						}

						// Calculate dx_c
						for c := 0; c < dim; c++ {
							idx, err := flatToIndices(rowBase+c, xShape)
							if err != nil {
								return err
							}
							xVal, err := x.Data.Float32At(idx...)
							if err != nil {
								return err
							}
							dy, err := out.Grad.Float32At(idx...)
							if err != nil {
								return err
							}
							wVal := float32(1.0)
							if weight != nil && weight.Data != nil {
								wIdx := broadcastWeightIndex(idx, weight.Shape())
								wVal, err = weight.Data.Float32At(wIdx...)
								if err != nil {
									return err
								}
							}
							g := dy * wVal
							dx := invRMS*g - (invRMS3/dimF32)*xVal*dot
							if err := xGrad.SetFloat32At(dx, idx...); err != nil {
								return err
							}
						}
					}

				case tensor.Float64:
					for r := 0; r < numRows; r++ {
						invRMS := invRMSList[r]
						invRMS3 := invRMS * invRMS * invRMS
						rowBase := r * dim

						// Calculate dot product sum_j (g_j * x_j)
						var dot float64
						for c := 0; c < dim; c++ {
							idx, err := flatToIndices(rowBase+c, xShape)
							if err != nil {
								return err
							}
							xVal, err := x.Data.Float64At(idx...)
							if err != nil {
								return err
							}
							dy, err := out.Grad.Float64At(idx...)
							if err != nil {
								return err
							}
							wVal := 1.0
							if weight != nil && weight.Data != nil {
								wIdx := broadcastWeightIndex(idx, weight.Shape())
								wVal, err = weight.Data.Float64At(wIdx...)
								if err != nil {
									return err
								}
							}
							g := dy * wVal
							dot += g * xVal
						}

						// Calculate dx_c
						for c := 0; c < dim; c++ {
							idx, err := flatToIndices(rowBase+c, xShape)
							if err != nil {
								return err
							}
							xVal, err := x.Data.Float64At(idx...)
							if err != nil {
								return err
							}
							dy, err := out.Grad.Float64At(idx...)
							if err != nil {
								return err
							}
							wVal := 1.0
							if weight != nil && weight.Data != nil {
								wIdx := broadcastWeightIndex(idx, weight.Shape())
								wVal, err = weight.Data.Float64At(wIdx...)
								if err != nil {
									return err
								}
							}
							g := dy * wVal
							dx := invRMS*g - (invRMS3/dimFloat)*xVal*dot
							if err := xGrad.SetFloat64At(dx, idx...); err != nil {
								return err
							}
						}
					}
				}
				if err := x.AccGrad(xGrad); err != nil {
					return err
				}
			}

			return nil
		})
	}

	return out, nil
}

// broadcastWeightIndex projects a coordinate index from input shape to weight shape.
func broadcastWeightIndex(idx []int, wShape []int) []int {
	wRank := len(wShape)
	res := make([]int, wRank)
	rankDiff := len(idx) - wRank
	for i := 0; i < wRank; i++ {
		dim := wShape[i]
		if dim == 1 {
			res[i] = 0
		} else {
			res[i] = idx[i+rankDiff]
		}
	}
	return res
}

// flatToIndices converts a flat linear element index into N-D coordinate indices.
func flatToIndices(flat int, shape []int) ([]int, error) {
	return autograd.FlatToIndices(flat, shape)
}
