package transformer

import (
	"fmt"
	"math"

	"m3tal-llm/internal/autograd"
	"m3tal-llm/internal/tensor"
)

// DefaultRoPEBase is the standard base frequency constant (10000.0) used in RoPE embeddings.
const DefaultRoPEBase = 10000.0

// RoPEFrequencies generates rotary embedding cos and sin tables for a given dimension and sequence length.
// The dimension dim must be even. Frequencies are dynamically computed as:
//
//	theta_i = base^(-2i / dim) for i in [0, dim/2)
//
// No hidden upper limit is placed on sequence length or dimension.
func RoPEFrequencies(dim, seqLen int, base float64) (cosTable, sinTable [][]float64, err error) {
	if dim <= 0 || dim%2 != 0 {
		return nil, nil, fmt.Errorf("%w: RoPE dimension must be positive and even, got %d", ErrInvalidDim, dim)
	}
	if seqLen < 0 {
		return nil, nil, fmt.Errorf("%w: seqLen cannot be negative, got %d", ErrInvalidDim, seqLen)
	}
	if base <= 0 {
		base = DefaultRoPEBase
	}

	halfDim := dim / 2
	cosTable = make([][]float64, seqLen)
	sinTable = make([][]float64, seqLen)

	thetas := make([]float64, halfDim)
	for i := 0; i < halfDim; i++ {
		thetas[i] = math.Pow(base, -2.0*float64(i)/float64(dim))
	}

	for m := 0; m < seqLen; m++ {
		cosRow := make([]float64, halfDim)
		sinRow := make([]float64, halfDim)
		for i := 0; i < halfDim; i++ {
			angle := float64(m) * thetas[i]
			cosRow[i] = math.Cos(angle)
			sinRow[i] = math.Sin(angle)
		}
		cosTable[m] = cosRow
		sinTable[m] = sinRow
	}

	return cosTable, sinTable, nil
}

// ApplyRoPE applies Rotary Position Embedding to input variable x.
// Input x can have arbitrary batch/head/sequence shapes:
//   - 2D: [seqLen, dim]
//   - 3D: [batch, seqLen, dim]
//   - 4D: [batch, seqLen, numHeads, headDim] or [batch, numHeads, seqLen, headDim]
//
// Parameter seqDim specifies which dimension index represents the sequence length (default: 0 if 2D, 1 if 3D/4D).
// If seqDim < 0, it defaults to:
//   - 0 for rank 2 ([seqLen, dim])
//   - 1 for rank >= 3 ([batch, seqLen, ...])
//
// posOffset allows rotary embeddings for cached tokens at an arbitrary starting position m = posOffset + i.
// The rotation on each 2D vector pair [x_{2i}, x_{2i+1}] is:
//
//	[ x_{2i} * cos(m*theta_i) - x_{2i+1} * sin(m*theta_i),
//	  x_{2i} * sin(m*theta_i) + x_{2i+1} * cos(m*theta_i) ]
//
// The backward pass applies the orthogonal transpose rotation (rotation by -angle).
func ApplyRoPE(x *autograd.Variable, posOffset int, base float64, seqDim ...int) (*autograd.Variable, error) {
	if x == nil || x.Data == nil {
		return nil, ErrNilInput
	}
	if base <= 0 {
		base = DefaultRoPEBase
	}
	if posOffset < 0 {
		return nil, fmt.Errorf("%w: posOffset cannot be negative, got %d", ErrInvalidDim, posOffset)
	}

	shape := x.Shape()
	rank := len(shape)
	if rank < 2 {
		return nil, fmt.Errorf("%w: RoPE requires rank >= 2, got rank %d", ErrShapeMismatch, rank)
	}

	dim := shape[rank-1]
	if dim <= 0 || dim%2 != 0 {
		return nil, fmt.Errorf("%w: RoPE last dimension must be positive and even, got %d", ErrInvalidDim, dim)
	}

	// Determine sequence dimension index
	sDim := 1
	if rank == 2 {
		sDim = 0
	}
	if len(seqDim) > 0 && seqDim[0] >= 0 {
		sDim = seqDim[0]
	}
	if sDim < 0 || sDim >= rank-1 {
		return nil, fmt.Errorf("%w: invalid sequence dimension index %d for rank %d", ErrInvalidDim, sDim, rank)
	}

	_ = shape[sDim]
	halfDim := dim / 2

	thetas := make([]float64, halfDim)
	for i := 0; i < halfDim; i++ {
		thetas[i] = math.Pow(base, -2.0*float64(i)/float64(dim))
	}

	dtype := x.DType()
	outTensor, err := tensor.New(shape, dtype)
	if err != nil {
		return nil, err
	}

	numel := x.Data.Numel()
	numPairs := numel / 2

	switch dtype {
	case tensor.Float32:
		for p := 0; p < numPairs; p++ {
			flatIdx0 := p * 2
			flatIdx1 := flatIdx0 + 1

			coord0, err := flatToIndices(flatIdx0, shape)
			if err != nil {
				return nil, err
			}
			coord1, err := flatToIndices(flatIdx1, shape)
			if err != nil {
				return nil, err
			}

			// Sequence position m
			seqPos := coord0[sDim] + posOffset
			// Frequency index i
			i := coord0[rank-1] / 2
			angle := float64(seqPos) * thetas[i]
			cosVal := float32(math.Cos(angle))
			sinVal := float32(math.Sin(angle))

			v0, err := x.Data.Float32At(coord0...)
			if err != nil {
				return nil, err
			}
			v1, err := x.Data.Float32At(coord1...)
			if err != nil {
				return nil, err
			}

			out0 := v0*cosVal - v1*sinVal
			out1 := v0*sinVal + v1*cosVal

			if err := outTensor.SetFloat32At(out0, coord0...); err != nil {
				return nil, err
			}
			if err := outTensor.SetFloat32At(out1, coord1...); err != nil {
				return nil, err
			}
		}

	case tensor.Float64:
		for p := 0; p < numPairs; p++ {
			flatIdx0 := p * 2
			flatIdx1 := flatIdx0 + 1

			coord0, err := flatToIndices(flatIdx0, shape)
			if err != nil {
				return nil, err
			}
			coord1, err := flatToIndices(flatIdx1, shape)
			if err != nil {
				return nil, err
			}

			seqPos := coord0[sDim] + posOffset
			i := coord0[rank-1] / 2
			angle := float64(seqPos) * thetas[i]
			cosVal := math.Cos(angle)
			sinVal := math.Sin(angle)

			v0, err := x.Data.Float64At(coord0...)
			if err != nil {
				return nil, err
			}
			v1, err := x.Data.Float64At(coord1...)
			if err != nil {
				return nil, err
			}

			out0 := v0*cosVal - v1*sinVal
			out1 := v0*sinVal + v1*cosVal

			if err := outTensor.SetFloat64At(out0, coord0...); err != nil {
				return nil, err
			}
			if err := outTensor.SetFloat64At(out1, coord1...); err != nil {
				return nil, err
			}
		}

	default:
		return nil, fmt.Errorf("%w: RoPE not implemented for dtype %s", autograd.ErrUnsupportedOp, dtype)
	}

	parents := make([]*autograd.Variable, 0, 1)
	if x.RequiresGrad {
		parents = append(parents, x)
	}

	out, err := autograd.NewVariable(outTensor, x.RequiresGrad)
	if err != nil {
		return nil, err
	}

	if x.RequiresGrad {
		out.SetBackward(parents, func() error {
			if out.Grad == nil {
				return nil
			}

			gradIn, err := tensor.New(shape, dtype)
			if err != nil {
				return err
			}

			// Rotation matrix is R = [cos, -sin; sin, cos]
			// Reverse pass: dL/dx = R^T * dL/dy
			// R^T = [cos, sin; -sin, cos]
			switch dtype {
			case tensor.Float32:
				for p := 0; p < numPairs; p++ {
					flatIdx0 := p * 2
					flatIdx1 := flatIdx0 + 1

					coord0, err := flatToIndices(flatIdx0, shape)
					if err != nil {
						return err
					}
					coord1, err := flatToIndices(flatIdx1, shape)
					if err != nil {
						return err
					}

					seqPos := coord0[sDim] + posOffset
					i := coord0[rank-1] / 2
					angle := float64(seqPos) * thetas[i]
					cosVal := float32(math.Cos(angle))
					sinVal := float32(math.Sin(angle))

					dy0, err := out.Grad.Float32At(coord0...)
					if err != nil {
						return err
					}
					dy1, err := out.Grad.Float32At(coord1...)
					if err != nil {
						return err
					}

					// R^T * [dy0, dy1]^T:
					// dx0 = dy0*cos + dy1*sin
					// dx1 = -dy0*sin + dy1*cos
					dx0 := dy0*cosVal + dy1*sinVal
					dx1 := -dy0*sinVal + dy1*cosVal

					if err := gradIn.SetFloat32At(dx0, coord0...); err != nil {
						return err
					}
					if err := gradIn.SetFloat32At(dx1, coord1...); err != nil {
						return err
					}
				}

			case tensor.Float64:
				for p := 0; p < numPairs; p++ {
					flatIdx0 := p * 2
					flatIdx1 := flatIdx0 + 1

					coord0, err := flatToIndices(flatIdx0, shape)
					if err != nil {
						return err
					}
					coord1, err := flatToIndices(flatIdx1, shape)
					if err != nil {
						return err
					}

					seqPos := coord0[sDim] + posOffset
					i := coord0[rank-1] / 2
					angle := float64(seqPos) * thetas[i]
					cosVal := math.Cos(angle)
					sinVal := math.Sin(angle)

					dy0, err := out.Grad.Float64At(coord0...)
					if err != nil {
						return err
					}
					dy1, err := out.Grad.Float64At(coord1...)
					if err != nil {
						return err
					}

					dx0 := dy0*cosVal + dy1*sinVal
					dx1 := -dy0*sinVal + dy1*cosVal

					if err := gradIn.SetFloat64At(dx0, coord0...); err != nil {
						return err
					}
					if err := gradIn.SetFloat64At(dx1, coord1...); err != nil {
						return err
					}
				}
			}

			return x.AccGrad(gradIn)
		})
	}

	return out, nil
}
