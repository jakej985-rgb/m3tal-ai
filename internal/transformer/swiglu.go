package transformer

import (
	"fmt"
	"math"
	"math/rand"

	"m3tal-ai/internal/autograd"
	"m3tal-ai/internal/tensor"
)

// SiLU applies the Sigmoid Linear Unit (Swish-1) activation function elementwise:
//
//	SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// The derivative is:
//
//	d/dx SiLU(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
//	             = sigma * (1 + x * (1 - sigma))
//	dL/dx = dL/dy * (sigma * (1 + x * (1 - sigma)))
func SiLU(x *autograd.Variable) (*autograd.Variable, error) {
	if x == nil || x.Data == nil {
		return nil, ErrNilInput
	}

	shape := x.Shape()
	dtype := x.DType()
	outTensor, err := tensor.New(shape, dtype)
	if err != nil {
		return nil, err
	}

	numel := x.Data.Numel()

	switch dtype {
	case tensor.Float32:
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, shape)
			if err != nil {
				return nil, err
			}
			v, err := x.Data.Float32At(idx...)
			if err != nil {
				return nil, err
			}
			sigma := float32(1.0 / (1.0 + math.Exp(-float64(v))))
			res := v * sigma
			if err := outTensor.SetFloat32At(res, idx...); err != nil {
				return nil, err
			}
		}

	case tensor.Float64:
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, shape)
			if err != nil {
				return nil, err
			}
			v, err := x.Data.Float64At(idx...)
			if err != nil {
				return nil, err
			}
			sigma := 1.0 / (1.0 + math.Exp(-v))
			res := v * sigma
			if err := outTensor.SetFloat64At(res, idx...); err != nil {
				return nil, err
			}
		}

	default:
		return nil, fmt.Errorf("%w: SiLU not implemented for dtype %s", autograd.ErrUnsupportedOp, dtype)
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

			switch dtype {
			case tensor.Float32:
				for i := 0; i < numel; i++ {
					idx, err := flatToIndices(i, shape)
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
					sigma := float32(1.0 / (1.0 + math.Exp(-float64(xVal))))
					dSiLU := sigma * (1.0 + xVal*(1.0-sigma))
					dx := dy * dSiLU
					if err := gradIn.SetFloat32At(dx, idx...); err != nil {
						return err
					}
				}

			case tensor.Float64:
				for i := 0; i < numel; i++ {
					idx, err := flatToIndices(i, shape)
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
					sigma := 1.0 / (1.0 + math.Exp(-xVal))
					dSiLU := sigma * (1.0 + xVal*(1.0-sigma))
					dx := dy * dSiLU
					if err := gradIn.SetFloat64At(dx, idx...); err != nil {
						return err
					}
				}
			}

			return x.AccGrad(gradIn)
		})
	}

	return out, nil
}

// SwiGLU computes the SwiGLU gated linear unit activation:
//
//	SwiGLU(gate, up) = SiLU(gate) * up
//
// Inputs gate and up must have matching shapes and data types.
// The backward pass computes:
//
//	dL/d(gate) = dL/dy * up * d/d(gate)[SiLU(gate)]
//	dL/d(up)   = dL/dy * SiLU(gate)
func SwiGLU(gate, up *autograd.Variable) (*autograd.Variable, error) {
	if gate == nil || up == nil || gate.Data == nil || up.Data == nil {
		return nil, ErrNilInput
	}
	if gate.DType() != up.DType() {
		return nil, fmt.Errorf("%w: SwiGLU input dtypes mismatch: %s and %s", ErrShapeMismatch, gate.DType(), up.DType())
	}

	gShape := gate.Shape()
	uShape := up.Shape()
	if len(gShape) != len(uShape) {
		return nil, fmt.Errorf("%w: SwiGLU input shapes mismatch: %v and %v", ErrShapeMismatch, gShape, uShape)
	}
	for i := range gShape {
		if gShape[i] != uShape[i] {
			return nil, fmt.Errorf("%w: SwiGLU input shapes mismatch: %v and %v", ErrShapeMismatch, gShape, uShape)
		}
	}

	siluGate, err := SiLU(gate)
	if err != nil {
		return nil, err
	}

	return autograd.Mul(siluGate, up)
}

// SwiGLULayer represents a feed-forward SwiGLU network block:
//
//	hidden = SwiGLU( x * W_gate, x * W_up )
//	output = hidden * W_down
//
// Parameters:
//
//	W_gate: [dim, hiddenDim]
//	W_up:   [dim, hiddenDim]
//	W_down: [hiddenDim, dim]
type SwiGLULayer struct {
	Dim       int
	HiddenDim int
	WGate     *autograd.Variable
	WUp       *autograd.Variable
	WDown     *autograd.Variable
}

// NewSwiGLU initializes a SwiGLULayer with Xavier/Glorot-like uniform initialization.
// hiddenDim is often set to (4 * dim * 2 / 3) rounded to a multiple of 64 or 256 in standard LLaMA-style architectures.
func NewSwiGLU(dim, hiddenDim int, rng *rand.Rand) (*SwiGLULayer, error) {
	if dim <= 0 || hiddenDim <= 0 {
		return nil, fmt.Errorf("%w: dim (%d) and hiddenDim (%d) must be positive", ErrInvalidDim, dim, hiddenDim)
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}

	initWeight := func(inD, outD int) (*autograd.Variable, error) {
		t, err := tensor.New([]int{inD, outD}, tensor.Float32)
		if err != nil {
			return nil, err
		}
		limit := float32(1.0 / math.Sqrt(float64(inD)))
		numel := inD * outD
		for i := 0; i < numel; i++ {
			idx, _ := flatToIndices(i, []int{inD, outD})
			val := (rng.Float32()*2.0 - 1.0) * limit
			if err := t.SetFloat32At(val, idx...); err != nil {
				return nil, err
			}
		}
		return autograd.NewVariable(t, true)
	}

	wGate, err := initWeight(dim, hiddenDim)
	if err != nil {
		return nil, err
	}
	wUp, err := initWeight(dim, hiddenDim)
	if err != nil {
		return nil, err
	}
	wDown, err := initWeight(hiddenDim, dim)
	if err != nil {
		return nil, err
	}

	return &SwiGLULayer{
		Dim:       dim,
		HiddenDim: hiddenDim,
		WGate:     wGate,
		WUp:       wUp,
		WDown:     wDown,
	}, nil
}

// Forward executes the feed-forward pass: (SiLU(x * WGate) * (x * WUp)) * WDown.
// Supports arbitrary batched tensor shapes (e.g. [batch, seqLen, dim]).
func (l *SwiGLULayer) Forward(x *autograd.Variable) (*autograd.Variable, error) {
	if x == nil || x.Data == nil {
		return nil, ErrNilInput
	}
	gateProj, err := autograd.MatMul(x, l.WGate)
	if err != nil {
		return nil, err
	}
	upProj, err := autograd.MatMul(x, l.WUp)
	if err != nil {
		return nil, err
	}
	hidden, err := SwiGLU(gateProj, upProj)
	if err != nil {
		return nil, err
	}
	return autograd.MatMul(hidden, l.WDown)
}
