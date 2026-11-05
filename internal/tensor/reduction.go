package tensor

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Sum computes the sum of tensor elements along specified axes.
// If axes is empty, all elements are summed into a scalar tensor (or rank-preserved 1s if keepDims is true).
func Sum(t *Tensor, axes []int, keepDims bool) (*Tensor, error) {
	return reduceOp(t, axes, keepDims, "Sum",
		func(accum, val float64, count int) float64 {
			return accum + val
		},
		func(accum float64, count int) float64 {
			return accum
		},
		0.0,
	)
}

// Mean computes the arithmetic mean of tensor elements along specified axes.
func Mean(t *Tensor, axes []int, keepDims bool) (*Tensor, error) {
	return reduceOp(t, axes, keepDims, "Mean",
		func(accum, val float64, count int) float64 {
			return accum + val
		},
		func(accum float64, count int) float64 {
			if count == 0 {
				return math.NaN()
			}
			return accum / float64(count)
		},
		0.0,
	)
}

// Max computes the maximum value along specified axes.
func Max(t *Tensor, axes []int, keepDims bool) (*Tensor, error) {
	if t.Numel() == 0 {
		return nil, fmt.Errorf("%w: cannot compute Max on empty tensor", ErrIndexOutOfBounds)
	}
	return reduceOp(t, axes, keepDims, "Max",
		func(accum, val float64, count int) float64 {
			if count == 0 || val > accum {
				return val
			}
			return accum
		},
		func(accum float64, count int) float64 {
			return accum
		},
		math.Inf(-1),
	)
}

// Min computes the minimum value along specified axes.
func Min(t *Tensor, axes []int, keepDims bool) (*Tensor, error) {
	if t.Numel() == 0 {
		return nil, fmt.Errorf("%w: cannot compute Min on empty tensor", ErrIndexOutOfBounds)
	}
	return reduceOp(t, axes, keepDims, "Min",
		func(accum, val float64, count int) float64 {
			if count == 0 || val < accum {
				return val
			}
			return accum
		},
		func(accum float64, count int) float64 {
			return accum
		},
		math.Inf(1),
	)
}

func reduceOp(t *Tensor, axes []int, keepDims bool, opName string,
	accumulate func(accum, val float64, count int) float64,
	finalize func(accum float64, count int) float64,
	initialVal float64) (*Tensor, error) {

	if t == nil {
		return nil, fmt.Errorf("%w: input tensor cannot be nil", ErrInvalidShape)
	}
	if !t.device.IsCPU() {
		return nil, fmt.Errorf("%w: %s only supported on CPU", ErrUnsupportedDevice, opName)
	}
	if t.dtype != Float32 && t.dtype != Float64 {
		return nil, fmt.Errorf("%w: %s not supported for dtype %s", ErrUnsupportedDType, opName, t.dtype)
	}

	rank := len(t.shape)
	isAxisReduced := make([]bool, rank)

	if len(axes) == 0 {
		for i := 0; i < rank; i++ {
			isAxisReduced[i] = true
		}
	} else {
		for _, axis := range axes {
			origAxis := axis
			if axis < 0 {
				axis += rank
			}
			if axis < 0 || axis >= rank {
				return nil, fmt.Errorf("%w: axis %d out of bounds for rank %d", ErrIndexOutOfBounds, origAxis, rank)
			}
			if isAxisReduced[axis] {
				return nil, fmt.Errorf("%w: duplicate axis %d in reduction", ErrInvalidShape, origAxis)
			}
			isAxisReduced[axis] = true
		}
	}

	// Calculate output shape
	outShape := make([]int, 0, rank)
	for i := 0; i < rank; i++ {
		if isAxisReduced[i] {
			if keepDims {
				outShape = append(outShape, 1)
			}
		} else {
			outShape = append(outShape, t.shape[i])
		}
	}

	out, err := New(outShape, t.dtype)
	if err != nil {
		return nil, err
	}

	outNumel := out.Numel()
	if outNumel == 0 {
		return out, nil
	}

	// Count number of reduced elements per output cell
	reducedCount := 1
	for i := 0; i < rank; i++ {
		if isAxisReduced[i] {
			reducedCount *= t.shape[i]
		}
	}

	// Iterate over all output positions, and for each output position iterate over reduced axes
	outIndices := make([]int, len(outShape))
	srcIndices := make([]int, rank)

	srcBytes := t.storage.Bytes()
	dstBytes := out.storage.Bytes()

	for o := 0; o < outNumel; o++ {
		// Map outIndices to the unreduced dimensions of srcIndices
		outIdx := 0
		for i := 0; i < rank; i++ {
			if isAxisReduced[i] {
				srcIndices[i] = 0
			} else {
				if keepDims {
					srcIndices[i] = outIndices[i]
				} else {
					srcIndices[i] = outIndices[outIdx]
					outIdx++
				}
			}
		}

		accum := initialVal
		for c := 0; c < reducedCount; c++ {
			srcOff, err := ComputeOffset(srcIndices, t.shape, t.strides, t.offset)
			if err != nil {
				return nil, err
			}

			var val float64
			if t.dtype == Float32 {
				bits := binary.LittleEndian.Uint32(srcBytes[srcOff*4 : (srcOff+1)*4])
				val = float64(math.Float32frombits(bits))
			} else {
				bits := binary.LittleEndian.Uint64(srcBytes[srcOff*8 : (srcOff+1)*8])
				val = math.Float64frombits(bits)
			}

			accum = accumulate(accum, val, c)

			// Advance only the reduced axes in srcIndices
			for d := rank - 1; d >= 0; d-- {
				if isAxisReduced[d] {
					srcIndices[d]++
					if srcIndices[d] < t.shape[d] {
						break
					}
					srcIndices[d] = 0
				}
			}
		}

		finalVal := finalize(accum, reducedCount)
		if t.dtype == Float32 {
			binary.LittleEndian.PutUint32(dstBytes[o*4:(o+1)*4], math.Float32bits(float32(finalVal)))
		} else {
			binary.LittleEndian.PutUint64(dstBytes[o*8:(o+1)*8], math.Float64bits(finalVal))
		}

		advanceIndices(outIndices, outShape)
	}

	return out, nil
}
