package tensor

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Add performs element-wise addition between a and b with broadcasting.
func Add(a, b *Tensor) (*Tensor, error) {
	return elementwiseBinary(a, b, "Add", func(x, y float64) float64 { return x + y })
}

// Sub performs element-wise subtraction (a - b) with broadcasting.
func Sub(a, b *Tensor) (*Tensor, error) {
	return elementwiseBinary(a, b, "Sub", func(x, y float64) float64 { return x - y })
}

// Mul performs element-wise multiplication between a and b with broadcasting.
func Mul(a, b *Tensor) (*Tensor, error) {
	return elementwiseBinary(a, b, "Mul", func(x, y float64) float64 { return x * y })
}

// Div performs element-wise division (a / b) with broadcasting.
func Div(a, b *Tensor) (*Tensor, error) {
	return elementwiseBinary(a, b, "Div", func(x, y float64) float64 { return x / y })
}

// Neg returns a new tensor with each element negated.
func (t *Tensor) Neg() (*Tensor, error) {
	if !t.device.IsCPU() {
		return nil, fmt.Errorf("%w: Neg only supported on CPU", ErrUnsupportedDevice)
	}
	out, err := New(t.shape, t.dtype)
	if err != nil {
		return nil, err
	}
	numel := t.Numel()
	if numel == 0 {
		return out, nil
	}

	indices := make([]int, len(t.shape))
	dstBytes := out.storage.Bytes()
	srcBytes := t.storage.Bytes()

	switch t.dtype {
	case Float32:
		for i := 0; i < numel; i++ {
			srcOff, err := ComputeOffset(indices, t.shape, t.strides, t.offset)
			if err != nil {
				return nil, err
			}
			bits := binary.LittleEndian.Uint32(srcBytes[srcOff*4 : (srcOff+1)*4])
			val := -math.Float32frombits(bits)
			binary.LittleEndian.PutUint32(dstBytes[i*4:(i+1)*4], math.Float32bits(val))
			advanceIndices(indices, t.shape)
		}
	case Float64:
		for i := 0; i < numel; i++ {
			srcOff, err := ComputeOffset(indices, t.shape, t.strides, t.offset)
			if err != nil {
				return nil, err
			}
			bits := binary.LittleEndian.Uint64(srcBytes[srcOff*8 : (srcOff+1)*8])
			val := -math.Float64frombits(bits)
			binary.LittleEndian.PutUint64(dstBytes[i*8:(i+1)*8], math.Float64bits(val))
			advanceIndices(indices, t.shape)
		}
	default:
		return nil, fmt.Errorf("%w: Neg not supported for dtype %s", ErrUnsupportedDType, t.dtype)
	}

	return out, nil
}


func elementwiseBinary(a, b *Tensor, opName string, op func(float64, float64) float64) (*Tensor, error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("%w: input tensor cannot be nil", ErrInvalidShape)
	}
	if !a.device.IsCPU() || !b.device.IsCPU() {
		return nil, fmt.Errorf("%w: %s only supported on CPU", ErrUnsupportedDevice, opName)
	}
	if a.dtype != b.dtype {
		return nil, fmt.Errorf("%w: %s requires identical dtypes, got %s and %s", ErrUnsupportedDType, opName, a.dtype, b.dtype)
	}
	if a.dtype != Float32 && a.dtype != Float64 {
		return nil, fmt.Errorf("%w: %s not supported for dtype %s", ErrUnsupportedDType, opName, a.dtype)
	}

	targetShape, err := BroadcastShapes(a.shape, b.shape)
	if err != nil {
		return nil, err
	}

	aView, err := a.BroadcastTo(targetShape)
	if err != nil {
		return nil, err
	}
	bView, err := b.BroadcastTo(targetShape)
	if err != nil {
		return nil, err
	}

	out, err := New(targetShape, a.dtype)
	if err != nil {
		return nil, err
	}

	total := out.Numel()
	if total == 0 {
		return out, nil
	}

	indices := make([]int, len(targetShape))
	aBytes := aView.storage.Bytes()
	bBytes := bView.storage.Bytes()
	dstBytes := out.storage.Bytes()

	if a.dtype == Float32 {
		for i := 0; i < total; i++ {
			aOff, err := ComputeOffset(indices, aView.shape, aView.strides, aView.offset)
			if err != nil {
				return nil, err
			}
			bOff, err := ComputeOffset(indices, bView.shape, bView.strides, bView.offset)
			if err != nil {
				return nil, err
			}

			aBits := binary.LittleEndian.Uint32(aBytes[aOff*4 : (aOff+1)*4])
			bBits := binary.LittleEndian.Uint32(bBytes[bOff*4 : (bOff+1)*4])
			valA := float64(math.Float32frombits(aBits))
			valB := float64(math.Float32frombits(bBits))
			res := float32(op(valA, valB))

			binary.LittleEndian.PutUint32(dstBytes[i*4:(i+1)*4], math.Float32bits(res))
			advanceIndices(indices, targetShape)
		}
	} else {
		for i := 0; i < total; i++ {
			aOff, err := ComputeOffset(indices, aView.shape, aView.strides, aView.offset)
			if err != nil {
				return nil, err
			}
			bOff, err := ComputeOffset(indices, bView.shape, bView.strides, bView.offset)
			if err != nil {
				return nil, err
			}

			aBits := binary.LittleEndian.Uint64(aBytes[aOff*8 : (aOff+1)*8])
			bBits := binary.LittleEndian.Uint64(bBytes[bOff*8 : (bOff+1)*8])
			valA := math.Float64frombits(aBits)
			valB := math.Float64frombits(bBits)
			res := op(valA, valB)

			binary.LittleEndian.PutUint64(dstBytes[i*8:(i+1)*8], math.Float64bits(res))
			advanceIndices(indices, targetShape)
		}
	}

	return out, nil
}

func advanceIndices(indices, shape []int) {
	for d := len(shape) - 1; d >= 0; d-- {
		indices[d]++
		if indices[d] < shape[d] {
			return
		}
		indices[d] = 0
	}
}
