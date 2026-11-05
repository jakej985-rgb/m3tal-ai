package tensor

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"m3tal-llm/internal/tensor/kernel"
)

// MatMul computes the matrix multiplication of two 2D tensors: C = A * B.
// Both tensors must reside on the CPU, have rank 2, and have compatible dimensions
// (a.shape[1] == b.shape[0]).
func (t *Tensor) MatMul(other *Tensor) (*Tensor, error) {
	if !t.device.IsCPU() || !other.device.IsCPU() {
		return nil, fmt.Errorf("%w: MatMul currently only supported on CPU", ErrUnsupportedDevice)
	}
	if t.Rank() != 2 || other.Rank() != 2 {
		return nil, fmt.Errorf("%w: MatMul requires 2D tensors, got ranks %d and %d", ErrShapeMismatch, t.Rank(), other.Rank())
	}
	if t.dtype != other.dtype {
		return nil, fmt.Errorf("%w: dtype mismatch %s vs %s", ErrUnsupportedDType, t.dtype, other.dtype)
	}
	M, K1 := t.shape[0], t.shape[1]
	K2, N := other.shape[0], other.shape[1]
	if K1 != K2 {
		return nil, fmt.Errorf("%w: inner dimension mismatch %d != %d", ErrShapeMismatch, K1, K2)
	}

	out, err := New([]int{M, N}, t.dtype)
	if err != nil {
		return nil, err
	}

	if M == 0 || N == 0 || K1 == 0 {
		return out, nil
	}

	// Ensure inputs are contiguous for high-performance blocked / SIMD kernels
	aContig := t
	if !t.IsContiguous() {
		aContig, err = t.Clone()
		if err != nil {
			return nil, err
		}
	}
	bContig := other
	if !other.IsContiguous() {
		bContig, err = other.Clone()
		if err != nil {
			return nil, err
		}
	}

	switch t.dtype {
	case Float32:
		aData := aContig.Float32Slice()
		bData := bContig.Float32Slice()
		cData := out.Float32Slice()
		kernel.MatMulFloat32Dispatch(M, N, K1, aData, bData, cData)
		out.SetFromFloat32Slice(cData)
	case Float64:
		aData := aContig.Float64Slice()
		bData := bContig.Float64Slice()
		cData := out.Float64Slice()
		kernel.MatMulFloat64Dispatch(M, N, K1, aData, bData, cData)
		out.SetFromFloat64Slice(cData)
	default:
		return nil, fmt.Errorf("%w: MatMul not implemented for %s", ErrUnsupportedDType, t.dtype)
	}

	return out, nil
}

// Add performs elementwise addition: C = A + B.
func (t *Tensor) Add(other *Tensor) (*Tensor, error) {
	return t.elementwiseBinary(other, "Add", kernel.AddFloat32Dispatch, kernel.AddFloat64Dispatch)
}

// Sub performs elementwise subtraction: C = A - B.
func (t *Tensor) Sub(other *Tensor) (*Tensor, error) {
	return t.elementwiseBinary(other, "Sub", kernel.SubFloat32Dispatch, kernel.SubFloat64Dispatch)
}

// Mul performs elementwise multiplication (Hadamard product): C = A * B.
func (t *Tensor) Mul(other *Tensor) (*Tensor, error) {
	return t.elementwiseBinary(other, "Mul", kernel.MulFloat32Dispatch, kernel.MulFloat64Dispatch)
}

// Div performs elementwise division: C = A / B.
func (t *Tensor) Div(other *Tensor) (*Tensor, error) {
	return t.elementwiseBinary(other, "Div", kernel.DivFloat32Dispatch, kernel.DivFloat64Dispatch)
}

func (t *Tensor) elementwiseBinary(
	other *Tensor,
	opName string,
	opF32 func(dst, a, b []float32),
	opF64 func(dst, a, b []float64),
) (*Tensor, error) {
	if !t.device.IsCPU() || !other.device.IsCPU() {
		return nil, fmt.Errorf("%w: %s only supported on CPU", ErrUnsupportedDevice, opName)
	}
	if t.dtype != other.dtype {
		return nil, fmt.Errorf("%w: dtype mismatch %s vs %s", ErrUnsupportedDType, t.dtype, other.dtype)
	}
	if len(t.shape) != len(other.shape) {
		return nil, fmt.Errorf("%w: %s requires identical shapes, got %v and %v", ErrShapeMismatch, opName, t.shape, other.shape)
	}
	for i := range t.shape {
		if t.shape[i] != other.shape[i] {
			return nil, fmt.Errorf("%w: %s requires identical shapes, got %v and %v", ErrShapeMismatch, opName, t.shape, other.shape)
		}
	}

	out, err := New(t.shape, t.dtype)
	if err != nil {
		return nil, err
	}
	if t.Numel() == 0 {
		return out, nil
	}

	aContig := t
	if !t.IsContiguous() {
		aContig, err = t.Clone()
		if err != nil {
			return nil, err
		}
	}
	bContig := other
	if !other.IsContiguous() {
		bContig, err = other.Clone()
		if err != nil {
			return nil, err
		}
	}

	switch t.dtype {
	case Float32:
		cData := out.Float32Slice()
		opF32(cData, aContig.Float32Slice(), bContig.Float32Slice())
		out.SetFromFloat32Slice(cData)
	case Float64:
		cData := out.Float64Slice()
		opF64(cData, aContig.Float64Slice(), bContig.Float64Slice())
		out.SetFromFloat64Slice(cData)
	default:
		return nil, fmt.Errorf("%w: %s not implemented for %s", ErrUnsupportedDType, opName, t.dtype)
	}
	return out, nil
}

// Scale multiplies all elements of tensor by scalar alpha: C = A * alpha.
func (t *Tensor) Scale(alpha float64) (*Tensor, error) {
	if !t.device.IsCPU() {
		return nil, fmt.Errorf("%w: Scale only supported on CPU", ErrUnsupportedDevice)
	}
	out, err := New(t.shape, t.dtype)
	if err != nil {
		return nil, err
	}
	if t.Numel() == 0 {
		return out, nil
	}

	aContig := t
	if !t.IsContiguous() {
		aContig, err = t.Clone()
		if err != nil {
			return nil, err
		}
	}

	switch t.dtype {
	case Float32:
		cData := out.Float32Slice()
		kernel.ScaleFloat32Fallback(cData, aContig.Float32Slice(), float32(alpha))
		out.SetFromFloat32Slice(cData)
	case Float64:
		cData := out.Float64Slice()
		kernel.ScaleFloat64Fallback(cData, aContig.Float64Slice(), alpha)
		out.SetFromFloat64Slice(cData)
	default:
		return nil, fmt.Errorf("%w: Scale not implemented for %s", ErrUnsupportedDType, t.dtype)
	}
	return out, nil
}

// Sum computes the sum of all elements in the tensor.
func (t *Tensor) Sum() (float64, error) {
	if !t.device.IsCPU() {
		return 0, fmt.Errorf("%w: Sum only supported on CPU", ErrUnsupportedDevice)
	}
	if t.Numel() == 0 {
		return 0, nil
	}
	aContig := t
	var err error
	if !t.IsContiguous() {
		aContig, err = t.Clone()
		if err != nil {
			return 0, err
		}
	}
	switch t.dtype {
	case Float32:
		return float64(kernel.SumFloat32Fallback(aContig.Float32Slice())), nil
	case Float64:
		return kernel.SumFloat64Fallback(aContig.Float64Slice()), nil
	default:
		return 0, fmt.Errorf("%w: Sum not implemented for %s", ErrUnsupportedDType, t.dtype)
	}
}

// Float32Slice returns the values as a newly allocated []float32 slice.
func (t *Tensor) Float32Slice() []float32 {
	if t.dtype != Float32 {
		panic("Float32Slice called on non-Float32 tensor")
	}
	numel := t.Numel()
	if numel == 0 {
		return nil
	}
	raw := t.storage.Bytes()
	byteStart := t.offset * 4
	byteEnd := byteStart + numel*4
	sub := raw[byteStart:byteEnd]

	res := make([]float32, numel)
	for i := 0; i < numel; i++ {
		bits := binary.LittleEndian.Uint32(sub[i*4 : (i+1)*4])
		res[i] = math.Float32frombits(bits)
	}
	return res
}

// SetFromFloat32Slice writes a slice of float32 values into the tensor storage.
func (t *Tensor) SetFromFloat32Slice(src []float32) {
	if t.dtype != Float32 {
		panic("SetFromFloat32Slice called on non-Float32 tensor")
	}
	numel := t.Numel()
	if len(src) != numel {
		panic("SetFromFloat32Slice: length mismatch")
	}
	raw := t.storage.Bytes()
	byteStart := t.offset * 4
	for i, v := range src {
		binary.LittleEndian.PutUint32(raw[byteStart+i*4:byteStart+(i+1)*4], math.Float32bits(v))
	}
}

// Float64Slice returns the values as a newly allocated []float64 slice.
func (t *Tensor) Float64Slice() []float64 {
	if t.dtype != Float64 {
		panic("Float64Slice called on non-Float64 tensor")
	}
	numel := t.Numel()
	if numel == 0 {
		return nil
	}
	raw := t.storage.Bytes()
	byteStart := t.offset * 8
	byteEnd := byteStart + numel*8
	sub := raw[byteStart:byteEnd]

	res := make([]float64, numel)
	for i := 0; i < numel; i++ {
		bits := binary.LittleEndian.Uint64(sub[i*8 : (i+1)*8])
		res[i] = math.Float64frombits(bits)
	}
	return res
}

// SetFromFloat64Slice writes a slice of float64 values into the tensor storage.
func (t *Tensor) SetFromFloat64Slice(src []float64) {
	if t.dtype != Float64 {
		panic("SetFromFloat64Slice called on non-Float64 tensor")
	}
	numel := t.Numel()
	if len(src) != numel {
		panic("SetFromFloat64Slice: length mismatch")
	}
	raw := t.storage.Bytes()
	byteStart := t.offset * 8
	for i, v := range src {
		binary.LittleEndian.PutUint64(raw[byteStart+i*8:byteStart+(i+1)*8], math.Float64bits(v))
	}
}

// Suppress unused import warning for unsafe
var _ = unsafe.Sizeof(0)
