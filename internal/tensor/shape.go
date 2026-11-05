package tensor

import (
	"fmt"
	"math"
)

// Shape represents the dimensions of a tensor.
type Shape []int

// Strides represents the step size in element counts along each dimension.
type Strides []int

// Numel returns the total number of elements implied by shape.
// It returns an error if any dimension is negative, or if total multiplication overflows math.MaxInt.
// A shape of length 0 (scalar) has 1 element.
func Numel(shape []int) (int, error) {
	if len(shape) == 0 {
		return 1, nil
	}
	total := 1
	for i, dim := range shape {
		if dim < 0 {
			return 0, fmt.Errorf("%w: dimension %d has negative size %d", ErrInvalidShape, i, dim)
		}
		if dim == 0 {
			// If any dimension is 0, total elements is 0. But continue checking remaining dims for negative values.
			total = 0
			continue
		}
		if total != 0 {
			if dim > math.MaxInt/total {
				return 0, fmt.Errorf("%w: shape %v overflows maximum integer size", ErrStorageOverflow, shape)
			}
			total *= dim
		}
	}
	return total, nil
}

// ByteSize calculates the total bytes needed for a shape with the given DType,
// guarding against integer overflow.
func ByteSize(shape []int, dtype DType) (int, error) {
	if !dtype.IsValid() {
		return 0, fmt.Errorf("%w: %s", ErrUnsupportedDType, dtype)
	}
	n, err := Numel(shape)
	if err != nil {
		return 0, err
	}
	size := dtype.Size()
	if n > math.MaxInt/size {
		return 0, fmt.Errorf("%w: byte size for shape %v with dtype %s overflows maximum integer size", ErrStorageOverflow, shape, dtype)
	}
	return n * size, nil
}

// ContiguousStrides calculates standard row-major (C-order) element strides for a given shape.
// For example, for shape [2, 3, 4], strides are [12, 4, 1].
// For a scalar (empty shape), strides are empty slice [].
func ContiguousStrides(shape []int) []int {
	if len(shape) == 0 {
		return []int{}
	}
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

// IsContiguous reports whether strides describe a standard contiguous row-major layout
// for the given shape.
func IsContiguous(shape []int, strides []int) bool {
	if len(shape) != len(strides) {
		return false
	}
	expected := ContiguousStrides(shape)
	for i := range strides {
		// If dimension length is 1 or 0, any stride is effectively contiguous,
		// but standard C-order layout uses expected stride.
		if strides[i] != expected[i] {
			return false
		}
	}
	return true
}

// ComputeOffset calculates the linear element offset for a multi-dimensional index
// relative to a base offset, validating each index against shape boundaries.
func ComputeOffset(indices, shape, strides []int, baseOffset int) (int, error) {
	if len(indices) != len(shape) {
		return 0, fmt.Errorf("%w: expected %d indices, got %d", ErrIndexOutOfBounds, len(shape), len(indices))
	}
	offset := baseOffset
	for i, idx := range indices {
		if idx < 0 || idx >= shape[i] {
			return 0, fmt.Errorf("%w: index %d out of bounds for axis %d with size %d", ErrIndexOutOfBounds, idx, i, shape[i])
		}
		offset += idx * strides[i]
	}
	return offset, nil
}
