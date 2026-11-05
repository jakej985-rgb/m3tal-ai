package tensor

import "errors"

var (
	// ErrShapeMismatch indicates tensor shapes are incompatible for an operation.
	ErrShapeMismatch = errors.New("tensor shape mismatch")
	// ErrInvalidShape indicates invalid dimension size (such as negative length).
	ErrInvalidShape = errors.New("invalid tensor shape")
	// ErrIndexOutOfBounds indicates an index falls outside the valid bounds.
	ErrIndexOutOfBounds = errors.New("tensor index out of bounds")
	// ErrStorageOverflow indicates requested dimensions or byte capacity overflow integer limits.
	ErrStorageOverflow = errors.New("tensor storage capacity overflow")
	// ErrUnsupportedDType indicates an invalid or unsupported data type was provided.
	ErrUnsupportedDType = errors.New("unsupported tensor data type")
	// ErrUnsupportedDevice indicates the requested hardware device backend is not supported.
	ErrUnsupportedDevice = errors.New("unsupported device")
	// ErrNonContiguous indicates an operation requires contiguous layout but a strided view was provided.
	ErrNonContiguous = errors.New("tensor is not contiguous")
	// ErrStorageExhausted indicates storage capacity is insufficient for the requested tensor slice or view.
	ErrStorageExhausted = errors.New("storage capacity exceeded")
)
