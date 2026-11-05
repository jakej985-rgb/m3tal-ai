package tensor

import "fmt"

// DType represents the numeric element data type stored in a Tensor.
type DType uint8

const (
	// Invalid indicates an uninitialized or unsupported data type.
	Invalid DType = iota
	// Float32 represents single-precision IEEE 754 floating point (32-bit).
	Float32
	// Float64 represents double-precision IEEE 754 floating point (64-bit).
	Float64
	// Int32 represents signed 32-bit integers.
	Int32
	// Int64 represents signed 64-bit integers.
	Int64
	// Uint8 represents unsigned 8-bit integers (bytes).
	Uint8
)

// Size returns the size in bytes of a single element of this DType.
// It returns 0 for Invalid or unsupported dtypes.
func (d DType) Size() int {
	switch d {
	case Float32, Int32:
		return 4
	case Float64, Int64:
		return 8
	case Uint8:
		return 1
	default:
		return 0
	}
}

// String returns a readable string representation of the DType.
func (d DType) String() string {
	switch d {
	case Float32:
		return "float32"
	case Float64:
		return "float64"
	case Int32:
		return "int32"
	case Int64:
		return "int64"
	case Uint8:
		return "uint8"
	default:
		return fmt.Sprintf("invalid_dtype(%d)", uint8(d))
	}
}

// IsValid reports whether d is a supported, valid data type.
func (d DType) IsValid() bool {
	return d.Size() > 0
}
