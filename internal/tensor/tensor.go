package tensor

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Tensor represents a multi-dimensional array with configurable shape,
// strides, element data type, and storage placement.
type Tensor struct {
	shape   []int
	strides []int
	offset  int // offset in element units from the start of Storage
	dtype   DType
	device  Device
	storage *Storage
}

// New creates a new contiguous Tensor of the given shape and DType on CPU.
func New(shape []int, dtype DType) (*Tensor, error) {
	if !dtype.IsValid() {
		return nil, fmt.Errorf("%w: %s", ErrUnsupportedDType, dtype)
	}
	bytesNeeded, err := ByteSize(shape, dtype)
	if err != nil {
		return nil, err
	}
	storage, err := NewCPUStorage(bytesNeeded)
	if err != nil {
		return nil, err
	}
	clonedShape := append([]int(nil), shape...)
	strides := ContiguousStrides(clonedShape)
	return &Tensor{
		shape:   clonedShape,
		strides: strides,
		offset:  0,
		dtype:   dtype,
		device:  CPUDevice,
		storage: storage,
	}, nil
}

// NewWithStorage creates a Tensor viewing an existing Storage with explicit strides and element offset.
func NewWithStorage(shape, strides []int, offset int, dtype DType, storage *Storage) (*Tensor, error) {
	if !dtype.IsValid() {
		return nil, fmt.Errorf("%w: %s", ErrUnsupportedDType, dtype)
	}
	if storage == nil {
		return nil, fmt.Errorf("%w: storage cannot be nil", ErrInvalidShape)
	}
	if len(shape) != len(strides) {
		return nil, fmt.Errorf("%w: shape rank %d does not match strides rank %d", ErrShapeMismatch, len(shape), len(strides))
	}
	if offset < 0 {
		return nil, fmt.Errorf("%w: offset %d cannot be negative", ErrIndexOutOfBounds, offset)
	}

	numel, err := Numel(shape)
	if err != nil {
		return nil, err
	}

	// Verify maximum index reachable does not exceed storage boundary
	if numel > 0 {
		maxOffset := offset
		for i, dim := range shape {
			if dim > 1 && strides[i] > 0 {
				maxOffset += (dim - 1) * strides[i]
			}
		}
		maxByte := (maxOffset + 1) * dtype.Size()
		if maxByte > storage.ByteSize() {
			return nil, fmt.Errorf("%w: requested layout requires %d bytes, storage only has %d", ErrStorageExhausted, maxByte, storage.ByteSize())
		}
	}

	return &Tensor{
		shape:   append([]int(nil), shape...),
		strides: append([]int(nil), strides...),
		offset:  offset,
		dtype:   dtype,
		device:  storage.Device(),
		storage: storage,
	}, nil
}

// FromFloat32 creates a contiguous CPU Tensor with Float32 dtype from a slice of float32 values.
func FromFloat32(shape []int, data []float32) (*Tensor, error) {
	n, err := Numel(shape)
	if err != nil {
		return nil, err
	}
	if len(data) != n {
		return nil, fmt.Errorf("%w: data has %d elements, shape requires %d", ErrShapeMismatch, len(data), n)
	}
	t, err := New(shape, Float32)
	if err != nil {
		return nil, err
	}
	raw := t.storage.Bytes()
	for i, v := range data {
		binary.LittleEndian.PutUint32(raw[i*4:(i+1)*4], math.Float32bits(v))
	}
	return t, nil
}

// FromFloat64 creates a contiguous CPU Tensor with Float64 dtype from a slice of float64 values.
func FromFloat64(shape []int, data []float64) (*Tensor, error) {
	n, err := Numel(shape)
	if err != nil {
		return nil, err
	}
	if len(data) != n {
		return nil, fmt.Errorf("%w: data has %d elements, shape requires %d", ErrShapeMismatch, len(data), n)
	}
	t, err := New(shape, Float64)
	if err != nil {
		return nil, err
	}
	raw := t.storage.Bytes()
	for i, v := range data {
		binary.LittleEndian.PutUint64(raw[i*8:(i+1)*8], math.Float64bits(v))
	}
	return t, nil
}

// Shape returns a copy of the tensor dimensions.
func (t *Tensor) Shape() []int {
	return append([]int(nil), t.shape...)
}

// Strides returns a copy of the step size along each dimension.
func (t *Tensor) Strides() []int {
	return append([]int(nil), t.strides...)
}

// Rank returns the number of dimensions (axes).
func (t *Tensor) Rank() int {
	return len(t.shape)
}

// Offset returns the linear element offset into the underlying Storage.
func (t *Tensor) Offset() int {
	return t.offset
}

// DType returns the element data type.
func (t *Tensor) DType() DType {
	return t.dtype
}

// Device returns the device hosting tensor memory.
func (t *Tensor) Device() Device {
	return t.device
}

// Storage returns the underlying memory buffer.
func (t *Tensor) Storage() *Storage {
	return t.storage
}

// Numel returns the total number of elements.
func (t *Tensor) Numel() int {
	n, _ := Numel(t.shape)
	return n
}

// IsContiguous reports whether the tensor is stored in contiguous C-order row-major layout.
func (t *Tensor) IsContiguous() bool {
	return IsContiguous(t.shape, t.strides)
}

// ByteSize returns the byte footprint of the elements logically in this tensor.
func (t *Tensor) ByteSize() int {
	return t.Numel() * t.dtype.Size()
}

// Reshape returns a new Tensor sharing the same storage with a modified shape,
// provided the tensor is contiguous and element counts match.
func (t *Tensor) Reshape(newShape []int) (*Tensor, error) {
	newNumel, err := Numel(newShape)
	if err != nil {
		return nil, err
	}
	if newNumel != t.Numel() {
		return nil, fmt.Errorf("%w: cannot reshape tensor with %d elements into shape %v (%d elements)", ErrShapeMismatch, t.Numel(), newShape, newNumel)
	}
	if !t.IsContiguous() {
		return nil, fmt.Errorf("%w: reshape requires contiguous tensor (copy first if needed)", ErrNonContiguous)
	}
	return NewWithStorage(newShape, ContiguousStrides(newShape), t.offset, t.dtype, t.storage)
}

// Slice creates a view sharing the same storage by slicing dimension `dim` between `start` and `end`.
// Half-open interval [start, end) is used.
func (t *Tensor) Slice(dim, start, end int) (*Tensor, error) {
	if dim < 0 || dim >= len(t.shape) {
		return nil, fmt.Errorf("%w: dimension %d out of range for rank %d", ErrIndexOutOfBounds, dim, len(t.shape))
	}
	dimSize := t.shape[dim]
	if start < 0 || end < start || end > dimSize {
		return nil, fmt.Errorf("%w: invalid slice range [%d:%d] for dimension %d with size %d", ErrIndexOutOfBounds, start, end, dim, dimSize)
	}

	newShape := append([]int(nil), t.shape...)
	newShape[dim] = end - start

	newStrides := append([]int(nil), t.strides...)
	newOffset := t.offset + start*t.strides[dim]

	return &Tensor{
		shape:   newShape,
		strides: newStrides,
		offset:  newOffset,
		dtype:   t.dtype,
		device:  t.device,
		storage: t.storage,
	}, nil
}

// Clone creates an independent, contiguous copy of the tensor on the same device.
func (t *Tensor) Clone() (*Tensor, error) {
	if !t.device.IsCPU() {
		return nil, fmt.Errorf("%w: clone only supported on CPU", ErrUnsupportedDevice)
	}
	out, err := New(t.shape, t.dtype)
	if err != nil {
		return nil, err
	}
	numel := t.Numel()
	if numel == 0 {
		return out, nil
	}

	elemSize := t.dtype.Size()
	srcBytes := t.storage.Bytes()
	dstBytes := out.storage.Bytes()

	if t.IsContiguous() {
		start := t.offset * elemSize
		end := start + numel*elemSize
		copy(dstBytes, srcBytes[start:end])
		return out, nil
	}

	// Copy element by element for strided layout
	indices := make([]int, len(t.shape))
	dstOffset := 0
	for i := 0; i < numel; i++ {
		srcOff, err := ComputeOffset(indices, t.shape, t.strides, t.offset)
		if err != nil {
			return nil, err
		}
		copy(dstBytes[dstOffset*elemSize:(dstOffset+1)*elemSize], srcBytes[srcOff*elemSize:(srcOff+1)*elemSize])
		dstOffset++

		// Advance indices in row-major order
		for d := len(t.shape) - 1; d >= 0; d-- {
			indices[d]++
			if indices[d] < t.shape[d] {
				break
			}
			indices[d] = 0
		}
	}
	return out, nil
}

// Float32At reads a float32 value at the specified coordinate.
func (t *Tensor) Float32At(indices ...int) (float32, error) {
	if t.dtype != Float32 {
		return 0, fmt.Errorf("%w: tensor dtype is %s, expected float32", ErrUnsupportedDType, t.dtype)
	}
	elemOffset, err := ComputeOffset(indices, t.shape, t.strides, t.offset)
	if err != nil {
		return 0, err
	}
	byteOffset := elemOffset * 4
	raw := t.storage.Bytes()
	bits := binary.LittleEndian.Uint32(raw[byteOffset : byteOffset+4])
	return math.Float32frombits(bits), nil
}

// SetFloat32At writes a float32 value at the specified coordinate.
func (t *Tensor) SetFloat32At(val float32, indices ...int) error {
	if t.dtype != Float32 {
		return fmt.Errorf("%w: tensor dtype is %s, expected float32", ErrUnsupportedDType, t.dtype)
	}
	elemOffset, err := ComputeOffset(indices, t.shape, t.strides, t.offset)
	if err != nil {
		return err
	}
	byteOffset := elemOffset * 4
	raw := t.storage.Bytes()
	binary.LittleEndian.PutUint32(raw[byteOffset:byteOffset+4], math.Float32bits(val))
	return nil
}

// Float64At reads a float64 value at the specified coordinate.
func (t *Tensor) Float64At(indices ...int) (float64, error) {
	if t.dtype != Float64 {
		return 0, fmt.Errorf("%w: tensor dtype is %s, expected float64", ErrUnsupportedDType, t.dtype)
	}
	elemOffset, err := ComputeOffset(indices, t.shape, t.strides, t.offset)
	if err != nil {
		return 0, err
	}
	byteOffset := elemOffset * 8
	raw := t.storage.Bytes()
	bits := binary.LittleEndian.Uint64(raw[byteOffset : byteOffset+8])
	return math.Float64frombits(bits), nil
}

// SetFloat64At writes a float64 value at the specified coordinate.
func (t *Tensor) SetFloat64At(val float64, indices ...int) error {
	if t.dtype != Float64 {
		return fmt.Errorf("%w: tensor dtype is %s, expected float64", ErrUnsupportedDType, t.dtype)
	}
	elemOffset, err := ComputeOffset(indices, t.shape, t.strides, t.offset)
	if err != nil {
		return err
	}
	byteOffset := elemOffset * 8
	raw := t.storage.Bytes()
	binary.LittleEndian.PutUint64(raw[byteOffset:byteOffset+8], math.Float64bits(val))
	return nil
}

// ToMatrix converts a 2D Float64 Tensor into the legacy Matrix type.
func (t *Tensor) ToMatrix() (Matrix, error) {
	if len(t.shape) != 2 {
		return Matrix{}, fmt.Errorf("%w: matrix requires 2D shape, got rank %d", ErrShapeMismatch, len(t.shape))
	}
	if t.dtype != Float64 {
		return Matrix{}, fmt.Errorf("%w: matrix requires float64, got %s", ErrUnsupportedDType, t.dtype)
	}
	rows, cols := t.shape[0], t.shape[1]
	values := make([]float64, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			v, err := t.Float64At(r, c)
			if err != nil {
				return Matrix{}, err
			}
			values[r*cols+c] = v
		}
	}
	return NewMatrix(rows, cols, values)
}

// FromMatrix converts a legacy Matrix into a 2D Float64 Tensor.
func FromMatrix(m Matrix) (*Tensor, error) {
	return FromFloat64([]int{m.Rows(), m.Cols()}, m.data)
}
