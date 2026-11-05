package tensor

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestDTypeProperties(t *testing.T) {
	tests := []struct {
		dtype    DType
		wantSize int
		wantStr  string
		wantVal  bool
	}{
		{Float32, 4, "float32", true},
		{Float64, 8, "float64", true},
		{Int32, 4, "int32", true},
		{Int64, 8, "int64", true},
		{Uint8, 1, "uint8", true},
		{Invalid, 0, "invalid_dtype(0)", false},
		{DType(99), 0, "invalid_dtype(99)", false},
	}
	for _, tc := range tests {
		if tc.dtype.Size() != tc.wantSize {
			t.Errorf("dtype %v size = %d, want %d", tc.dtype, tc.dtype.Size(), tc.wantSize)
		}
		if tc.dtype.String() != tc.wantStr {
			t.Errorf("dtype %v string = %q, want %q", tc.dtype, tc.dtype.String(), tc.wantStr)
		}
		if tc.dtype.IsValid() != tc.wantVal {
			t.Errorf("dtype %v is valid = %v, want %v", tc.dtype, tc.dtype.IsValid(), tc.wantVal)
		}
	}
}

func TestDeviceProperties(t *testing.T) {
	cpu := CPUDevice
	if !cpu.IsCPU() {
		t.Errorf("CPUDevice.IsCPU() = false, want true")
	}
	if cpu.String() != "cpu" {
		t.Errorf("CPUDevice.String() = %q, want 'cpu'", cpu.String())
	}

	cuda0 := CUDADevice(0)
	if cuda0.IsCPU() {
		t.Errorf("cuda0.IsCPU() = true, want false")
	}
	if cuda0.String() != "cuda:0" {
		t.Errorf("cuda0.String() = %q, want 'cuda:0'", cuda0.String())
	}
	if DeviceType(10).String() != "device(10)" {
		t.Errorf("unknown DeviceType string = %q, want 'device(10)'", DeviceType(10).String())
	}
}

func TestShapeNumelAndOverflow(t *testing.T) {
	// Scalar tensor
	numel, err := Numel([]int{})
	if err != nil || numel != 1 {
		t.Fatalf("scalar Numel = %d, err = %v; want 1, nil", numel, err)
	}

	// Normal shape
	numel, err = Numel([]int{2, 3, 4})
	if err != nil || numel != 24 {
		t.Fatalf("shape [2, 3, 4] Numel = %d, err = %v; want 24, nil", numel, err)
	}

	// Zero-dim shape
	numel, err = Numel([]int{2, 0, 4})
	if err != nil || numel != 0 {
		t.Fatalf("shape with 0 dim Numel = %d, err = %v; want 0, nil", numel, err)
	}

	// Negative dimension
	_, err = Numel([]int{2, -1, 4})
	if !errors.Is(err, ErrInvalidShape) {
		t.Fatalf("expected ErrInvalidShape for negative dim, got %v", err)
	}

	// Integer overflow in shape
	_, err = Numel([]int{math.MaxInt / 2, 4})
	if !errors.Is(err, ErrStorageOverflow) {
		t.Fatalf("expected ErrStorageOverflow for overflowing dimensions, got %v", err)
	}

	// ByteSize calculation and overflow
	bytesNeeded, err := ByteSize([]int{2, 3}, Float32)
	if err != nil || bytesNeeded != 24 {
		t.Fatalf("ByteSize = %d, err = %v; want 24, nil", bytesNeeded, err)
	}

	_, err = ByteSize([]int{math.MaxInt / 4, 2}, Float64)
	if !errors.Is(err, ErrStorageOverflow) {
		t.Fatalf("expected ErrStorageOverflow for overflowing byte size, got %v", err)
	}

	// Invalid dtype for ByteSize
	_, err = ByteSize([]int{2, 3}, Invalid)
	if !errors.Is(err, ErrUnsupportedDType) {
		t.Fatalf("expected ErrUnsupportedDType for invalid dtype, got %v", err)
	}
}

func TestStridesAndContiguity(t *testing.T) {
	// Scalar
	scalarStrides := ContiguousStrides([]int{})
	if len(scalarStrides) != 0 {
		t.Fatalf("scalar strides = %v, want []", scalarStrides)
	}
	if !IsContiguous([]int{}, scalarStrides) {
		t.Fatalf("scalar should be contiguous")
	}

	// 3D shape [2, 3, 4] -> strides [12, 4, 1]
	shape := []int{2, 3, 4}
	strides := ContiguousStrides(shape)
	expectedStrides := []int{12, 4, 1}
	if !reflect.DeepEqual(strides, expectedStrides) {
		t.Fatalf("strides = %v, want %v", strides, expectedStrides)
	}
	if !IsContiguous(shape, strides) {
		t.Fatalf("standard strides should be contiguous")
	}

	nonContigStrides := []int{1, 2, 6}
	if IsContiguous(shape, nonContigStrides) {
		t.Fatalf("transposed strides should not be contiguous")
	}
	if IsContiguous(shape, []int{12, 4}) {
		t.Fatalf("mismatched strides rank should not be contiguous")
	}
}

func TestComputeOffsetBoundsChecking(t *testing.T) {
	shape := []int{2, 3}
	strides := []int{3, 1}

	// Valid offsets
	off, err := ComputeOffset([]int{1, 2}, shape, strides, 0)
	if err != nil || off != 5 {
		t.Fatalf("ComputeOffset(1, 2) = %d, %v; want 5, nil", off, err)
	}

	// With base offset
	off, err = ComputeOffset([]int{1, 2}, shape, strides, 10)
	if err != nil || off != 15 {
		t.Fatalf("ComputeOffset(1, 2) base 10 = %d, %v; want 15, nil", off, err)
	}

	// Mismatched index count
	_, err = ComputeOffset([]int{1}, shape, strides, 0)
	if !errors.Is(err, ErrIndexOutOfBounds) {
		t.Fatalf("expected ErrIndexOutOfBounds for mismatched rank, got %v", err)
	}

	// Negative index
	_, err = ComputeOffset([]int{-1, 0}, shape, strides, 0)
	if !errors.Is(err, ErrIndexOutOfBounds) {
		t.Fatalf("expected ErrIndexOutOfBounds for negative index, got %v", err)
	}

	// Out of bounds index
	_, err = ComputeOffset([]int{2, 0}, shape, strides, 0)
	if !errors.Is(err, ErrIndexOutOfBounds) {
		t.Fatalf("expected ErrIndexOutOfBounds for index >= shape, got %v", err)
	}
}

func TestTensorAllocationAndAccessors(t *testing.T) {
	// Allocate valid float32 tensor
	tensor, err := New([]int{2, 3}, Float32)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	if tensor.Rank() != 2 || tensor.Numel() != 6 || tensor.ByteSize() != 24 {
		t.Fatalf("unexpected tensor properties: rank %d, numel %d, byteSize %d", tensor.Rank(), tensor.Numel(), tensor.ByteSize())
	}
	if !tensor.IsContiguous() {
		t.Fatalf("new tensor should be contiguous")
	}
	if !tensor.Device().IsCPU() {
		t.Fatalf("new tensor device should be CPU")
	}
	if !tensor.Storage().OwnsMemory() {
		t.Fatalf("new tensor storage should own memory")
	}

	// Invalid dtype
	if _, err := New([]int{2, 3}, Invalid); !errors.Is(err, ErrUnsupportedDType) {
		t.Fatalf("expected ErrUnsupportedDType, got %v", err)
	}

	// Invalid shape
	if _, err := New([]int{2, -3}, Float32); !errors.Is(err, ErrInvalidShape) {
		t.Fatalf("expected ErrInvalidShape, got %v", err)
	}

	// Write and read float32
	if err := tensor.SetFloat32At(3.14, 0, 1); err != nil {
		t.Fatalf("SetFloat32At failed: %v", err)
	}
	val, err := tensor.Float32At(0, 1)
	if err != nil || val != 3.14 {
		t.Fatalf("Float32At = %f, %v; want 3.14, nil", val, err)
	}

	// Type mismatch read
	if _, err := tensor.Float64At(0, 1); !errors.Is(err, ErrUnsupportedDType) {
		t.Fatalf("expected ErrUnsupportedDType on Float64At on float32 tensor, got %v", err)
	}
	if err := tensor.SetFloat64At(1.0, 0, 1); !errors.Is(err, ErrUnsupportedDType) {
		t.Fatalf("expected ErrUnsupportedDType on SetFloat64At on float32 tensor, got %v", err)
	}

	// Out of bounds read/write
	if _, err := tensor.Float32At(2, 0); !errors.Is(err, ErrIndexOutOfBounds) {
		t.Fatalf("expected ErrIndexOutOfBounds, got %v", err)
	}
	if err := tensor.SetFloat32At(1.0, 0, 5); !errors.Is(err, ErrIndexOutOfBounds) {
		t.Fatalf("expected ErrIndexOutOfBounds, got %v", err)
	}
}

func TestFromFloat32AndFromFloat64(t *testing.T) {
	// FromFloat32
	t32, err := FromFloat32([]int{2, 2}, []float32{1.5, 2.5, 3.5, 4.5})
	if err != nil {
		t.Fatalf("FromFloat32 failed: %v", err)
	}
	v32, err := t32.Float32At(1, 0)
	if err != nil || v32 != 3.5 {
		t.Fatalf("Float32At(1, 0) = %f, want 3.5", v32)
	}

	// Mismatched length
	if _, err := FromFloat32([]int{2, 2}, []float32{1.0}); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("expected ErrShapeMismatch, got %v", err)
	}

	// FromFloat64
	t64, err := FromFloat64([]int{2, 2}, []float64{10.0, 20.0, 30.0, 40.0})
	if err != nil {
		t.Fatalf("FromFloat64 failed: %v", err)
	}
	v64, err := t64.Float64At(0, 1)
	if err != nil || v64 != 20.0 {
		t.Fatalf("Float64At(0, 1) = %f, want 20.0", v64)
	}

	// ToMatrix and FromMatrix roundtrip
	mat, err := t64.ToMatrix()
	if err != nil {
		t.Fatalf("ToMatrix failed: %v", err)
	}
	if mat.Rows() != 2 || mat.Cols() != 2 || mat.At(1, 1) != 40.0 {
		t.Fatalf("unexpected matrix content: %v", mat)
	}
	t64Roundtrip, err := FromMatrix(mat)
	if err != nil {
		t.Fatalf("FromMatrix failed: %v", err)
	}
	vRoundtrip, err := t64Roundtrip.Float64At(1, 1)
	if err != nil || vRoundtrip != 40.0 {
		t.Fatalf("roundtrip Float64At(1, 1) = %f, want 40.0", vRoundtrip)
	}

	// ToMatrix type mismatch with float32
	if _, err := t32.ToMatrix(); !errors.Is(err, ErrUnsupportedDType) {
		t.Fatalf("expected ErrUnsupportedDType converting float32 to Matrix, got %v", err)
	}
}

func TestSlicingAndStorageSharing(t *testing.T) {
	// Shape [3, 4] with sequential float32 numbers
	data := make([]float32, 12)
	for i := range data {
		data[i] = float32(i)
	}
	tOrig, err := FromFloat32([]int{3, 4}, data)
	if err != nil {
		t.Fatal(err)
	}

	// Slice rows [1:3] -> shape [2, 4]
	sliceRow, err := tOrig.Slice(0, 1, 3)
	if err != nil {
		t.Fatalf("Slice failed: %v", err)
	}
	if !reflect.DeepEqual(sliceRow.Shape(), []int{2, 4}) {
		t.Fatalf("slice shape = %v, want [2, 4]", sliceRow.Shape())
	}
	if sliceRow.Offset() != 4 {
		t.Fatalf("slice offset = %d, want 4", sliceRow.Offset())
	}

	// Verify element reading in slice
	val, err := sliceRow.Float32At(0, 0)
	if err != nil || val != 4.0 {
		t.Fatalf("slice Float32At(0, 0) = %f, want 4.0", val)
	}

	// Mutating the slice directly affects the parent tensor! (Storage sharing)
	if err := sliceRow.SetFloat32At(999.0, 0, 0); err != nil {
		t.Fatal(err)
	}
	parentVal, err := tOrig.Float32At(1, 0)
	if err != nil || parentVal != 999.0 {
		t.Fatalf("parentVal = %f, want 999.0 (mutation should reflect in parent)", parentVal)
	}

	// Slice column [1:3] on sliceRow -> sub-slice
	subSlice, err := sliceRow.Slice(1, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(subSlice.Shape(), []int{2, 2}) {
		t.Fatalf("subSlice shape = %v, want [2, 2]", subSlice.Shape())
	}
	if subSlice.IsContiguous() {
		t.Fatalf("column sliced sub-matrix should NOT be contiguous")
	}

	// Invalid slice parameters
	if _, err := tOrig.Slice(-1, 0, 1); !errors.Is(err, ErrIndexOutOfBounds) {
		t.Fatalf("expected ErrIndexOutOfBounds for negative dim, got %v", err)
	}
	if _, err := tOrig.Slice(0, 2, 1); !errors.Is(err, ErrIndexOutOfBounds) {
		t.Fatalf("expected ErrIndexOutOfBounds for end < start, got %v", err)
	}
	if _, err := tOrig.Slice(0, 0, 5); !errors.Is(err, ErrIndexOutOfBounds) {
		t.Fatalf("expected ErrIndexOutOfBounds for end > dimSize, got %v", err)
	}
}

func TestReshapeAndClone(t *testing.T) {
	tOrig, err := FromFloat32([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}

	// Valid reshape on contiguous tensor
	tReshaped, err := tOrig.Reshape([]int{3, 2})
	if err != nil {
		t.Fatalf("Reshape failed: %v", err)
	}
	if !reflect.DeepEqual(tReshaped.Shape(), []int{3, 2}) {
		t.Fatalf("reshaped shape = %v, want [3, 2]", tReshaped.Shape())
	}

	// Storage is shared across reshape
	if err := tReshaped.SetFloat32At(100.0, 0, 0); err != nil {
		t.Fatal(err)
	}
	v, _ := tOrig.Float32At(0, 0)
	if v != 100.0 {
		t.Fatalf("mutating reshaped tensor should reflect in original, got %f", v)
	}

	// Reshape with element count mismatch
	if _, err := tOrig.Reshape([]int{4, 2}); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("expected ErrShapeMismatch, got %v", err)
	}

	// Reshape on non-contiguous tensor should fail with ErrNonContiguous
	colSlice, err := tOrig.Slice(1, 0, 2) // shape [2, 2], strides [3, 1]
	if err != nil {
		t.Fatal(err)
	}
	if colSlice.IsContiguous() {
		t.Fatal("colSlice should not be contiguous")
	}
	if _, err := colSlice.Reshape([]int{4}); !errors.Is(err, ErrNonContiguous) {
		t.Fatalf("expected ErrNonContiguous reshaping non-contiguous view, got %v", err)
	}

	// Clone non-contiguous view into contiguous tensor
	cloned, err := colSlice.Clone()
	if err != nil {
		t.Fatalf("Clone failed: %v", err)
	}
	if !cloned.IsContiguous() {
		t.Fatal("cloned tensor must be contiguous")
	}
	// Reshape on cloned tensor succeeds
	reshapedCloned, err := cloned.Reshape([]int{4})
	if err != nil {
		t.Fatalf("reshaping cloned tensor failed: %v", err)
	}
	if reshapedCloned.Numel() != 4 {
		t.Fatalf("reshaped numel = %d, want 4", reshapedCloned.Numel())
	}

	// Mutations on cloned tensor do not affect the original
	_ = cloned.SetFloat32At(555.0, 0, 0)
	origVal, _ := colSlice.Float32At(0, 0)
	if origVal == 555.0 {
		t.Fatalf("cloned tensor should not share storage with original")
	}
}

func TestNewWithStorageLayoutBounds(t *testing.T) {
	storage, err := NewCPUStorage(16) // 4 float32 elements
	if err != nil {
		t.Fatal(err)
	}

	// Layout that fits: shape [2, 2], strides [2, 1], offset 0 -> max index 3 (16 bytes)
	_, err = NewWithStorage([]int{2, 2}, []int{2, 1}, 0, Float32, storage)
	if err != nil {
		t.Fatalf("NewWithStorage should succeed, got %v", err)
	}

	// Layout that exceeds storage: shape [2, 2], strides [2, 1], offset 1 -> max index 4 (20 bytes)
	_, err = NewWithStorage([]int{2, 2}, []int{2, 1}, 1, Float32, storage)
	if !errors.Is(err, ErrStorageExhausted) {
		t.Fatalf("expected ErrStorageExhausted, got %v", err)
	}

	// Nil storage
	_, err = NewWithStorage([]int{2}, []int{1}, 0, Float32, nil)
	if !errors.Is(err, ErrInvalidShape) {
		t.Fatalf("expected ErrInvalidShape for nil storage, got %v", err)
	}

	// Negative offset
	_, err = NewWithStorage([]int{2}, []int{1}, -1, Float32, storage)
	if !errors.Is(err, ErrIndexOutOfBounds) {
		t.Fatalf("expected ErrIndexOutOfBounds for negative offset, got %v", err)
	}
}
