package tensor

import (
	"reflect"
	"testing"
)

func mustMatrix(t *testing.T, rows, cols int, values []float64) Matrix {
	t.Helper()
	matrix, err := NewMatrix(rows, cols, values)
	if err != nil {
		t.Fatal(err)
	}
	return matrix
}

func TestVectorAddAndDot(t *testing.T) {
	sum, err := Vector{1, 2, 3}.Add(Vector{4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(sum, Vector{5, 7, 9}) {
		t.Fatalf("sum = %v, want [5 7 9]", sum)
	}
	dot, err := (Vector{1, 2, 3}).Dot(Vector{4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	if dot != 32 {
		t.Fatalf("dot = %v, want 32", dot)
	}
}

func TestMatrixMultiplyTransposeAndVectorMultiply(t *testing.T) {
	a := mustMatrix(t, 2, 3, []float64{1, 2, 3, 4, 5, 6})
	b := mustMatrix(t, 3, 2, []float64{7, 8, 9, 10, 11, 12})
	product, err := a.MatMul(b)
	if err != nil {
		t.Fatal(err)
	}
	if product.Rows() != 2 || product.Cols() != 2 || product.At(0, 0) != 58 || product.At(0, 1) != 64 || product.At(1, 0) != 139 || product.At(1, 1) != 154 {
		t.Fatalf("unexpected matrix product: %v", product)
	}
	transposed := a.Transpose()
	if transposed.Rows() != 3 || transposed.Cols() != 2 || transposed.At(2, 1) != 6 {
		t.Fatalf("unexpected transpose: %v", transposed)
	}
	result, err := a.MulVec(Vector{1, 0, -1})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result, Vector{-2, -2}) {
		t.Fatalf("matrix/vector product = %v, want [-2 -2]", result)
	}
}

func TestMatrixShapeErrors(t *testing.T) {
	a := mustMatrix(t, 2, 3, make([]float64, 6))
	b := mustMatrix(t, 2, 2, make([]float64, 4))
	if _, err := a.MatMul(b); err == nil {
		t.Fatal("expected matrix multiplication shape error")
	}
	if _, err := NewMatrix(2, 2, []float64{1}); err == nil {
		t.Fatal("expected constructor shape error")
	}
}
