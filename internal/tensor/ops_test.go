package tensor

import (
	"math"
	"testing"
)

func TestTensorMatMul(t *testing.T) {
	// A: 2x3, B: 3x2 -> C: 2x2
	aData := []float32{1, 2, 3, 4, 5, 6}
	bData := []float32{7, 8, 9, 10, 11, 12}
	a, err := FromFloat32([]int{2, 3}, aData)
	if err != nil {
		t.Fatal(err)
	}
	b, err := FromFloat32([]int{3, 2}, bData)
	if err != nil {
		t.Fatal(err)
	}

	c, err := a.MatMul(b)
	if err != nil {
		t.Fatal(err)
	}
	if c.Rank() != 2 || c.Shape()[0] != 2 || c.Shape()[1] != 2 {
		t.Fatalf("unexpected shape: %v", c.Shape())
	}

	val00, _ := c.Float32At(0, 0)
	val01, _ := c.Float32At(0, 1)
	val10, _ := c.Float32At(1, 0)
	val11, _ := c.Float32At(1, 1)

	// 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
	// 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
	// 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
	// 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
	if val00 != 58 || val01 != 64 || val10 != 139 || val11 != 154 {
		t.Fatalf("unexpected matmul results: [[%f, %f], [%f, %f]]", val00, val01, val10, val11)
	}
}

func TestTensorElementwiseOpsAndReductions(t *testing.T) {
	a, _ := FromFloat32([]int{2, 2}, []float32{1, 2, 3, 4})
	b, _ := FromFloat32([]int{2, 2}, []float32{5, 6, 7, 8})

	sum, err := a.Add(b)
	if err != nil {
		t.Fatal(err)
	}
	v, _ := sum.Float32At(0, 0)
	if v != 6 {
		t.Fatalf("expected 6, got %f", v)
	}

	sub, err := b.Sub(a)
	if err != nil {
		t.Fatal(err)
	}
	v, _ = sub.Float32At(1, 1)
	if v != 4 {
		t.Fatalf("expected 4, got %f", v)
	}

	mul, err := a.Mul(b)
	if err != nil {
		t.Fatal(err)
	}
	v, _ = mul.Float32At(1, 1)
	if v != 32 {
		t.Fatalf("expected 32, got %f", v)
	}

	div, err := a.Div(b)
	if err != nil {
		t.Fatal(err)
	}
	v, _ = div.Float32At(0, 0)
	if math.Abs(float64(v-0.2)) > 1e-6 {
		t.Fatalf("expected 0.2, got %f", v)
	}

	scaled, err := a.Scale(2.5)
	if err != nil {
		t.Fatal(err)
	}
	v, _ = scaled.Float32At(1, 1)
	if v != 10 {
		t.Fatalf("expected 10, got %f", v)
	}

	totalSum, err := a.Sum()
	if err != nil {
		t.Fatal(err)
	}
	if totalSum != 10 {
		t.Fatalf("expected sum 10, got %f", totalSum)
	}
}

func TestTensorOpsEdgeCases(t *testing.T) {
	// 0-size tensor
	emptyA, _ := New([]int{0, 5}, Float32)
	emptyB, _ := New([]int{5, 3}, Float32)
	emptyC, err := emptyA.MatMul(emptyB)
	if err != nil {
		t.Fatal(err)
	}
	if emptyC.Shape()[0] != 0 || emptyC.Shape()[1] != 3 {
		t.Fatalf("unexpected shape for empty matmul: %v", emptyC.Shape())
	}

	// 1x1 matmul
	oneA, _ := FromFloat32([]int{1, 1}, []float32{3})
	oneB, _ := FromFloat32([]int{1, 1}, []float32{4})
	oneC, err := oneA.MatMul(oneB)
	if err != nil {
		t.Fatal(err)
	}
	val, _ := oneC.Float32At(0, 0)
	if val != 12 {
		t.Fatalf("expected 12, got %f", val)
	}

	// Shape mismatch errors
	badB, _ := FromFloat32([]int{4, 2}, make([]float32, 8))
	if _, err := oneA.MatMul(badB); err == nil {
		t.Fatal("expected inner dimension mismatch error")
	}

	// Strided view matmul
	big, _ := FromFloat32([]int{4, 4}, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	})
	sliceA, err := big.Slice(0, 0, 2) // [2, 4]
	if err != nil {
		t.Fatal(err)
	}
	sliceB, err := big.Slice(0, 0, 4) // [4, 4]
	if err != nil {
		t.Fatal(err)
	}
	sliceSubB, err := sliceB.Slice(1, 0, 2) // [4, 2]
	if err != nil {
		t.Fatal(err)
	}
	res, err := sliceA.MatMul(sliceSubB)
	if err != nil {
		t.Fatal(err)
	}
	if res.Shape()[0] != 2 || res.Shape()[1] != 2 {
		t.Fatalf("unexpected shape: %v", res.Shape())
	}
}
