package tensor

import (
	"errors"
	"reflect"
	"testing"
)

func TestBroadcastShapes(t *testing.T) {
	tests := []struct {
		s1, s2   []int
		expected []int
		hasErr   bool
	}{
		{[]int{2, 3}, []int{2, 3}, []int{2, 3}, false},
		{[]int{2, 1}, []int{3}, []int{2, 3}, false},
		{[]int{1, 4}, []int{3, 1}, []int{3, 4}, false},
		{[]int{2, 3, 4}, []int{4}, []int{2, 3, 4}, false},
		{[]int{2, 3, 4}, []int{1, 4}, []int{2, 3, 4}, false},
		{[]int{2, 3, 4}, []int{3, 1}, []int{2, 3, 4}, false},
		{[]int{}, []int{2, 3}, []int{2, 3}, false},
		{[]int{2, 3}, []int{2, 4}, nil, true},
		{[]int{-1, 3}, []int{3}, nil, true},
	}

	for _, tt := range tests {
		out, err := BroadcastShapes(tt.s1, tt.s2)
		if tt.hasErr {
			if err == nil {
				t.Fatalf("BroadcastShapes(%v, %v) expected error, got nil", tt.s1, tt.s2)
			}
		} else {
			if err != nil {
				t.Fatalf("BroadcastShapes(%v, %v) unexpected error: %v", tt.s1, tt.s2, err)
			}
			if !reflect.DeepEqual(out, tt.expected) {
				t.Fatalf("BroadcastShapes(%v, %v) = %v, want %v", tt.s1, tt.s2, out, tt.expected)
			}
		}
	}
}

func TestBroadcastTo(t *testing.T) {
	orig, err := FromFloat32([]int{1, 3}, []float32{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}

	bcast, err := orig.BroadcastTo([]int{2, 3})
	if err != nil {
		t.Fatalf("BroadcastTo failed: %v", err)
	}
	if !reflect.DeepEqual(bcast.Shape(), []int{2, 3}) {
		t.Fatalf("bcast shape = %v, want [2, 3]", bcast.Shape())
	}

	// Verify stride on expanded dimension is 0
	if bcast.Strides()[0] != 0 || bcast.Strides()[1] != 1 {
		t.Fatalf("bcast strides = %v, want [0, 1]", bcast.Strides())
	}

	// Access elements across rows
	v0, _ := bcast.Float32At(0, 1)
	v1, _ := bcast.Float32At(1, 1)
	if v0 != 2 || v1 != 2 {
		t.Fatalf("bcast values mismatch: %f, %f", v0, v1)
	}

	// Incompatible broadcast
	if _, err := orig.BroadcastTo([]int{2, 4}); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("expected ErrShapeMismatch, got %v", err)
	}
}

func TestShapeTransforms(t *testing.T) {
	// Transpose 2D
	t2d, err := FromFloat32([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	tTrans, err := t2d.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(tTrans.Shape(), []int{3, 2}) {
		t.Fatalf("transpose shape = %v, want [3, 2]", tTrans.Shape())
	}
	v, _ := tTrans.Float32At(2, 0)
	if v != 3 {
		t.Fatalf("tTrans(2, 0) = %f, want 3", v)
	}

	// Transpose N-D with permutation
	t3d, err := FromFloat32([]int{2, 3, 4}, make([]float32, 24))
	if err != nil {
		t.Fatal(err)
	}
	permuted, err := t3d.Transpose(2, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(permuted.Shape(), []int{4, 2, 3}) {
		t.Fatalf("permuted shape = %v, want [4, 2, 3]", permuted.Shape())
	}

	// Squeeze
	tSq, err := FromFloat32([]int{1, 3, 1, 4}, make([]float32, 12))
	if err != nil {
		t.Fatal(err)
	}
	sqAll, err := tSq.Squeeze()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(sqAll.Shape(), []int{3, 4}) {
		t.Fatalf("sqAll shape = %v, want [3, 4]", sqAll.Shape())
	}
	sqOne, err := tSq.Squeeze(0)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(sqOne.Shape(), []int{3, 1, 4}) {
		t.Fatalf("sqOne shape = %v, want [3, 1, 4]", sqOne.Shape())
	}

	// Unsqueeze
	unsq, err := sqAll.Unsqueeze(1)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(unsq.Shape(), []int{3, 1, 4}) {
		t.Fatalf("unsq shape = %v, want [3, 1, 4]", unsq.Shape())
	}
	unsqEnd, err := sqAll.Unsqueeze(2)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(unsqEnd.Shape(), []int{3, 4, 1}) {
		t.Fatalf("unsqEnd shape = %v, want [3, 4, 1]", unsqEnd.Shape())
	}

	// Flatten
	flat, err := permuted.Flatten(1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(flat.Shape(), []int{4, 6}) {
		t.Fatalf("flatten shape = %v, want [4, 6]", flat.Shape())
	}
}

func TestElementwiseArithmetic(t *testing.T) {
	a, err := FromFloat32([]int{2, 1}, []float32{10, 20})
	if err != nil {
		t.Fatal(err)
	}
	b, err := FromFloat32([]int{1, 3}, []float32{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}

	// Add with broadcast -> shape [2, 3]
	cAdd, err := Add(a, b)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if !reflect.DeepEqual(cAdd.Shape(), []int{2, 3}) {
		t.Fatalf("cAdd shape = %v, want [2, 3]", cAdd.Shape())
	}
	v00, _ := cAdd.Float32At(0, 0)
	v02, _ := cAdd.Float32At(0, 2)
	v10, _ := cAdd.Float32At(1, 0)
	v12, _ := cAdd.Float32At(1, 2)
	if v00 != 11 || v02 != 13 || v10 != 21 || v12 != 23 {
		t.Fatalf("unexpected Add results: %f, %f, %f, %f", v00, v02, v10, v12)
	}

	// Sub
	cSub, err := Sub(a, b)
	if err != nil {
		t.Fatal(err)
	}
	s00, _ := cSub.Float32At(0, 0)
	if s00 != 9 {
		t.Fatalf("Sub(0, 0) = %f, want 9", s00)
	}

	// Mul
	cMul, err := Mul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	m12, _ := cMul.Float32At(1, 2)
	if m12 != 60 {
		t.Fatalf("Mul(1, 2) = %f, want 60", m12)
	}

	// Div
	cDiv, err := Div(a, b)
	if err != nil {
		t.Fatal(err)
	}
	d01, _ := cDiv.Float32At(0, 1)
	if d01 != 5 {
		t.Fatalf("Div(0, 1) = %f, want 5", d01)
	}

	// Neg
	negA, err := a.Neg()
	if err != nil {
		t.Fatal(err)
	}
	n0, _ := negA.Float32At(0, 0)
	if n0 != -10 {
		t.Fatalf("Neg = %f, want -10", n0)
	}

	// Scale
	scaledB, err := b.Scale(10.0)
	if err != nil {
		t.Fatal(err)
	}
	sc2, _ := scaledB.Float32At(0, 2)
	if sc2 != 30 {
		t.Fatalf("Scale = %f, want 30", sc2)
	}

	// Float64 arithmetic
	a64, _ := FromFloat64([]int{2}, []float64{1.5, 2.5})
	b64, _ := FromFloat64([]int{2}, []float64{0.5, 1.5})
	res64, err := Add(a64, b64)
	if err != nil {
		t.Fatal(err)
	}
	v0, _ := res64.Float64At(0)
	v1, _ := res64.Float64At(1)
	if v0 != 2.0 || v1 != 4.0 {
		t.Fatalf("Float64 Add failed: %f, %f", v0, v1)
	}
}

func TestReductions(t *testing.T) {
	// Shape [2, 3]
	data := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	tensor, err := FromFloat32([]int{2, 3}, data)
	if err != nil {
		t.Fatal(err)
	}

	// Sum over all elements (axes = nil) -> scalar
	sumAll, err := Sum(tensor, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if sumAll.Rank() != 0 {
		t.Fatalf("sumAll rank = %d, want 0", sumAll.Rank())
	}
	vAll, _ := sumAll.Float32At()
	if vAll != 21 {
		t.Fatalf("sumAll = %f, want 21", vAll)
	}

	// Sum along axis 0 (columns) -> shape [3]
	sumAxis0, err := Sum(tensor, []int{0}, false)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(sumAxis0.Shape(), []int{3}) {
		t.Fatalf("sumAxis0 shape = %v, want [3]", sumAxis0.Shape())
	}
	c0, _ := sumAxis0.Float32At(0)
	c1, _ := sumAxis0.Float32At(1)
	c2, _ := sumAxis0.Float32At(2)
	if c0 != 5 || c1 != 7 || c2 != 9 {
		t.Fatalf("sumAxis0 values: %f, %f, %f, want 5, 7, 9", c0, c1, c2)
	}

	// Mean along axis 1 with keepDims -> shape [2, 1]
	meanAxis1, err := Mean(tensor, []int{1}, true)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(meanAxis1.Shape(), []int{2, 1}) {
		t.Fatalf("meanAxis1 shape = %v, want [2, 1]", meanAxis1.Shape())
	}
	r0, _ := meanAxis1.Float32At(0, 0)
	r1, _ := meanAxis1.Float32At(1, 0)
	if r0 != 2 || r1 != 5 {
		t.Fatalf("meanAxis1 values: %f, %f, want 2, 5", r0, r1)
	}

	// Max and Min
	maxVal, err := Max(tensor, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	mx, _ := maxVal.Float32At()
	if mx != 6 {
		t.Fatalf("maxVal = %f, want 6", mx)
	}

	minVal, err := Min(tensor, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	mn, _ := minVal.Float32At()
	if mn != 1 {
		t.Fatalf("minVal = %f, want 1", mn)
	}

	// Float64 reduction
	t64, _ := FromFloat64([]int{4}, []float64{10, 20, 30, 40})
	mean64, err := Mean(t64, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	m64, _ := mean64.Float64At()
	if m64 != 25.0 {
		t.Fatalf("mean64 = %f, want 25.0", m64)
	}

	// Empty tensor Max error
	emptyT, _ := New([]int{0}, Float32)
	if _, err := Max(emptyT, nil, false); !errors.Is(err, ErrIndexOutOfBounds) {
		t.Fatalf("expected ErrIndexOutOfBounds on empty tensor Max, got %v", err)
	}
}
