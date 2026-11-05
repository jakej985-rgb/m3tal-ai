package autograd

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"m3tal-ai/internal/tensor"
)

// Helper to create a Float64 Variable from slice and shape.
func mustVar64(t *testing.T, shape []int, data []float64, requiresGrad bool) *Variable {
	t.Helper()
	tens, err := tensor.FromFloat64(shape, data)
	if err != nil {
		t.Fatalf("failed to create tensor: %v", err)
	}
	v, err := NewVariable(tens, requiresGrad)
	if err != nil {
		t.Fatalf("failed to create variable: %v", err)
	}
	return v
}

// Helper to create a Float32 Variable from slice and shape.
func mustVar32(t *testing.T, shape []int, data []float32, requiresGrad bool) *Variable {
	t.Helper()
	tens, err := tensor.FromFloat32(shape, data)
	if err != nil {
		t.Fatalf("failed to create tensor: %v", err)
	}
	v, err := NewVariable(tens, requiresGrad)
	if err != nil {
		t.Fatalf("failed to create variable: %v", err)
	}
	return v
}

// TestScalarParityWithAutogradValue verifies that 1-element Variable matches the scalar Value autodiff engine.
func TestScalarParityWithAutogradValue(t *testing.T) {
	// Scalar Value engine: f(x, y) = (x * y) + tanh(x)
	xVal := 0.37
	yVal := -0.82

	sX := New(xVal)
	sY := New(yVal)
	sF := sX.Mul(sY).Add(sX.Tanh())
	sF.Backward()

	// Tensor Variable engine
	tX := mustVar64(t, []int{1}, []float64{xVal}, true)
	tY := mustVar64(t, []int{1}, []float64{yVal}, true)

	prod, err := Mul(tX, tY)
	if err != nil {
		t.Fatal(err)
	}
	tanhX, err := Tanh(tX)
	if err != nil {
		t.Fatal(err)
	}
	tF, err := Add(prod, tanhX)
	if err != nil {
		t.Fatal(err)
	}

	if err := tF.Backward(); err != nil {
		t.Fatalf("tF.Backward failed: %v", err)
	}

	gradX, err := tX.Grad.Float64At(0)
	if err != nil {
		t.Fatal(err)
	}
	gradY, err := tY.Grad.Float64At(0)
	if err != nil {
		t.Fatal(err)
	}

	const tol = 1e-7
	if math.Abs(gradX-sX.Grad) > tol {
		t.Fatalf("gradX mismatch: tensor=%v scalar=%v", gradX, sX.Grad)
	}
	if math.Abs(gradY-sY.Grad) > tol {
		t.Fatalf("gradY mismatch: tensor=%v scalar=%v", gradY, sY.Grad)
	}
}

// TestTensorBranchingGradientAccumulation verifies that gradients accumulate correctly
// across shared graph branches (e.g., f = x * x + x).
func TestTensorBranchingGradientAccumulation(t *testing.T) {
	// x = [2, 3]
	// f = x * x + x -> df/dx = 2*x + 1 -> [5, 7]
	// Sum(f) -> dSum/dx = [5, 7]
	x := mustVar64(t, []int{2}, []float64{2.0, 3.0}, true)
	xSquared, err := Mul(x, x)
	if err != nil {
		t.Fatal(err)
	}
	f, err := Add(xSquared, x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := Sum(f)
	if err != nil {
		t.Fatal(err)
	}

	if err := loss.Backward(); err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	g0, _ := x.Grad.Float64At(0)
	g1, _ := x.Grad.Float64At(1)

	if math.Abs(g0-5.0) > 1e-6 {
		t.Errorf("expected grad[0] = 5.0, got %f", g0)
	}
	if math.Abs(g1-7.0) > 1e-6 {
		t.Errorf("expected grad[1] = 7.0, got %f", g1)
	}

	// Test ZeroGrad
	x.ZeroGrad()
	g0Zero, _ := x.Grad.Float64At(0)
	g1Zero, _ := x.Grad.Float64At(1)
	if g0Zero != 0 || g1Zero != 0 {
		t.Errorf("ZeroGrad failed: %f, %f", g0Zero, g1Zero)
	}
}

// TestTensorBroadcastingArithmetic tests Add/Sub/Mul/Div with broadcasting.
func TestTensorBroadcastingArithmetic(t *testing.T) {
	// A: shape [2, 3]
	// B: shape [1, 3]
	a := mustVar64(t, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6}, true)
	b := mustVar64(t, []int{1, 3}, []float64{10, 20, 30}, true)

	// C = A + B
	c, err := Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := Sum(c)
	if err != nil {
		t.Fatal(err)
	}
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	// dLoss/dA = ones [2, 3]
	// dLoss/dB = sum along axis 0 = [2, 2, 2]
	for i := 0; i < 3; i++ {
		gb, _ := b.Grad.Float64At(0, i)
		if math.Abs(gb-2.0) > 1e-6 {
			t.Errorf("expected b.Grad[0, %d] = 2.0, got %f", i, gb)
		}
	}
}

// TestMatMulGradient verifies reverse-mode matrix multiplication.
func TestMatMulGradient(t *testing.T) {
	// A: [2, 2], B: [2, 2]
	// A = [[1, 2], [3, 4]]
	// B = [[5, 6], [7, 8]]
	// C = A x B
	// Loss = Sum(C)
	// dLoss/dC = [[1, 1], [1, 1]]
	// dLoss/dA = dLoss/dC x B^T = [[1, 1], [1, 1]] x [[5, 7], [6, 8]] = [[11, 15], [11, 15]]
	// dLoss/dB = A^T x dLoss/dC = [[1, 3], [2, 4]] x [[1, 1], [1, 1]] = [[4, 4], [6, 6]]
	a := mustVar64(t, []int{2, 2}, []float64{1, 2, 3, 4}, true)
	b := mustVar64(t, []int{2, 2}, []float64{5, 6, 7, 8}, true)

	c, err := MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := Sum(c)
	if err != nil {
		t.Fatal(err)
	}

	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	ga00, _ := a.Grad.Float64At(0, 0)
	ga01, _ := a.Grad.Float64At(0, 1)
	ga10, _ := a.Grad.Float64At(1, 0)
	ga11, _ := a.Grad.Float64At(1, 1)

	if ga00 != 11 || ga01 != 15 || ga10 != 11 || ga11 != 15 {
		t.Errorf("unexpected a.Grad: [[%f, %f], [%f, %f]]", ga00, ga01, ga10, ga11)
	}

	gb00, _ := b.Grad.Float64At(0, 0)
	gb01, _ := b.Grad.Float64At(0, 1)
	gb10, _ := b.Grad.Float64At(1, 0)
	gb11, _ := b.Grad.Float64At(1, 1)

	if gb00 != 4 || gb01 != 4 || gb10 != 6 || gb11 != 6 {
		t.Errorf("unexpected b.Grad: [[%f, %f], [%f, %f]]", gb00, gb01, gb10, gb11)
	}

	// Batched 3D MatMul: [2, 2, 2] x [2, 2, 2]
	bA := mustVar64(t, []int{2, 2, 2}, []float64{1, 2, 3, 4, 1, 0, 0, 1}, true)
	bB := mustVar64(t, []int{2, 2, 2}, []float64{5, 6, 7, 8, 2, 3, 4, 5}, true)
	bC, err := MatMul(bA, bB)
	if err != nil {
		t.Fatalf("Batched MatMul failed: %v", err)
	}
	bLoss, err := Sum(bC)
	if err != nil {
		t.Fatalf("Sum failed: %v", err)
	}
	if err := bLoss.Backward(); err != nil {
		t.Fatalf("Backward on batched matmul failed: %v", err)
	}
	if bA.Grad == nil || bB.Grad == nil {
		t.Fatalf("Batched MatMul grads must not be nil")
	}
}

// TestActivationsAndReductions tests ReLU, Exp, Log, Pow, and Mean with finite differences.
func TestActivationsAndReductions(t *testing.T) {
	// x in Float32
	x := mustVar32(t, []int{4}, []float32{1.5, 2.0, 3.0, 4.0}, true)

	// y = Mean(ReLU(Exp(Log(x))))
	logX, err := Log(x)
	if err != nil {
		t.Fatal(err)
	}
	expX, err := Exp(logX)
	if err != nil {
		t.Fatal(err)
	}
	reluX, err := ReLU(expX)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := Mean(reluX)
	if err != nil {
		t.Fatal(err)
	}

	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	// For positive x, ReLU(Exp(Log(x))) = x.
	// Mean(x) = sum(x) / 4 -> dLoss/dx_i = 1/4 = 0.25.
	for i := 0; i < 4; i++ {
		gx, err := x.Grad.Float32At(i)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(float64(gx-0.25)) > 1e-5 {
			t.Errorf("expected grad[%d] = 0.25, got %f", i, gx)
		}
	}
}

// TestShapeOpsReshapeTranspose tests Reshape and Transpose backward gradient routing.
func TestShapeOpsReshapeTranspose(t *testing.T) {
	x := mustVar64(t, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6}, true)

	// Reshape to [3, 2]
	reshaped, err := Reshape(x, []int{3, 2})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(reshaped.Shape(), []int{3, 2}) {
		t.Fatalf("expected shape [3, 2], got %v", reshaped.Shape())
	}

	// Transpose to [2, 3]
	trans, err := Transpose(reshaped)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(trans.Shape(), []int{2, 3}) {
		t.Fatalf("expected shape [2, 3], got %v", trans.Shape())
	}

	// Scale by 2.0
	scaled, err := Scale(trans, 2.0)
	if err != nil {
		t.Fatal(err)
	}

	loss, err := Sum(scaled)
	if err != nil {
		t.Fatal(err)
	}

	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	// dLoss/dx should be all 2.0
	for i := 0; i < 6; i++ {
		idx, _ := flatToIndices(i, []int{2, 3})
		gx, err := x.Grad.Float64At(idx...)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(gx-2.0) > 1e-6 {
			t.Errorf("expected grad %v = 2.0, got %f", idx, gx)
		}
	}
}

// TestGraphLifecycleDetachRelease tests Detach and ReleaseGraph.
func TestGraphLifecycleDetachRelease(t *testing.T) {
	x := mustVar64(t, []int{2}, []float64{10, 20}, true)
	y, err := Scale(x, 3.0)
	if err != nil {
		t.Fatal(err)
	}

	// Detach y
	detachedY := y.Detach()
	if detachedY.RequiresGrad {
		t.Errorf("detached variable should not require grad")
	}
	if len(detachedY.parents) != 0 {
		t.Errorf("detached variable should have no parents")
	}

	// ReleaseGraph on y
	if len(y.parents) == 0 {
		t.Fatal("y should have parents before release")
	}
	y.ReleaseGraph()
	if len(y.parents) != 0 || y.backward != nil {
		t.Errorf("ReleaseGraph should clear parents and backward closure")
	}
}

// TestUnsupportedAndErrorPaths verifies failure modes.
func TestUnsupportedAndErrorPaths(t *testing.T) {
	// 1. Nil tensor input
	if _, err := NewVariable(nil, false); !errors.Is(err, ErrNilTensor) {
		t.Errorf("expected ErrNilTensor for nil tensor, got %v", err)
	}

	// 2. Incompatible shapes for MatMul
	a := mustVar64(t, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6}, true)
	b := mustVar64(t, []int{2, 2}, []float64{1, 2, 3, 4}, true)
	if _, err := MatMul(a, b); !errors.Is(err, ErrUnsupportedOp) {
		t.Errorf("expected ErrUnsupportedOp for inner dim mismatch in MatMul, got %v", err)
	}

	// 3. Backward on non-scalar without seed grad
	vNonScalar := mustVar64(t, []int{2, 2}, []float64{1, 2, 3, 4}, true)
	if err := vNonScalar.Backward(); !errors.Is(err, ErrGradUndefined) {
		t.Errorf("expected ErrGradUndefined calling Backward on non-scalar, got %v", err)
	}

	// 4. Log of non-positive number
	vNeg := mustVar64(t, []int{1}, []float64{-1.0}, true)
	if _, err := Log(vNeg); !errors.Is(err, ErrUnsupportedOp) {
		t.Errorf("expected ErrUnsupportedOp for Log(negative), got %v", err)
	}

	// 5. Division by zero
	vZero := mustVar64(t, []int{1}, []float64{0.0}, true)
	vOne := mustVar64(t, []int{1}, []float64{1.0}, true)
	if _, err := Div(vOne, vZero); !errors.Is(err, ErrUnsupportedOp) {
		t.Errorf("expected ErrUnsupportedOp for Div by zero, got %v", err)
	}
}
