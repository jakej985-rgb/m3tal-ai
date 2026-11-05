package transformer

import (
	"math"
	"math/rand"
	"reflect"
	"testing"

	"m3tal-ai/internal/autograd"
	"m3tal-ai/internal/gradcheck"
	"m3tal-ai/internal/tensor"
)

// Helper to create a Float32 Variable with given shape and values.
func makeVar32(t *testing.T, shape []int, values []float32, requiresGrad bool) *autograd.Variable {
	t.Helper()
	tens, err := tensor.FromFloat32(shape, values)
	if err != nil {
		t.Fatalf("failed to create tensor: %v", err)
	}
	v, err := autograd.NewVariable(tens, requiresGrad)
	if err != nil {
		t.Fatalf("failed to create variable: %v", err)
	}
	return v
}

// Helper to create a Float64 Variable with given shape and values.
func makeVar64(t *testing.T, shape []int, values []float64, requiresGrad bool) *autograd.Variable {
	t.Helper()
	tens, err := tensor.FromFloat64(shape, values)
	if err != nil {
		t.Fatalf("failed to create tensor: %v", err)
	}
	v, err := autograd.NewVariable(tens, requiresGrad)
	if err != nil {
		t.Fatalf("failed to create variable: %v", err)
	}
	return v
}

// TestRMSNormNumericalReference verifies RMSNorm forward pass against an explicit analytical reference.
func TestRMSNormNumericalReference(t *testing.T) {
	// 2 rows, 4 columns
	xVals := []float64{
		1.0, 2.0, 3.0, 4.0,
		-1.0, 0.5, -0.5, 2.0,
	}
	wVals := []float64{1.0, 0.5, 2.0, 1.5}
	eps := 1e-5

	xVar := makeVar64(t, []int{2, 4}, xVals, true)
	wVar := makeVar64(t, []int{4}, wVals, true)

	out, err := RMSNorm(xVar, wVar, eps)
	if err != nil {
		t.Fatalf("RMSNorm failed: %v", err)
	}

	// Calculate expected values manually
	expected := make([]float64, 8)
	// Row 0: mean(1^2 + 2^2 + 3^2 + 4^2) = (1 + 4 + 9 + 16)/4 = 30/4 = 7.5
	rms0 := math.Sqrt(7.5 + eps)
	expected[0] = (1.0 / rms0) * 1.0
	expected[1] = (2.0 / rms0) * 0.5
	expected[2] = (3.0 / rms0) * 2.0
	expected[3] = (4.0 / rms0) * 1.5

	// Row 1: mean((-1)^2 + 0.5^2 + (-0.5)^2 + 2^2) = (1 + 0.25 + 0.25 + 4)/4 = 5.5/4 = 1.375
	rms1 := math.Sqrt(1.375 + eps)
	expected[4] = (-1.0 / rms1) * 1.0
	expected[5] = (0.5 / rms1) * 0.5
	expected[6] = (-0.5 / rms1) * 2.0
	expected[7] = (2.0 / rms1) * 1.5

	for i := 0; i < 8; i++ {
		idx, _ := flatToIndices(i, []int{2, 4})
		v, err := out.Data.Float64At(idx...)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(v-expected[i]) > 1e-6 {
			t.Errorf("mismatch at index %d: got %g, expected %g", i, v, expected[i])
		}
	}
}

// TestRMSNormLayer verifies RMSNormLayer initialization, forward shape and scaling.
func TestRMSNormLayer(t *testing.T) {
	dim := 16
	layer, err := NewRMSNorm(dim, 1e-5)
	if err != nil {
		t.Fatalf("NewRMSNorm failed: %v", err)
	}

	// Batch=2, Seq=3, Dim=16
	rng := rand.New(rand.NewSource(123))
	vals := make([]float32, 2*3*dim)
	for i := range vals {
		vals[i] = rng.Float32()*2 - 1
	}

	x := makeVar32(t, []int{2, 3, dim}, vals, true)
	out, err := layer.Forward(x)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	if !reflect.DeepEqual(out.Shape(), []int{2, 3, dim}) {
		t.Fatalf("expected shape %v, got %v", []int{2, 3, dim}, out.Shape())
	}

	// Output norm across last dimension should be close to sqrt(dim) because weights are 1.0
	for b := 0; b < 2; b++ {
		for s := 0; s < 3; s++ {
			var sumSq float64
			for d := 0; d < dim; d++ {
				v, _ := out.Data.Float32At(b, s, d)
				sumSq += float64(v * v)
			}
			meanSq := sumSq / float64(dim)
			// with weight=1.0, mean(y^2) should be approximately 1.0
			if math.Abs(meanSq-1.0) > 1e-3 {
				t.Errorf("RMS of output is %g, expected ~1.0", math.Sqrt(meanSq))
			}
		}
	}
}

// TestRMSNormEdgeCases verifies error handling on invalid shapes, dtypes, or non-positive dimensions.
func TestRMSNormEdgeCases(t *testing.T) {
	// Nil input
	if _, err := RMSNorm(nil, nil, 1e-5); err == nil {
		t.Errorf("expected error on nil input")
	}

	// Invalid dim
	if _, err := NewRMSNorm(0, 1e-5); err == nil {
		t.Errorf("expected error on dim 0")
	}

	// Weight shape mismatch
	x := makeVar32(t, []int{2, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, false)
	wBad := makeVar32(t, []int{5}, []float32{1, 1, 1, 1, 1}, false)
	if _, err := RMSNorm(x, wBad, 1e-5); err == nil {
		t.Errorf("expected error on mismatched weight dimension")
	}

	// DType mismatch
	wDType := makeVar64(t, []int{4}, []float64{1, 1, 1, 1}, false)
	if _, err := RMSNorm(x, wDType, 1e-5); err == nil {
		t.Errorf("expected error on dtype mismatch")
	}
}

// TestRoPENumericalReference verifies RoPE 2D rotation against exact trigonometric computation.
func TestRoPENumericalReference(t *testing.T) {
	// seqLen=2, dim=4
	xVals := []float64{
		1.0, 0.0, 0.0, 2.0, // pos 0
		1.0, 0.0, 0.0, 2.0, // pos 1
	}
	x := makeVar64(t, []int{2, 4}, xVals, true)
	base := 10000.0

	out, err := ApplyRoPE(x, 0, base)
	if err != nil {
		t.Fatalf("ApplyRoPE failed: %v", err)
	}

	theta0 := 1.0                                      // 10000^(-0/4) = 1.0
	theta1 := math.Pow(base, -2.0*1.0/4.0)             // 10000^(-0.5) = 1/100 = 0.01

	// Pos 0: angle = 0 -> identity
	expected0 := []float64{1.0, 0.0, 0.0, 2.0}
	for i := 0; i < 4; i++ {
		v, _ := out.Data.Float64At(0, i)
		if math.Abs(v-expected0[i]) > 1e-7 {
			t.Errorf("pos 0 index %d mismatch: got %g, expected %g", i, v, expected0[i])
		}
	}

	// Pos 1:
	// pair 0: x0=1.0, x1=0.0 -> out0 = 1*cos(theta0) - 0*sin(theta0) = cos(1)
	//                           out1 = 1*sin(theta0) + 0*cos(theta0) = sin(1)
	// pair 1: x2=0.0, x3=2.0 -> out2 = 0*cos(theta1) - 2*sin(theta1) = -2*sin(0.01)
	//                           out3 = 0*sin(theta1) + 2*cos(theta1) = 2*cos(0.01)
	expected1 := []float64{
		math.Cos(theta0),
		math.Sin(theta0),
		-2.0 * math.Sin(theta1),
		2.0 * math.Cos(theta1),
	}
	for i := 0; i < 4; i++ {
		v, _ := out.Data.Float64At(1, i)
		if math.Abs(v-expected1[i]) > 1e-7 {
			t.Errorf("pos 1 index %d mismatch: got %g, expected %g", i, v, expected1[i])
		}
	}
}

// TestRoPEProperties verifies mathematical invariants of RoPE:
// 1. Norm preservation: ||R(x)|| == ||x||
// 2. Relative position shift invariance: <R_m(u), R_n(v)> depends only on (m - n)
// 3. Orthogonality / invertibility: R(-m) * R(m) * x == x
func TestRoPEProperties(t *testing.T) {
	rng := rand.New(rand.NewSource(4242))
	dim := 32
	seqLen := 10
	vals := make([]float64, seqLen*dim)
	for i := range vals {
		vals[i] = rng.Float64()*4 - 2
	}

	x := makeVar64(t, []int{seqLen, dim}, vals, false)
	ropeOut, err := ApplyRoPE(x, 0, DefaultRoPEBase)
	if err != nil {
		t.Fatal(err)
	}

	// 1. Norm preservation
	for s := 0; s < seqLen; s++ {
		var origNormSq, ropeNormSq float64
		for d := 0; d < dim; d++ {
			orig, _ := x.Data.Float64At(s, d)
			origNormSq += orig * orig

			ropeVal, _ := ropeOut.Data.Float64At(s, d)
			ropeNormSq += ropeVal * ropeVal
		}
		if math.Abs(math.Sqrt(origNormSq)-math.Sqrt(ropeNormSq)) > 1e-6 {
			t.Errorf("norm not preserved at seq %d: orig=%g rope=%g", s, origNormSq, ropeNormSq)
		}
	}

	// 2. Relative position shift invariance:
	// Let u and v be vectors. For any shift k:
	// <R_m(u), R_n(v)> == <R_{m+k}(u), R_{n+k}(v)>
	uVals := make([]float64, dim)
	vVals := make([]float64, dim)
	for i := range uVals {
		uVals[i] = rng.Float64()*2 - 1
		vVals[i] = rng.Float64()*2 - 1
	}

	uVar := makeVar64(t, []int{1, dim}, uVals, false)
	vVar := makeVar64(t, []int{1, dim}, vVals, false)

	// Shift k = 7
	m, n, k := 3, 5, 7
	uRotM, _ := ApplyRoPE(uVar, m, DefaultRoPEBase)
	vRotN, _ := ApplyRoPE(vVar, n, DefaultRoPEBase)

	uRotMPlusK, _ := ApplyRoPE(uVar, m+k, DefaultRoPEBase)
	vRotNPlusK, _ := ApplyRoPE(vVar, n+k, DefaultRoPEBase)

	var dot1, dot2 float64
	for d := 0; d < dim; d++ {
		um, _ := uRotM.Data.Float64At(0, d)
		vn, _ := vRotN.Data.Float64At(0, d)
		dot1 += um * vn

		umk, _ := uRotMPlusK.Data.Float64At(0, d)
		vnk, _ := vRotNPlusK.Data.Float64At(0, d)
		dot2 += umk * vnk
	}

	if math.Abs(dot1-dot2) > 1e-6 {
		t.Fatalf("relative position property violated: dot1=%g, dot2=%g", dot1, dot2)
	}
}

// TestRoPEShapes verifies RoPE on 2D, 3D, and 4D tensor layouts (no hidden sequence caps).
func TestRoPEShapes(t *testing.T) {
	testShapes := [][]int{
		{128, 16},          // 2D: [seq, dim] with large sequence
		{2, 64, 32},        // 3D: [batch, seq, dim]
		{2, 16, 4, 32},     // 4D: [batch, seq, heads, headDim]
	}

	for _, shape := range testShapes {
		numel, _ := tensor.Numel(shape)
		vals := make([]float32, numel)
		for i := range vals {
			vals[i] = float32(i % 100)
		}
		x := makeVar32(t, shape, vals, false)
		out, err := ApplyRoPE(x, 0, DefaultRoPEBase)
		if err != nil {
			t.Fatalf("shape %v failed: %v", shape, err)
		}
		if !reflect.DeepEqual(out.Shape(), shape) {
			t.Fatalf("expected shape %v, got %v", shape, out.Shape())
		}
	}

	// Verify odd dim fails
	xOdd := makeVar32(t, []int{2, 7}, make([]float32, 14), false)
	if _, err := ApplyRoPE(xOdd, 0, DefaultRoPEBase); err == nil {
		t.Errorf("expected error for odd RoPE dim")
	}
}

// TestSiLUAndSwiGLUNumericalReference tests SiLU and SwiGLU forward values against math reference.
func TestSiLUAndSwiGLUNumericalReference(t *testing.T) {
	// 1. SiLU
	xVals := []float64{-2.0, -1.0, 0.0, 1.0, 2.0}
	x := makeVar64(t, []int{5}, xVals, true)
	siluOut, err := SiLU(x)
	if err != nil {
		t.Fatal(err)
	}

	for i, val := range xVals {
		sigma := 1.0 / (1.0 + math.Exp(-val))
		expected := val * sigma
		actual, _ := siluOut.Data.Float64At(i)
		if math.Abs(actual-expected) > 1e-7 {
			t.Errorf("SiLU mismatch at %d: got %g, expected %g", i, actual, expected)
		}
	}

	// 2. SwiGLU: SiLU(gate) * up
	gateVals := []float64{0.5, -0.5, 1.5}
	upVals := []float64{2.0, -1.0, 0.5}
	gate := makeVar64(t, []int{3}, gateVals, true)
	up := makeVar64(t, []int{3}, upVals, true)

	swigluOut, err := SwiGLU(gate, up)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 3; i++ {
		g := gateVals[i]
		u := upVals[i]
		sigma := 1.0 / (1.0 + math.Exp(-g))
		expected := (g * sigma) * u
		actual, _ := swigluOut.Data.Float64At(i)
		if math.Abs(actual-expected) > 1e-7 {
			t.Errorf("SwiGLU mismatch at %d: got %g, expected %g", i, actual, expected)
		}
	}
}

// TestSwiGLULayer verifies SwiGLULayer forward dimension propagation and parameter shapes.
func TestSwiGLULayer(t *testing.T) {
	dim := 16
	hiddenDim := 32
	rng := rand.New(rand.NewSource(999))
	layer, err := NewSwiGLU(dim, hiddenDim, rng)
	if err != nil {
		t.Fatal(err)
	}

	// Test with 3D input: [batch=2, seq=4, dim=16]
	xVals := make([]float32, 2*4*dim)
	for i := range xVals {
		xVals[i] = rng.Float32()*2 - 1
	}
	x := makeVar32(t, []int{2, 4, dim}, xVals, true)

	out, err := layer.Forward(x)
	if err != nil {
		t.Fatalf("SwiGLULayer Forward failed: %v", err)
	}

	expectedShape := []int{2, 4, dim}
	if !reflect.DeepEqual(out.Shape(), expectedShape) {
		t.Fatalf("expected shape %v, got %v", expectedShape, out.Shape())
	}
}

// TestRMSNormGradcheck performs finite-difference gradient checking on input and weight of RMSNorm.
func TestRMSNormGradcheck(t *testing.T) {
	cfg := gradcheck.DefaultConfig()
	rng := rand.New(rand.NewSource(101))

	rows, cols := 3, 4
	xVals := make([]float64, rows*cols)
	wVals := make([]float64, cols)
	upstreamVals := make([]float64, rows*cols)

	for i := range xVals {
		xVals[i] = rng.Float64()*2 - 1
		upstreamVals[i] = rng.Float64()*2 - 1
	}
	for i := range wVals {
		wVals[i] = rng.Float64() + 0.5
	}

	xVar := makeVar64(t, []int{rows, cols}, xVals, true)
	wVar := makeVar64(t, []int{cols}, wVals, true)
	upstreamVar := makeVar64(t, []int{rows, cols}, upstreamVals, false)

	normOut, err := RMSNorm(xVar, wVar, 1e-5)
	if err != nil {
		t.Fatal(err)
	}

	lossVar, err := autograd.Mul(normOut, upstreamVar)
	if err != nil {
		t.Fatal(err)
	}
	scalarLoss, err := autograd.Sum(lossVar)
	if err != nil {
		t.Fatal(err)
	}

	if err := scalarLoss.Backward(); err != nil {
		t.Fatal(err)
	}

	// 1. Check Input gradient
	evalX := func(vals []float64) float64 {
		tX, _ := tensor.FromFloat64([]int{rows, cols}, vals)
		vX, _ := autograd.NewVariable(tX, false)
		tW, _ := tensor.FromFloat64([]int{cols}, wVals)
		vW, _ := autograd.NewVariable(tW, false)
		y, err := RMSNorm(vX, vW, 1e-5)
		if err != nil {
			t.Fatal(err)
		}
		var loss float64
		for i := 0; i < rows*cols; i++ {
			idx, _ := flatToIndices(i, []int{rows, cols})
			vy, _ := y.Data.Float64At(idx...)
			loss += vy * upstreamVals[i]
		}
		return loss
	}

	xAnalytic := make([]float64, rows*cols)
	for i := 0; i < rows*cols; i++ {
		idx, _ := flatToIndices(i, []int{rows, cols})
		xAnalytic[i], _ = xVar.Grad.Float64At(idx...)
	}

	if err := gradcheck.CheckSlice("RMSNorm", "x", xVals, xAnalytic, evalX, cfg); err != nil {
		t.Fatalf("RMSNorm input gradcheck failed: %v", err)
	}

	// 2. Check Weight gradient
	evalW := func(vals []float64) float64 {
		tX, _ := tensor.FromFloat64([]int{rows, cols}, xVals)
		vX, _ := autograd.NewVariable(tX, false)
		tW, _ := tensor.FromFloat64([]int{cols}, vals)
		vW, _ := autograd.NewVariable(tW, false)
		y, err := RMSNorm(vX, vW, 1e-5)
		if err != nil {
			t.Fatal(err)
		}
		var loss float64
		for i := 0; i < rows*cols; i++ {
			idx, _ := flatToIndices(i, []int{rows, cols})
			vy, _ := y.Data.Float64At(idx...)
			loss += vy * upstreamVals[i]
		}
		return loss
	}

	wAnalytic := make([]float64, cols)
	for i := 0; i < cols; i++ {
		wAnalytic[i], _ = wVar.Grad.Float64At(i)
	}

	if err := gradcheck.CheckSlice("RMSNorm", "weight", wVals, wAnalytic, evalW, cfg); err != nil {
		t.Fatalf("RMSNorm weight gradcheck failed: %v", err)
	}
}

// TestRoPEGradcheck performs finite-difference gradient checking on RoPE input variable.
func TestRoPEGradcheck(t *testing.T) {
	cfg := gradcheck.DefaultConfig()
	rng := rand.New(rand.NewSource(202))

	seqLen, dim := 3, 4
	xVals := make([]float64, seqLen*dim)
	upstreamVals := make([]float64, seqLen*dim)

	for i := range xVals {
		xVals[i] = rng.Float64()*2 - 1
		upstreamVals[i] = rng.Float64()*2 - 1
	}

	xVar := makeVar64(t, []int{seqLen, dim}, xVals, true)
	upstreamVar := makeVar64(t, []int{seqLen, dim}, upstreamVals, false)

	ropeOut, err := ApplyRoPE(xVar, 1, DefaultRoPEBase)
	if err != nil {
		t.Fatal(err)
	}

	lossVar, err := autograd.Mul(ropeOut, upstreamVar)
	if err != nil {
		t.Fatal(err)
	}
	scalarLoss, err := autograd.Sum(lossVar)
	if err != nil {
		t.Fatal(err)
	}

	if err := scalarLoss.Backward(); err != nil {
		t.Fatal(err)
	}

	evalX := func(vals []float64) float64 {
		tX, _ := tensor.FromFloat64([]int{seqLen, dim}, vals)
		vX, _ := autograd.NewVariable(tX, false)
		y, err := ApplyRoPE(vX, 1, DefaultRoPEBase)
		if err != nil {
			t.Fatal(err)
		}
		var loss float64
		for i := 0; i < seqLen*dim; i++ {
			idx, _ := flatToIndices(i, []int{seqLen, dim})
			vy, _ := y.Data.Float64At(idx...)
			loss += vy * upstreamVals[i]
		}
		return loss
	}

	xAnalytic := make([]float64, seqLen*dim)
	for i := 0; i < seqLen*dim; i++ {
		idx, _ := flatToIndices(i, []int{seqLen, dim})
		xAnalytic[i], _ = xVar.Grad.Float64At(idx...)
	}

	if err := gradcheck.CheckSlice("RoPE", "x", xVals, xAnalytic, evalX, cfg); err != nil {
		t.Fatalf("RoPE input gradcheck failed: %v", err)
	}
}

// TestSwiGLUGradcheck performs finite-difference gradient checking on gate and up variables of SwiGLU.
func TestSwiGLUGradcheck(t *testing.T) {
	cfg := gradcheck.DefaultConfig()
	rng := rand.New(rand.NewSource(303))

	n := 6
	gVals := make([]float64, n)
	uVals := make([]float64, n)
	upstreamVals := make([]float64, n)

	for i := 0; i < n; i++ {
		gVals[i] = rng.Float64()*3 - 1.5
		uVals[i] = rng.Float64()*3 - 1.5
		upstreamVals[i] = rng.Float64()*2 - 1
	}

	gVar := makeVar64(t, []int{n}, gVals, true)
	uVar := makeVar64(t, []int{n}, uVals, true)
	upstreamVar := makeVar64(t, []int{n}, upstreamVals, false)

	swigluOut, err := SwiGLU(gVar, uVar)
	if err != nil {
		t.Fatal(err)
	}

	lossVar, err := autograd.Mul(swigluOut, upstreamVar)
	if err != nil {
		t.Fatal(err)
	}
	scalarLoss, err := autograd.Sum(lossVar)
	if err != nil {
		t.Fatal(err)
	}

	if err := scalarLoss.Backward(); err != nil {
		t.Fatal(err)
	}

	// 1. Check Gate gradient
	evalGate := func(vals []float64) float64 {
		tG, _ := tensor.FromFloat64([]int{n}, vals)
		vG, _ := autograd.NewVariable(tG, false)
		tU, _ := tensor.FromFloat64([]int{n}, uVals)
		vU, _ := autograd.NewVariable(tU, false)
		y, err := SwiGLU(vG, vU)
		if err != nil {
			t.Fatal(err)
		}
		var loss float64
		for i := 0; i < n; i++ {
			vy, _ := y.Data.Float64At(i)
			loss += vy * upstreamVals[i]
		}
		return loss
	}

	gAnalytic := make([]float64, n)
	for i := 0; i < n; i++ {
		gAnalytic[i], _ = gVar.Grad.Float64At(i)
	}

	if err := gradcheck.CheckSlice("SwiGLU", "gate", gVals, gAnalytic, evalGate, cfg); err != nil {
		t.Fatalf("SwiGLU gate gradcheck failed: %v", err)
	}

	// 2. Check Up gradient
	evalUp := func(vals []float64) float64 {
		tG, _ := tensor.FromFloat64([]int{n}, gVals)
		vG, _ := autograd.NewVariable(tG, false)
		tU, _ := tensor.FromFloat64([]int{n}, vals)
		vU, _ := autograd.NewVariable(tU, false)
		y, err := SwiGLU(vG, vU)
		if err != nil {
			t.Fatal(err)
		}
		var loss float64
		for i := 0; i < n; i++ {
			vy, _ := y.Data.Float64At(i)
			loss += vy * upstreamVals[i]
		}
		return loss
	}

	uAnalytic := make([]float64, n)
	for i := 0; i < n; i++ {
		uAnalytic[i], _ = uVar.Grad.Float64At(i)
	}

	if err := gradcheck.CheckSlice("SwiGLU", "up", uVals, uAnalytic, evalUp, cfg); err != nil {
		t.Fatalf("SwiGLU up gradcheck failed: %v", err)
	}
}
