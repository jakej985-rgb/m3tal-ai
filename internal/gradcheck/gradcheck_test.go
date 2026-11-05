package gradcheck

import (
	"errors"
	"math"
	"math/rand"
	"strings"
	"testing"

	"m3tal-ai/internal/autograd"
	"m3tal-ai/internal/lm"
	"m3tal-ai/internal/nn"
	"m3tal-ai/internal/tensor"
)

// TestGradientMismatchReporting verifies that deliberate gradient errors produce clear diagnostic strings
// reporting operation name, parameter name, index, analytic vs numerical, and diffs.
func TestGradientMismatchReporting(t *testing.T) {
	cfg := DefaultConfig()
	x := 2.5
	analyticGrad := 10.0 // Wrong! Derivative of x^2 at 2.5 is 5.0

	err := CheckScalar("square_op", "x", x, analyticGrad, func(val float64) float64 {
		return val * val
	}, cfg)

	if err == nil {
		t.Fatal("expected gradient mismatch error, got nil")
	}

	var mismatch *GradientMismatchError
	if !errors.As(err, &mismatch) {
		t.Fatalf("expected error of type *GradientMismatchError, got %T: %v", err, err)
	}

	if mismatch.OpName != "square_op" || mismatch.ParamName != "x" || mismatch.Index != "[0]" {
		t.Fatalf("unexpected metadata in mismatch error: %+v", mismatch)
	}

	errStr := err.Error()
	for _, substr := range []string{"square_op", "param \"x\"", "index [0]", "analytic=10", "numerical=5"} {
		if !strings.Contains(errStr, substr) {
			t.Errorf("error string %q should contain %q", errStr, substr)
		}
	}
}

// TestAutogradScalarOps tests every autograd scalar operation independently:
// Add, Mul, Div, Tanh, Exp, Log across randomized values.
func TestAutogradScalarOps(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	cfg := DefaultConfig()

	for trial := 0; trial < 10; trial++ {
		// 1. Add
		a := rng.Float64()*10 - 5
		b := rng.Float64()*10 - 5
		va := autograd.New(a)
		vb := autograd.New(b)
		vAdd := va.Add(vb)
		vAdd.Backward()

		if err := CheckScalar("autograd.Add", "a", a, va.Grad, func(x float64) float64 {
			return x + b
		}, cfg); err != nil {
			t.Fatal(err)
		}
		if err := CheckScalar("autograd.Add", "b", b, vb.Grad, func(y float64) float64 {
			return a + y
		}, cfg); err != nil {
			t.Fatal(err)
		}

		// 2. Mul
		va = autograd.New(a)
		vb = autograd.New(b)
		vMul := va.Mul(vb)
		vMul.Backward()

		if err := CheckScalar("autograd.Mul", "a", a, va.Grad, func(x float64) float64 {
			return x * b
		}, cfg); err != nil {
			t.Fatal(err)
		}
		if err := CheckScalar("autograd.Mul", "b", b, vb.Grad, func(y float64) float64 {
			return a * y
		}, cfg); err != nil {
			t.Fatal(err)
		}

		// 3. Div (ensure non-zero denominator)
		denom := rng.Float64()*5 + 0.5
		if rng.Float64() < 0.5 {
			denom = -denom
		}
		va = autograd.New(a)
		vb = autograd.New(denom)
		vDiv := va.Div(vb)
		vDiv.Backward()

		if err := CheckScalar("autograd.Div", "num", a, va.Grad, func(x float64) float64 {
			return x / denom
		}, cfg); err != nil {
			t.Fatal(err)
		}
		if err := CheckScalar("autograd.Div", "denom", denom, vb.Grad, func(y float64) float64 {
			return a / y
		}, cfg); err != nil {
			t.Fatal(err)
		}

		// 4. Tanh
		va = autograd.New(a)
		vTanh := va.Tanh()
		vTanh.Backward()

		if err := CheckScalar("autograd.Tanh", "a", a, va.Grad, func(x float64) float64 {
			return math.Tanh(x)
		}, cfg); err != nil {
			t.Fatal(err)
		}

		// 5. Exp (keep a in reasonable range to avoid overflow)
		aSmall := rng.Float64()*4 - 2
		va = autograd.New(aSmall)
		vExp := va.Exp()
		vExp.Backward()

		if err := CheckScalar("autograd.Exp", "a", aSmall, va.Grad, func(x float64) float64 {
			return math.Exp(x)
		}, cfg); err != nil {
			t.Fatal(err)
		}

		// 6. Log (positive domain)
		aPos := rng.Float64()*10 + 0.1
		va = autograd.New(aPos)
		vLog := va.Log()
		vLog.Backward()

		if err := CheckScalar("autograd.Log", "a", aPos, va.Grad, func(x float64) float64 {
			return math.Log(x)
		}, cfg); err != nil {
			t.Fatal(err)
		}
	}
}

// TestNNTanhLayer tests Tanh and TanhLayer forward/backward with randomized and boundary-shaped inputs.
func TestNNTanhLayer(t *testing.T) {
	cfg := DefaultConfig()
	shapes := [][2]int{
		{1, 1}, // Single element boundary case
		{1, 5}, // 1-row vector
		{7, 1}, // 1-column vector
		{3, 4}, // Regular rectangular matrix
		{2, 7}, // Odd dimensions
	}

	rng := rand.New(rand.NewSource(101))

	for _, shape := range shapes {
		rows, cols := shape[0], shape[1]
		inputVals := make([]float64, rows*cols)
		upstreamVals := make([]float64, rows*cols)
		for i := range inputVals {
			inputVals[i] = rng.Float64()*4 - 2
			upstreamVals[i] = rng.Float64()*2 - 1
		}

		input, err := tensor.NewMatrix(rows, cols, inputVals)
		if err != nil {
			t.Fatal(err)
		}
		upstream, err := tensor.NewMatrix(rows, cols, upstreamVals)
		if err != nil {
			t.Fatal(err)
		}

		layer := &nn.TanhLayer{}
		_, err = layer.Forward(input)
		if err != nil {
			t.Fatal(err)
		}
		inputGrad, err := layer.Backward(upstream)
		if err != nil {
			t.Fatal(err)
		}

		// Scalar objective: sum_ij (Tanh(X)_ij * Upstream_ij)
		evalFn := func(m tensor.Matrix) (float64, error) {
			out, err := nn.Tanh(m)
			if err != nil {
				return 0, err
			}
			var loss float64
			for r := 0; r < rows; r++ {
				for c := 0; c < cols; c++ {
					loss += out.At(r, c) * upstream.At(r, c)
				}
			}
			return loss, nil
		}

		if err := CheckMatrix("nn.TanhLayer", "input", input, inputGrad, evalFn, cfg); err != nil {
			t.Fatalf("shape %vx%v failed: %v", rows, cols, err)
		}
	}
}

// TestNNReLULayer tests ReLULayer forward/backward with non-zero inputs (avoiding non-differentiable x=0).
func TestNNReLULayer(t *testing.T) {
	cfg := DefaultConfig()
	shapes := [][2]int{
		{1, 1},
		{1, 4},
		{5, 1},
		{3, 3},
	}

	rng := rand.New(rand.NewSource(202))

	for _, shape := range shapes {
		rows, cols := shape[0], shape[1]
		inputVals := make([]float64, rows*cols)
		upstreamVals := make([]float64, rows*cols)
		for i := range inputVals {
			val := rng.Float64()*4 - 2
			// Avoid values within [-0.05, 0.05] to prevent finite difference perturbation from crossing 0
			if math.Abs(val) < 0.05 {
				val = 0.5
			}
			inputVals[i] = val
			upstreamVals[i] = rng.Float64()*2 - 1
		}

		input, err := tensor.NewMatrix(rows, cols, inputVals)
		if err != nil {
			t.Fatal(err)
		}
		upstream, err := tensor.NewMatrix(rows, cols, upstreamVals)
		if err != nil {
			t.Fatal(err)
		}

		layer := &nn.ReLULayer{}
		_, err = layer.Forward(input)
		if err != nil {
			t.Fatal(err)
		}
		inputGrad, err := layer.Backward(upstream)
		if err != nil {
			t.Fatal(err)
		}

		evalFn := func(m tensor.Matrix) (float64, error) {
			var loss float64
			for r := 0; r < rows; r++ {
				for c := 0; c < cols; c++ {
					v := m.At(r, c)
					if v > 0 {
						loss += v * upstream.At(r, c)
					}
				}
			}
			return loss, nil
		}

		if err := CheckMatrix("nn.ReLULayer", "input", input, inputGrad, evalFn, cfg); err != nil {
			t.Fatalf("shape %vx%v failed: %v", rows, cols, err)
		}
	}
}

// TestNNDenseLayer tests input, weight, and bias gradients of Dense layer across randomized and boundary shapes.
func TestNNDenseLayer(t *testing.T) {
	cfg := DefaultConfig()
	configs := []struct {
		batch      int
		inputSize  int
		outputSize int
	}{
		{batch: 1, inputSize: 1, outputSize: 1}, // 1x1x1 boundary case
		{batch: 1, inputSize: 3, outputSize: 2}, // batch 1
		{batch: 4, inputSize: 1, outputSize: 3}, // input size 1
		{batch: 3, inputSize: 4, outputSize: 1}, // output size 1
		{batch: 5, inputSize: 3, outputSize: 4}, // rectangular general case
	}

	rng := rand.New(rand.NewSource(303))

	for _, c := range configs {
		layer, err := nn.NewDense(c.inputSize, c.outputSize, rng)
		if err != nil {
			t.Fatal(err)
		}

		inputVals := make([]float64, c.batch*c.inputSize)
		upstreamVals := make([]float64, c.batch*c.outputSize)
		for i := range inputVals {
			inputVals[i] = rng.Float64()*2 - 1
		}
		for i := range upstreamVals {
			upstreamVals[i] = rng.Float64()*2 - 1
		}

		input, err := tensor.NewMatrix(c.batch, c.inputSize, inputVals)
		if err != nil {
			t.Fatal(err)
		}
		upstream, err := tensor.NewMatrix(c.batch, c.outputSize, upstreamVals)
		if err != nil {
			t.Fatal(err)
		}

		layer.ZeroGrad()
		_, err = layer.Forward(input)
		if err != nil {
			t.Fatal(err)
		}
		inputGrad, err := layer.Backward(upstream)
		if err != nil {
			t.Fatal(err)
		}

		weightGrad, biasGrad := layer.Gradients()

		// 1. Check Input gradient
		evalInput := func(m tensor.Matrix) (float64, error) {
			tempLayer := *layer
			out, err := tempLayer.Forward(m)
			if err != nil {
				return 0, err
			}
			var loss float64
			for r := 0; r < c.batch; r++ {
				for col := 0; col < c.outputSize; col++ {
					loss += out.At(r, col) * upstream.At(r, col)
				}
			}
			return loss, nil
		}
		if err := CheckMatrix("nn.Dense", "input", input, inputGrad, evalInput, cfg); err != nil {
			t.Fatalf("Dense input gradient check failed for config %+v: %v", c, err)
		}

		// 2. Check Weight gradient
		origWeights := layer.Parameters()[0].Values()
		evalWeights := func(w []float64) float64 {
			_ = layer.Parameters()[0].Load(w)
			out, err := layer.Forward(input)
			if err != nil {
				t.Fatal(err)
			}
			var loss float64
			for r := 0; r < c.batch; r++ {
				for col := 0; col < c.outputSize; col++ {
					loss += out.At(r, col) * upstream.At(r, col)
				}
			}
			return loss
		}
		if err := CheckSlice("nn.Dense", "weights", origWeights, weightGrad, evalWeights, cfg); err != nil {
			t.Fatalf("Dense weight gradient check failed for config %+v: %v", c, err)
		}
		_ = layer.Parameters()[0].Load(origWeights)

		// 3. Check Bias gradient
		origBias := layer.Parameters()[1].Values()
		evalBias := func(b []float64) float64 {
			_ = layer.Parameters()[1].Load(b)
			out, err := layer.Forward(input)
			if err != nil {
				t.Fatal(err)
			}
			var loss float64
			for r := 0; r < c.batch; r++ {
				for col := 0; col < c.outputSize; col++ {
					loss += out.At(r, col) * upstream.At(r, col)
				}
			}
			return loss
		}
		if err := CheckSlice("nn.Dense", "bias", origBias, biasGrad, evalBias, cfg); err != nil {
			t.Fatalf("Dense bias gradient check failed for config %+v: %v", c, err)
		}
		_ = layer.Parameters()[1].Load(origBias)
	}
}

// TestNNLosses tests MSE and SoftmaxCrossEntropy gradient computation with finite-difference check.
func TestNNLosses(t *testing.T) {
	cfg := DefaultConfig()
	rng := rand.New(rand.NewSource(404))

	// 1. MSE Loss
	mseShapes := [][2]int{
		{1, 1},
		{1, 5},
		{4, 1},
		{3, 4},
	}
	for _, shape := range mseShapes {
		rows, cols := shape[0], shape[1]
		predVals := make([]float64, rows*cols)
		targetVals := make([]float64, rows*cols)
		for i := range predVals {
			predVals[i] = rng.Float64()*4 - 2
			targetVals[i] = rng.Float64()*4 - 2
		}

		pred, err := tensor.NewMatrix(rows, cols, predVals)
		if err != nil {
			t.Fatal(err)
		}
		targets, err := tensor.NewMatrix(rows, cols, targetVals)
		if err != nil {
			t.Fatal(err)
		}

		mse := nn.MSE{}
		grad, err := mse.Gradient(pred, targets)
		if err != nil {
			t.Fatal(err)
		}

		evalFn := func(m tensor.Matrix) (float64, error) {
			return mse.Value(m, targets)
		}

		if err := CheckMatrix("nn.MSE", "prediction", pred, grad, evalFn, cfg); err != nil {
			t.Fatalf("MSE gradient check failed for shape %vx%v: %v", rows, cols, err)
		}
	}

	// 2. SoftmaxCrossEntropy Loss
	ceShapes := [][2]int{
		{1, 2}, // Smallest valid classification (min 2 classes)
		{1, 5},
		{3, 4},
		{5, 3},
	}
	for _, shape := range ceShapes {
		rows, cols := shape[0], shape[1]
		logitsVals := make([]float64, rows*cols)
		targetVals := make([]float64, rows*cols)

		for r := 0; r < rows; r++ {
			// One-hot targets
			targetCol := rng.Intn(cols)
			targetVals[r*cols+targetCol] = 1.0
			for c := 0; c < cols; c++ {
				logitsVals[r*cols+c] = rng.Float64()*2 - 1
			}
		}

		logits, err := tensor.NewMatrix(rows, cols, logitsVals)
		if err != nil {
			t.Fatal(err)
		}
		targets, err := tensor.NewMatrix(rows, cols, targetVals)
		if err != nil {
			t.Fatal(err)
		}

		ce := nn.SoftmaxCrossEntropy{}
		grad, err := ce.Gradient(logits, targets)
		if err != nil {
			t.Fatal(err)
		}

		evalFn := func(m tensor.Matrix) (float64, error) {
			return ce.Value(m, targets)
		}

		if err := CheckMatrix("nn.SoftmaxCrossEntropy", "logits", logits, grad, evalFn, cfg); err != nil {
			t.Fatalf("SoftmaxCrossEntropy gradient check failed for shape %vx%v: %v", rows, cols, err)
		}
	}
}

// TestLMTransformerPrimitivesGradcheck tests model-critical transformer operations in internal/lm
// against finite-difference estimates: linear layers, multi-layer multi-head causal attention, and masked loss.
func TestLMTransformerPrimitivesGradcheck(t *testing.T) {
	cfg := Float32Config()

	corpus := []string{"transformer primitives gradcheck testing café"}
	tokenizer, err := lm.TrainTokenizer(corpus, 256)
	if err != nil {
		t.Fatal(err)
	}

	model, err := lm.NewModel(lm.Config{
		Context: 8,
		Width:   8,
		Layers:  2,
		Heads:   2,
	}, tokenizer, 777)
	if err != nil {
		t.Fatal(err)
	}

	// Small input sequence
	prompt := "testing"
	ids := tokenizer.Encode(prompt)
	if len(ids) < 3 {
		ids = []int{10, 20, 30}
	}
	input := ids[:len(ids)-1]
	targets := ids[1:]

	// 1. Unmasked training step gradcheck
	modelParameters := model.ParametersInfo()
	model.ComputeGradients(input, targets, nil)

	evalObjective := func(pIdx int) func([]float32) float64 {
		return func(vals []float32) float64 {
			copy(modelParameters[pIdx].Values, vals)
			return model.EvaluateWindowLoss(input, targets, nil)
		}
	}

	for pIdx, p := range modelParameters {
		analyticGrads := p.Gradient
		originalVals := append([]float32(nil), p.Values...)
		paramName := p.Name
		if err := CheckSliceFloat32("lm.TransformerBlock", paramName, originalVals, analyticGrads, evalObjective(pIdx), cfg); err != nil {
			t.Fatalf("LM parameter %d (%s) gradcheck failed: %v", pIdx, paramName, err)
		}
		copy(p.Values, originalVals)
	}

	// 2. Masked training step gradcheck (instruction loss mask)
	mask := make([]bool, len(targets))
	for i := 1; i < len(mask); i++ {
		mask[i] = true
	}
	model.ComputeGradients(input, targets, mask)

	evalMaskedObjective := func(pIdx int) func([]float32) float64 {
		return func(vals []float32) float64 {
			copy(modelParameters[pIdx].Values, vals)
			return model.EvaluateWindowLoss(input, targets, mask)
		}
	}

	for pIdx, p := range modelParameters {
		analyticGrads := p.Gradient
		originalVals := append([]float32(nil), p.Values...)
		paramName := p.Name
		if err := CheckSliceFloat32("lm.TransformerBlock_Masked", paramName, originalVals, analyticGrads, evalMaskedObjective(pIdx), cfg); err != nil {
			t.Fatalf("LM masked parameter %d (%s) gradcheck failed: %v", pIdx, paramName, err)
		}
		copy(p.Values, originalVals)
	}
}
