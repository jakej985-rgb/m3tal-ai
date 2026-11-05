package nn

import (
	"math"
	"math/rand"
	"testing"

	"m3tal-llm/internal/autograd"
	"m3tal-llm/internal/gradcheck"
	"m3tal-llm/internal/tensor"
)

// TestCausalMaskingPreventsLeakage tests that modifying future input tokens at position > t
// has ZERO effect on activations and gradients at position t.
func TestCausalMaskingPreventsLeakage(t *testing.T) {
	rng := rand.New(rand.NewSource(123))
	cfg := CausalSelfAttentionConfig{
		Dim:       16,
		Heads:     2,
		HeadDim:   8,
		MaxSeqLen: 16,
		Bias:      true,
		DType:     tensor.Float32,
	}
	csa, err := NewCausalSelfAttention(cfg, rng)
	if err != nil {
		t.Fatal(err)
	}

	batchSize := 1
	seqLen := 4
	xData1, _ := tensor.New([]int{batchSize, seqLen, cfg.Dim}, tensor.Float32)
	xData2, _ := tensor.New([]int{batchSize, seqLen, cfg.Dim}, tensor.Float32)

	// Populate identical tokens for positions 0 and 1, but completely different at position 2 and 3
	for p := 0; p < seqLen; p++ {
		for d := 0; d < cfg.Dim; d++ {
			val := float32(rng.Float64()*2 - 1)
			_ = xData1.SetFloat32At(val, 0, p, d)
			if p <= 1 {
				_ = xData2.SetFloat32At(val, 0, p, d)
			} else {
				// Different token at future positions 2 and 3
				_ = xData2.SetFloat32At(val+100.0, 0, p, d)
			}
		}
	}

	xVar1, _ := autograd.NewVariable(xData1, true)
	xVar2, _ := autograd.NewVariable(xData2, true)

	out1, err := csa.Forward(xVar1)
	if err != nil {
		t.Fatal(err)
	}
	out2, err := csa.Forward(xVar2)
	if err != nil {
		t.Fatal(err)
	}

	// 1. Forward verification: outputs at pos 0 and 1 must be identical
	for p := 0; p <= 1; p++ {
		for d := 0; d < cfg.Dim; d++ {
			v1, _ := out1.Data.Float32At(0, p, d)
			v2, _ := out2.Data.Float32At(0, p, d)
			diff := math.Abs(float64(v1 - v2))
			if diff > 1e-6 {
				t.Fatalf("causal leakage in forward pass at pos %d, dim %d: v1=%f, v2=%f, diff=%e", p, d, v1, v2, diff)
			}
		}
	}

	// 2. Backward verification: seed gradient only at pos 1. Gradients at future pos 2 and 3 must be exactly 0!
	seedGrad, _ := tensor.New([]int{batchSize, seqLen, cfg.Dim}, tensor.Float32)
	for d := 0; d < cfg.Dim; d++ {
		_ = seedGrad.SetFloat32At(1.0, 0, 1, d)
	}

	if err := out1.Backward(seedGrad); err != nil {
		t.Fatal(err)
	}

	for p := 2; p < seqLen; p++ {
		for d := 0; d < cfg.Dim; d++ {
			g, _ := xVar1.Grad.Float32At(0, p, d)
			if g != 0 {
				t.Fatalf("gradient leakage to future token: xVar1.Grad[0, %d, %d] = %f, want 0", p, d, g)
			}
		}
	}
}

// TestCachedVsUncachedEquivalence tests that step-by-step autoregressive generation with KV cache
// produces identical logits to full uncached sequence forward pass.
func TestCachedVsUncachedEquivalence(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	cfg := CausalSelfAttentionConfig{
		Dim:       32,
		Heads:     4,
		HeadDim:   8,
		MaxSeqLen: 32,
		Bias:      true,
		DType:     tensor.Float32,
	}
	csa, err := NewCausalSelfAttention(cfg, rng)
	if err != nil {
		t.Fatal(err)
	}

	batchSize := 2
	seqLen := 6
	fullInput, _ := tensor.New([]int{batchSize, seqLen, cfg.Dim}, tensor.Float32)
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < cfg.Dim; d++ {
				_ = fullInput.SetFloat32At(float32(rng.Float64()*2-1), b, s, d)
			}
		}
	}

	// 1. Uncached forward (inference) over all tokens at once
	uncachedOut, err := csa.ForwardInference(fullInput, nil, 0)
	if err != nil {
		t.Fatalf("ForwardInference uncached failed: %v", err)
	}

	// 2. Cached token-by-token generation with KVCache
	cache, err := NewKVCache(1, batchSize, cfg.Heads, cfg.HeadDim, cfg.MaxSeqLen, tensor.Float32)
	if err != nil {
		t.Fatalf("NewKVCache failed: %v", err)
	}

	// Feed tokens one by one
	cachedOutTokens := make([]*tensor.Tensor, seqLen)
	for s := 0; s < seqLen; s++ {
		// Slice single token [B, 1, Dim]
		tokenInput, _ := tensor.New([]int{batchSize, 1, cfg.Dim}, tensor.Float32)
		for b := 0; b < batchSize; b++ {
			for d := 0; d < cfg.Dim; d++ {
				val, _ := fullInput.Float32At(b, s, d)
				_ = tokenInput.SetFloat32At(val, b, 0, d)
			}
		}

		tokOut, err := csa.ForwardInference(tokenInput, cache, 0)
		if err != nil {
			t.Fatalf("ForwardInference with cache at step %d failed: %v", s, err)
		}
		cachedOutTokens[s] = tokOut
	}

	// 3. Compare cached vs uncached outputs at every position
	for s := 0; s < seqLen; s++ {
		tokOut := cachedOutTokens[s]
		for b := 0; b < batchSize; b++ {
			for d := 0; d < cfg.Dim; d++ {
				expected, _ := uncachedOut.Float32At(b, s, d)
				actual, _ := tokOut.Float32At(b, 0, d)
				diff := math.Abs(float64(expected - actual))
				if diff > 1e-5 {
					t.Fatalf("mismatch at pos %d, batch %d, dim %d: uncached=%f, cached=%f, diff=%e", s, b, d, expected, actual, diff)
				}
			}
		}
	}
}

// TestAttentionShapeCoverage tests various batch, head, sequence length, and head dimension configurations.
func TestAttentionShapeCoverage(t *testing.T) {
	testCases := []struct {
		name    string
		batch   int
		heads   int
		headDim int
		dim     int
		seqLen  int
		bias    bool
		dtype   tensor.DType
	}{
		{"B1_H1_T1", 1, 1, 16, 16, 1, false, tensor.Float32},
		{"B2_H2_T5", 2, 2, 8, 16, 5, true, tensor.Float32},
		{"B4_H4_T17", 4, 4, 16, 64, 17, true, tensor.Float32},
		{"Float64_B1_H2_T3", 1, 2, 8, 16, 3, false, tensor.Float64},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(99))
			cfg := CausalSelfAttentionConfig{
				Dim:       tc.dim,
				Heads:     tc.heads,
				HeadDim:   tc.headDim,
				MaxSeqLen: tc.seqLen + 10,
				Bias:      tc.bias,
				DType:     tc.dtype,
			}
			csa, err := NewCausalSelfAttention(cfg, rng)
			if err != nil {
				t.Fatalf("NewCausalSelfAttention failed: %v", err)
			}

			// Test Forward (autograd)
			inputData, _ := tensor.New([]int{tc.batch, tc.seqLen, tc.dim}, tc.dtype)
			xVar, _ := autograd.NewVariable(inputData, true)

			outVar, err := csa.Forward(xVar)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}
			expectedShape := []int{tc.batch, tc.seqLen, tc.dim}
			for i, dim := range outVar.Shape() {
				if dim != expectedShape[i] {
					t.Fatalf("shape mismatch: got %v, want %v", outVar.Shape(), expectedShape)
				}
			}

			// Test backward on forward output
			loss, err := autograd.Mean(outVar)
			if err != nil {
				t.Fatal(err)
			}
			if err := loss.Backward(); err != nil {
				t.Fatalf("Backward failed: %v", err)
			}
			if xVar.Grad == nil {
				t.Fatalf("xVar.Grad is nil after Backward")
			}

			// Test ForwardInference with KVCache
			cache, err := NewKVCache(1, tc.batch, tc.heads, tc.headDim, cfg.MaxSeqLen, tc.dtype)
			if err != nil {
				t.Fatal(err)
			}
			inferOut, err := csa.ForwardInference(inputData, cache, 0)
			if err != nil {
				t.Fatalf("ForwardInference failed: %v", err)
			}
			for i, dim := range inferOut.Shape() {
				if dim != expectedShape[i] {
					t.Fatalf("inference shape mismatch: got %v, want %v", inferOut.Shape(), expectedShape)
				}
			}
		})
	}
}

// TestAttentionGradCheck validates analytical gradients against finite differences using internal/gradcheck.
func TestAttentionGradCheck(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	cfg := CausalSelfAttentionConfig{
		Dim:       8,
		Heads:     2,
		HeadDim:   4,
		MaxSeqLen: 8,
		Bias:      true,
		DType:     tensor.Float64, // float64 for high numerical precision in finite diff
	}
	csa, err := NewCausalSelfAttention(cfg, rng)
	if err != nil {
		t.Fatal(err)
	}

	batchSize := 1
	seqLen := 3
	xTensor, _ := tensor.New([]int{batchSize, seqLen, cfg.Dim}, tensor.Float64)
	for i := 0; i < xTensor.Numel(); i++ {
		val := (rng.Float64()*2 - 1) * 0.5
		idx := flatToCoords(i, xTensor.Shape())
		_ = xTensor.SetFloat64At(val, idx...)
	}

	csa.ZeroGrad()
	xVar, err := autograd.NewVariable(xTensor, true)
	if err != nil {
		t.Fatal(err)
	}
	out, err := csa.Forward(xVar)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := autograd.Mean(out)
	if err != nil {
		t.Fatal(err)
	}
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	numel := xTensor.Numel()
	vals := make([]float64, numel)
	analyticGrads := make([]float64, numel)
	for i := 0; i < numel; i++ {
		idx := flatToCoords(i, xTensor.Shape())
		v, _ := xTensor.Float64At(idx...)
		g, _ := xVar.Grad.Float64At(idx...)
		vals[i] = v
		analyticGrads[i] = g
	}

	evalFn := func(v []float64) float64 {
		tIn, _ := tensor.New([]int{batchSize, seqLen, cfg.Dim}, tensor.Float64)
		for i := 0; i < numel; i++ {
			idx := flatToCoords(i, tIn.Shape())
			_ = tIn.SetFloat64At(v[i], idx...)
		}
		csa.ZeroGrad()
		vVar, _ := autograd.NewVariable(tIn, false)
		outVar, err := csa.Forward(vVar)
		if err != nil {
			panic(err)
		}
		lossVar, err := autograd.Mean(outVar)
		if err != nil {
			panic(err)
		}
		val, _ := lossVar.Data.Float64At(0)
		return val
	}

	cfgCheck := gradcheck.Config{
		Epsilon:           1e-5,
		AbsoluteTolerance: 1e-4,
		RelativeTolerance: 1e-4,
	}
	if err := gradcheck.CheckSlice("CausalSelfAttention", "InputX", vals, analyticGrads, evalFn, cfgCheck); err != nil {
		t.Fatalf("GradCheck failed on input X: %v", err)
	}

	// Also check parameter gradients: Wq, Wk, Wv, Wo, Bq, Bk, Bv, Bo
	params := []struct {
		name string
		p    *Parameter
	}{
		{"Wq", csa.Wq},
		{"Wk", csa.Wk},
		{"Wv", csa.Wv},
		{"Wo", csa.Wo},
		{"Bq", csa.Bq},
		{"Bk", csa.Bk},
		{"Bv", csa.Bv},
		{"Bo", csa.Bo},
	}

	for _, pt := range params {
		p := pt.p
		// Populate fresh analytical gradients
		csa.ZeroGrad()
		xVarRec, _ := autograd.NewVariable(xTensor, true)
		outRec, _ := csa.Forward(xVarRec)
		lossRec, _ := autograd.Mean(outRec)
		_ = lossRec.Backward()

		pVals := append([]float64(nil), p.values...)
		pGrads := append([]float64(nil), p.grads...)

		// For weights (Wq, Wk, Wv, Wo) and biases (Bq, Bv, Bo), ensure analytical gradients are non-zero.
		// Bk has mathematically 0 gradient for self-attention because adding a constant to all key tokens
		// produces an identical scalar shift across all attention logits in a row, which exactly cancels in softmax.
		if pt.name != "Bk" {
			var hasNonZero bool
			for _, g := range p.grads {
				if math.Abs(g) > 1e-12 {
					hasNonZero = true
					break
				}
			}
			if !hasNonZero {
				t.Fatalf("parameter %s has zero gradient after Backward", pt.name)
			}
		}

		paramEvalFn := func(v []float64) float64 {
			copy(p.values, v)
			csa.ZeroGrad()
			vVar, _ := autograd.NewVariable(xTensor, false)
			outVar, err := csa.Forward(vVar)
			if err != nil {
				panic(err)
			}
			lossVar, err := autograd.Mean(outVar)
			if err != nil {
				panic(err)
			}
			val, _ := lossVar.Data.Float64At(0)
			return val
		}

		if err := gradcheck.CheckSlice("CausalSelfAttention", pt.name, pVals, pGrads, paramEvalFn, cfgCheck); err != nil {
			t.Fatalf("GradCheck failed on param %s: %v", pt.name, err)
		}
		copy(p.values, pVals)
	}
}
