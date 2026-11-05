package lm

import (
	"math"
	"strings"
	"testing"
)

// ── RMSNorm unit tests ──────────────────────────────────────────────────────

// TestRMSNormOutputHasUnitRMS verifies that rmsNorm with gamma=1 produces a
// row whose root-mean-square equals 1 (up to epsilon rounding).
func TestRMSNormOutputHasUnitRMS(t *testing.T) {
	x := []float32{3, 4, 0, -1} // some non-trivial vector
	gamma := []float32{1, 1, 1, 1}
	out, _ := rmsNorm(x, gamma, 1, 4)

	var sumSq float32
	for _, v := range out {
		sumSq += v * v
	}
	got := math.Sqrt(float64(sumSq / 4))
	if math.Abs(got-1) > 1e-5 {
		t.Fatalf("output RMS = %.8f, want ≈ 1.0", got)
	}
}

// TestRMSNormGammaScaling verifies that gamma acts as an element-wise scale
// on the normalised vector: out_i = gamma_i * x_i / rms.
func TestRMSNormGammaScaling(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	gamma := []float32{2, 0.5, 1, 3}
	out, rmsVals := rmsNorm(x, gamma, 1, 4)
	r := rmsVals[0]
	for d := 0; d < 4; d++ {
		want := gamma[d] * x[d] / r
		if math.Abs(float64(out[d]-want)) > 1e-6 {
			t.Fatalf("out[%d] = %g, want %g", d, out[d], want)
		}
	}
}

// TestRMSNormBackwardMatchesFiniteDifferences is a component-level gradient
// check on rmsNormBackward, isolated from the full model.
func TestRMSNormBackwardMatchesFiniteDifferences(t *testing.T) {
	rows, width := 3, 4
	x := []float32{1, -2, 3, 0.5, -1, 2, 0.3, 1.1, 0.7, -0.4, 1.5, 2.2}
	gamma := []float32{0.9, 1.1, 0.8, 1.2}

	// Forward once to get rmsVals.
	_, rmsVals := rmsNorm(x, gamma, rows, width)

	// Simple loss = sum(out); dL/d_out = all 1.
	gradOut := make([]float32, rows*width)
	for i := range gradOut {
		gradOut[i] = 1.0
	}

	gammaGrad := make([]float32, width)
	gradX := rmsNormBackward(gradOut, x, rmsVals, gamma, rows, width, gammaGrad)

	const eps = float32(1e-3)
	sumOut := func(gm, xv []float32) float64 {
		out, _ := rmsNorm(xv, gm, rows, width)
		var s float64
		for _, v := range out {
			s += float64(v)
		}
		return s
	}

	// Check gradX via finite differences.
	for i := 0; i < rows*width; i++ {
		orig := x[i]
		x[i] = orig + eps
		lossPlus := sumOut(gamma, x)
		x[i] = orig - eps
		lossMinus := sumOut(gamma, x)
		x[i] = orig
		numerical := float32((lossPlus - lossMinus) / (2 * float64(eps)))
		diff := math.Abs(float64(gradX[i] - numerical))
		scale := math.Max(1, math.Max(math.Abs(float64(gradX[i])), math.Abs(float64(numerical))))
		if diff > 5e-2 && diff/scale > 5e-2 {
			t.Fatalf("gradX[%d] analytic=%g numerical=%g diff=%g", i, gradX[i], numerical, diff)
		}
	}

	// Check gammaGrad via finite differences.
	for i := 0; i < width; i++ {
		orig := gamma[i]
		gamma[i] = orig + eps
		lossPlus := sumOut(gamma, x)
		gamma[i] = orig - eps
		lossMinus := sumOut(gamma, x)
		gamma[i] = orig
		numerical := float32((lossPlus - lossMinus) / (2 * float64(eps)))
		diff := math.Abs(float64(gammaGrad[i] - numerical))
		scale := math.Max(1, math.Max(math.Abs(float64(gammaGrad[i])), math.Abs(float64(numerical))))
		if diff > 5e-2 && diff/scale > 5e-2 {
			t.Fatalf("gammaGrad[%d] analytic=%g numerical=%g diff=%g", i, gammaGrad[i], numerical, diff)
		}
	}
}

// ── RoPE unit tests ─────────────────────────────────────────────────────────

// TestRoPEPositionZeroIsIdentity confirms that at sequence position 0,
// cos(0)=1 and sin(0)=0 so the rotation leaves the vectors unchanged.
func TestRoPEPositionZeroIsIdentity(t *testing.T) {
	q := []float32{1, 2, 3, 4}
	k := []float32{5, 6, 7, 8}
	qOrig := append([]float32(nil), q...)
	kOrig := append([]float32(nil), k...)
	applyRoPE(q, k, 1, 4, 4, 1) // 1 row (pos=0), width=4, headDim=4, 1 head
	for d := 0; d < 4; d++ {
		if math.Abs(float64(q[d]-qOrig[d])) > 1e-6 {
			t.Fatalf("RoPE at pos=0 changed q[%d]: got %g, want %g", d, q[d], qOrig[d])
		}
		if math.Abs(float64(k[d]-kOrig[d])) > 1e-6 {
			t.Fatalf("RoPE at pos=0 changed k[%d]: got %g, want %g", d, k[d], kOrig[d])
		}
	}
}

// TestRoPEInverseRoundTrip applies RoPE then its inverse and expects recovery
// of the original vectors to floating-point precision.
func TestRoPEInverseRoundTrip(t *testing.T) {
	// Two rows (positions 0 and 1), width=4, headDim=4, 1 head.
	q := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	k := []float32{9, 10, 11, 12, -1, -2, -3, -4}
	qOrig := append([]float32(nil), q...)
	kOrig := append([]float32(nil), k...)
	applyRoPE(q, k, 2, 4, 4, 1)
	applyInverseRoPE(q, k, 2, 4, 4, 1)
	for d := 0; d < 8; d++ {
		if math.Abs(float64(q[d]-qOrig[d])) > 1e-5 {
			t.Fatalf("RoPE inverse failed for q[%d]: got %g, want %g", d, q[d], qOrig[d])
		}
		if math.Abs(float64(k[d]-kOrig[d])) > 1e-5 {
			t.Fatalf("RoPE inverse failed for k[%d]: got %g, want %g", d, k[d], kOrig[d])
		}
	}
}

// TestRoPEPreservesNorm verifies that RoPE is an isometry: ‖q‖ is unchanged.
func TestRoPEPreservesNorm(t *testing.T) {
	q := []float32{3, 4, 1, 2}
	k := []float32{1, 0, 2, 1}
	norm := func(v []float32) float64 {
		var s float64
		for _, x := range v {
			s += float64(x * x)
		}
		return math.Sqrt(s)
	}
	normQBefore := norm(q)
	normKBefore := norm(k)
	applyRoPE(q, k, 1, 4, 4, 1)
	if math.Abs(norm(q)-normQBefore) > 1e-5 {
		t.Fatalf("RoPE changed ‖q‖: before %g after %g", normQBefore, norm(q))
	}
	if math.Abs(norm(k)-normKBefore) > 1e-5 {
		t.Fatalf("RoPE changed ‖k‖: before %g after %g", normKBefore, norm(k))
	}
}

// ── End-to-end model tests ──────────────────────────────────────────────────

// TestModelWithRMSNormAndRoPETrainingLowersLoss is the core regression guard:
// a model with the new architecture must still be able to overfit a small corpus.
func TestModelWithRMSNormAndRoPETrainingLowersLoss(t *testing.T) {
	text := strings.Repeat("hello world. m3tal learns fast. ", 12)
	tokenizer, err := TrainTokenizer([]string{text}, 256)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(Config{Context: 6, Width: 4}, tokenizer, 23)
	if err != nil {
		t.Fatal(err)
	}
	before, err := model.Evaluate([]string{text})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Train([]string{text}, 8, 20, 0.003); err != nil {
		t.Fatal(err)
	}
	after, err := model.Evaluate([]string{text})
	if err != nil {
		t.Fatal(err)
	}
	if !(after < before) {
		t.Fatalf("training did not lower loss: before %.4f after %.4f", before, after)
	}
}

// TestCheckpointRoundTripWithRoPEAndRMSNorm verifies that Save→Load→Generate
// produces bit-exact output for a multi-layer model with RMSNorm + RoPE.
func TestCheckpointRoundTripWithRoPEAndRMSNorm(t *testing.T) {
	text := "the quick brown fox jumps over the lazy dog. café 東京 🤖 "
	tokenizer, err := TrainTokenizer([]string{text + text}, 280)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(Config{Context: 8, Width: 4, Layers: 2, Heads: 2}, tokenizer, 13)
	if err != nil {
		t.Fatal(err)
	}
	before, err := model.Generate("the quick", 12, 0.8, 7)
	if err != nil {
		t.Fatal(err)
	}
	path := t.TempDir() + "/model_rope_rmsnorm.json"
	if err := model.SaveCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	after, err := loaded.Generate("the quick", 12, 0.8, 7)
	if err != nil {
		t.Fatal(err)
	}
	if before != after {
		t.Fatalf("generation changed after checkpoint round trip:\n  before: %q\n  after:  %q", before, after)
	}
}

// TestNormGammaGradientsMatchFiniteDifferences checks that the normAttn and
// normFF gamma gradients are correct end-to-end by running the same finite-
// difference sweep used for QKV weights — it covers both RMSNorm backward paths.
func TestNormGammaGradientsMatchFiniteDifferences(t *testing.T) {
	tokenizer, err := TrainTokenizer([]string{"rmsnorm rope gradient check café"}, 256)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(Config{Context: 6, Width: 4, Layers: 2, Heads: 2}, tokenizer, 61)
	if err != nil {
		t.Fatal(err)
	}
	ids := tokenizer.Encode("norm")
	if len(ids) < 3 {
		t.Fatal("expected prompt to encode to at least three tokens")
	}
	input := ids[:len(ids)-1]
	targets := ids[1:]
	zeroGradients(model.parameters)
	model.backward(model.forward(input), targets, nil)

	const epsilon float32 = 1e-3
	for paramIdx, param := range model.parameters {
		for i, analytic := range param.gradient {
			original := param.values[i]
			param.values[i] = original + epsilon
			plus := meanWindowLoss(model.forward(input), targets, nil)
			param.values[i] = original - epsilon
			minus := meanWindowLoss(model.forward(input), targets, nil)
			param.values[i] = original
			numerical := (plus - minus) / (2 * float64(epsilon))
			diff := math.Abs(float64(analytic) - numerical)
			scale := math.Max(1, math.Max(math.Abs(float64(analytic)), math.Abs(numerical)))
			if diff > 5e-2 && diff/scale > 5e-2 {
				t.Fatalf("param %d[%d] analytic=%g numerical=%g diff=%g",
					paramIdx, i, analytic, numerical, diff)
			}
		}
	}
}

// TestTokenizerSupports1024Merges verifies that TrainTokenizer accepts the new
// 1024-merge target without error.
func TestTokenizerSupports1024Merges(t *testing.T) {
	text := strings.Repeat("container restart traefik environment jellyfin m3tal. ", 60)
	tok, err := TrainTokenizer([]string{text}, 1024)
	if err != nil {
		t.Fatalf("TrainTokenizer(1024) failed: %v", err)
	}
	if len(tok.Tokens) < 256 {
		t.Fatalf("expected at least 256 tokens, got %d", len(tok.Tokens))
	}
	if len(tok.Tokens) > 1024 {
		t.Fatalf("expected at most 1024 tokens, got %d", len(tok.Tokens))
	}
	t.Logf("trained tokenizer with %d tokens (%d merges)", len(tok.Tokens), len(tok.Merges))
}
