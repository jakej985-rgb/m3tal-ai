//go:build cuda && cgo

package kernel

import (
	"math"
	"math/rand"
	"testing"
)

// cudaSelected selects the CUDA backend for the duration of a test and
// skips when no usable device exists.
func cudaSelected(t *testing.T) {
	t.Helper()
	if !CUDACompiled() {
		t.Skip("binary not built with -tags cuda")
	}
	if err := Select("cuda"); err != nil {
		t.Skipf("cuda backend unavailable: %v", err)
	}
	t.Cleanup(func() { _ = Select("cpu") })
}

func TestDispatchMatMulFloat32CUDAParity(t *testing.T) {
	cudaSelected(t)
	rng := rand.New(rand.NewSource(11))
	M, N, K := 96, 80, 64
	A := make([]float32, M*K)
	B := make([]float32, K*N)
	for i := range A {
		A[i] = rng.Float32()*2 - 1
	}
	for i := range B {
		B[i] = rng.Float32()*2 - 1
	}
	got := make([]float32, M*N)
	want := make([]float32, M*N)
	MatMulFloat32Dispatch(M, N, K, A, B, got)
	MatMulFloat32Reference(M, N, K, A, B, want)
	for i := range got {
		diff := math.Abs(float64(got[i] - want[i]))
		scale := math.Max(1, math.Max(math.Abs(float64(want[i])), math.Abs(float64(got[i]))))
		if diff > 1e-4*scale {
			t.Fatalf("element %d: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestDispatchElementwiseCUDAParity(t *testing.T) {
	cudaSelected(t)
	rng := rand.New(rand.NewSource(12))
	n := 4096
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = rng.Float32()*2 - 1
		b[i] = rng.Float32()*2 - 1
	}
	got := make([]float32, n)
	AddFloat32Dispatch(got, a, b)
	for i := range got {
		if math.Abs(float64(got[i]-(a[i]+b[i]))) > 1e-6 {
			t.Fatalf("add element %d: got %v want %v", i, got[i], a[i]+b[i])
		}
	}
	MulFloat32Dispatch(got, a, b)
	for i := range got {
		if math.Abs(float64(got[i]-a[i]*b[i])) > 1e-6 {
			t.Fatalf("mul element %d: got %v want %v", i, got[i], a[i]*b[i])
		}
	}
	// Sub/Div have no SIMD path; CUDA must still produce correct results.
	SubFloat32Dispatch(got, a, b)
	for i := range got {
		if math.Abs(float64(got[i]-(a[i]-b[i]))) > 1e-6 {
			t.Fatalf("sub element %d: got %v want %v", i, got[i], a[i]-b[i])
		}
	}
}

func TestDispatchSumCUDAParity(t *testing.T) {
	cudaSelected(t)
	rng := rand.New(rand.NewSource(13))
	src := make([]float32, 20011)
	for i := range src {
		src[i] = rng.Float32()
	}
	got := SumFloat32Dispatch(src)
	want := SumFloat32Fallback(src)
	if math.Abs(float64(got-want))/math.Max(1, math.Abs(float64(want))) > 1e-4 {
		t.Fatalf("sum: got %v want %v", got, want)
	}
}
