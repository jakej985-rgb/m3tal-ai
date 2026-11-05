//go:build cuda && cgo

package gpu

import (
	"math"
	"math/rand"
	"testing"
)

// requireGPU skips the test unless a CUDA device is actually usable.
func requireGPU(t *testing.T) {
	t.Helper()
	b, err := AllocF32(1)
	if err != nil {
		t.Skipf("no usable CUDA device: %v", err)
	}
	b.Free()
}

func randF32(n int, rng *rand.Rand) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = rng.Float32()*2 - 1
	}
	return out
}

func randF64(n int, rng *rand.Rand) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = rng.Float64()*2 - 1
	}
	return out
}

func refMatMulF32(M, N, K int, A, B, C []float32) {
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var s float32
			for k := 0; k < K; k++ {
				s += A[i*K+k] * B[k*N+j]
			}
			C[i*N+j] = s
		}
	}
}

func refMatMulF64(M, N, K int, A, B, C []float64) {
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var s float64
			for k := 0; k < K; k++ {
				s += A[i*K+k] * B[k*N+j]
			}
			C[i*N+j] = s
		}
	}
}

func assertF32Near(t *testing.T, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d want %d", len(got), len(want))
	}
	for i := range got {
		diff := float32(math.Abs(float64(got[i] - want[i])))
		scale := float32(math.Max(1, math.Max(float64(math.Abs(float64(want[i]))), float64(math.Abs(float64(got[i]))))))
		if diff > tol*scale {
			t.Fatalf("element %d: got %v want %v (diff %v > tol %v)", i, got[i], want[i], diff, tol*scale)
		}
	}
}

func assertF64Near(t *testing.T, got, want []float64, tol float64) {
	t.Helper()
	for i := range got {
		diff := math.Abs(got[i] - want[i])
		scale := math.Max(1, math.Max(math.Abs(want[i]), math.Abs(got[i])))
		if diff > tol*scale {
			t.Fatalf("element %d: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestMatMulFloat32Parity(t *testing.T) {
	requireGPU(t)
	rng := rand.New(rand.NewSource(1))
	for _, dims := range [][3]int{{1, 1, 1}, {4, 4, 4}, {33, 65, 17}, {128, 96, 64}, {200, 200, 200}} {
		M, N, K := dims[0], dims[1], dims[2]
		A, B := randF32(M*K, rng), randF32(K*N, rng)
		got := make([]float32, M*N)
		want := make([]float32, M*N)
		if err := MatMulFloat32(M, N, K, A, B, got); err != nil {
			t.Fatalf("MatMulFloat32(%dx%dx%d): %v", M, N, K, err)
		}
		refMatMulF32(M, N, K, A, B, want)
		assertF32Near(t, got, want, 1e-4)
	}
}

func TestMatMulFloat64Parity(t *testing.T) {
	requireGPU(t)
	rng := rand.New(rand.NewSource(2))
	M, N, K := 64, 48, 40
	A, B := randF64(M*K, rng), randF64(K*N, rng)
	got := make([]float64, M*N)
	want := make([]float64, M*N)
	if err := MatMulFloat64(M, N, K, A, B, got); err != nil {
		t.Fatalf("MatMulFloat64: %v", err)
	}
	refMatMulF64(M, N, K, A, B, want)
	assertF64Near(t, got, want, 1e-10)
}

func TestElemwiseF32Parity(t *testing.T) {
	requireGPU(t)
	rng := rand.New(rand.NewSource(3))
	n := 1000
	a, b := randF32(n, rng), randF32(n, rng)
	b[7] = 0 // div by zero element -> Inf on both paths
	ops := []struct {
		name string
		fn   func(a, b, out []float32) error
		ref  func(x, y float32) float32
	}{
		{"add", AddFloat32, func(x, y float32) float32 { return x + y }},
		{"sub", SubFloat32, func(x, y float32) float32 { return x - y }},
		{"mul", MulFloat32, func(x, y float32) float32 { return x * y }},
		{"div", DivFloat32, func(x, y float32) float32 { return x / y }},
	}
	for _, op := range ops {
		got := make([]float32, n)
		want := make([]float32, n)
		if err := op.fn(a, b, got); err != nil {
			t.Fatalf("%s: %v", op.name, err)
		}
		for i := range want {
			want[i] = op.ref(a[i], b[i])
		}
		assertF32Near(t, got, want, 1e-6)
	}
}

func TestElemwiseF64Parity(t *testing.T) {
	requireGPU(t)
	rng := rand.New(rand.NewSource(4))
	n := 500
	a, b := randF64(n, rng), randF64(n, rng)
	got := make([]float64, n)
	if err := AddFloat64(a, b, got); err != nil {
		t.Fatalf("AddFloat64: %v", err)
	}
	for i := range got {
		if math.Abs(got[i]-(a[i]+b[i])) > 1e-12 {
			t.Fatalf("element %d mismatch", i)
		}
	}
}

func TestScaleF32Parity(t *testing.T) {
	requireGPU(t)
	rng := rand.New(rand.NewSource(5))
	src := randF32(2048, rng)
	got := make([]float32, len(src))
	if err := ScaleFloat32(src, 1.5, got); err != nil {
		t.Fatalf("ScaleFloat32: %v", err)
	}
	for i, v := range got {
		if math.Abs(float64(v-src[i]*1.5)) > 1e-6 {
			t.Fatalf("element %d: got %v want %v", i, v, src[i]*1.5)
		}
	}
}

func TestSumMaxParity(t *testing.T) {
	requireGPU(t)
	rng := rand.New(rand.NewSource(6))
	src := randF32(10007, rng)
	src[123] = -42.5
	src[4567] = 99.25 // max

	sum, err := SumFloat32(src)
	if err != nil {
		t.Fatalf("SumFloat32: %v", err)
	}
	var wantSum float64
	wantMax := float32(math.Inf(-1))
	for _, v := range src {
		wantSum += float64(v)
		if v > wantMax {
			wantMax = v
		}
	}
	if math.Abs(float64(sum)-wantSum)/math.Max(1, math.Abs(wantSum)) > 1e-4 {
		t.Fatalf("sum: got %v want %v", sum, wantSum)
	}

	max, err := MaxFloat32(src)
	if err != nil {
		t.Fatalf("MaxFloat32: %v", err)
	}
	if max != wantMax {
		t.Fatalf("max: got %v want %v", max, wantMax)
	}
}

func TestSumMaxF64Parity(t *testing.T) {
	requireGPU(t)
	rng := rand.New(rand.NewSource(7))
	src := randF64(5000, rng)
	src[99] = 1234.5

	sum, err := SumFloat64(src)
	if err != nil {
		t.Fatalf("SumFloat64: %v", err)
	}
	var wantSum float64
	wantMax := math.Inf(-1)
	for _, v := range src {
		wantSum += v
		if v > wantMax {
			wantMax = v
		}
	}
	if math.Abs(sum-wantSum)/math.Max(1, math.Abs(wantSum)) > 1e-10 {
		t.Fatalf("sum: got %v want %v", sum, wantSum)
	}
	max, err := MaxFloat64(src)
	if err != nil {
		t.Fatalf("MaxFloat64: %v", err)
	}
	if max != wantMax {
		t.Fatalf("max: got %v want %v", max, wantMax)
	}
}

// BenchmarkMatMulFloat32CUDA measures the full host->device->kernel->host
// path, i.e. end-to-end cost including PCIe transfers.
func BenchmarkMatMulFloat32CUDA(b *testing.B) {
	if _, err := AllocF32(1); err != nil {
		b.Skipf("no usable CUDA device: %v", err)
	}
	rng := rand.New(rand.NewSource(8))
	M, N, K := 512, 512, 512
	A, B := randF32(M*K, rng), randF32(K*N, rng)
	C := make([]float32, M*N)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := MatMulFloat32(M, N, K, A, B, C); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMulFloat32CPU(b *testing.B) {
	rng := rand.New(rand.NewSource(8))
	M, N, K := 512, 512, 512
	A, B := randF32(M*K, rng), randF32(K*N, rng)
	C := make([]float32, M*N)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		refMatMulF32(M, N, K, A, B, C)
	}
}
