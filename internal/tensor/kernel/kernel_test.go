package kernel

import (
	"math"
	"math/rand"
	"testing"
)

func TestMatMulFloat32BlockedAndParallelAgainstReference(t *testing.T) {
	testSizes := [][3]int{
		{0, 0, 0},
		{1, 1, 1},
		{1, 5, 1},
		{5, 1, 7},
		{16, 16, 16},
		{32, 32, 32},
		{33, 31, 35}, // Non-multiples of tile sizes
		{64, 64, 64},
		{128, 64, 96},
	}

	rng := rand.New(rand.NewSource(42))

	for _, size := range testSizes {
		M, N, K := size[0], size[1], size[2]
		A := make([]float32, M*K)
		B := make([]float32, K*N)
		for i := range A {
			A[i] = rng.Float32()*2 - 1
		}
		for i := range B {
			B[i] = rng.Float32()*2 - 1
		}

		cRef := make([]float32, M*N)
		cBlocked := make([]float32, M*N)
		cParallel := make([]float32, M*N)

		MatMulFloat32Reference(M, N, K, A, B, cRef)
		MatMulFloat32Blocked(M, N, K, A, B, cBlocked)
		MatMulFloat32Parallel(M, N, K, A, B, cParallel)

		for i := 0; i < M*N; i++ {
			diffBlocked := float64(cBlocked[i] - cRef[i])
			if math.Abs(diffBlocked) > 1e-4 {
				t.Fatalf("MatMulFloat32Blocked size %vx%vx%v mismatch at %d: got %f, want %f",
					M, N, K, i, cBlocked[i], cRef[i])
			}
			diffParallel := float64(cParallel[i] - cRef[i])
			if math.Abs(diffParallel) > 1e-4 {
				t.Fatalf("MatMulFloat32Parallel size %vx%vx%v mismatch at %d: got %f, want %f",
					M, N, K, i, cParallel[i], cRef[i])
			}
		}
	}
}

func TestMatMulFloat64BlockedAndParallelAgainstReference(t *testing.T) {
	testSizes := [][3]int{
		{0, 0, 0},
		{1, 1, 1},
		{15, 23, 17},
		{32, 32, 32},
		{65, 33, 49},
		{100, 50, 75},
	}

	rng := rand.New(rand.NewSource(1337))

	for _, size := range testSizes {
		M, N, K := size[0], size[1], size[2]
		A := make([]float64, M*K)
		B := make([]float64, K*N)
		for i := range A {
			A[i] = rng.Float64()*2 - 1
		}
		for i := range B {
			B[i] = rng.Float64()*2 - 1
		}

		cRef := make([]float64, M*N)
		cBlocked := make([]float64, M*N)
		cParallel := make([]float64, M*N)

		MatMulFloat64Reference(M, N, K, A, B, cRef)
		MatMulFloat64Blocked(M, N, K, A, B, cBlocked)
		MatMulFloat64Parallel(M, N, K, A, B, cParallel)

		for i := 0; i < M*N; i++ {
			diffBlocked := cBlocked[i] - cRef[i]
			if math.Abs(diffBlocked) > 1e-9 {
				t.Fatalf("MatMulFloat64Blocked size %vx%vx%v mismatch at %d: got %f, want %f",
					M, N, K, i, cBlocked[i], cRef[i])
			}
			diffParallel := cParallel[i] - cRef[i]
			if math.Abs(diffParallel) > 1e-9 {
				t.Fatalf("MatMulFloat64Parallel size %vx%vx%v mismatch at %d: got %f, want %f",
					M, N, K, i, cParallel[i], cRef[i])
			}
		}
	}
}

func TestFallbackElementwiseAndReductions(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	dst := make([]float32, 4)

	AddFloat32Fallback(dst, a, b)
	if dst[0] != 6 || dst[3] != 12 {
		t.Fatalf("AddFloat32Fallback failed: %v", dst)
	}

	SubFloat32Fallback(dst, b, a)
	if dst[0] != 4 || dst[3] != 4 {
		t.Fatalf("SubFloat32Fallback failed: %v", dst)
	}

	MulFloat32Fallback(dst, a, b)
	if dst[0] != 5 || dst[3] != 32 {
		t.Fatalf("MulFloat32Fallback failed: %v", dst)
	}

	DivFloat32Fallback(dst, a, b)
	if dst[0] != 0.2 || math.Abs(float64(dst[3]-0.5)) > 1e-6 {
		t.Fatalf("DivFloat32Fallback failed: %v", dst)
	}

	ScaleFloat32Fallback(dst, a, 2.5)
	if dst[0] != 2.5 || dst[3] != 10 {
		t.Fatalf("ScaleFloat32Fallback failed: %v", dst)
	}

	if sum := SumFloat32Fallback(a); sum != 10 {
		t.Fatalf("SumFloat32Fallback failed: got %f, want 10", sum)
	}

	if max := MaxFloat32Fallback(a); max != 4 {
		t.Fatalf("MaxFloat32Fallback failed: got %f, want 4", max)
	}

	if dot := DotFloat32Fallback(a, b); dot != 70 {
		t.Fatalf("DotFloat32Fallback failed: got %f, want 70", dot)
	}
}
