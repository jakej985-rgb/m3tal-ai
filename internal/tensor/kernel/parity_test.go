package kernel

import (
	"math"
	"math/rand"
	"testing"

	"m3tal-llm/internal/tensor/cpu"
)

func TestSIMDParityAgainstFallback(t *testing.T) {
	sizes := []int{0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 513}
	rng := rand.New(rand.NewSource(999))

	for _, n := range sizes {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := range a {
			a[i] = rng.Float32()*20 - 10
			b[i] = rng.Float32()*20 - 10
		}

		// Test Add
		dstFallback := make([]float32, n)
		dstSIMD := make([]float32, n)
		AddFloat32Fallback(dstFallback, a, b)

		cpu.SetForceFallback(false)
		AddFloat32Dispatch(dstSIMD, a, b)
		for i := 0; i < n; i++ {
			if dstFallback[i] != dstSIMD[i] {
				t.Fatalf("Add parity mismatch at size %d, index %d: fallback=%f, simd=%f", n, i, dstFallback[i], dstSIMD[i])
			}
		}

		// Test Mul
		MulFloat32Fallback(dstFallback, a, b)
		MulFloat32Dispatch(dstSIMD, a, b)
		for i := 0; i < n; i++ {
			if dstFallback[i] != dstSIMD[i] {
				t.Fatalf("Mul parity mismatch at size %d, index %d: fallback=%f, simd=%f", n, i, dstFallback[i], dstSIMD[i])
			}
		}
	}
}

func TestMatMulSIMDParityAgainstFallback(t *testing.T) {
	testShapes := [][3]int{
		{1, 1, 1},
		{2, 3, 4},
		{7, 9, 11},
		{16, 16, 16},
		{32, 32, 32},
		{33, 17, 25},
		{64, 64, 64},
		{75, 45, 93},
		{128, 64, 32},
	}

	rng := rand.New(rand.NewSource(12345))

	for _, s := range testShapes {
		M, N, K := s[0], s[1], s[2]
		A := make([]float32, M*K)
		B := make([]float32, K*N)
		for i := range A {
			A[i] = rng.Float32()*2 - 1
		}
		for i := range B {
			B[i] = rng.Float32()*2 - 1
		}

		cFallback := make([]float32, M*N)
		cSIMD := make([]float32, M*N)

		// Force fallback
		cpu.SetForceFallback(true)
		MatMulFloat32Dispatch(M, N, K, A, B, cFallback)

		// Dispatch (SIMD if available)
		cpu.SetForceFallback(false)
		MatMulFloat32Dispatch(M, N, K, A, B, cSIMD)

		for i := 0; i < M*N; i++ {
			diff := float64(cSIMD[i] - cFallback[i])
			// FMA uses higher precision internally before rounding, so allow small epsilon
			if math.Abs(diff) > 1e-4 {
				t.Fatalf("MatMul SIMD vs Fallback mismatch at shape %vx%vx%v index %d: simd=%f, fallback=%f, diff=%e",
					M, N, K, i, cSIMD[i], cFallback[i], diff)
			}
		}
	}
}
