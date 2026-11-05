package kernel

import (
	"fmt"
	"math/rand"
	"testing"

	"m3tal-ai/internal/tensor/cpu"
)

func benchmarkMatMulFloat32(b *testing.B, size int, forceFallback bool) {
	cpu.SetForceFallback(forceFallback)
	defer cpu.SetForceFallback(false)

	M, N, K := size, size, size
	A := make([]float32, M*K)
	B := make([]float32, K*N)
	C := make([]float32, M*N)

	rng := rand.New(rand.NewSource(42))
	for i := range A {
		A[i] = rng.Float32()
	}
	for i := range B {
		B[i] = rng.Float32()
	}

	b.ResetTimer()
	b.SetBytes(int64(M * N * K * 4 * 2)) // Approximate FLOPS/bytes

	for i := 0; i < b.N; i++ {
		MatMulFloat32Dispatch(M, N, K, A, B, C)
	}
}

func BenchmarkMatMulFloat32(b *testing.B) {
	sizes := []int{32, 64, 128, 256, 512}
	for _, sz := range sizes {
		b.Run(fmt.Sprintf("Fallback_%dx%d", sz, sz), func(b *testing.B) {
			benchmarkMatMulFloat32(b, sz, true)
		})
		b.Run(fmt.Sprintf("Optimized_%dx%d", sz, sz), func(b *testing.B) {
			benchmarkMatMulFloat32(b, sz, false)
		})
	}
}

func BenchmarkElementwiseAddFloat32(b *testing.B) {
	sizes := []int{1024, 65536, 1048576}
	for _, sz := range sizes {
		a := make([]float32, sz)
		c := make([]float32, sz)
		dst := make([]float32, sz)
		b.Run(fmt.Sprintf("Fallback_N%d", sz), func(b *testing.B) {
			b.SetBytes(int64(sz * 4 * 2))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				AddFloat32Fallback(dst, a, c)
			}
		})
		b.Run(fmt.Sprintf("Optimized_N%d", sz), func(b *testing.B) {
			b.SetBytes(int64(sz * 4 * 2))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				AddFloat32Dispatch(dst, a, c)
			}
		})
	}
}
