package kernel

import (
	"math"
)

// Elementwise binary operations (float32)

func AddFloat32Fallback(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func SubFloat32Fallback(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

func MulFloat32Fallback(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func DivFloat32Fallback(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

// Elementwise binary operations (float64)

func AddFloat64Fallback(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func SubFloat64Fallback(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

func MulFloat64Fallback(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func DivFloat64Fallback(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

// Elementwise scalar operations

func ScaleFloat32Fallback(dst, src []float32, alpha float32) {
	for i := range dst {
		dst[i] = src[i] * alpha
	}
}

func ScaleFloat64Fallback(dst, src []float64, alpha float64) {
	for i := range dst {
		dst[i] = src[i] * alpha
	}
}

// Reductions

func SumFloat32Fallback(src []float32) float32 {
	var sum float32
	for _, v := range src {
		sum += v
	}
	return sum
}

func SumFloat64Fallback(src []float64) float64 {
	var sum float64
	for _, v := range src {
		sum += v
	}
	return sum
}

func MaxFloat32Fallback(src []float32) float32 {
	if len(src) == 0 {
		return 0
	}
	m := float32(math.Inf(-1))
	for _, v := range src {
		if v > m {
			m = v
		}
	}
	return m
}

func MaxFloat64Fallback(src []float64) float64 {
	if len(src) == 0 {
		return 0
	}
	m := math.Inf(-1)
	for _, v := range src {
		if v > m {
			m = v
		}
	}
	return m
}

// Dot products (inner loops)

func DotFloat32Fallback(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func DotFloat64Fallback(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// Naive reference GEMM: C = A * B where A is [M, K], B is [K, N], C is [M, N]
func MatMulFloat32Reference(M, N, K int, A, B, C []float32) {
	for i := 0; i < M*N; i++ {
		C[i] = 0
	}
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float32
			for k := 0; k < K; k++ {
				sum += A[i*K+k] * B[k*N+j]
			}
			C[i*N+j] = sum
		}
	}
}

// Naive reference GEMM: C = A * B where A is [M, K], B is [K, N], C is [M, N]
func MatMulFloat64Reference(M, N, K int, A, B, C []float64) {
	for i := 0; i < M*N; i++ {
		C[i] = 0
	}
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float64
			for k := 0; k < K; k++ {
				sum += A[i*K+k] * B[k*N+j]
			}
			C[i*N+j] = sum
		}
	}
}
