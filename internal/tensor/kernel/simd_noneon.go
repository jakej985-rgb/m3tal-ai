//go:build !arm64

package kernel

func MatMulFloat32NEON(M, N, K int, A, B, C []float32) {
	MatMulFloat32Parallel(M, N, K, A, B, C)
}

func MatMulFloat64NEON(M, N, K int, A, B, C []float64) {
	MatMulFloat64Parallel(M, N, K, A, B, C)
}

func AddFloat32NEON(dst, a, b []float32) {
	AddFloat32Fallback(dst, a, b)
}

func MulFloat32NEON(dst, a, b []float32) {
	MulFloat32Fallback(dst, a, b)
}

func AddFloat64NEON(dst, a, b []float64) {
	AddFloat64Fallback(dst, a, b)
}

func MulFloat64NEON(dst, a, b []float64) {
	MulFloat64Fallback(dst, a, b)
}
