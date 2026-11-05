package kernel

import (
	"m3tal-ai/internal/tensor/cpu"
)

// Dispatch wrappers that route to SIMD when available, otherwise parallel cache-blocked / fallback.

func MatMulFloat32Dispatch(M, N, K int, A, B, C []float32) {
	if cpu.UseAVX2() {
		MatMulFloat32AVX2(M, N, K, A, B, C)
		return
	}
	if cpu.UseNEON() {
		MatMulFloat32NEON(M, N, K, A, B, C)
		return
	}
	MatMulFloat32Parallel(M, N, K, A, B, C)
}

func MatMulFloat64Dispatch(M, N, K int, A, B, C []float64) {
	if cpu.UseAVX2() {
		MatMulFloat64AVX2(M, N, K, A, B, C)
		return
	}
	if cpu.UseNEON() {
		MatMulFloat64NEON(M, N, K, A, B, C)
		return
	}
	MatMulFloat64Parallel(M, N, K, A, B, C)
}

func AddFloat32Dispatch(dst, a, b []float32) {
	if cpu.UseAVX2() {
		AddFloat32AVX2(dst, a, b)
		return
	}
	if cpu.UseNEON() {
		AddFloat32NEON(dst, a, b)
		return
	}
	AddFloat32Fallback(dst, a, b)
}

func SubFloat32Dispatch(dst, a, b []float32) {
	SubFloat32Fallback(dst, a, b)
}

func MulFloat32Dispatch(dst, a, b []float32) {
	if cpu.UseAVX2() {
		MulFloat32AVX2(dst, a, b)
		return
	}
	if cpu.UseNEON() {
		MulFloat32NEON(dst, a, b)
		return
	}
	MulFloat32Fallback(dst, a, b)
}

func DivFloat32Dispatch(dst, a, b []float32) {
	DivFloat32Fallback(dst, a, b)
}

func AddFloat64Dispatch(dst, a, b []float64) {
	if cpu.UseAVX2() {
		AddFloat64AVX2(dst, a, b)
		return
	}
	if cpu.UseNEON() {
		AddFloat64NEON(dst, a, b)
		return
	}
	AddFloat64Fallback(dst, a, b)
}

func SubFloat64Dispatch(dst, a, b []float64) {
	SubFloat64Fallback(dst, a, b)
}

func MulFloat64Dispatch(dst, a, b []float64) {
	if cpu.UseAVX2() {
		MulFloat64AVX2(dst, a, b)
		return
	}
	if cpu.UseNEON() {
		MulFloat64NEON(dst, a, b)
		return
	}
	MulFloat64Fallback(dst, a, b)
}

func DivFloat64Dispatch(dst, a, b []float64) {
	DivFloat64Fallback(dst, a, b)
}
