package kernel

import (
	"m3tal-llm/internal/tensor/cpu"
)

// Dispatch wrappers that route to CUDA (when selected and compiled in),
// then SIMD when available, otherwise parallel cache-blocked / fallback.
// Any CUDA failure falls through to the CPU path.

func MatMulFloat32Dispatch(M, N, K int, A, B, C []float32) {
	if preferCUDA() && cudaMatMulFloat32(M, N, K, A, B, C) {
		return
	}
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
	if preferCUDA() && cudaMatMulFloat64(M, N, K, A, B, C) {
		return
	}
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
	if preferCUDA() && cudaAddFloat32(dst, a, b) {
		return
	}
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
	if preferCUDA() && cudaSubFloat32(dst, a, b) {
		return
	}
	SubFloat32Fallback(dst, a, b)
}

func MulFloat32Dispatch(dst, a, b []float32) {
	if preferCUDA() && cudaMulFloat32(dst, a, b) {
		return
	}
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
	if preferCUDA() && cudaDivFloat32(dst, a, b) {
		return
	}
	DivFloat32Fallback(dst, a, b)
}

func AddFloat64Dispatch(dst, a, b []float64) {
	if preferCUDA() && cudaAddFloat64(dst, a, b) {
		return
	}
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
	if preferCUDA() && cudaSubFloat64(dst, a, b) {
		return
	}
	SubFloat64Fallback(dst, a, b)
}

func MulFloat64Dispatch(dst, a, b []float64) {
	if preferCUDA() && cudaMulFloat64(dst, a, b) {
		return
	}
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
	if preferCUDA() && cudaDivFloat64(dst, a, b) {
		return
	}
	DivFloat64Fallback(dst, a, b)
}

func ScaleFloat32Dispatch(dst, src []float32, alpha float32) {
	if preferCUDA() && cudaScaleFloat32(dst, src, alpha) {
		return
	}
	ScaleFloat32Fallback(dst, src, alpha)
}

func ScaleFloat64Dispatch(dst, src []float64, alpha float64) {
	if preferCUDA() && cudaScaleFloat64(dst, src, alpha) {
		return
	}
	ScaleFloat64Fallback(dst, src, alpha)
}

func SumFloat32Dispatch(src []float32) float32 {
	if preferCUDA() {
		if v, ok := cudaSumFloat32(src); ok {
			return v
		}
	}
	return SumFloat32Fallback(src)
}

func SumFloat64Dispatch(src []float64) float64 {
	if preferCUDA() {
		if v, ok := cudaSumFloat64(src); ok {
			return v
		}
	}
	return SumFloat64Fallback(src)
}
