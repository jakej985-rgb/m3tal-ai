//go:build cuda

package kernel

import "m3tal-llm/internal/gpu"

// CUDA dispatch wrappers. Each returns true when the operation completed on
// the GPU; any CUDA error records gpu.LastError() and returns false so the
// caller falls back to the CPU path.

func cudaMatMulFloat32(M, N, K int, A, B, C []float32) bool {
	return gpu.MatMulFloat32(M, N, K, A, B, C) == nil
}

func cudaMatMulFloat64(M, N, K int, A, B, C []float64) bool {
	return gpu.MatMulFloat64(M, N, K, A, B, C) == nil
}

func cudaAddFloat32(dst, a, b []float32) bool {
	return gpu.AddFloat32(a, b, dst) == nil
}

func cudaSubFloat32(dst, a, b []float32) bool {
	return gpu.SubFloat32(a, b, dst) == nil
}

func cudaMulFloat32(dst, a, b []float32) bool {
	return gpu.MulFloat32(a, b, dst) == nil
}

func cudaDivFloat32(dst, a, b []float32) bool {
	return gpu.DivFloat32(a, b, dst) == nil
}

func cudaAddFloat64(dst, a, b []float64) bool {
	return gpu.AddFloat64(a, b, dst) == nil
}

func cudaSubFloat64(dst, a, b []float64) bool {
	return gpu.SubFloat64(a, b, dst) == nil
}

func cudaMulFloat64(dst, a, b []float64) bool {
	return gpu.MulFloat64(a, b, dst) == nil
}

func cudaDivFloat64(dst, a, b []float64) bool {
	return gpu.DivFloat64(a, b, dst) == nil
}

func cudaScaleFloat32(dst, src []float32, alpha float32) bool {
	return gpu.ScaleFloat32(src, alpha, dst) == nil
}

func cudaScaleFloat64(dst, src []float64, alpha float64) bool {
	return gpu.ScaleFloat64(src, alpha, dst) == nil
}

func cudaSumFloat32(src []float32) (float32, bool) {
	v, err := gpu.SumFloat32(src)
	return v, err == nil
}

func cudaSumFloat64(src []float64) (float64, bool) {
	v, err := gpu.SumFloat64(src)
	return v, err == nil
}
