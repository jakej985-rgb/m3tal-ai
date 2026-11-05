//go:build !cuda

package kernel

// CUDA dispatch stubs: report every operation as not handled so callers
// take the CPU path. The real implementations live in cuda_ops.go.

func cudaMatMulFloat32(M, N, K int, A, B, C []float32) bool { return false }
func cudaMatMulFloat64(M, N, K int, A, B, C []float64) bool { return false }
func cudaAddFloat32(dst, a, b []float32) bool               { return false }
func cudaSubFloat32(dst, a, b []float32) bool               { return false }
func cudaMulFloat32(dst, a, b []float32) bool               { return false }
func cudaDivFloat32(dst, a, b []float32) bool               { return false }
func cudaAddFloat64(dst, a, b []float64) bool               { return false }
func cudaSubFloat64(dst, a, b []float64) bool               { return false }
func cudaMulFloat64(dst, a, b []float64) bool               { return false }
func cudaDivFloat64(dst, a, b []float64) bool               { return false }
func cudaScaleFloat32(dst, src []float32, alpha float32) bool {
	return false
}
func cudaScaleFloat64(dst, src []float64, alpha float64) bool {
	return false
}
func cudaSumFloat32(src []float32) (float32, bool) { return 0, false }
func cudaSumFloat64(src []float64) (float64, bool) { return 0, false }
