//go:build !cuda || !cgo || !amd64

package gpu

// This stub keeps the package API available on builds without CUDA support
// (the default) and on non-amd64 architectures. Every operation reports
// ErrUnavailable so callers can fall back cleanly.

import (
	"errors"
	"fmt"
	"math"
	"unsafe"
)

// ErrUnavailable is returned when CUDA support is not compiled in or no
// usable device exists.
var ErrUnavailable = errors.New("CUDA backend unavailable: rebuild with CGO_ENABLED=1 go build -tags cuda (see docs/CUDA.md)")

// Buffer is an opaque device allocation placeholder for stub builds.
type Buffer struct {
	bytes int
	freed bool
}

func alloc(bytes int, what string) (*Buffer, error) {
	if bytes <= 0 {
		return &Buffer{}, nil
	}
	return nil, fmt.Errorf("%w: %s allocation of %d bytes", ErrUnavailable, what, bytes)
}

// AllocF32 allocates n float32 elements on the device.
func AllocF32(n int) (*Buffer, error) { return alloc(n*4, "float32") }

// AllocF64 allocates n float64 elements on the device.
func AllocF64(n int) (*Buffer, error) { return alloc(n*8, "float64") }

// AllocBytes allocates raw bytes on the device.
func AllocBytes(n int) (*Buffer, error) { return alloc(n, "bytes") }

// Free releases the device allocation. It is idempotent.
func (b *Buffer) Free() {
	if b != nil {
		b.freed = true
	}
}

// Bytes returns the allocation size in bytes.
func (b *Buffer) Bytes() int { return b.bytes }

func (b *Buffer) check(what string) error {
	return fmt.Errorf("%w: %s on stub buffer", ErrUnavailable, what)
}

// Upload copies a host float32 slice into the buffer.
func (b *Buffer) Upload(src []float32) error { return b.check("upload") }

// UploadF64 copies a host float64 slice into the buffer.
func (b *Buffer) UploadF64(src []float64) error { return b.check("upload") }

// UploadBytes copies raw host bytes into the buffer.
func (b *Buffer) UploadBytes(src []byte) error { return b.check("upload") }

// Download copies the buffer contents into a host float32 slice.
func (b *Buffer) Download(dst []float32) error { return b.check("download") }

// DownloadF64 copies the buffer contents into a host float64 slice.
func (b *Buffer) DownloadF64(dst []float64) error { return b.check("download") }

// Zero fills the buffer with zero bytes.
func (b *Buffer) Zero() error { return b.check("zero") }

// DevicePointer exposes the raw device pointer (always nil on stubs).
func (b *Buffer) DevicePointer() unsafe.Pointer { return nil }

// DeviceInfo describes the primary CUDA device (stub builds never have one).
type DeviceInfo struct {
	Count      int
	Name       string
	Major      int
	Minor      int
	DeviceArch string
}

// ProbeDevice always fails on stub builds.
func ProbeDevice() (DeviceInfo, error) { return DeviceInfo{}, ErrUnavailable }

// LastError returns the most recent CUDA error, or nil.
func LastError() error { return nil }

// Sync blocks until all queued work on the device completes.
func Sync() error { return ErrUnavailable }

// DecodeMaxKeyF32 converts an order-preserving uint32 key to float32.
func DecodeMaxKeyF32(key uint32) float32 {
	var u uint32
	if key&0x80000000 != 0 {
		u = key & 0x7fffffff
	} else {
		u = ^key
	}
	return math.Float32frombits(u)
}

func unavail(what string) error {
	return fmt.Errorf("%w: %s", ErrUnavailable, what)
}

// GemmF32 computes C[M,N] = A[M,K] * B[K,N] on device buffers.
func GemmF32(A, B, C *Buffer, M, N, K int) error { return unavail("gemm_f32") }

// GemmF64 computes C[M,N] = A[M,K] * B[K,N] on device buffers.
func GemmF64(A, B, C *Buffer, M, N, K int) error { return unavail("gemm_f64") }

// ElemwiseF32 computes out[i] = a[i] OP b[i] on device buffers.
func ElemwiseF32(op int, A, B, Out *Buffer, n int) error {
	return unavail("elemwise_f32")
}

// ElemwiseF64 computes out[i] = a[i] OP b[i] on device buffers.
func ElemwiseF64(op int, A, B, Out *Buffer, n int) error {
	return unavail("elemwise_f64")
}

// ScaleF32 computes out[i] = src[i] * alpha on device buffers.
func ScaleF32(Src *Buffer, alpha float32, Out *Buffer, n int) error {
	return unavail("scale_f32")
}

// ScaleF64 computes out[i] = src[i] * alpha on device buffers.
func ScaleF64(Src *Buffer, alpha float64, Out *Buffer, n int) error {
	return unavail("scale_f64")
}

// SumF32 reduces a device float32 buffer to a single value.
func SumF32(Src, Out *Buffer, n int) error { return unavail("sum_f32") }

// MaxF32 reduces a device float32 buffer to its maximum.
func MaxF32(Src, Out *Buffer, n int) error { return unavail("max_f32") }

// SumF64 reduces a device float64 buffer to a single value.
func SumF64(Src, Out *Buffer, n int) error { return unavail("sum_f64") }

// MaxF64 reduces a device float64 buffer to its maximum.
func MaxF64(Src, Out *Buffer, n int) error { return unavail("max_f64") }

// MatMulFloat32 computes C[M,N] = A[M,K] * B[K,N] from host slices.
func MatMulFloat32(M, N, K int, A, B, C []float32) error {
	return unavail("matmul_f32")
}

// MatMulFloat64 computes C[M,N] = A[M,K] * B[K,N] from host slices.
func MatMulFloat64(M, N, K int, A, B, C []float64) error {
	return unavail("matmul_f64")
}

// AddFloat32 computes out=a+b on host slices.
func AddFloat32(a, b, out []float32) error { return unavail("add_f32") }

// SubFloat32 computes out=a-b on host slices.
func SubFloat32(a, b, out []float32) error { return unavail("sub_f32") }

// MulFloat32 computes out=a*b on host slices.
func MulFloat32(a, b, out []float32) error { return unavail("mul_f32") }

// DivFloat32 computes out=a/b on host slices.
func DivFloat32(a, b, out []float32) error { return unavail("div_f32") }

// AddFloat64 computes out=a+b on host slices.
func AddFloat64(a, b, out []float64) error { return unavail("add_f64") }

// SubFloat64 computes out=a-b on host slices.
func SubFloat64(a, b, out []float64) error { return unavail("sub_f64") }

// MulFloat64 computes out=a*b on host slices.
func MulFloat64(a, b, out []float64) error { return unavail("mul_f64") }

// DivFloat64 computes out=a/b on host slices.
func DivFloat64(a, b, out []float64) error { return unavail("div_f64") }

// ScaleFloat32 computes out[i] = src[i] * alpha on host slices.
func ScaleFloat32(src []float32, alpha float32, out []float32) error {
	return unavail("scale_f32")
}

// ScaleFloat64 computes out[i] = src[i] * alpha on host slices.
func ScaleFloat64(src []float64, alpha float64, out []float64) error {
	return unavail("scale_f64")
}

// SumFloat32 reduces a host float32 slice on the GPU.
func SumFloat32(src []float32) (float32, error) { return 0, unavail("sum_f32") }

// MaxFloat32 reduces a host float32 slice on the GPU.
func MaxFloat32(src []float32) (float32, error) { return 0, unavail("max_f32") }

// SumFloat64 reduces a host float64 slice on the GPU.
func SumFloat64(src []float64) (float64, error) { return 0, unavail("sum_f64") }

// MaxFloat64 reduces a host float64 slice on the GPU.
func MaxFloat64(src []float64) (float64, error) { return 0, unavail("max_f64") }

// MemcpyD2D copies bytes between device buffers (stub).
func MemcpyD2D(dst *Buffer, dstOff int, src *Buffer, srcOff int, n int) error {
	return unavail("d2d memcpy")
}

// RMSNormF32 stub.
func RMSNormF32(x, gamma, out *Buffer, rows, width int, eps float32) error {
	return unavail("rmsnorm_f32")
}

// RoPEBatchF32 stub.
func RoPEBatchF32(q, k *Buffer, rows, width, headDim, heads int) error {
	return unavail("rope_batch")
}

// RoPESingleF32 stub.
func RoPESingleF32(q, k *Buffer, pos, width, headDim, heads int) error {
	return unavail("rope_single")
}

// SwiGLUF32 stub.
func SwiGLUF32(gate, up, out *Buffer, n int) error { return unavail("swiglu") }

// BiasAddF32 stub.
func BiasAddF32(c, bias *Buffer, M, N int) error { return unavail("bias_add") }

// EmbedF32 stub.
func EmbedF32(emb, ids, out *Buffer, rows, width int) error {
	return unavail("embed")
}

// AttnScoresF32 stub.
func AttnScoresF32(q, k, attn *Buffer, heads, qRows, kvLen, width, headDim int,
	scale float32, causal bool) error {
	return unavail("attn_scores")
}

// AttnSoftmaxF32 stub.
func AttnSoftmaxF32(attn *Buffer, heads, qRows, kvLen int) error {
	return unavail("attn_softmax")
}

// AttnWeightedF32 stub.
func AttnWeightedF32(attn, v, out *Buffer, heads, qRows, kvLen, width, headDim int) error {
	return unavail("attn_weighted")
}
