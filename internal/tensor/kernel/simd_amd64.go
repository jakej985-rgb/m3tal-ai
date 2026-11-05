//go:build amd64

package kernel

import (
	"runtime"
	"sync"
	"unsafe"
)

func addFloat32AVX2Asm(dst, a, b unsafe.Pointer, n int)
func mulFloat32AVX2Asm(dst, a, b unsafe.Pointer, n int)
func dotFloat32AVX2Asm(a, b unsafe.Pointer, n int) float32
func fmaRowFloat32AVX2Asm(dst, b unsafe.Pointer, alpha float32, n int)

func AddFloat32AVX2(dst, a, b []float32) {
	n := len(dst)
	if n == 0 {
		return
	}
	addFloat32AVX2Asm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), n)
}

func MulFloat32AVX2(dst, a, b []float32) {
	n := len(dst)
	if n == 0 {
		return
	}
	mulFloat32AVX2Asm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), n)
}

func AddFloat64AVX2(dst, a, b []float64) {
	AddFloat64Fallback(dst, a, b)
}

func MulFloat64AVX2(dst, a, b []float64) {
	MulFloat64Fallback(dst, a, b)
}

func MatMulFloat32AVX2(M, N, K int, A, B, C []float32) {
	if M == 0 || N == 0 || K == 0 {
		return
	}
	totalOps := M * N * K
	numWorkers := runtime.GOMAXPROCS(0)
	if numWorkers <= 1 || totalOps < ParallelThresholdOps || M < 2 {
		matmulFloat32AVX2Blocked(M, N, K, A, B, C)
		return
	}

	if numWorkers > M {
		numWorkers = M
	}
	chunkSize := (M + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		iStart := w * chunkSize
		if iStart >= M {
			break
		}
		iEnd := iStart + chunkSize
		if iEnd > M {
			iEnd = M
		}

		wg.Add(1)
		go func(rowStart, rowEnd int) {
			defer wg.Done()
			subM := rowEnd - rowStart
			subA := A[rowStart*K : rowEnd*K]
			subC := C[rowStart*N : rowEnd*N]
			matmulFloat32AVX2Blocked(subM, N, K, subA, B, subC)
		}(iStart, iEnd)
	}
	wg.Wait()
}

func matmulFloat32AVX2Blocked(M, N, K int, A, B, C []float32) {
	for i := 0; i < M*N; i++ {
		C[i] = 0
	}
	if M == 0 || N == 0 || K == 0 {
		return
	}

	for i0 := 0; i0 < M; i0 += TileM {
		iMax := i0 + TileM
		if iMax > M {
			iMax = M
		}
		for k0 := 0; k0 < K; k0 += TileK {
			kMax := k0 + TileK
			if kMax > K {
				kMax = K
			}
			for j0 := 0; j0 < N; j0 += TileN {
				jMax := j0 + TileN
				if jMax > N {
					jMax = N
				}
				numCols := jMax - j0

				for i := i0; i < iMax; i++ {
					cRowPtr := unsafe.Pointer(&C[i*N+j0])
					for k := k0; k < kMax; k++ {
						aElem := A[i*K+k]
						if aElem == 0 {
							continue
						}
						bRowPtr := unsafe.Pointer(&B[k*N+j0])
						fmaRowFloat32AVX2Asm(cRowPtr, bRowPtr, aElem, numCols)
					}
				}
			}
		}
	}
}

func MatMulFloat64AVX2(M, N, K int, A, B, C []float64) {
	MatMulFloat64Parallel(M, N, K, A, B, C)
}
