//go:build arm64

package kernel

import (
	"runtime"
	"sync"
)

// NEON (ASIMD) implementation for arm64.
// Go's arm64 compiler auto-vectorizes clean unrolled 4-wide float32 loops using NEON instructions.

func AddFloat32NEON(dst, a, b []float32) {
	n := len(dst)
	i := 0
	for ; i+4 <= n; i += 4 {
		dst[i] = a[i] + b[i]
		dst[i+1] = a[i+1] + b[i+1]
		dst[i+2] = a[i+2] + b[i+2]
		dst[i+3] = a[i+3] + b[i+3]
	}
	for ; i < n; i++ {
		dst[i] = a[i] + b[i]
	}
}

func MulFloat32NEON(dst, a, b []float32) {
	n := len(dst)
	i := 0
	for ; i+4 <= n; i += 4 {
		dst[i] = a[i] * b[i]
		dst[i+1] = a[i+1] * b[i+1]
		dst[i+2] = a[i+2] * b[i+2]
		dst[i+3] = a[i+3] * b[i+3]
	}
	for ; i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}

func AddFloat64NEON(dst, a, b []float64) {
	n := len(dst)
	i := 0
	for ; i+2 <= n; i += 2 {
		dst[i] = a[i] + b[i]
		dst[i+1] = a[i+1] + b[i+1]
	}
	for ; i < n; i++ {
		dst[i] = a[i] + b[i]
	}
}

func MulFloat64NEON(dst, a, b []float64) {
	n := len(dst)
	i := 0
	for ; i+2 <= n; i += 2 {
		dst[i] = a[i] * b[i]
		dst[i+1] = a[i+1] * b[i+1]
	}
	for ; i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}

func MatMulFloat32NEON(M, N, K int, A, B, C []float32) {
	if M == 0 || N == 0 || K == 0 {
		return
	}
	totalOps := M * N * K
	numWorkers := runtime.GOMAXPROCS(0)
	if numWorkers <= 1 || totalOps < ParallelThresholdOps || M < 2 {
		matmulFloat32NEONBlocked(M, N, K, A, B, C)
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
			matmulFloat32NEONBlocked(subM, N, K, subA, B, subC)
		}(iStart, iEnd)
	}
	wg.Wait()
}

func matmulFloat32NEONBlocked(M, N, K int, A, B, C []float32) {
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

				for i := i0; i < iMax; i++ {
					cRow := C[i*N : (i+1)*N]
					for k := k0; k < kMax; k++ {
						aElem := A[i*K+k]
						if aElem == 0 {
							continue
						}
						bRow := B[k*N : (k+1)*N]
						j := j0
						for ; j+4 <= jMax; j += 4 {
							cRow[j] += aElem * bRow[j]
							cRow[j+1] += aElem * bRow[j+1]
							cRow[j+2] += aElem * bRow[j+2]
							cRow[j+3] += aElem * bRow[j+3]
						}
						for ; j < jMax; j++ {
							cRow[j] += aElem * bRow[j]
						}
					}
				}
			}
		}
	}
}

func MatMulFloat64NEON(M, N, K int, A, B, C []float64) {
	MatMulFloat64Parallel(M, N, K, A, B, C)
}
