package kernel

import (
	"runtime"
	"sync"
)

const (
	// Tile sizes tuned for L1/L2 CPU cache blocking
	TileM = 32
	TileN = 32
	TileK = 32

	// Threshold of total multiply-accumulate operations to enable multithreading
	ParallelThresholdOps = 64 * 64 * 64
)

// MatMulFloat32Blocked computes C = A * B using cache-blocked loop reordering (ikj)
// for single-threaded cache locality.
func MatMulFloat32Blocked(M, N, K int, A, B, C []float32) {
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

				// Micro-loops over blocks with ikj order to maximize row-major streaming of B and C
				for i := i0; i < iMax; i++ {
					cRow := C[i*N : (i+1)*N]
					for k := k0; k < kMax; k++ {
						aElem := A[i*K+k]
						if aElem == 0 {
							continue
						}
						bRow := B[k*N : (k+1)*N]
						for j := j0; j < jMax; j++ {
							cRow[j] += aElem * bRow[j]
						}
					}
				}
			}
		}
	}
}

// MatMulFloat64Blocked computes C = A * B using cache-blocked loop reordering (ikj)
// for single-threaded cache locality.
func MatMulFloat64Blocked(M, N, K int, A, B, C []float64) {
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
						for j := j0; j < jMax; j++ {
							cRow[j] += aElem * bRow[j]
						}
					}
				}
			}
		}
	}
}

// MatMulFloat32Parallel partitions row-blocks of A and C across available CPU workers.
func MatMulFloat32Parallel(M, N, K int, A, B, C []float32) {
	if M == 0 || N == 0 || K == 0 {
		return
	}
	totalOps := M * N * K
	numWorkers := runtime.GOMAXPROCS(0)
	if numWorkers <= 1 || totalOps < ParallelThresholdOps || M < 2 {
		MatMulFloat32Blocked(M, N, K, A, B, C)
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
			MatMulFloat32Blocked(subM, N, K, subA, B, subC)
		}(iStart, iEnd)
	}
	wg.Wait()
}

// MatMulFloat64Parallel partitions row-blocks of A and C across available CPU workers.
func MatMulFloat64Parallel(M, N, K int, A, B, C []float64) {
	if M == 0 || N == 0 || K == 0 {
		return
	}
	totalOps := M * N * K
	numWorkers := runtime.GOMAXPROCS(0)
	if numWorkers <= 1 || totalOps < ParallelThresholdOps || M < 2 {
		MatMulFloat64Blocked(M, N, K, A, B, C)
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
			MatMulFloat64Blocked(subM, N, K, subA, B, subC)
		}(iStart, iEnd)
	}
	wg.Wait()
}
