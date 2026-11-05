package tensor

import (
	"encoding/binary"
	"fmt"
	"math"
)

// MatMul performs matrix multiplication between two tensors.
// It supports:
// - 2D x 2D: [M, K] x [K, N] -> [M, N]
// - Batched N-D: [..., M, K] x [..., K, N] -> [..., M, N]
// Batch dimensions are broadcast according to standard broadcasting rules.
func MatMul(a, b *Tensor) (*Tensor, error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("%w: input tensor cannot be nil", ErrInvalidShape)
	}
	if !a.device.IsCPU() || !b.device.IsCPU() {
		return nil, fmt.Errorf("%w: MatMul only supported on CPU", ErrUnsupportedDevice)
	}
	if a.dtype != b.dtype {
		return nil, fmt.Errorf("%w: MatMul requires matching dtypes, got %s and %s", ErrUnsupportedDType, a.dtype, b.dtype)
	}
	if a.dtype != Float32 && a.dtype != Float64 {
		return nil, fmt.Errorf("%w: MatMul not supported for dtype %s", ErrUnsupportedDType, a.dtype)
	}

	rA, rB := len(a.shape), len(b.shape)
	if rA < 2 || rB < 2 {
		return nil, fmt.Errorf("%w: MatMul requires tensors of rank at least 2, got %d and %d", ErrShapeMismatch, rA, rB)
	}

	M := a.shape[rA-2]
	Ka := a.shape[rA-1]
	Kb := b.shape[rB-2]
	N := b.shape[rB-1]

	if Ka != Kb {
		return nil, fmt.Errorf("%w: inner matrix dimension mismatch: %d vs %d", ErrShapeMismatch, Ka, Kb)
	}
	K := Ka

	// Broadcast batch dimensions
	batchA := a.shape[:rA-2]
	batchB := b.shape[:rB-2]
	broadcastBatch, err := BroadcastShapes(batchA, batchB)
	if err != nil {
		return nil, fmt.Errorf("%w: batch dimensions cannot be broadcast: %v and %v", err, batchA, batchB)
	}

	// Result shape: broadcastBatch + [M, N]
	outShape := make([]int, len(broadcastBatch)+2)
	copy(outShape, broadcastBatch)
	outShape[len(broadcastBatch)] = M
	outShape[len(broadcastBatch)+1] = N

	out, err := New(outShape, a.dtype)
	if err != nil {
		return nil, err
	}

	// Total batch count
	batchCount, err := Numel(broadcastBatch)
	if err != nil {
		return nil, err
	}

	if batchCount == 0 || M == 0 || N == 0 {
		return out, nil
	}

	// Prepare broadcasted view shapes for batch iterations:
	// a: broadcastBatch + [M, K]
	// b: broadcastBatch + [K, N]
	aTargetShape := make([]int, len(broadcastBatch)+2)
	copy(aTargetShape, broadcastBatch)
	aTargetShape[len(broadcastBatch)] = M
	aTargetShape[len(broadcastBatch)+1] = K

	bTargetShape := make([]int, len(broadcastBatch)+2)
	copy(bTargetShape, broadcastBatch)
	bTargetShape[len(broadcastBatch)] = K
	bTargetShape[len(broadcastBatch)+1] = N

	aView, err := a.BroadcastTo(aTargetShape)
	if err != nil {
		return nil, err
	}
	bView, err := b.BroadcastTo(bTargetShape)
	if err != nil {
		return nil, err
	}

	batchIndices := make([]int, len(broadcastBatch))

	for bIdx := 0; bIdx < batchCount; bIdx++ {
		// Calculate base element offset for batch slice
		aCoord := make([]int, len(aTargetShape))
		copy(aCoord, batchIndices)
		aCoord[len(batchIndices)] = 0
		aCoord[len(batchIndices)+1] = 0

		bCoord := make([]int, len(bTargetShape))
		copy(bCoord, batchIndices)
		bCoord[len(batchIndices)] = 0
		bCoord[len(batchIndices)+1] = 0

		cCoord := make([]int, len(outShape))
		copy(cCoord, batchIndices)
		cCoord[len(batchIndices)] = 0
		cCoord[len(batchIndices)+1] = 0

		aBaseOffset, err := ComputeOffset(aCoord, aView.shape, aView.strides, aView.offset)
		if err != nil {
			return nil, err
		}
		bBaseOffset, err := ComputeOffset(bCoord, bView.shape, bView.strides, bView.offset)
		if err != nil {
			return nil, err
		}
		cBaseOffset, err := ComputeOffset(cCoord, out.shape, out.strides, out.offset)
		if err != nil {
			return nil, err
		}

		strideRowA := aView.strides[len(aTargetShape)-2]
		strideColA := aView.strides[len(aTargetShape)-1]

		strideRowB := bView.strides[len(bTargetShape)-2]
		strideColB := bView.strides[len(bTargetShape)-1]

		strideRowC := out.strides[len(outShape)-2]
		strideColC := out.strides[len(outShape)-1]

		if a.dtype == Float32 {
			gemmBlockedFloat32(
				M, N, K,
				aView.storage.Bytes(), aBaseOffset, strideRowA, strideColA,
				bView.storage.Bytes(), bBaseOffset, strideRowB, strideColB,
				out.storage.Bytes(), cBaseOffset, strideRowC, strideColC,
			)
		} else {
			gemmBlockedFloat64(
				M, N, K,
				aView.storage.Bytes(), aBaseOffset, strideRowA, strideColA,
				bView.storage.Bytes(), bBaseOffset, strideRowB, strideColB,
				out.storage.Bytes(), cBaseOffset, strideRowC, strideColC,
			)
		}

		advanceIndices(batchIndices, broadcastBatch)
	}

	return out, nil
}

// Block sizes chosen to fit L1/L2 caches and facilitate CPU prefetching and loop unrolling.
const (
	blockSizeM = 64
	blockSizeN = 64
	blockSizeK = 64
)

func gemmBlockedFloat32(
	M, N, K int,
	aBytes []byte, aOffset, aRowStride, aColStride int,
	bBytes []byte, bOffset, bRowStride, bColStride int,
	cBytes []byte, cOffset, cRowStride, cColStride int,
) {
	if K == 0 {
		return
	}

	// Cache-blocked loops over M, N, K
	for bi := 0; bi < M; bi += blockSizeM {
		iend := bi + blockSizeM
		if iend > M {
			iend = M
		}
		for bk := 0; bk < K; bk += blockSizeK {
			kend := bk + blockSizeK
			if kend > K {
				kend = K
			}
			for bj := 0; bj < N; bj += blockSizeN {
				jend := bj + blockSizeN
				if jend > N {
					jend = N
				}

				// Inner kernel: i, k, j order for optimal spatial locality and streaming writes to C
				for i := bi; i < iend; i++ {
					aRowBase := aOffset + i*aRowStride
					cRowBase := cOffset + i*cRowStride

					for k := bk; k < kend; k++ {
						aElemOff := (aRowBase + k*aColStride) * 4
						aBits := binary.LittleEndian.Uint32(aBytes[aElemOff : aElemOff+4])
						aVal := math.Float32frombits(aBits)
						if aVal == 0 {
							continue
						}

						bRowBase := bOffset + k*bRowStride

						j := bj
						// Unroll by 4 for inner loop
						for ; j <= jend-4; j += 4 {
							bElemOff0 := (bRowBase + j*bColStride) * 4
							bElemOff1 := (bRowBase + (j+1)*bColStride) * 4
							bElemOff2 := (bRowBase + (j+2)*bColStride) * 4
							bElemOff3 := (bRowBase + (j+3)*bColStride) * 4

							cElemOff0 := (cRowBase + j*cColStride) * 4
							cElemOff1 := (cRowBase + (j+1)*cColStride) * 4
							cElemOff2 := (cRowBase + (j+2)*cColStride) * 4
							cElemOff3 := (cRowBase + (j+3)*cColStride) * 4

							bVal0 := math.Float32frombits(binary.LittleEndian.Uint32(bBytes[bElemOff0 : bElemOff0+4]))
							bVal1 := math.Float32frombits(binary.LittleEndian.Uint32(bBytes[bElemOff1 : bElemOff1+4]))
							bVal2 := math.Float32frombits(binary.LittleEndian.Uint32(bBytes[bElemOff2 : bElemOff2+4]))
							bVal3 := math.Float32frombits(binary.LittleEndian.Uint32(bBytes[bElemOff3 : bElemOff3+4]))

							cVal0 := math.Float32frombits(binary.LittleEndian.Uint32(cBytes[cElemOff0 : cElemOff0+4]))
							cVal1 := math.Float32frombits(binary.LittleEndian.Uint32(cBytes[cElemOff1 : cElemOff1+4]))
							cVal2 := math.Float32frombits(binary.LittleEndian.Uint32(cBytes[cElemOff2 : cElemOff2+4]))
							cVal3 := math.Float32frombits(binary.LittleEndian.Uint32(cBytes[cElemOff3 : cElemOff3+4]))

							cVal0 += aVal * bVal0
							cVal1 += aVal * bVal1
							cVal2 += aVal * bVal2
							cVal3 += aVal * bVal3

							binary.LittleEndian.PutUint32(cBytes[cElemOff0:cElemOff0+4], math.Float32bits(cVal0))
							binary.LittleEndian.PutUint32(cBytes[cElemOff1:cElemOff1+4], math.Float32bits(cVal1))
							binary.LittleEndian.PutUint32(cBytes[cElemOff2:cElemOff2+4], math.Float32bits(cVal2))
							binary.LittleEndian.PutUint32(cBytes[cElemOff3:cElemOff3+4], math.Float32bits(cVal3))
						}

						// Remaining elements
						for ; j < jend; j++ {
							bElemOff := (bRowBase + j*bColStride) * 4
							cElemOff := (cRowBase + j*cColStride) * 4

							bVal := math.Float32frombits(binary.LittleEndian.Uint32(bBytes[bElemOff : bElemOff+4]))
							cVal := math.Float32frombits(binary.LittleEndian.Uint32(cBytes[cElemOff : cElemOff+4]))

							cVal += aVal * bVal
							binary.LittleEndian.PutUint32(cBytes[cElemOff:cElemOff+4], math.Float32bits(cVal))
						}
					}
				}
			}
		}
	}
}

func gemmBlockedFloat64(
	M, N, K int,
	aBytes []byte, aOffset, aRowStride, aColStride int,
	bBytes []byte, bOffset, bRowStride, bColStride int,
	cBytes []byte, cOffset, cRowStride, cColStride int,
) {
	if K == 0 {
		return
	}

	for bi := 0; bi < M; bi += blockSizeM {
		iend := bi + blockSizeM
		if iend > M {
			iend = M
		}
		for bk := 0; bk < K; bk += blockSizeK {
			kend := bk + blockSizeK
			if kend > K {
				kend = K
			}
			for bj := 0; bj < N; bj += blockSizeN {
				jend := bj + blockSizeN
				if jend > N {
					jend = N
				}

				for i := bi; i < iend; i++ {
					aRowBase := aOffset + i*aRowStride
					cRowBase := cOffset + i*cRowStride

					for k := bk; k < kend; k++ {
						aElemOff := (aRowBase + k*aColStride) * 8
						aBits := binary.LittleEndian.Uint64(aBytes[aElemOff : aElemOff+8])
						aVal := math.Float64frombits(aBits)
						if aVal == 0 {
							continue
						}

						bRowBase := bOffset + k*bRowStride

						j := bj
						for ; j <= jend-4; j += 4 {
							bElemOff0 := (bRowBase + j*bColStride) * 8
							bElemOff1 := (bRowBase + (j+1)*bColStride) * 8
							bElemOff2 := (bRowBase + (j+2)*bColStride) * 8
							bElemOff3 := (bRowBase + (j+3)*bColStride) * 8

							cElemOff0 := (cRowBase + j*cColStride) * 8
							cElemOff1 := (cRowBase + (j+1)*cColStride) * 8
							cElemOff2 := (cRowBase + (j+2)*cColStride) * 8
							cElemOff3 := (cRowBase + (j+3)*cColStride) * 8

							bVal0 := math.Float64frombits(binary.LittleEndian.Uint64(bBytes[bElemOff0 : bElemOff0+8]))
							bVal1 := math.Float64frombits(binary.LittleEndian.Uint64(bBytes[bElemOff1 : bElemOff1+8]))
							bVal2 := math.Float64frombits(binary.LittleEndian.Uint64(bBytes[bElemOff2 : bElemOff2+8]))
							bVal3 := math.Float64frombits(binary.LittleEndian.Uint64(bBytes[bElemOff3 : bElemOff3+8]))

							cVal0 := math.Float64frombits(binary.LittleEndian.Uint64(cBytes[cElemOff0 : cElemOff0+8]))
							cVal1 := math.Float64frombits(binary.LittleEndian.Uint64(cBytes[cElemOff1 : cElemOff1+8]))
							cVal2 := math.Float64frombits(binary.LittleEndian.Uint64(cBytes[cElemOff2 : cElemOff2+8]))
							cVal3 := math.Float64frombits(binary.LittleEndian.Uint64(cBytes[cElemOff3 : cElemOff3+8]))

							cVal0 += aVal * bVal0
							cVal1 += aVal * bVal1
							cVal2 += aVal * bVal2
							cVal3 += aVal * bVal3

							binary.LittleEndian.PutUint64(cBytes[cElemOff0:cElemOff0+8], math.Float64bits(cVal0))
							binary.LittleEndian.PutUint64(cBytes[cElemOff1:cElemOff1+8], math.Float64bits(cVal1))
							binary.LittleEndian.PutUint64(cBytes[cElemOff2:cElemOff2+8], math.Float64bits(cVal2))
							binary.LittleEndian.PutUint64(cBytes[cElemOff3:cElemOff3+8], math.Float64bits(cVal3))
						}

						for ; j < jend; j++ {
							bElemOff := (bRowBase + j*bColStride) * 8
							cElemOff := (cRowBase + j*cColStride) * 8

							bVal := math.Float64frombits(binary.LittleEndian.Uint64(bBytes[bElemOff : bElemOff+8]))
							cVal := math.Float64frombits(binary.LittleEndian.Uint64(cBytes[cElemOff : cElemOff+8]))

							cVal += aVal * bVal
							binary.LittleEndian.PutUint64(cBytes[cElemOff:cElemOff+8], math.Float64bits(cVal))
						}
					}
				}
			}
		}
	}
}

// refMatMul is a straightforward naive triple-nested loop reference implementation used for testing and parity verification.
func refMatMul(a, b *Tensor) (*Tensor, error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("%w: input tensor cannot be nil", ErrInvalidShape)
	}
	rA, rB := len(a.shape), len(b.shape)
	if rA != 2 || rB != 2 {
		return nil, fmt.Errorf("%w: refMatMul only supports 2D tensors", ErrShapeMismatch)
	}
	M, Ka := a.shape[0], a.shape[1]
	Kb, N := b.shape[0], b.shape[1]
	if Ka != Kb {
		return nil, fmt.Errorf("%w: inner dimension mismatch", ErrShapeMismatch)
	}
	K := Ka

	out, err := New([]int{M, N}, a.dtype)
	if err != nil {
		return nil, err
	}

	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float64
			for k := 0; k < K; k++ {
				var va, vb float64
				if a.dtype == Float32 {
					valA, _ := a.Float32At(i, k)
					valB, _ := b.Float32At(k, j)
					va = float64(valA)
					vb = float64(valB)
				} else {
					va, _ = a.Float64At(i, k)
					vb, _ = b.Float64At(k, j)
				}
				sum += va * vb
			}
			if a.dtype == Float32 {
				_ = out.SetFloat32At(float32(sum), i, j)
			} else {
				_ = out.SetFloat64At(sum, i, j)
			}
		}
	}
	return out, nil
}
