package autograd

import (
	"fmt"

	"m3tal-ai/internal/tensor"
)

// MatMul performs matrix multiplication between a and b.
// Supports 2D matrices: (M, K) x (K, N) -> (M, N),
// as well as batched 3D/N-D matrices: (..., M, K) x (..., K, N) -> (..., M, N).
func MatMul(a, b *Variable) (*Variable, error) {
	if a == nil || b == nil || a.Data == nil || b.Data == nil {
		return nil, ErrNilTensor
	}
	if a.Data.Rank() < 2 || b.Data.Rank() < 2 {
		return nil, fmt.Errorf("%w: MatMul requires at least 2D tensors, got ranks %d and %d", ErrUnsupportedOp, a.Data.Rank(), b.Data.Rank())
	}
	if a.DType() != b.DType() {
		return nil, fmt.Errorf("%w: MatMul dtype mismatch %s and %s", ErrUnsupportedOp, a.DType(), b.DType())
	}

	shapeA := a.Shape()
	shapeB := b.Shape()
	rankA := len(shapeA)
	rankB := len(shapeB)

	mA, kA := shapeA[rankA-2], shapeA[rankA-1]
	kB, nB := shapeB[rankB-2], shapeB[rankB-1]
	if kA != kB {
		return nil, fmt.Errorf("%w: inner dimensions mismatch for MatMul: %d != %d", ErrUnsupportedOp, kA, kB)
	}

	// Batch shapes
	batchA := shapeA[:rankA-2]
	batchB := shapeB[:rankB-2]
	batchOut, err := broadcastShapes(batchA, batchB)
	if err != nil {
		return nil, fmt.Errorf("%w: incompatible batch shapes for MatMul: %v and %v", ErrUnsupportedOp, batchA, batchB)
	}

	outTensor, err := rawBatchedMatMul(a.Data, b.Data, batchOut, mA, kA, nB)
	if err != nil {
		return nil, err
	}

	requiresGrad := a.RequiresGrad || b.RequiresGrad
	parents := make([]*Variable, 0, 2)
	if a.RequiresGrad {
		parents = append(parents, a)
	}
	if b.RequiresGrad {
		parents = append(parents, b)
	}

	out := &Variable{
		Data:         outTensor,
		RequiresGrad: requiresGrad,
		parents:      parents,
	}

	out.backward = func() error {
		if out.Grad == nil {
			return nil
		}
		// dL/dA = grad * B^T
		if a.RequiresGrad {
			bT, err := transposeMatrixAxes(b.Data)
			if err != nil {
				return err
			}
			gradFullA, err := MatMulRaw(out.Grad, bT)
			if err != nil {
				return err
			}
			gradA, err := reduceToShape(gradFullA, a.Shape())
			if err != nil {
				return err
			}
			if err := a.AccGrad(gradA); err != nil {
				return err
			}
		}
		// dL/dB = A^T * grad
		if b.RequiresGrad {
			aT, err := transposeMatrixAxes(a.Data)
			if err != nil {
				return err
			}
			gradFullB, err := MatMulRaw(aT, out.Grad)
			if err != nil {
				return err
			}
			gradB, err := reduceToShape(gradFullB, b.Shape())
			if err != nil {
				return err
			}
			if err := b.AccGrad(gradB); err != nil {
				return err
			}
		}
		return nil
	}

	return out, nil
}

// MatMulRaw computes raw matrix multiplication for two tensors with batch broadcasting.
func MatMulRaw(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	shapeA := a.Shape()
	shapeB := b.Shape()
	rankA := len(shapeA)
	rankB := len(shapeB)
	if rankA < 2 || rankB < 2 {
		return nil, fmt.Errorf("%w: MatMulRaw requires at least 2D tensors", ErrUnsupportedOp)
	}
	mA, kA := shapeA[rankA-2], shapeA[rankA-1]
	kB, nB := shapeB[rankB-2], shapeB[rankB-1]
	if kA != kB {
		return nil, fmt.Errorf("%w: inner dimensions mismatch: %d != %d", ErrUnsupportedOp, kA, kB)
	}
	batchA := shapeA[:rankA-2]
	batchB := shapeB[:rankB-2]
	batchOut, err := broadcastShapes(batchA, batchB)
	if err != nil {
		return nil, err
	}
	return rawBatchedMatMul(a, b, batchOut, mA, kA, nB)
}

// transposeMatrixAxes transposes the last two dimensions (matrix dimensions) of an N-D tensor.
func transposeMatrixAxes(t *tensor.Tensor) (*tensor.Tensor, error) {
	rank := t.Rank()
	if rank < 2 {
		return nil, fmt.Errorf("%w: transposeMatrixAxes requires rank >= 2", ErrUnsupportedOp)
	}
	shape := t.Shape()
	outShape := append([]int(nil), shape...)
	outShape[rank-2], outShape[rank-1] = shape[rank-1], shape[rank-2]

	out, err := tensor.New(outShape, t.DType())
	if err != nil {
		return nil, err
	}
	numel := t.Numel()
	switch t.DType() {
	case tensor.Float32:
		for i := 0; i < numel; i++ {
			srcIdx, err := flatToIndices(i, shape)
			if err != nil {
				return nil, err
			}
			dstIdx := append([]int(nil), srcIdx...)
			dstIdx[rank-2], dstIdx[rank-1] = srcIdx[rank-1], srcIdx[rank-2]
			v, err := t.Float32At(srcIdx...)
			if err != nil {
				return nil, err
			}
			if err := out.SetFloat32At(v, dstIdx...); err != nil {
				return nil, err
			}
		}
	case tensor.Float64:
		for i := 0; i < numel; i++ {
			srcIdx, err := flatToIndices(i, shape)
			if err != nil {
				return nil, err
			}
			dstIdx := append([]int(nil), srcIdx...)
			dstIdx[rank-2], dstIdx[rank-1] = srcIdx[rank-1], srcIdx[rank-2]
			v, err := t.Float64At(srcIdx...)
			if err != nil {
				return nil, err
			}
			if err := out.SetFloat64At(v, dstIdx...); err != nil {
				return nil, err
			}
		}
	default:
		return nil, fmt.Errorf("%w: transposeMatrixAxes for dtype %s", ErrUnsupportedOp, t.DType())
	}
	return out, nil
}

// rawBatchedMatMul computes batched matrix multiplication over arbitrary batch dims.
func rawBatchedMatMul(a, b *tensor.Tensor, batchShape []int, m, k, n int) (*tensor.Tensor, error) {
	outShape := append(append([]int(nil), batchShape...), m, n)
	out, err := tensor.New(outShape, a.DType())
	if err != nil {
		return nil, err
	}

	batchNumel, _ := tensor.Numel(batchShape)
	shapeA := a.Shape()
	shapeB := b.Shape()
	batchRankA := len(shapeA) - 2
	batchRankB := len(shapeB) - 2

	switch a.DType() {
	case tensor.Float32:
		for batchIdx := 0; batchIdx < batchNumel; batchIdx++ {
			bCoord, err := flatToIndices(batchIdx, batchShape)
			if err != nil {
				return nil, err
			}
			bCoordA := broadcastIndex(bCoord, shapeA[:batchRankA])
			bCoordB := broadcastIndex(bCoord, shapeB[:batchRankB])

			idxA := make([]int, len(shapeA))
			copy(idxA, bCoordA)
			idxB := make([]int, len(shapeB))
			copy(idxB, bCoordB)
			idxOut := make([]int, len(outShape))
			copy(idxOut, bCoord)

			for i := 0; i < m; i++ {
				idxA[len(shapeA)-2] = i
				idxOut[len(outShape)-2] = i
				for j := 0; j < n; j++ {
					idxB[len(shapeB)-1] = j
					idxOut[len(outShape)-1] = j
					var sum float32
					for p := 0; p < k; p++ {
						idxA[len(shapeA)-1] = p
						idxB[len(shapeB)-2] = p
						va, err := a.Float32At(idxA...)
						if err != nil {
							return nil, err
						}
						vb, err := b.Float32At(idxB...)
						if err != nil {
							return nil, err
						}
						sum += va * vb
					}
					if err := out.SetFloat32At(sum, idxOut...); err != nil {
						return nil, err
					}
				}
			}
		}
	case tensor.Float64:
		for batchIdx := 0; batchIdx < batchNumel; batchIdx++ {
			bCoord, err := flatToIndices(batchIdx, batchShape)
			if err != nil {
				return nil, err
			}
			bCoordA := broadcastIndex(bCoord, shapeA[:batchRankA])
			bCoordB := broadcastIndex(bCoord, shapeB[:batchRankB])

			idxA := make([]int, len(shapeA))
			copy(idxA, bCoordA)
			idxB := make([]int, len(shapeB))
			copy(idxB, bCoordB)
			idxOut := make([]int, len(outShape))
			copy(idxOut, bCoord)

			for i := 0; i < m; i++ {
				idxA[len(shapeA)-2] = i
				idxOut[len(outShape)-2] = i
				for j := 0; j < n; j++ {
					idxB[len(shapeB)-1] = j
					idxOut[len(outShape)-1] = j
					var sum float64
					for p := 0; p < k; p++ {
						idxA[len(shapeA)-1] = p
						idxB[len(shapeB)-2] = p
						va, err := a.Float64At(idxA...)
						if err != nil {
							return nil, err
						}
						vb, err := b.Float64At(idxB...)
						if err != nil {
							return nil, err
						}
						sum += va * vb
					}
					if err := out.SetFloat64At(sum, idxOut...); err != nil {
						return nil, err
					}
				}
			}
		}
	default:
		return nil, fmt.Errorf("%w: rawBatchedMatMul for %s", ErrUnsupportedOp, a.DType())
	}

	return out, nil
}
