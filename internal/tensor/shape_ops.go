package tensor

import "fmt"

// BroadcastShapes computes the common broadcast shape for two input shapes according to standard broadcasting rules:
// Two dimensions are compatible when:
// 1. They are equal, or
// 2. One of them is 1.
// Trailing dimensions are aligned first.
func BroadcastShapes(s1, s2 []int) ([]int, error) {
	r1, r2 := len(s1), len(s2)
	maxRank := r1
	if r2 > maxRank {
		maxRank = r2
	}
	outShape := make([]int, maxRank)

	for i := 0; i < maxRank; i++ {
		dim1 := 1
		idx1 := r1 - 1 - i
		if idx1 >= 0 {
			dim1 = s1[idx1]
		}

		dim2 := 1
		idx2 := r2 - 1 - i
		if idx2 >= 0 {
			dim2 = s2[idx2]
		}

		if dim1 < 0 || dim2 < 0 {
			return nil, fmt.Errorf("%w: negative dimension in broadcast", ErrInvalidShape)
		}

		if dim1 == dim2 {
			outShape[maxRank-1-i] = dim1
		} else if dim1 == 1 {
			outShape[maxRank-1-i] = dim2
		} else if dim2 == 1 {
			outShape[maxRank-1-i] = dim1
		} else {
			return nil, fmt.Errorf("%w: cannot broadcast shapes %v and %v (dimension mismatch %d vs %d)", ErrShapeMismatch, s1, s2, dim1, dim2)
		}
	}

	return outShape, nil
}

// BroadcastTo returns a new Tensor view with targetShape by expanding singleton dimensions with 0 stride.
// The underlying storage is shared and not copied.
func (t *Tensor) BroadcastTo(targetShape []int) (*Tensor, error) {
	srcShape := t.shape
	srcStrides := t.strides

	if len(targetShape) < len(srcShape) {
		return nil, fmt.Errorf("%w: target shape rank %d cannot be smaller than tensor rank %d", ErrShapeMismatch, len(targetShape), len(srcShape))
	}

	for _, d := range targetShape {
		if d < 0 {
			return nil, fmt.Errorf("%w: target dimension %d cannot be negative", ErrInvalidShape, d)
		}
	}

	newStrides := make([]int, len(targetShape))
	diff := len(targetShape) - len(srcShape)

	// Preceding dimensions added to the left have stride 0
	for i := 0; i < diff; i++ {
		newStrides[i] = 0
	}

	// Trailing dimensions
	for i := 0; i < len(srcShape); i++ {
		targetDim := targetShape[diff+i]
		srcDim := srcShape[i]
		if srcDim == targetDim {
			newStrides[diff+i] = srcStrides[i]
		} else if srcDim == 1 {
			newStrides[diff+i] = 0
		} else {
			return nil, fmt.Errorf("%w: cannot broadcast shape %v to %v", ErrShapeMismatch, srcShape, targetShape)
		}
	}

	return NewWithStorage(targetShape, newStrides, t.offset, t.dtype, t.storage)
}

// Transpose permutes the axes of the tensor.
// If no dims are provided and rank is 2, it performs a 2D matrix transpose (swaps axes 0 and 1).
// If dims are provided, it must be a valid permutation of [0, 1, ..., Rank()-1].
func (t *Tensor) Transpose(dims ...int) (*Tensor, error) {
	rank := len(t.shape)
	if len(dims) == 0 {
		if rank != 2 {
			return nil, fmt.Errorf("%w: default transpose without dimensions requires 2D tensor, got rank %d", ErrShapeMismatch, rank)
		}
		dims = []int{1, 0}
	}

	if len(dims) != rank {
		return nil, fmt.Errorf("%w: expected %d permutation dimensions for rank %d, got %d", ErrShapeMismatch, rank, rank, len(dims))
	}

	seen := make([]bool, rank)
	newShape := make([]int, rank)
	newStrides := make([]int, rank)

	for i, d := range dims {
		if d < 0 || d >= rank {
			return nil, fmt.Errorf("%w: dimension %d out of bounds for rank %d", ErrIndexOutOfBounds, d, rank)
		}
		if seen[d] {
			return nil, fmt.Errorf("%w: duplicate dimension %d in transpose permutation", ErrInvalidShape, d)
		}
		seen[d] = true
		newShape[i] = t.shape[d]
		newStrides[i] = t.strides[d]
	}

	return NewWithStorage(newShape, newStrides, t.offset, t.dtype, t.storage)
}

// Squeeze returns a tensor with all specified dimensions of length 1 removed.
// If no dims are specified, all dimensions of size 1 are removed.
func (t *Tensor) Squeeze(dims ...int) (*Tensor, error) {
	rank := len(t.shape)
	toRemove := make(map[int]bool)

	if len(dims) == 0 {
		for i, dim := range t.shape {
			if dim == 1 {
				toRemove[i] = true
			}
		}
	} else {
		for _, d := range dims {
			origD := d
			if d < 0 {
				d += rank
			}
			if d < 0 || d >= rank {
				return nil, fmt.Errorf("%w: dimension %d out of bounds for rank %d", ErrIndexOutOfBounds, origD, rank)
			}
			if t.shape[d] != 1 {
				return nil, fmt.Errorf("%w: cannot squeeze dimension %d with size %d != 1", ErrInvalidShape, origD, t.shape[d])
			}
			toRemove[d] = true
		}
	}

	newShape := make([]int, 0, rank-len(toRemove))
	newStrides := make([]int, 0, rank-len(toRemove))

	for i := 0; i < rank; i++ {
		if !toRemove[i] {
			newShape = append(newShape, t.shape[i])
			newStrides = append(newStrides, t.strides[i])
		}
	}

	return NewWithStorage(newShape, newStrides, t.offset, t.dtype, t.storage)
}

// Unsqueeze returns a new tensor with a dimension of size 1 inserted at index dim.
func (t *Tensor) Unsqueeze(dim int) (*Tensor, error) {
	rank := len(t.shape)
	origDim := dim
	if dim < 0 {
		dim += rank + 1
	}
	if dim < 0 || dim > rank {
		return nil, fmt.Errorf("%w: dimension %d out of range for inserting into rank %d", ErrIndexOutOfBounds, origDim, rank)
	}

	newShape := make([]int, 0, rank+1)
	newStrides := make([]int, 0, rank+1)

	// Determine stride for the inserted dimension:
	// If dim < rank, its stride can match the stride of the dimension at that index.
	// If dim == rank, stride is 1.
	insertedStride := 1
	if dim < rank {
		insertedStride = t.shape[dim] * t.strides[dim]
	}

	for i := 0; i < rank; i++ {
		if i == dim {
			newShape = append(newShape, 1)
			newStrides = append(newStrides, insertedStride)
		}
		newShape = append(newShape, t.shape[i])
		newStrides = append(newStrides, t.strides[i])
	}
	if dim == rank {
		newShape = append(newShape, 1)
		newStrides = append(newStrides, insertedStride)
	}

	return NewWithStorage(newShape, newStrides, t.offset, t.dtype, t.storage)
}

// Flatten flattens dimensions between startDim and endDim (inclusive) into a single dimension.
func (t *Tensor) Flatten(startDim, endDim int) (*Tensor, error) {
	rank := len(t.shape)
	if rank == 0 {
		return t.Clone()
	}
	if startDim < 0 {
		startDim += rank
	}
	if endDim < 0 {
		endDim += rank
	}
	if startDim < 0 || startDim >= rank || endDim < 0 || endDim >= rank || startDim > endDim {
		return nil, fmt.Errorf("%w: invalid flatten range [%d, %d] for rank %d", ErrIndexOutOfBounds, startDim, endDim, rank)
	}

	if startDim == endDim {
		return t.Clone()
	}

	// Calculate product of dimensions to flatten
	flattenedSize := 1
	for i := startDim; i <= endDim; i++ {
		flattenedSize *= t.shape[i]
	}

	newShape := make([]int, 0, rank-(endDim-startDim))
	for i := 0; i < startDim; i++ {
		newShape = append(newShape, t.shape[i])
	}
	newShape = append(newShape, flattenedSize)
	for i := endDim + 1; i < rank; i++ {
		newShape = append(newShape, t.shape[i])
	}

	if t.IsContiguous() {
		return t.Reshape(newShape)
	}

	// If non-contiguous, clone first
	cloned, err := t.Clone()
	if err != nil {
		return nil, err
	}
	return cloned.Reshape(newShape)
}
