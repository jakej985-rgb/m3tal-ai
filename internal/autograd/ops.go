package autograd

import (
	"fmt"
	"reflect"

	"m3tal-ai/internal/tensor"
)

// Add returns the element-wise addition of a and b, supporting broadcasting.
func Add(a, b *Variable) (*Variable, error) {
	if a == nil || b == nil || a.Data == nil || b.Data == nil {
		return nil, ErrNilTensor
	}
	if a.DType() != b.DType() {
		return nil, fmt.Errorf("%w: cannot add tensors with different dtypes %s and %s", ErrUnsupportedOp, a.DType(), b.DType())
	}

	outShape, err := broadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnsupportedOp, err)
	}

	outTensor, err := tensor.New(outShape, a.DType())
	if err != nil {
		return nil, err
	}

	numel := outTensor.Numel()
	dtype := a.DType()
	switch dtype {
	case tensor.Float32:
		for i := 0; i < numel; i++ {
			outIdx, err := flatToIndices(i, outShape)
			if err != nil {
				return nil, err
			}
			aIdx := broadcastIndex(outIdx, a.Shape())
			bIdx := broadcastIndex(outIdx, b.Shape())
			va, err := a.Data.Float32At(aIdx...)
			if err != nil {
				return nil, err
			}
			vb, err := b.Data.Float32At(bIdx...)
			if err != nil {
				return nil, err
			}
			if err := outTensor.SetFloat32At(va+vb, outIdx...); err != nil {
				return nil, err
			}
		}
	case tensor.Float64:
		for i := 0; i < numel; i++ {
			outIdx, err := flatToIndices(i, outShape)
			if err != nil {
				return nil, err
			}
			aIdx := broadcastIndex(outIdx, a.Shape())
			bIdx := broadcastIndex(outIdx, b.Shape())
			va, err := a.Data.Float64At(aIdx...)
			if err != nil {
				return nil, err
			}
			vb, err := b.Data.Float64At(bIdx...)
			if err != nil {
				return nil, err
			}
			if err := outTensor.SetFloat64At(va+vb, outIdx...); err != nil {
				return nil, err
			}
		}
	default:
		return nil, fmt.Errorf("%w: Add for dtype %s", ErrUnsupportedOp, dtype)
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
		if a.RequiresGrad {
			gradA, err := reduceToShape(out.Grad, a.Shape())
			if err != nil {
				return err
			}
			if err := a.AccGrad(gradA); err != nil {
				return err
			}
		}
		if b.RequiresGrad {
			gradB, err := reduceToShape(out.Grad, b.Shape())
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

// Sub returns the element-wise subtraction of a and b (a - b).
func Sub(a, b *Variable) (*Variable, error) {
	if a == nil || b == nil || a.Data == nil || b.Data == nil {
		return nil, ErrNilTensor
	}
	if a.DType() != b.DType() {
		return nil, fmt.Errorf("%w: cannot subtract tensors with different dtypes %s and %s", ErrUnsupportedOp, a.DType(), b.DType())
	}

	outShape, err := broadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnsupportedOp, err)
	}

	outTensor, err := tensor.New(outShape, a.DType())
	if err != nil {
		return nil, err
	}

	numel := outTensor.Numel()
	dtype := a.DType()
	switch dtype {
	case tensor.Float32:
		for i := 0; i < numel; i++ {
			outIdx, err := flatToIndices(i, outShape)
			if err != nil {
				return nil, err
			}
			aIdx := broadcastIndex(outIdx, a.Shape())
			bIdx := broadcastIndex(outIdx, b.Shape())
			va, err := a.Data.Float32At(aIdx...)
			if err != nil {
				return nil, err
			}
			vb, err := b.Data.Float32At(bIdx...)
			if err != nil {
				return nil, err
			}
			if err := outTensor.SetFloat32At(va-vb, outIdx...); err != nil {
				return nil, err
			}
		}
	case tensor.Float64:
		for i := 0; i < numel; i++ {
			outIdx, err := flatToIndices(i, outShape)
			if err != nil {
				return nil, err
			}
			aIdx := broadcastIndex(outIdx, a.Shape())
			bIdx := broadcastIndex(outIdx, b.Shape())
			va, err := a.Data.Float64At(aIdx...)
			if err != nil {
				return nil, err
			}
			vb, err := b.Data.Float64At(bIdx...)
			if err != nil {
				return nil, err
			}
			if err := outTensor.SetFloat64At(va-vb, outIdx...); err != nil {
				return nil, err
			}
		}
	default:
		return nil, fmt.Errorf("%w: Sub for dtype %s", ErrUnsupportedOp, dtype)
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
		if a.RequiresGrad {
			gradA, err := reduceToShape(out.Grad, a.Shape())
			if err != nil {
				return err
			}
			if err := a.AccGrad(gradA); err != nil {
				return err
			}
		}
		if b.RequiresGrad {
			negGrad, err := negTensor(out.Grad)
			if err != nil {
				return err
			}
			gradB, err := reduceToShape(negGrad, b.Shape())
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

// Mul returns the element-wise multiplication of a and b.
func Mul(a, b *Variable) (*Variable, error) {
	if a == nil || b == nil || a.Data == nil || b.Data == nil {
		return nil, ErrNilTensor
	}
	if a.DType() != b.DType() {
		return nil, fmt.Errorf("%w: cannot multiply tensors with different dtypes %s and %s", ErrUnsupportedOp, a.DType(), b.DType())
	}

	outShape, err := broadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnsupportedOp, err)
	}

	outTensor, err := tensor.New(outShape, a.DType())
	if err != nil {
		return nil, err
	}

	numel := outTensor.Numel()
	dtype := a.DType()
	switch dtype {
	case tensor.Float32:
		for i := 0; i < numel; i++ {
			outIdx, err := flatToIndices(i, outShape)
			if err != nil {
				return nil, err
			}
			aIdx := broadcastIndex(outIdx, a.Shape())
			bIdx := broadcastIndex(outIdx, b.Shape())
			va, err := a.Data.Float32At(aIdx...)
			if err != nil {
				return nil, err
			}
			vb, err := b.Data.Float32At(bIdx...)
			if err != nil {
				return nil, err
			}
			if err := outTensor.SetFloat32At(va*vb, outIdx...); err != nil {
				return nil, err
			}
		}
	case tensor.Float64:
		for i := 0; i < numel; i++ {
			outIdx, err := flatToIndices(i, outShape)
			if err != nil {
				return nil, err
			}
			aIdx := broadcastIndex(outIdx, a.Shape())
			bIdx := broadcastIndex(outIdx, b.Shape())
			va, err := a.Data.Float64At(aIdx...)
			if err != nil {
				return nil, err
			}
			vb, err := b.Data.Float64At(bIdx...)
			if err != nil {
				return nil, err
			}
			if err := outTensor.SetFloat64At(va*vb, outIdx...); err != nil {
				return nil, err
			}
		}
	default:
		return nil, fmt.Errorf("%w: Mul for dtype %s", ErrUnsupportedOp, dtype)
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
		if a.RequiresGrad {
			// grad * b
			mult, err := elemwiseMul(out.Grad, b.Data)
			if err != nil {
				return err
			}
			gradA, err := reduceToShape(mult, a.Shape())
			if err != nil {
				return err
			}
			if err := a.AccGrad(gradA); err != nil {
				return err
			}
		}
		if b.RequiresGrad {
			// grad * a
			mult, err := elemwiseMul(out.Grad, a.Data)
			if err != nil {
				return err
			}
			gradB, err := reduceToShape(mult, b.Shape())
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

// Div returns the element-wise division of a by b (a / b).
func Div(a, b *Variable) (*Variable, error) {
	if a == nil || b == nil || a.Data == nil || b.Data == nil {
		return nil, ErrNilTensor
	}
	if a.DType() != b.DType() {
		return nil, fmt.Errorf("%w: cannot divide tensors with different dtypes %s and %s", ErrUnsupportedOp, a.DType(), b.DType())
	}

	outShape, err := broadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnsupportedOp, err)
	}

	outTensor, err := tensor.New(outShape, a.DType())
	if err != nil {
		return nil, err
	}

	numel := outTensor.Numel()
	dtype := a.DType()
	switch dtype {
	case tensor.Float32:
		for i := 0; i < numel; i++ {
			outIdx, err := flatToIndices(i, outShape)
			if err != nil {
				return nil, err
			}
			aIdx := broadcastIndex(outIdx, a.Shape())
			bIdx := broadcastIndex(outIdx, b.Shape())
			va, err := a.Data.Float32At(aIdx...)
			if err != nil {
				return nil, err
			}
			vb, err := b.Data.Float32At(bIdx...)
			if err != nil {
				return nil, err
			}
			if vb == 0 {
				return nil, fmt.Errorf("%w: division by zero", ErrUnsupportedOp)
			}
			if err := outTensor.SetFloat32At(va/vb, outIdx...); err != nil {
				return nil, err
			}
		}
	case tensor.Float64:
		for i := 0; i < numel; i++ {
			outIdx, err := flatToIndices(i, outShape)
			if err != nil {
				return nil, err
			}
			aIdx := broadcastIndex(outIdx, a.Shape())
			bIdx := broadcastIndex(outIdx, b.Shape())
			va, err := a.Data.Float64At(aIdx...)
			if err != nil {
				return nil, err
			}
			vb, err := b.Data.Float64At(bIdx...)
			if err != nil {
				return nil, err
			}
			if vb == 0 {
				return nil, fmt.Errorf("%w: division by zero", ErrUnsupportedOp)
			}
			if err := outTensor.SetFloat64At(va/vb, outIdx...); err != nil {
				return nil, err
			}
		}
	default:
		return nil, fmt.Errorf("%w: Div for dtype %s", ErrUnsupportedOp, dtype)
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
		if a.RequiresGrad {
			// dL/da = grad / b
			divA, err := elemwiseDiv(out.Grad, b.Data)
			if err != nil {
				return err
			}
			gradA, err := reduceToShape(divA, a.Shape())
			if err != nil {
				return err
			}
			if err := a.AccGrad(gradA); err != nil {
				return err
			}
		}
		if b.RequiresGrad {
			// dL/db = -grad * a / (b^2)
			bSq, err := elemwiseMul(b.Data, b.Data)
			if err != nil {
				return err
			}
			gradA, err := elemwiseMul(out.Grad, a.Data)
			if err != nil {
				return err
			}
			ratio, err := elemwiseDiv(gradA, bSq)
			if err != nil {
				return err
			}
			negRatio, err := negTensor(ratio)
			if err != nil {
				return err
			}
			gradB, err := reduceToShape(negRatio, b.Shape())
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

// Scale multiplies all elements in a by scalar alpha.
func Scale(a *Variable, alpha float64) (*Variable, error) {
	if a == nil || a.Data == nil {
		return nil, ErrNilTensor
	}
	outTensor, err := tensor.New(a.Shape(), a.DType())
	if err != nil {
		return nil, err
	}
	numel := outTensor.Numel()
	switch a.DType() {
	case tensor.Float32:
		alpha32 := float32(alpha)
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, a.Shape())
			if err != nil {
				return nil, err
			}
			v, err := a.Data.Float32At(idx...)
			if err != nil {
				return nil, err
			}
			if err := outTensor.SetFloat32At(v*alpha32, idx...); err != nil {
				return nil, err
			}
		}
	case tensor.Float64:
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, a.Shape())
			if err != nil {
				return nil, err
			}
			v, err := a.Data.Float64At(idx...)
			if err != nil {
				return nil, err
			}
			if err := outTensor.SetFloat64At(v*alpha, idx...); err != nil {
				return nil, err
			}
		}
	default:
		return nil, fmt.Errorf("%w: Scale for dtype %s", ErrUnsupportedOp, a.DType())
	}

	out := &Variable{
		Data:         outTensor,
		RequiresGrad: a.RequiresGrad,
	}
	if a.RequiresGrad {
		out.parents = []*Variable{a}
		out.backward = func() error {
			if out.Grad == nil {
				return nil
			}
			scaledGrad, err := scaleTensor(out.Grad, alpha)
			if err != nil {
				return err
			}
			return a.AccGrad(scaledGrad)
		}
	}
	return out, nil
}

// AddScalar adds scalar alpha to all elements in a.
func AddScalar(a *Variable, alpha float64) (*Variable, error) {
	if a == nil || a.Data == nil {
		return nil, ErrNilTensor
	}
	outTensor, err := tensor.New(a.Shape(), a.DType())
	if err != nil {
		return nil, err
	}
	numel := outTensor.Numel()
	switch a.DType() {
	case tensor.Float32:
		alpha32 := float32(alpha)
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, a.Shape())
			if err != nil {
				return nil, err
			}
			v, err := a.Data.Float32At(idx...)
			if err != nil {
				return nil, err
			}
			if err := outTensor.SetFloat32At(v+alpha32, idx...); err != nil {
				return nil, err
			}
		}
	case tensor.Float64:
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, a.Shape())
			if err != nil {
				return nil, err
			}
			v, err := a.Data.Float64At(idx...)
			if err != nil {
				return nil, err
			}
			if err := outTensor.SetFloat64At(v+alpha, idx...); err != nil {
				return nil, err
			}
		}
	default:
		return nil, fmt.Errorf("%w: AddScalar for dtype %s", ErrUnsupportedOp, a.DType())
	}

	out := &Variable{
		Data:         outTensor,
		RequiresGrad: a.RequiresGrad,
	}
	if a.RequiresGrad {
		out.parents = []*Variable{a}
		out.backward = func() error {
			if out.Grad == nil {
				return nil
			}
			return a.AccGrad(out.Grad)
		}
	}
	return out, nil
}


// broadcastShapes computes standard NumPy-style broadcasted output shape.
func broadcastShapes(a, b []int) ([]int, error) {
	rankA := len(a)
	rankB := len(b)
	maxRank := rankA
	if rankB > maxRank {
		maxRank = rankB
	}
	out := make([]int, maxRank)
	for i := 0; i < maxRank; i++ {
		dimA := 1
		idxA := rankA - 1 - i
		if idxA >= 0 {
			dimA = a[idxA]
		}
		dimB := 1
		idxB := rankB - 1 - i
		if idxB >= 0 {
			dimB = b[idxB]
		}
		if dimA != dimB && dimA != 1 && dimB != 1 {
			return nil, fmt.Errorf("incompatible broadcast dimensions: %v and %v", a, b)
		}
		outDim := dimA
		if dimB > outDim {
			outDim = dimB
		}
		out[maxRank-1-i] = outDim
	}
	return out, nil
}

// broadcastIndex maps coordinates from the broadcasted shape back to an operand shape.
func broadcastIndex(outIdx []int, shape []int) []int {
	if len(shape) == 0 {
		return []int{}
	}
	rankDiff := len(outIdx) - len(shape)
	res := make([]int, len(shape))
	for i := 0; i < len(shape); i++ {
		dim := shape[i]
		if dim == 1 {
			res[i] = 0
		} else {
			res[i] = outIdx[i+rankDiff]
		}
	}
	return res
}

// reduceToShape sums elements along dimensions that were broadcasted to match targetShape.
func reduceToShape(t *tensor.Tensor, targetShape []int) (*tensor.Tensor, error) {
	if reflect.DeepEqual(t.Shape(), targetShape) {
		return t.Clone()
	}
	out, err := tensor.New(targetShape, t.DType())
	if err != nil {
		return nil, err
	}
	numel := t.Numel()
	switch t.DType() {
	case tensor.Float32:
		for i := 0; i < numel; i++ {
			srcIdx, err := flatToIndices(i, t.Shape())
			if err != nil {
				return nil, err
			}
			dstIdx := broadcastIndex(srcIdx, targetShape)
			val, err := t.Float32At(srcIdx...)
			if err != nil {
				return nil, err
			}
			curr, err := out.Float32At(dstIdx...)
			if err != nil {
				return nil, err
			}
			if err := out.SetFloat32At(curr+val, dstIdx...); err != nil {
				return nil, err
			}
		}
	case tensor.Float64:
		for i := 0; i < numel; i++ {
			srcIdx, err := flatToIndices(i, t.Shape())
			if err != nil {
				return nil, err
			}
			dstIdx := broadcastIndex(srcIdx, targetShape)
			val, err := t.Float64At(srcIdx...)
			if err != nil {
				return nil, err
			}
			curr, err := out.Float64At(dstIdx...)
			if err != nil {
				return nil, err
			}
			if err := out.SetFloat64At(curr+val, dstIdx...); err != nil {
				return nil, err
			}
		}
	default:
		return nil, fmt.Errorf("%w: reduceToShape for %s", ErrUnsupportedOp, t.DType())
	}
	return out, nil
}

// elemwiseMul multiplies two tensors with broadcasted or matching shapes elementwise.
func elemwiseMul(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outShape, err := broadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}
	out, err := tensor.New(outShape, a.DType())
	if err != nil {
		return nil, err
	}
	numel := out.Numel()
	switch a.DType() {
	case tensor.Float32:
		for i := 0; i < numel; i++ {
			outIdx, err := flatToIndices(i, outShape)
			if err != nil {
				return nil, err
			}
			aIdx := broadcastIndex(outIdx, a.Shape())
			bIdx := broadcastIndex(outIdx, b.Shape())
			va, err := a.Float32At(aIdx...)
			if err != nil {
				return nil, err
			}
			vb, err := b.Float32At(bIdx...)
			if err != nil {
				return nil, err
			}
			if err := out.SetFloat32At(va*vb, outIdx...); err != nil {
				return nil, err
			}
		}
	case tensor.Float64:
		for i := 0; i < numel; i++ {
			outIdx, err := flatToIndices(i, outShape)
			if err != nil {
				return nil, err
			}
			aIdx := broadcastIndex(outIdx, a.Shape())
			bIdx := broadcastIndex(outIdx, b.Shape())
			va, err := a.Float64At(aIdx...)
			if err != nil {
				return nil, err
			}
			vb, err := b.Float64At(bIdx...)
			if err != nil {
				return nil, err
			}
			if err := out.SetFloat64At(va*vb, outIdx...); err != nil {
				return nil, err
			}
		}
	default:
		return nil, fmt.Errorf("%w: elemwiseMul for %s", ErrUnsupportedOp, a.DType())
	}
	return out, nil
}

// elemwiseDiv divides tensor a by tensor b elementwise.
func elemwiseDiv(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outShape, err := broadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}
	out, err := tensor.New(outShape, a.DType())
	if err != nil {
		return nil, err
	}
	numel := out.Numel()
	switch a.DType() {
	case tensor.Float32:
		for i := 0; i < numel; i++ {
			outIdx, err := flatToIndices(i, outShape)
			if err != nil {
				return nil, err
			}
			aIdx := broadcastIndex(outIdx, a.Shape())
			bIdx := broadcastIndex(outIdx, b.Shape())
			va, err := a.Float32At(aIdx...)
			if err != nil {
				return nil, err
			}
			vb, err := b.Float32At(bIdx...)
			if err != nil {
				return nil, err
			}
			if vb == 0 {
				return nil, fmt.Errorf("%w: division by zero", ErrUnsupportedOp)
			}
			if err := out.SetFloat32At(va/vb, outIdx...); err != nil {
				return nil, err
			}
		}
	case tensor.Float64:
		for i := 0; i < numel; i++ {
			outIdx, err := flatToIndices(i, outShape)
			if err != nil {
				return nil, err
			}
			aIdx := broadcastIndex(outIdx, a.Shape())
			bIdx := broadcastIndex(outIdx, b.Shape())
			va, err := a.Float64At(aIdx...)
			if err != nil {
				return nil, err
			}
			vb, err := b.Float64At(bIdx...)
			if err != nil {
				return nil, err
			}
			if vb == 0 {
				return nil, fmt.Errorf("%w: division by zero", ErrUnsupportedOp)
			}
			if err := out.SetFloat64At(va/vb, outIdx...); err != nil {
				return nil, err
			}
		}
	default:
		return nil, fmt.Errorf("%w: elemwiseDiv for %s", ErrUnsupportedOp, a.DType())
	}
	return out, nil
}

// scaleTensor multiplies all elements of t by alpha.
func scaleTensor(t *tensor.Tensor, alpha float64) (*tensor.Tensor, error) {
	out, err := tensor.New(t.Shape(), t.DType())
	if err != nil {
		return nil, err
	}
	numel := t.Numel()
	switch t.DType() {
	case tensor.Float32:
		alpha32 := float32(alpha)
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, t.Shape())
			if err != nil {
				return nil, err
			}
			v, err := t.Float32At(idx...)
			if err != nil {
				return nil, err
			}
			if err := out.SetFloat32At(v*alpha32, idx...); err != nil {
				return nil, err
			}
		}
	case tensor.Float64:
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, t.Shape())
			if err != nil {
				return nil, err
			}
			v, err := t.Float64At(idx...)
			if err != nil {
				return nil, err
			}
			if err := out.SetFloat64At(v*alpha, idx...); err != nil {
				return nil, err
			}
		}
	default:
		return nil, fmt.Errorf("%w: scaleTensor for %s", ErrUnsupportedOp, t.DType())
	}
	return out, nil
}

// negTensor returns elementwise -t.
func negTensor(t *tensor.Tensor) (*tensor.Tensor, error) {
	return scaleTensor(t, -1.0)
}
