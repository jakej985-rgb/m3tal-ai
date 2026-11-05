// Package autograd implements scalar and tensor automatic-differentiation engines.
package autograd

import (
	"errors"
	"fmt"
	"reflect"

	"m3tal-llm/internal/tensor"
)

var (
	// ErrNilTensor is returned when an operation receives a nil Tensor.
	ErrNilTensor = errors.New("autograd: nil tensor")
	// ErrGradUndefined is returned when backward is called without seed gradient on non-scalar.
	ErrGradUndefined = errors.New("autograd: grad can only be implicitly created for scalar outputs")
	// ErrRequiresGradMismatch is returned when gradient accumulation encounters uninitialized or mismatched grad.
	ErrRequiresGradMismatch = errors.New("autograd: gradient accumulation requires grad buffer with matching shape and dtype")
	// ErrCycleDetected is returned if a cycle is detected in the computation graph.
	ErrCycleDetected = errors.New("autograd: cycle detected in computation graph")
	// ErrUnsupportedOp is returned when an operation or gradient is not supported for the given operands.
	ErrUnsupportedOp = errors.New("autograd: unsupported operation")
)

// Variable represents a node in a tensor computation graph.
type Variable struct {
	// Data holds the forward tensor value.
	Data *tensor.Tensor
	// Grad holds accumulated reverse-mode gradients.
	Grad *tensor.Tensor
	// RequiresGrad specifies whether gradients should be tracked and accumulated.
	RequiresGrad bool

	parents  []*Variable
	backward func() error
}

// NewVariable wraps a tensor.Tensor into a Variable node.
func NewVariable(data *tensor.Tensor, requiresGrad bool) (*Variable, error) {
	if data == nil {
		return nil, ErrNilTensor
	}
	return &Variable{
		Data:         data,
		RequiresGrad: requiresGrad,
	}, nil
}

// Shape returns the shape of the underlying tensor data.
func (v *Variable) Shape() []int {
	if v.Data == nil {
		return nil
	}
	return v.Data.Shape()
}

// DType returns the element type of the underlying tensor data.
func (v *Variable) DType() tensor.DType {
	if v.Data == nil {
		return tensor.Invalid
	}
	return v.Data.DType()
}

// AccGrad accumulates incoming gradient into v.Grad.
// If v.Grad is nil, a cloned tensor of grad is created.
// If v.Grad already exists, element-wise addition is performed.
func (v *Variable) AccGrad(incoming *tensor.Tensor) error {
	if !v.RequiresGrad {
		return nil
	}
	if incoming == nil {
		return ErrNilTensor
	}
	if !reflect.DeepEqual(v.Shape(), incoming.Shape()) {
		return fmt.Errorf("%w: target shape %v vs incoming grad shape %v", ErrRequiresGradMismatch, v.Shape(), incoming.Shape())
	}
	if v.DType() != incoming.DType() {
		return fmt.Errorf("%w: target dtype %v vs incoming grad dtype %v", ErrRequiresGradMismatch, v.DType(), incoming.DType())
	}

	if v.Grad == nil {
		cloned, err := incoming.Clone()
		if err != nil {
			return err
		}
		v.Grad = cloned
		return nil
	}

	// In-place accumulation into v.Grad
	numel := v.Grad.Numel()
	switch v.DType() {
	case tensor.Float32:
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, v.Shape())
			if err != nil {
				return err
			}
			valCurrent, err := v.Grad.Float32At(idx...)
			if err != nil {
				return err
			}
			valInc, err := incoming.Float32At(idx...)
			if err != nil {
				return err
			}
			if err := v.Grad.SetFloat32At(valCurrent+valInc, idx...); err != nil {
				return err
			}
		}
	case tensor.Float64:
		for i := 0; i < numel; i++ {
			idx, err := flatToIndices(i, v.Shape())
			if err != nil {
				return err
			}
			valCurrent, err := v.Grad.Float64At(idx...)
			if err != nil {
				return err
			}
			valInc, err := incoming.Float64At(idx...)
			if err != nil {
				return err
			}
			if err := v.Grad.SetFloat64At(valCurrent+valInc, idx...); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("%w: gradient accumulation for dtype %s", ErrUnsupportedOp, v.DType())
	}
	return nil
}

// ZeroGrad zeroes out the gradient buffer if it exists.
func (v *Variable) ZeroGrad() {
	if v.Grad == nil {
		return
	}
	numel := v.Grad.Numel()
	switch v.DType() {
	case tensor.Float32:
		for i := 0; i < numel; i++ {
			idx, _ := flatToIndices(i, v.Shape())
			_ = v.Grad.SetFloat32At(0, idx...)
		}
	case tensor.Float64:
		for i := 0; i < numel; i++ {
			idx, _ := flatToIndices(i, v.Shape())
			_ = v.Grad.SetFloat64At(0, idx...)
		}
	}
}

// Detach creates a new leaf Variable that shares the underlying Tensor data
// but has no history and does not track gradients by default.
func (v *Variable) Detach() *Variable {
	return &Variable{
		Data:         v.Data,
		RequiresGrad: false,
	}
}

// ReleaseGraph clears parent links and backward closures across all upstream nodes,
// freeing intermediate references held in closures.
func (v *Variable) ReleaseGraph() {
	order, err := topologicalSort(v)
	if err != nil {
		// Even if sort fails, release direct links
		v.parents = nil
		v.backward = nil
		return
	}
	for _, node := range order {
		node.parents = nil
		node.backward = nil
	}
}

// Backward executes reverse-mode automatic differentiation starting from v.
// If seedGrad is provided, it is used as the incoming gradient on v;
// otherwise, v must be a scalar (numel == 1), and a gradient of 1 is seeded.
func (v *Variable) Backward(seedGrad ...*tensor.Tensor) error {
	var rootGrad *tensor.Tensor
	if len(seedGrad) > 0 && seedGrad[0] != nil {
		rootGrad = seedGrad[0]
		if !reflect.DeepEqual(rootGrad.Shape(), v.Shape()) {
			return fmt.Errorf("%w: root grad shape %v does not match variable shape %v", ErrRequiresGradMismatch, rootGrad.Shape(), v.Shape())
		}
		if rootGrad.DType() != v.DType() {
			return fmt.Errorf("%w: root grad dtype %v does not match variable dtype %v", ErrRequiresGradMismatch, rootGrad.DType(), v.DType())
		}
	} else {
		if v.Data.Numel() != 1 {
			return fmt.Errorf("%w: tensor has shape %v (numel %d)", ErrGradUndefined, v.Shape(), v.Data.Numel())
		}
		var err error
		rootGrad, err = tensor.New(v.Shape(), v.DType())
		if err != nil {
			return err
		}
		switch v.DType() {
		case tensor.Float32:
			err = rootGrad.SetFloat32At(1.0, make([]int, v.Data.Rank())...)
		case tensor.Float64:
			err = rootGrad.SetFloat64At(1.0, make([]int, v.Data.Rank())...)
		default:
			err = fmt.Errorf("%w: seed grad for %s", ErrUnsupportedOp, v.DType())
		}
		if err != nil {
			return err
		}
	}

	if err := v.AccGrad(rootGrad); err != nil {
		return err
	}

	order, err := topologicalSort(v)
	if err != nil {
		return err
	}

	for i := len(order) - 1; i >= 0; i-- {
		node := order[i]
		if node.backward != nil {
			if err := node.backward(); err != nil {
				return err
			}
		}
	}
	return nil
}

// topologicalSort computes reverse-post-order nodes reachable from root.
func topologicalSort(root *Variable) ([]*Variable, error) {
	order := make([]*Variable, 0)
	state := make(map[*Variable]int) // 0: unvisited, 1: visiting, 2: visited

	var visit func(*Variable) error
	visit = func(node *Variable) error {
		if node == nil {
			return nil
		}
		st := state[node]
		if st == 1 {
			return ErrCycleDetected
		}
		if st == 2 {
			return nil
		}
		state[node] = 1
		for _, p := range node.parents {
			if err := visit(p); err != nil {
				return err
			}
		}
		state[node] = 2
		order = append(order, node)
		return nil
	}

	if err := visit(root); err != nil {
		return nil, err
	}
	return order, nil
}

// NewOpVariable creates a new Variable resulting from an operation, registering parents and backward function.
func NewOpVariable(data *tensor.Tensor, parents []*Variable, backward func() error) *Variable {
	requiresGrad := false
	validParents := make([]*Variable, 0, len(parents))
	for _, p := range parents {
		if p != nil {
			if p.RequiresGrad {
				requiresGrad = true
			}
			validParents = append(validParents, p)
		}
	}
	v := &Variable{
		Data:         data,
		RequiresGrad: requiresGrad,
		parents:      validParents,
		backward:     backward,
	}
	return v
}

// SetBackward configures the computation graph topology and reverse-mode gradient closure for custom ops.
func (v *Variable) SetBackward(parents []*Variable, fn func() error) {
	v.parents = parents
	v.backward = fn
}

// FlatToIndices converts a flat linear element index into N-D coordinate indices.
func FlatToIndices(flat int, shape []int) ([]int, error) {
	return flatToIndices(flat, shape)
}

// OnBackward attaches an additional callback to be run during the backward pass of v.
func (v *Variable) OnBackward(cb func() error) {
	orig := v.backward
	v.backward = func() error {
		if orig != nil {
			if err := orig(); err != nil {
				return err
			}
		}
		if cb != nil {
			return cb()
		}
		return nil
	}
}

// FlatToIndices converts a flat linear element index into N-D coordinate indices.
func FlatToIndices(flat int, shape []int) ([]int, error) {
	return flatToIndices(flat, shape)
}

// flatToIndices converts a flat linear element index into N-D coordinate indices.
func flatToIndices(flat int, shape []int) ([]int, error) {
	if len(shape) == 0 {
		return []int{}, nil
	}
	idx := make([]int, len(shape))
	curr := flat
	for i := len(shape) - 1; i >= 0; i-- {
		dim := shape[i]
		if dim <= 0 {
			return nil, tensor.ErrInvalidShape
		}
		idx[i] = curr % dim
		curr /= dim
	}
	return idx, nil
}
