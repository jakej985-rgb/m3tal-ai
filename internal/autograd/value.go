// Package autograd implements a small scalar automatic-differentiation engine.
package autograd

import "math"

// Value is a scalar in a computation graph. Grad is the derivative of the
// final output with respect to this value after Backward has been called.
type Value struct {
	Data float64
	Grad float64

	parents  []*Value
	backward func()
}

// New creates a leaf value that does not require gradients from earlier ops.
func New(data float64) *Value {
	return &Value{Data: data}
}

// Add returns v + other.
func (v *Value) Add(other *Value) *Value {
	out := &Value{Data: v.Data + other.Data, parents: []*Value{v, other}}
	out.backward = func() {
		v.Grad += out.Grad
		other.Grad += out.Grad
	}
	return out
}

// Mul returns v * other.
func (v *Value) Mul(other *Value) *Value {
	out := &Value{Data: v.Data * other.Data, parents: []*Value{v, other}}
	out.backward = func() {
		v.Grad += other.Data * out.Grad
		other.Grad += v.Data * out.Grad
	}
	return out
}

// Tanh returns tanh(v).
func (v *Value) Tanh() *Value {
	out := &Value{Data: math.Tanh(v.Data), parents: []*Value{v}}
	out.backward = func() {
		v.Grad += (1 - out.Data*out.Data) * out.Grad
	}
	return out
}

// Backward computes gradients for all values that contributed to v.
func (v *Value) Backward() {
	order := make([]*Value, 0)
	seen := make(map[*Value]bool)
	var visit func(*Value)
	visit = func(node *Value) {
		if seen[node] {
			return
		}
		seen[node] = true
		for _, parent := range node.parents {
			visit(parent)
		}
		order = append(order, node)
	}
	visit(v)

	v.Grad = 1
	for i := len(order) - 1; i >= 0; i-- {
		if order[i].backward != nil {
			order[i].backward()
		}
	}
}

// ZeroGrad clears gradients on a set of values, usually model parameters.
func ZeroGrad(values []*Value) {
	for _, value := range values {
		value.Grad = 0
	}
}

// Step updates leaf values with a simple gradient-descent step.
func Step(values []*Value, learningRate float64) {
	for _, value := range values {
		value.Data -= learningRate * value.Grad
	}
}
