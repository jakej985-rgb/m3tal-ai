// Package tensor provides small dense vector and matrix operations.
package tensor

import "fmt"

// Vector is a dense one-dimensional collection of float64 values.
type Vector []float64

// Add returns the element-wise sum of v and other.
func (v Vector) Add(other Vector) (Vector, error) {
	if len(v) != len(other) {
		return nil, fmt.Errorf("vector size mismatch: %d and %d", len(v), len(other))
	}
	out := make(Vector, len(v))
	for i := range v {
		out[i] = v[i] + other[i]
	}
	return out, nil
}

// Dot returns the dot product of v and other.
func (v Vector) Dot(other Vector) (float64, error) {
	if len(v) != len(other) {
		return 0, fmt.Errorf("vector size mismatch: %d and %d", len(v), len(other))
	}
	var sum float64
	for i := range v {
		sum += v[i] * other[i]
	}
	return sum, nil
}
