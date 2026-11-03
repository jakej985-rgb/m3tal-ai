package tensor

import "fmt"

// Matrix is a dense row-major two-dimensional collection of float64 values.
type Matrix struct {
	rows int
	cols int
	data []float64
}

// NewMatrix creates a matrix and copies values so callers cannot mutate its
// storage accidentally. Values must contain rows*cols elements.
func NewMatrix(rows, cols int, values []float64) (Matrix, error) {
	if rows < 0 || cols < 0 || len(values) != rows*cols {
		return Matrix{}, fmt.Errorf("invalid matrix shape %dx%d for %d values", rows, cols, len(values))
	}
	data := append([]float64(nil), values...)
	return Matrix{rows: rows, cols: cols, data: data}, nil
}

// Rows returns the number of rows.
func (m Matrix) Rows() int { return m.rows }

// Cols returns the number of columns.
func (m Matrix) Cols() int { return m.cols }

// SelectRows returns a copy of the requested rows in the given order.
func (m Matrix) SelectRows(indices []int) (Matrix, error) {
	values := make([]float64, len(indices)*m.cols)
	for outRow, row := range indices {
		if row < 0 || row >= m.rows {
			return Matrix{}, fmt.Errorf("row index %d out of range for %d rows", row, m.rows)
		}
		copy(values[outRow*m.cols:(outRow+1)*m.cols], m.data[row*m.cols:(row+1)*m.cols])
	}
	return Matrix{rows: len(indices), cols: m.cols, data: values}, nil
}

// At returns the value at row, col. It panics when the indices are out of range.
func (m Matrix) At(row, col int) float64 {
	if row < 0 || row >= m.rows || col < 0 || col >= m.cols {
		panic("matrix index out of range")
	}
	return m.data[row*m.cols+col]
}

// Transpose returns a new matrix with rows and columns exchanged.
func (m Matrix) Transpose() Matrix {
	data := make([]float64, len(m.data))
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			data[col*m.rows+row] = m.data[row*m.cols+col]
		}
	}
	return Matrix{rows: m.cols, cols: m.rows, data: data}
}

// MatMul returns the matrix product m * other.
func (m Matrix) MatMul(other Matrix) (Matrix, error) {
	if m.cols != other.rows {
		return Matrix{}, fmt.Errorf("matrix shape mismatch: %dx%d times %dx%d", m.rows, m.cols, other.rows, other.cols)
	}
	data := make([]float64, m.rows*other.cols)
	for row := 0; row < m.rows; row++ {
		for col := 0; col < other.cols; col++ {
			for k := 0; k < m.cols; k++ {
				data[row*other.cols+col] += m.data[row*m.cols+k] * other.data[k*other.cols+col]
			}
		}
	}
	return Matrix{rows: m.rows, cols: other.cols, data: data}, nil
}

// MulVec returns m * v.
func (m Matrix) MulVec(v Vector) (Vector, error) {
	if m.cols != len(v) {
		return nil, fmt.Errorf("matrix/vector shape mismatch: %dx%d times vector of size %d", m.rows, m.cols, len(v))
	}
	out := make(Vector, m.rows)
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			out[row] += m.data[row*m.cols+col] * v[col]
		}
	}
	return out, nil
}
