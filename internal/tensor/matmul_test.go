package tensor

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"
)

func TestMatMul2DAndParity(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	testCases := []struct {
		name    string
		M, K, N int
	}{
		{"1x1", 1, 1, 1},
		{"1x5x1", 1, 5, 1},
		{"SquareSmall", 4, 4, 4},
		{"TallSkinny", 65, 10, 8},
		{"ShortWide", 8, 10, 65},
		{"CrossTile65", 65, 65, 65},
		{"LargePrime", 73, 97, 83},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			aData := make([]float32, tc.M*tc.K)
			for i := range aData {
				aData[i] = rng.Float32()*2 - 1
			}
			bData := make([]float32, tc.K*tc.N)
			for i := range bData {
				bData[i] = rng.Float32()*2 - 1
			}

			a, err := FromFloat32([]int{tc.M, tc.K}, aData)
			if err != nil {
				t.Fatal(err)
			}
			b, err := FromFloat32([]int{tc.K, tc.N}, bData)
			if err != nil {
				t.Fatal(err)
			}

			// Blocked MatMul
			actual, err := MatMul(a, b)
			if err != nil {
				t.Fatalf("MatMul failed: %v", err)
			}

			// Reference naive GEMM
			expected, err := refMatMul(a, b)
			if err != nil {
				t.Fatalf("refMatMul failed: %v", err)
			}

			if !reflect.DeepEqual(actual.Shape(), expected.Shape()) {
				t.Fatalf("shape mismatch: got %v, want %v", actual.Shape(), expected.Shape())
			}

			// Verify numerical parity within floating-point tolerance
			for i := 0; i < tc.M; i++ {
				for j := 0; j < tc.N; j++ {
					actVal, _ := actual.Float32At(i, j)
					expVal, _ := expected.Float32At(i, j)
					diff := math.Abs(float64(actVal - expVal))
					if diff > 1e-4 {
						t.Fatalf("parity mismatch at (%d, %d): actual %f, expected %f (diff %f)", i, j, actVal, expVal, diff)
					}
				}
			}
		})
	}
}

func TestMatMulFloat64Parity(t *testing.T) {
	rng := rand.New(rand.NewSource(1337))
	M, K, N := 35, 45, 55

	aData := make([]float64, M*K)
	for i := range aData {
		aData[i] = rng.Float64()
	}
	bData := make([]float64, K*N)
	for i := range bData {
		bData[i] = rng.Float64()
	}

	a, _ := FromFloat64([]int{M, K}, aData)
	b, _ := FromFloat64([]int{K, N}, bData)

	actual, err := MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	expected, err := refMatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			actVal, _ := actual.Float64At(i, j)
			expVal, _ := expected.Float64At(i, j)
			diff := math.Abs(actVal - expVal)
			if diff > 1e-7 {
				t.Fatalf("float64 parity mismatch at (%d, %d): %f vs %f", i, j, actVal, expVal)
			}
		}
	}
}

func TestMatMulBatchedAndBroadcast(t *testing.T) {
	// a: [2, 1, 3, 4]
	// b: [1, 5, 4, 2]
	// output: [2, 5, 3, 2]
	rng := rand.New(rand.NewSource(99))

	aData := make([]float32, 2*1*3*4)
	for i := range aData {
		aData[i] = float32(i + 1)
	}
	bData := make([]float32, 1*5*4*2)
	for i := range bData {
		bData[i] = rng.Float32()
	}

	a, err := FromFloat32([]int{2, 1, 3, 4}, aData)
	if err != nil {
		t.Fatal(err)
	}
	b, err := FromFloat32([]int{1, 5, 4, 2}, bData)
	if err != nil {
		t.Fatal(err)
	}

	out, err := MatMul(a, b)
	if err != nil {
		t.Fatalf("batched MatMul failed: %v", err)
	}
	if !reflect.DeepEqual(out.Shape(), []int{2, 5, 3, 2}) {
		t.Fatalf("out shape = %v, want [2, 5, 3, 2]", out.Shape())
	}

	// Verify slice parity: out[1, 3, :, :] should match a[1, 0, :, :] x b[0, 3, :, :]
	aSlice, err := a.Slice(0, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	aSub, err := aSlice.Squeeze(0, 1)
	if err != nil {
		t.Fatal(err)
	}

	bSlice, err := b.Slice(1, 3, 4)
	if err != nil {
		t.Fatal(err)
	}
	bSub, err := bSlice.Squeeze(0, 1)
	if err != nil {
		t.Fatal(err)
	}

	expectedSub, err := refMatMul(aSub, bSub)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			actVal, _ := out.Float32At(1, 3, i, j)
			expVal, _ := expectedSub.Float32At(i, j)
			diff := math.Abs(float64(actVal - expVal))
			if diff > 1e-4 {
				t.Fatalf("batch element mismatch at (%d, %d): %f vs %f", i, j, actVal, expVal)
			}
		}
	}
}

func TestMatMulNonContiguousAndEdgeCases(t *testing.T) {
	// Non-contiguous transposed input
	aOrig, _ := FromFloat32([]int{4, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	})
	// Transpose aOrig to shape [3, 4]
	aTrans, err := aOrig.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	if aTrans.IsContiguous() {
		t.Fatal("transposed tensor should be non-contiguous")
	}

	b, _ := FromFloat32([]int{4, 2}, []float32{
		1, 0,
		0, 1,
		1, 0,
		0, 1,
	})

	c, err := MatMul(aTrans, b)
	if err != nil {
		t.Fatalf("MatMul with non-contiguous input failed: %v", err)
	}
	if !reflect.DeepEqual(c.Shape(), []int{3, 2}) {
		t.Fatalf("c shape = %v, want [3, 2]", c.Shape())
	}
	// Row 0 of aTrans is col 0 of aOrig: [1, 4, 7, 10]
	// c[0, 0] = 1*1 + 4*0 + 7*1 + 10*0 = 8
	// c[0, 1] = 1*0 + 4*1 + 7*0 + 10*1 = 14
	v00, _ := c.Float32At(0, 0)
	v01, _ := c.Float32At(0, 1)
	if v00 != 8 || v01 != 14 {
		t.Fatalf("non-contiguous MatMul values: %f, %f, want 8, 14", v00, v01)
	}

	// Edge case: Dimension 0
	aEmpty, _ := New([]int{0, 4}, Float32)
	bEmpty, _ := New([]int{4, 2}, Float32)
	cEmpty, err := MatMul(aEmpty, bEmpty)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(cEmpty.Shape(), []int{0, 2}) || cEmpty.Numel() != 0 {
		t.Fatalf("empty matmul shape = %v, numel = %d", cEmpty.Shape(), cEmpty.Numel())
	}

	// Dimension mismatch
	if _, err := MatMul(aOrig, b); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("expected ErrShapeMismatch, got %v", err)
	}

	// Rank < 2
	scalar, _ := New([]int{}, Float32)
	if _, err := MatMul(scalar, b); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("expected ErrShapeMismatch for rank < 2, got %v", err)
	}
}

// Benchmarks across representative matrix sizes: 32, 64, 128, 256, 512
func BenchmarkMatMul(b *testing.B) {
	sizes := []int{32, 64, 128, 256, 512}

	for _, size := range sizes {
		N := size
		aData := make([]float32, N*N)
		bData := make([]float32, N*N)
		for i := range aData {
			aData[i] = 1.0
			bData[i] = 1.0
		}
		tensorA, _ := FromFloat32([]int{N, N}, aData)
		tensorB, _ := FromFloat32([]int{N, N}, bData)

		// FLOPs for NxN x NxN is 2 * N^3
		flops := 2.0 * float64(N) * float64(N) * float64(N)

		b.Run(fmt.Sprintf("%dx%d", N, N), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := MatMul(tensorA, tensorB); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			gflops := (flops * float64(b.N) / b.Elapsed().Seconds()) / 1e9
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}
