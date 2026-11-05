package nn

import (
	"errors"
	"testing"

	"m3tal-ai/internal/tensor"
)

func TestKVCacheLifecycle(t *testing.T) {
	numLayers := 2
	batchSize := 2
	numHeads := 4
	headDim := 16
	maxSeqLen := 8

	cache, err := NewKVCache(numLayers, batchSize, numHeads, headDim, maxSeqLen, tensor.Float32)
	if err != nil {
		t.Fatalf("NewKVCache failed: %v", err)
	}

	if cache.SeqLen() != 0 {
		t.Errorf("initial seqLen = %d, want 0", cache.SeqLen())
	}
	if cache.MaxSeqLen() != maxSeqLen {
		t.Errorf("maxSeqLen = %d, want %d", cache.MaxSeqLen(), maxSeqLen)
	}

	// Step 1: Add prompt of 3 tokens
	k1, err := tensor.New([]int{batchSize, numHeads, 3, headDim}, tensor.Float32)
	if err != nil {
		t.Fatal(err)
	}
	v1, err := tensor.New([]int{batchSize, numHeads, 3, headDim}, tensor.Float32)
	if err != nil {
		t.Fatal(err)
	}
	_ = k1.SetFloat32At(1.5, 0, 0, 0, 0)
	_ = v1.SetFloat32At(2.5, 0, 0, 0, 0)

	// Update layer 0
	cachedK, cachedV, err := cache.Update(0, k1, v1)
	if err != nil {
		t.Fatalf("Update layer 0 failed: %v", err)
	}
	if cachedK.Shape()[2] != 3 || cachedV.Shape()[2] != 3 {
		t.Errorf("expected cached shape [B, H, 3, D], got k=%v, v=%v", cachedK.Shape(), cachedV.Shape())
	}

	// Update layer 1
	cachedK1, cachedV1, err := cache.Update(1, k1, v1)
	if err != nil {
		t.Fatalf("Update layer 1 failed: %v", err)
	}
	if cachedK1.Shape()[2] != 3 || cachedV1.Shape()[2] != 3 {
		t.Errorf("expected cached shape [B, H, 3, D], got k=%v, v=%v", cachedK1.Shape(), cachedV1.Shape())
	}
	if cache.SeqLen() != 3 {
		t.Errorf("seqLen after prompt = %d, want 3", cache.SeqLen())
	}

	// Verify values at layer 0 and 1
	kGet, vGet, err := cache.Get(0)
	if err != nil {
		t.Fatal(err)
	}
	valK, _ := kGet.Float32At(0, 0, 0, 0)
	valV, _ := vGet.Float32At(0, 0, 0, 0)
	if valK != 1.5 || valV != 2.5 {
		t.Errorf("Get(0) values mismatch: got k=%f, v=%f", valK, valV)
	}

	// Step 2: Incrementally decode 1 token
	k2, _ := tensor.New([]int{batchSize, numHeads, 1, headDim}, tensor.Float32)
	v2, _ := tensor.New([]int{batchSize, numHeads, 1, headDim}, tensor.Float32)
	_ = k2.SetFloat32At(3.5, 0, 0, 0, 0)
	_ = v2.SetFloat32At(4.5, 0, 0, 0, 0)

	_, _, _ = cache.Update(0, k2, v2)
	cachedK2, cachedV2, err := cache.Update(1, k2, v2)
	if err != nil {
		t.Fatalf("Update single token failed: %v", err)
	}
	if cache.SeqLen() != 4 {
		t.Errorf("seqLen after 1 step = %d, want 4", cache.SeqLen())
	}
	if cachedK2.Shape()[2] != 4 || cachedV2.Shape()[2] != 4 {
		t.Errorf("expected cached length 4, got %v", cachedK2.Shape())
	}

	// Verify token at position 3 has the new value
	kGet, vGet, _ = cache.Get(1)
	valK2, _ := kGet.Float32At(0, 0, 3, 0)
	valV2, _ := vGet.Float32At(0, 0, 3, 0)
	if valK2 != 3.5 || valV2 != 4.5 {
		t.Errorf("new token position mismatch: got k=%f, v=%f", valK2, valV2)
	}

	// Step 3: Bounds checking
	// We are at length 4, max is 8. Trying to append 5 tokens should fail with ErrKVCacheBoundsExceeded
	kOver, _ := tensor.New([]int{batchSize, numHeads, 5, headDim}, tensor.Float32)
	vOver, _ := tensor.New([]int{batchSize, numHeads, 5, headDim}, tensor.Float32)
	_, _, err = cache.Update(0, kOver, vOver)
	if !errors.Is(err, ErrKVCacheBoundsExceeded) {
		t.Errorf("expected ErrKVCacheBoundsExceeded, got %v", err)
	}

	// Step 4: Reset
	cache.Reset()
	if cache.SeqLen() != 0 {
		t.Errorf("after Reset, seqLen = %d, want 0", cache.SeqLen())
	}

	// After reset, we can re-add up to maxSeqLen tokens
	kFresh, _ := tensor.New([]int{batchSize, numHeads, 8, headDim}, tensor.Float32)
	vFresh, _ := tensor.New([]int{batchSize, numHeads, 8, headDim}, tensor.Float32)
	_, _, err = cache.Update(0, kFresh, vFresh)
	if err != nil {
		t.Fatalf("Update after Reset failed: %v", err)
	}
	_, _, err = cache.Update(1, kFresh, vFresh)
	if err != nil {
		t.Fatalf("Update after Reset failed: %v", err)
	}
	if cache.SeqLen() != 8 {
		t.Errorf("seqLen = %d, want 8", cache.SeqLen())
	}
}

func TestKVCacheInvalidArgs(t *testing.T) {
	// Invalid layer index
	cache, err := NewKVCache(1, 1, 1, 4, 4, tensor.Float32)
	if err != nil {
		t.Fatal(err)
	}
	k, _ := tensor.New([]int{1, 1, 1, 4}, tensor.Float32)
	v, _ := tensor.New([]int{1, 1, 1, 4}, tensor.Float32)

	_, _, err = cache.Update(-1, k, v)
	if !errors.Is(err, ErrKVCacheInvalidLayer) {
		t.Errorf("expected ErrKVCacheInvalidLayer for layer -1, got %v", err)
	}
	_, _, err = cache.Update(1, k, v)
	if !errors.Is(err, ErrKVCacheInvalidLayer) {
		t.Errorf("expected ErrKVCacheInvalidLayer for layer 1, got %v", err)
	}

	// Shape mismatch
	kBad, _ := tensor.New([]int{1, 2, 1, 4}, tensor.Float32)
	_, _, err = cache.Update(0, kBad, v)
	if !errors.Is(err, ErrKVCacheShapeMismatch) {
		t.Errorf("expected ErrKVCacheShapeMismatch, got %v", err)
	}

	// DType mismatch
	kDType, _ := tensor.New([]int{1, 1, 1, 4}, tensor.Float64)
	vDType, _ := tensor.New([]int{1, 1, 1, 4}, tensor.Float64)
	_, _, err = cache.Update(0, kDType, vDType)
	if err == nil {
		t.Errorf("expected error for dtype mismatch, got nil")
	}
}
