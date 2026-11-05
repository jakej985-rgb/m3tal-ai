package nn

import (
	"errors"
	"fmt"
	"reflect"

	"m3tal-ai/internal/tensor"
)

var (
	// ErrKVCacheBoundsExceeded indicates sequence length exceeded max capacity.
	ErrKVCacheBoundsExceeded = errors.New("kv_cache: capacity bounds exceeded")
	// ErrKVCacheInvalidLayer indicates an out-of-range layer index was accessed.
	ErrKVCacheInvalidLayer = errors.New("kv_cache: invalid layer index")
	// ErrKVCacheShapeMismatch indicates incoming key or value shape mismatch.
	ErrKVCacheShapeMismatch = errors.New("kv_cache: shape mismatch")
)

// LayerKVCache stores pre-allocated key and value tensors for a single transformer layer.
// Storage shape for both keys and values is [BatchSize, NumHeads, MaxSeqLen, HeadDim].
type LayerKVCache struct {
	Key   *tensor.Tensor
	Value *tensor.Tensor
}

// KVCache manages key and value tensors across layers during auto-regressive decoding.
type KVCache struct {
	numLayers int
	batchSize int
	numHeads  int
	headDim   int
	maxSeqLen int
	seqLens   []int
	dtype     tensor.DType
	layers    []LayerKVCache
}

// NewKVCache allocates a new Key-Value cache buffer for all layers.
func NewKVCache(numLayers, batchSize, numHeads, headDim, maxSeqLen int, dtype tensor.DType) (*KVCache, error) {
	if numLayers <= 0 {
		return nil, fmt.Errorf("kv_cache: numLayers must be positive, got %d", numLayers)
	}
	if batchSize <= 0 {
		return nil, fmt.Errorf("kv_cache: batchSize must be positive, got %d", batchSize)
	}
	if numHeads <= 0 {
		return nil, fmt.Errorf("kv_cache: numHeads must be positive, got %d", numHeads)
	}
	if headDim <= 0 {
		return nil, fmt.Errorf("kv_cache: headDim must be positive, got %d", headDim)
	}
	if maxSeqLen <= 0 {
		return nil, fmt.Errorf("kv_cache: maxSeqLen must be positive, got %d", maxSeqLen)
	}
	if dtype != tensor.Float32 && dtype != tensor.Float64 {
		return nil, fmt.Errorf("kv_cache: unsupported dtype %s", dtype)
	}

	shape := []int{batchSize, numHeads, maxSeqLen, headDim}
	layers := make([]LayerKVCache, numLayers)
	for i := 0; i < numLayers; i++ {
		k, err := tensor.New(shape, dtype)
		if err != nil {
			return nil, err
		}
		v, err := tensor.New(shape, dtype)
		if err != nil {
			return nil, err
		}
		layers[i] = LayerKVCache{Key: k, Value: v}
	}

	return &KVCache{
		numLayers: numLayers,
		batchSize: batchSize,
		numHeads:  numHeads,
		headDim:   headDim,
		maxSeqLen: maxSeqLen,
		seqLens:   make([]int, numLayers),
		dtype:     dtype,
		layers:    layers,
	}, nil
}

// NumLayers returns the number of layers in the cache.
func (c *KVCache) NumLayers() int { return c.numLayers }

// BatchSize returns the batch size.
func (c *KVCache) BatchSize() int { return c.batchSize }

// NumHeads returns the number of heads.
func (c *KVCache) NumHeads() int { return c.numHeads }

// HeadDim returns the head dimension.
func (c *KVCache) HeadDim() int { return c.headDim }

// MaxSeqLen returns the maximum sequence capacity.
func (c *KVCache) MaxSeqLen() int { return c.maxSeqLen }

// SeqLen returns the current sequence length stored in the cache (across all layers).
func (c *KVCache) SeqLen() int {
	if len(c.seqLens) == 0 {
		return 0
	}
	minLen := c.seqLens[0]
	for _, l := range c.seqLens[1:] {
		if l < minLen {
			minLen = l
		}
	}
	return minLen
}

// LayerSeqLen returns the sequence length stored in a specific layer.
func (c *KVCache) LayerSeqLen(layerIdx int) int {
	if layerIdx < 0 || layerIdx >= c.numLayers {
		return 0
	}
	return c.seqLens[layerIdx]
}

// DType returns the element type of the cache.
func (c *KVCache) DType() tensor.DType { return c.dtype }

// Reset resets the cached sequence length to 0 without reallocating underlying buffers.
func (c *KVCache) Reset() {
	for i := range c.seqLens {
		c.seqLens[i] = 0
	}
}

// Update writes incoming k and v tensors into the cache at layerIdx starting at current SeqLen,
// and returns slices/views of cached keys and values up to the new sequence length:
// [BatchSize, NumHeads, newSeqLen, HeadDim].
// Expected shape of incoming k and v is [BatchSize, NumHeads, incomingSeqLen, HeadDim].
func (c *KVCache) Update(layerIdx int, k, v *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	if layerIdx < 0 || layerIdx >= c.numLayers {
		return nil, nil, fmt.Errorf("%w: got %d, want in [0, %d)", ErrKVCacheInvalidLayer, layerIdx, c.numLayers)
	}
	if k == nil || v == nil {
		return nil, nil, fmt.Errorf("kv_cache: incoming k and v cannot be nil")
	}
	if k.DType() != c.dtype || v.DType() != c.dtype {
		return nil, nil, fmt.Errorf("kv_cache: dtype mismatch: expected %s, got k=%s, v=%s", c.dtype, k.DType(), v.DType())
	}

	shapeK := k.Shape()
	shapeV := v.Shape()
	if !reflect.DeepEqual(shapeK, shapeV) {
		return nil, nil, fmt.Errorf("%w: k shape %v != v shape %v", ErrKVCacheShapeMismatch, shapeK, shapeV)
	}
	if len(shapeK) != 4 {
		return nil, nil, fmt.Errorf("%w: expected 4D tensor [B, H, T, D], got rank %d", ErrKVCacheShapeMismatch, len(shapeK))
	}
	b, h, incomingSeqLen, d := shapeK[0], shapeK[1], shapeK[2], shapeK[3]
	if b != c.batchSize || h != c.numHeads || d != c.headDim {
		return nil, nil, fmt.Errorf("%w: expected [B=%d, H=%d, T=?, D=%d], got %v", ErrKVCacheShapeMismatch, c.batchSize, c.numHeads, c.headDim, shapeK)
	}

	startPos := c.seqLens[layerIdx]
	newSeqLen := startPos + incomingSeqLen
	if newSeqLen > c.maxSeqLen {
		return nil, nil, fmt.Errorf("%w: required sequence length %d exceeds capacity %d", ErrKVCacheBoundsExceeded, newSeqLen, c.maxSeqLen)
	}

	layer := c.layers[layerIdx]

	// Copy incoming k and v into cache buffers
	if err := copyKV(layer.Key, k, startPos, b, h, incomingSeqLen, d, c.dtype); err != nil {
		return nil, nil, err
	}
	if err := copyKV(layer.Value, v, startPos, b, h, incomingSeqLen, d, c.dtype); err != nil {
		return nil, nil, err
	}

	c.seqLens[layerIdx] = newSeqLen

	// Return sliced view of cached keys and values [B, H, newSeqLen, D]
	cachedK, err := sliceKV(layer.Key, b, h, newSeqLen, d, c.dtype)
	if err != nil {
		return nil, nil, err
	}
	cachedV, err := sliceKV(layer.Value, b, h, newSeqLen, d, c.dtype)
	if err != nil {
		return nil, nil, err
	}

	return cachedK, cachedV, nil
}

// Get returns the cached keys and values for a layer up to its current sequence length: [B, H, SeqLen, D].
func (c *KVCache) Get(layerIdx int) (*tensor.Tensor, *tensor.Tensor, error) {
	if layerIdx < 0 || layerIdx >= c.numLayers {
		return nil, nil, fmt.Errorf("%w: got %d, want in [0, %d)", ErrKVCacheInvalidLayer, layerIdx, c.numLayers)
	}
	seqLen := c.seqLens[layerIdx]
	if seqLen == 0 {
		return nil, nil, nil
	}
	layer := c.layers[layerIdx]
	k, err := sliceKV(layer.Key, c.batchSize, c.numHeads, seqLen, c.headDim, c.dtype)
	if err != nil {
		return nil, nil, err
	}
	v, err := sliceKV(layer.Value, c.batchSize, c.numHeads, seqLen, c.headDim, c.dtype)
	if err != nil {
		return nil, nil, err
	}
	return k, v, nil
}

func copyKV(dst, src *tensor.Tensor, startPos, b, h, t, d int, dtype tensor.DType) error {
	switch dtype {
	case tensor.Float32:
		for bi := 0; bi < b; bi++ {
			for hi := 0; hi < h; hi++ {
				for ti := 0; ti < t; ti++ {
					for di := 0; di < d; di++ {
						val, err := src.Float32At(bi, hi, ti, di)
						if err != nil {
							return err
						}
						if err := dst.SetFloat32At(val, bi, hi, startPos+ti, di); err != nil {
							return err
						}
					}
				}
			}
		}
	case tensor.Float64:
		for bi := 0; bi < b; bi++ {
			for hi := 0; hi < h; hi++ {
				for ti := 0; ti < t; ti++ {
					for di := 0; di < d; di++ {
						val, err := src.Float64At(bi, hi, ti, di)
						if err != nil {
							return err
						}
						if err := dst.SetFloat64At(val, bi, hi, startPos+ti, di); err != nil {
							return err
						}
					}
				}
			}
		}
	}
	return nil
}

func sliceKV(src *tensor.Tensor, b, h, length, d int, dtype tensor.DType) (*tensor.Tensor, error) {
	out, err := tensor.New([]int{b, h, length, d}, dtype)
	if err != nil {
		return nil, err
	}
	switch dtype {
	case tensor.Float32:
		for bi := 0; bi < b; bi++ {
			for hi := 0; hi < h; hi++ {
				for ti := 0; ti < length; ti++ {
					for di := 0; di < d; di++ {
						val, err := src.Float32At(bi, hi, ti, di)
						if err != nil {
							return nil, err
						}
						if err := out.SetFloat32At(val, bi, hi, ti, di); err != nil {
							return nil, err
						}
					}
				}
			}
		}
	case tensor.Float64:
		for bi := 0; bi < b; bi++ {
			for hi := 0; hi < h; hi++ {
				for ti := 0; ti < length; ti++ {
					for di := 0; di < d; di++ {
						val, err := src.Float64At(bi, hi, ti, di)
						if err != nil {
							return nil, err
						}
						if err := out.SetFloat64At(val, bi, hi, ti, di); err != nil {
							return nil, err
						}
					}
				}
			}
		}
	}
	return out, nil
}
