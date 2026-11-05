package nn

import (
	"fmt"
	"math/rand"
	"testing"

	"m3tal-ai/internal/tensor"
)

func BenchmarkAttentionForwardInference(b *testing.B) {
	seqLengths := []int{16, 64, 256}
	for _, seqLen := range seqLengths {
		b.Run(fmt.Sprintf("SeqLen_%d", seqLen), func(b *testing.B) {
			rng := rand.New(rand.NewSource(42))
			cfg := CausalSelfAttentionConfig{
				Dim:       64,
				Heads:     4,
				HeadDim:   16,
				MaxSeqLen: seqLen + 10,
				Bias:      true,
				DType:     tensor.Float32,
			}
			csa, err := NewCausalSelfAttention(cfg, rng)
			if err != nil {
				b.Fatal(err)
			}
			cache, err := NewKVCache(1, 1, cfg.Heads, cfg.HeadDim, cfg.MaxSeqLen, tensor.Float32)
			if err != nil {
				b.Fatal(err)
			}

			inputTok, _ := tensor.New([]int{1, 1, cfg.Dim}, tensor.Float32)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				cache.Reset()
				for s := 0; s < seqLen; s++ {
					_, _ = csa.ForwardInference(inputTok, cache, 0)
				}
			}
		})
	}
}
