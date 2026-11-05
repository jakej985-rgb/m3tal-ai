//go:build cuda

package lm

import (
	"fmt"
	"math"

	"m3tal-llm/internal/gpu"
)

// cudaLayer holds one linear layer's weights on the device.
type cudaLayer struct {
	w *gpu.Buffer
	b *gpu.Buffer
}

// cudaBlock holds one transformer block's weights on the device.
type cudaBlock struct {
	normAttn, normFF     *gpu.Buffer
	query, key, value    cudaLayer
	attnOut              cudaLayer
	ffGate, ffUp, ffDown cudaLayer
}

// cudaWeights is the device-resident model snapshot.
type cudaWeights struct {
	tokenEmb *gpu.Buffer
	blocks   []cudaBlock
	output   cudaLayer
}

func freeLayer(l cudaLayer) {
	l.w.Free()
	l.b.Free()
}

func freeBlock(b cudaBlock) {
	b.normAttn.Free()
	b.normFF.Free()
	freeLayer(b.query)
	freeLayer(b.key)
	freeLayer(b.value)
	freeLayer(b.attnOut)
	freeLayer(b.ffGate)
	freeLayer(b.ffUp)
	freeLayer(b.ffDown)
}

func freeWeights(w cudaWeights) {
	w.tokenEmb.Free()
	for _, b := range w.blocks {
		freeBlock(b)
	}
	freeLayer(w.output)
}

func uploadF32(src []float32) (*gpu.Buffer, error) {
	b, err := gpu.AllocF32(len(src))
	if err != nil {
		return nil, err
	}
	if err := b.Upload(src); err != nil {
		b.Free()
		return nil, err
	}
	return b, nil
}

func uploadLayer(l inferenceLayer) (cudaLayer, error) {
	w, err := uploadF32(l.weights)
	if err != nil {
		return cudaLayer{}, err
	}
	b, err := uploadF32(l.bias)
	if err != nil {
		w.Free()
		return cudaLayer{}, err
	}
	return cudaLayer{w: w, b: b}, nil
}

// cudaInferenceBackend runs the model forward pass on CUDA. Weights are
// uploaded once at construction; activations stay on the device except
// token ids in and last-row logits out. A host weight clone is retained so
// any GPU failure falls back to the CPU path transparently.
type cudaInferenceBackend struct {
	config Config
	vocab  int
	w      cudaWeights
	host   inferenceWeights
}

// newCUDAInferenceBackend uploads a snapshot of the model weights to the
// GPU. It returns an error when CUDA is unavailable or allocation fails;
// callers should fall back to the CPU backend.
func newCUDAInferenceBackend(model *Model) (inferenceBackend, error) {
	state := model.parameterView()
	host := cloneInferenceWeights(state)

	var w cudaWeights
	fail := func(err error) (inferenceBackend, error) {
		freeWeights(w)
		return nil, err
	}

	tok, err := uploadF32(state.tokenEmb)
	if err != nil {
		return fail(fmt.Errorf("cuda token embedding upload: %w", err))
	}
	w.tokenEmb = tok

	w.blocks = make([]cudaBlock, len(state.blocks))
	for i, sb := range state.blocks {
		cb := cudaBlock{}
		step := func(err error) (inferenceBackend, error) {
			freeBlock(cb)
			return fail(err)
		}
		if cb.normAttn, err = uploadF32(sb.normAttn); err != nil {
			return step(fmt.Errorf("cuda block %d normAttn: %w", i, err))
		}
		if cb.normFF, err = uploadF32(sb.normFF); err != nil {
			return step(fmt.Errorf("cuda block %d normFF: %w", i, err))
		}
		if cb.query, err = uploadLayer(sb.query); err != nil {
			return step(fmt.Errorf("cuda block %d query: %w", i, err))
		}
		if cb.key, err = uploadLayer(sb.key); err != nil {
			return step(fmt.Errorf("cuda block %d key: %w", i, err))
		}
		if cb.value, err = uploadLayer(sb.value); err != nil {
			return step(fmt.Errorf("cuda block %d value: %w", i, err))
		}
		if cb.attnOut, err = uploadLayer(sb.attnOut); err != nil {
			return step(fmt.Errorf("cuda block %d attnOut: %w", i, err))
		}
		if cb.ffGate, err = uploadLayer(sb.ffGate); err != nil {
			return step(fmt.Errorf("cuda block %d ffGate: %w", i, err))
		}
		if cb.ffUp, err = uploadLayer(sb.ffUp); err != nil {
			return step(fmt.Errorf("cuda block %d ffUp: %w", i, err))
		}
		if cb.ffDown, err = uploadLayer(sb.ffDown); err != nil {
			return step(fmt.Errorf("cuda block %d ffDown: %w", i, err))
		}
		w.blocks[i] = cb
	}

	out, err := uploadLayer(state.output)
	if err != nil {
		return fail(fmt.Errorf("cuda output layer upload: %w", err))
	}
	w.output = out

	return &cudaInferenceBackend{
		config: model.Config,
		vocab:  len(model.Tokenizer.Tokens),
		w:      w,
		host:   host,
	}, nil
}

// sanitizeIDs mirrors forwardFloat32's id handling: trim to context, clamp
// out-of-range ids to 0, and default to token 0 for empty input.
func sanitizeIDs(config Config, vocab int, tokenIDs []int) []int {
	config.Normalize()
	if len(tokenIDs) > config.Context {
		tokenIDs = tokenIDs[len(tokenIDs)-config.Context:]
	}
	if len(tokenIDs) == 0 {
		tokenIDs = []int{0}
	}
	ids := append([]int(nil), tokenIDs...)
	for i, id := range ids {
		if id < 0 || id >= vocab {
			ids[i] = 0
		}
	}
	return ids
}

// cudaScratch holds the per-call device buffers for one forward pass.
type cudaScratch struct {
	ids      *gpu.Buffer
	x        *gpu.Buffer
	n1       *gpu.Buffer
	q, k, v  *gpu.Buffer
	attn     *gpu.Buffer
	attended *gpu.Buffer
	proj     *gpu.Buffer
	res      *gpu.Buffer
	n2       *gpu.Buffer
	gate     *gpu.Buffer
	up       *gpu.Buffer
	act      *gpu.Buffer
	ff       *gpu.Buffer
	logits   *gpu.Buffer
}

func (s *cudaScratch) free() {
	for _, b := range []*gpu.Buffer{s.ids, s.x, s.n1, s.q, s.k, s.v, s.attn,
		s.attended, s.proj, s.res, s.n2, s.gate, s.up, s.act, s.ff, s.logits} {
		b.Free()
	}
}

// allocScratch allocates every buffer a forward pass needs. kvLen is the
// attention key/value length (== rows for batch, pos+1 for decode steps).
func (b *cudaInferenceBackend) allocScratch(rows, kvLen int) (*cudaScratch, error) {
	b.config.Normalize()
	width := b.config.Width
	heads := b.config.Heads
	hidden := width * 2

	s := &cudaScratch{}
	var err error
	f32bufs := []struct {
		dst **gpu.Buffer
		n   int // element count
	}{
		{&s.x, rows * width},
		{&s.n1, rows * width},
		{&s.q, rows * width},
		{&s.k, rows * width},
		{&s.v, rows * width},
		{&s.attn, heads * rows * kvLen},
		{&s.attended, rows * width},
		{&s.proj, rows * width},
		{&s.res, rows * width},
		{&s.n2, rows * width},
		{&s.gate, rows * hidden},
		{&s.up, rows * hidden},
		{&s.act, rows * hidden},
		{&s.ff, rows * width},
		{&s.logits, rows * b.vocab},
	}
	for i := range f32bufs {
		*f32bufs[i].dst, err = gpu.AllocF32(f32bufs[i].n)
		if err != nil {
			s.free()
			return nil, err
		}
	}
	s.ids, err = gpu.AllocBytes(rows * 4)
	if err != nil {
		s.free()
		return nil, err
	}
	return s, nil
}

// linearF32 runs out[M,N] = in[M,K] * W[K,N] + bias on device buffers.
func linearF32(in, w, bias, out *gpu.Buffer, M, K, N int) error {
	if err := gpu.GemmF32(in, w, out, M, N, K); err != nil {
		return err
	}
	return gpu.BiasAddF32(out, bias, M, N)
}

// runTransformer executes embedding + all blocks + output projection on
// device buffers held in s. causal=true is the batch path; causal=false is
// the single-token KV-cache decode step where kSrc/vSrc point at the device
// KV cache (kvLen entries) instead of s.k/s.v.
func (b *cudaInferenceBackend) runTransformer(s *cudaScratch, rows, kvLen int,
	causal bool, kSrc, vSrc *gpu.Buffer) error {

	b.config.Normalize()
	width := b.config.Width
	heads := b.config.Heads
	headDim := width / heads
	hidden := width * 2
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// Token embedding gather.
	if err := gpu.EmbedF32(b.w.tokenEmb, s.ids, s.x, rows, width); err != nil {
		return err
	}

	for i := range b.w.blocks {
		cb := b.w.blocks[i]
		if err := gpu.RMSNormF32(s.x, cb.normAttn, s.n1, rows, width, 1e-6); err != nil {
			return err
		}
		if err := linearF32(s.n1, cb.query.w, cb.query.b, s.q, rows, width, width); err != nil {
			return err
		}
		if err := linearF32(s.n1, cb.key.w, cb.key.b, s.k, rows, width, width); err != nil {
			return err
		}
		if err := linearF32(s.n1, cb.value.w, cb.value.b, s.v, rows, width, width); err != nil {
			return err
		}

		ka, va := s.k, s.v
		if kSrc != nil {
			ka, va = kSrc, vSrc
		}
		if causal {
			if err := gpu.RoPEBatchF32(s.q, s.k, rows, width, headDim, heads); err != nil {
				return err
			}
		} else {
			if err := gpu.RoPESingleF32(s.q, s.k, kvLen-1, width, headDim, heads); err != nil {
				return err
			}
		}
		if err := gpu.AttnScoresF32(s.q, ka, s.attn, heads, rows, kvLen,
			width, headDim, scale, causal); err != nil {
			return err
		}
		if err := gpu.AttnSoftmaxF32(s.attn, heads, rows, kvLen); err != nil {
			return err
		}
		if err := gpu.AttnWeightedF32(s.attn, va, s.attended,
			heads, rows, kvLen, width, headDim); err != nil {
			return err
		}
		if err := linearF32(s.attended, cb.attnOut.w, cb.attnOut.b, s.proj, rows, width, width); err != nil {
			return err
		}
		if err := gpu.ElemwiseF32(gpu.OpAdd, s.x, s.proj, s.res, rows*width); err != nil {
			return err
		}
		if err := gpu.RMSNormF32(s.res, cb.normFF, s.n2, rows, width, 1e-6); err != nil {
			return err
		}
		if err := linearF32(s.n2, cb.ffGate.w, cb.ffGate.b, s.gate, rows, width, hidden); err != nil {
			return err
		}
		if err := linearF32(s.n2, cb.ffUp.w, cb.ffUp.b, s.up, rows, width, hidden); err != nil {
			return err
		}
		if err := gpu.SwiGLUF32(s.gate, s.up, s.act, rows*hidden); err != nil {
			return err
		}
		if err := linearF32(s.act, cb.ffDown.w, cb.ffDown.b, s.ff, rows, hidden, width); err != nil {
			return err
		}
		if err := gpu.ElemwiseF32(gpu.OpAdd, s.res, s.ff, s.x, rows*width); err != nil {
			return err
		}
	}

	return linearF32(s.x, b.w.output.w, b.w.output.b, s.logits, rows, width, b.vocab)
}

// uploadIDs writes token ids into the scratch ids buffer as little-endian int32.
func uploadIDs(s *cudaScratch, ids []int) error {
	buf := make([]byte, len(ids)*4)
	for i, id := range ids {
		v := int32(id)
		buf[i*4] = byte(v)
		buf[i*4+1] = byte(v >> 8)
		buf[i*4+2] = byte(v >> 16)
		buf[i*4+3] = byte(v >> 24)
	}
	return s.ids.UploadBytes(buf)
}

// Forward runs the batch forward pass on the GPU and returns the last
// row's logits. On any CUDA error it falls back to the CPU path.
func (b *cudaInferenceBackend) Forward(tokenIDs []int) []float32 {
	ids := sanitizeIDs(b.config, b.vocab, tokenIDs)
	rows := len(ids)

	s, err := b.allocScratch(rows, rows)
	if err != nil {
		return b.forwardCPU(ids)
	}
	defer s.free()

	if err := uploadIDs(s, ids); err != nil {
		return b.forwardCPU(ids)
	}
	if err := b.runTransformer(s, rows, rows, true, nil, nil); err != nil {
		return b.forwardCPU(ids)
	}

	last := make([]float32, b.vocab)
	rowBuf, err := gpu.AllocF32(b.vocab)
	if err != nil {
		return b.forwardCPU(ids)
	}
	defer rowBuf.Free()
	if err := gpu.MemcpyD2D(rowBuf, 0, s.logits, (rows-1)*b.vocab*4, b.vocab*4); err != nil {
		return b.forwardCPU(ids)
	}
	if err := rowBuf.Download(last); err != nil {
		return b.forwardCPU(ids)
	}
	return last
}

func (b *cudaInferenceBackend) forwardCPU(ids []int) []float32 {
	cache := forwardFloat32(b.config, b.vocab, b.host, ids)
	rows := len(cache.ids)
	return cache.logits[(rows-1)*b.vocab : rows*b.vocab]
}

func (b *cudaInferenceBackend) NewKVCache() *kvCache {
	return newKVCache(len(b.w.blocks))
}

// ensureKV grows a layer's device K/V cache to hold at least `rows` tokens.
func ensureKV(layer *kvLayerCache, width, rows int) error {
	need := rows * width * 4
	if layer.dk != nil && layer.dk.Bytes() >= need {
		return nil
	}
	newCap := rows
	if layer.dk != nil {
		newCap = layer.dk.Bytes() / (width * 4) * 2
		if newCap < rows {
			newCap = rows
		}
	}
	nk, err := gpu.AllocF32(newCap * width)
	if err != nil {
		return err
	}
	nv, err := gpu.AllocF32(newCap * width)
	if err != nil {
		nk.Free()
		return err
	}
	// Preserve existing contents.
	oldRows := 0
	if layer.dk != nil {
		oldRows = len(layer.k) / width
		if oldRows > 0 {
			if err := gpu.MemcpyD2D(nk, 0, layer.dk, 0, oldRows*width*4); err != nil {
				nk.Free()
				nv.Free()
				return err
			}
			if err := gpu.MemcpyD2D(nv, 0, layer.dv, 0, oldRows*width*4); err != nil {
				nk.Free()
				nv.Free()
				return err
			}
		}
		layer.dk.Free()
		layer.dv.Free()
	}
	layer.dk, layer.dv = nk, nv
	return nil
}

// ForwardStepKV runs one token through the model on the GPU, appending K/V
// to the device cache. The host cache mirrors every append so the CPU
// fallback stays correct; any GPU error marks the cache cpuOnly and
// delegates the remainder to the CPU path.
func (b *cudaInferenceBackend) ForwardStepKV(tokenID int, pos int, cache *kvCache) []float32 {
	b.config.Normalize()
	if cache == nil || cache.cpuOnly {
		return forwardStepKV(b.config, b.vocab, b.host, tokenID, pos, cache)
	}
	width := b.config.Width
	heads := b.config.Heads
	headDim := width / heads
	hidden := width * 2
	kvLen := pos + 1

	if tokenID < 0 || tokenID >= b.vocab {
		tokenID = 0
	}

	// Pre-allocate device KV caches for every layer before mutating state.
	for i := range cache.layers {
		if err := ensureKV(&cache.layers[i], width, kvLen); err != nil {
			cache.cpuOnly = true
			return forwardStepKV(b.config, b.vocab, b.host, tokenID, pos, cache)
		}
	}

	s, err := b.allocScratch(1, kvLen)
	if err != nil {
		cache.cpuOnly = true
		return forwardStepKV(b.config, b.vocab, b.host, tokenID, pos, cache)
	}
	defer s.free()

	if err := uploadIDs(s, []int{tokenID}); err != nil {
		cache.cpuOnly = true
		return forwardStepKV(b.config, b.vocab, b.host, tokenID, pos, cache)
	}

	// Token embedding for the single step token.
	if err := gpu.EmbedF32(b.w.tokenEmb, s.ids, s.x, 1, width); err != nil {
		cache.cpuOnly = true
		return forwardStepKV(b.config, b.vocab, b.host, tokenID, pos, cache)
	}

	// Track which layers appended K/V so a mid-pass failure can roll the
	// host mirror back before delegating to CPU.
	appended := 0
	rollback := func() {
		for i := 0; i < appended && i < len(cache.layers); i++ {
			l := &cache.layers[i]
			l.k = l.k[:pos*width]
			l.v = l.v[:pos*width]
		}
	}

	ok := true
	for i := range b.w.blocks {
		cb := b.w.blocks[i]
		lc := &cache.layers[i]

		// Reuse the shared transformer body but with per-layer KV handling:
		// run the block manually so we can append this layer's K/V first.
		if err := gpu.RMSNormF32(s.x, cb.normAttn, s.n1, 1, width, 1e-6); err != nil {
			ok = false
			break
		}
		if err := linearF32(s.n1, cb.query.w, cb.query.b, s.q, 1, width, width); err != nil {
			ok = false
			break
		}
		if err := linearF32(s.n1, cb.key.w, cb.key.b, s.k, 1, width, width); err != nil {
			ok = false
			break
		}
		if err := linearF32(s.n1, cb.value.w, cb.value.b, s.v, 1, width, width); err != nil {
			ok = false
			break
		}
		if err := gpu.RoPESingleF32(s.q, s.k, pos, width, headDim, heads); err != nil {
			ok = false
			break
		}
		// Append K/V to the device cache and mirror to the host slices.
		if err := gpu.MemcpyD2D(lc.dk, pos*width*4, s.k, 0, width*4); err != nil {
			ok = false
			break
		}
		if err := gpu.MemcpyD2D(lc.dv, pos*width*4, s.v, 0, width*4); err != nil {
			ok = false
			break
		}
		// Mirror to host cache for CPU fallback correctness.
		kSlice := make([]float32, width)
		vSlice := make([]float32, width)
		if err := s.k.Download(kSlice); err != nil {
			ok = false
			break
		}
		if err := s.v.Download(vSlice); err != nil {
			ok = false
			break
		}
		lc.k = append(lc.k, kSlice...)
		lc.v = append(lc.v, vSlice...)
		appended++

		// Attention against the full device cache (no causal mask: all
		// cached positions are valid history for a single query).
		scale := float32(1.0 / math.Sqrt(float64(headDim)))
		if err := gpu.AttnScoresF32(s.q, lc.dk, s.attn, heads, 1, kvLen,
			width, headDim, scale, false); err != nil {
			ok = false
			break
		}
		if err := gpu.AttnSoftmaxF32(s.attn, heads, 1, kvLen); err != nil {
			ok = false
			break
		}
		if err := gpu.AttnWeightedF32(s.attn, lc.dv, s.attended,
			heads, 1, kvLen, width, headDim); err != nil {
			ok = false
			break
		}
		if err := linearF32(s.attended, cb.attnOut.w, cb.attnOut.b, s.proj, 1, width, width); err != nil {
			ok = false
			break
		}
		if err := gpu.ElemwiseF32(gpu.OpAdd, s.x, s.proj, s.res, width); err != nil {
			ok = false
			break
		}
		if err := gpu.RMSNormF32(s.res, cb.normFF, s.n2, 1, width, 1e-6); err != nil {
			ok = false
			break
		}
		if err := linearF32(s.n2, cb.ffGate.w, cb.ffGate.b, s.gate, 1, width, hidden); err != nil {
			ok = false
			break
		}
		if err := linearF32(s.n2, cb.ffUp.w, cb.ffUp.b, s.up, 1, width, hidden); err != nil {
			ok = false
			break
		}
		if err := gpu.SwiGLUF32(s.gate, s.up, s.act, hidden); err != nil {
			ok = false
			break
		}
		if err := linearF32(s.act, cb.ffDown.w, cb.ffDown.b, s.ff, 1, hidden, width); err != nil {
			ok = false
			break
		}
		if err := gpu.ElemwiseF32(gpu.OpAdd, s.res, s.ff, s.x, width); err != nil {
			ok = false
			break
		}
	}

	if !ok {
		rollback()
		cache.cpuOnly = true
		return forwardStepKV(b.config, b.vocab, b.host, tokenID, pos, cache)
	}

	if err := linearF32(s.x, b.w.output.w, b.w.output.b, s.logits, 1, width, b.vocab); err != nil {
		rollback()
		cache.cpuOnly = true
		return forwardStepKV(b.config, b.vocab, b.host, tokenID, pos, cache)
	}

	out := make([]float32, b.vocab)
	if err := s.logits.Download(out); err != nil {
		rollback()
		cache.cpuOnly = true
		return forwardStepKV(b.config, b.vocab, b.host, tokenID, pos, cache)
	}
	return out
}
