package lm

import (
	"math"

	"m3tal-llm/internal/gpu"
)

type inferenceLayer struct {
	weights []float32 // row-major [inputSize, outputSize]
	bias    []float32
}

// inferenceBlock holds one transformer layer's weights plus the two RMSNorm
// gamma vectors and the SwiGLU feed-forward projections (gate, up, down).
type inferenceBlock struct {
	normAttn          []float32 // learnable gamma [width]
	normFF            []float32 // learnable gamma [width]
	query, key, value inferenceLayer
	attnOut           inferenceLayer
	ffGate, ffUp      inferenceLayer // [width, width*2]
	ffDown            inferenceLayer // [width*2, width]
}

// inferenceWeights holds all model weights for a single forward pass.
type inferenceWeights struct {
	tokenEmb []float32
	blocks   []inferenceBlock
	output   inferenceLayer
}

// blockCache stores every intermediate tensor needed by the backward pass.
type blockCache struct {
	x           []float32 // raw block input                 [rows, width]
	normX1      []float32 // pre-attention RMSNorm output    [rows, width]
	normRms1    []float32 // per-row RMS from norm #1        [rows]
	q           []float32 // query after RoPE                [rows, width]
	k           []float32 // key after RoPE                  [rows, width]
	v           []float32 // value (no RoPE)                 [rows, width]
	attention   []float32 // softmax attention weights       [heads, rows, rows]
	attended    []float32 // attended values                [rows, width]
	residual    []float32 // x + attn projection            [rows, width]
	normX2      []float32 // pre-FFN RMSNorm output          [rows, width]
	normRms2    []float32 // per-row RMS from norm #2        [rows]
	ffGateOut   []float32 // linear(normX2, ffGate)         [rows, width*2]
	ffUpOut     []float32 // linear(normX2, ffUp)           [rows, width*2]
	ffActivated []float32 // swish(gate) * up               [rows, width*2]
	encoded     []float32 // block output                    [rows, width]
}

type forwardCache struct {
	ids           []int
	blocks        []blockCache
	logits        []float32 // [rows, vocab]
	probabilities []float32 // [rows, vocab]
}

// kvLayerCache stores accumulated keys and values for one layer during autoregressive generation.
type kvLayerCache struct {
	k []float32 // row-major [pastTokens, width]
	v []float32 // row-major [pastTokens, width]
	// Device-resident K/V used by the CUDA inference backend (nil on CPU).
	// The host slices above are always kept in sync so the CPU fallback
	// path remains correct if a GPU step fails.
	dk *gpu.Buffer
	dv *gpu.Buffer
}

// kvCache stores state across all decoder layers for O(1) token-by-token generation.
type kvCache struct {
	layers []kvLayerCache
	// cpuOnly is set by the CUDA backend when a GPU step fails, forcing all
	// subsequent steps onto the CPU path for this cache.
	//lint:ignore U1000 only referenced under the cuda build tag
	cpuOnly bool
}

// Free releases device resources held by the cache (CUDA backend KV
// buffers). It is safe to call on CPU-only caches and multiple times.
func (c *kvCache) Free() {
	if c == nil {
		return
	}
	for i := range c.layers {
		c.layers[i].dk.Free()
		c.layers[i].dv.Free()
		c.layers[i].dk, c.layers[i].dv = nil, nil
	}
}

// newKVCache creates an empty KV cache for the given number of layers.
func newKVCache(numLayers int) *kvCache {
	return &kvCache{
		layers: make([]kvLayerCache, numLayers),
	}
}

// inferenceBackend owns one complete model snapshot and supports both batch
// evaluation and O(1) step generation with KV caching.
type inferenceBackend interface {
	Forward(tokenIDs []int) []float32
	NewKVCache() *kvCache
	ForwardStepKV(tokenID int, pos int, cache *kvCache) []float32
}

type cpuInferenceBackend struct {
	config Config
	vocab  int
	state  inferenceWeights
}

func newCPUInferenceBackend(model *Model) inferenceBackend {
	state := model.parameterView()
	state = cloneInferenceWeights(state)
	return &cpuInferenceBackend{config: model.Config, vocab: len(model.Tokenizer.Tokens), state: state}
}

func cloneInferenceWeights(src inferenceWeights) inferenceWeights {
	clone := func(values []float32) []float32 { return append([]float32(nil), values...) }
	cloneLayer := func(layer inferenceLayer) inferenceLayer {
		return inferenceLayer{weights: clone(layer.weights), bias: clone(layer.bias)}
	}
	blocks := make([]inferenceBlock, len(src.blocks))
	for i, b := range src.blocks {
		blocks[i] = inferenceBlock{
			normAttn: clone(b.normAttn),
			normFF:   clone(b.normFF),
			query:    cloneLayer(b.query),
			key:      cloneLayer(b.key),
			value:    cloneLayer(b.value),
			attnOut:  cloneLayer(b.attnOut),
			ffGate:   cloneLayer(b.ffGate),
			ffUp:     cloneLayer(b.ffUp),
			ffDown:   cloneLayer(b.ffDown),
		}
	}
	return inferenceWeights{
		tokenEmb: clone(src.tokenEmb),
		blocks:   blocks,
		output:   cloneLayer(src.output),
	}
}

func (backend *cpuInferenceBackend) Forward(tokenIDs []int) []float32 {
	cache := forwardFloat32(backend.config, backend.vocab, backend.state, tokenIDs)
	last := len(cache.ids) - 1
	vocab := backend.vocab
	return cache.logits[last*vocab : (last+1)*vocab]
}

func (backend *cpuInferenceBackend) NewKVCache() *kvCache {
	return newKVCache(len(backend.state.blocks))
}

func (backend *cpuInferenceBackend) ForwardStepKV(tokenID int, pos int, cache *kvCache) []float32 {
	return forwardStepKV(backend.config, backend.vocab, backend.state, tokenID, pos, cache)
}

// swish computes the Swish/SiLU non-linearity: z * sigmoid(z).
func swish(z float32) float32 {
	sig := float32(1.0 / (1.0 + math.Exp(-float64(z))))
	return z * sig
}

// swishPrime computes the derivative of Swish: sig * (1 + z*(1 - sig)).
func swishPrime(z float32) float32 {
	sig := float32(1.0 / (1.0 + math.Exp(-float64(z))))
	return sig * (1.0 + z*(1.0-sig))
}

// rmsNorm computes RMSNorm(x) * gamma for each row of x.
func rmsNorm(x, gamma []float32, rows, width int) (out, rmsVals []float32) {
	const eps = 1e-6
	out = make([]float32, rows*width)
	rmsVals = make([]float32, rows)
	for row := 0; row < rows; row++ {
		var sumSq float32
		for d := 0; d < width; d++ {
			v := x[row*width+d]
			sumSq += v * v
		}
		r := float32(math.Sqrt(float64(sumSq/float32(width)) + eps))
		rmsVals[row] = r
		for d := 0; d < width; d++ {
			out[row*width+d] = gamma[d] * x[row*width+d] / r
		}
	}
	return
}

// rmsNormBackward computes dL/dx and accumulates dL/d_gamma for one RMSNorm.
func rmsNormBackward(gradOut, x, rmsVals, gamma []float32, rows, width int, gammaGrad []float32) []float32 {
	gradX := make([]float32, rows*width)
	for row := 0; row < rows; row++ {
		r := rmsVals[row]
		for d := 0; d < width; d++ {
			gammaGrad[d] += gradOut[row*width+d] * x[row*width+d] / r
		}
		var dot float32
		for d := 0; d < width; d++ {
			dot += gradOut[row*width+d] * gamma[d] * (x[row*width+d] / r)
		}
		dot /= float32(width)
		for d := 0; d < width; d++ {
			normed := x[row*width+d] / r
			gradX[row*width+d] = (gamma[d]*gradOut[row*width+d] - normed*dot) / r
		}
	}
	return gradX
}

// applyRoPE rotates query and key vectors in-place using Rotary Position Embeddings.
func applyRoPE(q, k []float32, rows, width, headDim, heads int) {
	for row := 0; row < rows; row++ {
		pos := float64(row)
		for h := 0; h < heads; h++ {
			headOff := h * headDim
			for d := 0; d < headDim/2; d++ {
				theta := math.Pow(10000, -2*float64(d)/float64(headDim))
				cosVal := float32(math.Cos(pos * theta))
				sinVal := float32(math.Sin(pos * theta))
				qi := row*width + headOff + 2*d
				qj := qi + 1
				ki := row*width + headOff + 2*d
				kj := ki + 1
				q0, q1 := q[qi], q[qj]
				q[qi] = q0*cosVal - q1*sinVal
				q[qj] = q0*sinVal + q1*cosVal
				k0, k1 := k[ki], k[kj]
				k[ki] = k0*cosVal - k1*sinVal
				k[kj] = k0*sinVal + k1*cosVal
			}
		}
	}
}

// applyRoPESingle rotates a single token's Q and K vectors at a specific sequence position.
func applyRoPESingle(q, k []float32, pos, width, headDim, heads int) {
	posF := float64(pos)
	for h := 0; h < heads; h++ {
		headOff := h * headDim
		for d := 0; d < headDim/2; d++ {
			theta := math.Pow(10000, -2*float64(d)/float64(headDim))
			cosVal := float32(math.Cos(posF * theta))
			sinVal := float32(math.Sin(posF * theta))
			qi := headOff + 2*d
			qj := qi + 1
			ki := headOff + 2*d
			kj := ki + 1
			q0, q1 := q[qi], q[qj]
			q[qi] = q0*cosVal - q1*sinVal
			q[qj] = q0*sinVal + q1*cosVal
			k0, k1 := k[ki], k[kj]
			k[ki] = k0*cosVal - k1*sinVal
			k[kj] = k0*sinVal + k1*cosVal
		}
	}
}

// applyInverseRoPE applies the transpose (inverse) rotation R^T for the backward pass.
func applyInverseRoPE(dq, dk []float32, rows, width, headDim, heads int) {
	for row := 0; row < rows; row++ {
		pos := float64(row)
		for h := 0; h < heads; h++ {
			headOff := h * headDim
			for d := 0; d < headDim/2; d++ {
				theta := math.Pow(10000, -2*float64(d)/float64(headDim))
				cosVal := float32(math.Cos(pos * theta))
				sinVal := float32(math.Sin(pos * theta))
				qi := row*width + headOff + 2*d
				qj := qi + 1
				ki := row*width + headOff + 2*d
				kj := ki + 1
				dq0, dq1 := dq[qi], dq[qj]
				dq[qi] = dq0*cosVal + dq1*sinVal
				dq[qj] = -dq0*sinVal + dq1*cosVal
				dk0, dk1 := dk[ki], dk[kj]
				dk[ki] = dk0*cosVal + dk1*sinVal
				dk[kj] = -dk0*sinVal + dk1*cosVal
			}
		}
	}
}

// linear32 applies a dense linear layer (weights + bias) to every row.
func linear32(input []float32, rows, inputSize, outputSize int, layer inferenceLayer) []float32 {
	output := make([]float32, rows*outputSize)
	for row := 0; row < rows; row++ {
		outRow := output[row*outputSize : (row+1)*outputSize]
		copy(outRow, layer.bias)
		inRow := input[row*inputSize : (row+1)*inputSize]
		for k, val := range inRow {
			if val == 0 {
				continue
			}
			wRow := layer.weights[k*outputSize : (k+1)*outputSize]
			for col := 0; col < outputSize; col++ {
				outRow[col] += val * wRow[col]
			}
		}
	}
	return output
}

// forwardFloat32 is the pure-Go forward pass used for training: token embedding →
// stack of pre-norm transformer blocks (RMSNorm + RoPE attention + RMSNorm + SwiGLU FFN) →
// output projection → softmax probabilities.
func forwardFloat32(config Config, vocab int, weights inferenceWeights, tokenIDs []int) forwardCache {
	config.Normalize()
	if len(tokenIDs) > config.Context {
		tokenIDs = tokenIDs[len(tokenIDs)-config.Context:]
	}
	if len(tokenIDs) == 0 {
		tokenIDs = []int{0}
	}
	ids := append([]int(nil), tokenIDs...)
	width, rows := config.Width, len(ids)
	heads := config.Heads
	headDim := width / heads
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	x := make([]float32, rows*width)
	for row, id := range ids {
		if id < 0 || id >= vocab {
			id = 0
		}
		copy(x[row*width:(row+1)*width], weights.tokenEmb[id*width:(id+1)*width])
	}

	currentX := x
	bCaches := make([]blockCache, len(weights.blocks))

	for bIdx, block := range weights.blocks {
		bX := currentX

		// ── Pre-attention RMSNorm ──────────────────────────────────────
		normX1, normRms1 := rmsNorm(bX, block.normAttn, rows, width)

		// Q / K / V projections from normalized state.
		q := linear32(normX1, rows, width, width, block.query)
		k := linear32(normX1, rows, width, width, block.key)
		v := linear32(normX1, rows, width, width, block.value)

		// Apply RoPE in-place to Q and K (V is not rotated).
		applyRoPE(q, k, rows, width, headDim, heads)

		// ── Causal multi-head attention ────────────────────────────────
		attention := make([]float32, heads*rows*rows)
		for h := 0; h < heads; h++ {
			headOffset := h * headDim
			headMatrixOffset := h * rows * rows
			for i := 0; i < rows; i++ {
				maximum := float32(math.Inf(-1))
				for j := 0; j <= i; j++ {
					var score float32
					for d := 0; d < headDim; d++ {
						score += q[i*width+headOffset+d] * k[j*width+headOffset+d]
					}
					score *= scale
					attention[headMatrixOffset+i*rows+j] = score
					if score > maximum {
						maximum = score
					}
				}
				var normalizer float32
				for j := 0; j <= i; j++ {
					val := float32(math.Exp(float64(attention[headMatrixOffset+i*rows+j] - maximum)))
					attention[headMatrixOffset+i*rows+j] = val
					normalizer += val
				}
				for j := 0; j <= i; j++ {
					attention[headMatrixOffset+i*rows+j] /= normalizer
				}
			}
		}

		attended := make([]float32, rows*width)
		for h := 0; h < heads; h++ {
			headOffset := h * headDim
			headMatrixOffset := h * rows * rows
			for i := 0; i < rows; i++ {
				for d := 0; d < headDim; d++ {
					var sum float32
					for j := 0; j <= i; j++ {
						sum += attention[headMatrixOffset+i*rows+j] * v[j*width+headOffset+d]
					}
					attended[i*width+headOffset+d] = sum
				}
			}
		}

		// Attention output projection + first residual connection.
		attnProjection := linear32(attended, rows, width, width, block.attnOut)
		residual := make([]float32, rows*width)
		for i := 0; i < rows*width; i++ {
			residual[i] = bX[i] + attnProjection[i]
		}

		// ── Pre-FFN RMSNorm ────────────────────────────────────────────
		normX2, normRms2 := rmsNorm(residual, block.normFF, rows, width)

		// ── SwiGLU feed-forward network ────────────────────────────────
		hiddenDim := width * 2
		gate := linear32(normX2, rows, width, hiddenDim, block.ffGate)
		up := linear32(normX2, rows, width, hiddenDim, block.ffUp)
		activated := make([]float32, rows*hiddenDim)
		for i := 0; i < rows*hiddenDim; i++ {
			activated[i] = swish(gate[i]) * up[i]
		}
		feedForward := linear32(activated, rows, hiddenDim, width, block.ffDown)

		// Second residual connection.
		encoded := make([]float32, rows*width)
		for i := 0; i < rows*width; i++ {
			encoded[i] = residual[i] + feedForward[i]
		}

		bCaches[bIdx] = blockCache{
			x:           bX,
			normX1:      normX1,
			normRms1:    normRms1,
			q:           q,
			k:           k,
			v:           v,
			attention:   attention,
			attended:    attended,
			residual:    residual,
			normX2:      normX2,
			normRms2:    normRms2,
			ffGateOut:   gate,
			ffUpOut:     up,
			ffActivated: activated,
			encoded:     encoded,
		}
		currentX = encoded
	}

	// Final projection to vocabulary logits.
	logits := linear32(currentX, rows, width, vocab, weights.output)

	probabilities := make([]float32, rows*vocab)
	for i := 0; i < rows; i++ {
		maxLogit := float32(math.Inf(-1))
		for j := 0; j < vocab; j++ {
			if logits[i*vocab+j] > maxLogit {
				maxLogit = logits[i*vocab+j]
			}
		}
		var total float32
		for j := 0; j < vocab; j++ {
			probabilities[i*vocab+j] = float32(math.Exp(float64(logits[i*vocab+j] - maxLogit)))
			total += probabilities[i*vocab+j]
		}
		for j := 0; j < vocab; j++ {
			probabilities[i*vocab+j] /= total
		}
	}

	return forwardCache{
		ids:           ids,
		blocks:        bCaches,
		logits:        logits,
		probabilities: probabilities,
	}
}

// forwardStepKV executes a single token through the model at sequence position `pos`,
// caching K and V and performing causal attention against all cached tokens in O(1) time.
func forwardStepKV(config Config, vocab int, weights inferenceWeights, tokenID int, pos int, cache *kvCache) []float32 {
	config.Normalize()
	width := config.Width
	heads := config.Heads
	headDim := width / heads
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	if tokenID < 0 || tokenID >= vocab {
		tokenID = 0
	}
	x := make([]float32, width)
	copy(x, weights.tokenEmb[tokenID*width:(tokenID+1)*width])

	currentX := x
	for bIdx, block := range weights.blocks {
		bX := currentX

		// Pre-attention RMSNorm on single token (1 row).
		normX1, _ := rmsNorm(bX, block.normAttn, 1, width)

		q := linear32(normX1, 1, width, width, block.query)
		k := linear32(normX1, 1, width, width, block.key)
		v := linear32(normX1, 1, width, width, block.value)

		// Apply RoPE at rotation angle pos.
		applyRoPESingle(q, k, pos, width, headDim, heads)

		// Append new key and value to layer KV cache.
		cache.layers[bIdx].k = append(cache.layers[bIdx].k, k...)
		cache.layers[bIdx].v = append(cache.layers[bIdx].v, v...)

		cachedTokens := len(cache.layers[bIdx].k) / width
		kCached := cache.layers[bIdx].k
		vCached := cache.layers[bIdx].v

		// Multi-head attention against all cached tokens.
		attended := make([]float32, width)
		for h := 0; h < heads; h++ {
			headOffset := h * headDim

			maxScore := float32(math.Inf(-1))
			scores := make([]float32, cachedTokens)
			for j := 0; j < cachedTokens; j++ {
				var score float32
				for d := 0; d < headDim; d++ {
					score += q[headOffset+d] * kCached[j*width+headOffset+d]
				}
				score *= scale
				scores[j] = score
				if score > maxScore {
					maxScore = score
				}
			}

			var normalizer float32
			for j := 0; j < cachedTokens; j++ {
				val := float32(math.Exp(float64(scores[j] - maxScore)))
				scores[j] = val
				normalizer += val
			}

			for d := 0; d < headDim; d++ {
				var sum float32
				for j := 0; j < cachedTokens; j++ {
					sum += (scores[j] / normalizer) * vCached[j*width+headOffset+d]
				}
				attended[headOffset+d] = sum
			}
		}

		attnProjection := linear32(attended, 1, width, width, block.attnOut)
		residual := make([]float32, width)
		for i := 0; i < width; i++ {
			residual[i] = bX[i] + attnProjection[i]
		}

		// Pre-FFN RMSNorm.
		normX2, _ := rmsNorm(residual, block.normFF, 1, width)

		// SwiGLU feed-forward.
		hiddenDim := width * 2
		gate := linear32(normX2, 1, width, hiddenDim, block.ffGate)
		up := linear32(normX2, 1, width, hiddenDim, block.ffUp)
		activated := make([]float32, hiddenDim)
		for i := 0; i < hiddenDim; i++ {
			activated[i] = swish(gate[i]) * up[i]
		}
		feedForward := linear32(activated, 1, hiddenDim, width, block.ffDown)

		encoded := make([]float32, width)
		for i := 0; i < width; i++ {
			encoded[i] = residual[i] + feedForward[i]
		}
		currentX = encoded
	}

	logits := linear32(currentX, 1, width, vocab, weights.output)
	return logits
}
