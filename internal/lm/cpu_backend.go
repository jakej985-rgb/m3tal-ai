package lm

import "math"

type inferenceLayer struct {
	weights []float32 // row-major [inputSize, outputSize]
	bias    []float32
}

type inferenceBlock struct {
	query, key, value inferenceLayer
	attnOut           inferenceLayer
	ffIn, ffOut       inferenceLayer
}

type inferenceWeights struct {
	tokenEmb, posEmb []float32
	blocks           []inferenceBlock
	output           inferenceLayer
}

type blockCache struct {
	x         []float32 // [rows, width]
	q, k, v   []float32 // [rows, width]
	attention []float32 // [heads, rows, rows]
	attended  []float32 // [rows, width]
	residual  []float32 // [rows, width]
	hidden    []float32 // [rows, width*2]
	encoded   []float32 // [rows, width]
}

type forwardCache struct {
	ids           []int
	blocks        []blockCache
	logits        []float32 // [rows, vocab]
	probabilities []float32 // [rows, vocab]
}

// inferenceBackend owns one complete model snapshot. A GPU implementation can
// retain its weights and intermediate state for the lifetime of a session.
type inferenceBackend interface {
	Forward(tokenIDs []int) []float32
}

type cpuInferenceBackend struct {
	config Config
	vocab  int
	state  inferenceWeights
}

func newCPUInferenceBackend(model *Model) inferenceBackend {
	state := model.parameterView()
	state = cloneInferenceWeights(state)
	return &cpuInferenceBackend{config: model.Config, vocab: model.VocabSize(), state: state}
}

func cloneInferenceWeights(src inferenceWeights) inferenceWeights {
	clone := func(values []float32) []float32 { return append([]float32(nil), values...) }
	cloneLayer := func(layer inferenceLayer) inferenceLayer {
		return inferenceLayer{weights: clone(layer.weights), bias: clone(layer.bias)}
	}
	blocks := make([]inferenceBlock, len(src.blocks))
	for i, b := range src.blocks {
		blocks[i] = inferenceBlock{
			query:   cloneLayer(b.query),
			key:     cloneLayer(b.key),
			value:   cloneLayer(b.value),
			attnOut: cloneLayer(b.attnOut),
			ffIn:    cloneLayer(b.ffIn),
			ffOut:   cloneLayer(b.ffOut),
		}
	}
	return inferenceWeights{
		tokenEmb: clone(src.tokenEmb),
		posEmb:   clone(src.posEmb),
		blocks:   blocks,
		output:   cloneLayer(src.output),
	}
}

func (backend *cpuInferenceBackend) Forward(tokenIDs []int) []float32 {
	cache := forwardFloat32(backend.config, backend.vocab, backend.state, tokenIDs)
	// Return only the last position's logits for generation (backward compatibility)
	last := len(cache.ids) - 1
	vocab := backend.vocab
	return cache.logits[last*vocab : (last+1)*vocab]
}

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
		for d := 0; d < width; d++ {
			x[row*width+d] = weights.tokenEmb[id*width+d] + weights.posEmb[row*width+d]
		}
	}

	currentX := x
	bCaches := make([]blockCache, len(weights.blocks))

	for bIdx, block := range weights.blocks {
		bX := currentX
		q := linear32(bX, rows, width, width, block.query)
		k := linear32(bX, rows, width, width, block.key)
		v := linear32(bX, rows, width, width, block.value)

		// Multi-head causal attention
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

		// Attended values
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

		// Attention output projection + residual
		attnProjection := linear32(attended, rows, width, width, block.attnOut)
		residual := make([]float32, rows*width)
		for i := 0; i < rows*width; i++ {
			residual[i] = bX[i] + attnProjection[i]
		}

		// Feed-forward network
		hiddenPre := linear32(residual, rows, width, width*2, block.ffIn)
		hidden := make([]float32, rows*width*2)
		for i, value := range hiddenPre {
			hidden[i] = float32(math.Tanh(float64(value)))
		}
		feedForward := linear32(hidden, rows, width*2, width, block.ffOut)

		// Output projection + residual
		encoded := make([]float32, rows*width)
		for i := 0; i < rows*width; i++ {
			encoded[i] = residual[i] + feedForward[i]
		}

		bCaches[bIdx] = blockCache{
			x:         bX,
			q:         q,
			k:         k,
			v:         v,
			attention: attention,
			attended:  attended,
			residual:  residual,
			hidden:    hidden,
			encoded:   encoded,
		}

		currentX = encoded
	}

	// Final output layer
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
