package lm

import "math"

type inferenceLayer struct {
	weights []float32 // row-major [inputSize, outputSize]
	bias    []float32
}

type inferenceWeights struct {
	tokenEmb, posEmb []float32
	query, key, value inferenceLayer
	attnOut, ffIn, ffOut, output inferenceLayer
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
	return &cpuInferenceBackend{config: model.Config, vocab: len(model.Tokenizer.Tokens), state: state}
}

func cloneInferenceWeights(src inferenceWeights) inferenceWeights {
	clone := func(values []float32) []float32 { return append([]float32(nil), values...) }
	cloneLayer := func(layer inferenceLayer) inferenceLayer { return inferenceLayer{weights: clone(layer.weights), bias: clone(layer.bias)} }
	return inferenceWeights{
		tokenEmb: clone(src.tokenEmb), posEmb: clone(src.posEmb),
		query: cloneLayer(src.query), key: cloneLayer(src.key), value: cloneLayer(src.value),
		attnOut: cloneLayer(src.attnOut), ffIn: cloneLayer(src.ffIn), ffOut: cloneLayer(src.ffOut), output: cloneLayer(src.output),
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
		for col := 0; col < outputSize; col++ {
			sum := layer.bias[col]
			for k := 0; k < inputSize; k++ { sum += input[row*inputSize+k] * layer.weights[k*outputSize+col] }
			output[row*outputSize+col] = sum
		}
	}
	return output
}

func forwardFloat32(config Config, vocab int, weights inferenceWeights, tokenIDs []int) forwardCache {
	if len(tokenIDs) > config.Context { tokenIDs = tokenIDs[len(tokenIDs)-config.Context:] }
	if len(tokenIDs) == 0 { tokenIDs = []int{0} }
	ids := append([]int(nil), tokenIDs...)
	width, rows := config.Width, len(ids)
	x := make([]float32, rows*width)
	for row, id := range ids {
		if id < 0 || id >= vocab { id = 0 }
		for d := 0; d < width; d++ { x[row*width+d] = weights.tokenEmb[id*width+d] + weights.posEmb[row*width+d] }
	}
	q := linear32(x, rows, width, width, weights.query)
	k := linear32(x, rows, width, width, weights.key)
	v := linear32(x, rows, width, width, weights.value)
	
	// Causal attention: compute scores for all positions with masking
	scale := float32(1/math.Sqrt(float64(width)))
	attention := make([]float32, rows*rows) // [rows, rows]
	for i := 0; i < rows; i++ {
		maximum := float32(math.Inf(-1))
		for j := 0; j <= i; j++ { // causal mask: only attend to positions <= i
			var score float32
			for d := 0; d < width; d++ { score += q[i*width+d] * k[j*width+d] }
			score *= scale
			attention[i*rows+j] = score
			if score > maximum { maximum = score }
		}
		// Softmax with causal mask
		var normalizer float32
		for j := 0; j <= i; j++ {
			attention[i*rows+j] = float32(math.Exp(float64(attention[i*rows+j] - maximum)))
			normalizer += attention[i*rows+j]
		}
		for j := 0; j <= i; j++ { attention[i*rows+j] /= normalizer }
		// Positions > i get zero attention (already causal-masked)
	}
	
	// Compute attended values for all positions
	attended := make([]float32, rows*width)
	for i := 0; i < rows; i++ {
		for d := 0; d < width; d++ {
			for j := 0; j <= i; j++ {
				attended[i*width+d] += attention[i*rows+j] * v[j*width+d]
			}
		}
	}
	
	// Attention output + residual for all positions
	attnProjection := linear32(attended, rows, width, width, weights.attnOut)
	residual := make([]float32, rows*width)
	for i := 0; i < rows*width; i++ { residual[i] = x[i] + attnProjection[i] }
	
	// Feed-forward network for all positions
	hiddenPre := linear32(residual, rows, width, width*2, weights.ffIn)
	hidden := make([]float32, rows*width*2)
	for i, value := range hiddenPre { hidden[i] = float32(math.Tanh(float64(value))) }
	feedForward := linear32(hidden, rows, width*2, width, weights.ffOut)
	
	// Output projection + residual for all positions
	encoded := make([]float32, rows*width)
	for i := 0; i < rows*width; i++ { encoded[i] = residual[i] + feedForward[i] }
	
	// Logits for all positions
	logits := linear32(encoded, rows, width, vocab, weights.output)
	
	// Softmax probabilities for all positions
	probabilities := make([]float32, rows*vocab)
	for i := 0; i < rows; i++ {
		maxLogit := float32(math.Inf(-1))
		for j := 0; j < vocab; j++ {
			if logits[i*vocab+j] > maxLogit { maxLogit = logits[i*vocab+j] }
		}
		var total float32
		for j := 0; j < vocab; j++ {
			probabilities[i*vocab+j] = float32(math.Exp(float64(logits[i*vocab+j] - maxLogit)))
			total += probabilities[i*vocab+j]
		}
		for j := 0; j < vocab; j++ { probabilities[i*vocab+j] /= total }
	}
	
	return forwardCache{ids: ids, x: x, q: q, k: k, v: v, attention: attention, attended: attended, residual: residual, hidden: hidden, encoded: encoded, logits: logits, probabilities: probabilities}
}

