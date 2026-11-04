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
	return forwardFloat32(backend.config, backend.vocab, backend.state, tokenIDs).logits
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
	last, scale := rows-1, float32(1/math.Sqrt(float64(width)))
	scores := make([]float32, rows)
	maximum := float32(math.Inf(-1))
	for row := 0; row < rows; row++ {
		var score float32
		for d := 0; d < width; d++ { score += q[last*width+d] * k[row*width+d] }
		score *= scale
		scores[row] = score
		if score > maximum { maximum = score }
	}
	attention := make([]float32, rows)
	var normalizer float32
	for row, score := range scores { attention[row] = float32(math.Exp(float64(score-maximum))); normalizer += attention[row] }
	for row := range attention { attention[row] /= normalizer }
	attended := make([]float32, width)
	for row, alpha := range attention { for d := 0; d < width; d++ { attended[d] += alpha*v[row*width+d] } }
	residual := make([]float32, width)
	attnProjection := linear32(attended, 1, width, width, weights.attnOut)
	for d := range residual { residual[d] = x[last*width+d] + attnProjection[d] }
	hiddenPre := linear32(residual, 1, width, width*2, weights.ffIn)
	hidden := make([]float32, len(hiddenPre))
	for i, value := range hiddenPre { hidden[i] = float32(math.Tanh(float64(value))) }
	feedForward := linear32(hidden, 1, width*2, width, weights.ffOut)
	encoded := make([]float32, width)
	for d := range encoded { encoded[d] = residual[d] + feedForward[d] }
	logits := linear32(encoded, 1, width, vocab, weights.output)
	maxLogit := float32(math.Inf(-1))
	for _, value := range logits { if value > maxLogit { maxLogit = value } }
	probabilities := make([]float32, vocab)
	var total float32
	for i, value := range logits { probabilities[i] = float32(math.Exp(float64(value-maxLogit))); total += probabilities[i] }
	for i := range probabilities { probabilities[i] /= total }
	return forwardCache{ids: ids, x: x, q: q, k: k, v: v, attention: attention, attended: attended, residual: residual, hidden: hidden, encoded: encoded, logits: logits, probabilities: probabilities}
}

