package lm

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
)

// Config defines the multi-layer, multi-head decoder-only Transformer.
type Config struct {
	Context int `json:"context"`
	Width   int `json:"width"`
	Layers  int `json:"layers,omitempty"`
	Heads   int `json:"heads,omitempty"`
}

// Normalize ensures Layers and Heads have valid defaults.
func (c *Config) Normalize() {
	if c.Layers <= 0 {
		c.Layers = 1
	}
	if c.Heads <= 0 {
		c.Heads = 1
	}
}

// DefaultConfig fits an instruction-capable training prototype on CPU-only machines.
func DefaultConfig() Config { return Config{Context: 256, Width: 32, Layers: 2, Heads: 2} }

// parameter owns contiguous float32 values, gradients, and Adam state.
type parameter struct {
	values   []float32
	gradient []float32
	first    []float32
	second   []float32
}

type linear struct {
	weights *parameter // row-major [input, output]
	bias    *parameter
	input   int
	output  int
}

type transformerBlock struct {
	query   linear
	key     linear
	value   linear
	attnOut linear
	ffIn    linear
	ffOut   linear
}

// Model is a causal language model trained and run with contiguous float32 parameters.
type Model struct {
	Config     Config
	Tokenizer  *Tokenizer
	tokenEmb   *parameter // [vocabulary, width]
	posEmb     *parameter // [context, width]
	blocks     []transformerBlock
	output     linear
	parameters []*parameter
	step       int
	backend    inferenceBackend
}

// ParameterCount reports the number of trainable scalar weights.
func (m *Model) ParameterCount() int {
	count := 0
	for _, parameter := range m.parameters { count += len(parameter.values) }
	return count
}

func newParameter(size int) *parameter {
	return &parameter{values: make([]float32, size), gradient: make([]float32, size), first: make([]float32, size), second: make([]float32, size)}
}

func (m *Model) addParameter(parameter *parameter) *parameter {
	m.parameters = append(m.parameters, parameter)
	return parameter
}

// NewModel initializes weights with a deterministic seed.
func NewModel(config Config, tokenizer *Tokenizer, seed int64) (*Model, error) {
	config.Normalize()
	if config.Context < 2 || config.Context > 2048 || config.Width < 2 || config.Width > 256 || tokenizer == nil || len(tokenizer.Tokens) < 256 || len(tokenizer.Tokens) > 4096 {
		return nil, fmt.Errorf("invalid model config or tokenizer")
	}
	if config.Layers < 1 || config.Layers > 8 {
		return nil, fmt.Errorf("layers must be between 1 and 8")
	}
	if config.Heads < 1 || config.Heads > 16 || config.Width%config.Heads != 0 {
		return nil, fmt.Errorf("width %d must be divisible by heads %d", config.Width, config.Heads)
	}
	rng := rand.New(rand.NewSource(seed))
	m := &Model{Config: config, Tokenizer: tokenizer}
	newMatrix := func(rows, cols int) *parameter {
		parameter := newParameter(rows * cols)
		limit := float32(1 / math.Sqrt(float64(cols)))
		for i := range parameter.values { parameter.values[i] = (rng.Float32()*2 - 1) * limit }
		return m.addParameter(parameter)
	}
	newLinear := func(input, output int) linear {
		weights := newMatrix(input, output)
		bias := m.addParameter(newParameter(output))
		return linear{weights: weights, bias: bias, input: input, output: output}
	}
	width, vocab := config.Width, len(tokenizer.Tokens)
	m.tokenEmb = newMatrix(vocab, width)
	m.posEmb = newMatrix(config.Context, width)
	m.blocks = make([]transformerBlock, config.Layers)
	for l := 0; l < config.Layers; l++ {
		m.blocks[l] = transformerBlock{
			query:   newLinear(width, width),
			key:     newLinear(width, width),
			value:   newLinear(width, width),
			attnOut: newLinear(width, width),
			ffIn:    newLinear(width, width*2),
			ffOut:   newLinear(width*2, width),
		}
	}
	m.output = newLinear(width, vocab)
	return m, nil
}

func (m *Model) parameterView() inferenceWeights {
	m.Config.Normalize()
	blocks := make([]inferenceBlock, len(m.blocks))
	for i, b := range m.blocks {
		blocks[i] = inferenceBlock{
			query:   inferenceLayer{weights: b.query.weights.values, bias: b.query.bias.values},
			key:     inferenceLayer{weights: b.key.weights.values, bias: b.key.bias.values},
			value:   inferenceLayer{weights: b.value.weights.values, bias: b.value.bias.values},
			attnOut: inferenceLayer{weights: b.attnOut.weights.values, bias: b.attnOut.bias.values},
			ffIn:    inferenceLayer{weights: b.ffIn.weights.values, bias: b.ffIn.bias.values},
			ffOut:   inferenceLayer{weights: b.ffOut.weights.values, bias: b.ffOut.bias.values},
		}
	}
	return inferenceWeights{
		tokenEmb: m.tokenEmb.values,
		posEmb:   m.posEmb.values,
		blocks:   blocks,
		output:   inferenceLayer{weights: m.output.weights.values, bias: m.output.bias.values},
	}
}

func (m *Model) forward(ids []int) forwardCache {
	return forwardFloat32(m.Config, len(m.Tokenizer.Tokens), m.parameterView(), ids)
}

func (m *Model) backward(cache forwardCache, targets []int, mask []bool) {
	m.Config.Normalize()
	width, vocab, rows := m.Config.Width, len(m.Tokenizer.Tokens), len(cache.ids)
	heads := m.Config.Heads
	headDim := width / heads
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	activeCount := rows
	if mask != nil {
		activeCount = 0
		for i := 0; i < rows; i++ {
			if i < len(mask) && mask[i] {
				activeCount++
			}
		}
		if activeCount == 0 {
			return
		}
	}
	invRows := float32(1.0 / float64(activeCount))

	// 1. Logits layer gradient: softmax derivative at active positions
	gradientLogits := make([]float32, rows*vocab)
	for i := 0; i < rows; i++ {
		if mask != nil && (i >= len(mask) || !mask[i]) {
			continue
		}
		for j := 0; j < vocab; j++ {
			gradientLogits[i*vocab+j] = cache.probabilities[i*vocab+j] * invRows
		}
		gradientLogits[i*vocab+targets[i]] -= invRows
	}

	// 2. Output linear layer backprop
	lastBlockIdx := len(m.blocks) - 1
	gradientCurrentX := make([]float32, rows*width)
	for i := 0; i < rows; i++ {
		for d := 0; d < width; d++ {
			for token := 0; token < vocab; token++ {
				index := d*vocab + token
				m.output.weights.gradient[index] += cache.blocks[lastBlockIdx].encoded[i*width+d] * gradientLogits[i*vocab+token]
				gradientCurrentX[i*width+d] += m.output.weights.values[index] * gradientLogits[i*vocab+token]
			}
		}
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < vocab; j++ {
			m.output.bias.gradient[j] += gradientLogits[i*vocab+j]
		}
	}

	// 3. Backprop through transformer blocks in reverse order
	for l := len(m.blocks) - 1; l >= 0; l-- {
		block := m.blocks[l]
		bcache := cache.blocks[l]
		gradientEncoded := gradientCurrentX

		// Backprop through FF network
		gradientResidual := make([]float32, rows*width)
		copy(gradientResidual, gradientEncoded)
		gradientHidden := make([]float32, rows*width*2)

		for i := 0; i < rows; i++ {
			for h := 0; h < width*2; h++ {
				for d := 0; d < width; d++ {
					index := h*width + d
					block.ffOut.weights.gradient[index] += bcache.hidden[i*width*2+h] * gradientEncoded[i*width+d]
					gradientHidden[i*width*2+h] += block.ffOut.weights.values[index] * gradientEncoded[i*width+d]
				}
			}
		}
		for i := 0; i < rows; i++ {
			for d := 0; d < width; d++ {
				block.ffOut.bias.gradient[d] += gradientEncoded[i*width+d]
			}
		}

		for i := 0; i < rows; i++ {
			for h := 0; h < width*2; h++ {
				gradientHidden[i*width*2+h] *= 1 - bcache.hidden[i*width*2+h]*bcache.hidden[i*width*2+h]
			}
			for h := 0; h < width*2; h++ {
				block.ffIn.bias.gradient[h] += gradientHidden[i*width*2+h]
			}
			for d := 0; d < width; d++ {
				for h := 0; h < width*2; h++ {
					index := d*width*2 + h
					block.ffIn.weights.gradient[index] += bcache.residual[i*width+d] * gradientHidden[i*width*2+h]
					gradientResidual[i*width+d] += block.ffIn.weights.values[index] * gradientHidden[i*width*2+h]
				}
			}
		}

		// Backprop through attention output projection
		gradientAttended := make([]float32, rows*width)
		for i := 0; i < rows; i++ {
			for input := 0; input < width; input++ {
				for output := 0; output < width; output++ {
					index := input*width + output
					block.attnOut.weights.gradient[index] += bcache.attended[i*width+input] * gradientResidual[i*width+output]
					gradientAttended[i*width+input] += block.attnOut.weights.values[index] * gradientResidual[i*width+output]
				}
			}
		}
		for i := 0; i < rows; i++ {
			for output := 0; output < width; output++ {
				block.attnOut.bias.gradient[output] += gradientResidual[i*width+output]
			}
		}

		// Backprop through multi-head causal attention
		gradientX := make([]float32, rows*width)
		copy(gradientX, gradientResidual)
		gradientQuery := make([]float32, rows*width)
		gradientKeys := make([]float32, rows*width)
		gradientValues := make([]float32, rows*width)

		gradientAttention := make([]float32, heads*rows*rows)
		for h := 0; h < heads; h++ {
			headOffset := h * headDim
			headMatrixOffset := h * rows * rows
			for i := 0; i < rows; i++ {
				for d := 0; d < headDim; d++ {
					for j := 0; j <= i; j++ {
						gradientAttention[headMatrixOffset+i*rows+j] += gradientAttended[i*width+headOffset+d] * bcache.v[j*width+headOffset+d]
					}
				}
			}
		}

		for h := 0; h < heads; h++ {
			headOffset := h * headDim
			headMatrixOffset := h * rows * rows
			for i := 0; i < rows; i++ {
				var weightedGradient float32
				for j := 0; j <= i; j++ {
					weightedGradient += bcache.attention[headMatrixOffset+i*rows+j] * gradientAttention[headMatrixOffset+i*rows+j]
				}
				for j := 0; j <= i; j++ {
					gradientScore := bcache.attention[headMatrixOffset+i*rows+j] * (gradientAttention[headMatrixOffset+i*rows+j] - weightedGradient)
					for d := 0; d < headDim; d++ {
						gradientQuery[i*width+headOffset+d] += gradientScore * bcache.k[j*width+headOffset+d] * scale
						gradientKeys[j*width+headOffset+d] += gradientScore * bcache.q[i*width+headOffset+d] * scale
						gradientValues[j*width+headOffset+d] += bcache.attention[headMatrixOffset+i*rows+j] * gradientAttended[i*width+headOffset+d]
					}
				}
			}
		}

		// Backprop through QKV linear projections
		for input := 0; input < width; input++ {
			for output := 0; output < width; output++ {
				index := input*width + output
				for i := 0; i < rows; i++ {
					block.query.weights.gradient[index] += bcache.x[i*width+input] * gradientQuery[i*width+output]
					gradientX[i*width+input] += block.query.weights.values[index] * gradientQuery[i*width+output]
					block.key.weights.gradient[index] += bcache.x[i*width+input] * gradientKeys[i*width+output]
					gradientX[i*width+input] += block.key.weights.values[index] * gradientKeys[i*width+output]
					block.value.weights.gradient[index] += bcache.x[i*width+input] * gradientValues[i*width+output]
					gradientX[i*width+input] += block.value.weights.values[index] * gradientValues[i*width+output]
				}
			}
		}
		for i := 0; i < rows; i++ {
			for output := 0; output < width; output++ {
				block.query.bias.gradient[output] += gradientQuery[i*width+output]
				block.key.bias.gradient[output] += gradientKeys[i*width+output]
				block.value.bias.gradient[output] += gradientValues[i*width+output]
			}
		}

		// gradientX flows into preceding layer
		gradientCurrentX = gradientX
	}

	// 4. Backprop into token and position embeddings from block 0
	for row, id := range cache.ids {
		for d := 0; d < width; d++ {
			m.tokenEmb.gradient[id*width+d] += gradientCurrentX[row*width+d]
			m.posEmb.gradient[row*width+d] += gradientCurrentX[row*width+d]
		}
	}
}

func addInPlace(dst, src []float32) { for i := range dst { dst[i] += src[i] } }
func zeroGradients(parameters []*parameter) { for _, p := range parameters { clear(p.gradient) } }

func (m *Model) adamStep(learningRate float32) {
	m.step++
	beta1, beta2 := float32(0.9), float32(0.999)
	bias1 := float32(1 - math.Pow(float64(beta1), float64(m.step)))
	bias2 := float32(1 - math.Pow(float64(beta2), float64(m.step)))
	for _, p := range m.parameters {
		for i, gradient := range p.gradient {
			p.first[i] = beta1*p.first[i] + (1-beta1)*gradient
			p.second[i] = beta2*p.second[i] + (1-beta2)*gradient*gradient
			first := p.first[i] / bias1
			second := p.second[i] / bias2
			p.values[i] -= learningRate * first / (float32(math.Sqrt(float64(second))) + 1e-8)
		}
	}
}

func tokenLoss32(logits []float32, target int) float64 {
	maximum := float32(math.Inf(-1))
	for _, value := range logits { if value > maximum { maximum = value } }
	var total float64
	for _, value := range logits { total += math.Exp(float64(value-maximum)) }
	return math.Log(total) + float64(maximum) - float64(logits[target])
}

// Mean window loss: average cross-entropy over active positions (all if mask is nil)
func meanWindowLoss(cache forwardCache, targets []int, mask []bool) float64 {
	if len(targets) == 0 || len(cache.ids) == 0 {
		return 0
	}
	rows := len(cache.ids)
	if len(targets) < rows {
		rows = len(targets)
	}
	vocab := len(cache.logits) / len(cache.ids)
	totalLoss := 0.0
	activeCount := 0
	for i := 0; i < rows; i++ {
		if mask != nil && (i >= len(mask) || !mask[i]) {
			continue
		}
		rowLogits := cache.logits[i*vocab : (i+1)*vocab]
		totalLoss += tokenLoss32(rowLogits, targets[i])
		activeCount++
	}
	if activeCount == 0 {
		return 0
	}
	return totalLoss / float64(activeCount)
}

// Train performs causal full-window training with float32 parameters and Adam updates.
// It returns average pre-update cross-entropy per epoch.
func (m *Model) Train(texts []string, epochs, stepsPerEpoch int, learningRate float64) ([]float64, error) {
	if epochs <= 0 || stepsPerEpoch <= 0 || learningRate <= 0 || math.IsNaN(learningRate) || math.IsInf(learningRate, 0) { return nil, fmt.Errorf("epochs, steps, and finite positive learning rate are required") }
	m.backend = nil
	var ids []int
	for _, text := range texts { ids = append(ids, m.Tokenizer.Encode(text)...); ids = append(ids, m.Tokenizer.Encode("\n")...) }
	if len(ids) < 3 { return nil, fmt.Errorf("training corpus must encode to at least three tokens") }
	rng := rand.New(rand.NewSource(71))
	losses := make([]float64, 0, epochs)
	for epoch := 0; epoch < epochs; epoch++ {
		var totalLoss float64
		for step := 0; step < stepsPerEpoch; step++ {
			start := rng.Intn(len(ids)-1)
			end := start + m.Config.Context + 1
			if end > len(ids) { end = len(ids) }
			window := ids[start:end]
			if len(window) < 2 { continue }
			cache := m.forward(window[:len(window)-1])
			targets := window[1:]
			totalLoss += meanWindowLoss(cache, targets, nil)
			zeroGradients(m.parameters)
			m.backward(cache, targets, nil)
			m.adamStep(float32(learningRate))
		}
		losses = append(losses, totalLoss/float64(stepsPerEpoch))
	}
	return losses, nil
}

// TrainInstructions trains the model on instruction prompt-completion pairs
// using prompt loss masking. Gradients are only computed on completion tokens.
func (m *Model) TrainInstructions(examples []InstructionExample, epochs, stepsPerEpoch int, learningRate float64) ([]float64, error) {
	if epochs <= 0 || stepsPerEpoch <= 0 || learningRate <= 0 || math.IsNaN(learningRate) || math.IsInf(learningRate, 0) {
		return nil, fmt.Errorf("epochs, steps, and finite positive learning rate are required")
	}
	if len(examples) == 0 {
		return nil, fmt.Errorf("instruction examples required")
	}
	m.backend = nil

	type preparedExample struct {
		input   []int
		targets []int
		mask    []bool
	}

	var prepared []preparedExample
	for _, ex := range examples {
		promptIDs := m.Tokenizer.Encode(ex.Prompt)
		completionIDs := m.Tokenizer.Encode(" " + strings.TrimSpace(ex.Completion))
		ids := append(promptIDs, completionIDs...)
		if len(ids) < 2 {
			continue
		}
		// Take context window ending at completion
		start := 0
		if len(ids) > m.Config.Context+1 {
			start = len(ids) - (m.Config.Context + 1)
		}
		window := ids[start:]
		promptLen := len(promptIDs) - start
		if promptLen < 0 {
			promptLen = 0
		}
		input := window[:len(window)-1]
		targets := window[1:]
		mask := make([]bool, len(targets))
		hasActive := false
		for i := range targets {
			// Target i predicts window[i+1]; active if it's in completion
			if i+1 >= promptLen {
				mask[i] = true
				hasActive = true
			}
		}
		if hasActive {
			prepared = append(prepared, preparedExample{input: input, targets: targets, mask: mask})
		}
	}

	if len(prepared) == 0 {
		return nil, fmt.Errorf("no valid instruction examples within context length")
	}

	rng := rand.New(rand.NewSource(71))
	losses := make([]float64, 0, epochs)
	for epoch := 0; epoch < epochs; epoch++ {
		var totalLoss float64
		for step := 0; step < stepsPerEpoch; step++ {
			item := prepared[rng.Intn(len(prepared))]
			cache := m.forward(item.input)
			loss := meanWindowLoss(cache, item.targets, item.mask)
			totalLoss += loss
			zeroGradients(m.parameters)
			m.backward(cache, item.targets, item.mask)
			m.adamStep(float32(learningRate))
		}
		losses = append(losses, totalLoss/float64(stepsPerEpoch))
	}
	return losses, nil
}

// Evaluate returns mean next-token cross-entropy over a held-out corpus.
func (m *Model) Evaluate(texts []string) (float64, error) {
	var ids []int
	for _, text := range texts { ids = append(ids, m.Tokenizer.Encode(text)...); ids = append(ids, m.Tokenizer.Encode("\n")...) }
	if len(ids) < 2 { return 0, fmt.Errorf("evaluation corpus must encode to at least two tokens") }
	backend := m.inferenceBackend()
	var total float64
	for target := 1; target < len(ids); target++ {
		start := target-m.Config.Context
		if start < 0 { start = 0 }
		total += tokenLoss32(backend.Forward(ids[start:target]), ids[target])
	}
	return total/float64(len(ids)-1), nil
}

// GenerateOptions configures text generation parameters.
type GenerateOptions struct {
	MaxTokens         int
	Temperature       float64
	Seed              int64
	RepetitionPenalty float64 // > 1.0 penalizes already generated tokens (e.g. 1.15)
}

// GenerateWithOptions generates tokens using configurable sampling, repetition penalty,
// and UTF-8 validation.
func (m *Model) GenerateWithOptions(prompt string, opts GenerateOptions) (string, error) {
	if opts.MaxTokens < 0 || opts.Temperature <= 0 || math.IsNaN(opts.Temperature) || math.IsInf(opts.Temperature, 0) {
		return "", fmt.Errorf("max tokens must be non-negative and temperature finite and positive")
	}
	penalty := float32(opts.RepetitionPenalty)
	if penalty <= 0 {
		penalty = 1.0
	}
	ids := m.Tokenizer.Encode(prompt)
	if len(ids) == 0 {
		ids = m.Tokenizer.Encode(" ")
	}
	prefix := []byte(m.Tokenizer.Decode(ids))
	backend, rng := m.inferenceBackend(), rand.New(rand.NewSource(opts.Seed))

	generatedCounts := make(map[int]int)

	for n := 0; n < opts.MaxTokens; n++ {
		rawLogits := backend.Forward(ids)
		logits := append([]float32(nil), rawLogits...)

		if penalty > 1.0 {
			for id, count := range generatedCounts {
				if id >= 0 && id < len(logits) && count > 0 {
					tok := m.Tokenizer.Tokens[id]
					if tok == "{" || tok == "}" || tok == "[" || tok == "]" || tok == ":" || tok == "\"" || tok == "," || tok == " " || tok == "\n" {
						continue
					}
					if logits[id] > 0 {
						logits[id] /= penalty
					} else {
						logits[id] *= penalty
					}
				}
			}
		}

		maximum := float32(math.Inf(-1))
		for _, value := range logits {
			if value > maximum {
				maximum = value
			}
		}
		probabilities := make([]float64, len(logits))
		var sum float64
		for id, value := range logits {
			if id >= len(m.Tokenizer.Tokens) || !validUTF8Append(prefix, m.Tokenizer.Tokens[id]) {
				continue
			}
			probabilities[id] = math.Exp(float64(value-maximum) / opts.Temperature)
			sum += probabilities[id]
		}
		if sum == 0 {
			for id, token := range m.Tokenizer.Tokens {
				if validUTF8Append(prefix, token) {
					probabilities[id], sum = 1, 1
					break
				}
			}
		}
		draw, choice := rng.Float64()*sum, 0
		for id, probability := range probabilities {
			if probability == 0 {
				continue
			}
			if draw < probability {
				choice = id
				break
			}
			draw -= probability
		}
		ids = append(ids, choice)
		generatedCounts[choice]++
		prefix = appendString(prefix, m.Tokenizer.Tokens[choice])
	}
	return strings.ToValidUTF8(string(prefix), ""), nil
}

// Generate samples tokens at the requested temperature using deterministic
// seeded sampling. Sampling stays on valid printable UTF-8 boundaries.
func (m *Model) Generate(prompt string, maxTokens int, temperature float64, seed int64) (string, error) {
	return m.GenerateWithOptions(prompt, GenerateOptions{
		MaxTokens:         maxTokens,
		Temperature:       temperature,
		Seed:              seed,
		RepetitionPenalty: 1.15,
	})
}

func (m *Model) inferenceBackend() inferenceBackend {
	if m.backend == nil { m.backend = newCPUInferenceBackend(m) }
	return m.backend
}

func validUTF8Prefix(data []byte) bool { return validUTF8Append(data, "") }

func validUTF8Append(prefix []byte, suffix string) bool {
	total := len(prefix)+len(suffix)
	byteAt := func(index int) byte { if index < len(prefix) { return prefix[index] }; return suffix[index-len(prefix)] }
	for i := 0; i < total; {
		first := byteAt(i)
		if first < 0x80 {
			if (first < 0x20 && first != '\n' && first != '\r' && first != '\t') || first == 0x7F { return false }
			i++; continue
		}
		length := 0
		switch { case first >= 0xC2 && first <= 0xDF: length=2; case first >= 0xE0 && first <= 0xEF: length=3; case first >= 0xF0 && first <= 0xF4: length=4; default: return false }
		available, check := total-i, length
		incomplete := available < length
		if incomplete { check=available }
		for offset := 1; offset < check; offset++ {
			b := byteAt(i+offset)
			if b < 0x80 || b > 0xBF { return false }
			if offset == 1 && ((first == 0xE0 && b < 0xA0) || (first == 0xED && b > 0x9F) || (first == 0xF0 && b < 0x90) || (first == 0xF4 && b > 0x8F)) { return false }
		}
		if incomplete { return true }
		i += length
	}
	return true
}

func appendString(prefix []byte, suffix string) []byte { for i := 0; i < len(suffix); i++ { prefix=append(prefix,suffix[i]) }; return prefix }

type checkpoint struct {
	Config Config `json:"config"`
	Tokens [][]byte `json:"tokens"`
	Merges []checkpointMerge `json:"merges"`
	Values []float64 `json:"values"`
}
type checkpointMerge struct { Left, Right, Token []byte }

// SaveCheckpoint writes tokenizer and float32 weights to a local JSON checkpoint.
func (m *Model) SaveCheckpoint(path string) error {
	state := checkpoint{Config:m.Config, Tokens:make([][]byte,len(m.Tokenizer.Tokens)), Merges:make([]checkpointMerge,len(m.Tokenizer.Merges))}
	for i, token := range m.Tokenizer.Tokens { state.Tokens[i]=[]byte(token) }
	for i, merge := range m.Tokenizer.Merges { state.Merges[i]=checkpointMerge{[]byte(merge.Left),[]byte(merge.Right),[]byte(merge.Token)} }
	for _, p := range m.parameters {
		for _, value := range p.values {
			state.Values = append(state.Values, float64(value))
		}
	}
	data, err := json.Marshal(state); if err != nil { return err }
	if err := os.MkdirAll(filepath.Dir(path),0o700); err != nil { return err }
	tmp := path+".tmp"
	if err := os.WriteFile(tmp,data,0o600); err != nil { return err }
	return os.Rename(tmp,path)
}

// LoadCheckpoint restores model topology, tokenizer, and weights. Previous
// checkpoints with float64 JSON parameter values remain readable.
func LoadCheckpoint(path string) (*Model,error) {
	data,err:=os.ReadFile(path); if err!=nil{return nil,err}
	var state checkpoint
	if err:=json.Unmarshal(data,&state);err!=nil{return nil,fmt.Errorf("decode checkpoint: %w",err)}
	tokenizer:=&Tokenizer{Tokens:make([]string,len(state.Tokens)),Merges:make([]Merge,len(state.Merges))}
	for i,token:=range state.Tokens{tokenizer.Tokens[i]=string(token)}
	for i,merge:=range state.Merges{tokenizer.Merges[i]=Merge{string(merge.Left),string(merge.Right),string(merge.Token)}}
	tokenizer.reindex()
	model,err:=NewModel(state.Config,tokenizer,1);if err!=nil{return nil,err}
	if len(state.Values)!=model.ParameterCount(){return nil,fmt.Errorf("checkpoint has %d parameters; model requires %d",len(state.Values),model.ParameterCount())}
	index:=0
	for _,p:=range model.parameters{
		for i:=range p.values {
			p.values[i]=float32(state.Values[index])
			index++
		}
	}
	return model,nil
}
