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

// Config defines the small one-block, one-head decoder-only Transformer.
type Config struct {
	Context int `json:"context"`
	Width   int `json:"width"`
}

// DefaultConfig fits a small training prototype on CPU-only machines.
func DefaultConfig() Config { return Config{Context: 32, Width: 16} }

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

// Model is a small causal language model trained and run with contiguous
// float32 parameters. Training uses explicit backpropagation for its fixed
// one-block architecture.
type Model struct {
	Config    Config
	Tokenizer *Tokenizer
	tokenEmb  *parameter // [vocabulary, width]
	posEmb    *parameter // [context, width]
	query     linear
	key       linear
	value     linear
	attnOut   linear
	ffIn      linear
	ffOut     linear
	output    linear
	parameters []*parameter
	step      int
	backend   inferenceBackend
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
	if config.Context < 2 || config.Context > 256 || config.Width < 2 || config.Width > 64 || tokenizer == nil || len(tokenizer.Tokens) < 256 || len(tokenizer.Tokens) > 2048 {
		return nil, fmt.Errorf("invalid model config or tokenizer")
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
	m.query = newLinear(width, width)
	m.key = newLinear(width, width)
	m.value = newLinear(width, width)
	m.attnOut = newLinear(width, width)
	m.ffIn = newLinear(width, width*2)
	m.ffOut = newLinear(width*2, width)
	m.output = newLinear(width, vocab)
	return m, nil
}

type forwardCache struct {
	ids       []int
	x, q, k, v []float32
	attention []float32 // [rows, rows] - attention scores for causal masking
	attended  []float32 // [rows, width] - attended values per position
	residual  []float32 // [rows, width] - residual connections per position
	hidden    []float32 // [rows, width*2] - hidden activations per position
	encoded   []float32 // [rows, width] - encoded representations per position
	logits    []float32 // [rows, vocab] - logits for all positions
	probabilities []float32 // [rows, vocab] - softmax probabilities for all positions
}

func (m *Model) parameterView() inferenceWeights {
	return inferenceWeights{
		tokenEmb: m.tokenEmb.values, posEmb: m.posEmb.values,
		query: inferenceLayer{weights: m.query.weights.values, bias: m.query.bias.values},
		key: inferenceLayer{weights: m.key.weights.values, bias: m.key.bias.values},
		value: inferenceLayer{weights: m.value.weights.values, bias: m.value.bias.values},
		attnOut: inferenceLayer{weights: m.attnOut.weights.values, bias: m.attnOut.bias.values},
		ffIn: inferenceLayer{weights: m.ffIn.weights.values, bias: m.ffIn.bias.values},
		ffOut: inferenceLayer{weights: m.ffOut.weights.values, bias: m.ffOut.bias.values},
		output: inferenceLayer{weights: m.output.weights.values, bias: m.output.bias.values},
	}
}

func (m *Model) forward(ids []int) forwardCache {
	return forwardFloat32(m.Config, len(m.Tokenizer.Tokens), m.parameterView(), ids)
}

func (m *Model) backward(cache forwardCache, targets []int) {
	width, vocab, rows := m.Config.Width, len(m.Tokenizer.Tokens), len(cache.ids)
	
	// Gradient for logits: softmax derivative at all positions
	gradientLogits := make([]float32, rows*vocab)
	for i := 0; i < rows; i++ {
		for j := 0; j < vocab; j++ {
			gradientLogits[i*vocab+j] = cache.probabilities[i*vocab+j]
		}
		gradientLogits[i*vocab+targets[i]] -= 1
	}
	
	// Backprop through output layer (all positions)
	gradientEncoded := make([]float32, rows*width)
	for i := 0; i < rows; i++ {
		for d := 0; d < width; d++ {
			for token := 0; token < vocab; token++ {
				index := d*vocab + token
				m.output.weights.gradient[index] += cache.encoded[i*width+d] * gradientLogits[i*vocab+token]
				gradientEncoded[i*width+d] += m.output.weights.values[index] * gradientLogits[i*vocab+token]
			}
		}
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < vocab; j++ {
			m.output.bias.gradient[j] += gradientLogits[i*vocab+j]
		}
	}
	
	// Backprop through feed-forward network (all positions)
	gradientResidual := make([]float32, rows*width)
	copy(gradientResidual, gradientEncoded)
	gradientHidden := make([]float32, rows*width*2)
	for i := 0; i < rows; i++ {
		for h := 0; h < width*2; h++ {
			for d := 0; d < width; d++ {
				index := h*width + d
				m.ffOut.weights.gradient[index] += cache.hidden[i*width*2+h] * gradientEncoded[i*width+d]
				gradientHidden[i*width*2+h] += m.ffOut.weights.values[index] * gradientEncoded[i*width+d]
			}
		}
	}
	for i := 0; i < rows; i++ {
		for h := 0; h < width*2; h++ {
			m.ffIn.bias.gradient[h] += gradientHidden[i*width*2+h]
		}
	}
	for i := 0; i < rows; i++ {
		for h := 0; h < width*2; h++ {
			gradientHidden[i*width*2+h] *= 1 - cache.hidden[i*width*2+h]*cache.hidden[i*width*2+h]
		}
		for d := 0; d < width; d++ {
			for h := 0; h < width*2; h++ {
				index := d*width*2 + h
				m.ffIn.weights.gradient[index] += cache.residual[i*width+d] * gradientHidden[i*width*2+h]
				gradientResidual[i*width+d] += m.ffIn.weights.values[index] * gradientHidden[i*width*2+h]
			}
		}
	}
	
	// Backprop through attention (all positions)
	gradientAttended := make([]float32, rows*width)
	for i := 0; i < rows; i++ {
		for input := 0; input < width; input++ {
			for output := 0; output < width; output++ {
				index := input*width + output
				m.attnOut.weights.gradient[index] += cache.attended[i*width+input] * gradientResidual[i*width+output]
				gradientAttended[i*width+input] += m.attnOut.weights.values[index] * gradientResidual[i*width+output]
			}
		}
	}
	for i := 0; i < rows; i++ {
		for output := 0; output < width; output++ {
			m.attnOut.bias.gradient[output] += gradientResidual[i*width+output]
		}
	}
	
	// Backprop through attention weights (causal)
	gradientAttention := make([]float32, rows*rows)
	for i := 0; i < rows; i++ {
		for d := 0; d < width; d++ {
			for j := 0; j <= i; j++ {
				gradientAttention[i*rows+j] += gradientAttended[i*width+d] * cache.v[j*width+d]
			}
		}
	}
	
	gradientX := make([]float32, rows*width)
	gradientQuery := make([]float32, rows*width)
	gradientKeys := make([]float32, rows*width)
	gradientValues := make([]float32, rows*width)
	scale := float32(1 / math.Sqrt(float64(width)))
	
	for i := 0; i < rows; i++ {
		// Compute attention gradient contribution
		var weightedGradient float32
		for j := 0; j <= i; j++ {
			weightedGradient += cache.attention[i*rows+j] * gradientAttention[i*rows+j]
		}
		
		for j := 0; j <= i; j++ {
			gradientScore := cache.attention[i*rows+j] * (gradientAttention[i*rows+j] - weightedGradient)
			for output := 0; output < width; output++ {
				gradientQuery[i*width+output] += gradientScore * cache.k[j*width+output] * scale
				gradientKeys[j*width+output] += gradientScore * cache.q[i*width+output] * scale
				gradientValues[j*width+output] += cache.attention[i*rows+j] * gradientAttended[i*width+output]
			}
		}
	}
	
	// Backprop through QKV projections
	for input := 0; input < width; input++ {
		for output := 0; output < width; output++ {
			index := input*width + output
			for i := 0; i < rows; i++ {
				m.query.weights.gradient[index] += cache.x[i*width+input] * gradientQuery[i*width+output]
				gradientX[i*width+input] += m.query.weights.values[index] * gradientQuery[i*width+output]
				m.key.weights.gradient[index] += cache.x[i*width+input] * gradientKeys[i*width+output]
				gradientX[i*width+input] += m.key.weights.values[index] * gradientKeys[i*width+output]
				m.value.weights.gradient[index] += cache.x[i*width+input] * gradientValues[i*width+output]
				gradientX[i*width+input] += m.value.weights.values[index] * gradientValues[i*width+output]
			}
		}
	}
	
	// Bias gradients
	for i := 0; i < rows; i++ {
		for output := 0; output < width; output++ {
			m.query.bias.gradient[output] += gradientQuery[i*width+output]
			m.key.bias.gradient[output] += gradientKeys[i*width+output]
			m.value.bias.gradient[output] += gradientValues[i*width+output]
		}
	}
	
	// Backprop through embeddings
	for row, id := range cache.ids {
		for d := 0; d < width; d++ {
			tokenIndex := id*width + d
			posIndex := row*width + d
			m.tokenEmb.gradient[tokenIndex] += gradientX[row*width+d]
			m.posEmb.gradient[posIndex] += gradientX[row*width+d]
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

// Mean window loss: average cross-entropy over all positions
func meanWindowLoss(cache forwardCache, targets []int) float64 {
	totalLoss := 0.0
	for i := range targets {
		rowLogits := cache.logits[i*len(cache.probabilities)/len(targets) : (i+1)*len(cache.probabilities)/len(targets)]
		totalLoss += tokenLoss32(rowLogits, targets[i])
	}
	return totalLoss / float64(len(targets))
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
			totalLoss += meanWindowLoss(cache, targets)
			zeroGradients(m.parameters)
			m.backward(cache, targets)
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

// Generate samples tokens at the requested temperature using deterministic
// seeded sampling. Sampling stays on valid printable UTF-8 boundaries.
func (m *Model) Generate(prompt string, maxTokens int, temperature float64, seed int64) (string, error) {
	if maxTokens < 0 || temperature <= 0 || math.IsNaN(temperature) || math.IsInf(temperature, 0) { return "", fmt.Errorf("max tokens must be non-negative and temperature finite and positive") }
	ids := m.Tokenizer.Encode(prompt)
	if len(ids) == 0 { ids = m.Tokenizer.Encode(" ") }
	prefix := []byte(m.Tokenizer.Decode(ids))
	backend, rng := m.inferenceBackend(), rand.New(rand.NewSource(seed))
	for n := 0; n < maxTokens; n++ {
		logits := backend.Forward(ids)
		maximum := float32(math.Inf(-1))
		for _, value := range logits { if value > maximum { maximum = value } }
		probabilities := make([]float64, len(logits))
		var sum float64
		for id, value := range logits {
			if id >= len(m.Tokenizer.Tokens) || !validUTF8Append(prefix, m.Tokenizer.Tokens[id]) { continue }
			probabilities[id] = math.Exp(float64(value-maximum)/temperature)
			sum += probabilities[id]
		}
		if sum == 0 { for id, token := range m.Tokenizer.Tokens { if validUTF8Append(prefix, token) { probabilities[id], sum = 1, 1; break } } }
		draw, choice := rng.Float64()*sum, 0
		for id, probability := range probabilities {
			if probability == 0 { continue }
			if draw < probability { choice = id; break }
			draw -= probability
		}
		ids = append(ids, choice)
		prefix = appendString(prefix, m.Tokenizer.Tokens[choice])
	}
	return strings.ToValidUTF8(string(prefix), "�"), nil
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
