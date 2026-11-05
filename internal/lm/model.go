package lm

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"

	"m3tal-ai/internal/autograd"
	"m3tal-ai/internal/tensor"
)

// Config defines the multi-layer, multi-head decoder-only Transformer.
type Config struct {
	Context   int   `json:"context"`
	Width     int   `json:"width"`
	Layers    int   `json:"layers,omitempty"`
	Heads     int   `json:"heads,omitempty"`
	VocabSize int   `json:"vocab_size,omitempty"`
	Seed      int64 `json:"seed,omitempty"`
}

// Normalize ensures Layers, Heads, and Seed have valid defaults when omitted (0).
func (c *Config) Normalize() {
	if c.Layers == 0 {
		c.Layers = 1
	}
	if c.Heads == 0 {
		c.Heads = 1
	}
	if c.Seed == 0 {
		c.Seed = 1
	}
}

// DefaultConfig fits an instruction-capable training prototype on CPU-only machines.
func DefaultConfig() Config {
	return Config{Context: 256, Width: 64, Layers: 4, Heads: 4, VocabSize: 256, Seed: 1}
}

// ExpectedParameterCount calculates the exact expected count of trainable parameters
// for a model architecture configuration and vocabulary size from first principles:
//
//	token embeddings: vocab * width
//	position embeddings: context * width
//	each transformer block:
//	  q, k, v projections: 3 * (width * width + width)
//	  attn output projection: width * width + width
//	  ffIn projection: width * (2 * width) + (2 * width)
//	  ffOut projection: (2 * width) * width + width
//	  total per block: 8 * width^2 + 7 * width
//	final output projection: width * vocab + vocab
//	Total = (2 * vocab + context) * width + vocab + layers * (8 * width^2 + 7 * width)
func ExpectedParameterCount(config Config, vocabSize int) int {
	config.Normalize()
	if vocabSize <= 0 {
		vocabSize = config.VocabSize
	}
	w := config.Width
	v := vocabSize
	c := config.Context
	l := config.Layers

	blockParams := l * (8*w*w + 7*w)
	embAndHeadParams := (2*v+c)*w + v
	return embAndHeadParams + blockParams
}

// ExpectedParameterCount calculates the expected parameter count using the model's configuration and vocabulary size.
func (c Config) ExpectedParameterCount(vocabSize int) int {
	return ExpectedParameterCount(c, vocabSize)
}

// parameter owns contiguous float32 values, gradients, and Adam state, backed by an autograd.Variable.
type parameter struct {
	values   []float32
	gradient []float32
	first    []float32
	second   []float32
	variable *autograd.Variable
}

type linear struct {
	weights *parameter // row-major [input, output]
	bias    *parameter
	input   int
	output  int
}

// transformerBlock holds one decoder layer's parameters.
// normAttn and normFF are the learnable RMSNorm gamma vectors applied
// before the attention sub-layer and before the feed-forward sub-layer
// respectively (pre-norm architecture).
type transformerBlock struct {
	normAttn *parameter // RMSNorm gamma, pre-attention [width]
	normFF   *parameter // RMSNorm gamma, pre-FFN       [width]
	query    linear
	key      linear
	value    linear
	attnOut  linear
	ffGate   linear // SwiGLU gate projection [width, width*2]
	ffUp     linear // SwiGLU up projection   [width, width*2]
	ffDown   linear // SwiGLU down projection [width*2, width]
}

// Model is a causal language model trained and run with contiguous float32
// parameters. Position is encoded via RoPE inside each attention block;
// no learned position embedding table is maintained.
type Model struct {
	Config     Config
	Tokenizer  *Tokenizer
	tokenEmb   *parameter // [vocabulary, width]
	blocks     []transformerBlock
	output     linear
	parameters []*parameter
	step       int
	backend    inferenceBackend
}

// VocabSize returns the active vocabulary size of the model.
func (m *Model) VocabSize() int {
	if m.Tokenizer != nil && len(m.Tokenizer.Tokens) > 0 {
		return len(m.Tokenizer.Tokens)
	}
	return m.Config.VocabSize
}

// ParameterCount reports the number of trainable scalar weights.
func (m *Model) ParameterCount() int {
	count := 0
	for _, parameter := range m.parameters {
		count += len(parameter.values)
	}
	return count
}

func newParameterWithShape(shape []int) *parameter {
	numel, _ := tensor.Numel(shape)
	t, _ := tensor.New(shape, tensor.Float32)
	v, _ := autograd.NewVariable(t, true)
	return &parameter{
		values:   make([]float32, numel),
		gradient: make([]float32, numel),
		first:    make([]float32, numel),
		second:   make([]float32, numel),
		variable: v,
	}
}

func newParameter(size int) *parameter {
	return newParameterWithShape([]int{size})
}

func (p *parameter) syncToVariable() {
	if p.variable != nil && p.variable.Data != nil {
		p.variable.Data.SetFromFloat32Slice(p.values)
	}
}

func (p *parameter) syncFromVariable() {
	if p.variable != nil && p.variable.Data != nil {
		copy(p.values, p.variable.Data.Float32Slice())
	}
}

func (p *parameter) syncFromVariableGrad() {
	if p.variable != nil && p.variable.Grad != nil {
		copy(p.gradient, p.variable.Grad.Float32Slice())
	}
}

func (m *Model) addParameter(p *parameter) *parameter {
	m.parameters = append(m.parameters, p)
	return p
}

// Variables returns the underlying autograd.Variable nodes for all model parameters.
func (m *Model) Variables() []*autograd.Variable {
	vars := make([]*autograd.Variable, len(m.parameters))
	for i, p := range m.parameters {
		p.syncToVariable()
		vars[i] = p.variable
	}
	return vars
}

// NewModel initializes weights with a deterministic seed.
func NewModel(config Config, tokenizer *Tokenizer, seed int64) (*Model, error) {
	config.Normalize()
	if seed != 0 {
		config.Seed = seed
	}
	if tokenizer != nil && len(tokenizer.Tokens) > 0 {
		config.VocabSize = len(tokenizer.Tokens)
	}

	if config.Context < 2 {
		return nil, fmt.Errorf("context length must be at least 2, got %d", config.Context)
	}
	if config.Width < 2 {
		return nil, fmt.Errorf("model width must be at least 2, got %d", config.Width)
	}
	if config.Layers < 1 {
		return nil, fmt.Errorf("layers must be at least 1, got %d", config.Layers)
	}
	if config.Heads < 1 {
		return nil, fmt.Errorf("heads must be at least 1, got %d", config.Heads)
	}
	if config.Width%config.Heads != 0 {
		return nil, fmt.Errorf("width %d must be divisible by heads %d", config.Width, config.Heads)
	}
	if config.VocabSize < 2 {
		return nil, fmt.Errorf("vocabulary size must be at least 2, got %d", config.VocabSize)
	}

	rng := rand.New(rand.NewSource(config.Seed))
	m := &Model{Config: config, Tokenizer: tokenizer}
	newMatrix := func(rows, cols int) *parameter {
		parameter := newParameterWithShape([]int{rows, cols})
		limit := float32(1 / math.Sqrt(float64(cols)))
		for i := range parameter.values {
			parameter.values[i] = (rng.Float32()*2 - 1) * limit
		}
		parameter.syncToVariable()
		return m.addParameter(parameter)
	}
	newLinear := func(input, output int) linear {
		weights := newMatrix(input, output)
		bias := m.addParameter(newParameterWithShape([]int{output}))
		return linear{weights: weights, bias: bias, input: input, output: output}
	}
	width, vocab := config.Width, config.VocabSize
	m.tokenEmb = newMatrix(vocab, width)
	// No posEmb — RoPE encodes position inside each block's attention.

	m.blocks = make([]transformerBlock, config.Layers)
	for l := 0; l < config.Layers; l++ {
		m.blocks[l] = transformerBlock{
			normAttn: newNormGamma(width),
			normFF:   newNormGamma(width),
			query:    newLinear(width, width),
			key:      newLinear(width, width),
			value:    newLinear(width, width),
			attnOut:  newLinear(width, width),
			ffGate:   newLinear(width, width*2),
			ffUp:     newLinear(width, width*2),
			ffDown:   newLinear(width*2, width),
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
			normAttn: b.normAttn.values,
			normFF:   b.normFF.values,
			query:    inferenceLayer{weights: b.query.weights.values, bias: b.query.bias.values},
			key:      inferenceLayer{weights: b.key.weights.values, bias: b.key.bias.values},
			value:    inferenceLayer{weights: b.value.weights.values, bias: b.value.bias.values},
			attnOut:  inferenceLayer{weights: b.attnOut.weights.values, bias: b.attnOut.bias.values},
			ffGate:   inferenceLayer{weights: b.ffGate.weights.values, bias: b.ffGate.bias.values},
			ffUp:     inferenceLayer{weights: b.ffUp.weights.values, bias: b.ffUp.bias.values},
			ffDown:   inferenceLayer{weights: b.ffDown.weights.values, bias: b.ffDown.bias.values},
		}
	}
	return inferenceWeights{
		tokenEmb: m.tokenEmb.values,
		blocks:   blocks,
		output:   inferenceLayer{weights: m.output.weights.values, bias: m.output.bias.values},
	}
}

// ForwardLogits computes the forward pass over a batch of token sequences,
// returning a 3D float32 tensor of shape [batch, sequence, vocab] containing unnormalized logits.
// All sequences in the batch must have identical length S (1 <= S <= Context), and all
// token IDs must be in the range [0, VocabSize-1].
// All output logits are verified to be finite values.
func (m *Model) ForwardLogits(batch [][]int) (*tensor.Tensor, error) {
	if len(batch) == 0 {
		return nil, fmt.Errorf("batch cannot be empty")
	}
	batchSize := len(batch)
	seqLen := len(batch[0])
	if seqLen == 0 {
		return nil, fmt.Errorf("sequence length must be positive")
	}
	if seqLen > m.Config.Context {
		return nil, fmt.Errorf("sequence length %d exceeds model context length %d", seqLen, m.Config.Context)
	}
	vocab := m.VocabSize()

	for b, seq := range batch {
		if len(seq) != seqLen {
			return nil, fmt.Errorf("all sequences in batch must have identical length: batch[0] has %d, batch[%d] has %d", seqLen, b, len(seq))
		}
		for i, tok := range seq {
			if tok < 0 || tok >= vocab {
				return nil, fmt.Errorf("token %d at batch[%d][%d] out of vocabulary bounds [0, %d)", tok, b, i, vocab)
			}
		}
	}

	flatLogits := make([]float32, batchSize*seqLen*vocab)
	for b, seq := range batch {
		cache := m.forward(seq)
		offset := b * seqLen * vocab
		copy(flatLogits[offset:offset+seqLen*vocab], cache.logits)
	}

	for i, v := range flatLogits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("non-finite logit %f detected at index %d", v, i)
		}
	}

	return tensor.FromFloat32([]int{batchSize, seqLen, vocab}, flatLogits)
}

// ForwardSequence computes the forward pass over a single sequence of token IDs,
// returning a 2D float32 tensor of shape [sequence, vocab] containing unnormalized logits.
func (m *Model) ForwardSequence(tokenIDs []int) (*tensor.Tensor, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("tokenIDs cannot be empty")
	}
	t, err := m.ForwardLogits([][]int{tokenIDs})
	if err != nil {
		return nil, err
	}
	return t.Squeeze(0)
}

// ForwardAutograd runs the transformer forward pass using internal/autograd and internal/tensor primitives,
// building a fully connected computation graph that supports reverse-mode automatic differentiation.
// Returns an autograd.Variable with shape [seqLen, vocab].
func (m *Model) ForwardAutograd(tokenIDs []int) (*autograd.Variable, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("tokenIDs cannot be empty")
	}
	seqLen := len(tokenIDs)
	if seqLen > m.Config.Context {
		return nil, fmt.Errorf("sequence length %d exceeds context %d", seqLen, m.Config.Context)
	}
	width := m.Config.Width
	vocab := m.VocabSize()

	for i, tok := range tokenIDs {
		if tok < 0 || tok >= vocab {
			return nil, fmt.Errorf("token %d at index %d out of bounds [0, %d)", tok, i, vocab)
		}
	}

	// Sync current float parameter values into autograd variables
	for _, p := range m.parameters {
		p.syncToVariable()
	}

	// 1. One-hot selection matrix [seqLen, vocab] to differentiably embed tokens:
	// tokenEmbeddings = oneHot * tokenEmb
	oneHotData := make([]float32, seqLen*vocab)
	for i, tok := range tokenIDs {
		oneHotData[i*vocab+tok] = 1.0
	}
	oneHotTensor, err := tensor.FromFloat32([]int{seqLen, vocab}, oneHotData)
	if err != nil {
		return nil, err
	}
	oneHotVar, err := autograd.NewVariable(oneHotTensor, false)
	if err != nil {
		return nil, err
	}

	tokenEmbVar, err := autograd.MatMul(oneHotVar, m.tokenEmb.variable)
	if err != nil {
		return nil, err
	}

	// Position embeddings via one-hot selection [seqLen, context] * posEmb [context, width]
	posOneHotData := make([]float32, seqLen*m.Config.Context)
	for i := 0; i < seqLen; i++ {
		posOneHotData[i*m.Config.Context+i] = 1.0
	}
	posOneHotTensor, err := tensor.FromFloat32([]int{seqLen, m.Config.Context}, posOneHotData)
	if err != nil {
		return nil, err
	}
	posOneHotVar, err := autograd.NewVariable(posOneHotTensor, false)
	if err != nil {
		return nil, err
	}

	posEmbVar, err := autograd.MatMul(posOneHotVar, m.posEmb.variable)
	if err != nil {
		return nil, err
	}

	// Initial token representation: x = tokenEmb + posEmb
	x, err := autograd.Add(tokenEmbVar, posEmbVar)
	if err != nil {
		return nil, err
	}

	// 2. Transformer blocks
	scale := 1.0 / math.Sqrt(float64(width))
	for _, block := range m.blocks {
		// Q, K, V projections
		q, err := autograd.MatMul(x, block.query.weights.variable)
		if err != nil {
			return nil, err
		}
		k, err := autograd.MatMul(x, block.key.weights.variable)
		if err != nil {
			return nil, err
		}
		v, err := autograd.MatMul(x, block.value.weights.variable)
		if err != nil {
			return nil, err
		}

		// Self-attention scores: Q * K^T
		kTrans, err := autograd.Transpose(k)
		if err != nil {
			return nil, err
		}
		scores, err := autograd.MatMul(q, kTrans)
		if err != nil {
			return nil, err
		}
		scoresScaled, err := autograd.Scale(scores, scale)
		if err != nil {
			return nil, err
		}

		// Differentiable causal attention with mask
		// Mask future tokens with 0 in softmax attention
		scoresData := scoresScaled.Data.Float32Slice()
		attnWeights := make([]float32, seqLen*seqLen)
		for i := 0; i < seqLen; i++ {
			maxVal := float32(math.Inf(-1))
			for j := 0; j <= i; j++ {
				val := scoresData[i*seqLen+j]
				if val > maxVal {
					maxVal = val
				}
			}
			var sumExp float32
			for j := 0; j <= i; j++ {
				expVal := float32(math.Exp(float64(scoresData[i*seqLen+j] - maxVal)))
				attnWeights[i*seqLen+j] = expVal
				sumExp += expVal
			}
			for j := 0; j <= i; j++ {
				attnWeights[i*seqLen+j] /= sumExp
			}
		}
		attnTensor, err := tensor.FromFloat32([]int{seqLen, seqLen}, attnWeights)
		if err != nil {
			return nil, err
		}
		attnVar, err := autograd.NewVariable(attnTensor, false)
		if err != nil {
			return nil, err
		}

		attended, err := autograd.MatMul(attnVar, v)
		if err != nil {
			return nil, err
		}

		attnProj, err := autograd.MatMul(attended, block.attnOut.weights.variable)
		if err != nil {
			return nil, err
		}
		res1, err := autograd.Add(x, attnProj)
		if err != nil {
			return nil, err
		}

		// Feed-forward sublayer
		ffHidden, err := autograd.MatMul(res1, block.ffIn.weights.variable)
		if err != nil {
			return nil, err
		}
		ffAct, err := autograd.Tanh(ffHidden)
		if err != nil {
			return nil, err
		}
		ffOut, err := autograd.MatMul(ffAct, block.ffOut.weights.variable)
		if err != nil {
			return nil, err
		}
		x, err = autograd.Add(res1, ffOut)
		if err != nil {
			return nil, err
		}
	}

	// 3. Final projection to vocabulary
	logits, err := autograd.MatMul(x, m.output.weights.variable)
	if err != nil {
		return nil, err
	}
	return logits, nil
}

func (m *Model) forward(ids []int) forwardCache {
	return forwardFloat32(m.Config, m.VocabSize(), m.parameterView(), ids)
}

// backward propagates the cross-entropy loss gradient through the full model.
// If mask is non-nil, only positions where mask[i]==true contribute to the loss.
func (m *Model) backward(cache forwardCache, targets []int, mask []bool) {
	m.Config.Normalize()
	width, vocab, rows := m.Config.Width, m.VocabSize(), len(cache.ids)
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

	// ── 1. Logits → softmax gradient ──────────────────────────────────────
	gradLogits := make([]float32, rows*vocab)
	for i := 0; i < rows; i++ {
		if mask != nil && (i >= len(mask) || !mask[i]) {
			continue
		}
		for j := 0; j < vocab; j++ {
			gradLogits[i*vocab+j] = cache.probabilities[i*vocab+j] * invRows
		}
		gradLogits[i*vocab+targets[i]] -= invRows
	}

	// ── 2. Output linear layer ─────────────────────────────────────────────
	lastBlockIdx := len(m.blocks) - 1
	gradCurrentX := make([]float32, rows*width)
	for i := 0; i < rows; i++ {
		for d := 0; d < width; d++ {
			for tok := 0; tok < vocab; tok++ {
				idx := d*vocab + tok
				m.output.weights.gradient[idx] += cache.blocks[lastBlockIdx].encoded[i*width+d] * gradLogits[i*vocab+tok]
				gradCurrentX[i*width+d] += m.output.weights.values[idx] * gradLogits[i*vocab+tok]
			}
		}
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < vocab; j++ {
			m.output.bias.gradient[j] += gradLogits[i*vocab+j]
		}
	}

	// ── 3. Transformer blocks (reverse order) ─────────────────────────────
	for l := len(m.blocks) - 1; l >= 0; l-- {
		block := m.blocks[l]
		bc := cache.blocks[l]
		gradEncoded := gradCurrentX

		// ── 3a. FF residual split: encoded = residual + ff ────────────
		// gradResidual accumulates from the residual path; we start it with
		// gradEncoded (direct flow through the residual connection).
		gradResidual := make([]float32, rows*width)
		copy(gradResidual, gradEncoded)

		// ── 3b. SwiGLU: ffDown linear backward ────────────────────────
		// Forward: feedForward = activated · W_down + bias_down
		// Backward: dActivated[h] = Σ_d W_down[h*width+d] * dEncoded[d]
		hiddenDim := width * 2
		gradActivated := make([]float32, rows*hiddenDim)
		for i := 0; i < rows; i++ {
			for h := 0; h < hiddenDim; h++ {
				for d := 0; d < width; d++ {
					idx := h*width + d
					block.ffDown.weights.gradient[idx] += bc.ffActivated[i*hiddenDim+h] * gradEncoded[i*width+d]
					gradActivated[i*hiddenDim+h] += block.ffDown.weights.values[idx] * gradEncoded[i*width+d]
				}
			}
		}
		for i := 0; i < rows; i++ {
			for d := 0; d < width; d++ {
				block.ffDown.bias.gradient[d] += gradEncoded[i*width+d]
			}
		}

		// ── 3c. SwiGLU activation backward: activated = swish(gate) ⊙ up ──
		// dUp[h]   = dActivated[h] * swish(gate[h])
		// dGate[h] = dActivated[h] * up[h] * swishPrime(gate[h])
		gradGate := make([]float32, rows*hiddenDim)
		gradUp := make([]float32, rows*hiddenDim)
		for idx := 0; idx < rows*hiddenDim; idx++ {
			gradUp[idx] = gradActivated[idx] * swish(bc.ffGateOut[idx])
			gradGate[idx] = gradActivated[idx] * bc.ffUpOut[idx] * swishPrime(bc.ffGateOut[idx])
		}

		// ── 3d. ffGate and ffUp linear backward (input was normX2) ────
		// dNormX2[d] += Σ_h W_gate[d*hiddenDim+h]*dGate[h] + W_up[d*hiddenDim+h]*dUp[h]
		gradNormX2 := make([]float32, rows*width)
		for i := 0; i < rows; i++ {
			for h := 0; h < hiddenDim; h++ {
				block.ffGate.bias.gradient[h] += gradGate[i*hiddenDim+h]
				block.ffUp.bias.gradient[h] += gradUp[i*hiddenDim+h]
				for d := 0; d < width; d++ {
					idx := d*hiddenDim + h
					block.ffGate.weights.gradient[idx] += bc.normX2[i*width+d] * gradGate[i*hiddenDim+h]
					gradNormX2[i*width+d] += block.ffGate.weights.values[idx] * gradGate[i*hiddenDim+h]
					block.ffUp.weights.gradient[idx] += bc.normX2[i*width+d] * gradUp[i*hiddenDim+h]
					gradNormX2[i*width+d] += block.ffUp.weights.values[idx] * gradUp[i*hiddenDim+h]
				}
			}
		}

		// ── 3e. RMSNorm #2 backward (pre-FFN norm) ────────────────────
		// Input to norm was bc.residual; output was bc.normX2.
		gradRms2 := rmsNormBackward(gradNormX2, bc.residual, bc.normRms2, block.normFF.values, rows, width, block.normFF.gradient)
		for i := 0; i < rows*width; i++ {
			gradResidual[i] += gradRms2[i]
		}

		// ── 3f. Attn residual split: residual = x + attnProjection ────
		gradX := make([]float32, rows*width)
		copy(gradX, gradResidual) // direct flow through residual connection

		// ── 3g. attnOut linear ────────────────────────────────────────
		gradAttended := make([]float32, rows*width)
		for i := 0; i < rows; i++ {
			for inp := 0; inp < width; inp++ {
				for out := 0; out < width; out++ {
					idx := inp*width + out
					block.attnOut.weights.gradient[idx] += bc.attended[i*width+inp] * gradResidual[i*width+out]
					gradAttended[i*width+inp] += block.attnOut.weights.values[idx] * gradResidual[i*width+out]
				}
			}
		}
		for i := 0; i < rows; i++ {
			for out := 0; out < width; out++ {
				block.attnOut.bias.gradient[out] += gradResidual[i*width+out]
			}
		}

		// ── 3h. Multi-head causal attention backward ──────────────────
		// Gradients w.r.t. Q/K after RoPE, and V (no RoPE on V).
		gradQueryRope := make([]float32, rows*width)
		gradKeysRope := make([]float32, rows*width)
		gradValues := make([]float32, rows*width)

		gradAttention := make([]float32, heads*rows*rows)
		for h := 0; h < heads; h++ {
			headOffset := h * headDim
			headMatrixOffset := h * rows * rows
			for i := 0; i < rows; i++ {
				for d := 0; d < headDim; d++ {
					for j := 0; j <= i; j++ {
						gradAttention[headMatrixOffset+i*rows+j] += gradAttended[i*width+headOffset+d] * bc.v[j*width+headOffset+d]
						gradValues[j*width+headOffset+d] += bc.attention[headMatrixOffset+i*rows+j] * gradAttended[i*width+headOffset+d]
					}
				}
			}
		}

		for h := 0; h < heads; h++ {
			headOffset := h * headDim
			headMatrixOffset := h * rows * rows
			for i := 0; i < rows; i++ {
				var weightedGrad float32
				for j := 0; j <= i; j++ {
					weightedGrad += bc.attention[headMatrixOffset+i*rows+j] * gradAttention[headMatrixOffset+i*rows+j]
				}
				for j := 0; j <= i; j++ {
					gradScore := bc.attention[headMatrixOffset+i*rows+j] * (gradAttention[headMatrixOffset+i*rows+j] - weightedGrad)
					for d := 0; d < headDim; d++ {
						// bc.q and bc.k are the post-RoPE vectors used in the forward pass.
						gradQueryRope[i*width+headOffset+d] += gradScore * bc.k[j*width+headOffset+d] * scale
						gradKeysRope[j*width+headOffset+d] += gradScore * bc.q[i*width+headOffset+d] * scale
					}
				}
			}
		}

		// ── 3i. RoPE backward: inverse rotation on Q and K gradients ──
		applyInverseRoPE(gradQueryRope, gradKeysRope, rows, width, headDim, heads)

		// ── 3j. QKV linear projections backward (input was normX1) ────
		gradNormX1 := make([]float32, rows*width)
		for inp := 0; inp < width; inp++ {
			for out := 0; out < width; out++ {
				idx := inp*width + out
				for i := 0; i < rows; i++ {
					block.query.weights.gradient[idx] += bc.normX1[i*width+inp] * gradQueryRope[i*width+out]
					gradNormX1[i*width+inp] += block.query.weights.values[idx] * gradQueryRope[i*width+out]
					block.key.weights.gradient[idx] += bc.normX1[i*width+inp] * gradKeysRope[i*width+out]
					gradNormX1[i*width+inp] += block.key.weights.values[idx] * gradKeysRope[i*width+out]
					block.value.weights.gradient[idx] += bc.normX1[i*width+inp] * gradValues[i*width+out]
					gradNormX1[i*width+inp] += block.value.weights.values[idx] * gradValues[i*width+out]
				}
			}
		}
		for i := 0; i < rows; i++ {
			for out := 0; out < width; out++ {
				block.query.bias.gradient[out] += gradQueryRope[i*width+out]
				block.key.bias.gradient[out] += gradKeysRope[i*width+out]
				block.value.bias.gradient[out] += gradValues[i*width+out]
			}
		}

		// ── 3k. RMSNorm #1 backward (pre-attention norm) ──────────────
		// Input to norm was bc.x; output was bc.normX1.
		gradRms1 := rmsNormBackward(gradNormX1, bc.x, bc.normRms1, block.normAttn.values, rows, width, block.normAttn.gradient)
		for i := 0; i < rows*width; i++ {
			gradX[i] += gradRms1[i]
		}

		gradCurrentX = gradX
	}

	// ── 4. Token embedding gradients (no posEmb) ──────────────────────────
	for row, id := range cache.ids {
		for d := 0; d < width; d++ {
			m.tokenEmb.gradient[id*width+d] += gradCurrentX[row*width+d]
		}
	}
}

func zeroGradients(parameters []*parameter) {
	for _, p := range parameters {
		clear(p.gradient)
	}
}

func (m *Model) adamStep(learningRate float32) {
	m.step++
	beta1, beta2 := float32(0.9), float32(0.999)
	bias1 := float32(1 - math.Pow(float64(beta1), float64(m.step)))
	bias2 := float32(1 - math.Pow(float64(beta2), float64(m.step)))

	var totalNormSq float64
	for _, p := range m.parameters {
		for _, g := range p.gradient {
			totalNormSq += float64(g * g)
		}
	}
	clipScale := float32(1.0)
	if gradNorm := math.Sqrt(totalNormSq); gradNorm > 1.0 {
		clipScale = float32(1.0 / gradNorm)
	}

	for _, p := range m.parameters {
		for i, g := range p.gradient {
			gradient := g * clipScale
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
	for _, value := range logits {
		if value > maximum {
			maximum = value
		}
	}
	var total float64
	for _, value := range logits {
		total += math.Exp(float64(value - maximum))
	}
	return math.Log(total) + float64(maximum) - float64(logits[target])
}

// meanWindowLoss returns average cross-entropy over active positions (all if mask is nil).
func meanWindowLoss(cache forwardCache, targets []int, mask []bool) float64 {
	if len(targets) == 0 || len(cache.ids) == 0 {
		return 0
	}
	rows := len(cache.ids)
	if len(targets) < rows {
		rows = len(targets)
	}
	vocab := len(cache.logits) / len(cache.ids)
	var totalLoss float64
	activeCount := 0
	for i := 0; i < rows; i++ {
		if mask != nil && (i >= len(mask) || !mask[i]) {
			continue
		}
		totalLoss += tokenLoss32(cache.logits[i*vocab:(i+1)*vocab], targets[i])
		activeCount++
	}
	if activeCount == 0 {
		return 0
	}
	return totalLoss / float64(activeCount)
}

// TrainBatchStep performs a single optimization step (forward pass, loss computation,
// backward pass with gradient accumulation, and Adam parameter update) over a batch
// of input sequences and target sequences. Returns the average cross-entropy loss across the batch.
func (m *Model) TrainBatchStep(inputs, targets [][]int, learningRate float64) (float64, error) {
	if len(inputs) == 0 || len(targets) == 0 {
		return 0, fmt.Errorf("inputs and targets must not be empty")
	}
	if len(inputs) != len(targets) {
		return 0, fmt.Errorf("inputs batch size %d does not match targets batch size %d", len(inputs), len(targets))
	}
	if learningRate <= 0 || math.IsNaN(learningRate) || math.IsInf(learningRate, 0) {
		return 0, fmt.Errorf("learning rate must be positive and finite")
	}

	m.backend = nil
	zeroGradients(m.parameters)

	var totalLoss float64
	for b := range inputs {
		in := inputs[b]
		tgt := targets[b]
		if len(in) == 0 || len(tgt) == 0 {
			return 0, fmt.Errorf("sample %d has empty input or target", b)
		}
		if len(in) != len(tgt) {
			return 0, fmt.Errorf("sample %d input length %d does not match target length %d", b, len(in), len(tgt))
		}
		cache := m.forward(in)
		loss := meanWindowLoss(cache, tgt, nil)
		totalLoss += loss
		m.backward(cache, tgt, nil)
	}

	avgLoss := totalLoss / float64(len(inputs))
	m.adamStep(float32(learningRate))
	return avgLoss, nil
}

// Train performs causal full-window training with float32 parameters and Adam updates.
// It returns average pre-update cross-entropy per epoch.
func (m *Model) Train(texts []string, epochs, stepsPerEpoch int, learningRate float64) ([]float64, error) {
	if epochs <= 0 || stepsPerEpoch <= 0 || learningRate <= 0 || math.IsNaN(learningRate) || math.IsInf(learningRate, 0) {
		return nil, fmt.Errorf("epochs, steps, and finite positive learning rate are required")
	}
	m.backend = nil
	var ids []int
	for _, text := range texts {
		ids = append(ids, m.Tokenizer.Encode(text)...)
		ids = append(ids, m.Tokenizer.Encode("\n")...)
	}
	if len(ids) < 3 {
		return nil, fmt.Errorf("training corpus must encode to at least three tokens")
	}
	rng := rand.New(rand.NewSource(71))
	losses := make([]float64, 0, epochs)
	for epoch := 0; epoch < epochs; epoch++ {
		var totalLoss float64
		for step := 0; step < stepsPerEpoch; step++ {
			start := rng.Intn(len(ids) - 1)
			end := start + m.Config.Context + 1
			if end > len(ids) {
				end = len(ids)
			}
			window := ids[start:end]
			if len(window) < 2 {
				continue
			}
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

// TrainInstructions trains on prompt-completion pairs using prompt loss masking.
// Gradients are only computed on completion tokens.
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
		maskSlice := make([]bool, len(targets))
		hasActive := false
		for i := range targets {
			if i+1 >= promptLen {
				maskSlice[i] = true
				hasActive = true
			}
		}
		if hasActive {
			prepared = append(prepared, preparedExample{input: input, targets: targets, mask: maskSlice})
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
	for _, text := range texts {
		ids = append(ids, m.Tokenizer.Encode(text)...)
		ids = append(ids, m.Tokenizer.Encode("\n")...)
	}
	if len(ids) < 2 {
		return 0, fmt.Errorf("evaluation corpus must encode to at least two tokens")
	}
	backend := m.inferenceBackend()
	var total float64
	for target := 1; target < len(ids); target++ {
		start := target - m.Config.Context
		if start < 0 {
			start = 0
		}
		total += tokenLoss32(backend.Forward(ids[start:target]), ids[target])
	}
	return total / float64(len(ids)-1), nil
}

// GenerateOptions configures text generation parameters.
type GenerateOptions struct {
	MaxTokens         int
	Temperature       float64
	Seed              int64
	RepetitionPenalty float64 // > 1.0 penalizes already generated tokens (e.g. 1.15)
}

// GenerateWithOptions generates tokens using configurable sampling, repetition
// penalty, and UTF-8 validation. Generation uses a KV cache for O(1) per-token
// cost during the autoregressive phase; a full forward pass is used as fallback
// once the context window is exhausted.
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

	// ── KV-cache prefill: process all prompt tokens once ───────────────────
	// Each call to ForwardStepKV runs a single token through the model,
	// appending its K and V to the per-layer cache. After the loop, rawLogits
	// holds the next-token prediction for the last prompt position.
	m.Config.Normalize()
	kvCache := backend.NewKVCache()
	var rawLogits []float32
	for i, id := range ids {
		rawLogits = backend.ForwardStepKV(id, i, kvCache)
	}
	pos := len(ids) // next sequence position to write into the cache

	// ── Autoregressive generation ──────────────────────────────────────────
	for n := 0; n < opts.MaxTokens; n++ {
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

		// ── Next-token logits: KV step or full-forward fallback ────────
		if pos < m.Config.Context {
			rawLogits = backend.ForwardStepKV(choice, pos, kvCache)
			pos++
		} else {
			// Context window exhausted: full forward pass with truncation.
			rawLogits = backend.Forward(ids)
		}
	}
	return strings.ToValidUTF8(string(prefix), ""), nil
}


// Generate samples tokens at the requested temperature using deterministic
// seeded sampling with a 1.15 repetition penalty.
func (m *Model) Generate(prompt string, maxTokens int, temperature float64, seed int64) (string, error) {
	return m.GenerateWithOptions(prompt, GenerateOptions{
		MaxTokens:         maxTokens,
		Temperature:       temperature,
		Seed:              seed,
		RepetitionPenalty: 1.15,
	})
}

func (m *Model) inferenceBackend() inferenceBackend {
	if m.backend == nil {
		m.backend = newCPUInferenceBackend(m)
	}
	return m.backend
}

func validUTF8Append(prefix []byte, suffix string) bool {
	total := len(prefix) + len(suffix)
	byteAt := func(index int) byte {
		if index < len(prefix) {
			return prefix[index]
		}
		return suffix[index-len(prefix)]
	}
	for i := 0; i < total; {
		first := byteAt(i)
		if first < 0x80 {
			if (first < 0x20 && first != '\n' && first != '\r' && first != '\t') || first == 0x7F {
				return false
			}
			i++
			continue
		}
		length := 0
		switch {
		case first >= 0xC2 && first <= 0xDF:
			length = 2
		case first >= 0xE0 && first <= 0xEF:
			length = 3
		case first >= 0xF0 && first <= 0xF4:
			length = 4
		default:
			return false
		}
		available, check := total-i, length
		incomplete := available < length
		if incomplete {
			check = available
		}
		for offset := 1; offset < check; offset++ {
			b := byteAt(i + offset)
			if b < 0x80 || b > 0xBF {
				return false
			}
			if offset == 1 && ((first == 0xE0 && b < 0xA0) || (first == 0xED && b > 0x9F) || (first == 0xF0 && b < 0x90) || (first == 0xF4 && b > 0x8F)) {
				return false
			}
		}
		if incomplete {
			return true
		}
		i += length
	}
	return true
}

func appendString(prefix []byte, suffix string) []byte {
	for i := 0; i < len(suffix); i++ {
		prefix = append(prefix, suffix[i])
	}
	return prefix
}

// checkpoint is the on-disk format. All learnable parameters are stored as a
// flat float64 slice in the same order that NewModel allocates them.
type checkpoint struct {
	Config Config            `json:"config"`
	Tokens [][]byte          `json:"tokens"`
	Merges []checkpointMerge `json:"merges"`
	Values []float64         `json:"values"`
}

type checkpointMerge struct{ Left, Right, Token []byte }

// SaveCheckpoint writes tokenizer and float32 weights to a local JSON checkpoint.
func (m *Model) SaveCheckpoint(path string) error {
	state := checkpoint{
		Config: m.Config,
		Tokens: make([][]byte, len(m.Tokenizer.Tokens)),
		Merges: make([]checkpointMerge, len(m.Tokenizer.Merges)),
	}
	for i, token := range m.Tokenizer.Tokens {
		state.Tokens[i] = []byte(token)
	}
	for i, merge := range m.Tokenizer.Merges {
		state.Merges[i] = checkpointMerge{[]byte(merge.Left), []byte(merge.Right), []byte(merge.Token)}
	}
	for _, p := range m.parameters {
		for _, v := range p.values {
			state.Values = append(state.Values, float64(v))
		}
	}
	data, err := json.Marshal(state)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// LoadCheckpoint restores model topology, tokenizer, and weights.
func LoadCheckpoint(path string) (*Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var state checkpoint
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("decode checkpoint: %w", err)
	}
	tokenizer := &Tokenizer{
		Tokens: make([]string, len(state.Tokens)),
		Merges: make([]Merge, len(state.Merges)),
	}
	for i, token := range state.Tokens {
		tokenizer.Tokens[i] = string(token)
	}
	for i, merge := range state.Merges {
		tokenizer.Merges[i] = Merge{string(merge.Left), string(merge.Right), string(merge.Token)}
	}
	tokenizer.reindex()
	model, err := NewModel(state.Config, tokenizer, 1)
	if err != nil {
		return nil, err
	}
	if len(state.Values) != model.ParameterCount() {
		return nil, fmt.Errorf("checkpoint has %d parameters; model requires %d", len(state.Values), model.ParameterCount())
	}
	index := 0
	for _, p := range model.parameters {
		for i := range p.values {
			p.values[i] = float32(state.Values[index])
			index++
		}
	}
	return model, nil
}
