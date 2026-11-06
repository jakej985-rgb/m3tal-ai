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

// ParameterCount returns the analytic trainable-parameter count for this
// config and vocabulary size. It mirrors NewModel's layout exactly:
// token embedding (vocab*width), per-block RMSNorm gammas (2*width),
// attention Q/K/V/O projections (4*(width^2+width)), SwiGLU FFN
// (gate/up width->2*width, down 2*width->width = 6*width^2+5*width),
// and the output projection (width*vocab+vocab). Position encoding is
// RoPE, so there is no position table.
func (c Config) ParameterCount(vocab int) int {
	c.Normalize()
	w := c.Width
	perBlock := 10*w*w + 11*w
	return vocab*w + c.Layers*perBlock + w*vocab + vocab
}

// ParameterBreakdown splits ParameterCount into embedding, per-block,
// and output-projection components for reporting.
func (c Config) ParameterBreakdown(vocab int) (emb, blocks, output int) {
	c.Normalize()
	w := c.Width
	emb = vocab * w
	blocks = c.Layers * (10*w*w + 11*w)
	output = w*vocab + vocab
	return emb, blocks, output
}

// DefaultConfig fits an instruction-capable training prototype on CPU-only machines.
func DefaultConfig() Config { return Config{Context: 256, Width: 64, Layers: 4, Heads: 4} }

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
	step       int64
	seed       int64
	backend    inferenceBackend
}

// ParameterCount reports the number of trainable scalar weights.
func (m *Model) ParameterCount() int {
	count := 0
	for _, p := range m.parameters {
		count += len(p.values)
	}
	return count
}

func newParameter(size int) *parameter {
	return &parameter{
		values:   make([]float32, size),
		gradient: make([]float32, size),
		first:    make([]float32, size),
		second:   make([]float32, size),
	}
}

func (m *Model) addParameter(p *parameter) *parameter {
	m.parameters = append(m.parameters, p)
	return p
}

// NewModel initializes all weights with a deterministic seed.
// Every transformer block now carries two RMSNorm gamma vectors (pre-attention
// and pre-FFN). Position is encoded via RoPE; no posEmb table is allocated.
func NewModel(config Config, tokenizer *Tokenizer, seed int64) (*Model, error) {
	config.Normalize()
	if config.Context < 2 || config.Context > 2048 || config.Width < 2 || config.Width > 2048 ||
		tokenizer == nil || len(tokenizer.Tokens) < MinVocabSize || len(tokenizer.Tokens) > MaxVocabSize {
		return nil, fmt.Errorf("invalid model config or tokenizer")
	}
	if config.Layers < 1 || config.Layers > 16 {
		return nil, fmt.Errorf("layers must be between 1 and 16")
	}
	if config.Heads < 1 || config.Heads > 32 || config.Width%config.Heads != 0 {
		return nil, fmt.Errorf("width %d must be divisible by heads %d", config.Width, config.Heads)
	}
	rng := rand.New(rand.NewSource(seed))
	m := &Model{Config: config, Tokenizer: tokenizer, seed: seed}
	newMatrix := func(rows, cols int) *parameter {
		p := newParameter(rows * cols)
		limit := float32(1 / math.Sqrt(float64(cols)))
		for i := range p.values {
			p.values[i] = (rng.Float32()*2 - 1) * limit
		}
		return m.addParameter(p)
	}
	newLinear := func(input, output int) linear {
		weights := newMatrix(input, output)
		bias := m.addParameter(newParameter(output))
		return linear{weights: weights, bias: bias, input: input, output: output}
	}
	newNormGamma := func(width int) *parameter {
		p := m.addParameter(newParameter(width))
		for i := range p.values {
			p.values[i] = 1.0 // identity: no scaling at the start of training
		}
		return p
	}

	width, vocab := config.Width, len(tokenizer.Tokens)
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

func (m *Model) forward(ids []int) forwardCache {
	return forwardFloat32(m.Config, len(m.Tokenizer.Tokens), m.parameterView(), ids)
}

// backward propagates the cross-entropy loss gradient through the full model.
// If mask is non-nil, only positions where mask[i]==true contribute to the loss.
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

func (m *Model) adamStep(learningRate float32) { m.adamStepLR(learningRate, 1.0) }

// adamStepLR applies one Adam update with the given learning rate and
// global gradient-norm clip threshold. It returns the pre-clip gradient
// norm for logging.
func (m *Model) adamStepLR(learningRate float32, clipNorm float64) float64 {
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
	gradNorm := math.Sqrt(totalNormSq)
	clipScale := float32(1.0)
	if gradNorm > clipNorm {
		clipScale = float32(clipNorm / gradNorm)
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
	return gradNorm
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

// Train performs causal full-window training with Adam updates.
// Returns average pre-update cross-entropy per epoch.
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

func (m *Model) inferenceBackend() inferenceBackend {
	if m.backend == nil {
		m.backend = newInferenceBackend(m)
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
// FormatVersion 2 adds training metadata (step, seed, tokenizer fingerprint)
// and optional Adam optimizer state so interrupted runs can resume.
type checkpoint struct {
	FormatVersion int               `json:"format_version,omitempty"`
	Config        Config            `json:"config"`
	Tokens        [][]byte          `json:"tokens"`
	Merges        []checkpointMerge `json:"merges"`
	Values        []float64         `json:"values"`
	Step          int64             `json:"step,omitempty"`
	Seed          int64             `json:"seed,omitempty"`
	TokenizerHash string            `json:"tokenizer_hash,omitempty"`
	Optimizer     *optimizerState   `json:"optimizer,omitempty"`
}

// optimizerState stores Adam first/second moments aligned with Values order.
type optimizerState struct {
	First  []float64 `json:"first"`
	Second []float64 `json:"second"`
}

type checkpointMerge struct{ Left, Right, Token []byte }

// SaveCheckpoint writes tokenizer, weights, and training state (optimizer
// moments, step count, seed) to a local JSON checkpoint. The format is
// backward compatible: older readers ignore the added fields, and older
// checkpoints load with zeroed optimizer state.
func (m *Model) SaveCheckpoint(path string) error {
	state := checkpoint{
		FormatVersion: 2,
		Config:        m.Config,
		Step:          m.step,
		Seed:          m.seed,
		TokenizerHash: TokenizerFingerprint(m.Tokenizer),
		Tokens:        make([][]byte, len(m.Tokenizer.Tokens)),
		Merges:        make([]checkpointMerge, len(m.Tokenizer.Merges)),
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
	opt := &optimizerState{First: make([]float64, 0, len(state.Values)), Second: make([]float64, 0, len(state.Values))}
	for _, p := range m.parameters {
		for i := range p.values {
			opt.First = append(opt.First, float64(p.first[i]))
			opt.Second = append(opt.Second, float64(p.second[i]))
		}
	}
	state.Optimizer = opt
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
	if state.TokenizerHash != "" && TokenizerFingerprint(tokenizer) != state.TokenizerHash {
		return nil, fmt.Errorf("checkpoint tokenizer hash %q does not match serialized tokenizer", state.TokenizerHash)
	}
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
	if state.Step != 0 {
		model.step = state.Step
	}
	if state.Seed != 0 {
		model.seed = state.Seed
	}
	if state.Optimizer != nil {
		if len(state.Optimizer.First) != model.ParameterCount() || len(state.Optimizer.Second) != model.ParameterCount() {
			return nil, fmt.Errorf("checkpoint optimizer state has %d/%d moments; model requires %d",
				len(state.Optimizer.First), len(state.Optimizer.Second), model.ParameterCount())
		}
		idx := 0
		for _, p := range model.parameters {
			for i := range p.values {
				p.first[i] = float32(state.Optimizer.First[idx])
				p.second[i] = float32(state.Optimizer.Second[idx])
				idx++
			}
		}
	}
	return model, nil
}

// Step returns the optimizer step count stored in this model (0 for
// freshly initialized models). Used by model cards and packaging.
func (m *Model) Step() int64 { return m.step }
