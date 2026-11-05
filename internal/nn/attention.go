package nn

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"m3tal-ai/internal/autograd"
	"m3tal-ai/internal/tensor"
)

var (
	// ErrAttentionConfigInvalid indicates invalid attention configuration parameters.
	ErrAttentionConfigInvalid = errors.New("attention: invalid configuration")
	// ErrAttentionInputInvalid indicates invalid input dimensions or ranks.
	ErrAttentionInputInvalid = errors.New("attention: invalid input tensor")
)

// CausalSelfAttentionConfig holds configuration parameters for CausalSelfAttention.
type CausalSelfAttentionConfig struct {
	Dim       int          // Total model dimension (d_model)
	Heads     int          // Number of attention heads
	HeadDim   int          // Dimension per head (defaults to Dim / Heads if <= 0)
	MaxSeqLen int          // Maximum sequence length supported for causal masking/cache
	Bias      bool         // Whether linear projections use additive bias
	DType     tensor.DType // Data type: Float32 or Float64 (defaults to Float32)
}

// Normalize validates and fills default values for CausalSelfAttentionConfig.
func (c *CausalSelfAttentionConfig) Normalize() error {
	if c.Dim <= 0 {
		return fmt.Errorf("%w: Dim must be positive, got %d", ErrAttentionConfigInvalid, c.Dim)
	}
	if c.Heads <= 0 {
		return fmt.Errorf("%w: Heads must be positive, got %d", ErrAttentionConfigInvalid, c.Heads)
	}
	if c.HeadDim <= 0 {
		if c.Dim%c.Heads != 0 {
			return fmt.Errorf("%w: Dim %d must be divisible by Heads %d", ErrAttentionConfigInvalid, c.Dim, c.Heads)
		}
		c.HeadDim = c.Dim / c.Heads
	}
	if c.MaxSeqLen <= 0 {
		c.MaxSeqLen = 2048
	}
	if c.DType != tensor.Float32 && c.DType != tensor.Float64 {
		c.DType = tensor.Float32
	}
	return nil
}

// CausalSelfAttention implements multi-head causal self-attention with separate
// training (autograd-tracked batched parallel forward/backward) and inference (KV-cached token-by-token) paths.
type CausalSelfAttention struct {
	Config CausalSelfAttentionConfig

	// Weight matrices [inputDim, outputDim] where outputDim = Heads * HeadDim
	Wq *Parameter
	Wk *Parameter
	Wv *Parameter
	Wo *Parameter // [Heads * HeadDim, Dim]

	// Optional biases
	Bq *Parameter
	Bk *Parameter
	Bv *Parameter
	Bo *Parameter
}

// NewCausalSelfAttention initializes projection weights and optional biases deterministically.
func NewCausalSelfAttention(config CausalSelfAttentionConfig, rng *rand.Rand) (*CausalSelfAttention, error) {
	if err := config.Normalize(); err != nil {
		return nil, err
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}

	totalInner := config.Heads * config.HeadDim
	limit := 1.0 / math.Sqrt(float64(config.Dim))

	initWeights := func(rows, cols int) *Parameter {
		vals := make([]float64, rows*cols)
		for i := range vals {
			vals[i] = (rng.Float64()*2 - 1) * limit
		}
		return &Parameter{values: vals, grads: make([]float64, len(vals))}
	}

	initBias := func(size int) *Parameter {
		if !config.Bias {
			return nil
		}
		return &Parameter{values: make([]float64, size), grads: make([]float64, size)}
	}

	return &CausalSelfAttention{
		Config: config,
		Wq:     initWeights(config.Dim, totalInner),
		Wk:     initWeights(config.Dim, totalInner),
		Wv:     initWeights(config.Dim, totalInner),
		Wo:     initWeights(totalInner, config.Dim),
		Bq:     initBias(totalInner),
		Bk:     initBias(totalInner),
		Bv:     initBias(totalInner),
		Bo:     initBias(config.Dim),
	}, nil
}

// Parameters returns all trainable parameters of the layer.
func (c *CausalSelfAttention) Parameters() []*Parameter {
	params := []*Parameter{c.Wq, c.Wk, c.Wv, c.Wo}
	if c.Config.Bias {
		params = append(params, c.Bq, c.Bk, c.Bv, c.Bo)
	}
	return params
}

// ZeroGrad zeroes gradients for all parameters in the attention module.
func (c *CausalSelfAttention) ZeroGrad() {
	for _, p := range c.Parameters() {
		if p != nil {
			clear(p.grads)
		}
	}
}

func flatToCoords(flat int, shape []int) []int {
	if len(shape) == 0 {
		return []int{}
	}
	coords := make([]int, len(shape))
	curr := flat
	for i := len(shape) - 1; i >= 0; i-- {
		dim := shape[i]
		coords[i] = curr % dim
		curr /= dim
	}
	return coords
}

func parameterToVariable(p *Parameter, shape []int, dtype tensor.DType) (*autograd.Variable, error) {
	t, err := tensor.New(shape, dtype)
	if err != nil {
		return nil, err
	}
	numel := t.Numel()
	if dtype == tensor.Float32 {
		for i := 0; i < numel; i++ {
			idx := flatToCoords(i, shape)
			_ = t.SetFloat32At(float32(p.values[i]), idx...)
		}
	} else {
		for i := 0; i < numel; i++ {
			idx := flatToCoords(i, shape)
			_ = t.SetFloat64At(p.values[i], idx...)
		}
	}
	v, err := autograd.NewVariable(t, true)
	if err != nil {
		return nil, err
	}
	return v, nil
}

func accumulateGradToParameter(p *Parameter, v *autograd.Variable) error {
	if p == nil || v == nil || v.Grad == nil {
		return nil
	}
	numel := v.Grad.Numel()
	dtype := v.Grad.DType()
	shape := v.Grad.Shape()
	if dtype == tensor.Float32 {
		for i := 0; i < numel; i++ {
			idx := flatToCoords(i, shape)
			val, _ := v.Grad.Float32At(idx...)
			p.grads[i] += float64(val)
		}
	} else {
		for i := 0; i < numel; i++ {
			idx := flatToCoords(i, shape)
			val, _ := v.Grad.Float64At(idx...)
			p.grads[i] += val
		}
	}
	return nil
}

// Forward performs the training forward pass with causal masking and autograd graph tracking.
// Input x must have shape [BatchSize, SeqLen, Dim].
// Returns output variable with shape [BatchSize, SeqLen, Dim].
func (c *CausalSelfAttention) Forward(x *autograd.Variable) (*autograd.Variable, error) {
	if x == nil || x.Data == nil {
		return nil, ErrAttentionInputInvalid
	}
	shapeX := x.Shape()
	if len(shapeX) != 3 {
		return nil, fmt.Errorf("%w: expected 3D tensor [B, T, D], got %v", ErrAttentionInputInvalid, shapeX)
	}
	b, t, d := shapeX[0], shapeX[1], shapeX[2]
	if d != c.Config.Dim {
		return nil, fmt.Errorf("%w: input dim %d != config dim %d", ErrAttentionInputInvalid, d, c.Config.Dim)
	}
	dtype := x.DType()
	heads := c.Config.Heads
	headDim := c.Config.HeadDim
	totalInner := heads * headDim

	// Wrap parameters into autograd.Variables
	vWq, err := parameterToVariable(c.Wq, []int{d, totalInner}, dtype)
	if err != nil {
		return nil, err
	}
	vWk, err := parameterToVariable(c.Wk, []int{d, totalInner}, dtype)
	if err != nil {
		return nil, err
	}
	vWv, err := parameterToVariable(c.Wv, []int{d, totalInner}, dtype)
	if err != nil {
		return nil, err
	}
	vWo, err := parameterToVariable(c.Wo, []int{totalInner, d}, dtype)
	if err != nil {
		return nil, err
	}

	var vBq, vBk, vBv, vBo *autograd.Variable
	if c.Config.Bias {
		vBq, _ = parameterToVariable(c.Bq, []int{totalInner}, dtype)
		vBk, _ = parameterToVariable(c.Bk, []int{totalInner}, dtype)
		vBv, _ = parameterToVariable(c.Bv, []int{totalInner}, dtype)
		vBo, _ = parameterToVariable(c.Bo, []int{d}, dtype)
	}

	// 1. Projections: Q, K, V = X * W + b -> shape [B, T, totalInner]
	q, err := autograd.MatMul(x, vWq)
	if err != nil {
		return nil, err
	}
	k, err := autograd.MatMul(x, vWk)
	if err != nil {
		return nil, err
	}
	v, err := autograd.MatMul(x, vWv)
	if err != nil {
		return nil, err
	}
	if c.Config.Bias {
		q, _ = autograd.Add(q, vBq)
		k, _ = autograd.Add(k, vBk)
		v, _ = autograd.Add(v, vBv)
	}

	// 2. Custom differentiable multi-head causal attention forward & backward node
	// This performs:
	//   Q, K, V reshape & transpose -> [B, H, T, HeadDim]
	//   Scores = (Q * K^T) / sqrt(HeadDim)  [B, H, T, T]
	//   Causal Mask: Scores[..., i, j] = -inf for j > i
	//   Attn = Softmax(Scores, axis=-1)
	//   Attended = Attn * V -> [B, H, T, HeadDim]
	//   Reshape & transpose -> [B, T, totalInner]
	attended, err := causalAttentionOp(q, k, v, b, heads, t, headDim, dtype)
	if err != nil {
		return nil, err
	}

	// 3. Output projection: Attended * Wo + Bo -> [B, T, Dim]
	out, err := autograd.MatMul(attended, vWo)
	if err != nil {
		return nil, err
	}
	if c.Config.Bias {
		out, err = autograd.Add(out, vBo)
		if err != nil {
			return nil, err
		}
	}

	// Hook parameter gradient accumulation into each parameter variable's backward callback.
	// Leaf parameter variables have their backward callbacks executed after all incoming
	// gradients from downstream matrix multiplications and additions have accumulated.
	vWq.OnBackward(func() error { return accumulateGradToParameter(c.Wq, vWq) })
	vWk.OnBackward(func() error { return accumulateGradToParameter(c.Wk, vWk) })
	vWv.OnBackward(func() error { return accumulateGradToParameter(c.Wv, vWv) })
	vWo.OnBackward(func() error { return accumulateGradToParameter(c.Wo, vWo) })
	if c.Config.Bias {
		vBq.OnBackward(func() error { return accumulateGradToParameter(c.Bq, vBq) })
		vBk.OnBackward(func() error { return accumulateGradToParameter(c.Bk, vBk) })
		vBv.OnBackward(func() error { return accumulateGradToParameter(c.Bv, vBv) })
		vBo.OnBackward(func() error { return accumulateGradToParameter(c.Bo, vBo) })
	}

	return out, nil
}

// causalAttentionOp computes multi-head scaled causal attention with backward pass.
func causalAttentionOp(qVar, kVar, vVar *autograd.Variable, b, h, t, d int, dtype tensor.DType) (*autograd.Variable, error) {
	outTensor, err := tensor.New([]int{b, t, h * d}, dtype)
	if err != nil {
		return nil, err
	}

	scale := 1.0 / math.Sqrt(float64(d))
	scaleF32 := float32(scale)

	// Save attention weights for backward: [b, h, t, t]
	attnWeights := make([]float64, b*h*t*t)

	// Compute forward attention per batch and head
	for bi := 0; bi < b; bi++ {
		for hi := 0; hi < h; hi++ {
			headBase := (bi*h + hi) * t * t
			for i := 0; i < t; i++ {
				maxScore := -math.MaxFloat64
				scores := make([]float64, i+1)

				for j := 0; j <= i; j++ {
					var dot float64
					if dtype == tensor.Float32 {
						for di := 0; di < d; di++ {
							qVal, _ := qVar.Data.Float32At(bi, i, hi*d+di)
							kVal, _ := kVar.Data.Float32At(bi, j, hi*d+di)
							dot += float64(qVal * kVal)
						}
					} else {
						for di := 0; di < d; di++ {
							qVal, _ := qVar.Data.Float64At(bi, i, hi*d+di)
							kVal, _ := kVar.Data.Float64At(bi, j, hi*d+di)
							dot += qVal * kVal
						}
					}
					s := dot * scale
					scores[j] = s
					if s > maxScore {
						maxScore = s
					}
				}

				// Softmax over j <= i
				var sumExp float64
				for j := 0; j <= i; j++ {
					e := math.Exp(scores[j] - maxScore)
					scores[j] = e
					sumExp += e
				}
				for j := 0; j <= i; j++ {
					w := scores[j] / sumExp
					attnWeights[headBase+i*t+j] = w

					// Accumulate into attended output: Attn[i, j] * V[j]
					for di := 0; di < d; di++ {
						outIdx := hi*d + di
						if dtype == tensor.Float32 {
							vVal, _ := vVar.Data.Float32At(bi, j, outIdx)
							curr, _ := outTensor.Float32At(bi, i, outIdx)
							_ = outTensor.SetFloat32At(curr+float32(w)*vVal, bi, i, outIdx)
						} else {
							vVal, _ := vVar.Data.Float64At(bi, j, outIdx)
							curr, _ := outTensor.Float64At(bi, i, outIdx)
							_ = outTensor.SetFloat64At(curr+w*vVal, bi, i, outIdx)
						}
					}
				}
			}
		}
	}

	parents := make([]*autograd.Variable, 0, 3)
	if qVar.RequiresGrad {
		parents = append(parents, qVar)
	}
	if kVar.RequiresGrad {
		parents = append(parents, kVar)
	}
	if vVar.RequiresGrad {
		parents = append(parents, vVar)
	}

	var res *autograd.Variable
	res = autograd.NewOpVariable(outTensor, parents, func() error {
		if res.Grad == nil {
			return nil
		}

		var gradQ, gradK, gradV *tensor.Tensor
		if qVar.RequiresGrad {
			gradQ, _ = tensor.New([]int{b, t, h * d}, dtype)
		}
		if kVar.RequiresGrad {
			gradK, _ = tensor.New([]int{b, t, h * d}, dtype)
		}
		if vVar.RequiresGrad {
			gradV, _ = tensor.New([]int{b, t, h * d}, dtype)
		}

		for bi := 0; bi < b; bi++ {
			for hi := 0; hi < h; hi++ {
				headBase := (bi*h + hi) * t * t

				// 1. dL/dV[j, d] = sum_{i >= j} Attn[i, j] * gradOut[i, d]
				if vVar.RequiresGrad {
					for j := 0; j < t; j++ {
						for di := 0; di < d; di++ {
							idx := hi*d + di
							var sum float64
							for i := j; i < t; i++ {
								w := attnWeights[headBase+i*t+j]
								if dtype == tensor.Float32 {
									g, _ := res.Grad.Float32At(bi, i, idx)
									sum += float64(g) * w
								} else {
									g, _ := res.Grad.Float64At(bi, i, idx)
									sum += g * w
								}
							}
							if dtype == tensor.Float32 {
								_ = gradV.SetFloat32At(float32(sum), bi, j, idx)
							} else {
								_ = gradV.SetFloat64At(sum, bi, j, idx)
							}
						}
					}
				}

				// 2. dL/dAttn[i, j] = sum_d gradOut[i, d] * V[j, d]
				dAttn := make([]float64, t*t)
				for i := 0; i < t; i++ {
					for j := 0; j <= i; j++ {
						var sum float64
						for di := 0; di < d; di++ {
							idx := hi*d + di
							if dtype == tensor.Float32 {
								g, _ := res.Grad.Float32At(bi, i, idx)
								vVal, _ := vVar.Data.Float32At(bi, j, idx)
								sum += float64(g * vVal)
							} else {
								g, _ := res.Grad.Float64At(bi, i, idx)
								vVal, _ := vVar.Data.Float64At(bi, j, idx)
								sum += g * vVal
							}
						}
						dAttn[i*t+j] = sum
					}
				}

				// 3. dL/dScores[i, j]: Softmax backward
				// dScores[i, j] = Attn[i, j] * (dAttn[i, j] - sum_k Attn[i, k] * dAttn[i, k])
				dScores := make([]float64, t*t)
				for i := 0; i < t; i++ {
					var weightedSum float64
					for k := 0; k <= i; k++ {
						weightedSum += attnWeights[headBase+i*t+k] * dAttn[i*t+k]
					}
					for j := 0; j <= i; j++ {
						p := attnWeights[headBase+i*t+j]
						dScores[i*t+j] = p * (dAttn[i*t+j] - weightedSum) * scale
					}
				}

				// 4. dL/dQ[i, d] = sum_{j <= i} dScores[i, j] * K[j, d]
				if qVar.RequiresGrad {
					for i := 0; i < t; i++ {
						for di := 0; di < d; di++ {
							idx := hi*d + di
							var sum float64
							for j := 0; j <= i; j++ {
								if dtype == tensor.Float32 {
									kVal, _ := kVar.Data.Float32At(bi, j, idx)
									sum += dScores[i*t+j] * float64(kVal)
								} else {
									kVal, _ := kVar.Data.Float64At(bi, j, idx)
									sum += dScores[i*t+j] * kVal
								}
							}
							if dtype == tensor.Float32 {
								_ = gradQ.SetFloat32At(float32(sum), bi, i, idx)
							} else {
								_ = gradQ.SetFloat64At(sum, bi, i, idx)
							}
						}
					}
				}

				// 5. dL/dK[j, d] = sum_{i >= j} dScores[i, j] * Q[i, d]
				if kVar.RequiresGrad {
					for j := 0; j < t; j++ {
						for di := 0; di < d; di++ {
							idx := hi*d + di
							var sum float64
							for i := j; i < t; i++ {
								if dtype == tensor.Float32 {
									qVal, _ := qVar.Data.Float32At(bi, i, idx)
									sum += dScores[i*t+j] * float64(qVal)
								} else {
									qVal, _ := qVar.Data.Float64At(bi, i, idx)
									sum += dScores[i*t+j] * qVal
								}
							}
							if dtype == tensor.Float32 {
								_ = gradK.SetFloat32At(float32(sum), bi, j, idx)
							} else {
								_ = gradK.SetFloat64At(sum, bi, j, idx)
							}
						}
					}
				}
			}
		}

		if qVar.RequiresGrad {
			if err := qVar.AccGrad(gradQ); err != nil {
				return err
			}
		}
		if kVar.RequiresGrad {
			if err := kVar.AccGrad(gradK); err != nil {
				return err
			}
		}
		if vVar.RequiresGrad {
			if err := vVar.AccGrad(gradV); err != nil {
				return err
			}
		}
		_ = scaleF32
		return nil
	})

	return res, nil
}

// ForwardInference performs inference forward pass with optional KV cache support.
// Input x has shape [BatchSize, SeqLen, Dim].
// If cache is nil: performs standard causal self-attention without caching.
// If cache is non-nil: updates cache at layerIdx and attends over all cached keys/values.
// Returns output tensor of shape [BatchSize, SeqLen, Dim].
func (c *CausalSelfAttention) ForwardInference(x *tensor.Tensor, cache *KVCache, layerIdx int) (*tensor.Tensor, error) {
	if x == nil {
		return nil, ErrAttentionInputInvalid
	}
	shapeX := x.Shape()
	if len(shapeX) != 3 {
		return nil, fmt.Errorf("%w: expected 3D tensor [B, T, D], got %v", ErrAttentionInputInvalid, shapeX)
	}
	b, t, d := shapeX[0], shapeX[1], shapeX[2]
	if d != c.Config.Dim {
		return nil, fmt.Errorf("%w: input dim %d != config dim %d", ErrAttentionInputInvalid, d, c.Config.Dim)
	}
	dtype := x.DType()
	heads := c.Config.Heads
	headDim := c.Config.HeadDim
	totalInner := heads * headDim

	// 1. Compute Q, K, V projections for incoming tokens
	q, err := linearInference(x, c.Wq, c.Bq, d, totalInner, dtype)
	if err != nil {
		return nil, err
	}
	k, err := linearInference(x, c.Wk, c.Bk, d, totalInner, dtype)
	if err != nil {
		return nil, err
	}
	v, err := linearInference(x, c.Wv, c.Bv, d, totalInner, dtype)
	if err != nil {
		return nil, err
	}

	// Reshape K and V to [B, H, T, HeadDim] for cache update
	k4D, err := reshapeTo4D(k, b, t, heads, headDim, dtype)
	if err != nil {
		return nil, err
	}
	v4D, err := reshapeTo4D(v, b, t, heads, headDim, dtype)
	if err != nil {
		return nil, err
	}

	var allK, allV *tensor.Tensor
	var totalSeqLen int
	var startPos int

	if cache != nil {
		startPos = cache.LayerSeqLen(layerIdx)
		allK, allV, err = cache.Update(layerIdx, k4D, v4D)
		if err != nil {
			return nil, err
		}
		totalSeqLen = allK.Shape()[2]
	} else {
		allK = k4D
		allV = v4D
		totalSeqLen = t
		startPos = 0
	}

	// 2. Scaled Dot-Product Attention: Q of length t attending over allK of length totalSeqLen
	// For each token i in 0..t-1 (its global position is startPos + i):
	// It attends to keys j in 0..totalSeqLen-1 where j <= startPos + i (causal constraint).
	scale := 1.0 / math.Sqrt(float64(headDim))
	attended, err := tensor.New([]int{b, t, totalInner}, dtype)
	if err != nil {
		return nil, err
	}

	for bi := 0; bi < b; bi++ {
		for hi := 0; hi < heads; hi++ {
			for i := 0; i < t; i++ {
				globalPos := startPos + i
				validKeys := globalPos + 1
				if validKeys > totalSeqLen {
					validKeys = totalSeqLen
				}

				scores := make([]float64, validKeys)
				maxScore := -math.MaxFloat64

				for j := 0; j < validKeys; j++ {
					var dot float64
					if dtype == tensor.Float32 {
						for di := 0; di < headDim; di++ {
							qVal, _ := q.Float32At(bi, i, hi*headDim+di)
							kVal, _ := allK.Float32At(bi, hi, j, di)
							dot += float64(qVal * kVal)
						}
					} else {
						for di := 0; di < headDim; di++ {
							qVal, _ := q.Float64At(bi, i, hi*headDim+di)
							kVal, _ := allK.Float64At(bi, hi, j, di)
							dot += qVal * kVal
						}
					}
					s := dot * scale
					scores[j] = s
					if s > maxScore {
						maxScore = s
					}
				}

				var sumExp float64
				for j := 0; j < validKeys; j++ {
					e := math.Exp(scores[j] - maxScore)
					scores[j] = e
					sumExp += e
				}
				for j := 0; j < validKeys; j++ {
					w := scores[j] / sumExp
					for di := 0; di < headDim; di++ {
						outIdx := hi*headDim + di
						if dtype == tensor.Float32 {
							vVal, _ := allV.Float32At(bi, hi, j, di)
							curr, _ := attended.Float32At(bi, i, outIdx)
							_ = attended.SetFloat32At(curr+float32(w)*vVal, bi, i, outIdx)
						} else {
							vVal, _ := allV.Float64At(bi, hi, j, di)
							curr, _ := attended.Float64At(bi, i, outIdx)
							_ = attended.SetFloat64At(curr+w*vVal, bi, i, outIdx)
						}
					}
				}
			}
		}
	}

	// 3. Output projection: attended * Wo + Bo
	out, err := linearInference(attended, c.Wo, c.Bo, totalInner, d, dtype)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func linearInference(x *tensor.Tensor, w, b *Parameter, inDim, outDim int, dtype tensor.DType) (*tensor.Tensor, error) {
	shapeX := x.Shape()
	bSize, tSize := shapeX[0], shapeX[1]
	out, err := tensor.New([]int{bSize, tSize, outDim}, dtype)
	if err != nil {
		return nil, err
	}

	switch dtype {
	case tensor.Float32:
		for bi := 0; bi < bSize; bi++ {
			for ti := 0; ti < tSize; ti++ {
				for oi := 0; oi < outDim; oi++ {
					var sum float64
					if b != nil {
						sum = b.values[oi]
					}
					for ii := 0; ii < inDim; ii++ {
						xVal, _ := x.Float32At(bi, ti, ii)
						sum += float64(xVal) * w.values[ii*outDim+oi]
					}
					_ = out.SetFloat32At(float32(sum), bi, ti, oi)
				}
			}
		}
	case tensor.Float64:
		for bi := 0; bi < bSize; bi++ {
			for ti := 0; ti < tSize; ti++ {
				for oi := 0; oi < outDim; oi++ {
					var sum float64
					if b != nil {
						sum = b.values[oi]
					}
					for ii := 0; ii < inDim; ii++ {
						xVal, _ := x.Float64At(bi, ti, ii)
						sum += xVal * w.values[ii*outDim+oi]
					}
					_ = out.SetFloat64At(sum, bi, ti, oi)
				}
			}
		}
	}
	return out, nil
}

func reshapeTo4D(t *tensor.Tensor, b, seqLen, h, d int, dtype tensor.DType) (*tensor.Tensor, error) {
	out, err := tensor.New([]int{b, h, seqLen, d}, dtype)
	if err != nil {
		return nil, err
	}
	switch dtype {
	case tensor.Float32:
		for bi := 0; bi < b; bi++ {
			for si := 0; si < seqLen; si++ {
				for hi := 0; hi < h; hi++ {
					for di := 0; di < d; di++ {
						val, _ := t.Float32At(bi, si, hi*d+di)
						_ = out.SetFloat32At(val, bi, hi, si, di)
					}
				}
			}
		}
	case tensor.Float64:
		for bi := 0; bi < b; bi++ {
			for si := 0; si < seqLen; si++ {
				for hi := 0; hi < h; hi++ {
					for di := 0; di < d; di++ {
						val, _ := t.Float64At(bi, si, hi*d+di)
						_ = out.SetFloat64At(val, bi, hi, si, di)
					}
				}
			}
		}
	}
	return out, nil
}
