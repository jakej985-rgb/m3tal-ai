package lm

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
)

// This file implements preference data handling and DPO training
// (issue: Stage 3 optional DPO milestone).
//
// A preference dataset is JSONL; each line:
//
//	{"id":"...","source":"...","license":"...","category":"...",
//	 "prompt":[{"role":"user","content":"..."}],
//	 "chosen":"assistant reply A","rejected":"assistant reply B"}
//
// chosen/rejected are assistant-message contents appended to the prompt.
// Training uses a frozen reference model (a second load of the same
// checkpoint — its weights are never updated) and the standard DPO loss
//
//	L = -log sigmoid( beta * ( (lp_chosen - lp_rejected)_policy
//	                         - (lp_chosen - lp_rejected)_reference ) )
//
// where lp_* are summed log-probabilities over assistant (masked) tokens.

// PreferencePair is one chosen/rejected preference example.
type PreferencePair struct {
	ID       string        `json:"id"`
	Source   string        `json:"source,omitempty"`
	License  string        `json:"license,omitempty"`
	Category string        `json:"category,omitempty"`
	Prompt   []ChatMessage `json:"prompt"`
	Chosen   string        `json:"chosen"`
	Rejected string        `json:"rejected"`
}

// Validate checks the pair schema.
func (p PreferencePair) Validate() error {
	if p.ID == "" {
		return fmt.Errorf("missing id")
	}
	if len(p.Prompt) == 0 {
		return fmt.Errorf("%s: empty prompt", p.ID)
	}
	last := p.Prompt[len(p.Prompt)-1]
	if last.Role == RoleAssistant {
		return fmt.Errorf("%s: prompt must not end with an assistant turn", p.ID)
	}
	for i, m := range p.Prompt {
		switch m.Role {
		case RoleSystem, RoleUser, RoleTool:
		default:
			return fmt.Errorf("%s: prompt message %d has invalid role %q", p.ID, i, m.Role)
		}
		if strings.TrimSpace(m.Content) == "" {
			return fmt.Errorf("%s: prompt message %d empty", p.ID, i)
		}
	}
	if strings.TrimSpace(p.Chosen) == "" || strings.TrimSpace(p.Rejected) == "" {
		return fmt.Errorf("%s: chosen and rejected must be non-empty", p.ID)
	}
	if p.Chosen == p.Rejected {
		return fmt.Errorf("%s: chosen and rejected are identical", p.ID)
	}
	return nil
}

// Conversation returns the prompt plus the chosen assistant reply.
func (p PreferencePair) Conversation(chosen bool) Conversation {
	msgs := append([]ChatMessage(nil), p.Prompt...)
	content := p.Rejected
	if chosen {
		content = p.Chosen
	}
	msgs = append(msgs, ChatMessage{Role: RoleAssistant, Content: content})
	return Conversation{Messages: msgs}
}

// LoadPreferenceJSONL reads and validates a preference JSONL dataset.
// Malformed lines are counted and skipped.
func LoadPreferenceJSONL(path string) ([]PreferencePair, int64, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()
	var out []PreferencePair
	var records, invalid int64
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 16<<20)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		records++
		var p PreferencePair
		if err := json.Unmarshal([]byte(line), &p); err != nil || p.Validate() != nil {
			invalid++
			continue
		}
		out = append(out, p)
	}
	if err := scanner.Err(); err != nil {
		return nil, records, invalid, err
	}
	return out, records, invalid, nil
}

// dpoWindow is a prepared masked sequence for DPO (same shape as sftWindow).
type dpoWindow struct {
	input   []int
	targets []int
	mask    []bool
	active  int // number of masked target positions
}

// prepareDPOSequence serializes a conversation into a masked window.
func (m *Model) prepareDPOSequence(conv Conversation) (dpoWindow, error) {
	ids, idMask := conv.SerializedParts(m.Tokenizer)
	if len(ids) < 2 {
		return dpoWindow{}, fmt.Errorf("sequence too short")
	}
	if len(ids) > m.Config.Context+1 {
		ids = ids[len(ids)-(m.Config.Context+1):]
		idMask = idMask[len(idMask)-(m.Config.Context+1):]
	}
	w := dpoWindow{
		input:   make([]int, len(ids)-1),
		targets: make([]int, len(ids)-1),
		mask:    make([]bool, len(ids)-1),
	}
	for i := range w.input {
		w.input[i] = ids[i]
		w.targets[i] = ids[i+1]
		w.mask[i] = idMask[i+1]
		if w.mask[i] {
			w.active++
		}
	}
	if w.active == 0 {
		return dpoWindow{}, fmt.Errorf("no assistant tokens in window")
	}
	return w, nil
}

// maskedLogprobSum returns the summed log-probability of masked target
// tokens under the cached forward pass.
func maskedLogprobSum(cache forwardCache, w dpoWindow) float64 {
	vocab := len(cache.logits) / len(cache.ids)
	var sum float64
	for i, tgt := range w.targets {
		if !w.mask[i] {
			continue
		}
		sum += -tokenLoss32(cache.logits[i*vocab:(i+1)*vocab], tgt)
	}
	return sum
}

// scaleAllGrads multiplies every parameter gradient by factor.
func scaleAllGrads(m *Model, factor float64) {
	f := float32(factor)
	for _, p := range m.parameters {
		for i := range p.gradient {
			p.gradient[i] *= f
		}
	}
}

// sigmoid returns 1/(1+e^-x).
func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

// DPOConfig configures preference training.
type DPOConfig struct {
	TrainConfig
	Beta float64 // DPO temperature (typical 0.1)
}

// Normalize fills DPO defaults.
func (c *DPOConfig) Normalize() {
	c.TrainConfig.Normalize()
	if c.Beta <= 0 {
		c.Beta = 0.1
	}
}

// seqLogprob returns summed masked logprob of a conversation's assistant
// tokens under this model.
func (m *Model) seqLogprob(conv Conversation) (float64, error) {
	w, err := m.prepareDPOSequence(conv)
	if err != nil {
		return 0, err
	}
	cache := m.forward(w.input)
	return maskedLogprobSum(cache, w), nil
}

// dpoPairData is a prepared preference pair with frozen reference
// logprobs.
type dpoPairData struct {
	pair       PreferencePair
	chosen     dpoWindow
	rejected   dpoWindow
	refC, refR float64
}

// prepareDPOPair builds masked windows and reference logprobs for one
// preference pair. Reference logprobs are computed once (ref is frozen).
func (m *Model) prepareDPOPair(p PreferencePair, ref *Model) (dpoPairData, error) {
	cw, err := m.prepareDPOSequence(p.Conversation(true))
	if err != nil {
		return dpoPairData{}, err
	}
	rw, err := m.prepareDPOSequence(p.Conversation(false))
	if err != nil {
		return dpoPairData{}, err
	}
	lpC, err := ref.seqLogprob(p.Conversation(true))
	if err != nil {
		return dpoPairData{}, err
	}
	lpR, err := ref.seqLogprob(p.Conversation(false))
	if err != nil {
		return dpoPairData{}, err
	}
	return dpoPairData{pair: p, chosen: cw, rejected: rw, refC: lpC, refR: lpR}, nil
}

// dpoAccumGrads accumulates the DPO loss gradient for one prepared pair
// into m's parameter gradients (caller must zeroGradients first and call
// adamStepLR after). z is the DPO logit for this pair under the current
// policy.
//
// Derivation: L = -log sigmoid(z), z = beta*((lpC-lpR) - (refC-refR)).
// dL/dlpC_pol = -beta*sigmoid(-z), dL/dlpR_pol = +beta*sigmoid(-z).
// backward() yields the gradient of the MEAN masked CE = -(1/n)Σ∇logp,
// so Σ∇logp = -n·backwardOutput. Net scaling:
//
//	chosen:   (-beta·sig(-z)) · (-nC) = +beta·sig(-z)·nC
//	rejected: (+beta·sig(-z)) · (-nR) = -beta·sig(-z)·nR
func (m *Model) dpoAccumGrads(d dpoPairData, beta, z float64) {
	coef := beta * sigmoid(-z)
	polC := m.forward(d.chosen.input)
	polR := m.forward(d.rejected.input)
	m.backward(polC, d.chosen.targets, d.chosen.mask)
	scaleAllGrads(m, coef*float64(d.chosen.active))
	m.backward(polR, d.rejected.targets, d.rejected.mask)
	scaleAllGrads(m, -coef*float64(d.rejected.active))
}

// dpoLossValue computes the DPO loss for a prepared pair under the
// current policy weights (forward only — used by finite-difference
// tests).
func (m *Model) dpoLossValue(d dpoPairData, beta float64) float64 {
	lpC := maskedLogprobSum(m.forward(d.chosen.input), d.chosen)
	lpR := maskedLogprobSum(m.forward(d.rejected.input), d.rejected)
	z := beta * ((lpC - lpR) - (d.refC - d.refR))
	return -math.Log(sigmoid(z))
}

// TrainDPO runs DPO over preference pairs. ref must be a frozen copy of
// the policy model (same architecture/tokenizer, loaded from the same
// checkpoint); its weights are never modified. Returns mean DPO loss.
func (m *Model) TrainDPO(pairs []PreferencePair, ref *Model, cfg DPOConfig, logf func(TrainMetrics)) (float64, error) {
	cfg.Normalize()
	if cfg.MaxSteps < 0 || cfg.LearningRate <= 0 {
		return 0, fmt.Errorf("invalid DPO config")
	}
	if len(pairs) == 0 {
		return 0, fmt.Errorf("preference pairs required")
	}
	if ref == nil {
		return 0, fmt.Errorf("reference model required")
	}
	if ref.Tokenizer != m.Tokenizer && TokenizerFingerprint(ref.Tokenizer) != TokenizerFingerprint(m.Tokenizer) {
		return 0, fmt.Errorf("reference model tokenizer does not match policy")
	}

	data := make([]dpoPairData, 0, len(pairs))
	for _, p := range pairs {
		d, err := m.prepareDPOPair(p, ref)
		if err != nil {
			continue
		}
		data = append(data, d)
	}
	if len(data) == 0 {
		return 0, fmt.Errorf("no usable preference pairs within context length")
	}

	m.backend = nil
	rng := rand.New(rand.NewSource(cfg.Seed))
	var totalLoss float64
	var steps int64
	lastLog := m.step

	for cfg.MaxSteps <= 0 || m.step < cfg.MaxSteps {
		d := data[rng.Intn(len(data))]

		// Policy logprobs (forward only, for the loss value).
		polC := m.forward(d.chosen.input)
		polR := m.forward(d.rejected.input)
		lpC := maskedLogprobSum(polC, d.chosen)
		lpR := maskedLogprobSum(polR, d.rejected)

		z := cfg.Beta * ((lpC - lpR) - (d.refC - d.refR))
		loss := -math.Log(sigmoid(z))
		totalLoss += loss
		steps++

		zeroGradients(m.parameters)
		m.dpoAccumGrads(d, cfg.Beta, z)

		lr := cfg.learningRateAt(m.step + 1)
		gradNorm := m.adamStepLR(float32(lr), cfg.GradClip)

		if logf != nil && cfg.LogEvery > 0 && (m.step-lastLog) >= cfg.LogEvery {
			logf(TrainMetrics{
				Step:         m.step,
				Loss:         loss,
				LearningRate: lr,
				GradNorm:     gradNorm,
				TokensSeen:   steps * 2,
			})
			lastLog = m.step
		}
	}
	if steps == 0 {
		return 0, fmt.Errorf("no training steps completed")
	}
	return totalLoss / float64(steps), nil
}
