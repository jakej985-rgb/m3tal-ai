package lm

import (
	"math"
	"testing"
)

func dpoTestPair() PreferencePair {
	return PreferencePair{
		ID:      "p1",
		Source:  "self-owned",
		License: "self-owned",
		Prompt: []ChatMessage{
			{Role: RoleUser, Content: "What is 2+2?"},
		},
		Chosen:   "The answer is 4.",
		Rejected: "I don't know.",
	}
}

func TestPreferencePairValidate(t *testing.T) {
	good := dpoTestPair()
	if err := good.Validate(); err != nil {
		t.Fatalf("valid pair rejected: %v", err)
	}
	cases := map[string]func(*PreferencePair){
		"no-id":        func(p *PreferencePair) { p.ID = "" },
		"empty-prompt": func(p *PreferencePair) { p.Prompt = nil },
		"assistant-prompt-end": func(p *PreferencePair) {
			p.Prompt = append(p.Prompt, ChatMessage{Role: RoleAssistant, Content: "x"})
		},
		"empty-chosen": func(p *PreferencePair) { p.Chosen = " " },
		"identical":    func(p *PreferencePair) { p.Rejected = p.Chosen },
		"bad-role":     func(p *PreferencePair) { p.Prompt[0].Role = "robot" },
	}
	for name, mutate := range cases {
		p := dpoTestPair()
		mutate(&p)
		if err := p.Validate(); err == nil {
			t.Fatalf("%s: invalid pair accepted", name)
		}
	}
}

func TestLoadPreferenceJSONL(t *testing.T) {
	p := writeChatTestFile(t, []string{
		`{"id":"a","prompt":[{"role":"user","content":"q"}],"chosen":"yes","rejected":"no"}`,
		`{"id":"a","prompt":[{"role":"user","content":"q"}],"chosen":"yes","rejected":"no"}`, // dup content but loader doesn't dedup — both valid
		`{"id":"b","prompt":[],"chosen":"x","rejected":"y"}`,
		`garbage`,
	})
	pairs, records, invalid, err := LoadPreferenceJSONL(p)
	if err != nil {
		t.Fatal(err)
	}
	if records != 4 || invalid != 2 || len(pairs) != 2 {
		t.Fatalf("records=%d invalid=%d pairs=%d", records, invalid, len(pairs))
	}
}

func dpoModels(t *testing.T) (*Model, *Model) {
	t.Helper()
	tok := testChatTokenizer(t)
	cfg := Config{Context: 64, Width: 32, Layers: 2, Heads: 4}
	m, err := NewModel(cfg, tok, 1)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := NewModel(cfg, tok, 1)
	if err != nil {
		t.Fatal(err)
	}
	return m, ref
}

func TestDPOLossFiniteDifference(t *testing.T) {
	m, ref := dpoModels(t)
	pair := dpoTestPair()
	d, err := m.prepareDPOPair(pair, ref)
	if err != nil {
		t.Fatal(err)
	}

	// Analytic gradient via dpoAccumGrads.
	zeroGradients(m.parameters)
	lpC := maskedLogprobSum(m.forward(d.chosen.input), d.chosen)
	lpR := maskedLogprobSum(m.forward(d.rejected.input), d.rejected)
	z := 0.1 * ((lpC - lpR) - (d.refC - d.refR))
	m.dpoAccumGrads(d, 0.1, z)

	// Flatten analytic gradients in parameter order.
	flatGrad := []float64{}
	for _, p := range m.parameters {
		for _, g := range p.gradient {
			flatGrad = append(flatGrad, float64(g))
		}
	}

	// Sample a few parameters across tensors; compare to central
	// finite differences of the forward-only DPO loss.
	h := 1e-3
	type fp struct {
		p *parameter
		i int
	}
	var picks []fp
	for _, p := range m.parameters {
		step := len(p.values)/3 + 1
		for i := 0; i < len(p.values) && len(picks) < 200; i += step {
			picks = append(picks, fp{p, i})
		}
	}
	// Spread picks across the flat array.
	stride := len(picks) / 6
	if stride < 1 {
		stride = 1
	}
	checked := 0
	for k := 0; k < len(picks) && checked < 6; k += stride {
		f := picks[k]
		flatIdx := 0
		for _, p := range m.parameters {
			if p == f.p {
				flatIdx += f.i
				break
			}
			flatIdx += len(p.values)
		}
		orig := float64(f.p.values[f.i])
		f.p.values[f.i] = float32(orig + h)
		lp := m.dpoLossValue(d, 0.1)
		f.p.values[f.i] = float32(orig - h)
		lm := m.dpoLossValue(d, 0.1)
		f.p.values[f.i] = float32(orig)
		fd := (lp - lm) / (2 * h)
		an := flatGrad[flatIdx]
		if math.Abs(fd-an) > 2e-2*math.Max(1, math.Abs(fd)) {
			t.Fatalf("param %d: finite-diff %g vs analytic %g", flatIdx, fd, an)
		}
		checked++
	}
	if checked == 0 {
		t.Fatal("no parameters checked")
	}
}

func TestTrainDPORunsAndRefUnchanged(t *testing.T) {
	m, ref := dpoModels(t)
	// Snapshot ref weights.
	var refSnapshot []float64
	for _, p := range ref.parameters {
		for _, v := range p.values {
			refSnapshot = append(refSnapshot, float64(v))
		}
	}
	pairs := []PreferencePair{dpoTestPair(), {
		ID: "p2", Prompt: []ChatMessage{{Role: RoleUser, Content: "Say ok"}},
		Chosen: "ok", Rejected: "no way",
	}}
	cfg := DPOConfig{TrainConfig: TrainConfig{MaxSteps: 10, LearningRate: 0.01, Seed: 5}, Beta: 0.1}
	mean, err := m.TrainDPO(pairs, ref, cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	if mean <= 0 || math.IsNaN(mean) {
		t.Fatalf("bad DPO loss %v", mean)
	}
	if m.step != 10 {
		t.Fatalf("step=%d want 10", m.step)
	}
	// Ref weights must be untouched.
	idx := 0
	for _, p := range ref.parameters {
		for _, v := range p.values {
			if float64(v) != refSnapshot[idx] {
				t.Fatal("reference model weights modified")
			}
			idx++
		}
	}
}
