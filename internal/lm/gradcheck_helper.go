package lm

import "fmt"

// ParameterInfo holds a reference to a model parameter and its metadata for inspection and testing.
type ParameterInfo struct {
	Name     string
	Values   []float32
	Gradient []float32
}

// ParametersInfo returns a slice of all trainable parameters and gradients.
func (m *Model) ParametersInfo() []ParameterInfo {
	var info []ParameterInfo
	info = append(info, ParameterInfo{Name: "tokenEmb", Values: m.tokenEmb.values, Gradient: m.tokenEmb.gradient})
	for l, b := range m.blocks {
		prefix := fmt.Sprintf("block[%d].", l)
		info = append(info, ParameterInfo{Name: prefix + "normAttn", Values: b.normAttn.values, Gradient: b.normAttn.gradient})
		info = append(info, ParameterInfo{Name: prefix + "normFF", Values: b.normFF.values, Gradient: b.normFF.gradient})
		info = append(info, ParameterInfo{Name: prefix + "query.weights", Values: b.query.weights.values, Gradient: b.query.weights.gradient})
		info = append(info, ParameterInfo{Name: prefix + "query.bias", Values: b.query.bias.values, Gradient: b.query.bias.gradient})
		info = append(info, ParameterInfo{Name: prefix + "key.weights", Values: b.key.weights.values, Gradient: b.key.weights.gradient})
		info = append(info, ParameterInfo{Name: prefix + "key.bias", Values: b.key.bias.values, Gradient: b.key.bias.gradient})
		info = append(info, ParameterInfo{Name: prefix + "value.weights", Values: b.value.weights.values, Gradient: b.value.weights.gradient})
		info = append(info, ParameterInfo{Name: prefix + "value.bias", Values: b.value.bias.values, Gradient: b.value.bias.gradient})
		info = append(info, ParameterInfo{Name: prefix + "attnOut.weights", Values: b.attnOut.weights.values, Gradient: b.attnOut.weights.gradient})
		info = append(info, ParameterInfo{Name: prefix + "attnOut.bias", Values: b.attnOut.bias.values, Gradient: b.attnOut.bias.gradient})
		info = append(info, ParameterInfo{Name: prefix + "ffGate.weights", Values: b.ffGate.weights.values, Gradient: b.ffGate.weights.gradient})
		info = append(info, ParameterInfo{Name: prefix + "ffGate.bias", Values: b.ffGate.bias.values, Gradient: b.ffGate.bias.gradient})
		info = append(info, ParameterInfo{Name: prefix + "ffUp.weights", Values: b.ffUp.weights.values, Gradient: b.ffUp.weights.gradient})
		info = append(info, ParameterInfo{Name: prefix + "ffUp.bias", Values: b.ffUp.bias.values, Gradient: b.ffUp.bias.gradient})
		info = append(info, ParameterInfo{Name: prefix + "ffDown.weights", Values: b.ffDown.weights.values, Gradient: b.ffDown.weights.gradient})
		info = append(info, ParameterInfo{Name: prefix + "ffDown.bias", Values: b.ffDown.bias.values, Gradient: b.ffDown.bias.gradient})
	}
	info = append(info, ParameterInfo{Name: "output.weights", Values: m.output.weights.values, Gradient: m.output.weights.gradient})
	info = append(info, ParameterInfo{Name: "output.bias", Values: m.output.bias.values, Gradient: m.output.bias.gradient})
	return info
}

// ComputeGradients executes forward, loss, and backward pass on the given input, targets, and loss mask.
func (m *Model) ComputeGradients(input, targets []int, mask []bool) float64 {
	zeroGradients(m.parameters)
	cache := m.forward(input)
	loss := meanWindowLoss(cache, targets, mask)
	m.backward(cache, targets, mask)
	return loss
}

// EvaluateWindowLoss computes meanWindowLoss for the given input, targets, and mask without modifying gradients.
func (m *Model) EvaluateWindowLoss(input, targets []int, mask []bool) float64 {
	cache := m.forward(input)
	return meanWindowLoss(cache, targets, mask)
}
