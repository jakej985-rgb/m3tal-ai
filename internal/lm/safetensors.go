package lm

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// This file implements read/write for the safetensors weight format
// (https://github.com/huggingface/safetensors), a simple container of named
// raw tensors with a JSON header. Model weights are exported as F32 tensors
// named after their position in the parameter list; the parameter order is
// recorded in the "__metadata__" block so a reader can restore exact order.
//
// Safetensors carries weights only. Training state (step, optimizer moments,
// tokenizer) stays in the JSON checkpoint written by SaveCheckpoint.

const safetensorsFormat = "m3tal.safetensors"

// parameterSpecs returns every trainable parameter with a stable name,
// mirroring the naming used by gradcheck ParametersInfo.
func (m *Model) parameterSpecs() []struct {
	Name string
	P    *parameter
} {
	var specs []struct {
		Name string
		P    *parameter
	}
	add := func(name string, p *parameter) {
		specs = append(specs, struct {
			Name string
			P    *parameter
		}{name, p})
	}
	addLinear := func(prefix string, l linear) {
		add(prefix+".weights", l.weights)
		add(prefix+".bias", l.bias)
	}
	add("token_emb", m.tokenEmb)
	for i, b := range m.blocks {
		prefix := fmt.Sprintf("blocks.%d", i)
		add(prefix+".norm_attn", b.normAttn)
		add(prefix+".norm_ff", b.normFF)
		addLinear(prefix+".query", b.query)
		addLinear(prefix+".key", b.key)
		addLinear(prefix+".value", b.value)
		addLinear(prefix+".attn_out", b.attnOut)
		addLinear(prefix+".ff_gate", b.ffGate)
		addLinear(prefix+".ff_up", b.ffUp)
		addLinear(prefix+".ff_down", b.ffDown)
	}
	addLinear("output", m.output)
	return specs
}

type stHeaderEntry struct {
	Dtype       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets []int  `json:"data_offsets"`
}

// SaveSafetensors writes all model weights to path in safetensors format.
func (m *Model) SaveSafetensors(path string) error {
	specs := m.parameterSpecs()
	header := map[string]json.RawMessage{}
	meta := map[string]string{
		"format":          safetensorsFormat,
		"format_version":  "1",
		"model_config":    mustJSON(m.Config),
		"tokenizer_hash":  TokenizerFingerprint(m.Tokenizer),
		"parameter_count": fmt.Sprint(len(specs)),
	}
	header["__metadata__"], _ = json.Marshal(meta)

	var data bytes.Buffer
	offset := 0
	for _, s := range specs {
		entry := stHeaderEntry{
			Dtype:       "F32",
			Shape:       []int{len(s.P.values)},
			DataOffsets: []int{offset, offset + len(s.P.values)*4},
		}
		raw, _ := json.Marshal(entry)
		header[s.Name] = raw
		var b [4]byte
		for _, v := range s.P.values {
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
			data.Write(b[:])
		}
		offset += len(s.P.values) * 4
	}

	hj, err := json.Marshal(header)
	if err != nil {
		return err
	}
	out := &bytes.Buffer{}
	var lb [8]byte
	binary.LittleEndian.PutUint64(lb[:], uint64(len(hj)))
	out.Write(lb[:])
	out.Write(hj)
	out.Write(data.Bytes())

	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, out.Bytes(), 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func mustJSON(v interface{}) string {
	b, _ := json.Marshal(v)
	return string(b)
}

// LoadSafetensors reads a safetensors weight file and copies values into the
// model's parameters in parameterSpecs order. It verifies tensor count,
// names, shapes, and dtype before writing anything.
func (m *Model) LoadSafetensors(path string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if len(raw) < 8 {
		return fmt.Errorf("safetensors file too small")
	}
	hlen := binary.LittleEndian.Uint64(raw[:8])
	if hlen == 0 || 8+int(hlen) > len(raw) {
		return fmt.Errorf("safetensors header length %d out of bounds", hlen)
	}
	var header map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+hlen], &header); err != nil {
		return fmt.Errorf("decode safetensors header: %w", err)
	}
	delete(header, "__metadata__")

	specs := m.parameterSpecs()
	if len(header) != len(specs) {
		return fmt.Errorf("safetensors has %d tensors; model requires %d", len(header), len(specs))
	}
	base := 8 + int(hlen)
	for _, s := range specs {
		entryRaw, ok := header[s.Name]
		if !ok {
			return fmt.Errorf("safetensors missing tensor %q", s.Name)
		}
		var entry stHeaderEntry
		if err := json.Unmarshal(entryRaw, &entry); err != nil {
			return fmt.Errorf("bad header for %q: %w", s.Name, err)
		}
		if entry.Dtype != "F32" {
			return fmt.Errorf("tensor %q dtype %q, want F32", s.Name, entry.Dtype)
		}
		if len(entry.Shape) != 1 || entry.Shape[0] != len(s.P.values) {
			return fmt.Errorf("tensor %q shape %v, want [%d]", s.Name, entry.Shape, len(s.P.values))
		}
		start, end := entry.DataOffsets[0], entry.DataOffsets[1]
		if start < 0 || end-start != len(s.P.values)*4 || base+end > len(raw) {
			return fmt.Errorf("tensor %q data offsets %v out of bounds", s.Name, entry.DataOffsets)
		}
		payload := raw[base+start : base+end]
		for i := range s.P.values {
			s.P.values[i] = math.Float32frombits(binary.LittleEndian.Uint32(payload[i*4:]))
		}
	}
	return nil
}
