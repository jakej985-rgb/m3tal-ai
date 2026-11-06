// Package llmclient is a Go client for the m3tal-lm inference server.
// It implements the same Generate(prompt, maxTokens, temperature, seed)
// signature as the m3tal-ai assistant.Generator interface, so M3TAL
// plugins can route language-model requests to a local m3tal-lm server
// without coupling to model internals.
package llmclient

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Client calls a running `m3tal-lm serve` endpoint.
type Client struct {
	base string
	http *http.Client
}

// New returns a client for the server at baseURL (e.g.
// "http://127.0.0.1:8080").
func New(baseURL string) *Client {
	return &Client{
		base: strings.TrimRight(baseURL, "/"),
		http: &http.Client{Timeout: 5 * time.Minute},
	}
}

// Health checks server liveness.
func (c *Client) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, "GET", c.base+"/healthz", nil)
	if err != nil {
		return err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("health check: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check: status %d", resp.StatusCode)
	}
	return nil
}

// GenerateRequest mirrors the server's /v1/generate schema.
type GenerateRequest struct {
	Prompt            string   `json:"prompt"`
	MaxTokens         int      `json:"max_tokens,omitempty"`
	Temperature       float64  `json:"temperature,omitempty"`
	TopK              int      `json:"top_k,omitempty"`
	TopP              float64  `json:"top_p,omitempty"`
	Seed              int64    `json:"seed,omitempty"`
	Stop              []string `json:"stop,omitempty"`
	RepetitionPenalty float64  `json:"repetition_penalty,omitempty"`
}

// GenerateResponse is the /v1/generate reply.
type GenerateResponse struct {
	Text string `json:"text"`
}

// ChatMessage mirrors the server chat schema.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest is the /v1/chat schema.
type ChatRequest struct {
	GenerateRequest
	Messages []ChatMessage `json:"messages"`
	Stream   bool          `json:"stream,omitempty"`
}

// ChatResponse is the /v1/chat reply.
type ChatResponse struct {
	Text     string `json:"text"`
	Template string `json:"template"`
}

func (c *Client) post(ctx context.Context, path string, body any, out any) error {
	data, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", c.base+path, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		var e struct {
			Error string `json:"error"`
		}
		_ = json.NewDecoder(resp.Body).Decode(&e)
		if e.Error != "" {
			return fmt.Errorf("%s: %s (status %d)", path, e.Error, resp.StatusCode)
		}
		return fmt.Errorf("%s: status %d", path, resp.StatusCode)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// Generate calls POST /v1/generate.
func (c *Client) Generate(ctx context.Context, req GenerateRequest) (GenerateResponse, error) {
	var out GenerateResponse
	err := c.post(ctx, "/v1/generate", req, &out)
	return out, err
}

// Chat calls POST /v1/chat (non-streaming).
func (c *Client) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	var out ChatResponse
	err := c.post(ctx, "/v1/chat", req, &out)
	return out, err
}

// ChatStream calls POST /v1/chat with stream=true and invokes onDelta for
// each decoded text delta. It returns the concatenated output.
func (c *Client) ChatStream(ctx context.Context, req ChatRequest, onDelta func(string) error) (string, error) {
	req.Stream = true
	data, err := json.Marshal(req)
	if err != nil {
		return "", err
	}
	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.base+"/v1/chat", bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("/v1/chat stream: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return "", fmt.Errorf("/v1/chat stream: status %d: %s", resp.StatusCode, body)
	}
	var out strings.Builder
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 1<<20), 4<<20)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}
		var evt struct {
			Delta string `json:"delta"`
			Error string `json:"error"`
		}
		if err := json.Unmarshal([]byte(payload), &evt); err != nil {
			continue
		}
		if evt.Error != "" {
			return out.String(), fmt.Errorf("stream error: %s", evt.Error)
		}
		if evt.Delta != "" {
			out.WriteString(evt.Delta)
			if onDelta != nil {
				if err := onDelta(evt.Delta); err != nil {
					return out.String(), err
				}
			}
		}
	}
	return out.String(), scanner.Err()
}

// Generator adapts Client to the m3tal-ai assistant.Generator interface:
//
//	Generate(prompt string, maxTokens int, temperature float64, seed int64) (string, error)
type Generator struct {
	client      *Client
	MaxTokens   int     // default output cap override (0 = server default)
	Temperature float64 // override when > 0
	TopP        float64
	TopK        int
	Stop        []string
}

// NewGenerator returns a Generator backed by client.
func NewGenerator(client *Client) *Generator { return &Generator{client: client} }

// Generate implements the assistant.Generator contract.
func (g *Generator) Generate(prompt string, maxTokens int, temperature float64, seed int64) (string, error) {
	temp := temperature
	if g.Temperature > 0 {
		temp = g.Temperature
	}
	outCap := maxTokens
	if g.MaxTokens > 0 && (outCap <= 0 || g.MaxTokens < outCap) {
		outCap = g.MaxTokens
	}
	resp, err := g.client.Generate(context.Background(), GenerateRequest{
		Prompt:      prompt,
		MaxTokens:   outCap,
		Temperature: temp,
		TopP:        g.TopP,
		TopK:        g.TopK,
		Seed:        seed,
		Stop:        g.Stop,
	})
	if err != nil {
		return "", err
	}
	return resp.Text, nil
}
