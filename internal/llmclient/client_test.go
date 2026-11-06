package llmclient

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"m3tal-llm/internal/lm"
	"m3tal-llm/internal/server"
)

func testClient(t *testing.T) (*Client, *httptest.Server) {
	t.Helper()
	tok, err := lm.TrainTokenizer([]string{
		"<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there!<|im_end|>\n",
		"The quick brown fox jumps over the lazy dog.",
	}, 256)
	if err != nil {
		t.Fatal(err)
	}
	model, err := lm.NewModel(lm.Config{Context: 32, Width: 32, Layers: 2, Heads: 4}, tok, 1)
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(server.New(model, server.Policy{}).Handler())
	t.Cleanup(ts.Close)
	return New(ts.URL), ts
}

func TestClientHealth(t *testing.T) {
	c, _ := testClient(t)
	if err := c.Health(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestClientGenerate(t *testing.T) {
	c, _ := testClient(t)
	resp, err := c.Generate(context.Background(), GenerateRequest{
		Prompt: "Hello", MaxTokens: 4, Seed: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text == "" {
		t.Fatal("empty text")
	}
}

func TestClientChat(t *testing.T) {
	c, _ := testClient(t)
	resp, err := c.Chat(context.Background(), ChatRequest{
		GenerateRequest: GenerateRequest{MaxTokens: 4, Seed: 1},
		Messages:        []ChatMessage{{Role: "user", Content: "Hello"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Template != lm.ChatTemplateVersion {
		t.Fatalf("template %q", resp.Template)
	}
}

func TestClientChatStream(t *testing.T) {
	c, _ := testClient(t)
	var deltas []string
	full, err := c.ChatStream(context.Background(), ChatRequest{
		GenerateRequest: GenerateRequest{MaxTokens: 4, Seed: 1},
		Messages:        []ChatMessage{{Role: "user", Content: "Hello"}},
	}, func(d string) error {
		deltas = append(deltas, d)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(deltas) == 0 {
		t.Fatal("no deltas streamed")
	}
	if strings.Join(deltas, "") != full {
		t.Fatalf("deltas %q != full %q", strings.Join(deltas, ""), full)
	}
}

func TestGeneratorAdapter(t *testing.T) {
	c, _ := testClient(t)
	g := NewGenerator(c)
	out, err := g.Generate("Hello", 4, 0.5, 7)
	if err != nil {
		t.Fatal(err)
	}
	if out == "" {
		t.Fatal("empty generation")
	}
}

func TestUnavailableServer(t *testing.T) {
	// No server listening → clean error, not a panic.
	c := New("http://127.0.0.1:1")
	if err := c.Health(context.Background()); err == nil {
		t.Fatal("expected health error for unavailable server")
	}
	if _, err := c.Generate(context.Background(), GenerateRequest{Prompt: "x"}); err == nil {
		t.Fatal("expected generate error")
	}
}

func TestServerErrorPropagation(t *testing.T) {
	c, _ := testClient(t)
	// Empty prompt → 400 with error body.
	_, err := c.Generate(context.Background(), GenerateRequest{Prompt: "  "})
	if err == nil || !strings.Contains(err.Error(), "prompt") {
		t.Fatalf("expected prompt error, got %v", err)
	}
}

func TestHealthJSONShape(t *testing.T) {
	// Verify the health endpoint returns the documented schema.
	_, ts := testClient(t)
	resp, err := http.Get(ts.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body["status"] != "ok" {
		t.Fatalf("health body %v", body)
	}
}
