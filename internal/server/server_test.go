package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"m3tal-llm/internal/lm"
)

func testServer(t *testing.T, policy Policy) (*Server, *httptest.Server) {
	t.Helper()
	tok := testTokenizer(t)
	cfg := lm.Config{Context: 32, Width: 32, Layers: 2, Heads: 4}
	model, err := lm.NewModel(cfg, tok, 1)
	if err != nil {
		t.Fatal(err)
	}
	srv := New(model, policy)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return srv, ts
}

func testTokenizer(t *testing.T) *lm.Tokenizer {
	t.Helper()
	tok, err := lm.TrainTokenizer([]string{
		"<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there!<|im_end|>\n",
		"The quick brown fox jumps over the lazy dog.",
	}, 256)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func post(t *testing.T, url string, body any) (*http.Response, []byte) {
	t.Helper()
	data, _ := json.Marshal(body)
	resp, err := http.Post(url, "application/json", bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	buf := new(bytes.Buffer)
	_, _ = buf.ReadFrom(resp.Body)
	return resp, buf.Bytes()
}

func TestHealth(t *testing.T) {
	_, ts := testServer(t, Policy{})
	resp, err := http.Get(ts.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
}

func TestGenerateEndpoint(t *testing.T) {
	_, ts := testServer(t, Policy{})
	resp, body := post(t, ts.URL+"/v1/generate", map[string]any{
		"prompt": "Hello", "max_tokens": 4, "seed": 1,
	})
	if resp.StatusCode != 200 {
		t.Fatalf("status %d body %s", resp.StatusCode, body)
	}
	var out struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatal(err)
	}
	if out.Text == "" {
		t.Fatal("empty generation")
	}
}

func TestChatEndpoint(t *testing.T) {
	_, ts := testServer(t, Policy{})
	resp, body := post(t, ts.URL+"/v1/chat", map[string]any{
		"messages": []map[string]string{
			{"role": "user", "content": "Hello"},
		},
		"max_tokens": 4, "seed": 1,
	})
	if resp.StatusCode != 200 {
		t.Fatalf("status %d body %s", resp.StatusCode, body)
	}
	var out struct {
		Text     string `json:"text"`
		Template string `json:"template"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatal(err)
	}
	if out.Template != lm.ChatTemplateVersion {
		t.Fatalf("template %q", out.Template)
	}
	if strings.Contains(out.Text, "<|im_end|>") {
		t.Fatalf("stop marker leaked: %q", out.Text)
	}
}

func TestChatValidationErrors(t *testing.T) {
	_, ts := testServer(t, Policy{})
	cases := []map[string]any{
		{"messages": []any{}}, // empty
		{"messages": []map[string]string{{"role": "user", "content": ""}}},        // invalid conv
		{"messages": []map[string]string{{"role": "assistant", "content": "hi"}}}, // no non-assistant
	}
	for i, body := range cases {
		resp, respBody := post(t, ts.URL+"/v1/chat", body)
		if resp.StatusCode != 400 {
			t.Fatalf("case %d: status %d body %s", i, resp.StatusCode, respBody)
		}
	}
}

func TestPromptTokenLimit(t *testing.T) {
	_, ts := testServer(t, Policy{MaxPromptTokens: 8})
	longPrompt := strings.Repeat("word ", 100)
	resp, body := post(t, ts.URL+"/v1/generate", map[string]any{
		"prompt": longPrompt, "max_tokens": 2,
	})
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("status %d body %s", resp.StatusCode, body)
	}
}

func TestOutputTokenClamp(t *testing.T) {
	srv, ts := testServer(t, Policy{MaxOutputTokens: 3})
	if srv.Policy().MaxOutputTokens != 3 {
		t.Fatal("policy not applied")
	}
	resp, _ := post(t, ts.URL+"/v1/generate", map[string]any{
		"prompt": "Hi", "max_tokens": 100,
	})
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
}

func TestMalformedJSON(t *testing.T) {
	_, ts := testServer(t, Policy{})
	resp, err := http.Post(ts.URL+"/v1/generate", "application/json",
		strings.NewReader("{not json"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Fatalf("status %d", resp.StatusCode)
	}
}

func TestOversizedBody(t *testing.T) {
	_, ts := testServer(t, Policy{MaxBodyBytes: 128})
	big := map[string]any{"prompt": strings.Repeat("x", 1000)}
	resp, _ := post(t, ts.URL+"/v1/generate", big)
	if resp.StatusCode != 400 {
		t.Fatalf("status %d", resp.StatusCode)
	}
}

func TestChatStreamSSE(t *testing.T) {
	_, ts := testServer(t, Policy{})
	data, _ := json.Marshal(map[string]any{
		"messages":   []map[string]string{{"role": "user", "content": "Hi"}},
		"max_tokens": 3, "stream": true, "seed": 2,
	})
	resp, err := http.Post(ts.URL+"/v1/chat", "application/json", bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("content-type %q", ct)
	}
	body := new(bytes.Buffer)
	_, _ = body.ReadFrom(resp.Body)
	text := body.String()
	if !strings.Contains(text, "data: ") || !strings.Contains(text, "[DONE]") {
		t.Fatalf("bad SSE stream: %q", text)
	}
}

func TestDefaultPolicyApplied(t *testing.T) {
	srv, _ := testServer(t, Policy{})
	p := srv.Policy()
	if p.MaxPromptTokens != DefaultPolicy.MaxPromptTokens ||
		p.MaxOutputTokens != DefaultPolicy.MaxOutputTokens ||
		p.MaxBodyBytes != DefaultPolicy.MaxBodyBytes ||
		p.RequestTimeout != DefaultPolicy.RequestTimeout {
		t.Fatalf("defaults not applied: %+v", p)
	}
}
