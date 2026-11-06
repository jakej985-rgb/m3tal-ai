// Package server exposes a trained lm.Model over HTTP for M3TAL
// integration (issue: Stage 4 M3TAL router integration + safety
// controls). The safety policy is configuration, not model weights:
// request size, prompt/output token limits, and timeouts are enforced
// here, independently of the checkpoint.
package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"m3tal-llm/internal/lm"
)

// Policy bounds request cost and output size. Zero values mean "use
// DefaultPolicy".
type Policy struct {
	MaxBodyBytes    int64         // request body cap (default 1 MiB)
	MaxPromptTokens int           // encoded prompt cap (default 4096)
	MaxOutputTokens int           // generation cap (default 512)
	RequestTimeout  time.Duration // per-request cap (default 2m)
	AllowedRoles    map[string]bool
}

// DefaultPolicy is the built-in safety boundary.
var DefaultPolicy = Policy{
	MaxBodyBytes:    1 << 20,
	MaxPromptTokens: 4096,
	MaxOutputTokens: 512,
	RequestTimeout:  2 * time.Minute,
}

func (p Policy) withDefaults() Policy {
	if p.MaxBodyBytes <= 0 {
		p.MaxBodyBytes = DefaultPolicy.MaxBodyBytes
	}
	if p.MaxPromptTokens <= 0 {
		p.MaxPromptTokens = DefaultPolicy.MaxPromptTokens
	}
	if p.MaxOutputTokens <= 0 {
		p.MaxOutputTokens = DefaultPolicy.MaxOutputTokens
	}
	if p.RequestTimeout <= 0 {
		p.RequestTimeout = DefaultPolicy.RequestTimeout
	}
	return p
}

// Server serves lm.Model inference endpoints.
type Server struct {
	model  *lm.Model
	policy Policy
	mux    *http.ServeMux
}

// New creates an inference server around model.
func New(model *lm.Model, policy Policy) *Server {
	s := &Server{model: model, policy: policy.withDefaults(), mux: http.NewServeMux()}
	s.mux.HandleFunc("GET /healthz", s.handleHealth)
	s.mux.HandleFunc("POST /v1/generate", s.handleGenerate)
	s.mux.HandleFunc("POST /v1/chat", s.handleChat)
	return s
}

// Handler returns the HTTP handler.
func (s *Server) Handler() http.Handler { return s.mux }

// Model returns the underlying model (for metadata endpoints).
func (s *Server) Model() *lm.Model { return s.model }

// Policy returns the effective safety policy.
func (s *Server) Policy() Policy { return s.policy }

type errorResponse struct {
	Error string `json:"error"`
}

func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(errorResponse{Error: msg})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":     "ok",
		"parameters": s.model.ParameterCount(),
		"backend":    "cpu-or-cuda", // refined by caller; model abstracts backend
	})
}

// generateRequest is the /v1/generate schema.
type generateRequest struct {
	Prompt            string   `json:"prompt"`
	MaxTokens         int      `json:"max_tokens,omitempty"`
	Temperature       float64  `json:"temperature,omitempty"`
	TopK              int      `json:"top_k,omitempty"`
	TopP              float64  `json:"top_p,omitempty"`
	Seed              int64    `json:"seed,omitempty"`
	Stop              []string `json:"stop,omitempty"`
	RepetitionPenalty float64  `json:"repetition_penalty,omitempty"`
}

// chatRequest is the /v1/chat schema.
type chatRequest struct {
	generateRequest
	Messages []lm.ChatMessage `json:"messages"`
	Stream   bool             `json:"stream,omitempty"`
}

func clampOutput(n, cap int) int {
	if n <= 0 || n > cap {
		return cap
	}
	return n
}

func (s *Server) decode(w http.ResponseWriter, r *http.Request, v any) error {
	if r.Body == nil {
		return errors.New("empty request body")
	}
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, s.policy.MaxBodyBytes))
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	return nil
}

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), s.policy.RequestTimeout)
	defer cancel()
	r = r.WithContext(ctx)

	var req generateRequest
	if err := s.decode(w, r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if strings.TrimSpace(req.Prompt) == "" {
		writeError(w, http.StatusBadRequest, "prompt is required")
		return
	}
	promptTokens := len(s.model.Tokenizer.Encode(req.Prompt))
	if promptTokens > s.policy.MaxPromptTokens {
		writeError(w, http.StatusRequestEntityTooLarge,
			fmt.Sprintf("prompt has %d tokens; limit is %d", promptTokens, s.policy.MaxPromptTokens))
		return
	}
	opts := lm.GenerateOptions{
		MaxTokens:         clampOutput(req.MaxTokens, s.policy.MaxOutputTokens),
		Temperature:       req.Temperature,
		TopK:              req.TopK,
		TopP:              req.TopP,
		Seed:              req.Seed,
		RepetitionPenalty: req.RepetitionPenalty,
		Stop:              req.Stop,
	}
	out, err := s.model.GenerateWithOptions(req.Prompt, opts)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"text": out})
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), s.policy.RequestTimeout)
	defer cancel()
	r = r.WithContext(ctx)

	var req chatRequest
	if err := s.decode(w, r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages is required")
		return
	}
	// Generation prompts don't require a trailing assistant turn, so we
	// validate structure manually: known roles, non-empty content, and at
	// least one non-assistant turn to reply to.
	hasNonAssistant := false
	for i, m := range req.Messages {
		switch m.Role {
		case lm.RoleSystem, lm.RoleUser, lm.RoleAssistant, lm.RoleTool:
		default:
			writeError(w, http.StatusBadRequest, fmt.Sprintf("message %d: unknown role %q", i, m.Role))
			return
		}
		if strings.TrimSpace(m.Content) == "" {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("message %d: empty content", i))
			return
		}
		if m.Role != lm.RoleAssistant {
			hasNonAssistant = true
		}
	}
	if !hasNonAssistant {
		writeError(w, http.StatusBadRequest, "conversation has no user/system/tool turn to reply to")
		return
	}
	conv := lm.Conversation{Messages: req.Messages}
	promptIDs := conv.EncodePrompt(s.model.Tokenizer)
	if len(promptIDs) > s.policy.MaxPromptTokens {
		writeError(w, http.StatusRequestEntityTooLarge,
			fmt.Sprintf("prompt has %d tokens; limit is %d", len(promptIDs), s.policy.MaxPromptTokens))
		return
	}
	promptText := s.model.Tokenizer.Decode(promptIDs)
	opts := lm.GenerateOptions{
		MaxTokens:         clampOutput(req.MaxTokens, s.policy.MaxOutputTokens),
		Temperature:       req.Temperature,
		TopK:              req.TopK,
		TopP:              req.TopP,
		Seed:              req.Seed,
		RepetitionPenalty: req.RepetitionPenalty,
		Stop:              append(append([]string{}, req.Stop...), lm.StopMarkers()...),
	}
	if req.Stream {
		s.handleChatStream(w, promptText, opts)
		return
	}
	out, err := s.model.GenerateWithOptions(promptText, opts)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"text":     out,
		"template": lm.ChatTemplateVersion,
	})
}

// handleChatStream serves Server-Sent Events: each "data:" line carries
// {"delta": "..."}; the stream ends with "data: [DONE]".
func (s *Server) handleChatStream(w http.ResponseWriter, prompt string, opts lm.GenerateOptions) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	writeSSE := func(v any) bool {
		data, err := json.Marshal(v)
		if err != nil {
			return false
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
			return false
		}
		flusher.Flush()
		return true
	}
	_, err := s.model.GenerateStream(prompt, opts, func(delta string) error {
		if !writeSSE(map[string]string{"delta": delta}) {
			return fmt.Errorf("client disconnected")
		}
		return nil
	})
	if err != nil {
		writeSSE(map[string]string{"error": err.Error()})
	}
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}
