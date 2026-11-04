package assistant

import (
	"bytes"
	"context"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type stubGenerator struct {
	text string
	err  error
}

func (stub stubGenerator) Generate(string, int, float64, int64) (string, error) {
	return stub.text, stub.err
}

func TestScenarioSuite(t *testing.T) {
	directory := t.TempDir()
	guide := filepath.Join(directory, "install.md")
	if err := os.WriteFile(guide, []byte("# Install M3TAL\n\nRun the config wizard to install M3TAL Core.\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	core := NewCoreAPI("https://m3tal.test", "")
	core.Client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		switch {
		case request.Method == http.MethodGet && request.URL.Path == "/api/v2/system/stats":
			return mockResponse(request, http.StatusOK, `{"status":"success","data":{"cpu_usage":15}}`), nil
		case request.Method == http.MethodGet && request.URL.Path == "/api/v2/stacks":
			return mockResponse(request, http.StatusOK, `{"status":"success","data":[{"name":"media"},{"name":"medical"}]}`), nil
		case request.Method == http.MethodPost:
			t.Errorf("unexpected mutation %s", request.URL.Path)
			return mockResponse(request, http.StatusOK, `{"status":"success","data":{}}`), nil
		default:
			return mockResponse(request, http.StatusOK, `{"status":"success","data":{}}`), nil
		}
	})}
	agent, err := New([]string{directory}, core)
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name, query string
		confirm     func(string) bool
		want        string
		wantErr     string
	}{
		{name: "unknown", query: "qwerty blargh", want: "I'm not sure"},
		{name: "docs", query: "how do I install m3tal", want: "Install M3TAL"},
		{name: "denied confirm", query: "restart stack media", confirm: func(string) bool { return false }, want: "Canceled"},
		{name: "preview", query: "restart stack media", confirm: func(string) bool { return false }, want: "Preview:"},
		{name: "unlisted", query: "restart stack unknown", confirm: func(string) bool { return true }, wantErr: "allowlist"},
		{name: "ambiguous mutation", query: "restart stack and container", want: "I'm not sure"},
		{name: "fuzzy name suggestion", query: "restart stack med", confirm: func(string) bool { return true }, wantErr: "did you mean"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			var output bytes.Buffer
			err := agent.Ask(context.Background(), test.query, test.confirm, &output)
			if test.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantErr) {
					t.Fatalf("error = %v, want substring %q", err, test.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(output.String(), test.want) {
				t.Fatalf("output %q, want substring %q", output.String(), test.want)
			}
		})
	}
}

func TestInvalidPlanFallsBackToClassifier(t *testing.T) {
	core := NewCoreAPI("https://m3tal.test", "")
	core.Client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Path == "/api/v2/system/stats" {
			return mockResponse(request, http.StatusOK, `{"status":"success","data":{"cpu_usage":9}}`), nil
		}
		return mockResponse(request, http.StatusOK, `{"status":"success","data":{}}`), nil
	})}
	agent, err := New(nil, core)
	if err != nil {
		t.Fatal(err)
	}
	agent.SetLanguageModel(stubGenerator{text: "not json at all"})
	var output bytes.Buffer
	if err := agent.Ask(context.Background(), "show stats and health", nil, &output); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), "valid tool plan") {
		t.Fatalf("expected fallback notice, got %q", output.String())
	}
	if !strings.Contains(output.String(), "I'm not sure") && !strings.Contains(output.String(), "cpu_usage") {
		t.Fatalf("expected classifier routing after invalid plan, got %q", output.String())
	}
}

func TestValidPlanExecutesReadOnlySteps(t *testing.T) {
	core := NewCoreAPI("https://m3tal.test", "")
	core.Client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		switch request.URL.Path {
		case "/api/v2/system/stats":
			return mockResponse(request, http.StatusOK, `{"status":"success","data":{"cpu_usage":3}}`), nil
		case "/api/v2/system/health":
			return mockResponse(request, http.StatusOK, `{"status":"success","data":{"ok":true}}`), nil
		default:
			t.Errorf("unexpected path %s", request.URL.Path)
			return mockResponse(request, http.StatusOK, `{"status":"success","data":{}}`), nil
		}
	})}
	agent, err := New(nil, core)
	if err != nil {
		t.Fatal(err)
	}
	agent.SetLanguageModel(stubGenerator{text: `PLAN:{"steps":[{"tool":"stats"},{"tool":"health"}]}`})
	var output bytes.Buffer
	if err := agent.Ask(context.Background(), "show stats and health", nil, &output); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), "cpu_usage: 3") || !strings.Contains(output.String(), "ok: true") {
		t.Fatalf("plan output = %q", output.String())
	}
}

func TestUngroundedDraftFallsBackToExcerpts(t *testing.T) {
	directory := t.TempDir()
	path := filepath.Join(directory, "guide.md")
	if err := os.WriteFile(path, []byte("# Install\n\nUse the wizard to install M3TAL.\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	agent, err := New([]string{directory}, nil)
	if err != nil {
		t.Fatal(err)
	}
	agent.SetLanguageModel(stubGenerator{text: "I know this from memory without citing anything."})
	var output bytes.Buffer
	if err := agent.Ask(context.Background(), "how do I install m3tal", nil, &output); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), "not grounded") || !strings.Contains(output.String(), "Use the wizard") {
		t.Fatalf("output = %q", output.String())
	}
}

func TestGroundedDraftIsShownWhenItCitesASource(t *testing.T) {
	directory := t.TempDir()
	path := filepath.Join(directory, "guide.md")
	if err := os.WriteFile(path, []byte("# Install\n\nUse the wizard to install M3TAL.\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	agent, err := New([]string{directory}, nil)
	if err != nil {
		t.Fatal(err)
	}
	agent.SetLanguageModel(stubGenerator{text: "According to Install, use the wizard."})
	var output bytes.Buffer
	if err := agent.Ask(context.Background(), "how do I install m3tal", nil, &output); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), "According to Install") || !strings.Contains(output.String(), "Sources:") {
		t.Fatalf("output = %q", output.String())
	}
}

func TestLogsPronounUsesLastAllowlistedContainer(t *testing.T) {
	core := NewCoreAPI("https://m3tal.test", "")
	core.Client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		switch {
		case request.URL.Path == "/api/v2/services":
			return mockResponse(request, http.StatusOK, `{"status":"success","data":[{"names":["/jellyfin"]}]}`), nil
		case strings.Contains(request.URL.Path, "/logs"):
			if request.URL.Path != "/api/v2/containers/jellyfin/logs" {
				t.Errorf("logs path = %s", request.URL.Path)
			}
			return mockResponse(request, http.StatusOK, `{"status":"success","data":{"log":"ok"}}`), nil
		default:
			return mockResponse(request, http.StatusOK, `{"status":"success","data":{}}`), nil
		}
	})}
	agent, err := New(nil, core)
	if err != nil {
		t.Fatal(err)
	}
	if err := agent.Ask(context.Background(), "show logs for jellyfin", nil, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := agent.Ask(context.Background(), "show logs for it", nil, &output); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), "Recent logs for jellyfin") {
		t.Fatalf("output = %q", output.String())
	}
}

func TestJSONOutputKeepsRawCorePayload(t *testing.T) {
	core := NewCoreAPI("https://m3tal.test", "")
	core.Client = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return mockResponse(nil, http.StatusOK, `{"status":"success","data":{"cpu_usage":15}}`), nil
	})}
	agent, err := New(nil, core)
	if err != nil {
		t.Fatal(err)
	}
	agent.SetJSONOutput(true)
	var output bytes.Buffer
	if err := agent.Ask(context.Background(), "show system stats", nil, &output); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), `"cpu_usage": 15`) {
		t.Fatalf("json output = %q", output.String())
	}
}
