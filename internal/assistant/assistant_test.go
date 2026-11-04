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

func TestAskUsesCoreForStats(t *testing.T) {
	core := NewCoreAPI("https://m3tal.test", "")
	core.Client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Path != "/api/v2/system/stats" {
			t.Errorf("unexpected API path %q", request.URL.Path)
		}
		return mockResponse(request, http.StatusOK, `{"status":"success","data":{"cpu_usage":15}}`), nil
	})}

	agent, err := New(nil, core)
	if err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := agent.Ask(context.Background(), "show system stats", nil, &output); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), "cpu_usage: 15") {
		t.Fatalf("stats response missing from output: %s", output.String())
	}
}

func TestAskFiltersSpecificContainerStatus(t *testing.T) {
	core := NewCoreAPI("https://m3tal.test", "")
	core.Client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Path != "/api/v2/services" {
			t.Errorf("unexpected API path %q", request.URL.Path)
		}
		return mockResponse(request, http.StatusOK, `{"status":"success","data":[{"names":["/jellyfin"],"state":"running"},{"names":["/traefik"],"state":"stopped"}]}`), nil
	})}

	agent, err := New(nil, core)
	if err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := agent.Ask(context.Background(), "is jellyfin running", nil, &output); err != nil {
		t.Fatal(err)
	}
	out := output.String()
	if !strings.Contains(out, "jellyfin") || !strings.Contains(out, "running") {
		t.Fatalf("expected jellyfin running in output, got: %s", out)
	}
	if strings.Contains(out, "traefik") {
		t.Fatalf("expected traefik to be filtered out for targeted query, got: %s", out)
	}
}

func TestUnknownRequestAsksForClarificationWithoutCallingCore(t *testing.T) {
	agent, err := New(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := agent.Ask(context.Background(), "qwerty blargh", nil, &output); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), "I'm not sure") {
		t.Fatalf("unknown fallback = %q", output.String())
	}
}

func TestAskListsApplicationsFromHubManifest(t *testing.T) {
	directory := t.TempDir()
	manifest := filepath.Join(directory, "manifest.yml")
	contents := "apps:\n  - id: monster-lab\n    name: Monster Lab\n    status: ready\n    description: Robotics learning application.\n"
	if err := os.WriteFile(manifest, []byte(contents), 0o600); err != nil {
		t.Fatal(err)
	}
	agent, err := New([]string{manifest}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := agent.Ask(context.Background(), "what apps are in the M3TAL Hub", nil, &output); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), "Monster Lab") || !strings.Contains(output.String(), "Robotics learning application") {
		t.Fatalf("app list missing manifest data: %s", output.String())
	}
}

func TestAskConfirmsStackActionsBeforeCallingCore(t *testing.T) {
	postCalls := 0
	core := NewCoreAPI("https://m3tal.test", "")
	core.Client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.Method == http.MethodGet && request.URL.Path == "/api/v2/stacks" {
			return mockResponse(request, http.StatusOK, `{"status":"success","data":[{"name":"media"}]}`), nil
		}
		postCalls++
		if request.Method != http.MethodPost || (request.URL.Path != "/api/v2/stacks/media/down" && request.URL.Path != "/api/v2/stacks/media/up") {
			t.Errorf("unexpected request: %s %s", request.Method, request.URL.Path)
		}
		return mockResponse(request, http.StatusOK, `{"status":"success","data":{"ok":true}}`), nil
	})}

	agent, err := New(nil, core)
	if err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	confirmed := false
	err = agent.Ask(context.Background(), "restart the media stack", func(action string) bool {
		confirmed = true
		if action != "confirm restart stack media" {
			t.Errorf("confirmation prompt = %q", action)
		}
		return false
	}, &output)
	if err != nil {
		t.Fatal(err)
	}
	if !confirmed || postCalls != 0 || !strings.Contains(output.String(), "Canceled") {
		t.Fatalf("denied action state: confirmed=%v mutation calls=%d output=%q", confirmed, postCalls, output.String())
	}

	output.Reset()
	err = agent.Ask(context.Background(), "restart the media stack", func(string) bool { return true }, &output)
	if err != nil {
		t.Fatal(err)
	}
	if postCalls != 2 || !strings.Contains(output.String(), "stack restart media completed") {
		t.Fatalf("confirmed action state: mutation calls=%d output=%q", postCalls, output.String())
	}
}

func TestMutationTargetsMustBeInLiveAllowlist(t *testing.T) {
	cases := []struct {
		kind, path, data string
	}{
		{"stack", "/api/v2/stacks", `[{"name":"media"}]`},
		{"container", "/api/v2/services", `[{"names":["/jellyfin"]}]`},
		{"service", "/api/v2/system/services", `[{"name":"m3tal-api.service"}]`},
	}
	for _, test := range cases {
		t.Run(test.kind, func(t *testing.T) {
			core := NewCoreAPI("https://m3tal.test", "")
			core.Client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
				if request.Method != http.MethodGet || request.URL.Path != test.path {
					t.Errorf("unexpected request: %s %s", request.Method, request.URL.Path)
				}
				return mockResponse(request, http.StatusOK, `{"status":"success","data":`+test.data+`}`), nil
			})}
			agent := &Assistant{core: core}
			name := map[string]string{"stack": "media", "container": "jellyfin", "service": "m3tal-api.service"}[test.kind]
			if err := agent.validateTarget(context.Background(), test.kind, name); err != nil {
				t.Fatalf("valid target rejected: %v", err)
			}
			if err := agent.validateTarget(context.Background(), test.kind, "unlisted-target"); err == nil {
				t.Fatal("unlisted target accepted")
			}
		})
	}
}

func TestContainerAndServiceMutationsUseScopedEndpoints(t *testing.T) {
	for _, test := range []struct{ kind, name, want string }{
		{"container", "jellyfin", "/api/v2/containers/jellyfin/restart"},
		{"service", "m3tal-api.service", "/api/v2/system/services/m3tal-api.service/restart"},
	} {
		t.Run(test.kind, func(t *testing.T) {
			core := NewCoreAPI("https://m3tal.test", "")
			core.Client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
				if request.Method != http.MethodPost || request.URL.Path != test.want {
					t.Errorf("request = %s %s, want POST %s", request.Method, request.URL.Path, test.want)
				}
				return mockResponse(request, http.StatusOK, `{"status":"success","data":{"ok":true}}`), nil
			})}
			if _, err := (&Assistant{core: core}).mutate(context.Background(), test.kind, test.name, "restart"); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestUnlistedStackNeverReachesMutationEndpoint(t *testing.T) {
	postCalls := 0
	core := NewCoreAPI("https://m3tal.test", "")
	core.Client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.Method == http.MethodGet {
			return mockResponse(request, http.StatusOK, `{"status":"success","data":[]}`), nil
		}
		postCalls++
		return mockResponse(request, http.StatusOK, `{"status":"success","data":{}}`), nil
	})}
	agent, err := New(nil, core)
	if err != nil {
		t.Fatal(err)
	}
	if err := agent.Ask(context.Background(), "restart stack unknown", func(string) bool { return true }, &bytes.Buffer{}); err == nil {
		t.Fatal("expected unlisted stack to be rejected")
	}
	if postCalls != 0 {
		t.Fatalf("unlisted target caused %d mutation requests", postCalls)
	}
}

func TestStackRestartReportsPartialFailure(t *testing.T) {
	core := NewCoreAPI("https://m3tal.test", "")
	core.Client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Path == "/api/v2/stacks/media/down" {
			return mockResponse(request, http.StatusOK, `{"status":"success","data":{"ok":true}}`), nil
		}
		return mockResponse(request, http.StatusInternalServerError, `{"status":"error","error":{"code":"START_FAILED","message":"compose failed"}}`), nil
	})}
	_, err := (&Assistant{core: core}).mutate(context.Background(), "stack", "media", "restart")
	if err == nil || !strings.Contains(err.Error(), "stopped the stack but failed during start") {
		t.Fatalf("restart error = %v, want partial failure details", err)
	}
}
