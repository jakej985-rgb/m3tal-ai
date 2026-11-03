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
	core := NewCoreAPI("http://m3tal.test", "")
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
	if !strings.Contains(output.String(), `"cpu_usage": 15`) {
		t.Fatalf("stats response missing from output: %s", output.String())
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
	calls := 0
	core := NewCoreAPI("http://m3tal.test", "")
	core.Client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		calls++
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
		if action != "restart stack media" {
			t.Errorf("confirmation prompt = %q", action)
		}
		return false
	}, &output)
	if err != nil {
		t.Fatal(err)
	}
	if !confirmed || calls != 0 || !strings.Contains(output.String(), "Canceled") {
		t.Fatalf("denied action state: confirmed=%v calls=%d output=%q", confirmed, calls, output.String())
	}

	output.Reset()
	err = agent.Ask(context.Background(), "restart the media stack", func(string) bool { return true }, &output)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 || !strings.Contains(output.String(), "Stack restart media completed") {
		t.Fatalf("confirmed action state: calls=%d output=%q", calls, output.String())
	}
}
