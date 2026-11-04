package assistant

import (
	"strings"
	"testing"
)

func TestClassifierRoutesRepresentativeRequests(t *testing.T) {
	model, err := newClassifier()
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		query string
		want  intent
	}{
		{"show cpu and memory usage", intentStats},
		{"check component health status", intentHealth},
		{"list running containers", intentContainers},
		{"show logs for jellyfin", intentLogs},
		{"how do I install m3tal", intentDocs},
		{"restart the media stack", intentRestart},
		{"start stack downloads", intentStartStack},
		{"stop stack traefik", intentStopStack},
		{"restart the jellyfin container", intentRestartContainer},
		{"stop service docker", intentStopService},
	}
	for _, test := range cases {
		got, confidence, err := model.classify(test.query)
		if err != nil {
			t.Errorf("classify %q: %v", test.query, err)
			continue
		}
		if got != test.want {
			t.Errorf("classify %q = %q (%.2f confidence), want %q", test.query, got, confidence, test.want)
		}
	}
}

func TestHeldOutIntentSetAccuracy(t *testing.T) {
	model, err := newClassifier()
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		phrase string
		want   intent
	}{
		{"get host cpu stats", intentStats}, {"show memory usage", intentStats}, {"check disk metrics", intentStats}, {"show uptime", intentStats}, {"get hardware telemetry", intentStats}, {"what is cpu load", intentStats}, {"show system metrics", intentStats},
		{"is the daemon healthy", intentHealth}, {"run system health", intentHealth}, {"check unhealthy components", intentHealth}, {"show doctor results", intentHealth}, {"health of the host", intentHealth}, {"are components healthy", intentHealth}, {"system health report", intentHealth},
		{"show running containers", intentContainers}, {"list docker containers", intentContainers}, {"which containers are active", intentContainers}, {"container status list", intentContainers}, {"show all containers", intentContainers}, {"what is running in docker", intentContainers}, {"list container services", intentUnknown},
		{"show logs for jellyfin", intentLogs}, {"read container logs", intentLogs}, {"fetch logs from traefik", intentLogs}, {"view recent logs", intentLogs}, {"why did jellyfin fail show logs", intentLogs}, {"container log output", intentLogs}, {"get service logs", intentLogs},
		{"how to install m3tal", intentDocs}, {"explain M3TAL configuration", intentDocs}, {"find documentation", intentDocs}, {"configure the proxy", intentDocs}, {"explain the Hub", intentDocs}, {"help with installation", intentDocs}, {"where are docs", intentDocs},
		{"start stack media", intentStartStack}, {"stop stack media", intentStopStack}, {"restart the traefik stack", intentRestart}, {"start container jellyfin", intentStartContainer}, {"stop the jellyfin container", intentStopContainer}, {"restart container jellyfin", intentRestartContainer}, {"start service docker", intentStartService}, {"stop system service docker", intentStopService}, {"restart the m3tal-api service", intentRestartService},
		{"show stats and health", intentUnknown}, {"restart stack and show logs for media", intentUnknown}, {"tell me something random", intentUnknown}, {"qwerty blargh", intentUnknown}, {"show metrics and containers", intentUnknown}, {"start container and restart stack media", intentUnknown}, {"what is happening", intentUnknown},
	}
	correct := 0
	for _, test := range cases {
		got, _, err := model.classify(test.phrase)
		if err != nil {
			t.Errorf("classify %q: %v", test.phrase, err)
			continue
		}
		if got == test.want {
			correct++
		} else {
			t.Errorf("%q routed to %q, want %q", test.phrase, got, test.want)
		}
	}
	accuracy := float64(correct) / float64(len(cases))
	t.Logf("held-out routing accuracy: %d/%d (%.1f%%)", correct, len(cases), accuracy*100)
	if accuracy < 0.95 {
		t.Fatalf("routing accuracy %.1f%% is below 95%%", accuracy*100)
	}
}

func TestRequireNameSupportsStackBeforeOrAfterName(t *testing.T) {
	for query, expected := range map[string]string{
		"restart stack media":            "media",
		"restart the media stack":        "media",
		"restart the jellyfin container": "jellyfin",
	} {
		marker := "stack"
		if strings.Contains(query, "container") {
			marker = "container"
		}
		got, err := requireName(query, marker)
		if err != nil || got != expected {
			t.Errorf("requireName(%q) = %q, %v; want %q", query, got, err, expected)
		}
	}
}
