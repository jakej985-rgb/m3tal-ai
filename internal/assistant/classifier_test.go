package assistant

import "testing"

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

func TestRequireNameSupportsStackBeforeOrAfterName(t *testing.T) {
	for query, expected := range map[string]string{
		"restart stack media":     "media",
		"restart the media stack": "media",
	} {
		got, err := requireName(query, "stack")
		if err != nil || got != expected {
			t.Errorf("requireName(%q) = %q, %v; want %q", query, got, err, expected)
		}
	}
}
