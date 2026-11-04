package assistant

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (roundTrip roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return roundTrip(request)
}

func mockResponse(request *http.Request, status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
		Request:    request,
	}
}

func TestCoreAPIUnwrapsDataAndSendsToken(t *testing.T) {
	api := NewCoreAPI("https://m3tal.test", "test-token")
	api.Client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.Method != http.MethodGet || request.URL.Path != "/api/v2/system/stats" {
			t.Errorf("request = %s %s, want GET /api/v2/system/stats", request.Method, request.URL.Path)
		}
		if request.Header.Get("X-API-Token") != "test-token" {
			t.Errorf("API token header = %q, want test-token", request.Header.Get("X-API-Token"))
		}
		return mockResponse(request, http.StatusOK, `{"status":"success","data":{"cpu_usage":12}}`), nil
	})}
	data, err := api.get(context.Background(), "/api/v2/system/stats")
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `{"cpu_usage":12}` {
		t.Fatalf("unwrapped data = %s, want {\"cpu_usage\":12}", data)
	}
}

func TestAPIURLRequiresTLSForRemoteHosts(t *testing.T) {
	for _, test := range []struct {
		raw       string
		wantError bool
	}{
		{"https://core.example:5050", false},
		{"http://localhost:5050", false},
		{"http://127.0.0.1:5050", false},
		{"http://[::1]:5050", false},
		{"http://core.example:5050", true},
		{"http://192.168.1.20:5050", true},
		{"https://user:pass@core.example", true},
		{"https://core.example/?token=secret", true},
	} {
		_, err := validateAPIURL(test.raw)
		if (err != nil) != test.wantError {
			t.Errorf("validateAPIURL(%q) error = %v, wantError=%v", test.raw, err, test.wantError)
		}
	}
}

func TestCoreAPIErrorNeverIncludesToken(t *testing.T) {
	const token = "private-token-value"
	api := NewCoreAPI("https://m3tal.test", token)
	api.Client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		return mockResponse(request, http.StatusUnauthorized, `{"status":"error","error":{"code":"AUTH","message":"bad token private-token-value"}}`), nil
	})}
	_, err := api.get(context.Background(), "/api/v2/system/stats")
	if err == nil {
		t.Fatal("expected API error")
	}
	if strings.Contains(err.Error(), token) {
		t.Fatalf("error exposed token: %v", err)
	}
}
