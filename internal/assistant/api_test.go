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
	api := NewCoreAPI("http://m3tal.test", "test-token")
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
