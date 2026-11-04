package assistant

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// CoreAPI calls the local M3TAL Core HTTP API.
type CoreAPI struct {
	BaseURL string
	Token   string
	Client  *http.Client
}

// NewCoreAPI creates a client for M3TAL Core. An empty base URL uses localhost:5050.
func NewCoreAPI(baseURL, token string) *CoreAPI {
	if strings.TrimSpace(baseURL) == "" {
		baseURL = "http://localhost:5050"
	}
	return &CoreAPI{BaseURL: strings.TrimRight(baseURL, "/"), Token: token, Client: &http.Client{Timeout: 5 * time.Second}}
}

func (api *CoreAPI) request(ctx context.Context, method, path string) (json.RawMessage, error) {
	parsedURL, err := validateAPIURL(api.BaseURL)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, method, api.BaseURL+path, bytes.NewReader(nil))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if api.Token != "" {
		req.Header.Set("X-API-Token", api.Token)
	}
	response, err := api.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("M3TAL Core request to %s failed: %s", sanitizedURL(parsedURL), api.sanitize(err.Error()))
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 8*1024*1024))
	if err != nil {
		return nil, fmt.Errorf("read M3TAL Core response: %w", err)
	}
	var envelope struct {
		Status string          `json:"status"`
		Data   json.RawMessage `json:"data"`
		Error  *struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return nil, fmt.Errorf("M3TAL Core returned HTTP %d with invalid JSON", response.StatusCode)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 || envelope.Status == "error" || envelope.Error != nil {
		if envelope.Error != nil {
			return nil, fmt.Errorf("M3TAL Core error (%s): %s", api.sanitize(envelope.Error.Code), api.sanitize(envelope.Error.Message))
		}
		return nil, fmt.Errorf("M3TAL Core returned HTTP %d", response.StatusCode)
	}
	if len(envelope.Data) > 0 {
		return envelope.Data, nil
	}
	return json.RawMessage(body), nil
}

func validateAPIURL(raw string) (*url.URL, error) {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Hostname() == "" {
		return nil, fmt.Errorf("M3TAL_API_URL must be an absolute HTTP or HTTPS URL")
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, fmt.Errorf("M3TAL_API_URL cannot contain credentials, query parameters, or a fragment; pass the API token with M3TAL_API_TOKEN")
	}
	switch parsed.Scheme {
	case "https":
	case "http":
		host := strings.TrimSuffix(strings.ToLower(parsed.Hostname()), ".")
		ip := net.ParseIP(host)
		if host != "localhost" && (ip == nil || !ip.IsLoopback()) {
			return nil, fmt.Errorf("refusing non-TLS remote M3TAL_API_URL; use HTTPS (plain HTTP is allowed only for localhost or loopback)")
		}
	default:
		return nil, fmt.Errorf("M3TAL_API_URL scheme must be HTTP or HTTPS")
	}
	return parsed, nil
}

func sanitizedURL(parsed *url.URL) string {
	copyURL := *parsed
	copyURL.User = nil
	copyURL.RawQuery = ""
	copyURL.Fragment = ""
	return copyURL.String()
}

func (api *CoreAPI) sanitize(message string) string {
	if api.Token != "" {
		message = strings.ReplaceAll(message, api.Token, "[REDACTED]")
	}
	return message
}

func (api *CoreAPI) get(ctx context.Context, path string) (json.RawMessage, error) {
	return api.request(ctx, http.MethodGet, path)
}

func (api *CoreAPI) post(ctx context.Context, path string) (json.RawMessage, error) {
	return api.request(ctx, http.MethodPost, path)
}

func (api *CoreAPI) stackAction(ctx context.Context, name, action string) (json.RawMessage, error) {
	return api.post(ctx, "/api/v2/stacks/"+url.PathEscape(name)+"/"+action)
}
