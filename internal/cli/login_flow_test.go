package cli

import (
	"net/http"
	"net/url"
	"strings"
	"testing"
)

func TestBuildModelsLabLoginURL(t *testing.T) {
	loginURL, err := buildModelsLabLoginURL("https://modelslab.com", "http://127.0.0.1:4242/callback", "state-1", "openai-gpt-5.4-pro")
	if err != nil {
		t.Fatalf("buildModelsLabLoginURL() error = %v", err)
	}

	parsed, err := url.Parse(loginURL)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if parsed.Path != "/auth/fusion-cli/browser-login" {
		t.Fatalf("expected login path, got %q", parsed.Path)
	}
	if parsed.Query().Get("callback_url") != "http://127.0.0.1:4242/callback" {
		t.Fatalf("unexpected callback_url %q", parsed.Query().Get("callback_url"))
	}
	if parsed.Query().Get("state") != "state-1" {
		t.Fatalf("unexpected state %q", parsed.Query().Get("state"))
	}
}

func TestParseModelsLabLoginPayloadFromJSON(t *testing.T) {
	req, err := http.NewRequest(http.MethodPost, "http://127.0.0.1:4242/callback", strings.NewReader(`{"api_key":"ml-123","model_id":"openai-gpt-5.4-pro","state":"abc"}`))
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	payload, err := parseModelsLabLoginPayload(req)
	if err != nil {
		t.Fatalf("parseModelsLabLoginPayload() error = %v", err)
	}
	if payload.APIKey != "ml-123" || payload.State != "abc" {
		t.Fatalf("unexpected payload %#v", payload)
	}
}

func TestParseModelsLabLoginPayloadFromForm(t *testing.T) {
	form := url.Values{}
	form.Set("api_key", "ml-123")
	form.Set("model_id", "openai-gpt-5.4-pro")
	form.Set("state", "abc")

	req, err := http.NewRequest(http.MethodPost, "http://127.0.0.1:4242/callback", strings.NewReader(form.Encode()))
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	payload, err := parseModelsLabLoginPayload(req)
	if err != nil {
		t.Fatalf("parseModelsLabLoginPayload() error = %v", err)
	}
	if payload.Model != "openai-gpt-5.4-pro" {
		t.Fatalf("unexpected model %q", payload.Model)
	}
}

func TestParseModelsLabLoginPayloadRequiresState(t *testing.T) {
	req, err := http.NewRequest(http.MethodGet, "http://127.0.0.1:4242/callback?api_key=ml-123", nil)
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}

	if _, err := parseModelsLabLoginPayload(req); err == nil {
		t.Fatal("expected missing state error")
	}
}
