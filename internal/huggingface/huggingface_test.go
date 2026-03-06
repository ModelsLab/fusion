package huggingface

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestTokenFromEnvPrefersHFToken(t *testing.T) {
	t.Setenv(TokenEnvVar, "hf-primary")
	t.Setenv(LegacyTokenEnvVar, "hf-legacy")

	if got := TokenFromEnv(); got != "hf-primary" {
		t.Fatalf("expected HF_TOKEN to win, got %q", got)
	}
}

func TestWhoAmIUsesConfiguredWebBaseURL(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != WhoAmIPath {
			t.Fatalf("unexpected whoami path %q", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer hf-test" {
			t.Fatalf("unexpected authorization header %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"name":"tester","type":"user"}`))
	}))
	defer server.Close()

	t.Setenv("FUSION_HUGGINGFACE_WEB_URL", server.URL)
	payload, err := WhoAmI(context.Background(), "hf-test")
	if err != nil {
		t.Fatalf("WhoAmI() error = %v", err)
	}
	if payload["name"] != "tester" {
		t.Fatalf("expected whoami payload, got %#v", payload)
	}
}
