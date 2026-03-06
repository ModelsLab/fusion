package githubauth

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestTokenFromEnvPrefersGitHubToken(t *testing.T) {
	t.Setenv(TokenEnvVar, "gh-primary")
	t.Setenv(LegacyTokenEnvVar, "gh-legacy")

	if got := TokenFromEnv(); got != "gh-primary" {
		t.Fatalf("expected GITHUB_TOKEN to win, got %q", got)
	}
}

func TestWhoAmIUsesConfiguredAPIBaseURL(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != WhoAmIPath {
			t.Fatalf("unexpected user path %q", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer gh-test" {
			t.Fatalf("unexpected authorization header %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"login":"tester","id":1}`))
	}))
	defer server.Close()

	t.Setenv("FUSION_GITHUB_API_URL", server.URL)
	payload, err := WhoAmI(context.Background(), "gh-test")
	if err != nil {
		t.Fatalf("WhoAmI() error = %v", err)
	}
	if payload["login"] != "tester" {
		t.Fatalf("expected GitHub user payload, got %#v", payload)
	}
}
