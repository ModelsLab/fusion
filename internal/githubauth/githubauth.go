package githubauth

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	Name               = "GitHub"
	DefaultAPIBaseURL  = "https://api.github.com"
	TokenEnvVar        = "GITHUB_TOKEN"
	LegacyTokenEnvVar  = "GH_TOKEN"
	WhoAmIPath         = "/user"
	defaultHTTPTimeout = 30 * time.Second
)

func TokenFromEnv() string {
	return firstNonEmpty(
		os.Getenv(TokenEnvVar),
		os.Getenv(LegacyTokenEnvVar),
	)
}

func APIBaseURL() string {
	return firstNonEmpty(
		os.Getenv("FUSION_GITHUB_API_URL"),
		DefaultAPIBaseURL,
	)
}

func ShellEnv(token string) map[string]string {
	token = strings.TrimSpace(token)
	if token == "" {
		return nil
	}
	return map[string]string{
		TokenEnvVar:       token,
		LegacyTokenEnvVar: token,
	}
}

func WhoAmI(ctx context.Context, token string) (map[string]any, error) {
	token = strings.TrimSpace(token)
	if token == "" {
		return nil, fmt.Errorf("github token is required")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if _, hasDeadline := ctx.Deadline(); !hasDeadline {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultHTTPTimeout)
		defer cancel()
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(APIBaseURL(), "/")+WhoAmIPath, nil)
	if err != nil {
		return nil, fmt.Errorf("build GitHub whoami request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Accept", "application/vnd.github+json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("call GitHub whoami API: %w", err)
	}
	defer resp.Body.Close()

	var payload map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("decode GitHub whoami response: %w", err)
	}
	if resp.StatusCode >= 400 {
		if message := extractWhoAmIError(payload); message != "" {
			return nil, fmt.Errorf("github whoami error: %s", message)
		}
		return nil, fmt.Errorf("github whoami error: status %s", resp.Status)
	}
	return payload, nil
}

func extractWhoAmIError(payload map[string]any) string {
	for _, key := range []string{"message", "error"} {
		if value, ok := payload[key].(string); ok && strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			return value
		}
	}
	return ""
}
