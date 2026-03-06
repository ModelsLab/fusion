package cli

import (
	"fmt"
	"strings"

	"github.com/ModelsLab/fusion/internal/config"
	"github.com/ModelsLab/fusion/internal/githubauth"
	"github.com/ModelsLab/fusion/internal/huggingface"
	"github.com/ModelsLab/fusion/internal/kb"
)

func maskToken(token string) string {
	if token == "" {
		return ""
	}
	if len(token) <= 8 {
		return "********"
	}
	return token[:4] + strings.Repeat("*", len(token)-8) + token[len(token)-4:]
}

func joinOrFallback(values []string, fallback string) string {
	if len(values) == 0 {
		return fallback
	}
	return strings.Join(values, ", ")
}

func formatSourceList(sources []kb.Source) string {
	lines := make([]string, 0, len(sources))
	for _, source := range sources {
		lines = append(lines, fmt.Sprintf("%s (%s)", source.Title, source.URL))
	}
	return strings.Join(lines, "; ")
}

func resolveTarget(runtimeState *runtimeState, name string) (config.TargetConfig, string, error) {
	cfg, err := runtimeState.Config.Load()
	if err != nil {
		return config.TargetConfig{}, "", err
	}

	name = strings.TrimSpace(name)
	if name == "" {
		name = strings.TrimSpace(cfg.DefaultTarget)
	}
	if name == "" {
		return config.TargetConfig{
			Name: "local",
			Mode: "local",
		}, "local", nil
	}

	target, ok := cfg.Targets[name]
	if !ok {
		return config.TargetConfig{}, "", fmt.Errorf("target %q is not configured", name)
	}

	return target, name, nil
}

func resolveHuggingFaceToken(runtimeState *runtimeState) (string, error) {
	cfg, err := runtimeState.Config.Load()
	if err != nil {
		return "", err
	}

	token := strings.TrimSpace(cfg.HuggingFace.Token)
	if token == "" {
		token = huggingface.TokenFromEnv()
	}
	return token, nil
}

func resolveGitHubToken(runtimeState *runtimeState) (string, error) {
	cfg, err := runtimeState.Config.Load()
	if err != nil {
		return "", err
	}

	token := strings.TrimSpace(cfg.GitHub.Token)
	if token == "" {
		token = githubauth.TokenFromEnv()
	}
	return token, nil
}

func runtimeShellEnv(runtimeState *runtimeState) (map[string]string, error) {
	env := map[string]string{}

	hfToken, err := resolveHuggingFaceToken(runtimeState)
	if err != nil {
		return nil, err
	}
	for key, value := range huggingface.ShellEnv(hfToken) {
		env[key] = value
	}
	ghToken, err := resolveGitHubToken(runtimeState)
	if err != nil {
		return nil, err
	}
	for key, value := range githubauth.ShellEnv(ghToken) {
		env[key] = value
	}

	if len(env) == 0 {
		return nil, nil
	}
	return env, nil
}

func shellQuote(value string) string {
	if value == "" {
		return "''"
	}
	return "'" + strings.ReplaceAll(value, "'", `'"'"'`) + "'"
}
