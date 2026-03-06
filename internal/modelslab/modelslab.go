package modelslab

import (
	"os"
	"strings"
)

const (
	ProviderID        = "modelslab"
	Name              = "ModelsLab"
	DefaultAPIBaseURL = "https://modelslab.com/api/v7/llm"
	DefaultWebBaseURL = "https://modelslab.com"
	TokenEnvVar       = "MODELSLAB_API_KEY"
	ModelEnvVar       = "MODELSLAB_MODEL_ID"
	DefaultModelID    = "openai-gpt-5.4-pro"
	LoginPath         = "/auth/fusion-cli/browser-login"
)

func APIBaseURL() string {
	return firstNonEmpty(
		os.Getenv("FUSION_MODELSLAB_API_URL"),
		DefaultAPIBaseURL,
	)
}

func WebBaseURL() string {
	return firstNonEmpty(
		os.Getenv("FUSION_MODELSLAB_WEB_URL"),
		DefaultWebBaseURL,
	)
}

func NormalizeURL(value string, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		value = fallback
	}
	return strings.TrimRight(value, "/")
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
