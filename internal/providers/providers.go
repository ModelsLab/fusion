package providers

import "strings"

type Provider struct {
	ID          string
	Name        string
	BaseURL     string
	TokenEnvVar string
	DocsURL     string
	Notes       string
}

var registry = []Provider{
	{
		ID:          "openai",
		Name:        "OpenAI",
		BaseURL:     "https://api.openai.com/v1",
		TokenEnvVar: "OPENAI_API_KEY",
		DocsURL:     "https://platform.openai.com/docs/quickstart",
		Notes:       "Good default for planning, synthesis, and code generation.",
	},
	{
		ID:          "anthropic",
		Name:        "Anthropic",
		BaseURL:     "https://api.anthropic.com",
		TokenEnvVar: "ANTHROPIC_API_KEY",
		DocsURL:     "https://docs.anthropic.com",
		Notes:       "Strong long-context reasoning for strategy and review loops.",
	},
	{
		ID:          "openrouter",
		Name:        "OpenRouter",
		BaseURL:     "https://openrouter.ai/api/v1",
		TokenEnvVar: "OPENROUTER_API_KEY",
		DocsURL:     "https://openrouter.ai/docs",
		Notes:       "Useful when you want a single provider surface over many models.",
	},
	{
		ID:          "google",
		Name:        "Google Gemini",
		BaseURL:     "https://generativelanguage.googleapis.com",
		TokenEnvVar: "GEMINI_API_KEY",
		DocsURL:     "https://ai.google.dev/gemini-api/docs",
		Notes:       "Good when you need additional multimodal or long-context coverage.",
	},
	{
		ID:          "groq",
		Name:        "Groq",
		BaseURL:     "https://api.groq.com/openai/v1",
		TokenEnvVar: "GROQ_API_KEY",
		DocsURL:     "https://console.groq.com/docs",
		Notes:       "Fast hosted inference for planning and light agent loops.",
	},
}

func All() []Provider {
	out := make([]Provider, len(registry))
	copy(out, registry)
	return out
}

func Lookup(id string) (Provider, bool) {
	candidate := normalize(id)
	for _, provider := range registry {
		if normalize(provider.ID) == candidate {
			return provider, true
		}
	}
	return Provider{}, false
}

func normalize(value string) string {
	return strings.TrimSpace(strings.ToLower(value))
}
