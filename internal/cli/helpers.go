package cli

import (
	"fmt"
	"strings"

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
