package artifacts

import (
	"encoding/json"
	"strconv"
	"strings"
)

func ParseMetrics(text string) map[string]float64 {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	if metrics := parseJSONMetrics(text); len(metrics) > 0 {
		return metrics
	}

	return parseKeyValueMetrics(text)
}

func parseJSONMetrics(text string) map[string]float64 {
	var payload map[string]any
	if err := json.Unmarshal([]byte(text), &payload); err != nil {
		return nil
	}

	metrics := map[string]float64{}
	for key, value := range payload {
		switch typed := value.(type) {
		case float64:
			metrics[key] = typed
		case int:
			metrics[key] = float64(typed)
		}
	}
	return metrics
}

func parseKeyValueMetrics(text string) map[string]float64 {
	metrics := map[string]float64{}
	lines := strings.Split(text, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		separator := "="
		if !strings.Contains(line, separator) {
			separator = ":"
		}
		parts := strings.SplitN(line, separator, 2)
		if len(parts) != 2 {
			continue
		}

		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])
		number, err := strconv.ParseFloat(value, 64)
		if err != nil {
			continue
		}
		metrics[key] = number
	}

	if len(metrics) == 0 {
		return nil
	}
	return metrics
}
