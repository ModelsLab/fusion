package artifacts

import "testing"

func TestParseMetricsJSON(t *testing.T) {
	metrics := ParseMetrics(`{"tokens_per_sec": 123.5, "latency_ms": 9}`)
	if metrics["tokens_per_sec"] != 123.5 {
		t.Fatalf("expected tokens_per_sec to be parsed")
	}
	if metrics["latency_ms"] != 9 {
		t.Fatalf("expected latency_ms to be parsed")
	}
}

func TestParseMetricsKeyValue(t *testing.T) {
	metrics := ParseMetrics("tokens_per_sec=123.5\nlatency_ms=9")
	if metrics["tokens_per_sec"] != 123.5 {
		t.Fatalf("expected tokens_per_sec to be parsed")
	}
	if metrics["latency_ms"] != 9 {
		t.Fatalf("expected latency_ms to be parsed")
	}
}
