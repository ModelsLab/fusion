package artifacts

import "testing"

func TestParseMetricsParsesStandaloneJSON(t *testing.T) {
	metrics := ParseMetrics(`{"rtf":0.5,"x_real_time":2.0}`)
	if metrics["rtf"] != 0.5 {
		t.Fatalf("expected rtf metric, got %#v", metrics)
	}
	if metrics["x_real_time"] != 2.0 {
		t.Fatalf("expected x_real_time metric, got %#v", metrics)
	}
}

func TestParseMetricsParsesTrailingJSONLine(t *testing.T) {
	metrics := ParseMetrics("loaded PerthNet\n{\"gen_s\":2.5,\"rtf\":0.75}")
	if metrics["gen_s"] != 2.5 {
		t.Fatalf("expected gen_s metric, got %#v", metrics)
	}
	if metrics["rtf"] != 0.75 {
		t.Fatalf("expected rtf metric, got %#v", metrics)
	}
}

func TestParseMetricsFallsBackToKeyValuePairs(t *testing.T) {
	metrics := ParseMetrics("latency_ms=12.5\nthroughput: 42")
	if metrics["latency_ms"] != 12.5 {
		t.Fatalf("expected latency_ms metric, got %#v", metrics)
	}
	if metrics["throughput"] != 42 {
		t.Fatalf("expected throughput metric, got %#v", metrics)
	}
}
