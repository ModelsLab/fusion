package cli

import (
	"bytes"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

func TestLowerIsBetter(t *testing.T) {
	if !lowerIsBetter("latency_ms") {
		t.Fatal("expected latency_ms to be lower-is-better")
	}
	if !lowerIsBetter("kernel_time") {
		t.Fatal("expected kernel_time to be lower-is-better")
	}
	if !lowerIsBetter("gen_s") {
		t.Fatal("expected gen_s to be lower-is-better")
	}
	if !lowerIsBetter("rtf") {
		t.Fatal("expected rtf to be lower-is-better")
	}
	if lowerIsBetter("tokens_per_sec") {
		t.Fatal("expected tokens_per_sec to be higher-is-better")
	}
	if lowerIsBetter("x_real_time") {
		t.Fatal("expected x_real_time to be higher-is-better")
	}
}

func TestCommonMetricKeysReturnsSortedIntersection(t *testing.T) {
	keys := commonMetricKeys(
		map[string]float64{"z": 1, "tokens_per_sec": 2, "latency_ms": 3},
		map[string]float64{"latency_ms": 1, "tokens_per_sec": 1, "x": 1},
	)

	expected := []string{"latency_ms", "tokens_per_sec"}
	if len(keys) != len(expected) {
		t.Fatalf("expected %d keys, got %d", len(expected), len(keys))
	}
	for i := range expected {
		if keys[i] != expected[i] {
			t.Fatalf("expected key %q at index %d, got %q", expected[i], i, keys[i])
		}
	}
}

func TestPrintComparisonLineFormatsSpeedup(t *testing.T) {
	buffer := &bytes.Buffer{}
	cmd := &cobra.Command{}
	cmd.SetOut(buffer)

	printComparisonLine(cmd, "tokens_per_sec", 100, 125)
	output := buffer.String()

	if !strings.Contains(output, "speedup 1.25x") {
		t.Fatalf("expected speedup text, got %q", output)
	}
	if !strings.Contains(output, "higher is better") {
		t.Fatalf("expected direction hint, got %q", output)
	}
}

func TestPrintComparisonLineHandlesLatency(t *testing.T) {
	buffer := &bytes.Buffer{}
	cmd := &cobra.Command{}
	cmd.SetOut(buffer)

	printComparisonLine(cmd, "latency_ms", 10, 8)
	output := buffer.String()

	if !strings.Contains(output, "lower is better") {
		t.Fatalf("expected lower-is-better hint, got %q", output)
	}
	if !strings.Contains(output, "speedup 1.25x") {
		t.Fatalf("expected latency speedup, got %q", output)
	}
}

func TestPrintComparisonLineTreatsXRealTimeAsHigherIsBetter(t *testing.T) {
	buffer := &bytes.Buffer{}
	cmd := &cobra.Command{}
	cmd.SetOut(buffer)

	printComparisonLine(cmd, "x_real_time", 1.16, 1.24)
	output := buffer.String()

	if !strings.Contains(output, "higher is better") {
		t.Fatalf("expected higher-is-better hint, got %q", output)
	}
	if !strings.Contains(output, "speedup 1.07x") {
		t.Fatalf("expected higher-is-better speedup, got %q", output)
	}
}

func TestPrintComparisonLineTreatsAudioDurationAsContextual(t *testing.T) {
	buffer := &bytes.Buffer{}
	cmd := &cobra.Command{}
	cmd.SetOut(buffer)

	printComparisonLine(cmd, "audio_s", 3.4, 3.4)
	output := buffer.String()

	if !strings.Contains(output, "contextual metric") {
		t.Fatalf("expected contextual metric hint, got %q", output)
	}
}
