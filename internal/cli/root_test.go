package cli

import (
	"bytes"
	"strings"
	"testing"
)

func TestRootCommandHelpFlag(t *testing.T) {
	cmd := NewRootCommand()
	stdout := &bytes.Buffer{}
	stderr := &bytes.Buffer{}
	cmd.SetOut(stdout)
	cmd.SetErr(stderr)
	cmd.SetArgs([]string{"-h"})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute() error = %v\nstderr=%s", err, stderr.String())
	}

	output := stdout.String()
	if !strings.Contains(output, "Usage:") {
		t.Fatalf("expected help usage output, got %q", output)
	}
	if !strings.Contains(output, "fusion [flags]") {
		t.Fatalf("expected root usage line, got %q", output)
	}
	if !strings.Contains(output, "hf") {
		t.Fatalf("expected hf command in help output, got %q", output)
	}
	if !strings.Contains(output, "github") {
		t.Fatalf("expected github command in help output, got %q", output)
	}
}
