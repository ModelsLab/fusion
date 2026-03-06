package runner

import (
	"context"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/ModelsLab/fusion/internal/config"
)

func TestExecuteLocalTarget(t *testing.T) {
	result, err := Execute(Request{
		Target: config.TargetConfig{
			Name: "local",
			Mode: "local",
		},
		Command: "printf hello",
		Timeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}

	if result.Stdout != "hello" {
		t.Fatalf("expected stdout hello, got %q", result.Stdout)
	}
	if !result.Authoritative {
		t.Fatal("expected local run to be authoritative")
	}
	if result.Simulated {
		t.Fatal("expected local run to not be simulated")
	}
	if result.ExecutionMode != "local" {
		t.Fatalf("expected execution mode local, got %q", result.ExecutionMode)
	}
}

func TestExecuteSimTargetMarksResultNonAuthoritative(t *testing.T) {
	result, err := Execute(Request{
		Target: config.TargetConfig{
			Name:     "sim-h100-on-4090",
			Mode:     "sim",
			GPU:      "h100",
			ProxyGPU: "rtx4090",
		},
		Command: "printf ok",
		Timeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}

	if result.Stdout != "ok" {
		t.Fatalf("expected stdout ok, got %q", result.Stdout)
	}
	if result.Authoritative {
		t.Fatal("expected sim run to be non-authoritative")
	}
	if !result.Simulated {
		t.Fatal("expected sim run to be marked simulated")
	}
	if len(result.Warnings) == 0 {
		t.Fatal("expected sim warnings")
	}
}

func TestBuildCommandForSSHIncludesDestinationAndRemoteDir(t *testing.T) {
	command, err := buildCommand(context.Background(), config.TargetConfig{
		Name:         "lab-4090",
		Mode:         "ssh",
		Host:         "example.com",
		User:         "ubuntu",
		Port:         2222,
		IdentityFile: "/tmp/id",
		RemoteDir:    "/opt/fusion work",
	}, "printf hi", "", nil)
	if err != nil {
		t.Fatalf("buildCommand() error = %v", err)
	}

	if command.Path == "" || !strings.Contains(command.Path, "ssh") {
		t.Fatalf("expected ssh command path, got %q", command.Path)
	}

	args := strings.Join(command.Args, " ")
	if !strings.Contains(args, "ubuntu@example.com") {
		t.Fatalf("expected ssh destination in args, got %q", args)
	}
	if !strings.Contains(args, `mkdir -p "$1" && cd "$1" && exec sh -lc "$2"`) || !strings.Contains(args, "/opt/fusion work") {
		t.Fatalf("expected remote dir bootstrap with positional args, got %q", args)
	}
	if strings.Contains(args, "printf hi &&") {
		t.Fatalf("expected raw command to be passed as an argument, got %q", args)
	}
}

func TestBuildCommandForSSHIncludesEnvironmentExports(t *testing.T) {
	command, err := buildCommand(context.Background(), config.TargetConfig{
		Name: "lab-4090",
		Mode: "ssh",
		Host: "example.com",
	}, "printf %s \"$HF_TOKEN\"", "", map[string]string{
		"HF_TOKEN": "hf-secret",
	})
	if err != nil {
		t.Fatalf("buildCommand() error = %v", err)
	}

	args := strings.Join(command.Args, " ")
	if !strings.Contains(args, "export HF_TOKEN='hf-secret';") {
		t.Fatalf("expected ssh args to export HF_TOKEN, got %q", args)
	}
}

func TestLocalCommandUsesWindowsShellOnWindows(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("windows-specific shell behavior")
	}

	command := localCommand(context.Background(), config.TargetConfig{Name: "local", Mode: "local"}, "echo hi", "", nil)
	if len(command.Args) < 2 || command.Args[1] != "/C" {
		t.Fatalf("expected cmd /C on windows, got %v", command.Args)
	}
}

func TestLocalCommandAppliesWorkingDirectory(t *testing.T) {
	command := localCommand(context.Background(), config.TargetConfig{Name: "local", Mode: "local"}, "printf hi", t.TempDir(), nil)
	if command.Dir == "" {
		t.Fatal("expected working directory to be set")
	}
}

func TestExecuteLocalTargetAppliesEnvironmentOverrides(t *testing.T) {
	result, err := Execute(Request{
		Target: config.TargetConfig{
			Name: "local",
			Mode: "local",
		},
		Command: "printf %s \"$HF_TOKEN\"",
		Env: map[string]string{
			"HF_TOKEN": "hf-test-token",
		},
		Timeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if result.Stdout != "hf-test-token" {
		t.Fatalf("expected env override in stdout, got %q", result.Stdout)
	}
}
