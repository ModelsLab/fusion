package runner

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/ModelsLab/fusion/internal/config"
	"github.com/ModelsLab/fusion/internal/targets"
)

type Request struct {
	Target  config.TargetConfig
	Command string
	WorkDir string
	Env     map[string]string
	Timeout time.Duration
}

type Result struct {
	TargetName    string              `json:"target_name"`
	TargetMode    string              `json:"target_mode"`
	ExecutionMode string              `json:"execution_mode"`
	Command       string              `json:"command"`
	WorkDir       string              `json:"work_dir,omitempty"`
	StartedAt     time.Time           `json:"started_at"`
	FinishedAt    time.Time           `json:"finished_at"`
	DurationMS    int64               `json:"duration_ms"`
	ExitCode      int                 `json:"exit_code"`
	Stdout        string              `json:"stdout"`
	Stderr        string              `json:"stderr"`
	Authoritative bool                `json:"authoritative"`
	Simulated     bool                `json:"simulated"`
	Warnings      []string            `json:"warnings,omitempty"`
	Target        config.TargetConfig `json:"target"`
}

type CopyRequest struct {
	Target    config.TargetConfig
	Source    string
	Dest      string
	Recursive bool
	Timeout   time.Duration
}

type CopyResult struct {
	TargetName    string              `json:"target_name"`
	TargetMode    string              `json:"target_mode"`
	ExecutionMode string              `json:"execution_mode"`
	Source        string              `json:"source"`
	Dest          string              `json:"dest"`
	StartedAt     time.Time           `json:"started_at"`
	FinishedAt    time.Time           `json:"finished_at"`
	DurationMS    int64               `json:"duration_ms"`
	Warnings      []string            `json:"warnings,omitempty"`
	Target        config.TargetConfig `json:"target"`
}

func Execute(req Request) (Result, error) {
	if strings.TrimSpace(req.Command) == "" {
		return Result{}, fmt.Errorf("command is required")
	}

	timeout := req.Timeout
	if timeout <= 0 {
		timeout = 30 * time.Minute
	}

	started := time.Now().UTC()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	command, err := buildCommand(ctx, req.Target, req.Command, req.WorkDir, req.Env)
	if err != nil {
		return Result{}, err
	}

	stdout := &cappedBuffer{maxBytes: maxCommandOutputBytes}
	stderr := &cappedBuffer{maxBytes: maxCommandOutputBytes}
	command.Stdout = stdout
	command.Stderr = stderr

	runErr := command.Run()
	exitCode := 0
	if runErr != nil {
		exitCode = extractExitCode(runErr)
	}

	finished := time.Now().UTC()
	result := Result{
		TargetName:    req.Target.Name,
		TargetMode:    req.Target.Mode,
		ExecutionMode: executionMode(req.Target),
		Command:       req.Command,
		WorkDir:       strings.TrimSpace(req.WorkDir),
		StartedAt:     started,
		FinishedAt:    finished,
		DurationMS:    finished.Sub(started).Milliseconds(),
		ExitCode:      exitCode,
		Stdout:        strings.TrimSpace(stdout.String()),
		Stderr:        strings.TrimSpace(stderr.String()),
		Target:        req.Target,
		Authoritative: targets.Normalize(req.Target.Mode) != targets.ModeSim,
		Simulated:     targets.Normalize(req.Target.Mode) == targets.ModeSim,
		Warnings:      targets.Warnings(req.Target),
	}

	if runErr != nil {
		return result, fmt.Errorf("execute command: %w", runErr)
	}

	return result, nil
}

func Copy(req CopyRequest) (CopyResult, error) {
	if strings.TrimSpace(req.Source) == "" || strings.TrimSpace(req.Dest) == "" {
		return CopyResult{}, fmt.Errorf("source and destination are required")
	}

	timeout := req.Timeout
	if timeout <= 0 {
		timeout = 15 * time.Minute
	}

	started := time.Now().UTC()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	if targets.Normalize(req.Target.Mode) == targets.ModeSSH || (targets.Normalize(req.Target.Mode) == targets.ModeSim && req.Target.Host != "") {
		if err := scpCopy(ctx, req); err != nil {
			return CopyResult{}, err
		}
	} else {
		if err := localCopy(req.Source, req.Dest, req.Recursive); err != nil {
			return CopyResult{}, err
		}
	}

	finished := time.Now().UTC()
	return CopyResult{
		TargetName:    req.Target.Name,
		TargetMode:    req.Target.Mode,
		ExecutionMode: executionMode(req.Target),
		Source:        req.Source,
		Dest:          req.Dest,
		StartedAt:     started,
		FinishedAt:    finished,
		DurationMS:    finished.Sub(started).Milliseconds(),
		Warnings:      targets.Warnings(req.Target),
		Target:        req.Target,
	}, nil
}

func buildCommand(ctx context.Context, target config.TargetConfig, command string, workDir string, env map[string]string) (*exec.Cmd, error) {
	mode := targets.Normalize(target.Mode)
	switch mode {
	case targets.ModeLocal:
		return localCommand(ctx, target, command, workDir, env), nil
	case targets.ModeSSH:
		return sshCommand(ctx, target, command, workDir, env)
	case targets.ModeSim:
		if target.Host != "" {
			return sshCommand(ctx, target, command, workDir, env)
		}
		return localCommand(ctx, target, command, workDir, env), nil
	default:
		return nil, fmt.Errorf("unsupported target mode %q", target.Mode)
	}
}

func localCommand(ctx context.Context, target config.TargetConfig, command string, workDir string, env map[string]string) *exec.Cmd {
	var cmd *exec.Cmd
	shell := strings.TrimSpace(target.Shell)
	if shell != "" {
		cmd = exec.CommandContext(ctx, shell, "-lc", command)
	} else if runtime.GOOS == "windows" {
		cmd = exec.CommandContext(ctx, "cmd", "/C", command)
	} else {
		cmd = exec.CommandContext(ctx, "sh", "-lc", command)
	}
	if strings.TrimSpace(workDir) != "" {
		cmd.Dir = strings.TrimSpace(workDir)
	}
	if len(env) > 0 {
		cmd.Env = mergeEnv(os.Environ(), env)
	}
	return cmd
}

func sshCommand(ctx context.Context, target config.TargetConfig, command string, workDir string, env map[string]string) (*exec.Cmd, error) {
	if strings.TrimSpace(target.Host) == "" {
		return nil, fmt.Errorf("ssh target requires a host")
	}

	destination := target.Host
	if target.User != "" {
		destination = target.User + "@" + destination
	}

	remoteDir := strings.TrimSpace(target.RemoteDir)
	if strings.TrimSpace(workDir) != "" {
		remoteDir = strings.TrimSpace(workDir)
	}

	args := []string{}
	if target.Port > 0 {
		args = append(args, "-p", fmt.Sprintf("%d", target.Port))
	}
	if strings.TrimSpace(target.IdentityFile) != "" {
		args = append(args, "-i", target.IdentityFile)
	}

	envScript, err := shellEnvExports(env)
	if err != nil {
		return nil, err
	}

	remoteScript := envScript + "exec sh -lc " + shellQuote(command)
	if remoteDir != "" {
		remoteScript = envScript +
			"mkdir -p " + shellQuote(remoteDir) +
			" && cd " + shellQuote(remoteDir) +
			" && exec sh -lc " + shellQuote(command)
	}
	args = append(args, destination, remoteScript)
	return exec.CommandContext(ctx, "ssh", args...), nil
}

func executionMode(target config.TargetConfig) string {
	switch targets.Normalize(target.Mode) {
	case targets.ModeSSH:
		return "ssh"
	case targets.ModeSim:
		if target.Host != "" {
			return "ssh-proxy"
		}
		return "local-proxy"
	default:
		return "local"
	}
}

func extractExitCode(err error) int {
	var exitErr *exec.ExitError
	if !As(err, &exitErr) {
		return 1
	}

	if status, ok := exitErr.Sys().(syscall.WaitStatus); ok {
		return status.ExitStatus()
	}

	return 1
}

// As allows runner to avoid importing errors in multiple helpers.
func As(err error, target any) bool {
	return errorsAs(err, target)
}

func mergeEnv(base []string, overrides map[string]string) []string {
	if len(overrides) == 0 {
		return base
	}
	merged := append([]string{}, base...)
	indexByKey := map[string]int{}
	for i, entry := range merged {
		if key, _, ok := strings.Cut(entry, "="); ok {
			indexByKey[key] = i
		}
	}
	for key, value := range overrides {
		entry := key + "=" + value
		if index, ok := indexByKey[key]; ok {
			merged[index] = entry
			continue
		}
		merged = append(merged, entry)
	}
	return merged
}

func shellEnvExports(env map[string]string) (string, error) {
	if len(env) == 0 {
		return "", nil
	}
	keys := make([]string, 0, len(env))
	for key := range env {
		if !validEnvKey(key) {
			return "", fmt.Errorf("invalid environment variable name %q", key)
		}
		keys = append(keys, key)
	}
	sort.Strings(keys)

	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, key+"="+shellQuote(env[key]))
	}
	return "export " + strings.Join(parts, " ") + "; ", nil
}

func validEnvKey(key string) bool {
	if key == "" {
		return false
	}
	for i, r := range key {
		if i == 0 {
			if (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') || r == '_' {
				continue
			}
			return false
		}
		if (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '_' {
			continue
		}
		return false
	}
	return true
}

const maxCommandOutputBytes = 4 << 20

type cappedBuffer struct {
	buffer    bytes.Buffer
	maxBytes  int
	truncated bool
}

func (b *cappedBuffer) Write(p []byte) (int, error) {
	if b.maxBytes <= 0 {
		b.truncated = true
		return len(p), nil
	}

	remaining := b.maxBytes - b.buffer.Len()
	if remaining <= 0 {
		b.truncated = true
		return len(p), nil
	}
	if len(p) > remaining {
		if _, err := b.buffer.Write(p[:remaining]); err != nil {
			return 0, err
		}
		b.truncated = true
		return len(p), nil
	}
	return b.buffer.Write(p)
}

func (b *cappedBuffer) String() string {
	value := b.buffer.String()
	if !b.truncated {
		return value
	}
	if value != "" && !strings.HasSuffix(value, "\n") {
		value += "\n"
	}
	return value + "...[truncated]..."
}
