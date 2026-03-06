package cli

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/ModelsLab/fusion/internal/artifacts"
	"github.com/ModelsLab/fusion/internal/config"
	"github.com/ModelsLab/fusion/internal/optimize"
	"github.com/ModelsLab/fusion/internal/runner"
	"github.com/ModelsLab/fusion/internal/targets"
)

type workspaceRunRequest struct {
	Target       config.TargetConfig
	WorkspaceDir string
	Command      string
	Timeout      time.Duration
}

func needsRemoteWorkspace(target config.TargetConfig) bool {
	mode := targets.Normalize(target.Mode)
	return mode == targets.ModeSSH || (mode == targets.ModeSim && strings.TrimSpace(target.Host) != "")
}

func executeWorkspaceRun(req workspaceRunRequest) (runner.Result, string, error) {
	target := req.Target
	workspacePath := filepath.Clean(req.WorkspaceDir)

	if needsRemoteWorkspace(target) {
		remoteWorkspace := remoteWorkspacePath(target, workspacePath)
		copyTarget := target
		copyTarget.RemoteDir = ""

		if _, err := runner.Copy(runner.CopyRequest{
			Target:    copyTarget,
			Source:    workspacePath,
			Dest:      remoteWorkspace,
			Recursive: true,
			Timeout:   req.Timeout,
		}); err != nil {
			return runner.Result{}, "", fmt.Errorf("copy workspace to target: %w", err)
		}

		runResult, err := runner.Execute(runner.Request{
			Target:  target,
			Command: req.Command,
			WorkDir: remoteWorkspace,
			Timeout: req.Timeout,
		})
		return runResult, formatDisplayedCommand(remoteWorkspace, req.Command), err
	}

	runResult, err := runner.Execute(runner.Request{
		Target:  target,
		Command: req.Command,
		WorkDir: workspacePath,
		Timeout: req.Timeout,
	})
	return runResult, formatDisplayedCommand(workspacePath, req.Command), err
}

func remoteWorkspacePath(target config.TargetConfig, workspacePath string) string {
	root := strings.TrimSpace(target.RemoteDir)
	if root == "" {
		root = "~/fusion"
	}
	return strings.TrimRight(root, "/") + "/" + filepath.Base(workspacePath)
}

func formatDisplayedCommand(workdir, command string) string {
	workdir = strings.TrimSpace(workdir)
	if workdir == "" {
		return command
	}
	return "cd " + shellQuote(workdir) + " && " + command
}

func saveKernelStageArtifact(name, backend, stage, workspacePath string, target config.TargetConfig, executedCommand string, runResult runner.Result, outputPath string) (string, map[string]float64, error) {
	metrics := artifacts.ParseMetrics(runResult.Stdout)
	store, err := artifacts.NewStore()
	if err != nil {
		return "", nil, err
	}

	artifact := artifacts.KernelRunResult{
		Name:          name,
		Backend:       backend,
		Stage:         stage,
		Workspace:     workspacePath,
		TargetName:    target.Name,
		TargetMode:    target.Mode,
		Command:       executedCommand,
		Metrics:       metrics,
		StartedAt:     runResult.StartedAt,
		FinishedAt:    runResult.FinishedAt,
		DurationMS:    runResult.DurationMS,
		ExitCode:      runResult.ExitCode,
		Stdout:        runResult.Stdout,
		Stderr:        runResult.Stderr,
		Authoritative: runResult.Authoritative,
		Simulated:     runResult.Simulated,
		Warnings:      runResult.Warnings,
		Run:           runResult,
	}

	path, err := store.SaveKernelRun(artifact, outputPath)
	if err != nil {
		return "", nil, err
	}
	return path, metrics, nil
}

func recordCandidateStage(sessionID, candidateID, backend, workspacePath, stage, artifactPath, command string, exitCode int, metrics map[string]float64) error {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return nil
	}

	session, store, err := loadOptimizationSession(sessionID)
	if err != nil {
		return err
	}

	candidateID = strings.TrimSpace(candidateID)
	if candidateID == "" {
		if candidate, ok := session.CandidateByWorkspace(backend, workspacePath); ok {
			candidateID = candidate.ID
		}
	}
	if candidateID == "" {
		return fmt.Errorf("no candidate was registered for backend %s at %s in session %s", backend, workspacePath, session.ID)
	}

	if err := session.RecordCandidateStage(candidateID, stage, artifactPath, command, exitCode, metrics); err != nil {
		return err
	}
	_, err = store.Save(session)
	return err
}

func candidateFromSession(sessionID, candidateID string) (*optimize.Session, *optimize.Candidate, error) {
	session, _, err := loadOptimizationSession(sessionID)
	if err != nil {
		return nil, nil, err
	}
	candidate, ok := session.CandidateByID(candidateID)
	if !ok {
		return nil, nil, fmt.Errorf("candidate %q not found in session %s", candidateID, session.ID)
	}
	return session, candidate, nil
}

func targetNameFromSession(sessionID, explicitTarget string) string {
	explicitTarget = strings.TrimSpace(explicitTarget)
	if explicitTarget != "" || strings.TrimSpace(sessionID) == "" {
		return explicitTarget
	}
	session, _, err := loadOptimizationSession(sessionID)
	if err != nil {
		return explicitTarget
	}
	return strings.TrimSpace(session.Target)
}
