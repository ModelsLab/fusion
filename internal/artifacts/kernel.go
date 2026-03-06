package artifacts

import (
	"path/filepath"
	"strings"
	"time"

	"github.com/ModelsLab/fusion/internal/runner"
)

type KernelRunResult struct {
	Version       int                `json:"version"`
	ID            string             `json:"id"`
	Name          string             `json:"name"`
	Backend       string             `json:"backend"`
	Stage         string             `json:"stage"`
	Workspace     string             `json:"workspace"`
	TargetName    string             `json:"target_name"`
	TargetMode    string             `json:"target_mode"`
	Command       string             `json:"command"`
	Metrics       map[string]float64 `json:"metrics,omitempty"`
	StartedAt     time.Time          `json:"started_at"`
	FinishedAt    time.Time          `json:"finished_at"`
	DurationMS    int64              `json:"duration_ms"`
	ExitCode      int                `json:"exit_code"`
	Stdout        string             `json:"stdout,omitempty"`
	Stderr        string             `json:"stderr,omitempty"`
	Authoritative bool               `json:"authoritative"`
	Simulated     bool               `json:"simulated"`
	Warnings      []string           `json:"warnings,omitempty"`
	Run           runner.Result      `json:"run"`
}

func (s *Store) KernelRunPath(stage, id string) string {
	stage = strings.TrimSpace(strings.ToLower(stage))
	if stage == "" {
		stage = "run"
	}
	return filepath.Join(s.root, "kernel-runs", stage, id+".json")
}

func (s *Store) SaveKernelRun(result KernelRunResult, outputPath string) (string, error) {
	if result.ID == "" {
		result.ID = newID(result.Name)
	}
	if result.Version == 0 {
		result.Version = 1
	}
	if outputPath == "" {
		outputPath = s.KernelRunPath(result.Stage, result.ID)
	}
	return outputPath, writeJSON(outputPath, result)
}
