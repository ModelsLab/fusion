package artifacts

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ModelsLab/fusion/internal/runner"
)

type BenchmarkResult struct {
	Version       int                `json:"version"`
	ID            string             `json:"id"`
	Name          string             `json:"name"`
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

type ProfileResult struct {
	Version       int           `json:"version"`
	ID            string        `json:"id"`
	Name          string        `json:"name"`
	Tool          string        `json:"tool"`
	TargetName    string        `json:"target_name"`
	TargetMode    string        `json:"target_mode"`
	Command       string        `json:"command"`
	StartedAt     time.Time     `json:"started_at"`
	FinishedAt    time.Time     `json:"finished_at"`
	DurationMS    int64         `json:"duration_ms"`
	ExitCode      int           `json:"exit_code"`
	Stdout        string        `json:"stdout,omitempty"`
	Stderr        string        `json:"stderr,omitempty"`
	Authoritative bool          `json:"authoritative"`
	Simulated     bool          `json:"simulated"`
	Warnings      []string      `json:"warnings,omitempty"`
	Run           runner.Result `json:"run"`
}

type Store struct {
	root string
}

func NewStore() (*Store, error) {
	configDir, err := os.UserConfigDir()
	if err != nil {
		return nil, fmt.Errorf("resolve user config dir: %w", err)
	}
	return &Store{root: filepath.Join(configDir, "fusion", "artifacts")}, nil
}

func (s *Store) BenchmarkPath(id string) string {
	return filepath.Join(s.root, "benchmarks", safeArtifactID(id)+".json")
}

func (s *Store) ProfilePath(id string) string {
	return filepath.Join(s.root, "profiles", safeArtifactID(id)+".json")
}

func (s *Store) SaveBenchmark(result BenchmarkResult, outputPath string) (string, error) {
	if result.ID == "" {
		result.ID = newID(result.Name)
	}
	if result.Version == 0 {
		result.Version = 1
	}
	if outputPath == "" {
		outputPath = s.BenchmarkPath(result.ID)
	}
	return outputPath, writeJSON(outputPath, result)
}

func (s *Store) SaveProfile(result ProfileResult, outputPath string) (string, error) {
	if result.ID == "" {
		result.ID = newID(result.Name)
	}
	if result.Version == 0 {
		result.Version = 1
	}
	if outputPath == "" {
		outputPath = s.ProfilePath(result.ID)
	}
	return outputPath, writeJSON(outputPath, result)
}

func (s *Store) LoadBenchmark(path string) (BenchmarkResult, error) {
	var result BenchmarkResult
	if err := readJSON(path, &result); err != nil {
		return BenchmarkResult{}, err
	}
	return result, nil
}

func (s *Store) LoadProfile(path string) (ProfileResult, error) {
	var result ProfileResult
	if err := readJSON(path, &result); err != nil {
		return ProfileResult{}, err
	}
	return result, nil
}

func newID(name string) string {
	base := strings.TrimSpace(strings.ToLower(name))
	base = strings.NewReplacer(" ", "-", "/", "-", "_", "-").Replace(base)
	base = strings.Trim(base, "-")
	if base == "" {
		base = "run"
	}
	return time.Now().UTC().Format("20060102-150405") + "-" + base
}

func safeArtifactID(value string) string {
	value = strings.TrimSpace(strings.ToLower(value))
	if value == "" {
		return "run"
	}
	var builder strings.Builder
	builder.Grow(len(value))
	lastDash := false
	for _, r := range value {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			builder.WriteRune(r)
			lastDash = false
			continue
		}
		if r == '-' || r == '_' {
			builder.WriteRune(r)
			lastDash = false
			continue
		}
		if !lastDash {
			builder.WriteRune('-')
			lastDash = true
		}
	}
	safe := strings.Trim(builder.String(), "-")
	if safe == "" {
		return "run"
	}
	return safe
}

func writeJSON(path string, value any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create artifact dir: %w", err)
	}

	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("encode artifact: %w", err)
	}
	data = append(data, '\n')

	if err := os.WriteFile(path, data, 0o600); err != nil {
		return fmt.Errorf("write artifact: %w", err)
	}
	return nil
}

func readJSON(path string, target any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read artifact: %w", err)
	}
	if err := json.Unmarshal(data, target); err != nil {
		return fmt.Errorf("decode artifact: %w", err)
	}
	return nil
}
