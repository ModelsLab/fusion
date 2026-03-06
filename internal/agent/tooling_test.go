package agent

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ModelsLab/fusion/internal/config"
	"github.com/ModelsLab/fusion/internal/kb"
	"github.com/ModelsLab/fusion/internal/optimize"
)

func TestDefaultToolsExposeGenericCodingSurface(t *testing.T) {
	tools := DefaultTools(ToolContext{})
	names := map[string]bool{}
	for _, tool := range tools {
		names[tool.Definition.Name] = true
	}

	expected := []string{
		"append_file",
		"copy_path",
		"move_path",
		"stat_path",
		"run_command",
		"run_benchmark",
		"run_profile",
		"register_optimization_candidate",
		"record_optimization_stage",
	}
	for _, name := range expected {
		if !names[name] {
			t.Fatalf("expected tool %q to be exposed", name)
		}
	}

	unexpected := []string{
		"init_triton_workspace",
		"init_cuda_workspace",
		"init_cute_workspace",
		"build_triton_workspace",
		"build_cuda_workspace",
		"build_cute_workspace",
		"verify_triton_workspace",
		"verify_cuda_workspace",
		"verify_cute_workspace",
	}
	for _, name := range unexpected {
		if names[name] {
			t.Fatalf("did not expect backend-specific tool %q in DefaultTools", name)
		}
	}
}

func TestRegisterOptimizationCandidateAndRecordStage(t *testing.T) {
	configHome := t.TempDir()
	t.Setenv("HOME", configHome)
	t.Setenv("XDG_CONFIG_HOME", configHome)

	projectRoot := t.TempDir()
	store, err := optimize.NewSessionStore()
	if err != nil {
		t.Fatalf("NewSessionStore() error = %v", err)
	}

	session := store.NewSession(optimize.SessionCreateRequest{
		Name:        "agent-tools",
		ProjectRoot: projectRoot,
		Request: optimize.Request{
			GPU:      "h100",
			Workload: "decode",
		},
		Context: kb.ContextPacket{
			Request: kb.ContextRequest{GPU: "h100", Workload: "decode"},
		},
	})
	if _, err := store.Save(session); err != nil {
		t.Fatalf("Save() session error = %v", err)
	}

	registerTool := registerOptimizationCandidateTool(ToolContext{CWD: projectRoot})
	registerPayload, err := registerTool.Execute(context.Background(), mustJSON(t, map[string]any{
		"session":     session.ID,
		"name":        "baseline-bf16",
		"backend":     "baseline",
		"description": "runtime baseline",
	}))
	if err != nil {
		t.Fatalf("registerOptimizationCandidateTool() error = %v", err)
	}
	var registered struct {
		Candidate optimize.Candidate `json:"candidate"`
		Workspace string             `json:"workspace"`
	}
	if err := json.Unmarshal([]byte(registerPayload), &registered); err != nil {
		t.Fatalf("decode register payload: %v", err)
	}
	if registered.Candidate.ID == "" {
		t.Fatal("expected registered candidate id")
	}
	if registered.Workspace == "" {
		t.Fatal("expected registered workspace")
	}

	recordTool := recordOptimizationStageTool()
	if _, err := recordTool.Execute(context.Background(), mustJSON(t, map[string]any{
		"session":       session.ID,
		"candidate":     registered.Candidate.ID,
		"stage":         "baseline",
		"artifact_path": "/tmp/baseline.json",
		"command":       "python benchmark.py",
		"exit_code":     0,
		"metrics": map[string]float64{
			"tokens_per_sec": 42,
		},
	})); err != nil {
		t.Fatalf("recordOptimizationStageTool() error = %v", err)
	}

	loaded, err := store.Load(session.ID)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	candidate, ok := loaded.CandidateByID(registered.Candidate.ID)
	if !ok {
		t.Fatalf("expected candidate %s to persist", registered.Candidate.ID)
	}
	if candidate.Stages["baseline"].ArtifactPath != "/tmp/baseline.json" {
		t.Fatalf("expected recorded artifact path, got %+v", candidate.Stages["baseline"])
	}
	if candidate.Stages["baseline"].Metrics["tokens_per_sec"] != 42 {
		t.Fatalf("expected recorded metrics, got %+v", candidate.Stages["baseline"].Metrics)
	}
}

func TestRunCommandRecordsCandidateStageArtifact(t *testing.T) {
	configHome := t.TempDir()
	t.Setenv("HOME", configHome)
	t.Setenv("XDG_CONFIG_HOME", configHome)

	projectRoot := t.TempDir()
	cfg, err := config.NewManager()
	if err != nil {
		t.Fatalf("NewManager() error = %v", err)
	}
	store, err := optimize.NewSessionStore()
	if err != nil {
		t.Fatalf("NewSessionStore() error = %v", err)
	}

	session := store.NewSession(optimize.SessionCreateRequest{
		Name:        "run-command",
		ProjectRoot: projectRoot,
		Request: optimize.Request{
			GPU:      "h100",
			Workload: "decode",
		},
	})
	candidate := session.UpsertCandidate(optimize.Candidate{
		Name:      "baseline-bf16",
		Backend:   "baseline",
		Workspace: filepath.Join(session.WorkspaceRoot, "baseline-bf16"),
	})
	if err := os.MkdirAll(candidate.Workspace, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	if _, err := store.Save(session); err != nil {
		t.Fatalf("Save() error = %v", err)
	}

	tool := runCommandTool(ToolContext{CWD: projectRoot, Config: cfg})
	output, err := tool.Execute(context.Background(), mustJSON(t, map[string]any{
		"command":   "printf 'tokens_per_sec=42\\n'",
		"session":   session.ID,
		"candidate": candidate.ID,
		"stage":     "baseline",
	}))
	if err != nil {
		t.Fatalf("runCommandTool() error = %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal([]byte(output), &payload); err != nil {
		t.Fatalf("decode run_command output: %v", err)
	}
	if payload["artifact_path"] == "" {
		t.Fatalf("expected artifact_path in payload, got %v", payload)
	}

	loaded, err := store.Load(session.ID)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	updated, ok := loaded.CandidateByID(candidate.ID)
	if !ok {
		t.Fatalf("expected candidate %s", candidate.ID)
	}
	stage := updated.Stages["baseline"]
	if stage.ArtifactPath == "" {
		t.Fatalf("expected recorded artifact path, got %+v", stage)
	}
	if stage.Metrics["tokens_per_sec"] != 42 {
		t.Fatalf("expected parsed metrics, got %+v", stage.Metrics)
	}
}

func TestBenchmarkAndProfileToolsRecordCandidateStages(t *testing.T) {
	configHome := t.TempDir()
	t.Setenv("HOME", configHome)
	t.Setenv("XDG_CONFIG_HOME", configHome)

	projectRoot := t.TempDir()
	cfg, err := config.NewManager()
	if err != nil {
		t.Fatalf("NewManager() error = %v", err)
	}
	store, err := optimize.NewSessionStore()
	if err != nil {
		t.Fatalf("NewSessionStore() error = %v", err)
	}

	session := store.NewSession(optimize.SessionCreateRequest{
		Name:        "benchmark-profile",
		ProjectRoot: projectRoot,
		Request: optimize.Request{
			GPU:      "h100",
			Workload: "decode",
		},
	})
	candidate := session.UpsertCandidate(optimize.Candidate{
		Name:      "torch-compile",
		Backend:   "torch-compile",
		Workspace: filepath.Join(session.WorkspaceRoot, "torch-compile"),
	})
	if err := os.MkdirAll(candidate.Workspace, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	if _, err := store.Save(session); err != nil {
		t.Fatalf("Save() error = %v", err)
	}

	benchmarkTool := benchmarkRunTool(ToolContext{CWD: projectRoot, Config: cfg})
	if _, err := benchmarkTool.Execute(context.Background(), mustJSON(t, map[string]any{
		"command":   "printf 'tokens_per_sec=17\\nlatency_ms=9\\n'",
		"session":   session.ID,
		"candidate": candidate.ID,
	})); err != nil {
		t.Fatalf("benchmarkRunTool() error = %v", err)
	}

	profileTool := profileRunTool(ToolContext{CWD: projectRoot, Config: cfg})
	if _, err := profileTool.Execute(context.Background(), mustJSON(t, map[string]any{
		"command":   "printf 'profile ok\\n'",
		"tool":      "ncu",
		"session":   session.ID,
		"candidate": candidate.ID,
	})); err != nil {
		t.Fatalf("profileRunTool() error = %v", err)
	}

	loaded, err := store.Load(session.ID)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	updated, ok := loaded.CandidateByID(candidate.ID)
	if !ok {
		t.Fatalf("expected candidate %s", candidate.ID)
	}
	if updated.Stages["benchmark"].ArtifactPath == "" {
		t.Fatalf("expected benchmark stage artifact, got %+v", updated.Stages["benchmark"])
	}
	if updated.Stages["benchmark"].Metrics["tokens_per_sec"] != 17 {
		t.Fatalf("expected benchmark metrics, got %+v", updated.Stages["benchmark"].Metrics)
	}
	if updated.Stages["profile"].ArtifactPath == "" {
		t.Fatalf("expected profile stage artifact, got %+v", updated.Stages["profile"])
	}
}

func TestResolvePathRejectsEscapesOutsideWorkingDirectory(t *testing.T) {
	root := t.TempDir()
	if _, err := resolvePath(root, "../outside.txt"); err == nil {
		t.Fatal("expected resolvePath to reject traversal outside the working directory")
	}
}

func TestRunCommandRejectsDangerousRecursiveDelete(t *testing.T) {
	configHome := t.TempDir()
	t.Setenv("HOME", configHome)
	t.Setenv("XDG_CONFIG_HOME", configHome)

	projectRoot := t.TempDir()
	cfg, err := config.NewManager()
	if err != nil {
		t.Fatalf("NewManager() error = %v", err)
	}

	tool := runCommandTool(ToolContext{CWD: projectRoot, Config: cfg})
	output, err := tool.Execute(context.Background(), mustJSON(t, map[string]any{
		"command": "rm -rf build",
	}))
	if err != nil {
		t.Fatalf("runCommandTool() transport error = %v", err)
	}
	if !strings.Contains(output, "rm -rf") {
		t.Fatalf("expected rm -rf guidance in tool output, got %s", output)
	}
}

func TestBenchmarkAndProfileRejectDangerousRecursiveDelete(t *testing.T) {
	configHome := t.TempDir()
	t.Setenv("HOME", configHome)
	t.Setenv("XDG_CONFIG_HOME", configHome)

	projectRoot := t.TempDir()
	cfg, err := config.NewManager()
	if err != nil {
		t.Fatalf("NewManager() error = %v", err)
	}

	benchmarkTool := benchmarkRunTool(ToolContext{CWD: projectRoot, Config: cfg})
	benchmarkOutput, err := benchmarkTool.Execute(context.Background(), mustJSON(t, map[string]any{
		"command": "rm -f -r cache && echo hi",
	}))
	if err != nil {
		t.Fatalf("benchmarkRunTool() transport error = %v", err)
	}
	if !strings.Contains(benchmarkOutput, "rm -rf") {
		t.Fatalf("expected benchmark tool output to contain rm -rf guidance, got %s", benchmarkOutput)
	}

	profileTool := profileRunTool(ToolContext{CWD: projectRoot, Config: cfg})
	profileOutput, err := profileTool.Execute(context.Background(), mustJSON(t, map[string]any{
		"command": "bash -lc 'rm -rf traces'",
	}))
	if err != nil {
		t.Fatalf("profileRunTool() transport error = %v", err)
	}
	if !strings.Contains(profileOutput, "rm -rf") {
		t.Fatalf("expected profile tool output to contain rm -rf guidance, got %s", profileOutput)
	}
}

func mustJSON(t *testing.T, value any) string {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	return string(data)
}
