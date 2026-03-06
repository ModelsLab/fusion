package cli

import (
	"testing"

	"github.com/ModelsLab/fusion/internal/config"
)

func TestResolveTargetFallsBackToImplicitLocal(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	manager, err := config.NewManager()
	if err != nil {
		t.Fatalf("NewManager() error = %v", err)
	}

	target, name, err := resolveTarget(&runtimeState{Config: manager}, "")
	if err != nil {
		t.Fatalf("resolveTarget() error = %v", err)
	}

	if name != "local" {
		t.Fatalf("expected implicit local target name, got %q", name)
	}
	if target.Mode != "local" {
		t.Fatalf("expected local mode, got %q", target.Mode)
	}
}

func TestResolveTargetUsesDefaultConfiguredTarget(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	manager, err := config.NewManager()
	if err != nil {
		t.Fatalf("NewManager() error = %v", err)
	}

	cfg, err := manager.Load()
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	cfg.Targets["lab-4090"] = config.TargetConfig{
		Name: "lab-4090",
		Mode: "ssh",
		Host: "example.com",
		GPU:  "rtx4090",
	}
	cfg.DefaultTarget = "lab-4090"
	if err := manager.Save(cfg); err != nil {
		t.Fatalf("Save() error = %v", err)
	}

	target, name, err := resolveTarget(&runtimeState{Config: manager}, "")
	if err != nil {
		t.Fatalf("resolveTarget() error = %v", err)
	}

	if name != "lab-4090" {
		t.Fatalf("expected default target name lab-4090, got %q", name)
	}
	if target.GPU != "rtx4090" {
		t.Fatalf("expected target GPU rtx4090, got %q", target.GPU)
	}
}

func TestRuntimeShellEnvIncludesConfiguredHuggingFaceToken(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	manager, err := config.NewManager()
	if err != nil {
		t.Fatalf("NewManager() error = %v", err)
	}
	if err := manager.SetHuggingFaceToken("hf-configured"); err != nil {
		t.Fatalf("SetHuggingFaceToken() error = %v", err)
	}

	env, err := runtimeShellEnv(&runtimeState{Config: manager})
	if err != nil {
		t.Fatalf("runtimeShellEnv() error = %v", err)
	}
	if env["HF_TOKEN"] != "hf-configured" {
		t.Fatalf("expected HF_TOKEN in shell env, got %#v", env)
	}
	if env["HUGGING_FACE_HUB_TOKEN"] != "hf-configured" {
		t.Fatalf("expected HUGGING_FACE_HUB_TOKEN in shell env, got %#v", env)
	}
}

func TestRuntimeShellEnvIncludesConfiguredGitHubToken(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	manager, err := config.NewManager()
	if err != nil {
		t.Fatalf("NewManager() error = %v", err)
	}
	if err := manager.SetGitHubToken("gh-configured"); err != nil {
		t.Fatalf("SetGitHubToken() error = %v", err)
	}

	env, err := runtimeShellEnv(&runtimeState{Config: manager})
	if err != nil {
		t.Fatalf("runtimeShellEnv() error = %v", err)
	}
	if env["GITHUB_TOKEN"] != "gh-configured" {
		t.Fatalf("expected GITHUB_TOKEN in shell env, got %#v", env)
	}
	if env["GH_TOKEN"] != "gh-configured" {
		t.Fatalf("expected GH_TOKEN in shell env, got %#v", env)
	}
}
