package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadMigratesLegacyModelslabProviderConfig(t *testing.T) {
	root := t.TempDir()
	manager := &Manager{path: filepath.Join(root, "config.json")}

	legacy := `{
  "version": 1,
  "default_provider": "modelslab",
  "providers": {
    "modelslab": {
      "token": "ml-legacy",
      "model": "openai-gpt-5.4-pro"
    }
  }
}`
	if err := os.WriteFile(manager.path, []byte(legacy), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	cfg, err := manager.Load()
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.ModelsLab.Token != "ml-legacy" {
		t.Fatalf("expected migrated token, got %q", cfg.ModelsLab.Token)
	}
	if cfg.ModelsLab.Model != "openai-gpt-5.4-pro" {
		t.Fatalf("expected migrated model, got %q", cfg.ModelsLab.Model)
	}
}

func TestSaveWritesModelslabConfigOnly(t *testing.T) {
	root := t.TempDir()
	manager := &Manager{path: filepath.Join(root, "config.json")}

	cfg := defaultConfig()
	cfg.ModelsLab = ModelsLabConfig{
		Token: "ml-current",
		Model: "openai-gpt-5.4-pro",
	}
	cfg.HuggingFace = HuggingFaceConfig{
		Token: "hf-current",
	}
	cfg.GitHub = GitHubConfig{
		Token: "gh-current",
	}
	if err := manager.Save(cfg); err != nil {
		t.Fatalf("Save() error = %v", err)
	}

	data, err := os.ReadFile(manager.path)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	text := string(data)
	if !strings.Contains(text, `"modelslab"`) {
		t.Fatalf("expected modelslab block in saved config: %s", text)
	}
	if !strings.Contains(text, `"huggingface"`) {
		t.Fatalf("expected huggingface block in saved config: %s", text)
	}
	if !strings.Contains(text, `"github"`) {
		t.Fatalf("expected github block in saved config: %s", text)
	}
	if strings.Contains(text, `"providers"`) {
		t.Fatalf("did not expect legacy providers field in saved config: %s", text)
	}
}

func TestSetAndClearHuggingFaceToken(t *testing.T) {
	root := t.TempDir()
	manager := &Manager{path: filepath.Join(root, "config.json")}

	if err := manager.SetHuggingFaceToken("hf_test"); err != nil {
		t.Fatalf("SetHuggingFaceToken() error = %v", err)
	}
	cfg, err := manager.Load()
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.HuggingFace.Token != "hf_test" {
		t.Fatalf("expected Hugging Face token to persist, got %q", cfg.HuggingFace.Token)
	}

	if err := manager.ClearHuggingFace(); err != nil {
		t.Fatalf("ClearHuggingFace() error = %v", err)
	}
	cleared, err := manager.Load()
	if err != nil {
		t.Fatalf("Load() after clear error = %v", err)
	}
	if cleared.HuggingFace.Token != "" {
		t.Fatalf("expected Hugging Face token to be cleared, got %q", cleared.HuggingFace.Token)
	}
}

func TestSetAndClearGitHubToken(t *testing.T) {
	root := t.TempDir()
	manager := &Manager{path: filepath.Join(root, "config.json")}

	if err := manager.SetGitHubToken("gh_test"); err != nil {
		t.Fatalf("SetGitHubToken() error = %v", err)
	}
	cfg, err := manager.Load()
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.GitHub.Token != "gh_test" {
		t.Fatalf("expected GitHub token to persist, got %q", cfg.GitHub.Token)
	}

	if err := manager.ClearGitHub(); err != nil {
		t.Fatalf("ClearGitHub() error = %v", err)
	}
	cleared, err := manager.Load()
	if err != nil {
		t.Fatalf("Load() after clear error = %v", err)
	}
	if cleared.GitHub.Token != "" {
		t.Fatalf("expected GitHub token to be cleared, got %q", cleared.GitHub.Token)
	}
}
