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
	if strings.Contains(text, `"providers"`) {
		t.Fatalf("did not expect legacy providers field in saved config: %s", text)
	}
}
