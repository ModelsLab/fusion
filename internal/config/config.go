package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

const currentVersion = 1

type ModelsLabConfig struct {
	Token string `json:"token,omitempty"`
	Model string `json:"model,omitempty"`
}

type HuggingFaceConfig struct {
	Token string `json:"token,omitempty"`
}

type GitHubConfig struct {
	Token string `json:"token,omitempty"`
}

type TargetConfig struct {
	Name         string `json:"name"`
	Mode         string `json:"mode"`
	Description  string `json:"description,omitempty"`
	GPU          string `json:"gpu,omitempty"`
	ProxyGPU     string `json:"proxy_gpu,omitempty"`
	Host         string `json:"host,omitempty"`
	User         string `json:"user,omitempty"`
	Port         int    `json:"port,omitempty"`
	IdentityFile string `json:"identity_file,omitempty"`
	RemoteDir    string `json:"remote_dir,omitempty"`
	Shell        string `json:"shell,omitempty"`
}

type Config struct {
	Version       int                     `json:"version"`
	ModelsLab     ModelsLabConfig         `json:"modelslab,omitempty"`
	HuggingFace   HuggingFaceConfig       `json:"huggingface,omitempty"`
	GitHub        GitHubConfig            `json:"github,omitempty"`
	DefaultTarget string                  `json:"default_target,omitempty"`
	Targets       map[string]TargetConfig `json:"targets,omitempty"`
}

type Manager struct {
	path string
}

func NewManager() (*Manager, error) {
	configDir, err := os.UserConfigDir()
	if err != nil {
		return nil, fmt.Errorf("resolve user config dir: %w", err)
	}

	return &Manager{
		path: filepath.Join(configDir, "fusion", "config.json"),
	}, nil
}

func (m *Manager) Path() string {
	return m.path
}

func (m *Manager) Load() (Config, error) {
	data, err := os.ReadFile(m.path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return defaultConfig(), nil
		}
		return Config{}, fmt.Errorf("read config: %w", err)
	}

	type legacyProviderConfig struct {
		Token string `json:"token,omitempty"`
		Model string `json:"model,omitempty"`
	}

	type diskConfig struct {
		Version         int                             `json:"version"`
		ModelsLab       ModelsLabConfig                 `json:"modelslab,omitempty"`
		HuggingFace     HuggingFaceConfig               `json:"huggingface,omitempty"`
		GitHub          GitHubConfig                    `json:"github,omitempty"`
		DefaultTarget   string                          `json:"default_target,omitempty"`
		Targets         map[string]TargetConfig         `json:"targets,omitempty"`
		DefaultProvider string                          `json:"default_provider,omitempty"`
		Providers       map[string]legacyProviderConfig `json:"providers,omitempty"`
	}

	var disk diskConfig
	if err := json.Unmarshal(data, &disk); err != nil {
		return Config{}, fmt.Errorf("decode config: %w", err)
	}

	cfg := Config{
		Version:       disk.Version,
		ModelsLab:     disk.ModelsLab,
		HuggingFace:   disk.HuggingFace,
		GitHub:        disk.GitHub,
		DefaultTarget: disk.DefaultTarget,
		Targets:       disk.Targets,
	}
	if cfg.Version == 0 {
		cfg.Version = currentVersion
	}
	if cfg.Targets == nil {
		cfg.Targets = map[string]TargetConfig{}
	}
	if cfg.ModelsLab == (ModelsLabConfig{}) {
		if legacy, ok := disk.Providers["modelslab"]; ok {
			cfg.ModelsLab.Token = legacy.Token
			cfg.ModelsLab.Model = legacy.Model
		}
	}

	return cfg, nil
}

func (m *Manager) Save(cfg Config) error {
	if cfg.Version == 0 {
		cfg.Version = currentVersion
	}
	if cfg.Targets == nil {
		cfg.Targets = map[string]TargetConfig{}
	}

	if err := os.MkdirAll(filepath.Dir(m.path), 0o755); err != nil {
		return fmt.Errorf("create config dir: %w", err)
	}

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return fmt.Errorf("encode config: %w", err)
	}
	data = append(data, '\n')

	tmpPath := m.path + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0o600); err != nil {
		return fmt.Errorf("write temp config: %w", err)
	}
	if err := os.Rename(tmpPath, m.path); err != nil {
		return fmt.Errorf("replace config: %w", err)
	}

	return nil
}

func (m *Manager) SetModelsLabToken(token string) error {
	cfg, err := m.Load()
	if err != nil {
		return err
	}
	cfg.ModelsLab.Token = token
	return m.Save(cfg)
}

func (m *Manager) SetModelsLabModel(model string) error {
	cfg, err := m.Load()
	if err != nil {
		return err
	}
	cfg.ModelsLab.Model = model
	return m.Save(cfg)
}

func (m *Manager) ClearModelsLab() error {
	cfg, err := m.Load()
	if err != nil {
		return err
	}
	cfg.ModelsLab = ModelsLabConfig{}
	return m.Save(cfg)
}

func (m *Manager) SetHuggingFaceToken(token string) error {
	cfg, err := m.Load()
	if err != nil {
		return err
	}
	cfg.HuggingFace.Token = token
	return m.Save(cfg)
}

func (m *Manager) ClearHuggingFace() error {
	cfg, err := m.Load()
	if err != nil {
		return err
	}
	cfg.HuggingFace = HuggingFaceConfig{}
	return m.Save(cfg)
}

func (m *Manager) SetGitHubToken(token string) error {
	cfg, err := m.Load()
	if err != nil {
		return err
	}
	cfg.GitHub.Token = token
	return m.Save(cfg)
}

func (m *Manager) ClearGitHub() error {
	cfg, err := m.Load()
	if err != nil {
		return err
	}
	cfg.GitHub = GitHubConfig{}
	return m.Save(cfg)
}

func (m *Manager) SetTarget(target TargetConfig) error {
	cfg, err := m.Load()
	if err != nil {
		return err
	}
	if cfg.Targets == nil {
		cfg.Targets = map[string]TargetConfig{}
	}

	cfg.Targets[target.Name] = target
	return m.Save(cfg)
}

func (m *Manager) RemoveTarget(name string) error {
	cfg, err := m.Load()
	if err != nil {
		return err
	}

	delete(cfg.Targets, name)
	if cfg.DefaultTarget == name {
		cfg.DefaultTarget = ""
	}

	return m.Save(cfg)
}

func (m *Manager) SetDefaultTarget(name string) error {
	cfg, err := m.Load()
	if err != nil {
		return err
	}
	cfg.DefaultTarget = name
	return m.Save(cfg)
}

func defaultConfig() Config {
	return Config{
		Version: currentVersion,
		Targets: map[string]TargetConfig{},
	}
}
