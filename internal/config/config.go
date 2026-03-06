package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

const currentVersion = 1

type ProviderConfig struct {
	Token string `json:"token,omitempty"`
}

type Config struct {
	Version         int                       `json:"version"`
	DefaultProvider string                    `json:"default_provider,omitempty"`
	Providers       map[string]ProviderConfig `json:"providers,omitempty"`
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

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("decode config: %w", err)
	}

	if cfg.Version == 0 {
		cfg.Version = currentVersion
	}
	if cfg.Providers == nil {
		cfg.Providers = map[string]ProviderConfig{}
	}

	return cfg, nil
}

func (m *Manager) Save(cfg Config) error {
	if cfg.Version == 0 {
		cfg.Version = currentVersion
	}
	if cfg.Providers == nil {
		cfg.Providers = map[string]ProviderConfig{}
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

func (m *Manager) SetToken(provider, token string) error {
	cfg, err := m.Load()
	if err != nil {
		return err
	}
	cfg.Providers[provider] = ProviderConfig{Token: token}
	return m.Save(cfg)
}

func (m *Manager) RemoveProvider(provider string) error {
	cfg, err := m.Load()
	if err != nil {
		return err
	}

	delete(cfg.Providers, provider)
	if cfg.DefaultProvider == provider {
		cfg.DefaultProvider = ""
	}

	return m.Save(cfg)
}

func (m *Manager) SetDefaultProvider(provider string) error {
	cfg, err := m.Load()
	if err != nil {
		return err
	}
	cfg.DefaultProvider = provider
	return m.Save(cfg)
}

func defaultConfig() Config {
	return Config{
		Version:   currentVersion,
		Providers: map[string]ProviderConfig{},
	}
}
