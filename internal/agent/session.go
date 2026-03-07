package agent

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/ModelsLab/fusion/internal/modelslab"
)

type Store struct {
	root string
}

func NewStore() (*Store, error) {
	configDir, err := os.UserConfigDir()
	if err != nil {
		return nil, fmt.Errorf("resolve user config dir: %w", err)
	}

	return &Store{
		root: filepath.Join(configDir, "fusion", "sessions"),
	}, nil
}

func (s *Store) Root() string {
	return s.root
}

func (s *Store) SessionPath(id string) string {
	return filepath.Join(s.root, id+".json")
}

func (s *Store) NewSession(model, cwd, systemPrompt string) *Session {
	now := time.Now().UTC()
	return &Session{
		Version:      1,
		ID:           newSessionID(cwd),
		Provider:     modelslab.ProviderID,
		Model:        strings.TrimSpace(model),
		CWD:          strings.TrimSpace(cwd),
		SystemPrompt: systemPrompt,
		CreatedAt:    now,
		UpdatedAt:    now,
	}
}

func (s *Store) Save(session *Session) (string, error) {
	if session == nil {
		return "", fmt.Errorf("session is required")
	}
	session.UpdatedAt = time.Now().UTC()
	if session.Version == 0 {
		session.Version = 1
	}

	if err := os.MkdirAll(s.root, 0o755); err != nil {
		return "", fmt.Errorf("create sessions dir: %w", err)
	}

	path := s.SessionPath(session.ID)
	data, err := json.MarshalIndent(session, "", "  ")
	if err != nil {
		return "", fmt.Errorf("encode session: %w", err)
	}
	data = append(data, '\n')

	if err := os.WriteFile(path, data, 0o600); err != nil {
		return "", fmt.Errorf("write session: %w", err)
	}
	return path, nil
}

func (s *Store) Load(id string) (*Session, error) {
	path := s.SessionPath(strings.TrimSpace(id))
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read session: %w", err)
	}

	var session Session
	if err := json.Unmarshal(data, &session); err != nil {
		return nil, fmt.Errorf("decode session: %w", err)
	}
	return &session, nil
}

func (s *Store) List() ([]*Session, error) {
	entries, err := os.ReadDir(s.root)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read sessions dir: %w", err)
	}

	sessions := make([]*Session, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}
		session, loadErr := s.Load(strings.TrimSuffix(entry.Name(), ".json"))
		if loadErr != nil {
			return nil, loadErr
		}
		sessions = append(sessions, session)
	}

	sort.Slice(sessions, func(i, j int) bool {
		if sessions[i].UpdatedAt.Equal(sessions[j].UpdatedAt) {
			return sessions[i].CreatedAt.After(sessions[j].CreatedAt)
		}
		return sessions[i].UpdatedAt.After(sessions[j].UpdatedAt)
	})
	return sessions, nil
}

func (s *Store) FindLatestByCWD(cwd string) (*Session, error) {
	target := normalizeSessionCWD(cwd)
	if target == "" {
		return nil, nil
	}

	sessions, err := s.List()
	if err != nil {
		return nil, err
	}
	for _, session := range sessions {
		if normalizeSessionCWD(session.CWD) == target {
			return session, nil
		}
	}
	return nil, nil
}

func normalizeSessionCWD(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	abs, err := filepath.Abs(path)
	if err == nil {
		path = abs
	}
	return filepath.Clean(path)
}

func newSessionID(cwd string) string {
	base := strings.TrimSpace(strings.ToLower(filepath.Base(cwd)))
	if base == "." || base == "" || base == string(filepath.Separator) {
		base = "session"
	}
	base = strings.NewReplacer(" ", "-", "/", "-", "_", "-").Replace(base)
	base = strings.Trim(base, "-")
	return time.Now().UTC().Format("20060102-150405") + "-" + base
}
