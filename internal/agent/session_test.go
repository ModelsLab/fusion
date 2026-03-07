package agent

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestSessionStoreSaveAndLoad(t *testing.T) {
	root := t.TempDir()
	store := &Store{root: root}
	session := store.NewSession("gpt-test", "/tmp/project", "system")
	session.Messages = append(session.Messages, Message{
		Role:    "user",
		Content: "optimize this model",
	})

	path, err := store.Save(session)
	if err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("expected session file to exist: %v", err)
	}

	loaded, err := store.Load(session.ID)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if loaded.Provider != "modelslab" {
		t.Fatalf("expected provider modelslab, got %q", loaded.Provider)
	}
	if len(loaded.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(loaded.Messages))
	}
}

func TestNewSessionIDIncludesProjectName(t *testing.T) {
	id := newSessionID(filepath.Join("/tmp", "fusion-project"))
	if id == "" || id[len(id)-14:] == "" {
		t.Fatal("expected non-empty session id")
	}
}

func TestSessionStoreFindLatestByCWD(t *testing.T) {
	root := t.TempDir()
	store := &Store{root: root}

	oldSession := store.NewSession("gpt-test", "/tmp/project-a", "system")
	oldSession.CreatedAt = time.Now().UTC().Add(-2 * time.Hour)
	oldSession.UpdatedAt = time.Now().UTC().Add(-2 * time.Hour)
	if _, err := store.Save(oldSession); err != nil {
		t.Fatalf("Save(oldSession) error = %v", err)
	}

	latestSession := store.NewSession("gpt-test", "/tmp/project-a", "system")
	latestSession.CreatedAt = time.Now().UTC().Add(-1 * time.Hour)
	latestSession.UpdatedAt = time.Now().UTC().Add(-1 * time.Hour)
	if _, err := store.Save(latestSession); err != nil {
		t.Fatalf("Save(latestSession) error = %v", err)
	}

	otherSession := store.NewSession("gpt-test", "/tmp/project-b", "system")
	if _, err := store.Save(otherSession); err != nil {
		t.Fatalf("Save(otherSession) error = %v", err)
	}

	found, err := store.FindLatestByCWD("/tmp/project-a")
	if err != nil {
		t.Fatalf("FindLatestByCWD() error = %v", err)
	}
	if found == nil {
		t.Fatal("expected matching session")
	}
	if found.ID != latestSession.ID {
		t.Fatalf("expected latest session %q, got %q", latestSession.ID, found.ID)
	}
}
