package agent

import (
	"os"
	"path/filepath"
	"testing"
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
