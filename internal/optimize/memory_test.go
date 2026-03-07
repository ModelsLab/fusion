package optimize

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSaveSessionMemoryEntryAndIndex(t *testing.T) {
	store := &SessionStore{root: t.TempDir()}
	projectRoot := t.TempDir()
	session := store.NewSession(SessionCreateRequest{
		Name:        "memory",
		ProjectRoot: projectRoot,
		Request: Request{
			Model: "wan",
			Task:  "video-generation",
		},
	})
	if _, err := store.Save(session); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	path, err := SaveSessionMemoryEntry(session, SessionMemoryEntry{
		Title:   "winner",
		Summary: "Turbo path won",
		Outcome: "winner",
		Lessons: []string{"video tasks need stage-aware metrics"},
	})
	if err != nil {
		t.Fatalf("SaveSessionMemoryEntry() error = %v", err)
	}
	if filepath.Ext(path) != ".md" {
		t.Fatalf("expected markdown memory entry, got %q", path)
	}
	indexData, err := os.ReadFile(SessionMemoryIndexPath(session))
	if err != nil {
		t.Fatalf("read session memory index: %v", err)
	}
	index := string(indexData)
	if !strings.Contains(index, "video-generation") || !strings.Contains(index, "winner") {
		t.Fatalf("expected task and memory entry in index, got %s", index)
	}
}
