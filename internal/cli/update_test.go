package cli

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ModelsLab/fusion/internal/kb"
)

func TestUpdateKnowledgeCommandBootstrapsMarkdownAndBuildsSQLite(t *testing.T) {
	sourceRoot := filepath.Join(t.TempDir(), "knowledgebase")
	outputRoot := filepath.Join(t.TempDir(), "knowledge")

	cmd := newUpdateKnowledgeCommand()
	stdout := &bytes.Buffer{}
	stderr := &bytes.Buffer{}
	cmd.SetOut(stdout)
	cmd.SetErr(stderr)
	cmd.SetArgs([]string{
		"--source-root", sourceRoot,
		"--output-root", outputRoot,
		"--bootstrap",
	})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute() error = %v\nstderr=%s", err, stderr.String())
	}

	hasMarkdown, err := kb.HasMarkdownFiles(sourceRoot)
	if err != nil {
		t.Fatalf("HasMarkdownFiles() error = %v", err)
	}
	if !hasMarkdown {
		t.Fatal("expected update kb to bootstrap markdown docs")
	}

	dbPath := filepath.Join(outputRoot, "knowledge.db")
	if _, err := os.Stat(dbPath); err != nil {
		t.Fatalf("expected update kb to build knowledge.db: %v", err)
	}

	store, err := kb.LoadFromSQLitePath(dbPath)
	if err != nil {
		t.Fatalf("LoadFromSQLitePath() error = %v", err)
	}
	if len(store.Skills) == 0 {
		t.Fatal("expected generated knowledge db to include skills")
	}
	if !strings.Contains(stdout.String(), "SQLite index:") {
		t.Fatalf("expected sqlite output summary, got %q", stdout.String())
	}
}
