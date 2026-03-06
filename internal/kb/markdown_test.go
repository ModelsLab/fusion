package kb

import (
	"os"
	"path/filepath"
	"testing"
)

func TestExportMarkdownAndLoadMarkdownDir(t *testing.T) {
	store := &Store{
		Sources: []Source{
			{
				ID:           "nvidia-cute-dsl",
				Title:        "NVIDIA CuTe DSL",
				URL:          "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html",
				Type:         "official-docs",
				Category:     "cute",
				Reliability:  "official",
				ReviewStatus: "reviewed",
				Summary:      "CuTe DSL reference.",
				Tags:         []string{"cute", "cutlass"},
			},
		},
		Strategies: []Strategy{
			{
				ID:           "triton_for_memory_bound_fusions",
				Title:        "Triton For Memory Bound Fusions",
				Category:     "kernel-backend",
				Summary:      "Use Triton for memory-bound fused kernels.",
				SupportLevel: "recommended",
				Workloads:    []string{"decode"},
				Operators:    []string{"rmsnorm"},
				SourceIDs:    []string{"nvidia-cute-dsl"},
			},
		},
		Documents: []Document{
			{
				ID:           "blackwell-cutile-notes",
				Title:        "Blackwell cuTile Notes",
				Category:     "research-note",
				Summary:      "Long-form notes about Blackwell attention tuning.",
				SupportLevel: "recommended",
				Reliability:  "curated",
				ReviewStatus: "reviewed",
				GPUFamilies:  []string{"Blackwell"},
				Operators:    []string{"attention"},
				Backends:     []string{"cutile"},
				Path:         "documents/blackwell-cutile-notes.md",
				Body:         "## Summary\n\nBlackwell attention should try cuTile first.",
			},
		},
	}

	root := t.TempDir()
	if err := ExportMarkdown(store, root); err != nil {
		t.Fatalf("ExportMarkdown() error = %v", err)
	}

	loaded, err := LoadMarkdownDir(root)
	if err != nil {
		t.Fatalf("LoadMarkdownDir() error = %v", err)
	}
	if len(loaded.Sources) != 1 {
		t.Fatalf("expected 1 source, got %d", len(loaded.Sources))
	}
	if len(loaded.Strategies) != 1 {
		t.Fatalf("expected 1 strategy, got %d", len(loaded.Strategies))
	}
	if len(loaded.Documents) != 1 {
		t.Fatalf("expected 1 document, got %d", len(loaded.Documents))
	}
	if loaded.Documents[0].Body == "" {
		t.Fatal("expected markdown document body to round-trip")
	}

	outputRoot := filepath.Join(t.TempDir(), "knowledge")
	dbPath := filepath.Join(outputRoot, "knowledge.db")
	if err := WriteSQLiteIndex(loaded, dbPath); err != nil {
		t.Fatalf("WriteSQLiteIndex() error = %v", err)
	}
	if _, err := os.Stat(dbPath); err != nil {
		t.Fatalf("expected generated knowledge.db: %v", err)
	}

	reloaded, err := LoadFromSQLitePath(dbPath)
	if err != nil {
		t.Fatalf("LoadFromSQLitePath() error = %v", err)
	}
	if len(reloaded.Documents) != 1 {
		t.Fatalf("expected 1 document from sqlite, got %d", len(reloaded.Documents))
	}
	if reloaded.Documents[0].Body != loaded.Documents[0].Body {
		t.Fatal("expected markdown document body to persist through sqlite")
	}

	fsLoaded, err := LoadFromFS(os.DirFS(outputRoot))
	if err != nil {
		t.Fatalf("LoadFromFS() error = %v", err)
	}
	if len(fsLoaded.Strategies) != 1 {
		t.Fatalf("expected 1 strategy from sqlite fs load, got %d", len(fsLoaded.Strategies))
	}
	if fsLoaded.Documents[0].ID != "blackwell-cutile-notes" {
		t.Fatalf("unexpected document id %q", fsLoaded.Documents[0].ID)
	}
}

func TestLoadMarkdownDirLegacyDocumentWithoutFrontMatter(t *testing.T) {
	root := t.TempDir()
	docPath := filepath.Join(root, "documents", "legacy-note.md")
	if err := os.MkdirAll(filepath.Dir(docPath), 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	content := "# Legacy Note\n\nThis is an older knowledge document without structured front matter."
	if err := os.WriteFile(docPath, []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	loaded, err := LoadMarkdownDir(root)
	if err != nil {
		t.Fatalf("LoadMarkdownDir() error = %v", err)
	}
	if len(loaded.Documents) != 1 {
		t.Fatalf("expected 1 document, got %d", len(loaded.Documents))
	}
	if loaded.Documents[0].ID != "legacy-note" {
		t.Fatalf("unexpected document id %q", loaded.Documents[0].ID)
	}
	if loaded.Documents[0].Title != "Legacy Note" {
		t.Fatalf("unexpected document title %q", loaded.Documents[0].Title)
	}
	if loaded.Documents[0].Body != content {
		t.Fatal("expected legacy document body to be preserved")
	}
}

func TestLoadMarkdownDirLegacySourceWithoutFrontMatter(t *testing.T) {
	root := t.TempDir()
	sourcePath := filepath.Join(root, "sources", "legacy-source.md")
	if err := os.MkdirAll(filepath.Dir(sourcePath), 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	content := "# Legacy Source\n\nThis older source note predates structured front matter."
	if err := os.WriteFile(sourcePath, []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	loaded, err := LoadMarkdownDir(root)
	if err != nil {
		t.Fatalf("LoadMarkdownDir() error = %v", err)
	}
	if len(loaded.Sources) != 1 {
		t.Fatalf("expected 1 source, got %d", len(loaded.Sources))
	}
	if loaded.Sources[0].ID != "legacy-source" {
		t.Fatalf("unexpected source id %q", loaded.Sources[0].ID)
	}
	if loaded.Sources[0].Title != "Legacy Source" {
		t.Fatalf("unexpected source title %q", loaded.Sources[0].Title)
	}
}
