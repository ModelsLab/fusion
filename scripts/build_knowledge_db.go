package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/ModelsLab/fusion/internal/kb"
)

func main() {
	var (
		root         = flag.String("root", ".", "repository root")
		out          = flag.String("out", "knowledge/knowledge.db", "output SQLite path relative to root")
		markdownRoot = flag.String("markdown-root", "knowledgebase", "markdown knowledge root relative to repository root")
	)
	flag.Parse()

	rootPath, err := filepath.Abs(*root)
	if err != nil {
		fatalf("resolve root: %v", err)
	}

	knowledgeRoot := filepath.Join(rootPath, "knowledge")
	markdownPath := filepath.Join(rootPath, *markdownRoot)
	hasMarkdown, err := kb.HasMarkdownFiles(markdownPath)
	if err != nil {
		fatalf("scan markdown knowledge files: %v", err)
	}

	var fallback *kb.Store
	if !hasMarkdown {
		fallback, err = kb.LoadFromFS(os.DirFS(knowledgeRoot))
		if err != nil {
			fatalf("load fallback knowledge db: %v", err)
		}
	}

	store, err := kb.BuildMarkdownBackedStore(markdownPath, fallback)
	if err != nil {
		fatalf("build markdown-backed knowledge store: %v", err)
	}

	outputPath := filepath.Join(rootPath, *out)
	if err := kb.WriteSQLiteIndex(store, outputPath); err != nil {
		fatalf("write knowledge db: %v", err)
	}

	fmt.Printf("wrote %s\n", outputPath)
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
