package runner

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ModelsLab/fusion/internal/config"
)

func TestCopyLocalFile(t *testing.T) {
	tempDir := t.TempDir()
	src := filepath.Join(tempDir, "source.txt")
	dst := filepath.Join(tempDir, "nested", "dest.txt")

	if err := os.WriteFile(src, []byte("hello"), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	result, err := Copy(CopyRequest{
		Target:  config.TargetConfig{Name: "local", Mode: "local"},
		Source:  src,
		Dest:    dst,
		Timeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("Copy() error = %v", err)
	}

	data, err := os.ReadFile(dst)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	if string(data) != "hello" {
		t.Fatalf("expected copied contents, got %q", string(data))
	}
	if result.ExecutionMode != "local" {
		t.Fatalf("expected local execution mode, got %q", result.ExecutionMode)
	}
}

func TestCopyDirectoryRequiresRecursive(t *testing.T) {
	tempDir := t.TempDir()
	srcDir := filepath.Join(tempDir, "src")
	if err := os.MkdirAll(srcDir, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}

	_, err := Copy(CopyRequest{
		Target:  config.TargetConfig{Name: "local", Mode: "local"},
		Source:  srcDir,
		Dest:    filepath.Join(tempDir, "dest"),
		Timeout: 5 * time.Second,
	})
	if err == nil {
		t.Fatal("expected copy error for directory without recursive flag")
	}
}

func TestCopyDirectoryRecursive(t *testing.T) {
	tempDir := t.TempDir()
	srcDir := filepath.Join(tempDir, "src")
	nestedDir := filepath.Join(srcDir, "nested")
	if err := os.MkdirAll(nestedDir, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	srcFile := filepath.Join(nestedDir, "kernel.cu")
	if err := os.WriteFile(srcFile, []byte("extern \"C\""), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	dstDir := filepath.Join(tempDir, "dest")
	_, err := Copy(CopyRequest{
		Target:    config.TargetConfig{Name: "local", Mode: "local"},
		Source:    srcDir,
		Dest:      dstDir,
		Recursive: true,
		Timeout:   5 * time.Second,
	})
	if err != nil {
		t.Fatalf("Copy() error = %v", err)
	}

	data, err := os.ReadFile(filepath.Join(dstDir, "nested", "kernel.cu"))
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	if string(data) != "extern \"C\"" {
		t.Fatalf("expected recursive copy contents, got %q", string(data))
	}
}
