package cutedsl

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInitCreatesWorkspaceFiles(t *testing.T) {
	root := t.TempDir()
	outputDir := filepath.Join(root, "cute-add-one")

	workspace, err := Init(InitRequest{
		Name:      "cute-add-one",
		OutputDir: outputDir,
		GPUArch:   "sm100",
	})
	if err != nil {
		t.Fatalf("Init() error = %v", err)
	}

	if workspace.Backend != BackendID {
		t.Fatalf("expected backend %q, got %q", BackendID, workspace.Backend)
	}
	if workspace.GPUArch != "sm100" {
		t.Fatalf("expected gpu arch sm100, got %q", workspace.GPUArch)
	}

	expectedFiles := []string{
		MetadataFilename,
		workspace.Files.Kernel,
		workspace.Files.Build,
		workspace.Files.Verify,
		workspace.Files.Benchmark,
		workspace.Files.Requirements,
		workspace.Files.Readme,
	}
	for _, name := range expectedFiles {
		if _, err := os.Stat(filepath.Join(outputDir, name)); err != nil {
			t.Fatalf("expected file %s to exist: %v", name, err)
		}
	}
}

func TestBuildAndVerifyCommandsIncludeArguments(t *testing.T) {
	buildCommand := BuildCommand("/tmp/workspace", BuildArgs{
		GPUArch:   "sm90",
		OptLevel:  2,
		ExportDir: "/tmp/export",
	})
	if !strings.Contains(buildCommand, "build.py") {
		t.Fatalf("expected build.py in command, got %q", buildCommand)
	}
	if !strings.Contains(buildCommand, "--gpu-arch 'sm90'") {
		t.Fatalf("expected gpu arch in build command, got %q", buildCommand)
	}
	if !strings.Contains(buildCommand, "--export-dir '/tmp/export'") {
		t.Fatalf("expected export dir in build command, got %q", buildCommand)
	}

	verifyCommand := VerifyCommand("/tmp/workspace", VerifyArgs{
		GPUArch: "sm100",
		Size:    4096,
		ATol:    1e-4,
		RTol:    1e-3,
	})
	if !strings.Contains(verifyCommand, "verify.py") {
		t.Fatalf("expected verify.py in command, got %q", verifyCommand)
	}
	if !strings.Contains(verifyCommand, "--size 4096") {
		t.Fatalf("expected size in verify command, got %q", verifyCommand)
	}
	if !strings.Contains(verifyCommand, "--atol 0.0001") {
		t.Fatalf("expected atol in verify command, got %q", verifyCommand)
	}
}
