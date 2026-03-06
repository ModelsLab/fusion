package cudaworkspace

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInitCreatesWorkspaceFiles(t *testing.T) {
	root := t.TempDir()
	outputDir := filepath.Join(root, "cuda-add-one")

	workspace, err := Init(InitRequest{
		Name:      "cuda-add-one",
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
		workspace.Files.Readme,
	}
	for _, name := range expectedFiles {
		if _, err := os.Stat(filepath.Join(outputDir, name)); err != nil {
			t.Fatalf("expected file %s to exist: %v", name, err)
		}
	}
}

func TestCommandsIncludeArguments(t *testing.T) {
	buildCommand := BuildCommand("/tmp/workspace", BuildArgs{
		GPUArch: "sm90",
		Output:  "build/custom-binary",
	})
	if !strings.Contains(buildCommand, "build.py") {
		t.Fatalf("expected build.py in command, got %q", buildCommand)
	}
	if !strings.Contains(buildCommand, "--output 'build/custom-binary'") {
		t.Fatalf("expected output in build command, got %q", buildCommand)
	}

	verifyCommand := VerifyCommand("/tmp/workspace", VerifyArgs{
		GPUArch: "sm100",
		Binary:  "build/custom-binary",
		Size:    4096,
		ATol:    1e-4,
		RTol:    1e-3,
	})
	if !strings.Contains(verifyCommand, "--binary 'build/custom-binary'") {
		t.Fatalf("expected binary in verify command, got %q", verifyCommand)
	}
	if !strings.Contains(verifyCommand, "--size 4096") {
		t.Fatalf("expected size in verify command, got %q", verifyCommand)
	}

	benchmarkCommand := BenchmarkCommand("/tmp/workspace", BenchmarkArgs{
		GPUArch: "sm100",
		Binary:  "build/custom-binary",
		Size:    8192,
		Warmup:  5,
		Repeats: 12,
	})
	if !strings.Contains(benchmarkCommand, "--warmup 5") {
		t.Fatalf("expected warmup in benchmark command, got %q", benchmarkCommand)
	}
}
