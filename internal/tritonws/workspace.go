package tritonws

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	BackendID        = "triton"
	DefaultTemplate  = "vector_add_one"
	MetadataFilename = "fusion-triton-workspace.json"
)

type Workspace struct {
	Version   int            `json:"version"`
	Name      string         `json:"name"`
	Backend   string         `json:"backend"`
	Template  string         `json:"template"`
	Operation string         `json:"operation"`
	GPUArch   string         `json:"gpu_arch"`
	Target    string         `json:"target,omitempty"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
	Files     WorkspaceFiles `json:"files"`
}

type WorkspaceFiles struct {
	Metadata     string `json:"metadata"`
	Kernel       string `json:"kernel"`
	Build        string `json:"build"`
	Verify       string `json:"verify"`
	Benchmark    string `json:"benchmark"`
	Requirements string `json:"requirements"`
	Readme       string `json:"readme"`
}

type InitRequest struct {
	Name      string
	OutputDir string
	Template  string
	Operation string
	GPUArch   string
	Target    string
	Force     bool
}

type BuildArgs struct {
	PythonBin string
	GPUArch   string
	Size      int
}

type VerifyArgs struct {
	PythonBin string
	GPUArch   string
	Size      int
	ATol      float64
	RTol      float64
}

type BenchmarkArgs struct {
	PythonBin string
	GPUArch   string
	Size      int
	Warmup    int
	Repeats   int
}

func Init(req InitRequest) (Workspace, error) {
	name := strings.TrimSpace(req.Name)
	if name == "" {
		return Workspace{}, fmt.Errorf("workspace name is required")
	}

	template := strings.TrimSpace(req.Template)
	if template == "" {
		template = DefaultTemplate
	}
	if template != DefaultTemplate {
		return Workspace{}, fmt.Errorf("unsupported Triton template %q", template)
	}

	outputDir := strings.TrimSpace(req.OutputDir)
	if outputDir == "" {
		outputDir = name
	}

	absOutputDir, err := filepath.Abs(outputDir)
	if err != nil {
		return Workspace{}, fmt.Errorf("resolve output dir: %w", err)
	}
	if err := ensureWorkspaceDir(absOutputDir, req.Force); err != nil {
		return Workspace{}, err
	}

	now := time.Now().UTC()
	workspace := Workspace{
		Version:   1,
		Name:      name,
		Backend:   BackendID,
		Template:  template,
		Operation: valueOrDefault(req.Operation, "vector_add_one"),
		GPUArch:   valueOrDefault(req.GPUArch, "sm90"),
		Target:    strings.TrimSpace(req.Target),
		CreatedAt: now,
		UpdatedAt: now,
		Files: WorkspaceFiles{
			Metadata:     MetadataFilename,
			Kernel:       "kernel.py",
			Build:        "build.py",
			Verify:       "verify.py",
			Benchmark:    "benchmark.py",
			Requirements: "requirements.txt",
			Readme:       "README.md",
		},
	}
	metadata, err := marshalWorkspace(workspace)
	if err != nil {
		return Workspace{}, err
	}

	contents := map[string]string{
		workspace.Files.Metadata:     metadata,
		workspace.Files.Kernel:       renderKernel(),
		workspace.Files.Build:        renderBuild(workspace),
		workspace.Files.Verify:       renderVerify(workspace),
		workspace.Files.Benchmark:    renderBenchmark(workspace),
		workspace.Files.Requirements: renderRequirements(),
		workspace.Files.Readme:       renderReadme(workspace),
	}

	for relativePath, content := range contents {
		absolutePath := filepath.Join(absOutputDir, relativePath)
		if err := os.WriteFile(absolutePath, []byte(content), 0o644); err != nil {
			return Workspace{}, fmt.Errorf("write %s: %w", absolutePath, err)
		}
	}

	return workspace, nil
}

func Load(workspaceDir string) (Workspace, error) {
	data, err := os.ReadFile(filepath.Join(strings.TrimSpace(workspaceDir), MetadataFilename))
	if err != nil {
		return Workspace{}, fmt.Errorf("read workspace metadata: %w", err)
	}

	var workspace Workspace
	if err := json.Unmarshal(data, &workspace); err != nil {
		return Workspace{}, fmt.Errorf("decode workspace metadata: %w", err)
	}
	return workspace, nil
}

func BuildCommand(workspaceDir string, args BuildArgs) string {
	pythonBin := valueOrDefault(args.PythonBin, "python3")
	return strings.Join([]string{
		shellQuote(pythonBin),
		shellQuote(filepath.Join(workspaceDir, "build.py")),
		"--gpu-arch", shellQuote(valueOrDefault(args.GPUArch, "sm90")),
		"--size", strconv.Itoa(defaultInt(args.Size, 1<<20)),
	}, " ")
}

func VerifyCommand(workspaceDir string, args VerifyArgs) string {
	pythonBin := valueOrDefault(args.PythonBin, "python3")
	return strings.Join([]string{
		shellQuote(pythonBin),
		shellQuote(filepath.Join(workspaceDir, "verify.py")),
		"--gpu-arch", shellQuote(valueOrDefault(args.GPUArch, "sm90")),
		"--size", strconv.Itoa(defaultInt(args.Size, 1<<20)),
		"--atol", formatFloat(defaultFloat(args.ATol, 1e-6)),
		"--rtol", formatFloat(defaultFloat(args.RTol, 1e-5)),
	}, " ")
}

func BenchmarkCommand(workspaceDir string, args BenchmarkArgs) string {
	pythonBin := valueOrDefault(args.PythonBin, "python3")
	return strings.Join([]string{
		shellQuote(pythonBin),
		shellQuote(filepath.Join(workspaceDir, "benchmark.py")),
		"--gpu-arch", shellQuote(valueOrDefault(args.GPUArch, "sm90")),
		"--size", strconv.Itoa(defaultInt(args.Size, 1<<20)),
		"--warmup", strconv.Itoa(defaultInt(args.Warmup, 20)),
		"--repeats", strconv.Itoa(defaultInt(args.Repeats, 100)),
	}, " ")
}

func ensureWorkspaceDir(path string, force bool) error {
	if err := os.MkdirAll(path, 0o755); err != nil {
		return fmt.Errorf("create workspace dir: %w", err)
	}

	entries, err := os.ReadDir(path)
	if err != nil {
		return fmt.Errorf("read workspace dir: %w", err)
	}
	if len(entries) == 0 || force {
		return nil
	}
	return fmt.Errorf("workspace dir %s is not empty; pass --force to overwrite the scaffold", path)
}

func marshalWorkspace(workspace Workspace) (string, error) {
	data, err := json.MarshalIndent(workspace, "", "  ")
	if err != nil {
		return "", fmt.Errorf("encode workspace metadata: %w", err)
	}
	return string(append(data, '\n')), nil
}

func valueOrDefault(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	return value
}

func defaultInt(value, fallback int) int {
	if value <= 0 {
		return fallback
	}
	return value
}

func defaultFloat(value, fallback float64) float64 {
	if value <= 0 {
		return fallback
	}
	return value
}

func formatFloat(value float64) string {
	return strconv.FormatFloat(value, 'g', -1, 64)
}

func shellQuote(value string) string {
	if value == "" {
		return "''"
	}
	return "'" + strings.ReplaceAll(value, "'", `'"'"'`) + "'"
}

func renderKernel() string {
	return `from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def add_one_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, x + 1.0, mask=mask)


def run(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("Triton workspace requires a CUDA tensor")
    y = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_one_kernel[grid](x, y, n_elements, BLOCK_SIZE=block_size)
    return y
`
}

func renderBuild(workspace Workspace) string {
	return fmt.Sprintf(`from __future__ import annotations

import argparse
import json
import torch

import kernel


def main() -> None:
    parser = argparse.ArgumentParser(description="JIT compile the Triton workspace kernel.")
    parser.add_argument("--gpu-arch", default=%q)
    parser.add_argument("--size", type=int, default=%d)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Triton compilation.")

    x = torch.randn(args.size, device="cuda", dtype=torch.float32)
    kernel.run(x)
    torch.cuda.synchronize()
    print(json.dumps({
        "compile_status": 1,
        "elements": args.size,
        "gpu_arch": args.gpu_arch,
    }))


if __name__ == "__main__":
    main()
`, workspace.GPUArch, 1<<20)
}

func renderVerify(workspace Workspace) string {
	return fmt.Sprintf(`from __future__ import annotations

import argparse
import json
import torch

import kernel


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the Triton workspace kernel against a PyTorch reference.")
    parser.add_argument("--gpu-arch", default=%q)
    parser.add_argument("--size", type=int, default=%d)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Triton verification.")

    x = torch.randn(args.size, device="cuda", dtype=torch.float32)
    actual = kernel.run(x)
    expected = x + 1.0
    max_abs_diff = torch.max(torch.abs(actual - expected)).item()
    correctness_pass = torch.allclose(actual, expected, atol=args.atol, rtol=args.rtol)
    torch.cuda.synchronize()
    print(json.dumps({
        "correctness_pass": 1 if correctness_pass else 0,
        "max_abs_diff": max_abs_diff,
        "elements": args.size,
        "gpu_arch": args.gpu_arch,
    }))
    if not correctness_pass:
        raise SystemExit("verification failed")


if __name__ == "__main__":
    main()
`, workspace.GPUArch, 1<<20)
}

func renderBenchmark(workspace Workspace) string {
	return fmt.Sprintf(`from __future__ import annotations

import argparse
import json
import torch

import kernel


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the Triton workspace kernel.")
    parser.add_argument("--gpu-arch", default=%q)
    parser.add_argument("--size", type=int, default=%d)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Triton benchmarking.")

    x = torch.randn(args.size, device="cuda", dtype=torch.float32)
    for _ in range(args.warmup):
        kernel.run(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.repeats):
        kernel.run(x)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / max(args.repeats, 1)
    print(json.dumps({
        "avg_ms": avg_ms,
        "elements": args.size,
        "gpu_arch": args.gpu_arch,
    }))


if __name__ == "__main__":
    main()
`, workspace.GPUArch, 1<<20)
}

func renderRequirements() string {
	return strings.Join([]string{
		"torch",
		"triton",
		"",
	}, "\n")
}

func renderReadme(workspace Workspace) string {
	lines := []string{
		"# " + workspace.Name,
		"",
		"This workspace was generated by Fusion to give Triton kernels a first-class place in the optimization loop.",
		"",
		"Files",
		"- `kernel.py`: Triton kernel entrypoint.",
		"- `build.py`: JIT compilation warmup for the kernel.",
		"- `verify.py`: correctness verification against a PyTorch reference.",
		"- `benchmark.py`: simple CUDA event benchmark.",
		"",
		"Example commands",
		"```bash",
		"python3 -m venv .venv && source .venv/bin/activate",
		"pip install -r requirements.txt",
		"python3 build.py --gpu-arch " + workspace.GPUArch,
		"python3 verify.py --gpu-arch " + workspace.GPUArch,
		"python3 benchmark.py --gpu-arch " + workspace.GPUArch,
		"```",
		"",
		"Fusion commands",
		"```bash",
		"fusion optimize triton build --workspace " + workspace.Name + " --gpu-arch " + workspace.GPUArch,
		"fusion optimize triton verify --workspace " + workspace.Name + " --gpu-arch " + workspace.GPUArch,
		"fusion optimize triton benchmark --workspace " + workspace.Name + " --gpu-arch " + workspace.GPUArch,
		"```",
		"",
	}
	return strings.Join(lines, "\n")
}
