package cutedsl

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
	BackendID        = "cute_dsl"
	DefaultTemplate  = "add_one_tvm_ffi"
	MetadataFilename = "fusion-cute-workspace.json"
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
	OptLevel  int
	ExportDir string
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
		return Workspace{}, fmt.Errorf("unsupported CuTe template %q", template)
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
		Operation: valueOrDefault(req.Operation, "elementwise_add_one"),
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
		workspace.Files.Kernel:       renderKernel(workspace),
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
	metadataPath := filepath.Join(strings.TrimSpace(workspaceDir), MetadataFilename)
	data, err := os.ReadFile(metadataPath)
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
	command := []string{
		shellQuote(pythonBin),
		shellQuote(filepath.Join(workspaceDir, "build.py")),
		"--gpu-arch", shellQuote(valueOrDefault(args.GPUArch, "sm90")),
		"--opt-level", strconv.Itoa(defaultInt(args.OptLevel, 3)),
	}
	if strings.TrimSpace(args.ExportDir) != "" {
		command = append(command, "--export-dir", shellQuote(args.ExportDir))
	}
	return strings.Join(command, " ")
}

func VerifyCommand(workspaceDir string, args VerifyArgs) string {
	pythonBin := valueOrDefault(args.PythonBin, "python3")
	command := []string{
		shellQuote(pythonBin),
		shellQuote(filepath.Join(workspaceDir, "verify.py")),
		"--gpu-arch", shellQuote(valueOrDefault(args.GPUArch, "sm90")),
		"--size", strconv.Itoa(defaultInt(args.Size, 1<<20)),
		"--atol", formatFloat(defaultFloat(args.ATol, 1e-6)),
		"--rtol", formatFloat(defaultFloat(args.RTol, 1e-5)),
	}
	return strings.Join(command, " ")
}

func BenchmarkCommand(workspaceDir string, args BenchmarkArgs) string {
	pythonBin := valueOrDefault(args.PythonBin, "python3")
	command := []string{
		shellQuote(pythonBin),
		shellQuote(filepath.Join(workspaceDir, "benchmark.py")),
		"--gpu-arch", shellQuote(valueOrDefault(args.GPUArch, "sm90")),
		"--size", strconv.Itoa(defaultInt(args.Size, 1<<20)),
		"--warmup", strconv.Itoa(defaultInt(args.Warmup, 25)),
		"--repeats", strconv.Itoa(defaultInt(args.Repeats, 100)),
	}
	return strings.Join(command, " ")
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

func renderKernel(workspace Workspace) string {
	return fmt.Sprintf(`from __future__ import annotations

import json
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

try:
    import cutlass.cute as cute
except ImportError:
    cute = None

WORKSPACE_METADATA = Path(__file__).resolve().parent / %q


def _require_cute():
    if cute is None:
        raise RuntimeError(
            "CuTe DSL is not installed. Run pip install -r requirements.txt inside this workspace."
        )
    return cute


def workspace_metadata() -> dict:
    return json.loads(WORKSPACE_METADATA.read_text())


if cute is not None:
    @cute.kernel
    def device_add_one(a: cute.Tensor, b: cute.Tensor):
        threads_per_block = 128
        cta_x, _, _ = cute.arch.block_idx()
        tid_x, _, _ = cute.arch.thread_idx()
        tid = cta_x * threads_per_block + tid_x
        if tid < a.shape[0]:
            b[tid] = a[tid] + 1.0


    @cute.jit
    def add_one(a: cute.Tensor, b: cute.Tensor):
        size = a.shape[0]
        threads_per_block = 128
        blocks = (size + threads_per_block - 1) // threads_per_block
        device_add_one(a, b).launch(
            grid=(blocks, 1, 1),
            block=(threads_per_block, 1, 1),
        )
else:
    def add_one(*args, **kwargs):
        _require_cute()


def compile_kernel(gpu_arch: str = %q, opt_level: int = 3, enable_tvm_ffi: bool = True):
    cute_mod = _require_cute()
    symbolic_size = cute_mod.sym_int()
    fake_input = cute_mod.runtime.make_fake_compact_tensor(cute_mod.Float32, (symbolic_size,))
    fake_output = cute_mod.runtime.make_fake_compact_tensor(cute_mod.Float32, (symbolic_size,))

    options = [f"--opt-level {opt_level}"]
    if gpu_arch:
        options.append(f"--gpu-arch {gpu_arch}")
    if enable_tvm_ffi:
        options.append("--enable-tvm-ffi")

    return cute_mod.compile(add_one, fake_input, fake_output, options=" ".join(options))


def reference(input_tensor):
    if torch is None:
        raise RuntimeError("PyTorch is required for reference and verification flows.")
    return input_tensor + 1.0
`, MetadataFilename, workspace.GPUArch)
}

func renderBuild(workspace Workspace) string {
	return fmt.Sprintf(`from __future__ import annotations

import argparse
import json
from pathlib import Path

import kernel


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile the CuTe DSL workspace kernel.")
    parser.add_argument("--gpu-arch", default=%q)
    parser.add_argument("--opt-level", type=int, default=3)
    parser.add_argument("--export-dir", default="")
    parser.add_argument("--function-prefix", default="fusion_cute_")
    args = parser.parse_args()

    compiled = kernel.compile_kernel(gpu_arch=args.gpu_arch, opt_level=args.opt_level)

    exported = 0
    if args.export_dir:
        export_dir = Path(args.export_dir).resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(compiled, "export_to_c"):
            compiled.export_to_c(str(export_dir), "kernel", function_prefix=args.function_prefix)
            exported = 1
        else:
            raise RuntimeError(
                "The compiled CuTe function does not expose export_to_c on this installation."
            )

    print(
        json.dumps(
            {
                "compiled": 1,
                "aot_exported": exported,
                "opt_level": args.opt_level,
            }
        )
    )


if __name__ == "__main__":
    main()
`, workspace.GPUArch)
}

func renderVerify(workspace Workspace) string {
	return fmt.Sprintf(`from __future__ import annotations

import argparse
import json

import kernel

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is required for verification.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the CuTe DSL kernel against a PyTorch reference.")
    parser.add_argument("--gpu-arch", default=%q)
    parser.add_argument("--size", type=int, default=%d)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to verify the CuTe kernel.")

    compiled = kernel.compile_kernel(gpu_arch=args.gpu_arch)
    input_tensor = torch.arange(args.size, dtype=torch.float32, device="cuda")
    output_tensor = torch.empty_like(input_tensor)

    compiled(input_tensor, output_tensor)
    expected = kernel.reference(input_tensor)

    diff = torch.abs(output_tensor - expected)
    max_abs_error = float(diff.max().item())
    denom = torch.clamp(torch.abs(expected), min=args.atol)
    max_rel_error = float((diff / denom).max().item())
    passed = bool(torch.allclose(output_tensor, expected, atol=args.atol, rtol=args.rtol))

    print(
        json.dumps(
            {
                "passed": 1 if passed else 0,
                "max_abs_error": max_abs_error,
                "max_rel_error": max_rel_error,
            }
        )
    )

    if not passed:
        raise SystemExit("Verification failed.")


if __name__ == "__main__":
    main()
`, workspace.GPUArch, 1<<20)
}

func renderBenchmark(workspace Workspace) string {
	return fmt.Sprintf(`from __future__ import annotations

import argparse
import json

import kernel

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is required for benchmarking.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the CuTe DSL kernel with CUDA events.")
    parser.add_argument("--gpu-arch", default=%q)
    parser.add_argument("--size", type=int, default=%d)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to benchmark the CuTe kernel.")

    compiled = kernel.compile_kernel(gpu_arch=args.gpu_arch)
    input_tensor = torch.randn(args.size, device="cuda", dtype=torch.float32)
    output_tensor = torch.empty_like(input_tensor)

    for _ in range(args.warmup):
        compiled(input_tensor, output_tensor)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.repeats):
        compiled(input_tensor, output_tensor)
    end.record()
    torch.cuda.synchronize()

    kernel_time_ms = float(start.elapsed_time(end) / args.repeats)
    elements_per_sec = float(args.size / (kernel_time_ms / 1000.0))

    print(
        json.dumps(
            {
                "kernel_time_ms": kernel_time_ms,
                "elements_per_sec": elements_per_sec,
            }
        )
    )


if __name__ == "__main__":
    main()
`, workspace.GPUArch, 1<<20)
}

func renderRequirements() string {
	return strings.Join([]string{
		"nvidia-cutlass-dsl",
		"apache-tvm-ffi",
		"torch",
		"torch-c-dlpack-ext",
		"",
	}, "\n")
}

func renderReadme(workspace Workspace) string {
	return strings.Join([]string{
		"# " + workspace.Name,
		"",
		"This workspace was generated by Fusion to give CuTe DSL a first-class place in the optimization workflow.",
		"",
		"What is here:",
		"- kernel.py: a working CuTe DSL starter kernel using the TVM FFI-friendly JIT flow",
		"- build.py: compiles the kernel and can try AOT export when the installed CuTe build exposes export_to_c",
		"- verify.py: checks the kernel against a PyTorch reference with tolerances",
		"- benchmark.py: emits JSON metrics that Fusion can ingest",
		"",
		"Recommended environment:",
		"- Linux",
		"- Python 3.10 to 3.13",
		"- CUDA-compatible NVIDIA driver",
		"- pip install -r requirements.txt",
		"",
		"Quick usage:",
		"  python3 build.py --gpu-arch " + workspace.GPUArch,
		"  python3 verify.py --gpu-arch " + workspace.GPUArch,
		"  python3 benchmark.py --gpu-arch " + workspace.GPUArch,
		"",
		"To integrate this with Fusion:",
		"  fusion optimize cute build --workspace " + workspace.Name + " --gpu-arch " + workspace.GPUArch,
		"  fusion optimize cute verify --workspace " + workspace.Name + " --gpu-arch " + workspace.GPUArch,
		"  fusion benchmark run --command " + BenchmarkCommand(workspace.Name, BenchmarkArgs{GPUArch: workspace.GPUArch}),
		"",
		"For real inference kernels, replace the starter add-one kernel with your CuTe implementation and keep the compile_kernel and reference contract stable so Fusion commands continue to work.",
		"",
	}, "\n")
}
