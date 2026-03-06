package cudaworkspace

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
	BackendID        = "cuda"
	DefaultTemplate  = "vector_add_one"
	MetadataFilename = "fusion-cuda-workspace.json"
	DefaultBinary    = "build/kernel_runner"
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
	Metadata  string `json:"metadata"`
	Kernel    string `json:"kernel"`
	Build     string `json:"build"`
	Verify    string `json:"verify"`
	Benchmark string `json:"benchmark"`
	Readme    string `json:"readme"`
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
	Output    string
}

type VerifyArgs struct {
	PythonBin string
	GPUArch   string
	Binary    string
	Size      int
	ATol      float64
	RTol      float64
}

type BenchmarkArgs struct {
	PythonBin string
	GPUArch   string
	Binary    string
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
		return Workspace{}, fmt.Errorf("unsupported CUDA template %q", template)
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
			Metadata:  MetadataFilename,
			Kernel:    "kernel.cu",
			Build:     "build.py",
			Verify:    "verify.py",
			Benchmark: "benchmark.py",
			Readme:    "README.md",
		},
	}
	metadata, err := marshalWorkspace(workspace)
	if err != nil {
		return Workspace{}, err
	}

	contents := map[string]string{
		workspace.Files.Metadata:  metadata,
		workspace.Files.Kernel:    renderKernel(),
		workspace.Files.Build:     renderBuild(workspace),
		workspace.Files.Verify:    renderVerify(workspace),
		workspace.Files.Benchmark: renderBenchmark(workspace),
		workspace.Files.Readme:    renderReadme(workspace),
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
	output := valueOrDefault(args.Output, DefaultBinary)
	return strings.Join([]string{
		shellQuote(pythonBin),
		shellQuote(filepath.Join(workspaceDir, "build.py")),
		"--gpu-arch", shellQuote(valueOrDefault(args.GPUArch, "sm90")),
		"--output", shellQuote(output),
	}, " ")
}

func VerifyCommand(workspaceDir string, args VerifyArgs) string {
	pythonBin := valueOrDefault(args.PythonBin, "python3")
	binary := valueOrDefault(args.Binary, DefaultBinary)
	return strings.Join([]string{
		shellQuote(pythonBin),
		shellQuote(filepath.Join(workspaceDir, "verify.py")),
		"--gpu-arch", shellQuote(valueOrDefault(args.GPUArch, "sm90")),
		"--binary", shellQuote(binary),
		"--size", strconv.Itoa(defaultInt(args.Size, 1<<20)),
		"--atol", formatFloat(defaultFloat(args.ATol, 1e-6)),
		"--rtol", formatFloat(defaultFloat(args.RTol, 1e-5)),
	}, " ")
}

func BenchmarkCommand(workspaceDir string, args BenchmarkArgs) string {
	pythonBin := valueOrDefault(args.PythonBin, "python3")
	binary := valueOrDefault(args.Binary, DefaultBinary)
	return strings.Join([]string{
		shellQuote(pythonBin),
		shellQuote(filepath.Join(workspaceDir, "benchmark.py")),
		"--gpu-arch", shellQuote(valueOrDefault(args.GPUArch, "sm90")),
		"--binary", shellQuote(binary),
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
	return `#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

__global__ void add_one_kernel(const float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = x[idx] + 1.0f;
  }
}

int main(int argc, char** argv) {
  std::string mode = "verify";
  int size = 1 << 20;
  int warmup = 20;
  int repeats = 100;
  double atol = 1e-6;
  double rtol = 1e-5;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      mode = argv[++i];
    } else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
      size = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
      repeats = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--atol") == 0 && i + 1 < argc) {
      atol = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--rtol") == 0 && i + 1 < argc) {
      rtol = std::atof(argv[++i]);
    }
  }

  std::vector<float> host_x(size, 0.0f);
  std::vector<float> host_y(size, 0.0f);
  for (int i = 0; i < size; ++i) {
    host_x[i] = static_cast<float>((i % 1024) / 1024.0f);
  }

  float* device_x = nullptr;
  float* device_y = nullptr;
  cudaMalloc(&device_x, size * sizeof(float));
  cudaMalloc(&device_y, size * sizeof(float));
  cudaMemcpy(device_x, host_x.data(), size * sizeof(float), cudaMemcpyHostToDevice);

  int block = 256;
  int grid = (size + block - 1) / block;

  if (mode == "benchmark") {
    for (int i = 0; i < warmup; ++i) {
      add_one_kernel<<<grid, block>>>(device_x, device_y, size);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i) {
      add_one_kernel<<<grid, block>>>(device_x, device_y, size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / static_cast<float>(repeats > 0 ? repeats : 1);

    std::cout << "{\"avg_ms\":" << avg_ms << ",\"elements\":" << size << "}" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  } else {
    add_one_kernel<<<grid, block>>>(device_x, device_y, size);
    cudaDeviceSynchronize();
    cudaMemcpy(host_y.data(), device_y, size * sizeof(float), cudaMemcpyDeviceToHost);

    double max_abs_diff = 0.0;
    bool pass = true;
    for (int i = 0; i < size; ++i) {
      double expected = static_cast<double>(host_x[i]) + 1.0;
      double actual = static_cast<double>(host_y[i]);
      double abs_diff = std::fabs(actual - expected);
      double rel_limit = rtol * std::fabs(expected);
      if (abs_diff > max_abs_diff) {
        max_abs_diff = abs_diff;
      }
      if (abs_diff > atol + rel_limit) {
        pass = false;
        break;
      }
    }

    std::cout << "{\"correctness_pass\":" << (pass ? 1 : 0)
              << ",\"max_abs_diff\":" << max_abs_diff
              << ",\"elements\":" << size << "}" << std::endl;
    cudaFree(device_x);
    cudaFree(device_y);
    return pass ? 0 : 1;
  }

  cudaFree(device_x);
  cudaFree(device_y);
  return 0;
}
`
}

func renderBuild(workspace Workspace) string {
	return fmt.Sprintf(`from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def normalize_arch(value: str) -> str:
    value = value.strip().lower().replace("-", "").replace("_", "")
    if value.startswith("sm"):
        value = value[2:]
    return "sm_" + value


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile the CUDA workspace with nvcc.")
    parser.add_argument("--gpu-arch", default=%q)
    parser.add_argument("--output", default=%q)
    args = parser.parse_args()

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "nvcc",
        "-O3",
        "-std=c++17",
        "kernel.cu",
        "-o",
        str(output),
        "-arch=" + normalize_arch(args.gpu_arch),
    ]
    subprocess.run(command, check=True)
    print(json.dumps({
        "compile_status": 1,
        "binary": str(output),
        "gpu_arch": args.gpu_arch,
    }))


if __name__ == "__main__":
    main()
`, workspace.GPUArch, DefaultBinary)
}

func renderVerify(workspace Workspace) string {
	return fmt.Sprintf(`from __future__ import annotations

import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Run correctness verification for the CUDA workspace binary.")
    parser.add_argument("--gpu-arch", default=%q)
    parser.add_argument("--binary", default=%q)
    parser.add_argument("--size", type=int, default=%d)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    command = [
        args.binary,
        "--mode", "verify",
        "--size", str(args.size),
        "--atol", str(args.atol),
        "--rtol", str(args.rtol),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
`, workspace.GPUArch, DefaultBinary, 1<<20)
}

func renderBenchmark(workspace Workspace) string {
	return fmt.Sprintf(`from __future__ import annotations

import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the CUDA workspace binary.")
    parser.add_argument("--gpu-arch", default=%q)
    parser.add_argument("--binary", default=%q)
    parser.add_argument("--size", type=int, default=%d)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    command = [
        args.binary,
        "--mode", "benchmark",
        "--size", str(args.size),
        "--warmup", str(args.warmup),
        "--repeats", str(args.repeats),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
`, workspace.GPUArch, DefaultBinary, 1<<20)
}

func renderReadme(workspace Workspace) string {
	lines := []string{
		"# " + workspace.Name,
		"",
		"This workspace was generated by Fusion to give CUDA C++ kernels a first-class place in the optimization loop.",
		"",
		"Files",
		"- `kernel.cu`: CUDA source file with verify and benchmark entrypoints.",
		"- `build.py`: nvcc build wrapper.",
		"- `verify.py`: correctness run wrapper.",
		"- `benchmark.py`: benchmark run wrapper.",
		"",
		"Example commands",
		"```bash",
		"python3 build.py --gpu-arch " + workspace.GPUArch,
		"python3 verify.py --gpu-arch " + workspace.GPUArch,
		"python3 benchmark.py --gpu-arch " + workspace.GPUArch,
		"```",
		"",
		"Fusion commands",
		"```bash",
		"fusion optimize cuda build --workspace " + workspace.Name + " --gpu-arch " + workspace.GPUArch,
		"fusion optimize cuda verify --workspace " + workspace.Name + " --gpu-arch " + workspace.GPUArch,
		"fusion optimize cuda benchmark --workspace " + workspace.Name + " --gpu-arch " + workspace.GPUArch,
		"```",
		"",
	}
	return strings.Join(lines, "\n")
}
