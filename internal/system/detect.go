package system

import (
	"bytes"
	"fmt"
	"os/exec"
	"runtime"
	"strings"
)

type Host struct {
	OS   string
	Arch string
}

type ToolStatus struct {
	Name      string
	Available bool
	Path      string
	Version   string
	Notes     string
}

type GPU struct {
	Name     string
	Vendor   string
	Driver   string
	MemoryMB string
	Source   string
}

type Environment struct {
	Host  Host
	Tools []ToolStatus
	GPUs  []GPU
}

func DetectEnvironment() Environment {
	return Environment{
		Host: Host{
			OS:   runtime.GOOS,
			Arch: runtime.GOARCH,
		},
		Tools: detectToolchain(),
		GPUs:  DetectNVIDIAGPUs(),
	}
}

func DetectNVIDIAGPUs() []GPU {
	path, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return nil
	}

	output, err := exec.Command(path, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil
	}

	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	gpus := make([]GPU, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		parts := strings.Split(line, ",")
		if len(parts) < 3 {
			continue
		}

		gpus = append(gpus, GPU{
			Name:     strings.TrimSpace(parts[0]),
			Vendor:   "nvidia",
			Driver:   strings.TrimSpace(parts[1]),
			MemoryMB: strings.TrimSpace(parts[2]),
			Source:   "nvidia-smi",
		})
	}

	return gpus
}

func detectToolchain() []ToolStatus {
	tools := []struct {
		name    string
		args    []string
		notes   string
		version string
	}{
		{name: "go", args: []string{"version"}, notes: "Builds the Fusion CLI itself."},
		{name: "python3", args: []string{"--version"}, notes: "Useful for CuTe DSL, Triton kernels, and profiling glue."},
		{name: "ssh", args: []string{"-V"}, notes: "Needed for remote execution on Ubuntu GPU targets."},
		{name: "scp", args: []string{"-h"}, version: "scp", notes: "Needed for copying kernels and benchmark assets to remote targets."},
		{name: "nvidia-smi", args: []string{"--help"}, version: "nvidia-smi", notes: "Primary NVIDIA device detection on Linux hosts."},
		{name: "nvcc", args: []string{"--version"}, notes: "Required for CUDA C++ compilation flows."},
		{name: "ncu", args: []string{"--version"}, notes: "Nsight Compute CLI for kernel metrics and bottleneck analysis."},
		{name: "nsys", args: []string{"--version"}, notes: "Nsight Systems for end-to-end trace analysis."},
		{name: "uv", args: []string{"--version"}, notes: "Good fit for Python helper environments and Triton stacks."},
	}

	statuses := make([]ToolStatus, 0, len(tools))
	for _, tool := range tools {
		path, err := exec.LookPath(tool.name)
		if err != nil {
			statuses = append(statuses, ToolStatus{
				Name:      tool.name,
				Available: false,
				Notes:     tool.notes,
			})
			continue
		}

		version := tool.version
		if version == "" {
			version = firstNonEmptyLine(runCommand(path, tool.args...))
		}

		statuses = append(statuses, ToolStatus{
			Name:      tool.name,
			Available: true,
			Path:      path,
			Version:   version,
			Notes:     tool.notes,
		})
	}

	return statuses
}

func runCommand(binary string, args ...string) string {
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd := exec.Command(binary, args...)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return strings.TrimSpace(stderr.String())
	}

	output := strings.TrimSpace(stdout.String())
	if output == "" {
		output = strings.TrimSpace(stderr.String())
	}
	return output
}

func firstNonEmptyLine(value string) string {
	for _, line := range strings.Split(value, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			return line
		}
	}
	return ""
}

func FormatGPU(gpu GPU) string {
	if gpu.MemoryMB == "" {
		return gpu.Name
	}
	return fmt.Sprintf("%s (%s MB, driver %s)", gpu.Name, gpu.MemoryMB, gpu.Driver)
}
