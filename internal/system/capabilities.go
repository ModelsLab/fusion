package system

import "runtime"

type CapabilityReport struct {
	CanPlan           bool
	CanManageTargets  bool
	CanRunLocal       bool
	CanCompileCUDA    bool
	CanProfileCUDA    bool
	CanBenchmarkLocal bool
	CanUseSSH         bool
	HasNVIDIAGPU      bool
	Limitations       []string
	NextSteps         []string
}

func AssessCapabilities(env Environment) CapabilityReport {
	report := CapabilityReport{
		CanPlan:           true,
		CanManageTargets:  true,
		CanRunLocal:       true,
		CanBenchmarkLocal: true,
		CanUseSSH:         toolAvailable(env.Tools, "ssh"),
		HasNVIDIAGPU:      len(env.GPUs) > 0,
	}

	report.CanCompileCUDA = toolAvailable(env.Tools, "nvcc")
	report.CanProfileCUDA = toolAvailable(env.Tools, "ncu") || toolAvailable(env.Tools, "nsys")

	if runtime.GOOS == "darwin" {
		report.Limitations = append(report.Limitations,
			"macOS hosts do not support native NVIDIA CUDA execution in typical modern setups",
			"local Triton or CUDA kernel performance validation is not available here unless you hand work off to a remote Linux GPU target",
		)
		report.NextSteps = append(report.NextSteps,
			"use `fusion target add --mode ssh ...` to register a remote Ubuntu GPU box",
			"use `fusion optimize plan` and kernel generation locally, then benchmark/profile over SSH",
		)
	}

	if !report.HasNVIDIAGPU {
		report.Limitations = append(report.Limitations,
			"no NVIDIA GPU is currently visible on this host",
		)
		report.NextSteps = append(report.NextSteps,
			"pass `--gpu` to plan for another machine or add an SSH target",
		)
	}

	if !report.CanCompileCUDA {
		report.Limitations = append(report.Limitations,
			"`nvcc` is missing, so CUDA C++ compilation flows cannot run locally",
		)
	}

	if !report.CanProfileCUDA {
		report.Limitations = append(report.Limitations,
			"`ncu` or `nsys` is missing, so local Nsight profiling is unavailable",
		)
	}

	if !report.CanUseSSH {
		report.Limitations = append(report.Limitations,
			"`ssh` is missing, so remote target execution is unavailable from this host",
		)
	}

	if report.HasNVIDIAGPU && report.CanCompileCUDA && report.CanProfileCUDA {
		report.NextSteps = append(report.NextSteps,
			"this host is ready for local compile/profile loops",
		)
	}

	return report
}

func toolAvailable(tools []ToolStatus, name string) bool {
	for _, tool := range tools {
		if tool.Name == name && tool.Available {
			return true
		}
	}
	return false
}
