package system

import (
	"fmt"
	"strings"
)

type DoctorCheck struct {
	ID          string `json:"id"`
	Kind        string `json:"kind"`
	Requirement string `json:"requirement"`
	Severity    string `json:"severity"`
	OK          bool   `json:"ok"`
	Details     string `json:"details,omitempty"`
	FixHint     string `json:"fix_hint,omitempty"`
}

type DoctorReport struct {
	Backend              string        `json:"backend"`
	Host                 Host          `json:"host"`
	Ready                bool          `json:"ready"`
	Checks               []DoctorCheck `json:"checks"`
	MissingRequired      []string      `json:"missing_required,omitempty"`
	MissingRecommended   []string      `json:"missing_recommended,omitempty"`
	RecommendedFixScript string        `json:"recommended_fix_script,omitempty"`
}

type doctorRequirement struct {
	id          string
	kind        string
	severity    string
	tool        string
	anyTools    []string
	module      string
	requirement string
	details     string
	fixHint     string
}

func RunDoctor(backend string) DoctorReport {
	normalizedBackend := normalizeDoctorBackend(backend)
	env := DetectEnvironment()

	report := DoctorReport{
		Backend: normalizedBackend,
		Host:    env.Host,
		Ready:   true,
	}

	for _, requirement := range doctorRequirements(normalizedBackend) {
		check := evaluateDoctorRequirement(env, requirement)
		report.Checks = append(report.Checks, check)
		if check.OK {
			continue
		}
		if check.Severity == "required" {
			report.Ready = false
			report.MissingRequired = append(report.MissingRequired, check.Requirement)
		} else {
			report.MissingRecommended = append(report.MissingRecommended, check.Requirement)
		}
	}

	report.RecommendedFixScript = RecommendedFixScript(report)
	return report
}

func RecommendedFixScript(report DoctorReport) string {
	lines := []string{
		"#!/usr/bin/env bash",
		"set -euo pipefail",
		"",
		"# Best-effort Fusion environment bootstrap for Ubuntu-like hosts.",
		"# Review this before running it on a production GPU machine.",
		"",
	}

	needAptUpdate := false
	aptPackages := []string{}
	pipPackages := []string{}
	notes := []string{}

	for _, check := range report.Checks {
		if check.OK {
			continue
		}
		switch check.ID {
		case "python3":
			needAptUpdate = true
			aptPackages = append(aptPackages, "python3", "python3-venv", "python3-pip")
		case "git":
			needAptUpdate = true
			aptPackages = append(aptPackages, "git")
		case "ssh":
			needAptUpdate = true
			aptPackages = append(aptPackages, "openssh-client")
		case "scp":
			needAptUpdate = true
			aptPackages = append(aptPackages, "openssh-client")
		case "rg":
			needAptUpdate = true
			aptPackages = append(aptPackages, "ripgrep")
		case "uv":
			pipPackages = append(pipPackages, "uv")
		case "module:torch":
			pipPackages = append(pipPackages, "torch")
		case "module:triton":
			pipPackages = append(pipPackages, "triton")
		case "module:cutlass.cute":
			pipPackages = append(pipPackages, "nvidia-cutlass-dsl", "apache-tvm-ffi", "torch-c-dlpack-ext")
		case "nvcc", "nvidia-smi", "cuda-gpu":
			notes = append(notes, "Install an NVIDIA driver and CUDA toolkit that expose nvidia-smi and nvcc before running Fusion CUDA/Triton/CuTe loops.")
		case "nsight":
			notes = append(notes, "Install Nsight Compute or Nsight Systems from the CUDA toolkit if you want Fusion profiling and bottleneck analysis.")
		}
	}

	aptPackages = uniqueStrings(aptPackages)
	pipPackages = uniqueStrings(pipPackages)
	notes = uniqueStrings(notes)

	if needAptUpdate && len(aptPackages) > 0 {
		lines = append(lines,
			"sudo apt-get update",
			"sudo apt-get install -y "+strings.Join(aptPackages, " "),
			"",
		)
	}

	if len(pipPackages) > 0 {
		lines = append(lines,
			"python3 -m pip install --upgrade pip",
			"python3 -m pip install --upgrade "+strings.Join(pipPackages, " "),
			"",
		)
	}

	for _, note := range notes {
		lines = append(lines, "echo "+shellQuote(note))
	}

	return strings.TrimSpace(strings.Join(lines, "\n")) + "\n"
}

func doctorRequirements(backend string) []doctorRequirement {
	requirements := []doctorRequirement{
		{
			id:          "python3",
			kind:        "tool",
			severity:    "required",
			tool:        "python3",
			requirement: "python3",
			details:     "required for Triton, CuTe DSL, and helper scripts",
			fixHint:     "install Python 3 and pip",
		},
	}

	switch backend {
	case "chat":
		requirements = append(requirements,
			doctorRequirement{
				id:          "git",
				kind:        "tool",
				severity:    "recommended",
				tool:        "git",
				requirement: "git",
				details:     "useful for project inspection and patch review inside chat workflows",
				fixHint:     "install git",
			},
			doctorRequirement{
				id:          "rg",
				kind:        "tool",
				severity:    "recommended",
				tool:        "rg",
				requirement: "ripgrep",
				details:     "Fusion falls back to grep, but ripgrep keeps agent file search fast",
				fixHint:     "install ripgrep",
			},
		)
	case "cuda":
		requirements = append(requirements,
			cudaEnvironmentRequirements()...,
		)
	case "triton":
		requirements = append(requirements, gpuExecutionRequirements()...)
		requirements = append(requirements,
			doctorRequirement{
				id:          "module:torch",
				kind:        "python-module",
				severity:    "required",
				module:      "torch",
				requirement: "python module torch",
				details:     "required by Triton build, verify, and benchmark scripts",
				fixHint:     "install PyTorch with CUDA support",
			},
			doctorRequirement{
				id:          "module:triton",
				kind:        "python-module",
				severity:    "required",
				module:      "triton",
				requirement: "python module triton",
				details:     "required by Triton workspace kernels",
				fixHint:     "install the Triton Python package",
			},
			doctorRequirement{
				id:          "uv",
				kind:        "tool",
				severity:    "recommended",
				tool:        "uv",
				requirement: "uv",
				details:     "recommended for isolated Python environments on the target host",
				fixHint:     "install uv",
			},
		)
	case "cute":
		requirements = append(requirements, gpuExecutionRequirements()...)
		requirements = append(requirements,
			doctorRequirement{
				id:          "module:torch",
				kind:        "python-module",
				severity:    "required",
				module:      "torch",
				requirement: "python module torch",
				details:     "required by CuTe verification and benchmark flows",
				fixHint:     "install PyTorch with CUDA support",
			},
			doctorRequirement{
				id:          "module:cutlass.cute",
				kind:        "python-module",
				severity:    "required",
				module:      "cutlass.cute",
				requirement: "python module cutlass.cute",
				details:     "required by CuTe DSL code generation and compilation",
				fixHint:     "install nvidia-cutlass-dsl and related CuTe Python packages",
			},
			doctorRequirement{
				id:          "uv",
				kind:        "tool",
				severity:    "recommended",
				tool:        "uv",
				requirement: "uv",
				details:     "recommended for isolated Python environments on the target host",
				fixHint:     "install uv",
			},
		)
	case "profile":
		requirements = append(requirements, gpuExecutionRequirements()...)
		requirements = append(requirements,
			doctorRequirement{
				id:          "nsight",
				kind:        "tool-any",
				severity:    "required",
				anyTools:    []string{"ncu", "nsys"},
				requirement: "ncu or nsys",
				details:     "required for Nsight-based profiling workflows",
				fixHint:     "install Nsight Compute or Nsight Systems from the CUDA toolkit",
			},
		)
	default:
		requirements = append(requirements, cudaEnvironmentRequirements()...)
		requirements = append(requirements, doctorRequirements("triton")[1:]...)
		requirements = append(requirements, doctorRequirements("cute")[1:]...)
		requirements = append(requirements, doctorRequirements("profile")[1:]...)
		requirements = append(requirements, doctorRequirements("chat")[1:]...)
	}

	return uniqueRequirements(requirements)
}

func cudaEnvironmentRequirements() []doctorRequirement {
	return append(gpuExecutionRequirements(),
		doctorRequirement{
			id:          "nvcc",
			kind:        "tool",
			severity:    "required",
			tool:        "nvcc",
			requirement: "nvcc",
			details:     "required for CUDA C++ compilation flows",
			fixHint:     "install the CUDA toolkit that matches the target driver",
		},
	)
}

func gpuExecutionRequirements() []doctorRequirement {
	return []doctorRequirement{
		{
			id:          "nvidia-smi",
			kind:        "tool",
			severity:    "required",
			tool:        "nvidia-smi",
			requirement: "nvidia-smi",
			details:     "required to confirm that NVIDIA drivers are installed",
			fixHint:     "install an NVIDIA driver on the target host",
		},
		{
			id:          "cuda-gpu",
			kind:        "gpu",
			severity:    "required",
			requirement: "visible NVIDIA GPU",
			details:     "required for authoritative Triton, CuTe DSL, and CUDA kernel execution",
			fixHint:     "run on a host with a visible NVIDIA GPU or configure a remote GPU target",
		},
	}
}

func evaluateDoctorRequirement(env Environment, requirement doctorRequirement) DoctorCheck {
	check := DoctorCheck{
		ID:          requirement.id,
		Kind:        requirement.kind,
		Requirement: requirement.requirement,
		Severity:    requirement.severity,
		FixHint:     requirement.fixHint,
	}

	switch requirement.kind {
	case "tool":
		status, ok := toolStatusByName(env.Tools, requirement.tool)
		check.OK = ok && status.Available
		if check.OK {
			check.Details = fmt.Sprintf("%s [%s]", firstNonEmptyLine(status.Version), status.Path)
		} else {
			check.Details = requirement.details
		}
	case "tool-any":
		available := []string{}
		for _, tool := range requirement.anyTools {
			status, ok := toolStatusByName(env.Tools, tool)
			if ok && status.Available {
				available = append(available, tool)
			}
		}
		check.OK = len(available) > 0
		if check.OK {
			check.Details = "available: " + strings.Join(available, ", ")
		} else {
			check.Details = requirement.details
		}
	case "gpu":
		check.OK = len(env.GPUs) > 0
		if check.OK {
			check.Details = env.GPUs[0].Name
		} else {
			check.Details = requirement.details
		}
	case "python-module":
		if !toolAvailable(env.Tools, "python3") {
			check.OK = false
			check.Details = "python3 is missing, so module checks cannot run"
			break
		}
		check.OK = pythonModuleAvailable(requirement.module)
		if check.OK {
			check.Details = "importable via python3"
		} else {
			check.Details = requirement.details
		}
	default:
		check.OK = false
		check.Details = "unsupported doctor requirement kind"
	}

	return check
}

func normalizeDoctorBackend(backend string) string {
	switch strings.TrimSpace(strings.ToLower(backend)) {
	case "", "all":
		return "all"
	case "chat":
		return "chat"
	case "cuda":
		return "cuda"
	case "triton":
		return "triton"
	case "cute", "cute_dsl", "cute-dsl":
		return "cute"
	case "profile", "nsight":
		return "profile"
	default:
		return "all"
	}
}

func pythonModuleAvailable(module string) bool {
	script := "import importlib,sys\n" +
		"module = " + fmt.Sprintf("%q", module) + "\n" +
		"try:\n" +
		"    importlib.import_module(module)\n" +
		"except Exception:\n" +
		"    sys.exit(1)\n" +
		"sys.exit(0)\n"
	output := runCommand("python3", "-c", script)
	return strings.TrimSpace(output) == ""
}

func toolStatusByName(tools []ToolStatus, name string) (ToolStatus, bool) {
	for _, tool := range tools {
		if tool.Name == name {
			return tool, true
		}
	}
	return ToolStatus{}, false
}

func uniqueRequirements(values []doctorRequirement) []doctorRequirement {
	seen := map[string]struct{}{}
	out := make([]doctorRequirement, 0, len(values))
	for _, value := range values {
		if _, ok := seen[value.id]; ok {
			continue
		}
		seen[value.id] = struct{}{}
		out = append(out, value)
	}
	return out
}

func uniqueStrings(values []string) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}

func shellQuote(value string) string {
	if value == "" {
		return "''"
	}
	return "'" + strings.ReplaceAll(value, "'", `'"'"'`) + "'"
}
