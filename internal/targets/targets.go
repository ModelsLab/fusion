package targets

import (
	"fmt"
	"strings"

	"github.com/ModelsLab/fusion/internal/config"
	"github.com/ModelsLab/fusion/internal/kb"
)

const (
	ModeLocal = "local"
	ModeSSH   = "ssh"
	ModeSim   = "sim"
)

type Validation struct {
	Warnings []string
	Errors   []string
}

func Normalize(mode string) string {
	return strings.TrimSpace(strings.ToLower(mode))
}

func IsSupportedMode(mode string) bool {
	switch Normalize(mode) {
	case ModeLocal, ModeSSH, ModeSim:
		return true
	default:
		return false
	}
}

func Validate(target config.TargetConfig, store *kb.Store) Validation {
	mode := Normalize(target.Mode)
	result := Validation{}

	if strings.TrimSpace(target.Name) == "" {
		result.Errors = append(result.Errors, "target name is required")
	}
	if !IsSupportedMode(mode) {
		result.Errors = append(result.Errors, fmt.Sprintf("unsupported target mode %q", target.Mode))
		return result
	}

	if target.Port == 0 && mode != ModeLocal {
		target.Port = 22
	}

	if mode == ModeSSH || mode == ModeSim {
		if strings.TrimSpace(target.Host) == "" && mode == ModeSSH {
			result.Errors = append(result.Errors, "ssh targets require --host")
		}
	}

	if mode == ModeSim {
		if strings.TrimSpace(target.GPU) == "" {
			result.Errors = append(result.Errors, "sim targets require --gpu for the intended target GPU")
		}
		if strings.TrimSpace(target.ProxyGPU) == "" {
			result.Warnings = append(result.Warnings, "sim target has no --proxy-gpu; Fusion will treat measurements as generic non-authoritative proxy runs")
		}
		result.Warnings = append(result.Warnings,
			"sim mode is for compatibility, code generation, and rough iteration only; it is not a performance-faithful GPU simulator",
		)
	}

	if mode == ModeSSH && strings.TrimSpace(target.GPU) == "" {
		result.Warnings = append(result.Warnings, "ssh target has no declared --gpu; planning will fall back to generic strategies until a GPU is set")
	}

	if mode == ModeLocal && strings.TrimSpace(target.ProxyGPU) != "" {
		result.Warnings = append(result.Warnings, "local targets ignore --proxy-gpu; use mode=sim if you want non-authoritative proxy execution")
	}

	if store != nil {
		if strings.TrimSpace(target.GPU) != "" {
			if _, ok := store.GPUByID(target.GPU); !ok {
				result.Warnings = append(result.Warnings, fmt.Sprintf("target GPU %q is not in the curated knowledge base yet", target.GPU))
			}
		}
		if strings.TrimSpace(target.ProxyGPU) != "" {
			if _, ok := store.GPUByID(target.ProxyGPU); !ok {
				result.Warnings = append(result.Warnings, fmt.Sprintf("proxy GPU %q is not in the curated knowledge base yet", target.ProxyGPU))
			}
		}
	}

	return result
}

func ExecutionSummary(target config.TargetConfig) string {
	mode := Normalize(target.Mode)
	switch mode {
	case ModeLocal:
		if target.GPU != "" {
			return fmt.Sprintf("local execution on %s", target.GPU)
		}
		return "local execution"
	case ModeSSH:
		location := target.Host
		if target.User != "" {
			location = target.User + "@" + location
		}
		if target.GPU != "" {
			return fmt.Sprintf("ssh execution on %s targeting %s", location, target.GPU)
		}
		return fmt.Sprintf("ssh execution on %s", location)
	case ModeSim:
		runner := "local proxy execution"
		if target.Host != "" {
			runner = "ssh proxy execution"
		}
		if target.ProxyGPU != "" {
			return fmt.Sprintf("%s on %s while simulating %s", runner, target.ProxyGPU, target.GPU)
		}
		return fmt.Sprintf("%s while simulating %s", runner, target.GPU)
	default:
		return "unknown target mode"
	}
}

func Warnings(target config.TargetConfig) []string {
	mode := Normalize(target.Mode)
	warnings := []string{}

	switch mode {
	case ModeSim:
		warnings = append(warnings,
			"sim mode does not provide trustworthy cross-GPU performance numbers",
			"use sim mode for syntax, integration, and rough kernel iteration, then validate on the actual GPU",
		)
	case ModeSSH:
		warnings = append(warnings,
			"ssh results are authoritative only if the remote machine actually matches the declared GPU and software stack",
		)
	}

	return warnings
}
