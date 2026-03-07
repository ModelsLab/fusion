package optimize

import (
	"fmt"
	"strings"
	"time"
)

type HotspotMatch struct {
	KernelName string    `json:"kernel_name"`
	Component  string    `json:"component,omitempty"`
	Stage      string    `json:"stage,omitempty"`
	Operator   string    `json:"operator,omitempty"`
	Weight     float64   `json:"weight,omitempty"`
	Confidence float64   `json:"confidence,omitempty"`
	Evidence   []string  `json:"evidence,omitempty"`
	RecordedAt time.Time `json:"recorded_at"`
}

type HotspotAttribution struct {
	Version     int            `json:"version"`
	Task        string         `json:"task,omitempty"`
	Runtime     string         `json:"runtime,omitempty"`
	Workload    string         `json:"workload,omitempty"`
	Matches     []HotspotMatch `json:"matches,omitempty"`
	Notes       []string       `json:"notes,omitempty"`
	GeneratedAt time.Time      `json:"generated_at"`
}

func InferHotspotAttribution(task, runtime, workload string, kernelNames []string) HotspotAttribution {
	out := HotspotAttribution{
		Version:     1,
		Task:        NormalizeTask(task),
		Runtime:     strings.TrimSpace(runtime),
		Workload:    strings.TrimSpace(workload),
		Matches:     make([]HotspotMatch, 0, len(kernelNames)),
		GeneratedAt: time.Now().UTC(),
	}
	for _, kernelName := range kernelNames {
		match := inferKernelHotspot(strings.TrimSpace(kernelName))
		out.Matches = append(out.Matches, match)
	}
	if len(out.Matches) == 0 {
		out.Notes = append(out.Notes, "no kernel names were supplied for hotspot attribution")
	}
	return out
}

func SaveHotspotAttribution(session *Session, candidateID string, round int, attribution HotspotAttribution) (string, error) {
	if session == nil {
		return "", fmt.Errorf("session is required")
	}
	if strings.TrimSpace(candidateID) == "" {
		return "", fmt.Errorf("candidate id is required")
	}
	if round <= 0 {
		return "", fmt.Errorf("round must be >= 1")
	}
	return SaveRoundArtifact(session, candidateID, round, "hotspots", attribution)
}

func inferKernelHotspot(kernelName string) HotspotMatch {
	lower := strings.ToLower(strings.TrimSpace(kernelName))
	match := HotspotMatch{
		KernelName: kernelName,
		Weight:     1,
		Confidence: 0.45,
		RecordedAt: time.Now().UTC(),
	}
	addEvidence := func(values ...string) {
		match.Evidence = append(match.Evidence, values...)
	}

	switch {
	case containsKernelTerm(lower, "flash", "attn", "attention", "sdpa"):
		match.Component = "attention"
		match.Operator = "attention"
		match.Stage = "transformer"
		match.Confidence = 0.9
		addEvidence("kernel name suggests an attention path")
	case containsKernelTerm(lower, "kv", "cache"):
		match.Component = "kv-cache"
		match.Operator = "cache-transform"
		match.Stage = "decode"
		match.Confidence = 0.82
		addEvidence("kernel name suggests kv-cache activity")
	case containsKernelTerm(lower, "unet", "downblock", "upblock", "resnet"):
		match.Component = "unet"
		match.Operator = "convolution-or-attention-block"
		match.Stage = "denoiser"
		match.Confidence = 0.88
		addEvidence("kernel name matches common diffusion UNet terminology")
	case containsKernelTerm(lower, "dit", "transformer", "mma", "gemm", "matmul"):
		match.Component = "transformer"
		match.Operator = "matmul"
		match.Stage = "core-model"
		match.Confidence = 0.72
		addEvidence("kernel name matches transformer or matmul terminology")
	case containsKernelTerm(lower, "vae", "decoder", "encoder"):
		match.Component = "vae"
		match.Operator = "encode-decode"
		match.Stage = "latent-conversion"
		match.Confidence = 0.86
		addEvidence("kernel name suggests VAE encode/decode work")
	case containsKernelTerm(lower, "upsample", "upscaler", "superres"):
		match.Component = "upscaler"
		match.Operator = "upsample"
		match.Stage = "refinement"
		match.Confidence = 0.84
		addEvidence("kernel name suggests upscaling or refinement")
	case containsKernelTerm(lower, "text_encoder", "clip", "t5"):
		match.Component = "text-encoder"
		match.Operator = "embedding"
		match.Stage = "conditioning"
		match.Confidence = 0.83
		addEvidence("kernel name suggests text conditioning")
	case containsKernelTerm(lower, "scheduler", "sampler", "euler", "dpm"):
		match.Component = "scheduler"
		match.Operator = "step-update"
		match.Stage = "sampling"
		match.Confidence = 0.78
		addEvidence("kernel name suggests scheduler or sampler activity")
	default:
		match.Component = "unknown"
		match.Operator = "unknown"
		match.Stage = "unknown"
		addEvidence("kernel name did not match a known generic component pattern")
	}
	return match
}

func containsKernelTerm(lower string, terms ...string) bool {
	for _, term := range terms {
		if strings.Contains(lower, term) {
			return true
		}
	}
	return false
}
