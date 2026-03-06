package optimize

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/ModelsLab/fusion/internal/kb"
)

type Request struct {
	GPU                 string
	Model               string
	Workload            string
	Operators           []string
	Precision           string
	Bottleneck          string
	Goals               []string
	BatchSize           int
	ContextLength       int
	IncludeExperimental bool
}

type Recommendation struct {
	Strategy kb.Strategy
	Score    int
	Reasons  []string
	Sources  []kb.Source
}

type KernelBackendRecommendation struct {
	ID           string
	Name         string
	Summary      string
	SupportLevel string
	Score        int
	Reasons      []string
	Strengths    []string
	Tradeoffs    []string
	Sources      []kb.Source
}

type ModelPathRecommendation struct {
	ID           string
	Name         string
	Format       string
	Summary      string
	SupportLevel string
	Score        int
	Reasons      []string
	Actions      []string
	Tradeoffs    []string
	Sources      []kb.Source
}

type Plan struct {
	Request                  Request
	GPU                      *kb.GPUProfile
	ResolvedGPU              string
	LikelyBottleneck         string
	BottleneckReason         string
	Priorities               []string
	MeasurementLoop          []string
	ModelPaths               []ModelPathRecommendation
	KernelBackends           []KernelBackendRecommendation
	Recommendations          []Recommendation
	SupportingSources        []kb.Source
	Warnings                 []string
	PreferredPrecisionHints  []string
	ExperimentalPrecisionSet []string
}

type Planner struct {
	store *kb.Store
}

func NewPlanner(store *kb.Store) *Planner {
	return &Planner{store: store}
}

func (p *Planner) Build(input Request) (*Plan, error) {
	req := normalizeRequest(input)
	plan := &Plan{Request: req}

	var warnings []string
	if req.GPU != "" {
		gpu, ok := p.store.GPUByID(req.GPU)
		if ok {
			plan.GPU = gpu
			plan.ResolvedGPU = gpu.Name
			plan.PreferredPrecisionHints = append([]string{}, gpu.PreferredPrecisions...)
			plan.ExperimentalPrecisionSet = append([]string{}, gpu.ExperimentalPrecisons...)
		} else {
			warnings = append(warnings, fmt.Sprintf("GPU %q is not in the curated knowledge base yet; recommendations are falling back to general strategies.", req.GPU))
		}
	}

	measurement, ok := p.store.StrategyByID("baseline_profile_first")
	if ok {
		plan.MeasurementLoop = append(plan.MeasurementLoop, measurement.Actions...)
	}

	plan.LikelyBottleneck, plan.BottleneckReason = inferBottleneck(req, plan.GPU)
	plan.Priorities = derivePriorities(req, plan.GPU, plan.LikelyBottleneck)
	plan.ModelPaths = p.recommendModelPaths(req, plan.GPU, plan.LikelyBottleneck)
	plan.KernelBackends = p.recommendKernelBackends(req, plan.GPU, plan.LikelyBottleneck)
	plan.Warnings = append(plan.Warnings, warnings...)
	plan.Warnings = append(plan.Warnings, precisionSupportWarnings(plan.GPU)...)

	recommendations := []Recommendation{}
	supportingSourceIDs := []string{}
	for _, strategy := range p.store.Strategies {
		if strategy.ID == "baseline_profile_first" {
			supportingSourceIDs = append(supportingSourceIDs, strategy.SourceIDs...)
			continue
		}

		recommendation, ok := p.scoreStrategy(strategy, req, plan.GPU, plan.LikelyBottleneck)
		if !ok {
			continue
		}

		recommendation.Sources = p.store.SourcesForIDs(strategy.SourceIDs)
		supportingSourceIDs = append(supportingSourceIDs, strategy.SourceIDs...)
		recommendations = append(recommendations, recommendation)
	}

	sort.Slice(recommendations, func(i, j int) bool {
		if recommendations[i].Score == recommendations[j].Score {
			return recommendations[i].Strategy.Title < recommendations[j].Strategy.Title
		}
		return recommendations[i].Score > recommendations[j].Score
	})

	if len(recommendations) > 6 {
		recommendations = recommendations[:6]
	}

	plan.Recommendations = recommendations
	plan.SupportingSources = p.store.SourcesForIDs(supportingSourceIDs)
	return plan, nil
}

type kernelBackendSpec struct {
	ID           string
	Name         string
	Summary      string
	SupportLevel string
	Strengths    []string
	Tradeoffs    []string
	SourceIDs    []string
}

type modelPathSpec struct {
	ID           string
	Name         string
	Format       string
	Summary      string
	SupportLevel string
	Actions      []string
	Tradeoffs    []string
	SourceIDs    []string
}

func (p *Planner) recommendModelPaths(req Request, gpu *kb.GPUProfile, bottleneck string) []ModelPathRecommendation {
	specs := []modelPathSpec{
		{
			ID:           "bf16_baseline",
			Name:         "BF16 Baseline",
			Format:       "bf16",
			Summary:      "Keep a clean bf16 or fp16 reference so every lower-precision or custom-kernel path has an accuracy and speed baseline.",
			SupportLevel: "recommended",
			Actions: []string{
				"Measure bf16 or fp16 first with the exact production-like prompt mix.",
				"Capture tokens/sec, TTFT, inter-token latency, and peak memory before changing precision.",
			},
			Tradeoffs: []string{
				"Uses the most memory and may underutilize newer low-precision Tensor Core paths.",
			},
			SourceIDs: []string{"nvidia-cuda-programming-guide"},
		},
		{
			ID:           "awq_int4_weights",
			Name:         "AWQ Weight-Only INT4",
			Format:       "awq-int4",
			Summary:      "Best first quantization track when memory traffic or VRAM footprint is the dominant problem, especially on consumer GPUs.",
			SupportLevel: "recommended",
			Actions: []string{
				"Benchmark an AWQ or equivalent weight-only INT4 path before writing large custom kernels.",
				"Fuse dequantization into GEMM or attention hot paths instead of materializing higher-precision weights.",
				"Treat prefill and decode as separate tracks because the best quantization choice can differ.",
			},
			Tradeoffs: []string{
				"Quality still depends on calibration quality and runtime kernel coverage.",
				"Weight-only INT4 helps memory pressure more than raw compute throughput.",
			},
			SourceIDs: []string{"awq-activation-aware-weight-quantization", "tensorrt-llm-docs"},
		},
		{
			ID:           "fp8_serving",
			Name:         "FP8 Serving Path",
			Format:       "fp8",
			Summary:      "Preferred mature low-precision path on Hopper and Blackwell when the model and runtime tolerate FP8 calibration.",
			SupportLevel: "recommended",
			Actions: []string{
				"Validate an FP8 runtime path before hand-writing custom tensor-core kernels.",
				"Track quality drift and tensor-core utilization together.",
			},
			Tradeoffs: []string{
				"Needs calibration and selective fallbacks for sensitive model blocks.",
			},
			SourceIDs: []string{"nvidia-h100", "nvidia-blackwell-architecture", "tensorrt-llm-docs"},
		},
		{
			ID:           "nvfp4_block_scaled",
			Name:         "NVFP4 / Block-Scaled Low Precision",
			Format:       "nvfp4",
			Summary:      "High-upside experimental Blackwell track for models that are still limited by memory footprint or Tensor Core throughput after FP8.",
			SupportLevel: "experimental",
			Actions: []string{
				"Use FP8 as the control path before evaluating NVFP4 or block-scaled FP4.",
				"Gate rollout on calibration drift and real end-to-end serving metrics, not only GEMM microbenchmarks.",
			},
			Tradeoffs: []string{
				"Tooling and runtime coverage are still evolving.",
				"Portability outside Blackwell stacks is weak.",
			},
			SourceIDs: []string{"nvidia-blackwell-architecture", "triton-block-scaled-matmul"},
		},
		{
			ID:           "kv_cache_quantized",
			Name:         "KV Cache Quantization",
			Format:       "fp8/int8/int4-kv",
			Summary:      "For long-context decode and serving, KV compression can beat another round of GEMM tuning if memory movement dominates.",
			SupportLevel: "experimental",
			Actions: []string{
				"Benchmark KV compression separately from weight quantization.",
				"Track memory footprint, throughput, and quality together on representative long-context prompts.",
			},
			Tradeoffs: []string{
				"Adds cache-management complexity and quality sensitivity varies by model and workload.",
			},
			SourceIDs: []string{"flashinfer-docs", "xkv-paper", "titanus-paper"},
		},
	}

	recommendations := make([]ModelPathRecommendation, 0, len(specs))
	for _, spec := range specs {
		score, reasons := scoreModelPath(spec, req, gpu, bottleneck)
		recommendations = append(recommendations, ModelPathRecommendation{
			ID:           spec.ID,
			Name:         spec.Name,
			Format:       spec.Format,
			Summary:      spec.Summary,
			SupportLevel: spec.SupportLevel,
			Score:        score,
			Reasons:      reasons,
			Actions:      append([]string{}, spec.Actions...),
			Tradeoffs:    append([]string{}, spec.Tradeoffs...),
			Sources:      p.store.SourcesForIDs(spec.SourceIDs),
		})
	}

	sort.Slice(recommendations, func(i, j int) bool {
		if recommendations[i].Score == recommendations[j].Score {
			return recommendations[i].Name < recommendations[j].Name
		}
		return recommendations[i].Score > recommendations[j].Score
	})

	if len(recommendations) > 4 {
		recommendations = recommendations[:4]
	}

	return recommendations
}

func (p *Planner) recommendKernelBackends(req Request, gpu *kb.GPUProfile, bottleneck string) []KernelBackendRecommendation {
	specs := []kernelBackendSpec{
		{
			ID:           "cute_dsl",
			Name:         "CuTe DSL",
			Summary:      "Best fit for NVIDIA-first kernels when you want CUTLASS-grade building blocks, JIT/AOT options, and direct Tensor Core control.",
			SupportLevel: "recommended",
			Strengths: []string{
				"Best path for NVIDIA-specific matrix, attention, and epilogue kernels.",
				"Supports JIT plus AOT-style export flows for production integration.",
				"Maps closely to CUTLASS/CuTe hardware concepts on Ampere, Hopper, and Blackwell.",
			},
			Tradeoffs: []string{
				"Python authoring still assumes a Linux NVIDIA runtime for real validation.",
				"Lower-level than Triton, so simple fused elementwise kernels are usually faster to prototype elsewhere.",
			},
			SourceIDs: []string{"nvidia-cutlass-overview", "nvidia-cute-dsl"},
		},
		{
			ID:           "triton",
			Name:         "Triton",
			Summary:      "Strong default for fast iteration on fused pointwise, reduction, dequantization, and custom attention-adjacent kernels.",
			SupportLevel: "recommended",
			Strengths: []string{
				"Fastest path for trying many fusion ideas in Python.",
				"Good fit for memory-bound decode kernels and custom data-layout transforms.",
				"Large public ecosystem of examples for GEMM, attention, and fused ops.",
			},
			Tradeoffs: []string{
				"Peak kernels sometimes still require dropping to CuTe or CUDA C++.",
				"Production packaging and low-level debugging can be rougher than CUDA-native flows.",
			},
			SourceIDs: []string{"triton-tutorials"},
		},
		{
			ID:           "cuda_cutlass_cpp",
			Name:         "CUDA C++ / CUTLASS",
			Summary:      "Use when you need full control over launch structure, runtime integration, or a kernel shape that DSLs do not express well yet.",
			SupportLevel: "recommended",
			Strengths: []string{
				"Most control over code generation, memory movement, and runtime integration.",
				"Easiest backend to embed in mature C++ inference runtimes.",
				"Still the escape hatch for kernels that need explicit low-level control.",
			},
			Tradeoffs: []string{
				"Longest iteration cycle and the highest implementation cost.",
				"Template-heavy CUTLASS code can be harder to search, repair, and auto-generate than Python DSLs.",
			},
			SourceIDs: []string{"nvidia-cuda-programming-guide", "nvidia-cutlass-overview"},
		},
	}

	recommendations := make([]KernelBackendRecommendation, 0, len(specs))
	for _, spec := range specs {
		score, reasons := scoreKernelBackend(spec, req, gpu, bottleneck)
		recommendations = append(recommendations, KernelBackendRecommendation{
			ID:           spec.ID,
			Name:         spec.Name,
			Summary:      spec.Summary,
			SupportLevel: spec.SupportLevel,
			Score:        score,
			Reasons:      reasons,
			Strengths:    append([]string{}, spec.Strengths...),
			Tradeoffs:    append([]string{}, spec.Tradeoffs...),
			Sources:      p.store.SourcesForIDs(spec.SourceIDs),
		})
	}

	sort.Slice(recommendations, func(i, j int) bool {
		if recommendations[i].Score == recommendations[j].Score {
			return recommendations[i].Name < recommendations[j].Name
		}
		return recommendations[i].Score > recommendations[j].Score
	})

	return recommendations
}

func scoreModelPath(spec modelPathSpec, req Request, gpu *kb.GPUProfile, bottleneck string) (int, []string) {
	score := 35
	reasons := []string{}
	modelParamsB := parseModelParamsBillions(req.Model)
	bf16FootprintGB := estimateWeightFootprintGB(modelParamsB, "bf16")

	switch spec.ID {
	case "bf16_baseline":
		score += 18
		reasons = append(reasons, "keeps a reliable reference path before quantization or custom kernels")
		if req.Workload == "prefill" || req.Workload == "decode" || req.Workload == "serving" {
			score += 6
			reasons = append(reasons, fmt.Sprintf("gives a clean baseline for %s benchmarking", req.Workload))
		}
	case "awq_int4_weights":
		if bottleneck == "memory" || bottleneck == "mixed" {
			score += 24
			reasons = append(reasons, fmt.Sprintf("best early track for a %s bottleneck", bottleneck))
		}
		if gpu != nil && strings.EqualFold(gpu.Market, "consumer") {
			score += 16
			reasons = append(reasons, "consumer GPUs usually benefit quickly from weight-only INT4 paths")
		}
		if containsAny(req.Goals, "memory", "cost", "throughput") {
			score += 8
			reasons = append(reasons, "aligns with memory and serving cost goals")
		}
		if req.Workload == "decode" || req.Workload == "serving" {
			score += 8
			reasons = append(reasons, fmt.Sprintf("%s often improves once weight traffic drops", req.Workload))
		}
		if gpu != nil && bf16FootprintGB > 0 && bf16FootprintGB > float64(gpu.MemoryGB)*0.85 {
			score += 16
			reasons = append(reasons, fmt.Sprintf("the model looks too large for a clean bf16 fit on %s", gpu.Name))
		}
	case "fp8_serving":
		if gpu != nil && matchesBucket(gpu.Family, []string{"Hopper", "Blackwell"}) {
			score += 28
			reasons = append(reasons, fmt.Sprintf("%s is the best GPU family for mature FP8 paths", gpu.Family))
		} else if gpu != nil {
			score -= 24
			reasons = append(reasons, fmt.Sprintf("%s does not provide a first-class native FP8 tensor-core serving path", gpu.Name))
		}
		if bottleneck == "compute" {
			score += 14
			reasons = append(reasons, "compute-bound workloads usually test FP8 before custom matmul kernels")
		}
		if containsAny(req.Goals, "throughput", "cost") {
			score += 10
			reasons = append(reasons, "FP8 often improves both throughput and cost per token")
		}
		if gpu != nil && modelParamsB > 0 && estimateWeightFootprintGB(modelParamsB, "fp8") <= float64(gpu.MemoryGB) {
			score += 8
			reasons = append(reasons, "FP8 may fit the model cleanly where bf16 does not")
		}
	case "nvfp4_block_scaled":
		if gpu != nil && matchesBucket(gpu.Family, []string{"Blackwell"}) {
			score += 34
			reasons = append(reasons, "Blackwell is the first real home for NVFP4-style serving experiments")
		} else if gpu != nil {
			score -= 32
			reasons = append(reasons, fmt.Sprintf("NVFP4 is not a native path on %s", gpu.Name))
		}
		if bottleneck == "compute" || bottleneck == "memory" {
			score += 10
			reasons = append(reasons, fmt.Sprintf("can matter after FP8 when %s pressure remains", bottleneck))
		}
		if containsAny(req.Goals, "throughput", "memory", "cost") {
			score += 8
			reasons = append(reasons, "targets aggressive low-byte serving paths")
		}
	case "kv_cache_quantized":
		if req.Workload == "decode" || req.Workload == "serving" {
			score += 24
			reasons = append(reasons, fmt.Sprintf("%s workloads are often limited by KV movement", req.Workload))
		}
		if bottleneck == "memory" {
			score += 18
			reasons = append(reasons, "memory-bound decode usually needs KV decisions, not only GEMM tuning")
		}
		if containsAny(req.Operators, "kv-cache", "attention", "paged-attention") {
			score += 14
			reasons = append(reasons, "operators point directly at KV-heavy attention paths")
		}
		if req.ContextLength >= 16000 {
			score += 10
			reasons = append(reasons, fmt.Sprintf("context length %d makes KV footprint more important", req.ContextLength))
		}
	}

	if len(reasons) == 0 {
		reasons = append(reasons, spec.Summary)
	}
	return score, reasons
}

func precisionSupportWarnings(gpu *kb.GPUProfile) []string {
	if gpu == nil {
		return nil
	}

	family := canonicalGPUFamily(gpu.Family)
	warnings := []string{}

	if family != "hopper" && family != "blackwell" {
		warnings = append(warnings, fmt.Sprintf("%s is not a Hopper or Blackwell GPU, so native FP8 serving should not be treated as a first-line optimization path here.", gpu.Name))
	}
	if family != "blackwell" {
		warnings = append(warnings, fmt.Sprintf("%s is not a Blackwell GPU, so NVFP4 or block-scaled FP4 should be treated as unsupported for production optimization on this target.", gpu.Name))
	}

	return warnings
}

func scoreKernelBackend(spec kernelBackendSpec, req Request, gpu *kb.GPUProfile, bottleneck string) (int, []string) {
	score := 40
	reasons := []string{}

	if gpu != nil {
		switch spec.ID {
		case "cute_dsl":
			switch canonicalGPUFamily(gpu.Family) {
			case "blackwell", "hopper":
				score += 28
				reasons = append(reasons, fmt.Sprintf("best aligned with %s Tensor Core generations", gpu.Family))
			case "ada", "ampere":
				score += 20
				reasons = append(reasons, fmt.Sprintf("fits %s NVIDIA tensor-core kernels well", gpu.Family))
			default:
				score += 10
			}
		case "triton":
			score += 16
			reasons = append(reasons, "good default for rapid NVIDIA kernel iteration")
		case "cuda_cutlass_cpp":
			score += 14
			reasons = append(reasons, "always available as the lowest-level fallback")
		}
	}

	switch spec.ID {
	case "cute_dsl":
		if containsAny(req.Operators, "matmul", "gemm", "attention", "paged-attention", "moe") {
			score += 22
			reasons = append(reasons, "well suited for tensor-core-heavy kernels")
		}
		if containsAny(req.Goals, "throughput", "latency") {
			score += 10
			reasons = append(reasons, "fits production-oriented performance work")
		}
		if matchesBucket(req.Precision, []string{"bf16", "fp16", "fp8", "fp4", "int8", "int4"}) {
			score += 10
			reasons = append(reasons, fmt.Sprintf("supports low-precision NVIDIA paths like %s", req.Precision))
		}
	case "triton":
		if containsAny(req.Operators, "layernorm", "rmsnorm", "silu", "gelu", "dequantization", "attention", "kv-cache", "softmax") {
			score += 22
			reasons = append(reasons, "strong for fused memory-bound Python-authored kernels")
		}
		if bottleneck == "memory" || bottleneck == "mixed" {
			score += 10
			reasons = append(reasons, fmt.Sprintf("good fit for %s bottlenecks and fast iteration loops", bottleneck))
		}
		if req.Workload == "decode" || req.Workload == "serving" {
			score += 8
			reasons = append(reasons, fmt.Sprintf("common choice for %s kernel experiments", req.Workload))
		}
	case "cuda_cutlass_cpp":
		if req.Workload == "decode" || bottleneck == "latency" {
			score += 12
			reasons = append(reasons, "useful when launch structure and runtime integration matter more than authoring speed")
		}
		if containsAny(req.Goals, "throughput", "latency") {
			score += 8
			reasons = append(reasons, "good escape hatch when the winning kernel needs full low-level control")
		}
		if containsAny(req.Operators, "attention", "paged-attention", "collective", "matmul", "gemm") {
			score += 8
			reasons = append(reasons, "handles specialized kernels that outgrow higher-level DSLs")
		}
	}

	if len(reasons) == 0 {
		reasons = append(reasons, spec.Summary)
	}

	return score, reasons
}

func canonicalGPUFamily(value string) string {
	return strings.TrimSpace(strings.ToLower(value))
}

func parseModelParamsBillions(model string) float64 {
	model = canonicalText(model)
	if model == "" {
		return 0
	}

	re := regexp.MustCompile(`(\d+(?:\.\d+)?)\s*b`)
	match := re.FindStringSubmatch(model)
	if len(match) < 2 {
		return 0
	}
	value, err := strconv.ParseFloat(match[1], 64)
	if err != nil {
		return 0
	}
	return value
}

func estimateWeightFootprintGB(paramsB float64, precision string) float64 {
	if paramsB <= 0 {
		return 0
	}
	bytesPerParam := map[string]float64{
		"bf16":     2.0,
		"fp16":     2.0,
		"fp8":      1.0,
		"int8":     1.0,
		"int4":     0.5,
		"awq-int4": 0.5,
		"nvfp4":    0.5,
	}
	return paramsB * bytesPerParam[canonicalText(precision)]
}

func canonicalText(value string) string {
	return strings.TrimSpace(strings.ToLower(value))
}

func (p *Planner) scoreStrategy(strategy kb.Strategy, req Request, gpu *kb.GPUProfile, bottleneck string) (Recommendation, bool) {
	if !req.IncludeExperimental && strings.EqualFold(strategy.SupportLevel, "experimental") {
		return Recommendation{}, false
	}

	score := strategy.Priority
	reasons := []string{}

	if !matchesBucket(req.Workload, strategy.Workloads) {
		if len(strategy.Workloads) > 0 {
			return Recommendation{}, false
		}
	} else if req.Workload != "" {
		score += 18
		reasons = append(reasons, fmt.Sprintf("targets %s workloads", req.Workload))
	}

	operatorOverlap := overlap(req.Operators, strategy.Operators)
	if len(req.Operators) > 0 && len(strategy.Operators) > 0 && !containsWildcard(strategy.Operators) {
		if len(operatorOverlap) == 0 && strategy.Category != "workflow" {
			return Recommendation{}, false
		}
	}
	if len(operatorOverlap) > 0 {
		score += len(operatorOverlap) * 10
		reasons = append(reasons, fmt.Sprintf("covers operators: %s", strings.Join(operatorOverlap, ", ")))
	}

	if gpu != nil {
		if len(strategy.GPUIDs) > 0 || len(strategy.GPUFamilies) > 0 {
			switch {
			case matchesBucket(gpu.ID, strategy.GPUIDs):
				score += 18
				reasons = append(reasons, fmt.Sprintf("directly tuned for %s", gpu.Name))
			case matchesBucket(gpu.Family, strategy.GPUFamilies):
				score += 12
				reasons = append(reasons, fmt.Sprintf("fits the %s GPU family", gpu.Family))
			default:
				return Recommendation{}, false
			}
		}
	}

	if req.Precision != "" && len(strategy.Precision) > 0 && !containsWildcard(strategy.Precision) {
		if !matchesBucket(req.Precision, strategy.Precision) {
			if strategy.Category != "workflow" {
				return Recommendation{}, false
			}
		} else {
			score += 10
			reasons = append(reasons, fmt.Sprintf("applies to %s precision paths", req.Precision))
		}
	}

	if bottleneck != "" && len(strategy.Bottlenecks) > 0 && !containsWildcard(strategy.Bottlenecks) {
		if matchesBucket(bottleneck, strategy.Bottlenecks) {
			score += 12
			reasons = append(reasons, fmt.Sprintf("addresses a %s bottleneck", bottleneck))
		}
	}

	goalOverlap := overlap(req.Goals, strategy.Goals)
	if len(goalOverlap) > 0 {
		score += len(goalOverlap) * 6
		reasons = append(reasons, fmt.Sprintf("advances goals: %s", strings.Join(goalOverlap, ", ")))
	}

	if len(reasons) == 0 {
		reasons = append(reasons, strategy.Summary)
	}

	return Recommendation{
		Strategy: strategy,
		Score:    score,
		Reasons:  reasons,
	}, true
}

func inferBottleneck(req Request, gpu *kb.GPUProfile) (string, string) {
	if req.Bottleneck != "" {
		return req.Bottleneck, "bottleneck supplied on the command line"
	}

	if req.Workload == "decode" {
		if containsAny(req.Operators, "kv-cache", "attention", "paged-attention") {
			return "memory", "decode workloads are usually dominated by KV-cache traffic and memory movement"
		}
		return "latency", "steady-state decode is usually dominated by launch overhead and per-token latency"
	}

	if req.Workload == "prefill" {
		if req.BatchSize >= 8 || containsAny(req.Operators, "matmul", "gemm", "moe") {
			return "compute", "prefill usually leans compute-bound when GEMMs dominate the step"
		}
		return "memory", "prefill with long sequences often turns into an HBM bandwidth problem"
	}

	if req.Workload == "serving" && containsAny(req.Operators, "attention", "kv-cache") {
		return "memory", "multi-request serving usually amplifies KV-cache pressure"
	}

	if gpu != nil && strings.EqualFold(gpu.Market, "consumer") && containsAny(req.Operators, "quantization", "matmul", "gemm") {
		return "memory", "consumer GPUs often benefit first from lower-byte paths and fused dequant kernels"
	}

	return "mixed", "no strong workload prior was available, so Fusion is mixing memory and compute strategies"
}

func derivePriorities(req Request, gpu *kb.GPUProfile, bottleneck string) []string {
	priorities := []string{}

	switch bottleneck {
	case "memory":
		priorities = append(priorities,
			"Cut HBM and KV-cache traffic before chasing more math throughput.",
			"Prefer fused kernels that remove intermediate writes and reads.",
		)
	case "compute":
		priorities = append(priorities,
			"Maximize tensor core utilization before adding more kernel variants.",
			"Favor GEMM tiling, grouped GEMM, and low-precision tensor-core paths.",
		)
	case "latency":
		priorities = append(priorities,
			"Reduce launch overhead and stabilize shapes for steady-state decode.",
			"Capture repeatable decode steps with CUDA graphs when the runtime allows it.",
		)
	default:
		priorities = append(priorities,
			"Measure first and only keep strategies that move actual end-to-end tokens/sec or latency.",
		)
	}

	if gpu != nil {
		if strings.EqualFold(gpu.Market, "consumer") {
			priorities = append(priorities, "Consumer cards usually reward weight/KV quantization and fused dequant GEMMs early.")
		}
		if containsAny(gpu.PreferredPrecisions, "fp8") {
			priorities = append(priorities, "Validate FP8 paths before writing custom kernels if the model tolerates it.")
		}
	}

	if req.Workload == "decode" {
		priorities = append(priorities, "Treat paged attention, KV-cache layout, and graph capture as the first serving levers.")
	}

	return dedupe(priorities)
}

func normalizeRequest(req Request) Request {
	req.GPU = strings.TrimSpace(req.GPU)
	req.Model = strings.TrimSpace(req.Model)
	req.Workload = firstOrEmpty(normalizeList([]string{req.Workload}))
	req.Precision = firstOrEmpty(normalizeList([]string{req.Precision}))
	req.Bottleneck = firstOrEmpty(normalizeList([]string{req.Bottleneck}))
	req.Operators = normalizeList(req.Operators)
	req.Goals = normalizeList(req.Goals)

	if req.Workload == "" {
		req.Workload = "decode"
	}
	if req.Precision == "" {
		req.Precision = "bf16"
	}
	if len(req.Goals) == 0 {
		switch req.Workload {
		case "decode":
			req.Goals = []string{"latency", "throughput", "memory"}
		case "prefill":
			req.Goals = []string{"throughput", "compute"}
		default:
			req.Goals = []string{"throughput", "latency"}
		}
	}

	return req
}

func firstOrEmpty(values []string) string {
	if len(values) == 0 {
		return ""
	}
	return values[0]
}

func normalizeList(values []string) []string {
	out := make([]string, 0, len(values))
	seen := map[string]struct{}{}
	for _, value := range values {
		value = strings.TrimSpace(strings.ToLower(value))
		if value == "" {
			continue
		}
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}

func matchesBucket(target string, values []string) bool {
	if target == "" {
		return false
	}
	target = strings.TrimSpace(strings.ToLower(target))
	for _, value := range values {
		value = strings.TrimSpace(strings.ToLower(value))
		if value == "" {
			continue
		}
		if value == "all" || value == "*" || value == target {
			return true
		}
	}
	return false
}

func containsWildcard(values []string) bool {
	return matchesBucket("all", values)
}

func overlap(left, right []string) []string {
	set := map[string]struct{}{}
	for _, item := range normalizeList(left) {
		set[item] = struct{}{}
	}

	out := []string{}
	for _, item := range normalizeList(right) {
		if item == "all" || item == "*" {
			continue
		}
		if _, exists := set[item]; exists {
			out = append(out, item)
		}
	}

	return dedupe(out)
}

func containsAny(values []string, candidates ...string) bool {
	valueSet := map[string]struct{}{}
	for _, value := range normalizeList(values) {
		valueSet[value] = struct{}{}
	}
	for _, candidate := range normalizeList(candidates) {
		if _, exists := valueSet[candidate]; exists {
			return true
		}
	}
	return false
}

func dedupe(values []string) []string {
	out := []string{}
	seen := map[string]struct{}{}
	for _, value := range values {
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}
