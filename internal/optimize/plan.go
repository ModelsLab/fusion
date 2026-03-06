package optimize

import (
	"fmt"
	"sort"
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

type Plan struct {
	Request                  Request
	GPU                      *kb.GPUProfile
	ResolvedGPU              string
	LikelyBottleneck         string
	BottleneckReason         string
	Priorities               []string
	MeasurementLoop          []string
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
	plan.Warnings = warnings

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
