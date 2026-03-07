package optimize

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

type NsightProfile struct {
	Version     int                `json:"version"`
	Tool        string             `json:"tool"`
	KernelName  string             `json:"kernel_name,omitempty"`
	Metrics     map[string]float64 `json:"metrics,omitempty"`
	RawMetrics  map[string]float64 `json:"raw_metrics,omitempty"`
	Notes       []string           `json:"notes,omitempty"`
	GeneratedAt time.Time          `json:"generated_at"`
}

type BottleneckReport struct {
	Version         int                `json:"version"`
	Category        string             `json:"category"`
	Efficiency      float64            `json:"efficiency"`
	EvidenceMetrics map[string]float64 `json:"evidence_metrics,omitempty"`
	RootCauses      []string           `json:"root_causes,omitempty"`
	Confidence      float64            `json:"confidence"`
	Summary         string             `json:"summary"`
	GeneratedAt     time.Time          `json:"generated_at"`
}

type Prescription struct {
	Version                 int               `json:"version"`
	Category                string            `json:"category"`
	Summary                 string            `json:"summary"`
	Fixes                   []PrescriptionFix `json:"fixes,omitempty"`
	AffectedKnobs           []string          `json:"affected_knobs,omitempty"`
	ExpectedHardwareEffects []string          `json:"expected_hardware_effects,omitempty"`
	Risk                    string            `json:"risk,omitempty"`
	GeneratedAt             time.Time         `json:"generated_at"`
}

type PrescriptionFix struct {
	Action                 string   `json:"action"`
	AffectedKnobs          []string `json:"affected_knobs,omitempty"`
	ExpectedHardwareEffect string   `json:"expected_hardware_effect,omitempty"`
	Risk                   string   `json:"risk,omitempty"`
}

type Reflexion struct {
	Version       int       `json:"version"`
	Round         int       `json:"round"`
	Hypothesis    string    `json:"hypothesis,omitempty"`
	Outcome       string    `json:"outcome,omitempty"`
	Effective     bool      `json:"effective"`
	Lessons       []string  `json:"lessons,omitempty"`
	AvoidPatterns []string  `json:"avoid_patterns,omitempty"`
	TryNext       []string  `json:"try_next,omitempty"`
	RecordedAt    time.Time `json:"recorded_at"`
}

type BenchmarkProtocol struct {
	WarmupRuns        int      `json:"warmup_runs"`
	MeasuredRuns      int      `json:"measured_runs"`
	VarianceThreshold float64  `json:"variance_threshold"`
	NormalizedMetrics []string `json:"normalized_metrics,omitempty"`
	LockName          string   `json:"lock_name,omitempty"`
}

type BenchmarkRunSample struct {
	Name       string             `json:"name,omitempty"`
	Metrics    map[string]float64 `json:"metrics,omitempty"`
	DurationMS int64              `json:"duration_ms,omitempty"`
	ExitCode   int                `json:"exit_code,omitempty"`
	Warmup     bool               `json:"warmup,omitempty"`
}

type BenchmarkMetricSummary struct {
	Mean           float64 `json:"mean"`
	StdDev         float64 `json:"stddev"`
	RelativeStdDev float64 `json:"relative_stddev"`
	Samples        int     `json:"samples"`
}

type BenchmarkAssessment struct {
	Protocol      BenchmarkProtocol                 `json:"protocol"`
	Stable        bool                              `json:"stable"`
	MetricStats   map[string]BenchmarkMetricSummary `json:"metric_stats,omitempty"`
	PrimaryMetric string                            `json:"primary_metric,omitempty"`
	Notes         []string                          `json:"notes,omitempty"`
}

type SearchMode string

const (
	SearchModeGreedy SearchMode = "greedy"
	SearchModeBeam   SearchMode = "beam"
	SearchModeBandit SearchMode = "bandit"
)

type SearchConfig struct {
	Mode                   SearchMode `json:"mode"`
	BeamWidth              int        `json:"beam_width"`
	EarlyCancelScoreMargin float64    `json:"early_cancel_score_margin"`
	EarlyCancelMinRound    int        `json:"early_cancel_min_round"`
	NoveltyWeight          float64    `json:"novelty_weight,omitempty"`
}

type SearchCandidateState struct {
	CandidateID string              `json:"candidate_id"`
	Round       int                 `json:"round"`
	Verified    bool                `json:"verified"`
	BuildPassed bool                `json:"build_passed"`
	Assessment  BenchmarkAssessment `json:"assessment"`
	Efficiency  float64             `json:"efficiency"`
	Metrics     map[string]float64  `json:"metrics,omitempty"`
	Diagnosis   *BottleneckReport   `json:"diagnosis,omitempty"`
	Metadata    map[string]string   `json:"metadata,omitempty"`
}

type CandidateScore struct {
	CandidateID string             `json:"candidate_id"`
	Score       float64            `json:"score"`
	Round       int                `json:"round"`
	Reasons     []string           `json:"reasons,omitempty"`
	Metrics     map[string]float64 `json:"metrics,omitempty"`
}

type SearchSelection struct {
	Config            SearchConfig     `json:"config"`
	Ranked            []CandidateScore `json:"ranked"`
	Survivors         []CandidateScore `json:"survivors"`
	EarlyCancel       bool             `json:"early_cancel"`
	EarlyCancelReason string           `json:"early_cancel_reason,omitempty"`
}

type OuterLoopFamilyStatus struct {
	Family       string   `json:"family"`
	Required     bool     `json:"required"`
	Status       string   `json:"status"`
	Reason       string   `json:"reason,omitempty"`
	CandidateIDs []string `json:"candidate_ids,omitempty"`
}

type OuterLoopStatus struct {
	Families          []OuterLoopFamilyStatus `json:"families,omitempty"`
	Exhausted         bool                    `json:"exhausted"`
	ReadyForInnerLoop bool                    `json:"ready_for_inner_loop"`
	CurrentBestID     string                  `json:"current_best_id,omitempty"`
}

type SearchManager struct {
	Config SearchConfig
}

func DefaultBenchmarkProtocol() BenchmarkProtocol {
	return BenchmarkProtocol{
		WarmupRuns:        1,
		MeasuredRuns:      5,
		VarianceThreshold: 0.08,
		NormalizedMetrics: []string{"tokens_per_sec", "x_real_time", "rtf"},
		LockName:          "benchmark",
	}
}

func DefaultSearchConfig() SearchConfig {
	return SearchConfig{
		Mode:                   SearchModeBeam,
		BeamWidth:              3,
		EarlyCancelScoreMargin: 0.2,
		EarlyCancelMinRound:    2,
		NoveltyWeight:          0.15,
	}
}

var nsightMetricPattern = regexp.MustCompile(`(?m)([A-Za-z][A-Za-z0-9_./:%-]+)\s*[:=]\s*(-?[0-9][0-9,]*\.?[0-9]*)`)
var nsysReportPattern = regexp.MustCompile(`^\[\d+/\d+\] Executing '([^']+)' stats report`)
var nsysTableSplitPattern = regexp.MustCompile(`\s{2,}`)

func ParseNsightProfile(tool, stdout, stderr string) NsightProfile {
	profile := NsightProfile{
		Version:     1,
		Tool:        canonicalTool(tool),
		Metrics:     map[string]float64{},
		RawMetrics:  map[string]float64{},
		GeneratedAt: time.Now().UTC(),
	}
	text := strings.TrimSpace(stdout + "\n" + stderr)
	if profile.Tool == "nsys" {
		parseNsysStatsText(text, &profile)
	}
	for _, match := range nsightMetricPattern.FindAllStringSubmatch(text, -1) {
		rawKey := strings.TrimSpace(match[1])
		value, err := parseMetricNumber(match[2])
		if err != nil {
			continue
		}
		profile.RawMetrics[rawKey] = value
		key := canonicalNsightMetric(rawKey)
		if key == "" {
			continue
		}
		profile.Metrics[key] = value
	}
	if kernel := extractKernelName(text); kernel != "" {
		profile.KernelName = kernel
	}
	if len(profile.Metrics) == 0 {
		profile.Notes = append(profile.Notes, "no normalized Nsight metrics were detected in the profile output")
	}
	return profile
}

func AnalyzeRoofline(profile NsightProfile) BottleneckReport {
	metrics := cloneMetricMap(profile.Metrics)
	sm := firstMetric(metrics, "sm_pct_of_peak", "sm_active_pct", "sm_efficiency_pct")
	tensor := firstMetric(metrics, "tensor_pct_of_peak", "tensor_active_pct")
	dram := firstMetric(metrics, "dram_pct_of_peak", "dram_bandwidth_pct")
	occupancy := firstMetric(metrics, "occupancy_pct", "achieved_occupancy_pct")
	launch := firstMetric(metrics, "launch_overhead_pct", "cuda_api_pct")
	h2d := firstMetric(metrics, "h2d_memcpy_time_pct")
	d2d := firstMetric(metrics, "d2d_memcpy_time_pct")
	d2h := firstMetric(metrics, "d2h_memcpy_time_pct")
	memcpyAPI := firstMetric(metrics, "cuda_memcpy_api_pct")
	sync := firstMetric(metrics, "cuda_sync_pct")
	topKernel := firstMetric(metrics, "top_kernel_pct")
	efficiency := maxFloat(sm, tensor, dram, topKernel) / 100.0
	if efficiency == 0 && maxFloat(h2d, d2d, d2h, memcpyAPI, sync) > 0 {
		efficiency = math.Max(0.05, 1.0-maxFloat(h2d, memcpyAPI)/100.0)
	}

	category := "mixed"
	rootCauses := []string{}
	confidence := 0.45

	switch {
	case memcpyAPI >= 35 && h2d >= 50:
		category = "memory"
		rootCauses = append(rootCauses,
			"host-to-device staging dominates the traced transfer time",
			"runtime is spending substantial time in cudaMemcpyAsync before kernels can do useful work",
		)
		if firstMetric(metrics, "cuda_host_alloc_pct") >= 10 {
			rootCauses = append(rootCauses, "pinned host allocation and staging behavior is a material part of the traced runtime")
		}
		confidence = 0.9
	case launch >= 18 && maxFloat(sm, tensor, dram) < 50:
		category = "launch"
		rootCauses = append(rootCauses,
			"launch overhead dominates relative to achieved GPU throughput",
			"kernel fusion, CUDA Graphs, or batching should be considered before deeper kernel rewrites",
		)
		confidence = 0.82
	case sync >= 20 && launch < 18:
		category = "launch"
		rootCauses = append(rootCauses,
			"device or stream synchronization is consuming a large share of traced API time",
			"the hot path may include avoidable CPU-GPU barriers or phase boundaries before deeper kernel work",
		)
		confidence = 0.78
	case occupancy > 0 && occupancy < 35:
		category = "occupancy"
		rootCauses = append(rootCauses,
			"achieved occupancy is low enough to constrain throughput",
			"register pressure, shared memory footprint, or tile shape likely limits active warps",
		)
		confidence = 0.78
	case dram >= 65 && dram >= sm+10:
		category = "memory"
		rootCauses = append(rootCauses,
			"DRAM throughput is much closer to peak than SM throughput",
			"global memory traffic or poor locality is likely the dominant limit",
		)
		confidence = 0.84
	case maxFloat(sm, tensor) >= 65 && dram > 0 && maxFloat(sm, tensor) >= dram+8:
		category = "compute"
		rootCauses = append(rootCauses,
			"SM or Tensor Core throughput is closer to peak than memory bandwidth",
			"tile shape, instruction mix, tensor-core usage, or math precision likely dominate",
		)
		confidence = 0.84
	default:
		rootCauses = append(rootCauses,
			"available metrics point to a mixed bottleneck",
			"collect more counters or compare across shapes before overfitting one optimization path",
		)
	}
	if profile.Tool == "nsys" && firstMetric(metrics, "top_kernel_pct") == 0 {
		rootCauses = append(rootCauses, "Nsight Systems reports transfer and timeline behavior but not full roofline counters; use Nsight Compute if you need kernel-level DRAM vs SM attribution")
	}

	summary := fmt.Sprintf("%s-bound kernel with %.0f%% estimated roofline efficiency", strings.Title(category), efficiency*100)
	if profile.Tool == "nsys" && maxFloat(sm, tensor, dram) == 0 {
		summary = fmt.Sprintf("%s-bound traced path with %.0f%% estimated efficiency", strings.Title(category), efficiency*100)
	}
	return BottleneckReport{
		Version:         1,
		Category:        category,
		Efficiency:      roundFloat(efficiency, 4),
		EvidenceMetrics: metrics,
		RootCauses:      rootCauses,
		Confidence:      roundFloat(confidence, 4),
		Summary:         summary,
		GeneratedAt:     time.Now().UTC(),
	}
}

func PrescribeFromReport(report BottleneckReport, req Request, candidate Candidate) Prescription {
	fixes := []PrescriptionFix{}
	knobs := []string{}
	effects := []string{}
	risk := "medium"

	addFix := func(action string, affected []string, effect string, fixRisk string) {
		fixes = append(fixes, PrescriptionFix{
			Action:                 action,
			AffectedKnobs:          append([]string{}, affected...),
			ExpectedHardwareEffect: effect,
			Risk:                   fixRisk,
		})
		knobs = append(knobs, affected...)
		effects = append(effects, effect)
		if risk == "low" || (risk == "medium" && fixRisk == "high") {
			risk = fixRisk
		}
		if risk == "medium" && fixRisk == "high" {
			risk = "high"
		}
	}

	switch report.Category {
	case "memory":
		if firstMetric(report.EvidenceMetrics, "h2d_memcpy_time_pct") >= 50 || firstMetric(report.EvidenceMetrics, "cuda_memcpy_api_pct") >= 35 {
			addFix("Reduce host-to-device staging before deeper kernel work: keep model components resident, disable unnecessary offload, reuse pinned buffers, and cache reusable conditioning or embeddings.", []string{"offload-policy", "residency", "staging-buffers", "conditioning-cache"}, "lower host-device transfer time and expose the real steady-state kernel bottleneck", "low")
			addFix("Separate load or warmup profiling from steady-state generation so one-time checkpoint movement does not hide the real hot path.", []string{"profile-phase", "warmup", "benchmark-protocol"}, "focus optimization effort on the objective phase instead of startup costs", "low")
		}
		addFix("Increase fusion around the hot operator and remove unnecessary intermediate reads or writes.", []string{"fusion-boundary", "epilogue", "layout"}, "reduce DRAM traffic and increase locality", "medium")
		addFix("Try Triton or CuTe variants with vectorized loads, wider transactions, and tile shapes that improve L2 reuse.", []string{"block_m", "block_n", "block_k", "num_warps"}, "raise effective bandwidth and L2 hit rate", "medium")
		addFix("Prefer weight-only or KV-cache quantization before deeper tensor-core rewrites if memory still dominates.", []string{"precision", "kv-cache", "awq"}, "reduce bytes moved per token", "low")
	case "compute":
		addFix("Retile for tensor-core-friendly dimensions and verify tensor-core paths are actually active.", []string{"mma-shape", "tile-shape", "pipeline-stages"}, "increase SM and Tensor Core utilization", "medium")
		addFix("Try lower-precision compute branches like FP8 before hand-written CUDA if the runtime supports them.", []string{"precision", "transformer-engine", "modelopt"}, "increase math throughput and lower register pressure", "medium")
		addFix("Escalate from Triton to CuTe or CUTLASS when the winning shape depends on hardware-specific tensor-core scheduling.", []string{"backend", "warp-specialization"}, "improve instruction scheduling and peak throughput", "high")
	case "occupancy":
		addFix("Reduce register and shared memory pressure by shrinking tile shapes or simplifying epilogues.", []string{"block-size", "shared-memory", "register-usage"}, "raise active warps per SM", "medium")
		addFix("Split fused kernels that over-consume registers if occupancy losses outweigh fusion wins.", []string{"fusion-boundary", "epilogue"}, "recover parallelism and latency hiding", "medium")
	case "launch":
		addFix("Fuse short kernels and remove optional hooks or debug paths from the hot loop.", []string{"fusion-boundary", "runtime-flags"}, "reduce launch overhead and CPU-GPU sync points", "low")
		addFix("Try torch.compile and CUDA Graphs on the stabilized path before writing a custom kernel.", []string{"torch-compile", "cuda-graphs"}, "amortize dispatch overhead", "low")
	default:
		addFix("Collect a richer Nsight profile and compare multiple shapes before committing to a backend rewrite.", []string{"profile", "shape-set"}, "improve diagnosis confidence", "low")
		addFix("Continue outer-loop experiments until runtime, precision, and attention backend branches are exhausted.", []string{"outer-loop"}, "avoid overfitting custom kernels too early", "low")
	}

	if strings.Contains(strings.ToLower(candidate.Backend), "triton") && report.Category == "compute" {
		addFix("If Triton plateaus, branch to CuTe or CUDA/CUTLASS instead of burning more rounds on the same DSL.", []string{"backend"}, "avoid local optima in one backend", "medium")
	}
	if strings.EqualFold(req.Workload, "decode") && report.Category == "memory" {
		addFix("Test paged KV, KV quantization, or attention backend changes before a new GEMM kernel.", []string{"kv-cache", "attention-backend"}, "reduce decode-time memory pressure", "low")
	}

	return Prescription{
		Version:                 1,
		Category:                report.Category,
		Summary:                 report.Summary,
		Fixes:                   fixes,
		AffectedKnobs:           dedupeStrings(knobs),
		ExpectedHardwareEffects: dedupeStrings(effects),
		Risk:                    risk,
		GeneratedAt:             time.Now().UTC(),
	}
}

func (p BenchmarkProtocol) Evaluate(samples []BenchmarkRunSample) BenchmarkAssessment {
	protocol := p
	if protocol.WarmupRuns <= 0 {
		protocol.WarmupRuns = 1
	}
	if protocol.MeasuredRuns <= 0 {
		protocol.MeasuredRuns = 5
	}
	if protocol.VarianceThreshold <= 0 {
		protocol.VarianceThreshold = 0.08
	}
	if len(protocol.NormalizedMetrics) == 0 {
		protocol.NormalizedMetrics = DefaultBenchmarkProtocol().NormalizedMetrics
	}

	series := map[string][]float64{}
	for _, sample := range samples {
		if sample.Warmup || sample.ExitCode != 0 {
			continue
		}
		for key, value := range sample.Metrics {
			series[key] = append(series[key], value)
		}
	}

	stats := map[string]BenchmarkMetricSummary{}
	notes := []string{}
	stable := true
	for _, key := range protocol.NormalizedMetrics {
		values := series[key]
		if len(values) == 0 {
			continue
		}
		mean := average(values)
		stddev := stddev(values, mean)
		rel := 0.0
		if mean != 0 {
			rel = math.Abs(stddev / mean)
		}
		stats[key] = BenchmarkMetricSummary{
			Mean:           roundFloat(mean, 6),
			StdDev:         roundFloat(stddev, 6),
			RelativeStdDev: roundFloat(rel, 6),
			Samples:        len(values),
		}
		if rel > protocol.VarianceThreshold {
			stable = false
			notes = append(notes, fmt.Sprintf("%s variance %.4f exceeded threshold %.4f", key, rel, protocol.VarianceThreshold))
		}
	}
	if len(stats) == 0 {
		stable = false
		notes = append(notes, "no measured benchmark samples matched the protocol metrics")
	}

	primary := ""
	if len(protocol.NormalizedMetrics) > 0 {
		for _, key := range protocol.NormalizedMetrics {
			if _, ok := stats[key]; ok {
				primary = key
				break
			}
		}
	}
	return BenchmarkAssessment{
		Protocol:      protocol,
		Stable:        stable,
		MetricStats:   stats,
		PrimaryMetric: primary,
		Notes:         notes,
	}
}

func (m SearchManager) Select(states []SearchCandidateState) SearchSelection {
	config := m.Config
	if config.Mode == "" {
		config = DefaultSearchConfig()
	}
	if config.BeamWidth <= 0 {
		config.BeamWidth = 1
	}
	if config.NoveltyWeight < 0 {
		config.NoveltyWeight = 0
	}

	metricRanges := map[string][2]float64{}
	for _, state := range states {
		for key, summary := range state.Assessment.MetricStats {
			current := metricRanges[key]
			value := summary.Mean
			if current == [2]float64{} {
				metricRanges[key] = [2]float64{value, value}
				continue
			}
			metricRanges[key] = [2]float64{math.Min(current[0], value), math.Max(current[1], value)}
		}
	}

	ranked := make([]CandidateScore, 0, len(states))
	for _, state := range states {
		score := 0.0
		reasons := []string{}
		if state.BuildPassed {
			score += 0.2
			reasons = append(reasons, "build passed")
		} else {
			score -= 1.0
			reasons = append(reasons, "build failed")
		}
		if state.Verified {
			score += 0.8
			reasons = append(reasons, "verify passed")
		} else {
			score -= 2.0
			reasons = append(reasons, "verify missing or failed")
		}
		if state.Assessment.Stable {
			score += 0.4
			reasons = append(reasons, "benchmark variance within protocol")
		}
		if state.Efficiency > 0 {
			score += state.Efficiency
			reasons = append(reasons, fmt.Sprintf("roofline efficiency %.2f", state.Efficiency))
		}
		for key, summary := range state.Assessment.MetricStats {
			rng := metricRanges[key]
			normalized := normalizeMetric(summary.Mean, rng[0], rng[1], metricDirection(key))
			score += normalized
		}
		ranked = append(ranked, CandidateScore{
			CandidateID: state.CandidateID,
			Score:       roundFloat(score, 6),
			Round:       state.Round,
			Reasons:     reasons,
			Metrics:     state.Metrics,
		})
	}

	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].Score == ranked[j].Score {
			return ranked[i].CandidateID < ranked[j].CandidateID
		}
		return ranked[i].Score > ranked[j].Score
	})

	survivorCount := 1
	switch config.Mode {
	case SearchModeBeam:
		survivorCount = minInt(config.BeamWidth, len(ranked))
	case SearchModeBandit:
		survivorCount = minInt(maxInt(config.BeamWidth, 2), len(ranked))
	default:
		survivorCount = minInt(1, len(ranked))
	}
	survivors := selectDiverseSurvivors(ranked, states, survivorCount, config)
	selection := SearchSelection{
		Config:    config,
		Ranked:    ranked,
		Survivors: survivors,
	}
	if len(ranked) >= 2 && ranked[0].Score-ranked[1].Score >= config.EarlyCancelScoreMargin && ranked[0].Round >= config.EarlyCancelMinRound {
		selection.EarlyCancel = true
		selection.EarlyCancelReason = fmt.Sprintf("candidate %s dominates by %.3f after round %d", ranked[0].CandidateID, ranked[0].Score-ranked[1].Score, ranked[0].Round)
	}
	if len(ranked) == 1 && ranked[0].Round >= config.EarlyCancelMinRound {
		selection.EarlyCancel = true
		selection.EarlyCancelReason = fmt.Sprintf("single verified search branch %s remains after round %d", ranked[0].CandidateID, ranked[0].Round)
	}
	return selection
}

func selectDiverseSurvivors(ranked []CandidateScore, states []SearchCandidateState, survivorCount int, config SearchConfig) []CandidateScore {
	if survivorCount <= 0 || len(ranked) == 0 {
		return nil
	}
	if survivorCount >= len(ranked) {
		return append([]CandidateScore{}, ranked...)
	}
	if config.Mode == SearchModeGreedy || config.NoveltyWeight <= 0 {
		return append([]CandidateScore{}, ranked[:survivorCount]...)
	}

	stateByID := make(map[string]SearchCandidateState, len(states))
	for _, state := range states {
		stateByID[state.CandidateID] = state
	}

	survivors := make([]CandidateScore, 0, survivorCount)
	survivors = append(survivors, ranked[0])
	selected := map[string]bool{ranked[0].CandidateID: true}

	for len(survivors) < survivorCount {
		bestIdx := -1
		bestScore := math.Inf(-1)
		for idx, candidate := range ranked {
			if selected[candidate.CandidateID] {
				continue
			}
			novelty := 1.0
			if len(survivors) > 0 {
				maxSimilarity := 0.0
				state := stateByID[candidate.CandidateID]
				for _, selectedCandidate := range survivors {
					similarity := candidateSimilarity(state, stateByID[selectedCandidate.CandidateID])
					if similarity > maxSimilarity {
						maxSimilarity = similarity
					}
				}
				novelty = 1.0 - maxSimilarity
			}
			mmrScore := candidate.Score + config.NoveltyWeight*novelty
			if mmrScore > bestScore || (mmrScore == bestScore && (bestIdx == -1 || candidate.CandidateID < ranked[bestIdx].CandidateID)) {
				bestIdx = idx
				bestScore = mmrScore
			}
		}
		if bestIdx == -1 {
			break
		}
		survivors = append(survivors, ranked[bestIdx])
		selected[ranked[bestIdx].CandidateID] = true
	}
	return survivors
}

func candidateSimilarity(a, b SearchCandidateState) float64 {
	if strings.TrimSpace(a.CandidateID) == "" || strings.TrimSpace(b.CandidateID) == "" {
		return 0
	}
	signatureA := strings.TrimSpace(a.Metadata["signature"])
	signatureB := strings.TrimSpace(b.Metadata["signature"])
	if signatureA != "" && signatureA == signatureB {
		return 1
	}

	similarity := tokenSetSimilarity(candidateSearchTokens(a), candidateSearchTokens(b))

	laneA := strings.TrimSpace(a.Metadata["search_lane"])
	laneB := strings.TrimSpace(b.Metadata["search_lane"])
	if laneA != "" && laneA == laneB {
		similarity = math.Max(similarity, 0.85)
	}

	backendA := strings.TrimSpace(a.Metadata["backend"])
	backendB := strings.TrimSpace(b.Metadata["backend"])
	if backendA != "" && backendA == backendB {
		similarity = math.Max(similarity, 0.55)
	}

	return similarity
}

func candidateSearchTokens(state SearchCandidateState) map[string]struct{} {
	raw := []string{
		state.CandidateID,
		state.Metadata["backend"],
		state.Metadata["search_lane"],
		state.Metadata["signature"],
		state.Metadata["family"],
		state.Metadata["hypothesis"],
	}
	tokens := make(map[string]struct{})
	for _, value := range raw {
		for _, token := range splitSearchTokens(value) {
			tokens[token] = struct{}{}
		}
	}
	return tokens
}

func splitSearchTokens(value string) []string {
	value = strings.TrimSpace(strings.ToLower(value))
	if value == "" {
		return nil
	}
	replacer := strings.NewReplacer("/", " ", "_", " ", "-", " ", ".", " ", ":", " ", ",", " ")
	value = replacer.Replace(value)
	fields := strings.Fields(value)
	if len(fields) == 0 {
		return nil
	}
	out := make([]string, 0, len(fields))
	for _, field := range fields {
		if field != "" {
			out = append(out, field)
		}
	}
	return out
}

func tokenSetSimilarity(a, b map[string]struct{}) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	intersection := 0
	union := len(a)
	for token := range b {
		if _, ok := a[token]; ok {
			intersection++
		} else {
			union++
		}
	}
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

func EvaluateOuterLoopStatus(session *Session) OuterLoopStatus {
	if session == nil {
		return OuterLoopStatus{}
	}
	families := []OuterLoopFamilyStatus{
		evaluateOuterLoopFamily(session, "baseline", true),
		evaluateOuterLoopFamily(session, "profile", true),
		evaluateOuterLoopFamily(session, "model-family", true),
		evaluateOuterLoopFamily(session, "runtime", true),
		evaluateOuterLoopFamily(session, "quantization", true),
		evaluateOuterLoopFamily(session, "compile", true),
	}
	if requiresAttentionBackendFamily(session) {
		families = append(families, evaluateOuterLoopFamily(session, "attention-backend", true))
	}

	exhausted := true
	for _, family := range families {
		if family.Required && family.Status == "pending" {
			exhausted = false
			break
		}
	}
	bestID := strings.TrimSpace(session.CurrentBestID)
	if bestID == "" {
		for _, candidate := range session.Candidates {
			if candidate.Winner {
				bestID = candidate.ID
				break
			}
		}
	}
	return OuterLoopStatus{
		Families:          families,
		Exhausted:         exhausted,
		ReadyForInnerLoop: exhausted,
		CurrentBestID:     bestID,
	}
}

func SaveRoundArtifact(session *Session, candidateID string, round int, kind string, value any) (string, error) {
	if session == nil {
		return "", fmt.Errorf("session is required")
	}
	candidateID = safeSessionPathID(candidateID)
	if candidateID == "" || candidateID == "session" {
		return "", fmt.Errorf("candidate id is required")
	}
	if round <= 0 {
		return "", fmt.Errorf("round must be >= 1")
	}
	kind = canonicalLoopValue(kind)
	if kind == "" {
		return "", fmt.Errorf("artifact kind is required")
	}
	roundDir := filepath.Join(session.WorkspaceRoot, "candidates", candidateID, "rounds", fmt.Sprintf("%03d", round))
	if err := os.MkdirAll(roundDir, 0o755); err != nil {
		return "", fmt.Errorf("create round artifact dir: %w", err)
	}
	path := filepath.Join(roundDir, kind+".json")
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return "", fmt.Errorf("encode round artifact: %w", err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return "", fmt.Errorf("write round artifact: %w", err)
	}
	return path, nil
}

func LoadRoundArtifact(path string, target any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read round artifact: %w", err)
	}
	if err := json.Unmarshal(data, target); err != nil {
		return fmt.Errorf("decode round artifact: %w", err)
	}
	return nil
}

func evaluateOuterLoopFamily(session *Session, family string, required bool) OuterLoopFamilyStatus {
	if decision, ok := session.LoopDecision("outer", family); ok {
		return OuterLoopFamilyStatus{
			Family:       family,
			Required:     required,
			Status:       valueOrFallback(decision.Status, "pending"),
			Reason:       decision.Reason,
			CandidateIDs: compactCandidateIDs(decision.CandidateID),
		}
	}

	candidateIDs := []string{}
	for _, candidate := range session.Candidates {
		if !candidateMatchesOuterLoopFamily(candidate, family) {
			continue
		}
		if family == "profile" {
			candidateIDs = append(candidateIDs, candidate.ID)
			continue
		}
		if candidateStagePassed(candidate, "benchmark") || candidateStagePassed(candidate, "model-benchmark") || candidateStagePassed(candidate, "verify") {
			candidateIDs = append(candidateIDs, candidate.ID)
		}
	}
	status := "pending"
	reason := "no candidate or explicit decision recorded yet"
	if len(candidateIDs) > 0 {
		status = "tested"
		reason = "at least one candidate in this family completed a decisive stage"
	}
	return OuterLoopFamilyStatus{
		Family:       family,
		Required:     required,
		Status:       status,
		Reason:       reason,
		CandidateIDs: candidateIDs,
	}
}

func candidateMatchesOuterLoopFamily(candidate Candidate, family string) bool {
	backend := canonicalLoopValue(candidate.Backend)
	name := canonicalLoopValue(candidate.Name)
	template := canonicalLoopValue(candidate.Template)
	joined := backend + " " + name + " " + template
	switch family {
	case "baseline":
		return strings.Contains(joined, "baseline")
	case "profile":
		_, ok := candidate.Stages[canonicalLoopValue("profile")]
		return ok
	case "model-family":
		return stringContainsAny(joined, "turbo", "distilled", "checkpoint", "model-family", "packaged", "variant")
	case "runtime":
		return stringContainsAny(joined, "runtime", "no-attn", "flashinfer", "flash-attn", "flashattention", "flex-attention", "sdpa", "attention-impl")
	case "quantization":
		return stringContainsAny(joined, "awq", "int4", "int8", "fp8", "nvfp4", "quant", "kv-quant")
	case "compile":
		return stringContainsAny(joined, "compile", "inductor", "cuda-graphs", "cudagraphs")
	case "attention-backend":
		return stringContainsAny(joined, "flash-attn", "flashattention", "flashinfer", "flex-attention", "sdpa", "attention-backend")
	default:
		return false
	}
}

func candidateStagePassed(candidate Candidate, stage string) bool {
	record, ok := candidate.Stages[canonicalLoopValue(stage)]
	return ok && record.ExitCode == 0
}

func requiresAttentionBackendFamily(session *Session) bool {
	if session == nil {
		return false
	}
	if strings.EqualFold(session.Request.Workload, "decode") || strings.EqualFold(session.Request.Workload, "prefill") {
		return true
	}
	for _, operator := range session.Request.Operators {
		if strings.Contains(strings.ToLower(operator), "attention") {
			return true
		}
	}
	return false
}

func canonicalTool(tool string) string {
	tool = canonicalLoopValue(tool)
	switch tool {
	case "ncu", "nsight-compute":
		return "ncu"
	case "nsys", "nsight-systems":
		return "nsys"
	default:
		return tool
	}
}

func extractKernelName(text string) string {
	lines := strings.Split(text, "\n")
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(strings.ToLower(trimmed), "kernel") {
			if parts := strings.SplitN(trimmed, ":", 2); len(parts) == 2 {
				return strings.TrimSpace(parts[1])
			}
		}
	}
	return ""
}

func parseNsysStatsText(text string, profile *NsightProfile) {
	if profile == nil {
		return
	}
	sections := extractNsysSections(text)
	if len(sections) == 0 {
		return
	}
	parseNsysCUDAAPISum(sections["cuda_api_sum"], profile)
	parseNsysGPUMemTimeSum(sections["cuda_gpu_mem_time_sum"], profile)
	parseNsysGPUMemSizeSum(sections["cuda_gpu_mem_size_sum"], profile)
	parseNsysGPUKernelSum(sections["cuda_gpu_kern_sum"], profile)
}

func extractNsysSections(text string) map[string]string {
	sections := map[string][]string{}
	current := ""
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		if match := nsysReportPattern.FindStringSubmatch(trimmed); len(match) == 2 {
			current = strings.TrimSpace(match[1])
			continue
		}
		if strings.HasPrefix(trimmed, "Generated:") {
			current = ""
			continue
		}
		if current == "" {
			continue
		}
		sections[current] = append(sections[current], line)
	}
	out := make(map[string]string, len(sections))
	for key, lines := range sections {
		out[key] = strings.Join(lines, "\n")
	}
	return out
}

func parseNsysCUDAAPISum(section string, profile *NsightProfile) {
	for _, columns := range parseNsysTableRows(section) {
		if len(columns) < 2 {
			continue
		}
		name := columns[len(columns)-1]
		switch strings.TrimSpace(name) {
		case "cudaMemcpyAsync":
			setProfileMetric(profile, "cuda_memcpy_api_pct", parseMetricPercent(columns[0]))
			setProfileMetric(profile, "cuda_memcpy_api_time_ms", nsToMS(parseMetricNumberFallback(columns[1])))
		case "cudaHostAlloc":
			setProfileMetric(profile, "cuda_host_alloc_pct", parseMetricPercent(columns[0]))
		case "cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC_v11060":
			addProfileMetric(profile, "cuda_launch_pct", parseMetricPercent(columns[0]))
		case "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaStreamSynchronize":
			addProfileMetric(profile, "cuda_sync_pct", parseMetricPercent(columns[0]))
		}
	}
}

func parseNsysGPUMemTimeSum(section string, profile *NsightProfile) {
	for _, columns := range parseNsysTableRows(section) {
		if len(columns) < 2 {
			continue
		}
		operation := columns[len(columns)-1]
		switch strings.TrimSpace(operation) {
		case "[CUDA memcpy Host-to-Device]":
			setProfileMetric(profile, "h2d_memcpy_time_pct", parseMetricPercent(columns[0]))
			setProfileMetric(profile, "h2d_memcpy_time_ms", nsToMS(parseMetricNumberFallback(columns[1])))
		case "[CUDA memcpy Device-to-Device]":
			setProfileMetric(profile, "d2d_memcpy_time_pct", parseMetricPercent(columns[0]))
			setProfileMetric(profile, "d2d_memcpy_time_ms", nsToMS(parseMetricNumberFallback(columns[1])))
		case "[CUDA memcpy Device-to-Host]":
			setProfileMetric(profile, "d2h_memcpy_time_pct", parseMetricPercent(columns[0]))
			setProfileMetric(profile, "d2h_memcpy_time_ms", nsToMS(parseMetricNumberFallback(columns[1])))
		}
	}
}

func parseNsysGPUMemSizeSum(section string, profile *NsightProfile) {
	for _, columns := range parseNsysTableRows(section) {
		if len(columns) < 2 {
			continue
		}
		operation := columns[len(columns)-1]
		switch strings.TrimSpace(operation) {
		case "[CUDA memcpy Host-to-Device]":
			setProfileMetric(profile, "h2d_total_mb", parseMetricNumberFallback(columns[0]))
		case "[CUDA memcpy Device-to-Device]":
			setProfileMetric(profile, "d2d_total_mb", parseMetricNumberFallback(columns[0]))
		case "[CUDA memcpy Device-to-Host]":
			setProfileMetric(profile, "d2h_total_mb", parseMetricNumberFallback(columns[0]))
		}
	}
}

func parseNsysGPUKernelSum(section string, profile *NsightProfile) {
	rows := parseNsysTableRows(section)
	if len(rows) == 0 {
		return
	}
	top := rows[0]
	if len(top) < 2 {
		return
	}
	setProfileMetric(profile, "top_kernel_pct", parseMetricPercent(top[0]))
	setProfileMetric(profile, "top_kernel_time_ms", nsToMS(parseMetricNumberFallback(top[1])))
	if strings.TrimSpace(profile.KernelName) == "" {
		profile.KernelName = strings.TrimSpace(top[len(top)-1])
	}
}

func parseNsysTableRows(section string) [][]string {
	rows := [][]string{}
	for _, line := range strings.Split(section, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		if strings.HasPrefix(trimmed, "Time (%)") || strings.HasPrefix(trimmed, "Total (MB)") || strings.HasPrefix(trimmed, "--------") {
			continue
		}
		if strings.HasPrefix(trimmed, "[") && strings.Contains(trimmed, "Executing") {
			continue
		}
		if strings.HasPrefix(trimmed, "Generated:") {
			break
		}
		columns := nsysTableSplitPattern.Split(trimmed, -1)
		if len(columns) < 2 {
			continue
		}
		rows = append(rows, columns)
	}
	return rows
}

func setProfileMetric(profile *NsightProfile, key string, value float64) {
	if profile == nil || strings.TrimSpace(key) == "" || value == 0 {
		return
	}
	profile.RawMetrics[key] = roundFloat(value, 6)
	profile.Metrics[key] = roundFloat(value, 6)
}

func addProfileMetric(profile *NsightProfile, key string, value float64) {
	if profile == nil || strings.TrimSpace(key) == "" || value == 0 {
		return
	}
	total := profile.Metrics[key] + value
	profile.RawMetrics[key] = roundFloat(total, 6)
	profile.Metrics[key] = roundFloat(total, 6)
}

func parseMetricPercent(value string) float64 {
	parsed, err := parseMetricNumber(value)
	if err != nil {
		return 0
	}
	return parsed
}

func parseMetricNumberFallback(value string) float64 {
	parsed, err := parseMetricNumber(value)
	if err != nil {
		return 0
	}
	return parsed
}

func nsToMS(value float64) float64 {
	if value == 0 {
		return 0
	}
	return value / 1_000_000.0
}

func parseMetricNumber(value string) (float64, error) {
	clean := strings.ReplaceAll(strings.TrimSpace(value), ",", "")
	return strconv.ParseFloat(clean, 64)
}

func canonicalNsightMetric(key string) string {
	lower := strings.ToLower(strings.TrimSpace(key))
	switch {
	case strings.Contains(lower, "dram__throughput.avg.pct_of_peak"), strings.Contains(lower, "dram throughput"), strings.Contains(lower, "dram_bw_pct"):
		return "dram_pct_of_peak"
	case strings.Contains(lower, "sm__throughput.avg.pct_of_peak"), strings.Contains(lower, "sm throughput"), strings.Contains(lower, "sm_pct"):
		return "sm_pct_of_peak"
	case strings.Contains(lower, "tensor") && strings.Contains(lower, "pct_of_peak"):
		return "tensor_pct_of_peak"
	case strings.Contains(lower, "achieved_occupancy"), strings.Contains(lower, "occupancy") && strings.Contains(lower, "pct"):
		return "occupancy_pct"
	case strings.Contains(lower, "l2") && strings.Contains(lower, "hit") && strings.Contains(lower, "pct"):
		return "l2_hit_rate_pct"
	case strings.Contains(lower, "cuda_api") && strings.Contains(lower, "pct"), strings.Contains(lower, "launch_overhead_pct"):
		return "launch_overhead_pct"
	case strings.Contains(lower, "gpu time"), strings.Contains(lower, "gpu__time_duration"):
		return "gpu_time_ms"
	default:
		return ""
	}
}

func cloneMetricMap(metrics map[string]float64) map[string]float64 {
	if len(metrics) == 0 {
		return nil
	}
	out := make(map[string]float64, len(metrics))
	for key, value := range metrics {
		out[key] = value
	}
	return out
}

func firstMetric(metrics map[string]float64, keys ...string) float64 {
	for _, key := range keys {
		if value, ok := metrics[key]; ok {
			return value
		}
	}
	return 0
}

func maxFloat(values ...float64) float64 {
	best := 0.0
	for _, value := range values {
		if value > best {
			best = value
		}
	}
	return best
}

func roundFloat(value float64, places int) float64 {
	scale := math.Pow10(places)
	return math.Round(value*scale) / scale
}

func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, value := range values {
		sum += value
	}
	return sum / float64(len(values))
}

func stddev(values []float64, mean float64) float64 {
	if len(values) <= 1 {
		return 0
	}
	sum := 0.0
	for _, value := range values {
		diff := value - mean
		sum += diff * diff
	}
	return math.Sqrt(sum / float64(len(values)))
}

func metricDirection(key string) string {
	key = strings.ToLower(strings.TrimSpace(key))
	switch {
	case strings.Contains(key, "tokens_per_sec"), strings.Contains(key, "throughput"), strings.Contains(key, "x_real_time"), strings.Contains(key, "utilization"), strings.Contains(key, "occupancy"):
		return "higher"
	default:
		return "lower"
	}
}

func normalizeMetric(value, minValue, maxValue float64, direction string) float64 {
	if maxValue == minValue {
		return 0.5
	}
	norm := (value - minValue) / (maxValue - minValue)
	if direction == "lower" {
		return 1 - norm
	}
	return norm
}

func minInt(left, right int) int {
	if left < right {
		return left
	}
	return right
}

func maxInt(left, right int) int {
	if left > right {
		return left
	}
	return right
}

func stringContainsAny(haystack string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(haystack, needle) {
			return true
		}
	}
	return false
}

func compactCandidateIDs(value string) []string {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil
	}
	return []string{value}
}

func dedupeStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	seen := map[string]struct{}{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" {
			continue
		}
		if _, ok := seen[trimmed]; ok {
			continue
		}
		seen[trimmed] = struct{}{}
		out = append(out, trimmed)
	}
	sort.Strings(out)
	return out
}
