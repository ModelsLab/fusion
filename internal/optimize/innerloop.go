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
	Version    int                `json:"version"`
	Tool       string             `json:"tool"`
	KernelName string             `json:"kernel_name,omitempty"`
	Metrics    map[string]float64 `json:"metrics,omitempty"`
	RawMetrics map[string]float64 `json:"raw_metrics,omitempty"`
	Notes      []string           `json:"notes,omitempty"`
	GeneratedAt time.Time         `json:"generated_at"`
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
	Name      string             `json:"name,omitempty"`
	Metrics    map[string]float64 `json:"metrics,omitempty"`
	DurationMS int64              `json:"duration_ms,omitempty"`
	ExitCode   int                `json:"exit_code,omitempty"`
	Warmup     bool               `json:"warmup,omitempty"`
}

type BenchmarkMetricSummary struct {
	Mean          float64 `json:"mean"`
	StdDev        float64 `json:"stddev"`
	RelativeStdDev float64 `json:"relative_stddev"`
	Samples       int     `json:"samples"`
}

type BenchmarkAssessment struct {
	Protocol      BenchmarkProtocol                  `json:"protocol"`
	Stable        bool                               `json:"stable"`
	MetricStats   map[string]BenchmarkMetricSummary  `json:"metric_stats,omitempty"`
	PrimaryMetric string                             `json:"primary_metric,omitempty"`
	Notes         []string                           `json:"notes,omitempty"`
}

type SearchMode string

const (
	SearchModeGreedy SearchMode = "greedy"
	SearchModeBeam   SearchMode = "beam"
	SearchModeBandit SearchMode = "bandit"
)

type SearchConfig struct {
	Mode                  SearchMode `json:"mode"`
	BeamWidth             int        `json:"beam_width"`
	EarlyCancelScoreMargin float64   `json:"early_cancel_score_margin"`
	EarlyCancelMinRound   int        `json:"early_cancel_min_round"`
}

type SearchCandidateState struct {
	CandidateID string                        `json:"candidate_id"`
	Round       int                           `json:"round"`
	Verified    bool                          `json:"verified"`
	BuildPassed bool                          `json:"build_passed"`
	Assessment  BenchmarkAssessment           `json:"assessment"`
	Efficiency  float64                       `json:"efficiency"`
	Metrics     map[string]float64            `json:"metrics,omitempty"`
	Diagnosis   *BottleneckReport             `json:"diagnosis,omitempty"`
	Metadata    map[string]string             `json:"metadata,omitempty"`
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
	}
}

var nsightMetricPattern = regexp.MustCompile(`(?m)([A-Za-z][A-Za-z0-9_./:%-]+)\s*[:=]\s*(-?[0-9][0-9,]*\.?[0-9]*)`)

func ParseNsightProfile(tool, stdout, stderr string) NsightProfile {
	profile := NsightProfile{
		Version:     1,
		Tool:        canonicalTool(tool),
		Metrics:     map[string]float64{},
		RawMetrics:  map[string]float64{},
		GeneratedAt: time.Now().UTC(),
	}
	text := strings.TrimSpace(stdout + "\n" + stderr)
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
	efficiency := maxFloat(sm, tensor, dram) / 100.0

	category := "mixed"
	rootCauses := []string{}
	confidence := 0.45

	switch {
	case launch >= 18 && maxFloat(sm, tensor, dram) < 50:
		category = "launch"
		rootCauses = append(rootCauses,
			"launch overhead dominates relative to achieved GPU throughput",
			"kernel fusion, CUDA Graphs, or batching should be considered before deeper kernel rewrites",
		)
		confidence = 0.82
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

	summary := fmt.Sprintf("%s-bound kernel with %.0f%% estimated roofline efficiency", strings.Title(category), efficiency*100)
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
	survivors := append([]CandidateScore{}, ranked[:survivorCount]...)
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

func EvaluateOuterLoopStatus(session *Session) OuterLoopStatus {
	if session == nil {
		return OuterLoopStatus{}
	}
	families := []OuterLoopFamilyStatus{
		evaluateOuterLoopFamily(session, "baseline", true),
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
