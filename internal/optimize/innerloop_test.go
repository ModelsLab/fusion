package optimize

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestParseNsightProfileAndAnalyzeRoofline(t *testing.T) {
	stdout := `
Kernel: fused_attention_kernel
sm__throughput.avg.pct_of_peak_sustained_elapsed: 41.5
dram__throughput.avg.pct_of_peak_sustained_elapsed: 82.1
achieved_occupancy: 47.0
`
	profile := ParseNsightProfile("ncu", stdout, "")
	if profile.Tool != "ncu" {
		t.Fatalf("expected ncu tool, got %q", profile.Tool)
	}
	if profile.KernelName != "fused_attention_kernel" {
		t.Fatalf("expected kernel name to be parsed, got %q", profile.KernelName)
	}
	if profile.Metrics["dram_pct_of_peak"] != 82.1 {
		t.Fatalf("expected DRAM metric to be normalized, got %+v", profile.Metrics)
	}

	report := AnalyzeRoofline(profile)
	if report.Category != "memory" {
		t.Fatalf("expected memory bottleneck, got %+v", report)
	}
	if report.Efficiency <= 0 {
		t.Fatalf("expected positive efficiency, got %+v", report)
	}

	prescription := PrescribeFromReport(report, Request{Workload: "decode"}, Candidate{Backend: "triton"})
	if len(prescription.Fixes) == 0 {
		t.Fatal("expected at least one prescription fix")
	}
}

func TestParseNsysProfileAndAnalyzeTransfers(t *testing.T) {
	stdout := `
[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)   Max (ns)   StdDev (ns)                Name
 --------  ---------------  ---------  ----------  ----------  --------  ----------  -----------  ---------------------------------
     66.3      23293888689      15381   1514458.7      9713.0      3186  1569352568   24192314.8  cudaMemcpyAsync
     22.9       8035853481        129  62293437.8  62825269.0     23062   125906375   13789607.6  cudaHostAlloc
      1.5        540286578      24988     21621.8      5032.0      2359    76770363     700090.9  cudaLaunchKernel
      1.3        451144243        166   2717736.4   1068425.0      1649    24338896    4382823.0  cudaDeviceSynchronize

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)    Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                                                  Name
 --------  ---------------  ---------  ----------  ----------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     55.1      10094043637       1392   7251468.1   5476800.0    276960  15456982    4390049.3  void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(...)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)   StdDev (ns)            Operation
 --------  ---------------  -----  --------  --------  --------  ---------  -----------  ------------------------------
     94.4       8338403682  12240  681242.1    3264.0       384  341626389    3743971.7  [CUDA memcpy Host-to-Device]
      5.6        495160827   2802  176716.9    1888.0      1184     762688     318187.0  [CUDA memcpy Device-to-Device]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation
 ----------  -----  --------  --------  --------  --------  -----------  ------------------------------
 206860.305   2802    73.826     0.010     0.000   319.488      134.326  [CUDA memcpy Device-to-Device]
  51647.823  12240     4.220     0.010     0.000  2100.298       23.096  [CUDA memcpy Host-to-Device]
`
	profile := ParseNsightProfile("nsys", stdout, "")
	if profile.Tool != "nsys" {
		t.Fatalf("expected nsys tool, got %q", profile.Tool)
	}
	if profile.KernelName == "" {
		t.Fatalf("expected top kernel to be inferred, got %+v", profile)
	}
	if profile.Metrics["h2d_memcpy_time_pct"] != 94.4 {
		t.Fatalf("expected H2D transfer metric, got %+v", profile.Metrics)
	}
	if profile.Metrics["cuda_memcpy_api_pct"] != 66.3 {
		t.Fatalf("expected memcpy API metric, got %+v", profile.Metrics)
	}

	report := AnalyzeRoofline(profile)
	if report.Category != "memory" {
		t.Fatalf("expected memory bottleneck, got %+v", report)
	}
	if report.Confidence < 0.85 {
		t.Fatalf("expected high-confidence transfer diagnosis, got %+v", report)
	}

	prescription := PrescribeFromReport(report, Request{Task: "video-generation", Workload: "sampling"}, Candidate{Backend: "runtime"})
	foundResidencyFix := false
	for _, fix := range prescription.Fixes {
		if strings.Contains(strings.ToLower(fix.Action), "host-to-device staging") {
			foundResidencyFix = true
			break
		}
	}
	if !foundResidencyFix {
		t.Fatalf("expected staging/residency fix in prescription, got %+v", prescription.Fixes)
	}
}

func TestBenchmarkProtocolEvaluate(t *testing.T) {
	protocol := BenchmarkProtocol{
		WarmupRuns:        1,
		MeasuredRuns:      3,
		VarianceThreshold: 0.05,
		NormalizedMetrics: []string{"tokens_per_sec"},
	}
	assessment := protocol.Evaluate([]BenchmarkRunSample{
		{Name: "warmup", Metrics: map[string]float64{"tokens_per_sec": 90}, Warmup: true},
		{Name: "run1", Metrics: map[string]float64{"tokens_per_sec": 100}},
		{Name: "run2", Metrics: map[string]float64{"tokens_per_sec": 101}},
		{Name: "run3", Metrics: map[string]float64{"tokens_per_sec": 99}},
	})
	if !assessment.Stable {
		t.Fatalf("expected stable benchmark assessment, got %+v", assessment)
	}
	stats, ok := assessment.MetricStats["tokens_per_sec"]
	if !ok || stats.Samples != 3 {
		t.Fatalf("expected tokens_per_sec stats with 3 samples, got %+v", assessment.MetricStats)
	}
}

func TestSearchManagerSelectBeamAndEarlyCancel(t *testing.T) {
	manager := SearchManager{
		Config: SearchConfig{
			Mode:                   SearchModeBeam,
			BeamWidth:              2,
			EarlyCancelScoreMargin: 0.2,
			EarlyCancelMinRound:    2,
			NoveltyWeight:          0.15,
		},
	}
	selection := manager.Select([]SearchCandidateState{
		{
			CandidateID: "cand-a",
			Round:       2,
			Verified:    true,
			BuildPassed: true,
			Efficiency:  0.82,
			Assessment: BenchmarkAssessment{
				Stable: true,
				MetricStats: map[string]BenchmarkMetricSummary{
					"tokens_per_sec": {Mean: 120, Samples: 3},
				},
			},
		},
		{
			CandidateID: "cand-b",
			Round:       2,
			Verified:    true,
			BuildPassed: true,
			Efficiency:  0.55,
			Assessment: BenchmarkAssessment{
				Stable: true,
				MetricStats: map[string]BenchmarkMetricSummary{
					"tokens_per_sec": {Mean: 80, Samples: 3},
				},
			},
		},
		{
			CandidateID: "cand-c",
			Round:       2,
			Verified:    false,
			BuildPassed: true,
			Efficiency:  0.50,
			Assessment: BenchmarkAssessment{
				Stable: false,
				MetricStats: map[string]BenchmarkMetricSummary{
					"tokens_per_sec": {Mean: 100, Samples: 3},
				},
			},
		},
	})

	if len(selection.Survivors) != 2 {
		t.Fatalf("expected 2 beam survivors, got %+v", selection)
	}
	if selection.Ranked[0].CandidateID != "cand-a" {
		t.Fatalf("expected cand-a to rank first, got %+v", selection.Ranked)
	}
	if !selection.EarlyCancel {
		t.Fatalf("expected early cancel to trigger, got %+v", selection)
	}
}

func TestSearchManagerPreservesDiverseLanes(t *testing.T) {
	manager := SearchManager{
		Config: SearchConfig{
			Mode:                   SearchModeBeam,
			BeamWidth:              2,
			EarlyCancelScoreMargin: 10,
			EarlyCancelMinRound:    99,
			NoveltyWeight:          1.0,
		},
	}
	selection := manager.Select([]SearchCandidateState{
		{
			CandidateID: "triton-fast-a",
			Round:       2,
			Verified:    true,
			BuildPassed: true,
			Efficiency:  0.82,
			Metadata: map[string]string{
				"backend":     "triton",
				"search_lane": "lane-triton",
				"signature":   "tile128-stage3",
			},
			Assessment: BenchmarkAssessment{
				Stable: true,
				MetricStats: map[string]BenchmarkMetricSummary{
					"tokens_per_sec": {Mean: 120, Samples: 3},
				},
			},
		},
		{
			CandidateID: "triton-fast-b",
			Round:       2,
			Verified:    true,
			BuildPassed: true,
			Efficiency:  0.81,
			Metadata: map[string]string{
				"backend":     "triton",
				"search_lane": "lane-triton",
				"signature":   "tile128-stage4",
			},
			Assessment: BenchmarkAssessment{
				Stable: true,
				MetricStats: map[string]BenchmarkMetricSummary{
					"tokens_per_sec": {Mean: 119, Samples: 3},
				},
			},
		},
		{
			CandidateID: "cute-alt",
			Round:       2,
			Verified:    true,
			BuildPassed: true,
			Efficiency:  0.79,
			Metadata: map[string]string{
				"backend":     "cute",
				"search_lane": "lane-cute",
				"signature":   "warp-specialized",
			},
			Assessment: BenchmarkAssessment{
				Stable: true,
				MetricStats: map[string]BenchmarkMetricSummary{
					"tokens_per_sec": {Mean: 119, Samples: 3},
				},
			},
		},
	})

	if len(selection.Survivors) != 2 {
		t.Fatalf("expected 2 survivors, got %+v", selection.Survivors)
	}
	if selection.Survivors[0].CandidateID != "triton-fast-a" {
		t.Fatalf("expected top survivor to remain best-scoring candidate, got %+v", selection.Survivors)
	}
	if selection.Survivors[1].CandidateID != "cute-alt" {
		t.Fatalf("expected second survivor to preserve lane diversity, got %+v", selection.Survivors)
	}
}

func TestEvaluateOuterLoopStatusUsesDecisionsAndCandidates(t *testing.T) {
	session := (&SessionStore{root: t.TempDir()}).NewSession(SessionCreateRequest{
		Name:        "outer-loop",
		ProjectRoot: t.TempDir(),
		Request: Request{
			Workload:  "decode",
			Operators: []string{"attention"},
		},
	})
	baseline := session.UpsertCandidate(Candidate{
		Name:      "baseline",
		Backend:   "baseline",
		Workspace: filepath.Join(session.WorkspaceRoot, "baseline"),
	})
	if err := session.RecordCandidateStage(baseline.ID, "benchmark", "/tmp/baseline.json", "python bench.py", 0, map[string]float64{"x_real_time": 1.2}); err != nil {
		t.Fatalf("record baseline stage: %v", err)
	}
	if err := session.RecordCandidateStage(baseline.ID, "profile", "/tmp/baseline-profile.json", "nsys profile python bench.py", 0, nil); err != nil {
		t.Fatalf("record profile stage: %v", err)
	}
	session.RecordLoopDecision("outer", "model-family", "tested", "turbo", "packaged turbo evaluated")
	session.RecordLoopDecision("outer", "runtime", "tested", "no-attn", "runtime flags explored")
	session.RecordLoopDecision("outer", "quantization", "blocked", "", "runtime lacks calibrated fp8 path")
	session.RecordLoopDecision("outer", "compile", "tested", "compile", "torch.compile explored")
	session.RecordLoopDecision("outer", "attention-backend", "tested", "flash-attn", "attention backend compared")
	session.SetCurrentBestCandidate(baseline.ID)

	status := EvaluateOuterLoopStatus(session)
	if !status.Exhausted || !status.ReadyForInnerLoop {
		t.Fatalf("expected inner loop gate to be ready, got %+v", status)
	}
	if status.CurrentBestID != baseline.ID {
		t.Fatalf("expected best candidate to be preserved, got %+v", status)
	}
}

func TestSaveRoundArtifactLayout(t *testing.T) {
	store := &SessionStore{root: t.TempDir()}
	session := store.NewSession(SessionCreateRequest{
		Name:        "round-store",
		ProjectRoot: t.TempDir(),
	})
	candidate := session.UpsertCandidate(Candidate{
		Name:      "triton-attention",
		Backend:   "triton",
		Workspace: filepath.Join(session.WorkspaceRoot, "triton-attention"),
	})

	path, err := SaveRoundArtifact(session, candidate.ID, 1, "diagnosis", map[string]any{"category": "memory"})
	if err != nil {
		t.Fatalf("SaveRoundArtifact() error = %v", err)
	}
	expected := filepath.Join(session.WorkspaceRoot, "candidates", candidate.ID, "rounds", "001", "diagnosis.json")
	if path != expected {
		t.Fatalf("expected %q, got %q", expected, path)
	}

	var payload map[string]any
	if err := LoadRoundArtifact(path, &payload); err != nil {
		t.Fatalf("LoadRoundArtifact() error = %v", err)
	}
	if payload["category"] != "memory" {
		t.Fatalf("expected saved payload to round-trip, got %+v", payload)
	}
}
