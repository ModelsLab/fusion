package optimize

import (
	"path/filepath"
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
