package optimize

import (
	"testing"

	"github.com/ModelsLab/fusion/internal/kb"
)

func TestDecodePlanPrefersPagedKVCache(t *testing.T) {
	store, err := kb.LoadDefault()
	if err != nil {
		t.Fatalf("LoadDefault() error = %v", err)
	}

	planner := NewPlanner(store)
	plan, err := planner.Build(Request{
		GPU:       "rtx4090",
		Workload:  "decode",
		Operators: []string{"attention", "kv-cache"},
		Precision: "bf16",
	})
	if err != nil {
		t.Fatalf("Build() error = %v", err)
	}

	if len(plan.Recommendations) == 0 {
		t.Fatal("expected recommendations")
	}

	found := false
	for _, recommendation := range plan.Recommendations {
		if recommendation.Strategy.ID == "paged_kv_cache" {
			found = true
		}
	}

	if !found {
		t.Fatal("expected paged_kv_cache in top recommendations")
	}
}

func TestFP8PlanOnB200IncludesFP8Strategy(t *testing.T) {
	store, err := kb.LoadDefault()
	if err != nil {
		t.Fatalf("LoadDefault() error = %v", err)
	}

	planner := NewPlanner(store)
	plan, err := planner.Build(Request{
		GPU:       "b200",
		Workload:  "prefill",
		Operators: []string{"matmul"},
		Precision: "fp8",
	})
	if err != nil {
		t.Fatalf("Build() error = %v", err)
	}

	found := false
	for _, recommendation := range plan.Recommendations {
		if recommendation.Strategy.ID == "fp8_transformer_engine" {
			found = true
		}
	}

	if !found {
		t.Fatal("expected fp8_transformer_engine in recommendations")
	}
}

func TestPrefillPlanRecommendsCuteBackendOnHopper(t *testing.T) {
	store, err := kb.LoadDefault()
	if err != nil {
		t.Fatalf("LoadDefault() error = %v", err)
	}

	planner := NewPlanner(store)
	plan, err := planner.Build(Request{
		GPU:       "h100",
		Workload:  "prefill",
		Operators: []string{"matmul", "attention"},
		Precision: "fp8",
		Goals:     []string{"throughput"},
	})
	if err != nil {
		t.Fatalf("Build() error = %v", err)
	}

	if len(plan.KernelBackends) == 0 {
		t.Fatal("expected kernel backend recommendations")
	}
	if plan.KernelBackends[0].ID != "cute_dsl" {
		t.Fatalf("expected CuTe DSL to be ranked first, got %q", plan.KernelBackends[0].ID)
	}
}

func TestDecodePlanIncludesTritonBackend(t *testing.T) {
	store, err := kb.LoadDefault()
	if err != nil {
		t.Fatalf("LoadDefault() error = %v", err)
	}

	planner := NewPlanner(store)
	plan, err := planner.Build(Request{
		GPU:        "rtx4090",
		Workload:   "decode",
		Operators:  []string{"attention", "kv-cache", "layernorm"},
		Precision:  "bf16",
		Bottleneck: "memory",
	})
	if err != nil {
		t.Fatalf("Build() error = %v", err)
	}

	found := false
	for _, backend := range plan.KernelBackends {
		if backend.ID == "triton" {
			found = true
		}
	}

	if !found {
		t.Fatal("expected Triton in kernel backend recommendations")
	}
}

func TestLargeConsumerDecodePlanPrefersAWQTrack(t *testing.T) {
	store, err := kb.LoadDefault()
	if err != nil {
		t.Fatalf("LoadDefault() error = %v", err)
	}

	planner := NewPlanner(store)
	plan, err := planner.Build(Request{
		GPU:        "rtx4090",
		Model:      "qwen2.5-72b",
		Workload:   "decode",
		Operators:  []string{"attention", "kv-cache", "quantization"},
		Bottleneck: "memory",
		Goals:      []string{"memory", "cost"},
	})
	if err != nil {
		t.Fatalf("Build() error = %v", err)
	}

	if len(plan.ModelPaths) == 0 {
		t.Fatal("expected model path recommendations")
	}
	if plan.ModelPaths[0].ID != "awq_int4_weights" {
		t.Fatalf("expected AWQ INT4 to be ranked first, got %q", plan.ModelPaths[0].ID)
	}
}

func TestBlackwellPlanIncludesNVFP4Track(t *testing.T) {
	store, err := kb.LoadDefault()
	if err != nil {
		t.Fatalf("LoadDefault() error = %v", err)
	}

	planner := NewPlanner(store)
	plan, err := planner.Build(Request{
		GPU:       "b200",
		Model:     "llama-3.1-70b",
		Workload:  "prefill",
		Operators: []string{"matmul", "attention"},
		Goals:     []string{"throughput"},
	})
	if err != nil {
		t.Fatalf("Build() error = %v", err)
	}

	found := false
	for _, track := range plan.ModelPaths {
		if track.ID == "nvfp4_block_scaled" {
			found = true
		}
	}
	if !found {
		t.Fatal("expected NVFP4 model track for Blackwell plan")
	}
}
