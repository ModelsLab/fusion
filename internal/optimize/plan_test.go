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
