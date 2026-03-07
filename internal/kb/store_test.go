package kb

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadDefault(t *testing.T) {
	store, err := LoadDefault()
	if err != nil {
		t.Fatalf("LoadDefault() error = %v", err)
	}

	if len(store.GPUs) == 0 {
		t.Fatal("expected curated GPU profiles")
	}
	if len(store.Strategies) == 0 {
		t.Fatal("expected curated strategies")
	}
	if len(store.Sources) == 0 {
		t.Fatal("expected curated sources")
	}
	if len(store.Skills) == 0 {
		t.Fatal("expected curated skills")
	}
	if len(store.Examples) == 0 {
		t.Fatal("expected curated examples")
	}
}

func TestSearchFindsAttentionStrategy(t *testing.T) {
	store, err := LoadDefault()
	if err != nil {
		t.Fatalf("LoadDefault() error = %v", err)
	}

	results := store.Search("paged attention", "strategy", 5)
	if len(results) == 0 {
		t.Fatal("expected paged attention search hits")
	}
}

func TestSearchFindsBlackwellSkill(t *testing.T) {
	store, err := LoadDefault()
	if err != nil {
		t.Fatalf("LoadDefault() error = %v", err)
	}

	results := store.Search("blackwell attention cutile", "skill", 5)
	if len(results) == 0 {
		t.Fatal("expected blackwell skill search hits")
	}
	if results[0].Kind != "skill" {
		t.Fatalf("expected skill hit, got %q", results[0].Kind)
	}
}

func TestContextPacketForB200DecodeReturnsBlackwellGuidance(t *testing.T) {
	store, err := LoadDefault()
	if err != nil {
		t.Fatalf("LoadDefault() error = %v", err)
	}

	packet := store.BuildContextPacket(ContextRequest{
		Query:               "optimize qwen decode attention on b200",
		GPU:                 "b200",
		Model:               "qwen2.5-72b",
		Workload:            "decode",
		Operators:           []string{"attention", "kv-cache"},
		Precision:           "fp8",
		Runtime:             "vllm",
		Goals:               []string{"latency", "throughput"},
		IncludeExperimental: true,
		Limit:               4,
	})

	if packet.GPU == nil || packet.GPU.ID != "b200" {
		t.Fatal("expected packet to resolve the B200 profile")
	}

	foundStrategy := false
	for _, match := range packet.Strategies {
		if match.Strategy.ID == "blackwell_attention_cutile" {
			foundStrategy = true
		}
	}
	if !foundStrategy {
		t.Fatal("expected blackwell attention strategy in context packet")
	}

	foundSkill := false
	for _, match := range packet.Skills {
		if match.Skill.ID == "skill_blackwell_attention_tuning" {
			foundSkill = true
		}
	}
	if !foundSkill {
		t.Fatal("expected blackwell attention skill in context packet")
	}

	foundExample := false
	for _, match := range packet.Examples {
		if match.Example.ID == "cuda_tile_flash_attention_blackwell" {
			foundExample = true
		}
	}
	if !foundExample {
		t.Fatal("expected Blackwell CUDA Tile example in context packet")
	}
}

func TestContextPacketKeepsMultimodalWorkloadUnsetByDefault(t *testing.T) {
	store, err := LoadDefault()
	if err != nil {
		t.Fatalf("LoadDefault() error = %v", err)
	}

	packet := store.BuildContextPacket(ContextRequest{
		Task:  "video-generation",
		Query: "optimize a video generation pipeline for latency and quality",
		Limit: 4,
	})

	if packet.Request.Workload != "" {
		t.Fatalf("expected multimodal task to keep workload unset when none is provided, got %q", packet.Request.Workload)
	}
}

func TestLoadDefaultPrefersUserKnowledgeDB(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(home, ".config"))

	configDir, err := os.UserConfigDir()
	if err != nil {
		t.Fatalf("UserConfigDir() error = %v", err)
	}

	localStore := &Store{
		Sources: []Source{
			{
				ID:           "local-source",
				Title:        "Local Source",
				URL:          "https://example.com/local",
				Type:         "note",
				Category:     "local",
				Reliability:  "curated",
				ReviewStatus: "reviewed",
				Summary:      "Local-only knowledge for test coverage.",
			},
		},
		Documents: []Document{
			{
				ID:           "local-doc",
				Title:        "Local Doc",
				Category:     "test",
				Summary:      "Should override embedded KB.",
				SupportLevel: "recommended",
				Reliability:  "curated",
				ReviewStatus: "reviewed",
				Body:         "This document only exists in the user-local database.",
			},
		},
	}

	dbPath := filepath.Join(configDir, "fusion", "knowledge", "knowledge.db")
	if err := WriteSQLiteIndex(localStore, dbPath); err != nil {
		t.Fatalf("WriteSQLiteIndex() error = %v", err)
	}

	loaded, err := LoadDefault()
	if err != nil {
		t.Fatalf("LoadDefault() error = %v", err)
	}
	if len(loaded.Documents) != 1 || loaded.Documents[0].ID != "local-doc" {
		t.Fatalf("expected LoadDefault to prefer user-local knowledge db, got %#v", loaded.Documents)
	}
	if _, ok := loaded.SourceByID("local-source"); !ok {
		t.Fatal("expected local source to be loaded from user-local knowledge db")
	}
}
