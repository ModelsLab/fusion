package kb

import "testing"

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
