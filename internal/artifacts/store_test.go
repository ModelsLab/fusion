package artifacts

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestStorePathsSanitizeIDs(t *testing.T) {
	store := &Store{root: t.TempDir()}

	benchmarkPath := store.BenchmarkPath("../../etc/passwd")
	if !strings.HasPrefix(benchmarkPath, filepath.Join(store.root, "benchmarks")+string(filepath.Separator)) {
		t.Fatalf("expected benchmark path to remain under store root, got %q", benchmarkPath)
	}
	if strings.Contains(benchmarkPath, "..") {
		t.Fatalf("expected benchmark path to be sanitized, got %q", benchmarkPath)
	}

	profilePath := store.ProfilePath("../../../var/log")
	if !strings.HasPrefix(profilePath, filepath.Join(store.root, "profiles")+string(filepath.Separator)) {
		t.Fatalf("expected profile path to remain under store root, got %q", profilePath)
	}
	if strings.Contains(profilePath, "..") {
		t.Fatalf("expected profile path to be sanitized, got %q", profilePath)
	}
}
