package optimize

import "testing"

func TestDefaultHarnessManifestForVideoTask(t *testing.T) {
	manifest := DefaultHarnessManifest(Request{
		Task:     "video-generation",
		Workload: "sampling",
	}, "python")
	if manifest.Task != "video-generation" {
		t.Fatalf("expected normalized video-generation task, got %+v", manifest)
	}
	if manifest.PrimaryMetric != "video_frames_per_sec" {
		t.Fatalf("expected video_frames_per_sec primary metric, got %+v", manifest)
	}
	if len(manifest.Metrics) == 0 {
		t.Fatal("expected default metric specs")
	}
}

func TestAssessHarnessQualityAndBenchmark(t *testing.T) {
	manifest := DefaultHarnessManifest(Request{
		Task: "image-editing",
	}, "python")
	result := AssessHarness(manifest, []BenchmarkRunSample{
		{Metrics: map[string]float64{"images_per_sec": 2.1}},
		{Metrics: map[string]float64{"images_per_sec": 2.2}},
		{Metrics: map[string]float64{"images_per_sec": 2.0}},
	}, map[string]float64{
		"ssim":  0.95,
		"lpips": 0.1,
	})
	if result.PrimaryMetric != "images_per_sec" {
		t.Fatalf("expected images_per_sec primary metric, got %+v", result)
	}
	if !result.Benchmark.Stable {
		t.Fatalf("expected stable benchmark assessment, got %+v", result.Benchmark)
	}
}
