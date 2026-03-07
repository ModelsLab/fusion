package optimize

import (
	"strings"
	"time"
)

type MetricDirection string

const (
	MetricDirectionHigherBetter MetricDirection = "higher"
	MetricDirectionLowerBetter  MetricDirection = "lower"
	MetricDirectionNeutral      MetricDirection = "neutral"
)

type MetricSpec struct {
	Key        string          `json:"key"`
	Label      string          `json:"label,omitempty"`
	Unit       string          `json:"unit,omitempty"`
	Direction  MetricDirection `json:"direction,omitempty"`
	Normalized bool            `json:"normalized,omitempty"`
}

type QualityCheck struct {
	Key       string          `json:"key"`
	Label     string          `json:"label,omitempty"`
	Direction MetricDirection `json:"direction,omitempty"`
	Threshold float64         `json:"threshold"`
	Required  bool            `json:"required,omitempty"`
}

type HarnessManifest struct {
	Version          int               `json:"version"`
	Name             string            `json:"name"`
	Task             string            `json:"task,omitempty"`
	Runtime          string            `json:"runtime,omitempty"`
	Workload         string            `json:"workload,omitempty"`
	PrimaryMetric    string            `json:"primary_metric,omitempty"`
	BenchmarkCommand string            `json:"benchmark_command,omitempty"`
	QualityCommand   string            `json:"quality_command,omitempty"`
	Target           string            `json:"target,omitempty"`
	WorkDir          string            `json:"workdir,omitempty"`
	Protocol         BenchmarkProtocol `json:"protocol"`
	Metrics          []MetricSpec      `json:"metrics,omitempty"`
	QualityChecks    []QualityCheck    `json:"quality_checks,omitempty"`
	Metadata         map[string]string `json:"metadata,omitempty"`
}

type QualityResult struct {
	Key       string          `json:"key"`
	Value     float64         `json:"value"`
	Threshold float64         `json:"threshold"`
	Direction MetricDirection `json:"direction"`
	Required  bool            `json:"required"`
	Passed    bool            `json:"passed"`
}

type QualityAssessment struct {
	Passed  bool            `json:"passed"`
	Results []QualityResult `json:"results,omitempty"`
	Notes   []string        `json:"notes,omitempty"`
}

type HarnessResult struct {
	Version       int                 `json:"version"`
	Manifest      HarnessManifest     `json:"manifest"`
	PrimaryMetric string              `json:"primary_metric,omitempty"`
	Metrics       map[string]float64  `json:"metrics,omitempty"`
	Benchmark     BenchmarkAssessment `json:"benchmark"`
	Quality       QualityAssessment   `json:"quality"`
	StartedAt     time.Time           `json:"started_at,omitempty"`
	FinishedAt    time.Time           `json:"finished_at,omitempty"`
}

func NormalizeTask(task string) string {
	task = strings.TrimSpace(strings.ToLower(task))
	task = strings.NewReplacer("_", "-", " ", "-").Replace(task)
	switch task {
	case "", "auto":
		return ""
	case "text", "textgen", "llm", "chat", "completion":
		return "text-generation"
	case "image", "img", "imagegen":
		return "image-generation"
	case "image-edit", "edit":
		return "image-editing"
	case "video", "videogen":
		return "video-generation"
	case "audio", "tts", "speech":
		return "audio-generation"
	default:
		return task
	}
}

func DefaultHarnessManifest(req Request, runtime string) HarnessManifest {
	task := NormalizeTask(req.Task)
	return HarnessManifest{
		Version:       1,
		Name:          "default-harness",
		Task:          task,
		Runtime:       strings.TrimSpace(runtime),
		Workload:      strings.TrimSpace(req.Workload),
		PrimaryMetric: DefaultPrimaryMetric(task),
		Protocol:      defaultHarnessProtocol(task),
		Metrics:       DefaultMetricSpecs(task),
		QualityChecks: DefaultQualityChecks(task),
	}
}

func defaultHarnessProtocol(task string) BenchmarkProtocol {
	protocol := DefaultBenchmarkProtocol()
	protocol.NormalizedMetrics = defaultNormalizedMetrics(task)
	return protocol
}

func defaultNormalizedMetrics(task string) []string {
	metrics := DefaultMetricSpecs(task)
	keys := make([]string, 0, len(metrics))
	for _, metric := range metrics {
		if !metric.Normalized {
			continue
		}
		keys = append(keys, metric.Key)
	}
	if len(keys) == 0 {
		return append([]string{}, DefaultBenchmarkProtocol().NormalizedMetrics...)
	}
	return keys
}

func DefaultPrimaryMetric(task string) string {
	switch NormalizeTask(task) {
	case "text-generation":
		return "tokens_per_sec"
	case "image-generation", "image-editing":
		return "images_per_sec"
	case "video-generation":
		return "video_frames_per_sec"
	case "audio-generation":
		return "x_real_time"
	default:
		return "samples_per_sec"
	}
}

func DefaultMetricSpecs(task string) []MetricSpec {
	switch NormalizeTask(task) {
	case "text-generation":
		return []MetricSpec{
			{Key: "tokens_per_sec", Unit: "tok/s", Direction: MetricDirectionHigherBetter, Normalized: true},
			{Key: "ttft_ms", Unit: "ms", Direction: MetricDirectionLowerBetter},
			{Key: "itl_ms", Unit: "ms", Direction: MetricDirectionLowerBetter},
			{Key: "memory_mb", Unit: "MB", Direction: MetricDirectionLowerBetter},
		}
	case "image-generation", "image-editing":
		return []MetricSpec{
			{Key: "images_per_sec", Unit: "img/s", Direction: MetricDirectionHigherBetter, Normalized: true},
			{Key: "latency_ms", Unit: "ms", Direction: MetricDirectionLowerBetter},
			{Key: "steps_per_sec", Unit: "steps/s", Direction: MetricDirectionHigherBetter},
			{Key: "memory_mb", Unit: "MB", Direction: MetricDirectionLowerBetter},
		}
	case "video-generation":
		return []MetricSpec{
			{Key: "video_frames_per_sec", Unit: "frames/s", Direction: MetricDirectionHigherBetter, Normalized: true},
			{Key: "clips_per_hour", Unit: "clips/h", Direction: MetricDirectionHigherBetter},
			{Key: "latency_ms", Unit: "ms", Direction: MetricDirectionLowerBetter},
			{Key: "memory_mb", Unit: "MB", Direction: MetricDirectionLowerBetter},
		}
	case "audio-generation":
		return []MetricSpec{
			{Key: "x_real_time", Unit: "x", Direction: MetricDirectionHigherBetter, Normalized: true},
			{Key: "rtf", Unit: "ratio", Direction: MetricDirectionLowerBetter, Normalized: true},
			{Key: "latency_ms", Unit: "ms", Direction: MetricDirectionLowerBetter},
			{Key: "memory_mb", Unit: "MB", Direction: MetricDirectionLowerBetter},
		}
	default:
		return []MetricSpec{
			{Key: "samples_per_sec", Unit: "samples/s", Direction: MetricDirectionHigherBetter, Normalized: true},
			{Key: "latency_ms", Unit: "ms", Direction: MetricDirectionLowerBetter},
			{Key: "memory_mb", Unit: "MB", Direction: MetricDirectionLowerBetter},
		}
	}
}

func DefaultQualityChecks(task string) []QualityCheck {
	switch NormalizeTask(task) {
	case "text-generation":
		return []QualityCheck{
			{Key: "quality_drift_pct", Direction: MetricDirectionLowerBetter, Threshold: 1.0, Required: true},
		}
	case "image-generation", "image-editing":
		return []QualityCheck{
			{Key: "clip_score", Direction: MetricDirectionHigherBetter, Threshold: 0.0, Required: false},
			{Key: "lpips", Direction: MetricDirectionLowerBetter, Threshold: 0.25, Required: false},
			{Key: "ssim", Direction: MetricDirectionHigherBetter, Threshold: 0.9, Required: false},
		}
	case "video-generation":
		return []QualityCheck{
			{Key: "fvd", Direction: MetricDirectionLowerBetter, Threshold: 120.0, Required: false},
			{Key: "temporal_consistency", Direction: MetricDirectionHigherBetter, Threshold: 0.9, Required: false},
		}
	case "audio-generation":
		return []QualityCheck{
			{Key: "wer", Direction: MetricDirectionLowerBetter, Threshold: 0.1, Required: false},
			{Key: "mos", Direction: MetricDirectionHigherBetter, Threshold: 3.8, Required: false},
		}
	default:
		return nil
	}
}

func EvaluateQuality(metrics map[string]float64, checks []QualityCheck) QualityAssessment {
	if len(checks) == 0 {
		return QualityAssessment{Passed: true}
	}
	results := make([]QualityResult, 0, len(checks))
	passed := true
	notes := []string{}
	for _, check := range checks {
		value, ok := metrics[check.Key]
		if !ok {
			if check.Required {
				passed = false
				notes = append(notes, "missing required quality metric "+check.Key)
			}
			continue
		}
		checkPassed := true
		switch check.Direction {
		case MetricDirectionLowerBetter:
			checkPassed = value <= check.Threshold
		case MetricDirectionHigherBetter:
			checkPassed = value >= check.Threshold
		default:
			checkPassed = true
		}
		if check.Required && !checkPassed {
			passed = false
		}
		results = append(results, QualityResult{
			Key:       check.Key,
			Value:     value,
			Threshold: check.Threshold,
			Direction: check.Direction,
			Required:  check.Required,
			Passed:    checkPassed,
		})
	}
	return QualityAssessment{
		Passed:  passed,
		Results: results,
		Notes:   notes,
	}
}

func AssessHarness(manifest HarnessManifest, samples []BenchmarkRunSample, qualityMetrics map[string]float64) HarnessResult {
	if manifest.Version == 0 {
		manifest.Version = 1
	}
	if manifest.PrimaryMetric == "" {
		manifest.PrimaryMetric = DefaultPrimaryMetric(manifest.Task)
	}
	if len(manifest.Metrics) == 0 {
		manifest.Metrics = DefaultMetricSpecs(manifest.Task)
	}
	if len(manifest.QualityChecks) == 0 {
		manifest.QualityChecks = DefaultQualityChecks(manifest.Task)
	}
	if len(manifest.Protocol.NormalizedMetrics) == 0 {
		manifest.Protocol.NormalizedMetrics = defaultNormalizedMetrics(manifest.Task)
	}
	assessment := manifest.Protocol.Evaluate(samples)
	aggregated := map[string]float64{}
	for key, stat := range assessment.MetricStats {
		aggregated[key] = stat.Mean
	}
	quality := EvaluateQuality(qualityMetrics, manifest.QualityChecks)
	return HarnessResult{
		Version:       1,
		Manifest:      manifest,
		PrimaryMetric: manifest.PrimaryMetric,
		Metrics:       aggregated,
		Benchmark:     assessment,
		Quality:       quality,
	}
}
