package cli

import (
	"sort"
	"strings"
	"time"

	"github.com/ModelsLab/fusion/internal/artifacts"
	"github.com/ModelsLab/fusion/internal/runner"
	"github.com/spf13/cobra"
)

func newBenchmarkCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "benchmark",
		Short: "Run and compare benchmark artifacts on local, SSH, or sim targets",
	}

	cmd.AddCommand(
		newBenchmarkRunCommand(),
		newBenchmarkCompareCommand(),
	)
	return cmd
}

func newBenchmarkRunCommand() *cobra.Command {
	var targetName string
	var runName string
	var command string
	var timeout time.Duration
	var outputPath string
	var metricsText string

	cmd := &cobra.Command{
		Use:   "run",
		Short: "Execute a benchmark command and persist the artifact",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			target, resolvedName, err := resolveTarget(runtimeState, targetName)
			if err != nil {
				return err
			}
			if runName == "" {
				runName = resolvedName + "-benchmark"
			}

			runResult, err := runner.Execute(runner.Request{
				Target:  target,
				Command: command,
				Timeout: timeout,
			})
			if err != nil && runResult.ExitCode == 0 {
				return err
			}

			metrics := artifacts.ParseMetrics(runResult.Stdout)
			if metricsText != "" {
				for key, value := range artifacts.ParseMetrics(metricsText) {
					if metrics == nil {
						metrics = map[string]float64{}
					}
					metrics[key] = value
				}
			}

			store, err := artifacts.NewStore()
			if err != nil {
				return err
			}

			artifact := artifacts.BenchmarkResult{
				Name:          runName,
				TargetName:    target.Name,
				TargetMode:    target.Mode,
				Command:       command,
				Metrics:       metrics,
				StartedAt:     runResult.StartedAt,
				FinishedAt:    runResult.FinishedAt,
				DurationMS:    runResult.DurationMS,
				ExitCode:      runResult.ExitCode,
				Stdout:        runResult.Stdout,
				Stderr:        runResult.Stderr,
				Authoritative: runResult.Authoritative,
				Simulated:     runResult.Simulated,
				Warnings:      runResult.Warnings,
				Run:           runResult,
			}

			path, saveErr := store.SaveBenchmark(artifact, outputPath)
			if saveErr != nil {
				return saveErr
			}

			cmd.Printf("Saved benchmark artifact: %s\n", path)
			cmd.Printf("Target: %s (%s)\n", target.Name, target.Mode)
			cmd.Printf("Wall time: %d ms\n", artifact.DurationMS)
			if len(metrics) > 0 {
				cmd.Println("Metrics")
				keys := sortedMetricKeys(metrics)
				for _, key := range keys {
					cmd.Printf("- %s: %.4f\n", key, metrics[key])
				}
			}
			for _, warning := range artifact.Warnings {
				cmd.Printf("warning: %s\n", warning)
			}
			if err != nil {
				cmd.Printf("command exited with error: %v\n", err)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&targetName, "target", "", "target name; defaults to the configured default target or implicit local")
	cmd.Flags().StringVar(&runName, "name", "", "benchmark run name")
	cmd.Flags().StringVar(&command, "command", "", "shell command to execute; JSON metrics in stdout will be parsed automatically")
	cmd.Flags().StringVar(&metricsText, "metrics", "", "optional inline metrics like 'tokens_per_sec=42 latency_ms=10.5'")
	cmd.Flags().StringVar(&outputPath, "output", "", "custom artifact output path")
	cmd.Flags().DurationVar(&timeout, "timeout", 30*time.Minute, "command timeout")
	cmd.MarkFlagRequired("command")
	return cmd
}

func newBenchmarkCompareCommand() *cobra.Command {
	var beforePath string
	var afterPath string

	cmd := &cobra.Command{
		Use:   "compare",
		Short: "Compare before and after benchmark artifacts",
		RunE: func(cmd *cobra.Command, args []string) error {
			store, err := artifacts.NewStore()
			if err != nil {
				return err
			}

			before, err := store.LoadBenchmark(beforePath)
			if err != nil {
				return err
			}
			after, err := store.LoadBenchmark(afterPath)
			if err != nil {
				return err
			}

			cmd.Printf("Before: %s\n", beforePath)
			cmd.Printf("After:  %s\n", afterPath)
			cmd.Printf("Targets: %s -> %s\n", before.TargetName, after.TargetName)

			printComparisonLine(cmd, "wall_time_ms", float64(before.DurationMS), float64(after.DurationMS))

			keys := commonMetricKeys(before.Metrics, after.Metrics)
			if len(keys) > 0 {
				cmd.Println("\nMetrics")
				for _, key := range keys {
					printComparisonLine(cmd, key, before.Metrics[key], after.Metrics[key])
				}
			}

			if before.Simulated || after.Simulated {
				cmd.Println("\nWarnings")
				cmd.Println("- At least one artifact came from sim mode, so performance deltas are not authoritative.")
			}

			return nil
		},
	}

	cmd.Flags().StringVar(&beforePath, "before", "", "path to the baseline benchmark artifact")
	cmd.Flags().StringVar(&afterPath, "after", "", "path to the optimized benchmark artifact")
	cmd.MarkFlagRequired("before")
	cmd.MarkFlagRequired("after")
	return cmd
}

func printComparisonLine(cmd *cobra.Command, key string, before, after float64) {
	if before == 0 {
		cmd.Printf("- %s: before=%.4f after=%.4f\n", key, before, after)
		return
	}

	delta := after - before
	deltaPct := (delta / before) * 100
	if lowerIsBetter(key) {
		speedup := before / after
		cmd.Printf("- %s: %.4f -> %.4f (delta %.2f%%, speedup %.2fx, lower is better)\n", key, before, after, deltaPct, speedup)
		return
	}

	speedup := after / before
	cmd.Printf("- %s: %.4f -> %.4f (delta %.2f%%, speedup %.2fx, higher is better)\n", key, before, after, deltaPct, speedup)
}

func lowerIsBetter(key string) bool {
	key = strings.ToLower(key)
	return strings.Contains(key, "latency") ||
		strings.Contains(key, "time") ||
		strings.Contains(key, "_ms") ||
		strings.Contains(key, "memory") ||
		strings.Contains(key, "bytes")
}

func commonMetricKeys(left, right map[string]float64) []string {
	if len(left) == 0 || len(right) == 0 {
		return nil
	}
	keys := []string{}
	for key := range left {
		if _, ok := right[key]; ok {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	return keys
}

func sortedMetricKeys(metrics map[string]float64) []string {
	keys := make([]string, 0, len(metrics))
	for key := range metrics {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
