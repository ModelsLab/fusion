package cli

import (
	"encoding/json"
	"os"
	"strings"

	"github.com/ModelsLab/fusion/internal/artifacts"
	"github.com/ModelsLab/fusion/internal/optimize"
	"github.com/spf13/cobra"
)

func newOptimizeHarnessCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "harness",
		Short: "Create and assess generic model benchmark and quality harness manifests",
	}
	cmd.AddCommand(
		newOptimizeHarnessInitCommand(),
		newOptimizeHarnessAssessCommand(),
	)
	return cmd
}

func newOptimizeHarnessInitCommand() *cobra.Command {
	var request optimize.Request
	var runtimeName string
	var name string
	var outputPath string

	cmd := &cobra.Command{
		Use:   "init",
		Short: "Create a generic harness manifest for the current optimization task",
		RunE: func(cmd *cobra.Command, args []string) error {
			manifest := optimize.DefaultHarnessManifest(request, runtimeName)
			if strings.TrimSpace(name) != "" {
				manifest.Name = strings.TrimSpace(name)
			}
			data, err := json.MarshalIndent(manifest, "", "  ")
			if err != nil {
				return err
			}
			data = append(data, '\n')
			if outputPath == "" {
				outputPath = manifest.Name + ".harness.json"
			}
			if err := os.WriteFile(outputPath, data, 0o600); err != nil {
				return err
			}
			cmd.Printf("Saved harness manifest: %s\n", outputPath)
			cmd.Printf("Task: %s\n", valueOrFallback(manifest.Task, "unspecified"))
			cmd.Printf("Primary metric: %s\n", valueOrFallback(manifest.PrimaryMetric, "unspecified"))
			return nil
		},
	}

	cmd.Flags().StringVar(&name, "name", "default-harness", "manifest name")
	cmd.Flags().StringVar(&outputPath, "output", "", "output path for the harness manifest JSON")
	cmd.Flags().StringVar(&request.Task, "task", "", "task family like text-generation, image-generation, image-editing, video-generation, or audio-generation")
	cmd.Flags().StringVar(&request.Workload, "workload", "", "workload description like decode, denoise, sampling, or refinement")
	cmd.Flags().StringVar(&runtimeName, "runtime", "", "runtime label")
	return cmd
}

func newOptimizeHarnessAssessCommand() *cobra.Command {
	var manifestPath string
	var samplesPath string
	var qualityText string
	var outputPath string

	cmd := &cobra.Command{
		Use:   "assess",
		Short: "Assess repeated benchmark samples and quality metrics against a harness manifest",
		RunE: func(cmd *cobra.Command, args []string) error {
			var manifest optimize.HarnessManifest
			data, err := os.ReadFile(manifestPath)
			if err != nil {
				return err
			}
			if err := json.Unmarshal(data, &manifest); err != nil {
				return err
			}

			var samples []optimize.BenchmarkRunSample
			sampleData, err := os.ReadFile(samplesPath)
			if err != nil {
				return err
			}
			if err := json.Unmarshal(sampleData, &samples); err != nil {
				return err
			}

			qualityMetrics := map[string]float64{}
			if strings.TrimSpace(qualityText) != "" {
				for key, value := range artifacts.ParseMetrics(qualityText) {
					qualityMetrics[key] = value
				}
			}

			result := optimize.AssessHarness(manifest, samples, qualityMetrics)
			encoded, err := json.MarshalIndent(result, "", "  ")
			if err != nil {
				return err
			}
			encoded = append(encoded, '\n')
			if outputPath == "" {
				outputPath = strings.TrimSuffix(manifestPath, ".json") + ".result.json"
			}
			if err := os.WriteFile(outputPath, encoded, 0o600); err != nil {
				return err
			}
			cmd.Printf("Saved harness result: %s\n", outputPath)
			cmd.Printf("Primary metric: %s\n", valueOrFallback(result.PrimaryMetric, "unspecified"))
			cmd.Printf("Stable benchmark: %t\n", result.Benchmark.Stable)
			cmd.Printf("Quality passed: %t\n", result.Quality.Passed)
			return nil
		},
	}

	cmd.Flags().StringVar(&manifestPath, "manifest", "", "path to a harness manifest JSON")
	cmd.Flags().StringVar(&samplesPath, "samples", "", "path to repeated benchmark samples JSON")
	cmd.Flags().StringVar(&qualityText, "quality", "", "optional inline quality metrics like 'ssim=0.93 lpips=0.12'")
	cmd.Flags().StringVar(&outputPath, "output", "", "optional output path for the harness result JSON")
	cmd.MarkFlagRequired("manifest")
	cmd.MarkFlagRequired("samples")
	return cmd
}
