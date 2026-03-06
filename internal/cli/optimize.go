package cli

import (
	"strings"

	"github.com/ModelsLab/fusion/internal/optimize"
	"github.com/ModelsLab/fusion/internal/system"
	"github.com/spf13/cobra"
)

func newOptimizeCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "optimize",
		Short: "Plan model and kernel optimization work for a target GPU",
	}

	cmd.AddCommand(newOptimizePlanCommand())
	return cmd
}

func newOptimizePlanCommand() *cobra.Command {
	var request optimize.Request

	cmd := &cobra.Command{
		Use:   "plan",
		Short: "Build a ranked optimization plan from the embedded knowledge base",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			if strings.TrimSpace(request.GPU) == "" {
				if detected := system.DetectNVIDIAGPUs(); len(detected) > 0 {
					request.GPU = detected[0].Name
				}
			}

			planner := optimize.NewPlanner(runtimeState.KB)
			plan, err := planner.Build(request)
			if err != nil {
				return err
			}

			cmd.Println("Target")
			if plan.GPU != nil {
				cmd.Printf("- GPU: %s (%s, cc %s)\n", plan.GPU.Name, plan.GPU.Family, plan.GPU.ComputeCapability)
			} else if request.GPU != "" {
				cmd.Printf("- GPU: %s (not yet normalized in the knowledge base)\n", request.GPU)
			} else {
				cmd.Println("- GPU: unspecified")
			}
			cmd.Printf("- Model: %s\n", valueOrFallback(request.Model, "unspecified"))
			cmd.Printf("- Workload: %s\n", valueOrFallback(plan.Request.Workload, "decode"))
			cmd.Printf("- Precision: %s\n", valueOrFallback(plan.Request.Precision, "bf16"))
			cmd.Printf("- Operators: %s\n", joinOrFallback(plan.Request.Operators, "general transformer path"))
			cmd.Printf("- Likely bottleneck: %s\n", plan.LikelyBottleneck)
			cmd.Printf("  %s\n", plan.BottleneckReason)

			if len(plan.PreferredPrecisionHints) > 0 {
				cmd.Printf("- GPU-preferred precisions: %s\n", strings.Join(plan.PreferredPrecisionHints, ", "))
			}
			if len(plan.ExperimentalPrecisionSet) > 0 {
				cmd.Printf("- Experimental precisions: %s\n", strings.Join(plan.ExperimentalPrecisionSet, ", "))
			}

			if len(plan.Warnings) > 0 {
				cmd.Println("\nWarnings")
				for _, warning := range plan.Warnings {
					cmd.Printf("- %s\n", warning)
				}
			}

			if len(plan.Priorities) > 0 {
				cmd.Println("\nPriorities")
				for _, priority := range plan.Priorities {
					cmd.Printf("- %s\n", priority)
				}
			}

			if len(plan.MeasurementLoop) > 0 {
				cmd.Println("\nMeasurement Loop")
				for i, step := range plan.MeasurementLoop {
					cmd.Printf("%d. %s\n", i+1, step)
				}
			}

			cmd.Println("\nTop Strategies")
			if len(plan.Recommendations) == 0 {
				cmd.Println("- no strategies matched the current request")
			}
			for i, recommendation := range plan.Recommendations {
				cmd.Printf("%d. %s [%s, score %d]\n", i+1, recommendation.Strategy.Title, recommendation.Strategy.SupportLevel, recommendation.Score)
				cmd.Printf("   %s\n", recommendation.Strategy.Summary)
				cmd.Printf("   Why: %s\n", strings.Join(recommendation.Reasons, "; "))
				cmd.Printf("   Actions: %s\n", joinOrFallback(recommendation.Strategy.Actions, "none"))
				cmd.Printf("   Metrics: %s\n", joinOrFallback(recommendation.Strategy.Metrics, "none"))
				cmd.Printf("   Tradeoffs: %s\n", joinOrFallback(recommendation.Strategy.Tradeoffs, "none"))
				cmd.Printf("   Sources: %s\n", formatSourceList(recommendation.Sources))
			}

			if len(plan.SupportingSources) > 0 {
				cmd.Println("\nKnowledge Base Coverage")
				maxSources := len(plan.SupportingSources)
				if maxSources > 8 {
					maxSources = 8
				}
				for _, source := range plan.SupportingSources[:maxSources] {
					cmd.Printf("- %s [%s] %s\n", source.Title, source.Reliability, source.URL)
				}
			}

			return nil
		},
	}

	cmd.Flags().StringVar(&request.GPU, "gpu", "", "target GPU id or name; defaults to the first detected NVIDIA GPU when available")
	cmd.Flags().StringVar(&request.Model, "model", "", "model name or family, for example llama-3.1-8b or qwen2.5-32b")
	cmd.Flags().StringVar(&request.Workload, "workload", "decode", "workload shape: decode, prefill, serving, training-prep")
	cmd.Flags().StringSliceVar(&request.Operators, "operator", nil, "operator families to optimize; repeat or comma-separate")
	cmd.Flags().StringVar(&request.Precision, "precision", "bf16", "target precision, for example bf16, fp16, fp8, int4")
	cmd.Flags().StringVar(&request.Bottleneck, "bottleneck", "", "override the inferred bottleneck: memory, compute, latency, mixed")
	cmd.Flags().StringSliceVar(&request.Goals, "goal", nil, "optimization goals such as throughput, latency, memory, cost")
	cmd.Flags().IntVar(&request.BatchSize, "batch-size", 1, "representative batch size")
	cmd.Flags().IntVar(&request.ContextLength, "context-length", 0, "representative prompt or total context length")
	cmd.Flags().BoolVar(&request.IncludeExperimental, "experimental", false, "include experimental strategies from the knowledge base")
	return cmd
}

func valueOrFallback(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	return value
}
