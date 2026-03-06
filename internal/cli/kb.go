package cli

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"
)

func newKnowledgeCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "kb",
		Short: "Browse the embedded optimization knowledge base",
	}

	cmd.AddCommand(
		newKBListCommand(),
		newKBSearchCommand(),
		newKBShowCommand(),
	)

	return cmd
}

func newKBListCommand() *cobra.Command {
	var kind string

	cmd := &cobra.Command{
		Use:   "list",
		Short: "List embedded knowledge objects",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			switch strings.ToLower(strings.TrimSpace(kind)) {
			case "sources", "source":
				for _, source := range runtimeState.KB.Sources {
					cmd.Printf("%s [%s/%s]\n", source.ID, source.Reliability, source.ReviewStatus)
					cmd.Printf("  %s\n", source.Title)
					cmd.Printf("  %s\n", source.URL)
				}
			case "gpus", "gpu":
				for _, gpu := range runtimeState.KB.GPUs {
					cmd.Printf("%s (%s, cc %s)\n", gpu.Name, gpu.Family, gpu.ComputeCapability)
					cmd.Printf("  preferred: %s\n", joinOrFallback(gpu.PreferredPrecisions, "n/a"))
					cmd.Printf("  strengths: %s\n", joinOrFallback(gpu.Strengths, "n/a"))
				}
			default:
				for _, strategy := range runtimeState.KB.Strategies {
					cmd.Printf("%s [%s]\n", strategy.Title, strategy.SupportLevel)
					cmd.Printf("  id: %s\n", strategy.ID)
					cmd.Printf("  summary: %s\n", strategy.Summary)
					cmd.Printf("  workloads: %s\n", joinOrFallback(strategy.Workloads, "all"))
					cmd.Printf("  operators: %s\n", joinOrFallback(strategy.Operators, "general"))
				}
			}

			return nil
		},
	}

	cmd.Flags().StringVar(&kind, "kind", "strategies", "kind to list: strategies, gpus, sources")
	return cmd
}

func newKBSearchCommand() *cobra.Command {
	var kind string
	var limit int

	cmd := &cobra.Command{
		Use:   "search [query]",
		Short: "Search strategies, GPUs, and sources",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			query := strings.Join(args, " ")
			hits := runtimeState.KB.Search(query, kind, limit)
			if len(hits) == 0 {
				cmd.Printf("No knowledge hits for %q\n", query)
				return nil
			}

			for _, hit := range hits {
				cmd.Printf("[%s] %s\n", hit.Kind, hit.Title)
				cmd.Printf("  id: %s\n", hit.ID)
				cmd.Printf("  score: %d\n", hit.Score)
				cmd.Printf("  %s\n", hit.Summary)
			}

			return nil
		},
	}

	cmd.Flags().StringVar(&kind, "kind", "all", "kind to search: all, strategies, gpus, sources")
	cmd.Flags().IntVar(&limit, "limit", 8, "maximum number of results")
	return cmd
}

func newKBShowCommand() *cobra.Command {
	var kind string
	var id string

	cmd := &cobra.Command{
		Use:   "show",
		Short: "Show one knowledge object in detail",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			switch strings.ToLower(strings.TrimSpace(kind)) {
			case "source":
				source, ok := runtimeState.KB.SourceByID(id)
				if !ok {
					return fmt.Errorf("source %q not found", id)
				}
				cmd.Printf("%s\n", source.Title)
				cmd.Printf("id: %s\n", source.ID)
				cmd.Printf("type: %s/%s\n", source.Type, source.Category)
				cmd.Printf("reliability: %s\n", source.Reliability)
				cmd.Printf("review status: %s\n", source.ReviewStatus)
				cmd.Printf("url: %s\n", source.URL)
				cmd.Printf("summary: %s\n", source.Summary)
				cmd.Printf("tags: %s\n", joinOrFallback(source.Tags, "n/a"))
			case "gpu":
				gpu, ok := runtimeState.KB.GPUByID(id)
				if !ok {
					return fmt.Errorf("gpu %q not found", id)
				}
				cmd.Printf("%s\n", gpu.Name)
				cmd.Printf("id: %s\n", gpu.ID)
				cmd.Printf("family: %s\n", gpu.Family)
				cmd.Printf("market: %s\n", gpu.Market)
				cmd.Printf("compute capability: %s\n", gpu.ComputeCapability)
				cmd.Printf("memory: %d GB\n", gpu.MemoryGB)
				cmd.Printf("bandwidth: %d GB/s\n", gpu.MemoryBandwidthGBps)
				cmd.Printf("preferred precisions: %s\n", joinOrFallback(gpu.PreferredPrecisions, "n/a"))
				cmd.Printf("strengths: %s\n", joinOrFallback(gpu.Strengths, "n/a"))
				cmd.Printf("constraints: %s\n", joinOrFallback(gpu.Constraints, "n/a"))
				cmd.Printf("sources: %s\n", joinOrFallback(gpu.SourceIDs, "n/a"))
			default:
				strategy, ok := runtimeState.KB.StrategyByID(id)
				if !ok {
					return fmt.Errorf("strategy %q not found", id)
				}
				cmd.Printf("%s\n", strategy.Title)
				cmd.Printf("id: %s\n", strategy.ID)
				cmd.Printf("category: %s\n", strategy.Category)
				cmd.Printf("support: %s\n", strategy.SupportLevel)
				cmd.Printf("summary: %s\n", strategy.Summary)
				cmd.Printf("workloads: %s\n", joinOrFallback(strategy.Workloads, "all"))
				cmd.Printf("operators: %s\n", joinOrFallback(strategy.Operators, "general"))
				cmd.Printf("precision: %s\n", joinOrFallback(strategy.Precision, "any"))
				cmd.Printf("goals: %s\n", joinOrFallback(strategy.Goals, "n/a"))
				cmd.Printf("preconditions: %s\n", joinOrFallback(strategy.Preconditions, "none"))
				cmd.Printf("actions: %s\n", joinOrFallback(strategy.Actions, "none"))
				cmd.Printf("metrics: %s\n", joinOrFallback(strategy.Metrics, "none"))
				cmd.Printf("tradeoffs: %s\n", joinOrFallback(strategy.Tradeoffs, "none"))
				cmd.Printf("sources: %s\n", joinOrFallback(strategy.SourceIDs, "n/a"))
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&kind, "kind", "strategy", "kind to show: strategy, gpu, source")
	cmd.Flags().StringVar(&id, "id", "", "knowledge object id")
	cmd.MarkFlagRequired("id")
	return cmd
}
