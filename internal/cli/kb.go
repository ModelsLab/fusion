package cli

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ModelsLab/fusion/internal/kb"
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
		newKBContextCommand(),
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
			case "skills", "skill":
				for _, skill := range runtimeState.KB.Skills {
					cmd.Printf("%s [%s]\n", skill.Title, skill.SupportLevel)
					cmd.Printf("  id: %s\n", skill.ID)
					cmd.Printf("  summary: %s\n", skill.Summary)
					cmd.Printf("  backends: %s\n", joinOrFallback(skill.PreferredBackends, "n/a"))
					cmd.Printf("  runtimes: %s\n", joinOrFallback(skill.RuntimeAdapters, "n/a"))
				}
			case "examples", "example", "patterns", "pattern":
				for _, example := range runtimeState.KB.Examples {
					cmd.Printf("%s [%s]\n", example.Title, example.SupportLevel)
					cmd.Printf("  id: %s\n", example.ID)
					cmd.Printf("  backend: %s\n", example.Backend)
					cmd.Printf("  summary: %s\n", example.Summary)
				}
			case "documents", "document", "docs", "doc", "notes", "note":
				for _, document := range runtimeState.KB.Documents {
					cmd.Printf("%s [%s/%s]\n", document.Title, valueOrFallback(document.Reliability, "curated"), valueOrFallback(document.ReviewStatus, "reviewed"))
					cmd.Printf("  id: %s\n", document.ID)
					cmd.Printf("  path: %s\n", document.Path)
					cmd.Printf("  summary: %s\n", document.Summary)
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

	cmd.Flags().StringVar(&kind, "kind", "strategies", "kind to list: strategies, gpus, sources, skills, examples, documents")
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

	cmd.Flags().StringVar(&kind, "kind", "all", "kind to search: all, strategies, gpus, sources, skills, examples, documents")
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
			case "skill":
				skill, ok := runtimeState.KB.SkillByID(id)
				if !ok {
					return fmt.Errorf("skill %q not found", id)
				}
				cmd.Printf("%s\n", skill.Title)
				cmd.Printf("id: %s\n", skill.ID)
				cmd.Printf("category: %s\n", skill.Category)
				cmd.Printf("support: %s\n", skill.SupportLevel)
				cmd.Printf("summary: %s\n", skill.Summary)
				cmd.Printf("gpu families: %s\n", joinOrFallback(skill.Triggers.GPUFamilies, "all"))
				cmd.Printf("workloads: %s\n", joinOrFallback(skill.Triggers.Workloads, "all"))
				cmd.Printf("operators: %s\n", joinOrFallback(skill.Triggers.Operators, "general"))
				cmd.Printf("precision: %s\n", joinOrFallback(skill.Triggers.Precision, "any"))
				cmd.Printf("runtimes: %s\n", joinOrFallback(skill.RuntimeAdapters, "n/a"))
				cmd.Printf("backends: %s\n", joinOrFallback(skill.PreferredBackends, "n/a"))
				cmd.Printf("tools: %s\n", joinOrFallback(skill.RequiredTools, "n/a"))
				cmd.Printf("steps: %s\n", joinOrFallback(skill.Steps, "none"))
				cmd.Printf("verification: %s\n", joinOrFallback(skill.Verification, "none"))
				cmd.Printf("benchmark rubric: %s\n", joinOrFallback(skill.BenchmarkRubric, "none"))
				cmd.Printf("recovery: %s\n", joinOrFallback(skill.FailureRecovery, "none"))
				cmd.Printf("artifacts: %s\n", joinOrFallback(skill.ArtifactsToSave, "none"))
				cmd.Printf("sources: %s\n", joinOrFallback(skill.ReferenceSourceIDs, "n/a"))
			case "example":
				example, ok := runtimeState.KB.ExampleByID(id)
				if !ok {
					return fmt.Errorf("example %q not found", id)
				}
				cmd.Printf("%s\n", example.Title)
				cmd.Printf("id: %s\n", example.ID)
				cmd.Printf("category: %s\n", example.Category)
				cmd.Printf("backend: %s\n", example.Backend)
				cmd.Printf("support: %s\n", example.SupportLevel)
				cmd.Printf("summary: %s\n", example.Summary)
				cmd.Printf("gpu families: %s\n", joinOrFallback(example.GPUFamilies, "all"))
				cmd.Printf("workloads: %s\n", joinOrFallback(example.Workloads, "all"))
				cmd.Printf("operators: %s\n", joinOrFallback(example.Operators, "general"))
				cmd.Printf("precision: %s\n", joinOrFallback(example.Precision, "any"))
				cmd.Printf("runtimes: %s\n", joinOrFallback(example.Runtimes, "n/a"))
				cmd.Printf("use cases: %s\n", joinOrFallback(example.UseCases, "n/a"))
				cmd.Printf("notes: %s\n", joinOrFallback(example.Notes, "none"))
				cmd.Printf("sources: %s\n", joinOrFallback(example.SourceIDs, "n/a"))
			case "document":
				document, ok := runtimeState.KB.DocumentByID(id)
				if !ok {
					return fmt.Errorf("document %q not found", id)
				}
				cmd.Printf("%s\n", document.Title)
				cmd.Printf("id: %s\n", document.ID)
				cmd.Printf("category: %s\n", document.Category)
				cmd.Printf("support: %s\n", valueOrFallback(document.SupportLevel, "curated"))
				cmd.Printf("reliability: %s\n", valueOrFallback(document.Reliability, "curated"))
				cmd.Printf("review status: %s\n", valueOrFallback(document.ReviewStatus, "reviewed"))
				cmd.Printf("path: %s\n", document.Path)
				cmd.Printf("url: %s\n", valueOrFallback(document.URL, "n/a"))
				cmd.Printf("summary: %s\n", document.Summary)
				cmd.Printf("gpu families: %s\n", joinOrFallback(document.GPUFamilies, "all"))
				cmd.Printf("workloads: %s\n", joinOrFallback(document.Workloads, "all"))
				cmd.Printf("operators: %s\n", joinOrFallback(document.Operators, "general"))
				cmd.Printf("precision: %s\n", joinOrFallback(document.Precision, "any"))
				cmd.Printf("runtimes: %s\n", joinOrFallback(document.Runtimes, "n/a"))
				cmd.Printf("backends: %s\n", joinOrFallback(document.Backends, "n/a"))
				cmd.Printf("sources: %s\n", joinOrFallback(document.SourceIDs, "n/a"))
				if strings.TrimSpace(document.Body) != "" {
					cmd.Printf("body:\n%s\n", document.Body)
				}
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

	cmd.Flags().StringVar(&kind, "kind", "strategy", "kind to show: strategy, gpu, source, skill, example, document")
	cmd.Flags().StringVar(&id, "id", "", "knowledge object id")
	cmd.MarkFlagRequired("id")
	return cmd
}

func newKBContextCommand() *cobra.Command {
	var req kb.ContextRequest

	cmd := &cobra.Command{
		Use:   "context",
		Short: "Build a ranked context packet for a GPU optimization request",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			packet := runtimeState.KB.BuildContextPacket(req)
			data, err := json.MarshalIndent(packet, "", "  ")
			if err != nil {
				return err
			}
			cmd.Println(string(data))
			return nil
		},
	}

	cmd.Flags().StringVar(&req.Query, "query", "", "free-form query text")
	cmd.Flags().StringVar(&req.GPU, "gpu", "", "GPU id or name")
	cmd.Flags().StringVar(&req.Model, "model", "", "model name or family")
	cmd.Flags().StringVar(&req.Workload, "workload", "", "decode, prefill, serving, or training-prep")
	cmd.Flags().StringSliceVar(&req.Operators, "operators", nil, "operator families")
	cmd.Flags().StringVar(&req.Precision, "precision", "", "precision or quantization path")
	cmd.Flags().StringVar(&req.Bottleneck, "bottleneck", "", "memory, compute, latency, or mixed")
	cmd.Flags().StringVar(&req.Runtime, "runtime", "", "runtime like vllm, tensorrt-llm, transformers, or sglang")
	cmd.Flags().StringSliceVar(&req.Goals, "goals", nil, "optimization goals")
	cmd.Flags().BoolVar(&req.IncludeExperimental, "experimental", false, "include experimental strategies, skills, and examples")
	cmd.Flags().IntVar(&req.Limit, "limit", 4, "maximum number of strategies, skills, and examples to return")
	return cmd
}
