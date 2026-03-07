package cli

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ModelsLab/fusion/internal/kb"
	"github.com/ModelsLab/fusion/internal/optimize"
	"github.com/ModelsLab/fusion/internal/system"
	"github.com/spf13/cobra"
)

func newOptimizeSessionCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "session",
		Short: "Manage persistent optimization sessions and backend candidates",
	}

	cmd.AddCommand(
		newOptimizeSessionCreateCommand(),
		newOptimizeSessionListCommand(),
		newOptimizeSessionShowCommand(),
		newOptimizeSessionGateCommand(),
		newOptimizeSessionDecisionCommand(),
	)

	return cmd
}

func newOptimizeSessionCreateCommand() *cobra.Command {
	var request optimize.Request
	var name string
	var query string
	var runtimeName string
	var workspaceRoot string
	var targetName string
	var includeExperimental bool
	var limit int

	cmd := &cobra.Command{
		Use:   "create",
		Short: "Create an optimization session with a retrieved context packet",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			projectRoot, err := os.Getwd()
			if err != nil {
				return fmt.Errorf("resolve current working directory: %w", err)
			}

			if strings.TrimSpace(targetName) != "" {
				target, _, err := resolveTarget(runtimeState, targetName)
				if err != nil {
					return err
				}
				if strings.TrimSpace(request.GPU) == "" {
					request.GPU = target.GPU
				}
			}

			if strings.TrimSpace(request.GPU) == "" {
				if detected := system.DetectNVIDIAGPUs(); len(detected) > 0 {
					request.GPU = detected[0].Name
				}
			}

			context := runtimeState.KB.BuildContextPacket(kb.ContextRequest{
				Query:               query,
				GPU:                 request.GPU,
				Model:               request.Model,
				Task:                request.Task,
				Workload:            request.Workload,
				Operators:           request.Operators,
				Precision:           request.Precision,
				Bottleneck:          request.Bottleneck,
				Runtime:             runtimeName,
				Goals:               request.Goals,
				IncludeExperimental: includeExperimental,
				Limit:               limit,
			})
			request.IncludeExperimental = includeExperimental

			store, err := optimize.NewSessionStore()
			if err != nil {
				return err
			}

			session := store.NewSession(optimize.SessionCreateRequest{
				Name:          name,
				ProjectRoot:   projectRoot,
				WorkspaceRoot: workspaceRoot,
				Target:        targetName,
				Runtime:       runtimeName,
				Query:         query,
				Request:       request,
				Context:       context,
				Notes:         context.Notes,
			})
			session.Status = "ready"

			path, err := store.Save(session)
			if err != nil {
				return err
			}

			cmd.Printf("Created optimization session: %s\n", session.ID)
			cmd.Printf("Metadata: %s\n", path)
			cmd.Printf("Workspace root: %s\n", session.WorkspaceRoot)
			cmd.Printf("Memory: %s\n", optimize.SessionMemoryIndexPath(session))
			if session.Context.GPU != nil {
				cmd.Printf("GPU: %s\n", session.Context.GPU.Name)
			}
			cmd.Printf("Workload: %s\n", valueOrFallback(session.Request.Workload, "decode"))
			cmd.Printf("Runtime: %s\n", valueOrFallback(session.Runtime, "unspecified"))
			if len(session.Context.Skills) > 0 {
				cmd.Println("Top skills")
				for _, match := range session.Context.Skills {
					cmd.Printf("- %s (score %d)\n", match.Skill.Title, match.Score)
				}
			}
			if len(session.Context.Strategies) > 0 {
				cmd.Println("Top strategies")
				for _, match := range session.Context.Strategies {
					cmd.Printf("- %s (score %d)\n", match.Strategy.Title, match.Score)
				}
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&name, "name", "", "session name")
	cmd.Flags().StringVar(&query, "query", "", "free-form optimization request")
	cmd.Flags().StringVar(&runtimeName, "runtime", "", "runtime like vllm, tensorrt-llm, transformers, or sglang")
	cmd.Flags().StringVar(&workspaceRoot, "workspace-root", "", "override the session workspace root; defaults to ./.fusion/optimize/<session-id>")
	cmd.Flags().StringVar(&targetName, "target", "", "configured target name; uses the target GPU when --gpu is omitted")
	cmd.Flags().StringVar(&request.GPU, "gpu", "", "target GPU id or name")
	cmd.Flags().StringVar(&request.Model, "model", "", "model name or family")
	cmd.Flags().StringVar(&request.Task, "task", "", "task family like text-generation, image-generation, image-editing, video-generation, or audio-generation")
	cmd.Flags().StringVar(&request.Workload, "workload", "decode", "workload shape: decode, prefill, serving, training-prep")
	cmd.Flags().StringSliceVar(&request.Operators, "operator", nil, "operator families to optimize; repeat or comma-separate")
	cmd.Flags().StringVar(&request.Precision, "precision", "bf16", "target precision, for example bf16, fp16, fp8, int4")
	cmd.Flags().StringVar(&request.Bottleneck, "bottleneck", "", "override the inferred bottleneck: memory, compute, latency, mixed")
	cmd.Flags().StringSliceVar(&request.Goals, "goal", nil, "optimization goals such as throughput, latency, memory, cost")
	cmd.Flags().IntVar(&request.BatchSize, "batch-size", 1, "representative batch size")
	cmd.Flags().IntVar(&request.ContextLength, "context-length", 0, "representative prompt or total context length")
	cmd.Flags().BoolVar(&includeExperimental, "experimental", false, "include experimental strategies, skills, and examples")
	cmd.Flags().IntVar(&limit, "limit", 4, "maximum number of strategies, skills, and examples to store in context")
	return cmd
}

func newOptimizeSessionListCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List saved optimization sessions",
		RunE: func(cmd *cobra.Command, args []string) error {
			store, err := optimize.NewSessionStore()
			if err != nil {
				return err
			}
			summaries, err := store.List()
			if err != nil {
				return err
			}
			if len(summaries) == 0 {
				cmd.Println("No optimization sessions found.")
				return nil
			}
			for _, summary := range summaries {
				cmd.Printf("%s\n", summary.ID)
				cmd.Printf("  name: %s\n", summary.Name)
				cmd.Printf("  project: %s\n", summary.ProjectRoot)
				cmd.Printf("  workspace: %s\n", summary.WorkspaceRoot)
				cmd.Printf("  gpu/workload: %s / %s\n", valueOrFallback(summary.GPU, "unspecified"), valueOrFallback(summary.Workload, "unspecified"))
				cmd.Printf("  runtime: %s\n", valueOrFallback(summary.Runtime, "unspecified"))
				cmd.Printf("  status: %s\n", valueOrFallback(summary.Status, "ready"))
				cmd.Printf("  candidates: %d\n", summary.CandidateCount)
			}
			return nil
		},
	}
}

func newOptimizeSessionShowCommand() *cobra.Command {
	var id string

	cmd := &cobra.Command{
		Use:   "show",
		Short: "Show one optimization session in detail",
		RunE: func(cmd *cobra.Command, args []string) error {
			session, _, err := loadOptimizationSession(id)
			if err != nil {
				return err
			}

			cmd.Printf("%s\n", session.Name)
			cmd.Printf("id: %s\n", session.ID)
			cmd.Printf("project: %s\n", session.ProjectRoot)
			cmd.Printf("workspace root: %s\n", session.WorkspaceRoot)
			cmd.Printf("memory: %s\n", optimize.SessionMemoryIndexPath(session))
			cmd.Printf("target: %s\n", valueOrFallback(session.Target, "unspecified"))
			cmd.Printf("runtime: %s\n", valueOrFallback(session.Runtime, "unspecified"))
			cmd.Printf("status: %s\n", valueOrFallback(session.Status, "ready"))
			cmd.Printf("gpu: %s\n", valueOrFallback(session.Request.GPU, "unspecified"))
			cmd.Printf("model: %s\n", valueOrFallback(session.Request.Model, "unspecified"))
			cmd.Printf("task: %s\n", valueOrFallback(session.Request.Task, "unspecified"))
			cmd.Printf("workload: %s\n", valueOrFallback(session.Request.Workload, "decode"))
			cmd.Printf("precision: %s\n", valueOrFallback(session.Request.Precision, "bf16"))
			cmd.Printf("operators: %s\n", joinOrFallback(session.Request.Operators, "general"))
			if len(session.Context.Skills) > 0 {
				cmd.Println("skills")
				for _, match := range session.Context.Skills {
					cmd.Printf("- %s (score %d)\n", match.Skill.Title, match.Score)
				}
			}
			if len(session.Candidates) > 0 {
				cmd.Println("candidates")
				for _, candidate := range session.Candidates {
					cmd.Printf("- %s [%s]\n", candidate.Name, candidate.Backend)
					cmd.Printf("  id: %s\n", candidate.ID)
					cmd.Printf("  workspace: %s\n", candidate.Workspace)
					if len(candidate.Stages) > 0 {
						stageNames := make([]string, 0, len(candidate.Stages))
						for stage := range candidate.Stages {
							stageNames = append(stageNames, stage)
						}
						sort.Strings(stageNames)
						for _, stage := range stageNames {
							record := candidate.Stages[stage]
							cmd.Printf("  %s: exit=%d artifact=%s\n", stage, record.ExitCode, valueOrFallback(record.ArtifactPath, "n/a"))
						}
					}
				}
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&id, "id", "", "optimization session id")
	cmd.MarkFlagRequired("id")
	return cmd
}

func newOptimizeSessionGateCommand() *cobra.Command {
	var id string

	cmd := &cobra.Command{
		Use:   "gate",
		Short: "Show whether the outer loop is exhausted and the inner kernel loop is ready to start",
		RunE: func(cmd *cobra.Command, args []string) error {
			session, _, err := loadOptimizationSession(id)
			if err != nil {
				return err
			}
			status := optimize.EvaluateOuterLoopStatus(session)
			cmd.Printf("session: %s\n", session.ID)
			cmd.Printf("outer_loop_exhausted: %t\n", status.Exhausted)
			cmd.Printf("ready_for_inner_loop: %t\n", status.ReadyForInnerLoop)
			cmd.Printf("current_best: %s\n", valueOrFallback(status.CurrentBestID, "unset"))
			cmd.Println("families")
			for _, family := range status.Families {
				cmd.Printf("- %s: %s\n", family.Family, family.Status)
				if family.Reason != "" {
					cmd.Printf("  reason: %s\n", family.Reason)
				}
				if len(family.CandidateIDs) > 0 {
					cmd.Printf("  candidates: %s\n", strings.Join(family.CandidateIDs, ", "))
				}
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&id, "id", "", "optimization session id")
	cmd.MarkFlagRequired("id")
	return cmd
}

func newOptimizeSessionDecisionCommand() *cobra.Command {
	var id string
	var phase string
	var family string
	var status string
	var reason string
	var candidateID string

	cmd := &cobra.Command{
		Use:   "decide",
		Short: "Record an explicit outer-loop or inner-loop decision for orchestration and gating",
		RunE: func(cmd *cobra.Command, args []string) error {
			session, store, err := loadOptimizationSession(id)
			if err != nil {
				return err
			}
			session.RecordLoopDecision(phase, family, status, candidateID, reason)
			if _, err := store.Save(session); err != nil {
				return err
			}
			cmd.Printf("Recorded %s decision for %s: %s\n", phase, family, status)
			return nil
		},
	}

	cmd.Flags().StringVar(&id, "id", "", "optimization session id")
	cmd.Flags().StringVar(&phase, "phase", "outer", "loop phase, for example outer or inner")
	cmd.Flags().StringVar(&family, "family", "", "decision family like baseline, model-family, runtime, quantization, compile, or attention-backend")
	cmd.Flags().StringVar(&status, "status", "", "decision status like tested, blocked, skipped, regressed, or winner")
	cmd.Flags().StringVar(&reason, "reason", "", "human-readable reason for the decision")
	cmd.Flags().StringVar(&candidateID, "candidate", "", "optional candidate id associated with the decision")
	cmd.MarkFlagRequired("id")
	cmd.MarkFlagRequired("family")
	cmd.MarkFlagRequired("status")
	return cmd
}

func loadOptimizationSession(id string) (*optimize.Session, *optimize.SessionStore, error) {
	store, err := optimize.NewSessionStore()
	if err != nil {
		return nil, nil, err
	}
	session, err := store.Load(id)
	if err != nil {
		return nil, nil, err
	}
	return session, store, nil
}

func attachSessionCandidate(session *optimize.Session, store *optimize.SessionStore, candidate optimize.Candidate) error {
	session.UpsertCandidate(candidate)
	_, err := store.Save(session)
	return err
}

func resolveSessionWorkspaceRoot(session *optimize.Session, backend, name, output string) string {
	output = strings.TrimSpace(output)
	if output != "" {
		return output
	}
	base := strings.TrimSpace(name)
	if base == "" {
		base = backend + "-candidate"
	}
	return filepath.Join(session.WorkspaceRoot, backend+"-"+sanitizeWorkspaceName(base))
}

func sanitizeWorkspaceName(value string) string {
	value = strings.TrimSpace(strings.ToLower(value))
	value = strings.NewReplacer(" ", "-", "/", "-", "_", "-", ".", "-").Replace(value)
	value = strings.Trim(value, "-")
	if value == "" {
		return "workspace"
	}
	return value
}
