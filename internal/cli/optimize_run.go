package cli

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ModelsLab/fusion/internal/agent"
	"github.com/ModelsLab/fusion/internal/kb"
	"github.com/ModelsLab/fusion/internal/modelslab"
	"github.com/ModelsLab/fusion/internal/optimize"
	"github.com/ModelsLab/fusion/internal/system"
	"github.com/spf13/cobra"
)

func newOptimizeRunCommand() *cobra.Command {
	var request optimize.Request
	var llmModel string
	var sessionID string
	var agentSessionID string
	var cwd string
	var name string
	var query string
	var runtimeName string
	var targetName string
	var maxRounds int
	var limit int

	cmd := &cobra.Command{
		Use:   "run [query]",
		Short: "Run the ModelsLab-backed autonomous optimization agent inside a persistent optimization session",
		RunE: func(cmd *cobra.Command, args []string) error {
			if strings.TrimSpace(query) == "" && len(args) > 0 {
				query = strings.Join(args, " ")
			}
			return runOptimizeAgent(cmd, optimizeRunOptions{
				LLMModel:       llmModel,
				SessionID:      sessionID,
				AgentSessionID: agentSessionID,
				CWD:            cwd,
				Name:           name,
				Query:          query,
				Runtime:        runtimeName,
				Target:         targetName,
				Request:        request,
				MaxRounds:      maxRounds,
				Limit:          limit,
			})
		},
	}

	cmd.Flags().StringVar(&llmModel, "llm-model", "", "ModelsLab model id for the orchestration agent; defaults to the configured model")
	cmd.Flags().StringVar(&sessionID, "session", "", "resume an existing optimization session id")
	cmd.Flags().StringVar(&agentSessionID, "agent-session", "", "resume an existing chat agent session id")
	cmd.Flags().StringVar(&cwd, "cwd", "", "project working directory; defaults to the current directory")
	cmd.Flags().StringVar(&name, "name", "", "session name when creating a new optimization session")
	cmd.Flags().StringVar(&query, "query", "", "optimization request; falls back to positional arguments or derived session text")
	cmd.Flags().StringVar(&runtimeName, "runtime", "", "runtime like vllm, tensorrt-llm, transformers, or sglang")
	cmd.Flags().StringVar(&targetName, "target", "", "configured target name; uses the target GPU when --gpu is omitted")
	cmd.Flags().IntVar(&maxRounds, "max-rounds", 20, "maximum tool-calling rounds for the optimization turn")
	cmd.Flags().IntVar(&limit, "limit", 5, "maximum number of strategies, skills, and examples to refresh in the session context")

	cmd.Flags().StringVar(&request.GPU, "gpu", "", "target GPU id or name")
	cmd.Flags().StringVar(&request.Model, "model", "", "model name or family to optimize")
	cmd.Flags().StringVar(&request.Workload, "workload", "", "workload shape: decode, prefill, serving, training-prep")
	cmd.Flags().StringSliceVar(&request.Operators, "operator", nil, "operator families to optimize; repeat or comma-separate")
	cmd.Flags().StringVar(&request.Precision, "precision", "", "target precision or quantization path")
	cmd.Flags().StringVar(&request.Bottleneck, "bottleneck", "", "override the inferred bottleneck: memory, compute, latency, mixed")
	cmd.Flags().StringSliceVar(&request.Goals, "goal", nil, "optimization goals such as throughput, latency, memory, cost")
	cmd.Flags().IntVar(&request.BatchSize, "batch-size", 0, "representative batch size")
	cmd.Flags().IntVar(&request.ContextLength, "context-length", 0, "representative prompt or total context length")
	cmd.Flags().BoolVar(&request.IncludeExperimental, "experimental", false, "include experimental strategies, skills, and examples")
	return cmd
}

type optimizeRunOptions struct {
	LLMModel       string
	SessionID      string
	AgentSessionID string
	CWD            string
	Name           string
	Query          string
	Runtime        string
	Target         string
	Request        optimize.Request
	MaxRounds      int
	Limit          int
}

func runOptimizeAgent(cmd *cobra.Command, opts optimizeRunOptions) error {
	runtimeState, err := loadRuntime()
	if err != nil {
		return err
	}

	projectRoot := strings.TrimSpace(opts.CWD)
	if projectRoot == "" {
		projectRoot, err = os.Getwd()
		if err != nil {
			return fmt.Errorf("resolve current working directory: %w", err)
		}
	}
	projectRoot, err = filepath.Abs(projectRoot)
	if err != nil {
		return fmt.Errorf("resolve absolute project path: %w", err)
	}

	modelID, token, err := resolveChatAccess(runtimeState, opts.LLMModel)
	if err != nil {
		return err
	}

	session, store, err := prepareOptimizeSession(runtimeState, projectRoot, opts)
	if err != nil {
		return err
	}

	agentStore, err := agent.NewStore()
	if err != nil {
		return err
	}

	chatSession, agentPath, err := prepareOptimizeAgentSession(agentStore, session, modelID, projectRoot, valueOrFallback(opts.AgentSessionID, session.AgentSessionID))
	if err != nil {
		return err
	}

	session.Status = "running"
	session.AgentSessionID = chatSession.ID
	if _, err := store.Save(session); err != nil {
		return err
	}

	registry := agent.NewRegistry(agent.DefaultTools(agent.ToolContext{
		CWD:    chatSession.CWD,
		Config: runtimeState.Config,
		KB:     runtimeState.KB,
	}))
	engine := agent.NewEngine(agent.NewClient(token), registry, opts.MaxRounds)
	engine.SetToolHooks(
		func(call agent.ToolCall) {
			cmd.Printf("tool> %s %s\n", call.Name, compactJSON(call.Arguments))
		},
		func(call agent.ToolCall, output string, execErr error) {
			status := "ok"
			if execErr != nil {
				status = "error"
			}
			cmd.Printf("tool< %s [%s] %s\n", call.Name, status, summarizeToolOutput(output))
		},
	)

	sessionPath := store.SessionPath(session.ID)
	cmd.Printf("Optimization session: %s\n", session.ID)
	cmd.Printf("Session file: %s\n", sessionPath)
	cmd.Printf("Agent session: %s\n", chatSession.ID)
	cmd.Printf("Agent session file: %s\n", agentPath)
	cmd.Printf("Provider: %s\n", modelslab.Name)
	cmd.Printf("LLM model: %s\n", chatSession.Model)
	cmd.Printf("Project root: %s\n", chatSession.CWD)

	reply, runErr := engine.RunTurn(context.Background(), chatSession, buildOptimizeRunPrompt(session))
	saveAgentErr := saveChatSession(agentStore, chatSession)

	if runErr != nil {
		session.Status = "failed"
		session.LastResult = runErr.Error()
	} else {
		session.Status = "completed"
		session.LastResult = strings.TrimSpace(reply)
	}

	saveSessionErr := error(nil)
	if _, err := store.Save(session); err != nil {
		saveSessionErr = err
	}

	if strings.TrimSpace(reply) != "" {
		cmd.Println(reply)
	}

	if runErr != nil {
		if saveAgentErr != nil {
			cmd.Printf("warning: failed to save agent session: %v\n", saveAgentErr)
		}
		if saveSessionErr != nil {
			cmd.Printf("warning: failed to save optimization session: %v\n", saveSessionErr)
		}
		return runErr
	}
	if saveAgentErr != nil {
		return saveAgentErr
	}
	return saveSessionErr
}

func prepareOptimizeSession(runtimeState *runtimeState, projectRoot string, opts optimizeRunOptions) (*optimize.Session, *optimize.SessionStore, error) {
	store, err := optimize.NewSessionStore()
	if err != nil {
		return nil, nil, err
	}

	var session *optimize.Session
	if strings.TrimSpace(opts.SessionID) != "" {
		session, err = store.Load(opts.SessionID)
		if err != nil {
			return nil, nil, err
		}
		if strings.TrimSpace(opts.CWD) != "" {
			session.ProjectRoot = projectRoot
		}
	} else {
		request := opts.Request
		if strings.TrimSpace(request.Workload) == "" {
			request.Workload = "decode"
		}
		if strings.TrimSpace(request.Precision) == "" {
			request.Precision = "bf16"
		}
		if request.BatchSize <= 0 {
			request.BatchSize = 1
		}
		if strings.TrimSpace(opts.Target) != "" {
			target, _, err := resolveTarget(runtimeState, opts.Target)
			if err != nil {
				return nil, nil, err
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
		request.IncludeExperimental = opts.Request.IncludeExperimental

		session = store.NewSession(optimize.SessionCreateRequest{
			Name:        opts.Name,
			ProjectRoot: projectRoot,
			Target:      opts.Target,
			Runtime:     opts.Runtime,
			Query:       strings.TrimSpace(opts.Query),
			Request:     request,
		})
	}

	applyOptimizeRunOverrides(session, opts)
	refreshSessionContext(runtimeState.KB, session, opts.Limit)

	if strings.TrimSpace(session.Query) == "" {
		session.Query = defaultOptimizationQuery(session)
	}
	if strings.TrimSpace(session.Runtime) == "" {
		session.Runtime = strings.TrimSpace(opts.Runtime)
	}
	if strings.TrimSpace(session.ProjectRoot) == "" {
		session.ProjectRoot = projectRoot
	}
	if strings.TrimSpace(session.WorkspaceRoot) == "" {
		session.WorkspaceRoot = optimize.DefaultWorkspaceRoot(session.ProjectRoot, session.ID)
	}

	if _, err := store.Save(session); err != nil {
		return nil, nil, err
	}
	return session, store, nil
}

func prepareOptimizeAgentSession(store *agent.Store, session *optimize.Session, llmModel, projectRoot, requestedAgentSessionID string) (*agent.Session, string, error) {
	agentSessionID := strings.TrimSpace(requestedAgentSessionID)
	var chatSession *agent.Session
	var err error
	if agentSessionID != "" {
		chatSession, err = store.Load(agentSessionID)
		if err != nil {
			return nil, "", err
		}
	} else {
		chatSession = store.NewSession(llmModel, projectRoot, buildSystemPrompt(projectRoot))
	}

	chatSession.Provider = modelslab.ProviderID
	chatSession.Model = llmModel
	chatSession.CWD = projectRoot
	chatSession.SystemPrompt = buildSystemPrompt(projectRoot)

	path, err := store.Save(chatSession)
	if err != nil {
		return nil, "", err
	}
	return chatSession, path, nil
}

func applyOptimizeRunOverrides(session *optimize.Session, opts optimizeRunOptions) {
	if session == nil {
		return
	}
	if strings.TrimSpace(opts.Name) != "" {
		session.Name = strings.TrimSpace(opts.Name)
	}
	if strings.TrimSpace(opts.Query) != "" {
		session.Query = strings.TrimSpace(opts.Query)
	}
	if strings.TrimSpace(opts.Target) != "" {
		session.Target = strings.TrimSpace(opts.Target)
	}
	if strings.TrimSpace(opts.Runtime) != "" {
		session.Runtime = strings.TrimSpace(opts.Runtime)
	}

	request := session.Request
	override := opts.Request
	if strings.TrimSpace(override.GPU) != "" {
		request.GPU = strings.TrimSpace(override.GPU)
	}
	if strings.TrimSpace(override.Model) != "" {
		request.Model = strings.TrimSpace(override.Model)
	}
	if strings.TrimSpace(override.Workload) != "" {
		request.Workload = strings.TrimSpace(override.Workload)
	}
	if len(override.Operators) > 0 {
		request.Operators = append([]string{}, override.Operators...)
	}
	if strings.TrimSpace(override.Precision) != "" {
		request.Precision = strings.TrimSpace(override.Precision)
	}
	if strings.TrimSpace(override.Bottleneck) != "" {
		request.Bottleneck = strings.TrimSpace(override.Bottleneck)
	}
	if len(override.Goals) > 0 {
		request.Goals = append([]string{}, override.Goals...)
	}
	if override.BatchSize > 0 {
		request.BatchSize = override.BatchSize
	}
	if override.ContextLength > 0 {
		request.ContextLength = override.ContextLength
	}
	if override.IncludeExperimental {
		request.IncludeExperimental = true
	}
	session.Request = request
}

func refreshSessionContext(store *kb.Store, session *optimize.Session, limit int) {
	if store == nil || session == nil {
		return
	}
	session.Context = store.BuildContextPacket(kb.ContextRequest{
		Query:               session.Query,
		GPU:                 session.Request.GPU,
		Model:               session.Request.Model,
		Workload:            session.Request.Workload,
		Operators:           session.Request.Operators,
		Precision:           session.Request.Precision,
		Bottleneck:          session.Request.Bottleneck,
		Runtime:             session.Runtime,
		Goals:               session.Request.Goals,
		IncludeExperimental: session.Request.IncludeExperimental,
		Limit:               limit,
	})
	session.Notes = append([]string{}, session.Context.Notes...)
}

func defaultOptimizationQuery(session *optimize.Session) string {
	parts := []string{"optimize this project"}
	if model := strings.TrimSpace(session.Request.Model); model != "" {
		parts = append(parts, "for "+model)
	}
	if gpu := strings.TrimSpace(session.Request.GPU); gpu != "" {
		parts = append(parts, "on "+gpu)
	}
	if workload := strings.TrimSpace(session.Request.Workload); workload != "" {
		parts = append(parts, "for "+workload)
	}
	if runtimeName := strings.TrimSpace(session.Runtime); runtimeName != "" {
		parts = append(parts, "in "+runtimeName)
	}
	if len(session.Request.Goals) > 0 {
		parts = append(parts, "with goals: "+strings.Join(session.Request.Goals, ", "))
	}
	return strings.Join(parts, " ")
}
