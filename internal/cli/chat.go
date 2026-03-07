package cli

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ModelsLab/fusion/internal/agent"
	"github.com/ModelsLab/fusion/internal/modelslab"
	"github.com/spf13/cobra"
)

type chatOptions struct {
	Model     string
	SessionID string
	CWD       string
	MaxRounds int
	Prompt    string
	New       bool
}

func newChatCommand() *cobra.Command {
	var opts chatOptions

	cmd := &cobra.Command{
		Use:   "chat [prompt]",
		Short: "Start Fusion's interactive optimization agent through ModelsLab",
		RunE: func(cmd *cobra.Command, args []string) error {
			if opts.Prompt == "" && len(args) > 0 {
				opts.Prompt = strings.Join(args, " ")
			}
			return runChatSession(cmd, opts)
		},
	}

	cmd.Flags().StringVar(&opts.Model, "model", "", "model id; defaults to the configured ModelsLab model")
	cmd.Flags().StringVar(&opts.SessionID, "session", "", "resume a previous chat session by id or use latest")
	cmd.Flags().StringVar(&opts.CWD, "cwd", "", "working directory for chat tools; defaults to the current directory")
	cmd.Flags().IntVar(&opts.MaxRounds, "max-rounds", 12, "maximum tool-calling rounds per user turn")
	cmd.Flags().StringVar(&opts.Prompt, "prompt", "", "single prompt to run non-interactively")
	cmd.Flags().BoolVar(&opts.New, "new", false, "start a fresh chat session instead of auto-resuming the latest one for this directory")
	return cmd
}

func runDefaultChat(cmd *cobra.Command, args []string) error {
	return runChatSession(cmd, chatOptions{
		Prompt: strings.Join(args, " "),
	})
}

func runChatSession(cmd *cobra.Command, opts chatOptions) error {
	runtimeState, err := loadRuntime()
	if err != nil {
		return err
	}

	cwd := strings.TrimSpace(opts.CWD)
	if cwd == "" {
		cwd, err = os.Getwd()
		if err != nil {
			return fmt.Errorf("resolve current working directory: %w", err)
		}
	}
	cwd, err = filepath.Abs(cwd)
	if err != nil {
		return fmt.Errorf("resolve absolute working directory: %w", err)
	}

	store, err := agent.NewStore()
	if err != nil {
		return err
	}

	session, autoResumed, err := resolveInitialChatSession(runtimeState, store, opts, cwd)
	if err != nil {
		return err
	}

	model, token, err := resolveChatAccess(runtimeState, valueOrFallback(session.Model, opts.Model))
	if err != nil {
		return err
	}
	session.Provider = modelslab.ProviderID
	session.Model = model
	if strings.TrimSpace(session.CWD) == "" {
		session.CWD = cwd
	}
	if strings.TrimSpace(session.SystemPrompt) == "" {
		session.SystemPrompt = buildSystemPrompt(session.CWD)
	}

	registry, engine := buildChatRuntime(runtimeState, token, session, opts.MaxRounds, cmd)

	sessionPath, err := store.Save(session)
	if err != nil {
		return err
	}

	if strings.TrimSpace(opts.Prompt) != "" {
		reply, err := engine.RunTurn(context.Background(), session, opts.Prompt)
		if saveErr := saveChatSession(store, session); saveErr != nil && err == nil {
			err = saveErr
		}
		if strings.TrimSpace(reply) != "" {
			cmd.Println(reply)
		}
		return err
	}

	cmd.Printf("Fusion chat session: %s\n", session.ID)
	cmd.Printf("Session file: %s\n", sessionPath)
	if autoResumed {
		cmd.Println("Mode: resumed latest session for this working directory")
	} else {
		cmd.Println("Mode: new session")
	}
	cmd.Printf("Provider: %s\n", modelslab.Name)
	cmd.Printf("Model: %s\n", session.Model)
	cmd.Printf("Working directory: %s\n", session.CWD)
	cmd.Println("Type /help for local commands, /new for a fresh session, or /exit to quit.")

	reader := bufio.NewReader(cmd.InOrStdin())
	for {
		cmd.Print("fusion> ")
		line, readErr := reader.ReadString('\n')
		if readErr != nil && strings.TrimSpace(line) == "" {
			return nil
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if line == "exit" || line == "quit" {
			return saveChatSession(store, session)
		}

		if strings.HasPrefix(line, "/") {
			handled, nextSession, nextRegistry, nextEngine, commandErr := handleLocalChatCommand(cmd, store, runtimeState, session, registry, engine, opts.MaxRounds, line)
			if commandErr != nil {
				cmd.Printf("error: %v\n", commandErr)
				continue
			}
			if handled {
				if nextSession == nil {
					return nil
				}
				session = nextSession
				registry = nextRegistry
				engine = nextEngine
				continue
			}
		}

		reply, turnErr := engine.RunTurn(context.Background(), session, line)
		saveErr := saveChatSession(store, session)
		if turnErr != nil {
			cmd.Printf("error: %v\n", turnErr)
			if saveErr != nil {
				cmd.Printf("warning: failed to save session: %v\n", saveErr)
			}
			continue
		}
		if saveErr != nil {
			cmd.Printf("warning: failed to save session: %v\n", saveErr)
		}
		if strings.TrimSpace(reply) != "" {
			cmd.Println(reply)
		}
	}
}

func resolveInitialChatSession(runtimeState *runtimeState, store *agent.Store, opts chatOptions, cwd string) (*agent.Session, bool, error) {
	var session *agent.Session
	var err error

	switch {
	case strings.TrimSpace(opts.SessionID) != "":
		if strings.EqualFold(strings.TrimSpace(opts.SessionID), "latest") {
			session, err = store.FindLatestByCWD(cwd)
			if err != nil {
				return nil, false, err
			}
			if session == nil {
				return nil, false, fmt.Errorf("no saved Fusion chat session exists for %s", cwd)
			}
		} else {
			session, err = store.Load(opts.SessionID)
			if err != nil {
				return nil, false, err
			}
		}
	case !opts.New:
		session, err = store.FindLatestByCWD(cwd)
		if err != nil {
			return nil, false, err
		}
	}

	autoResumed := session != nil && strings.TrimSpace(opts.SessionID) == "" && !opts.New
	if session == nil {
		model, _, err := resolveChatAccess(runtimeState, opts.Model)
		if err != nil {
			return nil, false, err
		}
		session = store.NewSession(model, cwd, buildSystemPrompt(cwd))
	}

	if strings.TrimSpace(opts.Model) != "" {
		session.Model = opts.Model
	}
	if strings.TrimSpace(opts.CWD) != "" || strings.TrimSpace(session.CWD) == "" {
		session.CWD = cwd
		session.SystemPrompt = buildSystemPrompt(cwd)
	}

	return session, autoResumed, nil
}

func buildChatRuntime(runtimeState *runtimeState, token string, session *agent.Session, maxRounds int, cmd *cobra.Command) (*agent.ToolRegistry, *agent.Engine) {
	registry := agent.NewRegistry(agent.DefaultTools(agent.ToolContext{
		CWD:    session.CWD,
		Config: runtimeState.Config,
		KB:     runtimeState.KB,
	}))
	engine := agent.NewEngine(agent.NewClient(token), registry, maxRounds)
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
	return registry, engine
}

func handleLocalChatCommand(cmd *cobra.Command, store *agent.Store, runtimeState *runtimeState, session *agent.Session, registry *agent.ToolRegistry, engine *agent.Engine, maxRounds int, line string) (bool, *agent.Session, *agent.ToolRegistry, *agent.Engine, error) {
	fields := strings.Fields(line)
	if len(fields) == 0 {
		return true, session, registry, engine, nil
	}

	switch fields[0] {
	case "/exit":
		if err := saveChatSession(store, session); err != nil {
			return true, session, registry, engine, err
		}
		return true, nil, nil, nil, nil
	case "/help":
		printChatHelp(cmd)
		return true, session, registry, engine, nil
	case "/tools":
		for _, tool := range registry.Definitions() {
			cmd.Printf("- %s: %s\n", tool.Name, tool.Description)
		}
		return true, session, registry, engine, nil
	case "/session":
		cmd.Printf("Session: %s\n", session.ID)
		cmd.Printf("Provider: %s\n", modelslab.Name)
		cmd.Printf("Model: %s\n", session.Model)
		cmd.Printf("Working directory: %s\n", session.CWD)
		cmd.Printf("Messages: %d\n", len(session.Messages))
		return true, session, registry, engine, nil
	case "/save":
		path, err := store.Save(session)
		if err != nil {
			return true, session, registry, engine, err
		}
		cmd.Printf("Saved session: %s\n", path)
		return true, session, registry, engine, nil
	case "/history":
		limit := 12
		if len(fields) > 1 {
			parsed, err := strconv.Atoi(fields[1])
			if err != nil {
				return true, session, registry, engine, fmt.Errorf("invalid history count %q", fields[1])
			}
			if parsed > 0 {
				limit = parsed
			}
		}
		printSessionHistory(cmd, session, limit)
		return true, session, registry, engine, nil
	case "/sessions":
		limit := 10
		if len(fields) > 1 {
			parsed, err := strconv.Atoi(fields[1])
			if err != nil {
				return true, session, registry, engine, fmt.Errorf("invalid session count %q", fields[1])
			}
			if parsed > 0 {
				limit = parsed
			}
		}
		sessions, err := store.List()
		if err != nil {
			return true, session, registry, engine, err
		}
		if len(sessions) == 0 {
			cmd.Println("No saved sessions.")
			return true, session, registry, engine, nil
		}
		for i, item := range sessions {
			if i >= limit {
				break
			}
			marker := " "
			if item.ID == session.ID {
				marker = "*"
			}
			cmd.Printf("%s %s  %s  %s\n", marker, item.ID, item.Model, item.CWD)
		}
		return true, session, registry, engine, nil
	case "/resume":
		if len(fields) < 2 {
			return true, session, registry, engine, fmt.Errorf("usage: /resume <session-id|latest>")
		}
		var next *agent.Session
		var err error
		if strings.EqualFold(fields[1], "latest") {
			next, err = store.FindLatestByCWD(session.CWD)
		} else {
			next, err = store.Load(fields[1])
		}
		if err != nil {
			return true, session, registry, engine, err
		}
		if next == nil {
			return true, session, registry, engine, fmt.Errorf("no matching session found")
		}
		if strings.TrimSpace(next.SystemPrompt) == "" {
			next.SystemPrompt = buildSystemPrompt(next.CWD)
		}
		model, token, err := resolveChatAccess(runtimeState, next.Model)
		if err != nil {
			return true, session, registry, engine, err
		}
		next.Provider = modelslab.ProviderID
		next.Model = model
		nextRegistry, nextEngine := buildChatRuntime(runtimeState, token, next, maxRounds, cmd)
		if _, err := store.Save(next); err != nil {
			return true, session, registry, engine, err
		}
		cmd.Printf("Resumed session: %s\n", next.ID)
		cmd.Printf("Working directory: %s\n", next.CWD)
		return true, next, nextRegistry, nextEngine, nil
	case "/new":
		model, token, err := resolveChatAccess(runtimeState, session.Model)
		if err != nil {
			return true, session, registry, engine, err
		}
		next := store.NewSession(model, session.CWD, buildSystemPrompt(session.CWD))
		next.Provider = modelslab.ProviderID
		if _, err := store.Save(next); err != nil {
			return true, session, registry, engine, err
		}
		nextRegistry, nextEngine := buildChatRuntime(runtimeState, token, next, maxRounds, cmd)
		cmd.Printf("Created new session: %s\n", next.ID)
		return true, next, nextRegistry, nextEngine, nil
	case "/model":
		if len(fields) == 1 {
			cmd.Printf("Model: %s\n", session.Model)
			return true, session, registry, engine, nil
		}
		model, token, err := resolveChatAccess(runtimeState, fields[1])
		if err != nil {
			return true, session, registry, engine, err
		}
		session.Model = model
		if _, err := store.Save(session); err != nil {
			return true, session, registry, engine, err
		}
		nextRegistry, nextEngine := buildChatRuntime(runtimeState, token, session, maxRounds, cmd)
		cmd.Printf("Switched model: %s\n", session.Model)
		return true, session, nextRegistry, nextEngine, nil
	case "/cd":
		if len(fields) == 1 {
			cmd.Printf("Working directory: %s\n", session.CWD)
			return true, session, registry, engine, nil
		}
		nextPath, err := filepath.Abs(strings.Join(fields[1:], " "))
		if err != nil {
			return true, session, registry, engine, fmt.Errorf("resolve path: %w", err)
		}
		info, err := os.Stat(nextPath)
		if err != nil {
			return true, session, registry, engine, fmt.Errorf("access path: %w", err)
		}
		if !info.IsDir() {
			return true, session, registry, engine, fmt.Errorf("%s is not a directory", nextPath)
		}
		session.CWD = nextPath
		session.SystemPrompt = buildSystemPrompt(session.CWD)
		model, token, err := resolveChatAccess(runtimeState, session.Model)
		if err != nil {
			return true, session, registry, engine, err
		}
		session.Model = model
		if _, err := store.Save(session); err != nil {
			return true, session, registry, engine, err
		}
		nextRegistry, nextEngine := buildChatRuntime(runtimeState, token, session, maxRounds, cmd)
		cmd.Printf("Working directory changed to: %s\n", session.CWD)
		return true, session, nextRegistry, nextEngine, nil
	default:
		return false, session, registry, engine, nil
	}
}

func printSessionHistory(cmd *cobra.Command, session *agent.Session, limit int) {
	if len(session.Messages) == 0 {
		cmd.Println("No messages in this session yet.")
		return
	}
	if limit <= 0 {
		limit = 12
	}
	start := 0
	if len(session.Messages) > limit {
		start = len(session.Messages) - limit
	}
	for _, message := range session.Messages[start:] {
		switch message.Role {
		case "tool":
			cmd.Printf("[tool:%s] %s\n", message.ToolName, summarizeToolOutput(message.Content))
		default:
			cmd.Printf("[%s] %s\n", message.Role, summarizeToolOutput(message.Content))
		}
	}
}

func resolveChatAccess(runtimeState *runtimeState, requestedModel string) (string, string, error) {
	cfg, err := runtimeState.Config.Load()
	if err != nil {
		return "", "", err
	}

	token := strings.TrimSpace(cfg.ModelsLab.Token)
	if token == "" {
		token = strings.TrimSpace(os.Getenv(modelslab.TokenEnvVar))
	}
	if token == "" {
		return "", "", fmt.Errorf("no ModelsLab API key is configured; run `fusion login` or `fusion auth set --token ...` first")
	}

	model := strings.TrimSpace(requestedModel)
	if model == "" {
		model = strings.TrimSpace(cfg.ModelsLab.Model)
	}
	if model == "" {
		model = strings.TrimSpace(os.Getenv(modelslab.ModelEnvVar))
	}
	if model == "" {
		model = modelslab.DefaultModelID
	}

	return model, token, nil
}

func buildSystemPrompt(cwd string) string {
	return strings.TrimSpace(`You are Fusion, an optimization agent running inside a CLI.

Fusion is the tool and memory layer. You do the engineering work.

Core operating rules:
- Use tools instead of guessing about files, commands, environment state, or benchmark results.
- Inspect the local project first with list_files, search_files, and read_file before proposing an optimization plan.
- Understand the model/runtime layout, inference entrypoints, tests, build scripts, benchmark harnesses, and existing custom kernels before editing code.
- Treat the project as task-specific. Do not assume every workload is text generation or that tokens/sec is the right metric.
- Build an applicability matrix for optimization candidates before spending time. Mark branches as applicable, blocked, or unsupported for the current GPU, runtime, and model.
- Read files before editing them and prefer minimal, working changes over speculative rewrites.
- Destructive shell deletes like rm -rf are blocked. Use bounded file tools like delete_path, write_file, replace_in_file, move_path, and copy_path instead.
- If Hugging Face access is configured, shell commands automatically receive HF_TOKEN and HUGGING_FACE_HUB_TOKEN for model download and upload flows.
- If GitHub access is configured, shell commands automatically receive GITHUB_TOKEN and GH_TOKEN. For private HTTPS git operations, prefer gh commands or git with an Authorization header using $GITHUB_TOKEN instead of embedding secrets into URLs.
- For optimization tasks, create or reuse an optimization session so the work stays attached to one persistent record.
- After local inspection, establish a baseline candidate first. Benchmark it, profile the target phase you actually care about, analyze the profile, and then refresh context with build_context_packet and search_knowledge_base so you retrieve strategies based on measured bottlenecks instead of one giant prompt.
- Use detect_runtime_environment when the codebase could be transformers, diffusers, vllm, sglang, or another Python runtime.
- Register each optimization path as a candidate with register_optimization_candidate. This includes baseline/runtime-only candidates as well as packaged turbo or distilled model variants, Triton, CuTe, CUDA, torch.compile, AWQ, FP8, synthesized FP8 conversions, NVFP4, or any other backend or quantization path you choose.
- When the codebase already has a working path, create a conservative verified seed before aggressive rewrites. Prefer a correctness-preserving fast path first, then optimize bounded hot regions.
- Keep load, download, compilation, and warmup overhead separate from the steady-state phase you are trying to optimize. Do not let one-time costs hide the real hot path.
- Do not stop at the first small win. Exhaust the applicable low-hanging search ladder first: baseline, profile and diagnose, packaged model-family or checkpoint variants, runtime flags and attention implementation, dtype or quant or checkpoint variants, including synthesized FP8 conversion when no packaged FP8 artifact exists, torch.compile or CUDA graphs if supported, then custom kernels.
- Skip unsupported branches explicitly with a reason. Example: native FP8 is Hopper or Blackwell-first, synthesized FP8 still requires runtime and calibration support, and NVFP4 is Blackwell-only.
- Do not assume hardcoded backend helpers exist. Choose the backend yourself and use generic file tools plus run_command to write, edit, build, verify, and benchmark code.
- When a shell command belongs to a candidate stage, call run_command with session, candidate, and stage so Fusion saves the artifact and stage record.
- Use run_benchmark and run_profile with session and candidate when you want benchmark/profile stages attached to the candidate history.
- Use analyze_profile before escalating to deeper kernel work. Let measured bottlenecks determine whether the next branch should target residency, transfer reduction, quantization, attention/runtime changes, compile, or custom kernels.
- For text, image, video, and audio workloads, create a task-aware harness with create_harness_manifest and evaluate it with assess_harness instead of forcing every model into one benchmark shape.
- Use infer_hotspots when kernel names need to be mapped back to stages like attention, transformer, unet, vae, scheduler, or upscaler.
- Run inner-loop search in phases: explore early with multiple distinct search lanes or architectural families, then exploit later around the strongest survivors. Avoid spending the whole budget on one lineage too early.
- Use rank_search_candidates with metadata like backend, search_lane, signature, and hypothesis so Fusion can preserve diverse survivors instead of collapsing to near-duplicate candidates.
- Treat failed compile, verify, and runtime candidates as reusable negative examples. Record them with record_reflexion and session memory so later rounds avoid repeating the same class of mistake.
- If compile, correctness, inference, or performance issues happen, inspect the outputs, patch the code, and retry. Do not stop at the first fixable error.
- Verify correctness before claiming success, and use benchmark/profile evidence before claiming a performance win. For FP8 or other converted quantization paths, persist calibration details, fallback modules, and quality drift evidence with the candidate.
- Write session memory with write_session_memory after wins, failures, blockers, and environment changes so later turns can resume from markdown evidence.
- Prefer normalized steady-state metrics over raw wall time when model families or output lengths differ. Keep compile, download, and warmup overhead separate from steady-state generation speed.
- Maintain a current best candidate. If a later candidate regresses or fails, fall back to the current best and keep going.
- Keep user-facing responses concise, concrete, and action-oriented.

Current working directory: ` + cwd)
}

func saveChatSession(store *agent.Store, session *agent.Session) error {
	_, err := store.Save(session)
	return err
}

func printChatHelp(cmd *cobra.Command) {
	cmd.Println("/help         show local chat commands")
	cmd.Println("/tools        list available agent tools")
	cmd.Println("/session      show the active chat session")
	cmd.Println("/history [n]  show recent message history")
	cmd.Println("/sessions [n] list recent sessions")
	cmd.Println("/resume <id>  resume a saved session or /resume latest")
	cmd.Println("/new          start a fresh session in the current working directory")
	cmd.Println("/model [id]   show or switch the active model")
	cmd.Println("/cd [path]    show or change the working directory")
	cmd.Println("/save         save the current session immediately")
	cmd.Println("/exit         save and quit")
}

func compactJSON(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "{}"
	}
	var decoded any
	if err := json.Unmarshal([]byte(value), &decoded); err != nil {
		return value
	}
	data, err := json.Marshal(decoded)
	if err != nil {
		return value
	}
	return string(data)
}

func summarizeToolOutput(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "(empty)"
	}
	if len(value) > 220 {
		return value[:220] + "..."
	}
	return value
}
