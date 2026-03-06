package cli

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
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
	cmd.Flags().StringVar(&opts.SessionID, "session", "", "resume a previous chat session by id")
	cmd.Flags().StringVar(&opts.CWD, "cwd", "", "working directory for chat tools; defaults to the current directory")
	cmd.Flags().IntVar(&opts.MaxRounds, "max-rounds", 12, "maximum tool-calling rounds per user turn")
	cmd.Flags().StringVar(&opts.Prompt, "prompt", "", "single prompt to run non-interactively")
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

	store, err := agent.NewStore()
	if err != nil {
		return err
	}

	var session *agent.Session
	if strings.TrimSpace(opts.SessionID) != "" {
		session, err = store.Load(opts.SessionID)
		if err != nil {
			return err
		}
		if strings.TrimSpace(opts.Model) != "" {
			session.Model = opts.Model
		}
		if strings.TrimSpace(opts.CWD) != "" {
			session.CWD = cwd
		}
	} else {
		model, _, err := resolveChatAccess(runtimeState, opts.Model)
		if err != nil {
			return err
		}
		session = store.NewSession(model, cwd, buildSystemPrompt(cwd))
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

	client := agent.NewClient(token)
	registry := agent.NewRegistry(agent.DefaultTools(agent.ToolContext{
		CWD:    session.CWD,
		Config: runtimeState.Config,
		KB:     runtimeState.KB,
	}))
	engine := agent.NewEngine(client, registry, opts.MaxRounds)
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
	cmd.Printf("Provider: %s\n", modelslab.Name)
	cmd.Printf("Model: %s\n", session.Model)
	cmd.Printf("Working directory: %s\n", session.CWD)
	cmd.Println("Type /help for local commands or /exit to quit.")

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
		switch line {
		case "/exit", "exit", "quit":
			return saveChatSession(store, session)
		case "/help":
			printChatHelp(cmd)
			continue
		case "/tools":
			for _, tool := range registry.Definitions() {
				cmd.Printf("- %s: %s\n", tool.Name, tool.Description)
			}
			continue
		case "/session":
			cmd.Printf("Session: %s\n", session.ID)
			cmd.Printf("Provider: %s\n", modelslab.Name)
			cmd.Printf("Model: %s\n", session.Model)
			cmd.Printf("Working directory: %s\n", session.CWD)
			continue
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
- Build an applicability matrix for optimization candidates before spending time. Mark branches as applicable, blocked, or unsupported for the current GPU, runtime, and model.
- Read files before editing them and prefer minimal, working changes over speculative rewrites.
- Destructive shell deletes like rm -rf are blocked. Use bounded file tools like delete_path, write_file, replace_in_file, move_path, and copy_path instead.
- If Hugging Face access is configured, shell commands automatically receive HF_TOKEN and HUGGING_FACE_HUB_TOKEN for model download and upload flows.
- If GitHub access is configured, shell commands automatically receive GITHUB_TOKEN and GH_TOKEN. For private HTTPS git operations, prefer gh commands or git with an Authorization header using $GITHUB_TOKEN instead of embedding secrets into URLs.
- For optimization tasks, create or reuse an optimization session so the work stays attached to one persistent record.
- After local inspection, call build_context_packet and search_knowledge_base so you retrieve the most relevant strategies, skills, examples, and sources instead of relying on one giant prompt.
- Register each optimization path as a candidate with register_optimization_candidate. This includes baseline/runtime-only candidates as well as packaged turbo or distilled model variants, Triton, CuTe, CUDA, torch.compile, AWQ, FP8, synthesized FP8 conversions, NVFP4, or any other backend or quantization path you choose.
- Do not stop at the first small win. Exhaust the applicable low-hanging search ladder first: baseline, packaged model-family or checkpoint variants, runtime flags and attention implementation, dtype or quant or checkpoint variants, including synthesized FP8 conversion when no packaged FP8 artifact exists, torch.compile or CUDA graphs if supported, then custom kernels.
- Skip unsupported branches explicitly with a reason. Example: native FP8 is Hopper or Blackwell-first, synthesized FP8 still requires runtime and calibration support, and NVFP4 is Blackwell-only.
- Do not assume hardcoded backend helpers exist. Choose the backend yourself and use generic file tools plus run_command to write, edit, build, verify, and benchmark code.
- When a shell command belongs to a candidate stage, call run_command with session, candidate, and stage so Fusion saves the artifact and stage record.
- Use run_benchmark and run_profile with session and candidate when you want benchmark/profile stages attached to the candidate history.
- If compile, correctness, inference, or performance issues happen, inspect the outputs, patch the code, and retry. Do not stop at the first fixable error.
- Verify correctness before claiming success, and use benchmark/profile evidence before claiming a performance win. For FP8 or other converted quantization paths, persist calibration details, fallback modules, and quality drift evidence with the candidate.
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
	cmd.Println("/help    show local chat commands")
	cmd.Println("/tools   list available agent tools")
	cmd.Println("/session show the active chat session")
	cmd.Println("/exit    save and quit")
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
