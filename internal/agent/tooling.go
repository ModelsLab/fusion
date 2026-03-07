package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/ModelsLab/fusion/internal/artifacts"
	"github.com/ModelsLab/fusion/internal/config"
	"github.com/ModelsLab/fusion/internal/cutedsl"
	"github.com/ModelsLab/fusion/internal/githubauth"
	"github.com/ModelsLab/fusion/internal/huggingface"
	"github.com/ModelsLab/fusion/internal/kb"
	"github.com/ModelsLab/fusion/internal/optimize"
	"github.com/ModelsLab/fusion/internal/runner"
	"github.com/ModelsLab/fusion/internal/system"
	"github.com/ModelsLab/fusion/internal/targets"
)

type Tool struct {
	Definition ToolDefinition
	Execute    func(ctx context.Context, arguments string) (string, error)
}

type ToolRegistry struct {
	tools map[string]Tool
}

type ToolContext struct {
	CWD    string
	Config *config.Manager
	KB     *kb.Store
}

func NewRegistry(tools []Tool) *ToolRegistry {
	registry := &ToolRegistry{
		tools: map[string]Tool{},
	}
	for _, tool := range tools {
		registry.tools[tool.Definition.Name] = tool
	}
	return registry
}

func (r *ToolRegistry) Definitions() []ToolDefinition {
	out := make([]ToolDefinition, 0, len(r.tools))
	for _, tool := range r.tools {
		out = append(out, tool.Definition)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Name < out[j].Name
	})
	return out
}

func (r *ToolRegistry) Execute(ctx context.Context, call ToolCall) (string, error) {
	tool, ok := r.tools[call.Name]
	if !ok {
		return "", fmt.Errorf("unknown tool %q", call.Name)
	}
	return tool.Execute(ctx, call.Arguments)
}

func DefaultTools(toolCtx ToolContext) []Tool {
	return []Tool{
		listFilesTool(toolCtx),
		searchFilesTool(toolCtx),
		readFileTool(toolCtx),
		writeFileTool(toolCtx),
		appendFileTool(toolCtx),
		replaceInFileTool(toolCtx),
		makeDirectoryTool(toolCtx),
		copyPathTool(toolCtx),
		movePathTool(toolCtx),
		statPathTool(toolCtx),
		deletePathTool(toolCtx),
		runCommandTool(toolCtx),
		searchKnowledgeBaseTool(toolCtx),
		buildContextPacketTool(toolCtx),
		createOptimizationSessionTool(toolCtx),
		showOptimizationSessionTool(),
		registerOptimizationCandidateTool(toolCtx),
		recordOptimizationStageTool(),
		showOuterLoopStatusTool(),
		recordLoopDecisionTool(),
		analyzeProfileTool(toolCtx),
		assessBenchmarkRunsTool(),
		rankSearchCandidatesTool(),
		saveRoundArtifactTool(toolCtx),
		recordReflexionTool(toolCtx),
		detectRuntimeEnvironmentTool(toolCtx),
		applyRuntimePatchTool(toolCtx),
		revertRuntimePatchTool(),
		createHarnessManifestTool(toolCtx),
		assessHarnessTool(toolCtx),
		inferHotspotsTool(toolCtx),
		writeSessionMemoryTool(toolCtx),
		planOptimizationTool(toolCtx),
		detectEnvironmentTool(),
		listTargetsTool(toolCtx),
		benchmarkRunTool(toolCtx),
		profileRunTool(toolCtx),
	}
}

func listFilesTool(toolCtx ToolContext) Tool {
	type input struct {
		Path       string `json:"path"`
		Recursive  bool   `json:"recursive"`
		MaxDepth   int    `json:"max_depth"`
		MaxEntries int    `json:"max_entries"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "list_files",
			Description: "List files and directories inside the current project or a child path.",
			InputSchema: objectSchema(
				map[string]any{
					"path":        stringSchema("relative or absolute path to inspect"),
					"recursive":   boolSchema("whether to recurse into subdirectories"),
					"max_depth":   intSchema("maximum recursion depth when recursive is true"),
					"max_entries": intSchema("maximum number of entries to return"),
				},
				nil,
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}

			root, err := resolvePath(toolCtx.CWD, valueOrDefault(req.Path, "."))
			if err != nil {
				return "", err
			}
			maxEntries := defaultInt(req.MaxEntries, 200)
			maxDepth := defaultInt(req.MaxDepth, 4)

			type entry struct {
				Path  string `json:"path"`
				IsDir bool   `json:"is_dir"`
			}
			entries := []entry{}

			if !req.Recursive {
				items, err := os.ReadDir(root)
				if err != nil {
					return "", fmt.Errorf("read dir: %w", err)
				}
				for _, item := range items {
					if len(entries) >= maxEntries {
						break
					}
					entries = append(entries, entry{
						Path:  filepath.Join(root, item.Name()),
						IsDir: item.IsDir(),
					})
				}
			} else {
				err = filepath.WalkDir(root, func(path string, d os.DirEntry, walkErr error) error {
					if walkErr != nil {
						return walkErr
					}
					if path == root {
						return nil
					}
					rel, err := filepath.Rel(root, path)
					if err != nil {
						return err
					}
					depth := len(strings.Split(rel, string(filepath.Separator)))
					if depth > maxDepth {
						if d.IsDir() {
							return filepath.SkipDir
						}
						return nil
					}
					if len(entries) >= maxEntries {
						return fmt.Errorf("entry limit reached")
					}
					entries = append(entries, entry{
						Path:  path,
						IsDir: d.IsDir(),
					})
					return nil
				})
				if err != nil && err.Error() != "entry limit reached" {
					return "", fmt.Errorf("walk dir: %w", err)
				}
			}

			return marshalPretty(map[string]any{
				"path":        root,
				"entry_count": len(entries),
				"entries":     entries,
			})
		},
	}
}

func searchFilesTool(toolCtx ToolContext) Tool {
	type input struct {
		Pattern    string `json:"pattern"`
		Path       string `json:"path"`
		MaxResults int    `json:"max_results"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "search_files",
			Description: "Search file contents with ripgrep-style matching inside the project.",
			InputSchema: objectSchema(
				map[string]any{
					"pattern":     stringSchema("search pattern"),
					"path":        stringSchema("optional directory or file to search inside"),
					"max_results": intSchema("maximum number of matches"),
				},
				[]string{"pattern"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			root, err := resolvePath(toolCtx.CWD, valueOrDefault(req.Path, "."))
			if err != nil {
				return "", err
			}
			maxResults := defaultInt(req.MaxResults, 100)

			var output bytes.Buffer
			var cmd *exec.Cmd
			if rgPath, err := exec.LookPath("rg"); err == nil {
				cmd = exec.CommandContext(ctx, rgPath, "-n", "-S", "--hidden", "--max-count", strconv.Itoa(maxResults), req.Pattern, root)
			} else {
				cmd = exec.CommandContext(ctx, "grep", "-RIn", req.Pattern, root)
			}
			cmd.Stdout = &output
			cmd.Stderr = &output
			err = cmd.Run()
			result := strings.TrimSpace(output.String())
			if err != nil {
				if result == "" {
					return marshalPretty(map[string]any{
						"path":    root,
						"pattern": req.Pattern,
						"matches": []string{},
					})
				}
			}

			lines := strings.Split(result, "\n")
			if len(lines) > maxResults {
				lines = lines[:maxResults]
			}
			return marshalPretty(map[string]any{
				"path":    root,
				"pattern": req.Pattern,
				"matches": lines,
			})
		},
	}
}

func readFileTool(toolCtx ToolContext) Tool {
	type input struct {
		Path      string `json:"path"`
		StartLine int    `json:"start_line"`
		EndLine   int    `json:"end_line"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "read_file",
			Description: "Read a file, optionally restricted to a line range.",
			InputSchema: objectSchema(
				map[string]any{
					"path":       stringSchema("relative or absolute path to the file"),
					"start_line": intSchema("1-based starting line"),
					"end_line":   intSchema("1-based ending line"),
				},
				[]string{"path"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}

			path, err := resolvePath(toolCtx.CWD, req.Path)
			if err != nil {
				return "", err
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return "", fmt.Errorf("read file: %w", err)
			}

			lines := strings.Split(string(data), "\n")
			start := req.StartLine
			if start <= 0 {
				start = 1
			}
			end := req.EndLine
			if end <= 0 || end > len(lines) {
				end = len(lines)
			}
			if start > end {
				return "", fmt.Errorf("start_line must be less than or equal to end_line")
			}

			rendered := make([]string, 0, end-start+1)
			for i := start - 1; i < end; i++ {
				rendered = append(rendered, fmt.Sprintf("%d | %s", i+1, lines[i]))
			}

			return marshalPretty(map[string]any{
				"path":       path,
				"start_line": start,
				"end_line":   end,
				"content":    strings.Join(rendered, "\n"),
			})
		},
	}
}

func writeFileTool(toolCtx ToolContext) Tool {
	type input struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "write_file",
			Description: "Create or overwrite a file with the provided content.",
			InputSchema: objectSchema(
				map[string]any{
					"path":    stringSchema("relative or absolute path to the file"),
					"content": stringSchema("full file contents to write"),
				},
				[]string{"path", "content"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			path, err := resolvePath(toolCtx.CWD, req.Path)
			if err != nil {
				return "", err
			}
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				return "", fmt.Errorf("create parent dir: %w", err)
			}
			if err := os.WriteFile(path, []byte(req.Content), 0o644); err != nil {
				return "", fmt.Errorf("write file: %w", err)
			}
			return marshalPretty(map[string]any{
				"path":  path,
				"bytes": len(req.Content),
			})
		},
	}
}

func appendFileTool(toolCtx ToolContext) Tool {
	type input struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "append_file",
			Description: "Append content to a file, creating it if it does not exist.",
			InputSchema: objectSchema(
				map[string]any{
					"path":    stringSchema("relative or absolute path to the file"),
					"content": stringSchema("text to append"),
				},
				[]string{"path", "content"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			path, err := resolvePath(toolCtx.CWD, req.Path)
			if err != nil {
				return "", err
			}
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				return "", fmt.Errorf("create parent dir: %w", err)
			}
			file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
			if err != nil {
				return "", fmt.Errorf("open file: %w", err)
			}
			defer file.Close()
			if _, err := file.WriteString(req.Content); err != nil {
				return "", fmt.Errorf("append file: %w", err)
			}
			return marshalPretty(map[string]any{
				"path":  path,
				"bytes": len(req.Content),
			})
		},
	}
}

func replaceInFileTool(toolCtx ToolContext) Tool {
	type input struct {
		Path       string `json:"path"`
		OldText    string `json:"old_text"`
		NewText    string `json:"new_text"`
		ReplaceAll bool   `json:"replace_all"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "replace_in_file",
			Description: "Replace text in an existing file.",
			InputSchema: objectSchema(
				map[string]any{
					"path":        stringSchema("relative or absolute path to the file"),
					"old_text":    stringSchema("the exact text to replace"),
					"new_text":    stringSchema("replacement text"),
					"replace_all": boolSchema("replace every occurrence instead of the first"),
				},
				[]string{"path", "old_text", "new_text"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			path, err := resolvePath(toolCtx.CWD, req.Path)
			if err != nil {
				return "", err
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return "", fmt.Errorf("read file: %w", err)
			}
			content := string(data)
			count := strings.Count(content, req.OldText)
			if count == 0 {
				return "", fmt.Errorf("old_text was not found in %s", path)
			}

			replaced := content
			replacements := 1
			if req.ReplaceAll {
				replaced = strings.ReplaceAll(content, req.OldText, req.NewText)
				replacements = count
			} else {
				replaced = strings.Replace(content, req.OldText, req.NewText, 1)
			}
			if err := os.WriteFile(path, []byte(replaced), 0o644); err != nil {
				return "", fmt.Errorf("write file: %w", err)
			}
			return marshalPretty(map[string]any{
				"path":         path,
				"replacements": replacements,
			})
		},
	}
}

func makeDirectoryTool(toolCtx ToolContext) Tool {
	type input struct {
		Path string `json:"path"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "make_directory",
			Description: "Create a directory and any missing parent directories.",
			InputSchema: objectSchema(
				map[string]any{
					"path": stringSchema("directory path"),
				},
				[]string{"path"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			path, err := resolvePath(toolCtx.CWD, req.Path)
			if err != nil {
				return "", err
			}
			if err := os.MkdirAll(path, 0o755); err != nil {
				return "", fmt.Errorf("create directory: %w", err)
			}
			return marshalPretty(map[string]any{
				"path": path,
			})
		},
	}
}

func copyPathTool(toolCtx ToolContext) Tool {
	type input struct {
		Source    string `json:"source"`
		Dest      string `json:"dest"`
		Recursive bool   `json:"recursive"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "copy_path",
			Description: "Copy a file or directory inside the current project.",
			InputSchema: objectSchema(
				map[string]any{
					"source":    stringSchema("source file or directory"),
					"dest":      stringSchema("destination path"),
					"recursive": boolSchema("copy directories recursively"),
				},
				[]string{"source", "dest"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			source, err := resolvePath(toolCtx.CWD, req.Source)
			if err != nil {
				return "", err
			}
			dest, err := resolvePath(toolCtx.CWD, req.Dest)
			if err != nil {
				return "", err
			}
			info, err := os.Stat(source)
			if err != nil {
				return "", fmt.Errorf("stat source: %w", err)
			}
			if info.IsDir() {
				if !req.Recursive {
					return "", fmt.Errorf("source %s is a directory; set recursive=true to copy it", source)
				}
				if err := copyDirectory(source, dest); err != nil {
					return "", err
				}
			} else {
				if err := copyFile(source, dest); err != nil {
					return "", err
				}
			}
			return marshalPretty(map[string]any{
				"source": source,
				"dest":   dest,
			})
		},
	}
}

func movePathTool(toolCtx ToolContext) Tool {
	type input struct {
		Source string `json:"source"`
		Dest   string `json:"dest"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "move_path",
			Description: "Rename or move a file or directory.",
			InputSchema: objectSchema(
				map[string]any{
					"source": stringSchema("source path"),
					"dest":   stringSchema("destination path"),
				},
				[]string{"source", "dest"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			source, err := resolvePath(toolCtx.CWD, req.Source)
			if err != nil {
				return "", err
			}
			dest, err := resolvePath(toolCtx.CWD, req.Dest)
			if err != nil {
				return "", err
			}
			if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
				return "", fmt.Errorf("create destination parent dir: %w", err)
			}
			if err := os.Rename(source, dest); err != nil {
				return "", fmt.Errorf("move path: %w", err)
			}
			return marshalPretty(map[string]any{
				"source": source,
				"dest":   dest,
			})
		},
	}
}

func statPathTool(toolCtx ToolContext) Tool {
	type input struct {
		Path string `json:"path"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "stat_path",
			Description: "Return metadata about a file or directory.",
			InputSchema: objectSchema(
				map[string]any{
					"path": stringSchema("path to inspect"),
				},
				[]string{"path"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			path, err := resolvePath(toolCtx.CWD, req.Path)
			if err != nil {
				return "", err
			}
			info, err := os.Stat(path)
			if err != nil {
				return "", fmt.Errorf("stat path: %w", err)
			}
			return marshalPretty(map[string]any{
				"path":        path,
				"is_dir":      info.IsDir(),
				"size_bytes":  info.Size(),
				"mode":        info.Mode().String(),
				"modified_at": info.ModTime().UTC(),
				"basename":    filepath.Base(path),
				"parent":      filepath.Dir(path),
			})
		},
	}
}

func deletePathTool(toolCtx ToolContext) Tool {
	type input struct {
		Path string `json:"path"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "delete_path",
			Description: "Delete a file or directory tree.",
			InputSchema: objectSchema(
				map[string]any{
					"path": stringSchema("path to delete"),
				},
				[]string{"path"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			path, err := resolvePath(toolCtx.CWD, req.Path)
			if err != nil {
				return "", err
			}
			if err := os.RemoveAll(path); err != nil {
				return "", fmt.Errorf("delete path: %w", err)
			}
			return marshalPretty(map[string]any{
				"path": path,
			})
		},
	}
}

func runCommandTool(toolCtx ToolContext) Tool {
	type input struct {
		Command        string `json:"command"`
		Target         string `json:"target"`
		Workdir        string `json:"workdir"`
		Session        string `json:"session"`
		Candidate      string `json:"candidate"`
		Stage          string `json:"stage"`
		TimeoutSeconds int    `json:"timeout_seconds"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "run_command",
			Description: "Run a shell command locally or on a configured target. When session, candidate, and stage are set, persist the result as a candidate stage artifact.",
			InputSchema: objectSchema(
				map[string]any{
					"command":         stringSchema("shell command to execute"),
					"target":          stringSchema("optional configured target name"),
					"workdir":         stringSchema("optional working directory"),
					"session":         stringSchema("optional optimization session id"),
					"candidate":       stringSchema("optional candidate id when recording an optimization stage"),
					"stage":           stringSchema("optional stage name like inspect, baseline, build, verify, benchmark, profile, patch, or model-benchmark"),
					"timeout_seconds": intSchema("command timeout in seconds"),
				},
				[]string{"command"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}

			session, candidate, workdir, targetName, err := resolveCandidateExecutionContext(toolCtx, req.Session, req.Candidate, req.Workdir, req.Target)
			if err != nil {
				return "", err
			}

			result, executedCommand, err := executeTargetCommand(ctx, toolCtx, targetName, workdir, req.Command, time.Duration(defaultInt(req.TimeoutSeconds, 600))*time.Second)
			payload := map[string]any{
				"session":     strings.TrimSpace(req.Session),
				"candidate":   strings.TrimSpace(req.Candidate),
				"target_name": result.TargetName,
				"target_mode": result.TargetMode,
				"command":     executedCommand,
				"workdir":     workdir,
				"duration_ms": result.DurationMS,
				"exit_code":   result.ExitCode,
				"stdout":      truncate(result.Stdout, 24000),
				"stderr":      truncate(result.Stderr, 12000),
				"warnings":    result.Warnings,
			}

			stage := strings.TrimSpace(req.Stage)
			if stage != "" {
				if session == nil || candidate == nil {
					return "", fmt.Errorf("session, candidate, and stage must all resolve before recording a command artifact")
				}
				target, _, targetErr := resolveTarget(toolCtx, targetName)
				if targetErr != nil {
					return "", targetErr
				}
				artifactPath, metrics, artifactErr := saveKernelArtifact(candidate.Backend, stage, candidate.Name, candidate.Workspace, target, executedCommand, result)
				if artifactErr != nil {
					payload["artifact_error"] = artifactErr.Error()
				} else {
					payload["artifact_path"] = artifactPath
					if len(metrics) > 0 {
						payload["metrics"] = metrics
					}
					if stageErr := updateOptimizationSessionStage(session.ID, candidate.ID, candidate.Backend, candidate.Workspace, stage, artifactPath, executedCommand, result.ExitCode, metrics); stageErr != nil {
						payload["session_error"] = stageErr.Error()
					}
				}
			}

			if err != nil {
				payload["error"] = err.Error()
			}
			return marshalPretty(payload)
		},
	}
}

func searchKnowledgeBaseTool(toolCtx ToolContext) Tool {
	type input struct {
		Query string `json:"query"`
		Kind  string `json:"kind"`
		Limit int    `json:"limit"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "search_knowledge_base",
			Description: "Search Fusion's embedded GPU and optimization knowledge base across sources, GPUs, strategies, skills, and examples.",
			InputSchema: objectSchema(
				map[string]any{
					"query": stringSchema("search query"),
					"kind":  stringSchema("optional kind: all, source, gpu, strategy, skill, example"),
					"limit": intSchema("maximum number of hits"),
				},
				[]string{"query"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			hits := toolCtx.KB.Search(req.Query, req.Kind, defaultInt(req.Limit, 10))
			return marshalPretty(map[string]any{
				"query": req.Query,
				"kind":  valueOrDefault(req.Kind, "all"),
				"hits":  hits,
			})
		},
	}
}

func buildContextPacketTool(toolCtx ToolContext) Tool {
	type input struct {
		Query               string   `json:"query"`
		GPU                 string   `json:"gpu"`
		Model               string   `json:"model"`
		Task                string   `json:"task"`
		Workload            string   `json:"workload"`
		Operators           []string `json:"operators"`
		Precision           string   `json:"precision"`
		Bottleneck          string   `json:"bottleneck"`
		Runtime             string   `json:"runtime"`
		Goals               []string `json:"goals"`
		IncludeExperimental bool     `json:"include_experimental"`
		Limit               int      `json:"limit"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "build_context_packet",
			Description: "Build a ranked retrieval packet with strategies, skills, examples, and sources for a specific optimization request.",
			InputSchema: objectSchema(
				map[string]any{
					"query":                stringSchema("free-form request or retrieval hint"),
					"gpu":                  stringSchema("GPU id or name"),
					"model":                stringSchema("model name or family"),
					"task":                 stringSchema("task family like text-generation, image-generation, image-editing, video-generation, or audio-generation"),
					"workload":             stringSchema("decode, prefill, serving, or training-prep"),
					"operators":            stringArraySchema("operator families"),
					"precision":            stringSchema("precision or quantization path"),
					"bottleneck":           stringSchema("memory, compute, latency, or mixed"),
					"runtime":              stringSchema("runtime like vllm, tensorrt-llm, transformers, or sglang"),
					"goals":                stringArraySchema("optimization goals"),
					"include_experimental": boolSchema("include experimental strategies, skills, and examples"),
					"limit":                intSchema("maximum number of strategies, skills, and examples to return"),
				},
				nil,
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}

			packet := toolCtx.KB.BuildContextPacket(kb.ContextRequest{
				Query:               req.Query,
				GPU:                 req.GPU,
				Model:               req.Model,
				Task:                req.Task,
				Workload:            req.Workload,
				Operators:           req.Operators,
				Precision:           req.Precision,
				Bottleneck:          req.Bottleneck,
				Runtime:             req.Runtime,
				Goals:               req.Goals,
				IncludeExperimental: req.IncludeExperimental,
				Limit:               req.Limit,
			})
			return marshalPretty(packet)
		},
	}
}

func planOptimizationTool(toolCtx ToolContext) Tool {
	type input struct {
		Target              string   `json:"target"`
		GPU                 string   `json:"gpu"`
		Model               string   `json:"model"`
		Task                string   `json:"task"`
		Workload            string   `json:"workload"`
		Operators           []string `json:"operators"`
		Precision           string   `json:"precision"`
		Bottleneck          string   `json:"bottleneck"`
		Goals               []string `json:"goals"`
		BatchSize           int      `json:"batch_size"`
		ContextLength       int      `json:"context_length"`
		IncludeExperimental bool     `json:"include_experimental"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "plan_optimization",
			Description: "Build a GPU-aware optimization plan including model paths, kernel backends, and strategy recommendations.",
			InputSchema: objectSchema(
				map[string]any{
					"target":               stringSchema("optional target name"),
					"gpu":                  stringSchema("GPU id or name"),
					"model":                stringSchema("model name or family"),
					"task":                 stringSchema("task family like text-generation, image-generation, image-editing, video-generation, or audio-generation"),
					"workload":             stringSchema("decode, prefill, serving, or training-prep"),
					"operators":            stringArraySchema("operator families"),
					"precision":            stringSchema("precision or quantization path"),
					"bottleneck":           stringSchema("memory, compute, latency, or mixed"),
					"goals":                stringArraySchema("optimization goals"),
					"batch_size":           intSchema("representative batch size"),
					"context_length":       intSchema("representative context length"),
					"include_experimental": boolSchema("include experimental strategies"),
				},
				nil,
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}

			request := optimize.Request{
				GPU:                 req.GPU,
				Model:               req.Model,
				Task:                req.Task,
				Workload:            req.Workload,
				Operators:           req.Operators,
				Precision:           req.Precision,
				Bottleneck:          req.Bottleneck,
				Goals:               req.Goals,
				BatchSize:           req.BatchSize,
				ContextLength:       req.ContextLength,
				IncludeExperimental: req.IncludeExperimental,
			}

			if strings.TrimSpace(req.Target) != "" {
				target, _, err := resolveTarget(toolCtx, req.Target)
				if err != nil {
					return "", err
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

			planner := optimize.NewPlanner(toolCtx.KB)
			plan, err := planner.Build(request)
			if err != nil {
				return "", err
			}
			return marshalPretty(plan)
		},
	}
}

func detectEnvironmentTool() Tool {
	return Tool{
		Definition: ToolDefinition{
			Name:        "detect_environment",
			Description: "Inspect the host OS, detected GPUs, available tools, and execution capabilities.",
			InputSchema: objectSchema(nil, nil),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			env := system.DetectEnvironment()
			capabilities := system.AssessCapabilities(env)
			return marshalPretty(map[string]any{
				"environment":  env,
				"capabilities": capabilities,
			})
		},
	}
}

func listTargetsTool(toolCtx ToolContext) Tool {
	return Tool{
		Definition: ToolDefinition{
			Name:        "list_targets",
			Description: "List configured local, SSH, and sim targets.",
			InputSchema: objectSchema(nil, nil),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			cfg, err := toolCtx.Config.Load()
			if err != nil {
				return "", err
			}
			names := make([]string, 0, len(cfg.Targets))
			for name := range cfg.Targets {
				names = append(names, name)
			}
			sort.Strings(names)

			out := make([]config.TargetConfig, 0, len(names))
			for _, name := range names {
				out = append(out, cfg.Targets[name])
			}
			return marshalPretty(map[string]any{
				"default_target": cfg.DefaultTarget,
				"targets":        out,
			})
		},
	}
}

func benchmarkRunTool(toolCtx ToolContext) Tool {
	type input struct {
		Target         string `json:"target"`
		Name           string `json:"name"`
		Command        string `json:"command"`
		Workdir        string `json:"workdir"`
		Metrics        string `json:"metrics"`
		Session        string `json:"session"`
		Candidate      string `json:"candidate"`
		Stage          string `json:"stage"`
		TimeoutSeconds int    `json:"timeout_seconds"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "run_benchmark",
			Description: "Run a benchmark command and persist a benchmark artifact. When session and candidate are set, also record the benchmark stage on that candidate.",
			InputSchema: objectSchema(
				map[string]any{
					"target":          stringSchema("optional configured target name"),
					"name":            stringSchema("benchmark artifact name"),
					"command":         stringSchema("shell command to execute"),
					"workdir":         stringSchema("optional working directory"),
					"metrics":         stringSchema("optional inline metrics like tokens_per_sec=42 latency_ms=8"),
					"session":         stringSchema("optional optimization session id"),
					"candidate":       stringSchema("optional candidate id"),
					"stage":           stringSchema("optional stage name; defaults to benchmark when session and candidate are set"),
					"timeout_seconds": intSchema("command timeout in seconds"),
				},
				[]string{"command"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}

			session, candidate, workdir, targetName, err := resolveCandidateExecutionContext(toolCtx, req.Session, req.Candidate, req.Workdir, req.Target)
			if err != nil {
				return "", err
			}

			runResult, executedCommand, err := executeTargetCommand(ctx, toolCtx, targetName, workdir, req.Command, time.Duration(defaultInt(req.TimeoutSeconds, 1800))*time.Second)

			metrics := artifacts.ParseMetrics(runResult.Stdout)
			if strings.TrimSpace(req.Metrics) != "" {
				for key, value := range artifacts.ParseMetrics(req.Metrics) {
					if metrics == nil {
						metrics = map[string]float64{}
					}
					metrics[key] = value
				}
			}

			store, storeErr := artifacts.NewStore()
			if storeErr != nil {
				return "", storeErr
			}

			name := strings.TrimSpace(req.Name)
			if name == "" {
				name = runResult.TargetName + "-benchmark"
			}
			artifact := artifacts.BenchmarkResult{
				Name:          name,
				TargetName:    runResult.TargetName,
				TargetMode:    runResult.TargetMode,
				Command:       executedCommand,
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
			path, saveErr := store.SaveBenchmark(artifact, "")
			if saveErr != nil {
				return "", saveErr
			}
			payload := map[string]any{
				"session":       strings.TrimSpace(req.Session),
				"candidate":     strings.TrimSpace(req.Candidate),
				"artifact_path": path,
				"metrics":       metrics,
				"workdir":       workdir,
				"duration_ms":   artifact.DurationMS,
				"exit_code":     artifact.ExitCode,
				"warnings":      artifact.Warnings,
			}
			if session != nil && candidate != nil {
				stage := valueOrDefault(req.Stage, "benchmark")
				if stageErr := updateOptimizationSessionStage(session.ID, candidate.ID, candidate.Backend, candidate.Workspace, stage, path, executedCommand, artifact.ExitCode, metrics); stageErr != nil {
					payload["session_error"] = stageErr.Error()
				}
			}
			if err != nil {
				payload["error"] = err.Error()
			}
			return marshalPretty(payload)
		},
	}
}

func profileRunTool(toolCtx ToolContext) Tool {
	type input struct {
		Target         string `json:"target"`
		Name           string `json:"name"`
		ToolLabel      string `json:"tool"`
		Command        string `json:"command"`
		Workdir        string `json:"workdir"`
		Session        string `json:"session"`
		Candidate      string `json:"candidate"`
		Stage          string `json:"stage"`
		TimeoutSeconds int    `json:"timeout_seconds"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "run_profile",
			Description: "Run a profiling command and persist a raw profile artifact. When session and candidate are set, also record the profile stage on that candidate.",
			InputSchema: objectSchema(
				map[string]any{
					"target":          stringSchema("optional configured target name"),
					"name":            stringSchema("profile artifact name"),
					"tool":            stringSchema("label like ncu or nsys"),
					"command":         stringSchema("shell command to execute"),
					"workdir":         stringSchema("optional working directory"),
					"session":         stringSchema("optional optimization session id"),
					"candidate":       stringSchema("optional candidate id"),
					"stage":           stringSchema("optional stage name; defaults to profile when session and candidate are set"),
					"timeout_seconds": intSchema("command timeout in seconds"),
				},
				[]string{"command"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}

			session, candidate, workdir, targetName, err := resolveCandidateExecutionContext(toolCtx, req.Session, req.Candidate, req.Workdir, req.Target)
			if err != nil {
				return "", err
			}

			runResult, executedCommand, err := executeTargetCommand(ctx, toolCtx, targetName, workdir, req.Command, time.Duration(defaultInt(req.TimeoutSeconds, 1800))*time.Second)

			store, storeErr := artifacts.NewStore()
			if storeErr != nil {
				return "", storeErr
			}

			name := strings.TrimSpace(req.Name)
			if name == "" {
				name = runResult.TargetName + "-profile"
			}
			artifact := artifacts.ProfileResult{
				Name:          name,
				Tool:          valueOrDefault(req.ToolLabel, "custom"),
				TargetName:    runResult.TargetName,
				TargetMode:    runResult.TargetMode,
				Command:       executedCommand,
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
			path, saveErr := store.SaveProfile(artifact, "")
			if saveErr != nil {
				return "", saveErr
			}
			payload := map[string]any{
				"session":       strings.TrimSpace(req.Session),
				"candidate":     strings.TrimSpace(req.Candidate),
				"artifact_path": path,
				"workdir":       workdir,
				"duration_ms":   artifact.DurationMS,
				"exit_code":     artifact.ExitCode,
				"warnings":      artifact.Warnings,
			}
			if session != nil && candidate != nil {
				stage := valueOrDefault(req.Stage, "profile")
				if stageErr := updateOptimizationSessionStage(session.ID, candidate.ID, candidate.Backend, candidate.Workspace, stage, path, executedCommand, artifact.ExitCode, nil); stageErr != nil {
					payload["session_error"] = stageErr.Error()
				}
			}
			if err != nil {
				payload["error"] = err.Error()
			}
			return marshalPretty(payload)
		},
	}
}

func initCuteWorkspaceTool(toolCtx ToolContext) Tool {
	type input struct {
		Name      string `json:"name"`
		Output    string `json:"output"`
		Template  string `json:"template"`
		Operation string `json:"operation"`
		GPUArch   string `json:"gpu_arch"`
		Target    string `json:"target"`
		Force     bool   `json:"force"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "init_cute_workspace",
			Description: "Create a CuTe DSL workspace scaffold with kernel, build, verify, and benchmark scripts.",
			InputSchema: objectSchema(
				map[string]any{
					"name":      stringSchema("workspace name"),
					"output":    stringSchema("output directory"),
					"template":  stringSchema("CuTe template name"),
					"operation": stringSchema("logical kernel operation label"),
					"gpu_arch":  stringSchema("default GPU arch like sm90"),
					"target":    stringSchema("optional target label to record in metadata"),
					"force":     boolSchema("overwrite scaffold in a non-empty output directory"),
				},
				[]string{"name"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			output := req.Output
			if strings.TrimSpace(output) != "" {
				resolved, err := resolvePath(toolCtx.CWD, output)
				if err != nil {
					return "", err
				}
				output = resolved
			}
			workspace, err := cutedsl.Init(cutedsl.InitRequest{
				Name:      req.Name,
				OutputDir: output,
				Template:  req.Template,
				Operation: req.Operation,
				GPUArch:   req.GPUArch,
				Target:    req.Target,
				Force:     req.Force,
			})
			if err != nil {
				return "", err
			}
			if output == "" {
				output, _ = filepath.Abs(req.Name)
			}
			return marshalPretty(map[string]any{
				"workspace": workspace,
				"path":      output,
			})
		},
	}
}

func buildCuteWorkspaceTool(toolCtx ToolContext) Tool {
	type input struct {
		Workspace      string `json:"workspace"`
		Target         string `json:"target"`
		GPUArch        string `json:"gpu_arch"`
		Python         string `json:"python"`
		ExportDir      string `json:"export_dir"`
		OptLevel       int    `json:"opt_level"`
		TimeoutSeconds int    `json:"timeout_seconds"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "build_cute_workspace",
			Description: "Compile a CuTe DSL workspace on a local or remote target.",
			InputSchema: objectSchema(
				map[string]any{
					"workspace":       stringSchema("workspace path"),
					"target":          stringSchema("optional configured target name"),
					"gpu_arch":        stringSchema("override GPU arch"),
					"python":          stringSchema("python interpreter"),
					"export_dir":      stringSchema("optional AOT export directory"),
					"opt_level":       intSchema("CuTe optimization level"),
					"timeout_seconds": intSchema("command timeout in seconds"),
				},
				[]string{"workspace"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			return executeCuteTool(ctx, toolCtx, req.Workspace, req.Target, time.Duration(defaultInt(req.TimeoutSeconds, 1800))*time.Second, cutedsl.BuildCommand(".", cutedsl.BuildArgs{
				PythonBin: req.Python,
				GPUArch:   req.GPUArch,
				OptLevel:  req.OptLevel,
				ExportDir: req.ExportDir,
			}))
		},
	}
}

func verifyCuteWorkspaceTool(toolCtx ToolContext) Tool {
	type input struct {
		Workspace      string  `json:"workspace"`
		Target         string  `json:"target"`
		GPUArch        string  `json:"gpu_arch"`
		Python         string  `json:"python"`
		Size           int     `json:"size"`
		ATol           float64 `json:"atol"`
		RTol           float64 `json:"rtol"`
		TimeoutSeconds int     `json:"timeout_seconds"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "verify_cute_workspace",
			Description: "Run correctness verification for a CuTe DSL workspace on a local or remote target.",
			InputSchema: objectSchema(
				map[string]any{
					"workspace":       stringSchema("workspace path"),
					"target":          stringSchema("optional configured target name"),
					"gpu_arch":        stringSchema("override GPU arch"),
					"python":          stringSchema("python interpreter"),
					"size":            intSchema("number of elements"),
					"atol":            numberSchema("absolute tolerance"),
					"rtol":            numberSchema("relative tolerance"),
					"timeout_seconds": intSchema("command timeout in seconds"),
				},
				[]string{"workspace"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			return executeCuteTool(ctx, toolCtx, req.Workspace, req.Target, time.Duration(defaultInt(req.TimeoutSeconds, 1800))*time.Second, cutedsl.VerifyCommand(".", cutedsl.VerifyArgs{
				PythonBin: req.Python,
				GPUArch:   req.GPUArch,
				Size:      req.Size,
				ATol:      req.ATol,
				RTol:      req.RTol,
			}))
		},
	}
}

func executeCuteTool(ctx context.Context, toolCtx ToolContext, workspacePath, targetName string, timeout time.Duration, command string) (string, error) {
	workspacePath, err := resolvePath(toolCtx.CWD, workspacePath)
	if err != nil {
		return "", err
	}
	workspace, err := cutedsl.Load(workspacePath)
	if err != nil {
		return "", err
	}

	target, _, err := resolveTarget(toolCtx, targetName)
	if err != nil {
		return "", err
	}

	executedCommand := formatDisplayedToolCommand(workspacePath, command)
	runReq := runner.Request{
		Target:  target,
		Command: command,
		WorkDir: workspacePath,
		Timeout: timeout,
	}
	if needsRemoteWorkspace(target) {
		remoteWorkspace := remoteCuteWorkspace(target, workspacePath)
		copyTarget := target
		copyTarget.RemoteDir = ""
		if _, err := runner.Copy(runner.CopyRequest{
			Target:    copyTarget,
			Source:    workspacePath,
			Dest:      remoteWorkspace,
			Recursive: true,
			Timeout:   timeout,
		}); err != nil {
			return "", fmt.Errorf("copy CuTe workspace to target: %w", err)
		}
		runReq.WorkDir = remoteWorkspace
		executedCommand = formatDisplayedToolCommand(remoteWorkspace, command)
	}

	runReq.Target = target
	runResult, runErr := runner.Execute(runReq)

	payload := map[string]any{
		"workspace":   workspace.Name,
		"path":        workspacePath,
		"command":     executedCommand,
		"duration_ms": runResult.DurationMS,
		"exit_code":   runResult.ExitCode,
		"stdout":      truncate(runResult.Stdout, 24000),
		"stderr":      truncate(runResult.Stderr, 12000),
		"warnings":    runResult.Warnings,
	}
	if metrics := artifacts.ParseMetrics(runResult.Stdout); len(metrics) > 0 {
		payload["metrics"] = metrics
	}
	if runErr != nil {
		payload["error"] = runErr.Error()
	}
	return marshalPretty(payload)
}

func executeTargetCommand(ctx context.Context, toolCtx ToolContext, targetName, workdir, command string, timeout time.Duration) (runner.Result, string, error) {
	if err := validateSafeAgentCommand(command); err != nil {
		return runner.Result{}, "", err
	}

	target, _, err := resolveTarget(toolCtx, targetName)
	if err != nil {
		return runner.Result{}, "", err
	}

	mode := targets.Normalize(target.Mode)
	executedCommand := command
	runReq := runner.Request{
		Target:  target,
		Command: command,
		Timeout: timeout,
	}
	if env := shellEnvFromConfig(toolCtx.Config); len(env) > 0 {
		runReq.Env = env
	}
	if mode == targets.ModeLocal || (mode == targets.ModeSim && strings.TrimSpace(target.Host) == "") {
		baseDir := toolCtx.CWD
		if strings.TrimSpace(workdir) != "" {
			baseDir, err = resolvePath(toolCtx.CWD, workdir)
			if err != nil {
				return runner.Result{}, "", err
			}
		}
		runReq.WorkDir = baseDir
		executedCommand = formatDisplayedToolCommand(baseDir, command)
	} else if strings.TrimSpace(workdir) != "" {
		runReq.WorkDir = strings.TrimSpace(workdir)
		executedCommand = formatDisplayedToolCommand(runReq.WorkDir, command)
	}

	runResult, runErr := runner.Execute(runReq)
	return runResult, executedCommand, runErr
}

func formatDisplayedToolCommand(workdir, command string) string {
	workdir = strings.TrimSpace(workdir)
	if workdir == "" {
		return command
	}
	return "cd " + shellQuote(workdir) + " && " + command
}

func validateSafeAgentCommand(command string) error {
	command = strings.TrimSpace(command)
	if command == "" {
		return fmt.Errorf("command is required")
	}
	normalized := strings.ToLower(command)
	normalized = strings.NewReplacer("\n", " ", "\r", " ", "\t", " ").Replace(normalized)
	normalized = strings.Join(strings.Fields(normalized), " ")

	if containsDangerousRecursiveRemove(normalized) {
		return fmt.Errorf("blocked dangerous shell command: recursive force-delete commands like `rm -rf` are not allowed; use bounded file tools such as delete_path instead")
	}
	return nil
}

func containsDangerousRecursiveRemove(command string) bool {
	patterns := []string{
		`rm -rf`,
		`rm -fr`,
		`rm -r -f`,
		`rm -f -r`,
		`rm --recursive --force`,
		`rm --force --recursive`,
	}
	for _, pattern := range patterns {
		if strings.Contains(command, pattern) {
			return true
		}
	}
	return false
}

func shellEnvFromConfig(manager *config.Manager) map[string]string {
	if manager == nil {
		return nil
	}
	cfg, err := manager.Load()
	if err != nil {
		return nil
	}
	token := strings.TrimSpace(cfg.HuggingFace.Token)
	if token == "" {
		token = huggingface.TokenFromEnv()
	}
	env := map[string]string{}
	for key, value := range huggingface.ShellEnv(token) {
		env[key] = value
	}
	githubToken := strings.TrimSpace(cfg.GitHub.Token)
	if githubToken == "" {
		githubToken = githubauth.TokenFromEnv()
	}
	for key, value := range githubauth.ShellEnv(githubToken) {
		env[key] = value
	}
	if len(env) == 0 {
		return nil
	}
	return env
}

func resolveTarget(toolCtx ToolContext, name string) (config.TargetConfig, string, error) {
	cfg, err := toolCtx.Config.Load()
	if err != nil {
		return config.TargetConfig{}, "", err
	}

	name = strings.TrimSpace(name)
	if name == "" {
		name = strings.TrimSpace(cfg.DefaultTarget)
	}
	if name == "" {
		return config.TargetConfig{
			Name: "local",
			Mode: "local",
		}, "local", nil
	}

	target, ok := cfg.Targets[name]
	if !ok {
		return config.TargetConfig{}, "", fmt.Errorf("target %q is not configured", name)
	}
	return target, name, nil
}

func resolvePath(cwd, path string) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return "", fmt.Errorf("path is required")
	}
	root, err := filepath.Abs(strings.TrimSpace(cwd))
	if err != nil {
		return "", fmt.Errorf("resolve tool root: %w", err)
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(root, path)
	}
	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("resolve path: %w", err)
	}
	rel, err := filepath.Rel(root, absPath)
	if err != nil {
		return "", fmt.Errorf("check path boundary: %w", err)
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path %s escapes the working directory %s", absPath, root)
	}
	return absPath, nil
}

func decodeArguments(arguments string, target any) error {
	if strings.TrimSpace(arguments) == "" {
		arguments = "{}"
	}
	if err := json.Unmarshal([]byte(arguments), target); err != nil {
		return fmt.Errorf("decode tool arguments: %w", err)
	}
	return nil
}

func marshalPretty(value any) (string, error) {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return "", fmt.Errorf("encode tool output: %w", err)
	}
	return string(data), nil
}

func truncate(value string, max int) string {
	if max <= 0 || len(value) <= max {
		return value
	}
	return value[:max] + "\n...[truncated]..."
}

func objectSchema(properties map[string]any, required []string) map[string]any {
	if properties == nil {
		properties = map[string]any{}
	}
	schema := map[string]any{
		"type":       "object",
		"properties": properties,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema
}

func stringSchema(description string) map[string]any {
	return map[string]any{
		"type":        "string",
		"description": description,
	}
}

func boolSchema(description string) map[string]any {
	return map[string]any{
		"type":        "boolean",
		"description": description,
	}
}

func intSchema(description string) map[string]any {
	return map[string]any{
		"type":        "integer",
		"description": description,
	}
}

func numberSchema(description string) map[string]any {
	return map[string]any{
		"type":        "number",
		"description": description,
	}
}

func stringArraySchema(description string) map[string]any {
	return map[string]any{
		"type":        "array",
		"description": description,
		"items": map[string]any{
			"type": "string",
		},
	}
}

func defaultInt(value, fallback int) int {
	if value <= 0 {
		return fallback
	}
	return value
}

func valueOrDefault(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	return value
}

func shellQuote(value string) string {
	if value == "" {
		return "''"
	}
	return "'" + strings.ReplaceAll(value, "'", `'"'"'`) + "'"
}

func needsRemoteWorkspace(target config.TargetConfig) bool {
	mode := targets.Normalize(target.Mode)
	return mode == targets.ModeSSH || (mode == targets.ModeSim && strings.TrimSpace(target.Host) != "")
}

func remoteCuteWorkspace(target config.TargetConfig, workspacePath string) string {
	root := strings.TrimSpace(target.RemoteDir)
	if root == "" {
		root = "~/fusion"
	}
	return strings.TrimRight(root, "/") + "/" + filepath.Base(workspacePath)
}
