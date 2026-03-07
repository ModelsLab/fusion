package agent

import (
	"context"
	"encoding/json"
	"os"
	"strings"

	"github.com/ModelsLab/fusion/internal/optimize"
)

func detectRuntimeEnvironmentTool(toolCtx ToolContext) Tool {
	type input struct {
		Root string `json:"root"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "detect_runtime_environment",
			Description: "Inspect the checked-out project and infer likely runtime environments such as transformers, diffusers, vllm, sglang, or generic python.",
			InputSchema: objectSchema(
				map[string]any{
					"root": stringSchema("optional project root; defaults to the current working directory"),
				},
				nil,
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			root := strings.TrimSpace(req.Root)
			if root == "" {
				root = toolCtx.CWD
			}
			resolved, err := resolvePath(toolCtx.CWD, root)
			if err != nil {
				return "", err
			}
			detections, err := optimize.DetectRuntimeAdapters(resolved)
			if err != nil {
				return "", err
			}
			return marshalPretty(map[string]any{
				"root":       resolved,
				"detections": detections,
			})
		},
	}
}

func applyRuntimePatchTool(toolCtx ToolContext) Tool {
	type operation struct {
		Path       string `json:"path"`
		Content    string `json:"content"`
		CreateOnly bool   `json:"create_only"`
	}
	type input struct {
		Root       string      `json:"root"`
		Adapter    string      `json:"adapter"`
		Notes      []string    `json:"notes"`
		Operations []operation `json:"operations"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "apply_runtime_patch",
			Description: "Apply a generic runtime patch transaction with backups so the agent can revert later.",
			InputSchema: objectSchema(
				map[string]any{
					"root":    stringSchema("project root"),
					"adapter": stringSchema("runtime label such as transformers, diffusers, vllm, sglang, or generic-python"),
					"notes":   map[string]any{"type": "array", "items": stringSchema("note")},
					"operations": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"path":        stringSchema("path relative to the project root"),
								"content":     stringSchema("full file contents to write"),
								"create_only": boolSchema("fail if the file already exists"),
							},
						},
					},
				},
				[]string{"root", "adapter", "operations"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			root, err := resolvePath(toolCtx.CWD, req.Root)
			if err != nil {
				return "", err
			}
			ops := make([]optimize.RuntimePatchOperation, 0, len(req.Operations))
			for _, op := range req.Operations {
				ops = append(ops, optimize.RuntimePatchOperation{
					Path:       op.Path,
					Content:    op.Content,
					CreateOnly: op.CreateOnly,
				})
			}
			state, statePath, err := optimize.ApplyRuntimePatch(optimize.RuntimePatchPlan{
				Version:     1,
				Adapter:     req.Adapter,
				ProjectRoot: root,
				Notes:       append([]string{}, req.Notes...),
				Operations:  ops,
			})
			if err != nil {
				return "", err
			}
			return marshalPretty(map[string]any{
				"state_path": statePath,
				"state":      state,
			})
		},
	}
}

func revertRuntimePatchTool() Tool {
	type input struct {
		State string `json:"state"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "revert_runtime_patch",
			Description: "Revert a previously applied generic runtime patch transaction from its state file.",
			InputSchema: objectSchema(
				map[string]any{
					"state": stringSchema("runtime patch state JSON path"),
				},
				[]string{"state"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			state, err := optimize.RevertRuntimePatch(req.State)
			if err != nil {
				return "", err
			}
			return marshalPretty(state)
		},
	}
}

func createHarnessManifestTool(toolCtx ToolContext) Tool {
	type input struct {
		Task     string `json:"task"`
		Runtime  string `json:"runtime"`
		Workload string `json:"workload"`
		Name     string `json:"name"`
		SavePath string `json:"save_path"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "create_harness_manifest",
			Description: "Create a generic workload-aware harness manifest for text, image, video, audio, or editing workloads.",
			InputSchema: objectSchema(
				map[string]any{
					"task":      stringSchema("task family"),
					"runtime":   stringSchema("runtime label"),
					"workload":  stringSchema("workload label"),
					"name":      stringSchema("manifest name"),
					"save_path": stringSchema("optional path to save the manifest JSON"),
				},
				nil,
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			manifest := optimize.DefaultHarnessManifest(optimize.Request{
				Task:     req.Task,
				Workload: req.Workload,
			}, req.Runtime)
			if strings.TrimSpace(req.Name) != "" {
				manifest.Name = strings.TrimSpace(req.Name)
			}
			payload := map[string]any{"manifest": manifest}
			if strings.TrimSpace(req.SavePath) != "" {
				path, err := resolvePath(toolCtx.CWD, req.SavePath)
				if err != nil {
					return "", err
				}
				data, err := json.MarshalIndent(manifest, "", "  ")
				if err != nil {
					return "", err
				}
				data = append(data, '\n')
				if err := os.WriteFile(path, data, 0o600); err != nil {
					return "", err
				}
				payload["path"] = path
			}
			return marshalPretty(payload)
		},
	}
}

func assessHarnessTool(toolCtx ToolContext) Tool {
	type input struct {
		ManifestPath string                        `json:"manifest_path"`
		Samples      []optimize.BenchmarkRunSample `json:"samples"`
		Quality      map[string]float64            `json:"quality"`
		SavePath     string                        `json:"save_path"`
		Session      string                        `json:"session"`
		Candidate    string                        `json:"candidate"`
		Stage        string                        `json:"stage"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "assess_harness",
			Description: "Evaluate repeated benchmark samples and quality metrics against a saved generic harness manifest.",
			InputSchema: objectSchema(
				map[string]any{
					"manifest_path": stringSchema("path to a harness manifest JSON"),
					"save_path":     stringSchema("optional output path for the harness result JSON"),
					"session":       stringSchema("optional optimization session id"),
					"candidate":     stringSchema("optional candidate id"),
					"stage":         stringSchema("optional stage name; defaults to model-benchmark"),
					"quality":       map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "number"}},
					"samples":       map[string]any{"type": "array", "items": map[string]any{"type": "object"}},
				},
				[]string{"manifest_path", "samples"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			path, err := resolvePath(toolCtx.CWD, req.ManifestPath)
			if err != nil {
				return "", err
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return "", err
			}
			var manifest optimize.HarnessManifest
			if err := json.Unmarshal(data, &manifest); err != nil {
				return "", err
			}
			result := optimize.AssessHarness(manifest, req.Samples, req.Quality)
			payload := map[string]any{"result": result}
			savePath := strings.TrimSpace(req.SavePath)
			if savePath != "" {
				resolved, err := resolvePath(toolCtx.CWD, savePath)
				if err != nil {
					return "", err
				}
				encoded, err := json.MarshalIndent(result, "", "  ")
				if err != nil {
					return "", err
				}
				encoded = append(encoded, '\n')
				if err := os.WriteFile(resolved, encoded, 0o600); err != nil {
					return "", err
				}
				payload["path"] = resolved
				if strings.TrimSpace(req.Session) != "" && strings.TrimSpace(req.Candidate) != "" {
					if err := updateOptimizationSessionStage(req.Session, req.Candidate, "", "", valueOrDefault(req.Stage, "model-benchmark"), resolved, manifest.BenchmarkCommand, 0, result.Metrics); err != nil {
						payload["session_error"] = err.Error()
					}
				}
			}
			return marshalPretty(payload)
		},
	}
}

func inferHotspotsTool(toolCtx ToolContext) Tool {
	type input struct {
		Task      string   `json:"task"`
		Runtime   string   `json:"runtime"`
		Workload  string   `json:"workload"`
		Kernels   []string `json:"kernels"`
		Session   string   `json:"session"`
		Candidate string   `json:"candidate"`
		Round     int      `json:"round"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "infer_hotspots",
			Description: "Infer generic model-component hotspot attribution from kernel names and optionally persist it under a round artifact.",
			InputSchema: objectSchema(
				map[string]any{
					"task":      stringSchema("task family"),
					"runtime":   stringSchema("runtime label"),
					"workload":  stringSchema("workload label"),
					"kernels":   map[string]any{"type": "array", "items": stringSchema("kernel name")},
					"session":   stringSchema("optional session id"),
					"candidate": stringSchema("optional candidate id"),
					"round":     intSchema("optional round number"),
				},
				[]string{"kernels"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			attribution := optimize.InferHotspotAttribution(req.Task, req.Runtime, req.Workload, req.Kernels)
			payload := map[string]any{"attribution": attribution}
			if strings.TrimSpace(req.Session) != "" && strings.TrimSpace(req.Candidate) != "" && req.Round > 0 {
				session, _, err := loadOptimizationSession(req.Session)
				if err != nil {
					return "", err
				}
				path, err := optimize.SaveHotspotAttribution(session, req.Candidate, req.Round, attribution)
				if err != nil {
					return "", err
				}
				payload["path"] = path
			}
			return marshalPretty(payload)
		},
	}
}

func writeSessionMemoryTool(toolCtx ToolContext) Tool {
	type input struct {
		Session   string             `json:"session"`
		Title     string             `json:"title"`
		Category  string             `json:"category"`
		Summary   string             `json:"summary"`
		Outcome   string             `json:"outcome"`
		Candidate string             `json:"candidate"`
		Metrics   map[string]float64 `json:"metrics"`
		Lessons   []string           `json:"lessons"`
		NextSteps []string           `json:"next_steps"`
		Files     []string           `json:"files"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "write_session_memory",
			Description: "Write a markdown memory entry into the current optimization session so future runs can resume from structured history.",
			InputSchema: objectSchema(
				map[string]any{
					"session":    stringSchema("optimization session id"),
					"title":      stringSchema("memory entry title"),
					"category":   stringSchema("memory category"),
					"summary":    stringSchema("summary of what happened"),
					"outcome":    stringSchema("outcome label"),
					"candidate":  stringSchema("optional candidate id"),
					"metrics":    map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "number"}},
					"lessons":    map[string]any{"type": "array", "items": stringSchema("lesson")},
					"next_steps": map[string]any{"type": "array", "items": stringSchema("next step")},
					"files":      map[string]any{"type": "array", "items": stringSchema("file path")},
				},
				[]string{"session", "title", "summary"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			session, _, err := loadOptimizationSession(req.Session)
			if err != nil {
				return "", err
			}
			path, err := optimize.SaveSessionMemoryEntry(session, optimize.SessionMemoryEntry{
				Title:       req.Title,
				Category:    req.Category,
				Summary:     req.Summary,
				Outcome:     req.Outcome,
				CandidateID: req.Candidate,
				Metrics:     req.Metrics,
				Lessons:     append([]string{}, req.Lessons...),
				NextSteps:   append([]string{}, req.NextSteps...),
				Files:       append([]string{}, req.Files...),
			})
			if err != nil {
				return "", err
			}
			return marshalPretty(map[string]any{
				"session": session.ID,
				"path":    path,
			})
		},
	}
}
