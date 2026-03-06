package agent

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/ModelsLab/fusion/internal/optimize"
)

func registerOptimizationCandidateTool(toolCtx ToolContext) Tool {
	type input struct {
		Session         string `json:"session"`
		Name            string `json:"name"`
		Backend         string `json:"backend"`
		Workspace       string `json:"workspace"`
		Template        string `json:"template"`
		Operation       string `json:"operation"`
		GPUArch         string `json:"gpu_arch"`
		Description     string `json:"description"`
		CreateWorkspace *bool  `json:"create_workspace"`
	}

	return Tool{
		Definition: ToolDefinition{
			Name:        "register_optimization_candidate",
			Description: "Register a candidate backend/workspace inside an optimization session. Use this for baselines, Triton, CuTe, CUDA, torch.compile, quantization, or any other strategy the model chooses.",
			InputSchema: objectSchema(
				map[string]any{
					"session":          stringSchema("optimization session id"),
					"name":             stringSchema("candidate name"),
					"backend":          stringSchema("candidate backend label like baseline, triton, cute, cuda, torch-compile, awq, fp8, or nvfp4"),
					"workspace":        stringSchema("optional workspace path; defaults to <session-workspace>/<backend>-<name>"),
					"template":         stringSchema("optional template or recipe label"),
					"operation":        stringSchema("optional operation label such as attention, rmsnorm, or decode"),
					"gpu_arch":         stringSchema("optional GPU architecture label like sm89, sm90, or sm100"),
					"description":      stringSchema("optional candidate summary"),
					"create_workspace": boolSchema("create the workspace directory when omitted or true"),
				},
				[]string{"session", "backend"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}

			session, store, err := loadOptimizationSession(req.Session)
			if err != nil {
				return "", err
			}

			workspace := strings.TrimSpace(req.Workspace)
			if workspace == "" {
				baseName := strings.TrimSpace(req.Name)
				if baseName == "" {
					baseName = strings.TrimSpace(req.Operation)
				}
				if baseName == "" {
					baseName = strings.TrimSpace(req.Backend)
				}
				workspace = filepath.Join(session.WorkspaceRoot, strings.TrimSpace(req.Backend)+"-"+sanitizeWorkspaceName(baseName))
			}
			workspace, err = resolvePath(toolCtx.CWD, workspace)
			if err != nil {
				return "", err
			}

			createWorkspace := true
			if req.CreateWorkspace != nil {
				createWorkspace = *req.CreateWorkspace
			}
			if createWorkspace {
				if err := os.MkdirAll(workspace, 0o755); err != nil {
					return "", fmt.Errorf("create candidate workspace: %w", err)
				}
			}

			candidate := session.UpsertCandidate(optimize.Candidate{
				Name:        strings.TrimSpace(req.Name),
				Backend:     strings.TrimSpace(req.Backend),
				Template:    strings.TrimSpace(req.Template),
				Operation:   strings.TrimSpace(req.Operation),
				GPUArch:     strings.TrimSpace(req.GPUArch),
				Workspace:   workspace,
				Description: strings.TrimSpace(req.Description),
			})
			if _, err := store.Save(session); err != nil {
				return "", err
			}

			return marshalPretty(map[string]any{
				"session":          session.ID,
				"workspace_root":   session.WorkspaceRoot,
				"workspace":        workspace,
				"candidate":        candidate,
				"create_workspace": createWorkspace,
			})
		},
	}
}

func recordOptimizationStageTool() Tool {
	type input struct {
		Session      string             `json:"session"`
		Candidate    string             `json:"candidate"`
		Backend      string             `json:"backend"`
		Workspace    string             `json:"workspace"`
		Stage        string             `json:"stage"`
		ArtifactPath string             `json:"artifact_path"`
		Command      string             `json:"command"`
		ExitCode     int                `json:"exit_code"`
		Metrics      map[string]float64 `json:"metrics"`
	}

	return Tool{
		Definition: ToolDefinition{
			Name:        "record_optimization_stage",
			Description: "Attach an existing artifact or result to a candidate stage inside an optimization session.",
			InputSchema: objectSchema(
				map[string]any{
					"session":       stringSchema("optimization session id"),
					"candidate":     stringSchema("candidate id"),
					"backend":       stringSchema("candidate backend; used only when candidate is omitted"),
					"workspace":     stringSchema("candidate workspace path; used only when candidate is omitted"),
					"stage":         stringSchema("stage name like baseline, build, verify, benchmark, profile, patch, or model-benchmark"),
					"artifact_path": stringSchema("artifact path to attach"),
					"command":       stringSchema("command or workflow description"),
					"exit_code":     intSchema("exit code for the stage"),
					"metrics":       map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "number"}, "description": "optional numeric metrics"},
				},
				[]string{"session", "stage"},
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

			workspace := strings.TrimSpace(req.Workspace)
			if workspace != "" {
				resolved, err := resolvePath(session.ProjectRoot, workspace)
				if err != nil {
					return "", err
				}
				workspace = resolved
			}

			if err := updateOptimizationSessionStage(req.Session, req.Candidate, req.Backend, workspace, req.Stage, req.ArtifactPath, req.Command, req.ExitCode, req.Metrics); err != nil {
				return "", err
			}
			if strings.TrimSpace(req.Candidate) == "" && workspace != "" {
				if candidate, ok := session.CandidateByWorkspace(req.Backend, workspace); ok {
					req.Candidate = candidate.ID
				}
			}

			return marshalPretty(map[string]any{
				"session":   session.ID,
				"candidate": strings.TrimSpace(req.Candidate),
				"stage":     strings.TrimSpace(req.Stage),
				"recorded":  true,
			})
		},
	}
}

func resolveCandidateExecutionContext(toolCtx ToolContext, sessionID, candidateID, workdir, targetName string) (*optimize.Session, *optimize.Candidate, string, string, error) {
	sessionID = strings.TrimSpace(sessionID)
	candidateID = strings.TrimSpace(candidateID)
	workdir = strings.TrimSpace(workdir)
	targetName = strings.TrimSpace(targetName)

	if sessionID == "" {
		return nil, nil, workdir, targetName, nil
	}

	session, _, err := loadOptimizationSession(sessionID)
	if err != nil {
		return nil, nil, "", "", err
	}
	if targetName == "" {
		targetName = strings.TrimSpace(session.Target)
	}

	var candidate *optimize.Candidate
	if candidateID != "" {
		resolved, ok := session.CandidateByID(candidateID)
		if !ok {
			return nil, nil, "", "", fmt.Errorf("candidate %q not found in session %s", candidateID, session.ID)
		}
		candidate = resolved
		if workdir == "" {
			workdir = candidate.Workspace
		}
	} else if workdir == "" {
		workdir = session.ProjectRoot
	}

	return session, candidate, workdir, targetName, nil
}

func copyFile(source, dest string) error {
	input, err := os.Open(source)
	if err != nil {
		return fmt.Errorf("open source file: %w", err)
	}
	defer input.Close()

	info, err := input.Stat()
	if err != nil {
		return fmt.Errorf("stat source file: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return fmt.Errorf("create destination parent dir: %w", err)
	}

	output, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, info.Mode())
	if err != nil {
		return fmt.Errorf("open destination file: %w", err)
	}
	defer output.Close()

	if _, err := io.Copy(output, input); err != nil {
		return fmt.Errorf("copy file: %w", err)
	}
	return nil
}

func copyDirectory(source, dest string) error {
	return filepath.WalkDir(source, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}

		rel, err := filepath.Rel(source, path)
		if err != nil {
			return fmt.Errorf("resolve relative path: %w", err)
		}
		targetPath := dest
		if rel != "." {
			targetPath = filepath.Join(dest, rel)
		}

		if d.IsDir() {
			info, err := d.Info()
			if err != nil {
				return fmt.Errorf("stat source directory: %w", err)
			}
			if err := os.MkdirAll(targetPath, info.Mode()); err != nil {
				return fmt.Errorf("create destination directory: %w", err)
			}
			return nil
		}

		return copyFile(path, targetPath)
	})
}
