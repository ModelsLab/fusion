package agent

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/ModelsLab/fusion/internal/artifacts"
	"github.com/ModelsLab/fusion/internal/config"
	"github.com/ModelsLab/fusion/internal/cudaworkspace"
	"github.com/ModelsLab/fusion/internal/cutedsl"
	"github.com/ModelsLab/fusion/internal/kb"
	"github.com/ModelsLab/fusion/internal/optimize"
	"github.com/ModelsLab/fusion/internal/runner"
	"github.com/ModelsLab/fusion/internal/system"
	"github.com/ModelsLab/fusion/internal/targets"
	"github.com/ModelsLab/fusion/internal/tritonws"
)

func createOptimizationSessionTool(toolCtx ToolContext) Tool {
	type input struct {
		Name                string   `json:"name"`
		Query               string   `json:"query"`
		Target              string   `json:"target"`
		GPU                 string   `json:"gpu"`
		Model               string   `json:"model"`
		Workload            string   `json:"workload"`
		Operators           []string `json:"operators"`
		Precision           string   `json:"precision"`
		Bottleneck          string   `json:"bottleneck"`
		Runtime             string   `json:"runtime"`
		Goals               []string `json:"goals"`
		WorkspaceRoot       string   `json:"workspace_root"`
		IncludeExperimental bool     `json:"include_experimental"`
		Limit               int      `json:"limit"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "create_optimization_session",
			Description: "Create a persistent optimization session with a retrieved context packet and workspace root.",
			InputSchema: objectSchema(
				map[string]any{
					"name":                 stringSchema("session name"),
					"query":                stringSchema("free-form optimization request"),
					"target":               stringSchema("optional configured target name"),
					"gpu":                  stringSchema("GPU id or name"),
					"model":                stringSchema("model name or family"),
					"workload":             stringSchema("decode, prefill, serving, or training-prep"),
					"operators":            stringArraySchema("operator families"),
					"precision":            stringSchema("precision or quantization path"),
					"bottleneck":           stringSchema("memory, compute, latency, or mixed"),
					"runtime":              stringSchema("runtime like vllm, tensorrt-llm, transformers, or sglang"),
					"goals":                stringArraySchema("optimization goals"),
					"workspace_root":       stringSchema("optional override for the session workspace root"),
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

			request := optimize.Request{
				GPU:                 req.GPU,
				Model:               req.Model,
				Workload:            req.Workload,
				Operators:           req.Operators,
				Precision:           req.Precision,
				Bottleneck:          req.Bottleneck,
				Goals:               req.Goals,
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

			packet := toolCtx.KB.BuildContextPacket(kb.ContextRequest{
				Query:               req.Query,
				GPU:                 request.GPU,
				Model:               request.Model,
				Workload:            request.Workload,
				Operators:           request.Operators,
				Precision:           request.Precision,
				Bottleneck:          request.Bottleneck,
				Runtime:             req.Runtime,
				Goals:               request.Goals,
				IncludeExperimental: req.IncludeExperimental,
				Limit:               req.Limit,
			})

			store, err := optimize.NewSessionStore()
			if err != nil {
				return "", err
			}

			workspaceRoot := req.WorkspaceRoot
			if strings.TrimSpace(workspaceRoot) != "" {
				workspaceRoot, err = resolvePath(toolCtx.CWD, workspaceRoot)
				if err != nil {
					return "", err
				}
			}

			session := store.NewSession(optimize.SessionCreateRequest{
				Name:          req.Name,
				ProjectRoot:   toolCtx.CWD,
				WorkspaceRoot: workspaceRoot,
				Target:        req.Target,
				Runtime:       req.Runtime,
				Query:         req.Query,
				Request:       request,
				Context:       packet,
				Notes:         packet.Notes,
			})
			session.Status = "ready"
			path, err := store.Save(session)
			if err != nil {
				return "", err
			}

			return marshalPretty(map[string]any{
				"session":        session,
				"metadata_path":  path,
				"workspace_root": session.WorkspaceRoot,
			})
		},
	}
}

func showOptimizationSessionTool() Tool {
	type input struct {
		Session string `json:"session"`
	}
	return Tool{
		Definition: ToolDefinition{
			Name:        "show_optimization_session",
			Description: "Load a saved optimization session, including candidates and stage artifacts.",
			InputSchema: objectSchema(
				map[string]any{
					"session": stringSchema("optimization session id"),
				},
				[]string{"session"},
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
			return marshalPretty(session)
		},
	}
}

func initTritonWorkspaceTool(toolCtx ToolContext) Tool {
	type input struct {
		Session   string `json:"session"`
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
			Name:        "init_triton_workspace",
			Description: "Create a Triton workspace scaffold and optionally attach it to an optimization session.",
			InputSchema: objectSchema(
				map[string]any{
					"session":   stringSchema("optional optimization session id"),
					"name":      stringSchema("workspace name"),
					"output":    stringSchema("output directory"),
					"template":  stringSchema("Triton template name"),
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

			output, session, store, err := resolveBackendOutput(toolCtx.CWD, req.Session, tritonws.BackendID, req.Name, req.Output)
			if err != nil {
				return "", err
			}
			target := valueOrDefault(req.Target, sessionTarget(session))
			workspace, err := tritonws.Init(tritonws.InitRequest{
				Name:      req.Name,
				OutputDir: output,
				Template:  req.Template,
				Operation: req.Operation,
				GPUArch:   req.GPUArch,
				Target:    target,
				Force:     req.Force,
			})
			if err != nil {
				return "", err
			}
			if session != nil && store != nil {
				candidate := session.UpsertCandidate(optimize.Candidate{
					Name:      req.Name,
					Backend:   tritonws.BackendID,
					Template:  workspace.Template,
					Operation: workspace.Operation,
					GPUArch:   workspace.GPUArch,
					Workspace: output,
				})
				if _, err := store.Save(session); err != nil {
					return "", err
				}
				return marshalPretty(map[string]any{
					"workspace": workspace,
					"path":      output,
					"candidate": candidate,
					"session":   session.ID,
				})
			}
			return marshalPretty(map[string]any{
				"workspace": workspace,
				"path":      output,
			})
		},
	}
}

func initCUDAWorkspaceTool(toolCtx ToolContext) Tool {
	type input struct {
		Session   string `json:"session"`
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
			Name:        "init_cuda_workspace",
			Description: "Create a CUDA C++ workspace scaffold and optionally attach it to an optimization session.",
			InputSchema: objectSchema(
				map[string]any{
					"session":   stringSchema("optional optimization session id"),
					"name":      stringSchema("workspace name"),
					"output":    stringSchema("output directory"),
					"template":  stringSchema("CUDA template name"),
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

			output, session, store, err := resolveBackendOutput(toolCtx.CWD, req.Session, cudaworkspace.BackendID, req.Name, req.Output)
			if err != nil {
				return "", err
			}
			target := valueOrDefault(req.Target, sessionTarget(session))
			workspace, err := cudaworkspace.Init(cudaworkspace.InitRequest{
				Name:      req.Name,
				OutputDir: output,
				Template:  req.Template,
				Operation: req.Operation,
				GPUArch:   req.GPUArch,
				Target:    target,
				Force:     req.Force,
			})
			if err != nil {
				return "", err
			}
			if session != nil && store != nil {
				candidate := session.UpsertCandidate(optimize.Candidate{
					Name:      req.Name,
					Backend:   cudaworkspace.BackendID,
					Template:  workspace.Template,
					Operation: workspace.Operation,
					GPUArch:   workspace.GPUArch,
					Workspace: output,
				})
				if _, err := store.Save(session); err != nil {
					return "", err
				}
				return marshalPretty(map[string]any{
					"workspace": workspace,
					"path":      output,
					"candidate": candidate,
					"session":   session.ID,
				})
			}
			return marshalPretty(map[string]any{
				"workspace": workspace,
				"path":      output,
			})
		},
	}
}

func initCuteWorkspaceSessionTool(toolCtx ToolContext) Tool {
	type input struct {
		Session   string `json:"session"`
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
			Description: "Create a CuTe DSL workspace scaffold and optionally attach it to an optimization session.",
			InputSchema: objectSchema(
				map[string]any{
					"session":   stringSchema("optional optimization session id"),
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

			output, session, store, err := resolveBackendOutput(toolCtx.CWD, req.Session, cutedsl.BackendID, req.Name, req.Output)
			if err != nil {
				return "", err
			}
			target := valueOrDefault(req.Target, sessionTarget(session))
			workspace, err := cutedsl.Init(cutedsl.InitRequest{
				Name:      req.Name,
				OutputDir: output,
				Template:  req.Template,
				Operation: req.Operation,
				GPUArch:   req.GPUArch,
				Target:    target,
				Force:     req.Force,
			})
			if err != nil {
				return "", err
			}
			if session != nil && store != nil {
				candidate := session.UpsertCandidate(optimize.Candidate{
					Name:      req.Name,
					Backend:   cutedsl.BackendID,
					Template:  workspace.Template,
					Operation: workspace.Operation,
					GPUArch:   workspace.GPUArch,
					Workspace: output,
				})
				if _, err := store.Save(session); err != nil {
					return "", err
				}
				return marshalPretty(map[string]any{
					"workspace": workspace,
					"path":      output,
					"candidate": candidate,
					"session":   session.ID,
				})
			}
			return marshalPretty(map[string]any{
				"workspace": workspace,
				"path":      output,
			})
		},
	}
}

func buildTritonWorkspaceTool(toolCtx ToolContext) Tool {
	type input struct {
		Session        string `json:"session"`
		Candidate      string `json:"candidate"`
		Workspace      string `json:"workspace"`
		Target         string `json:"target"`
		GPUArch        string `json:"gpu_arch"`
		Python         string `json:"python"`
		Size           int    `json:"size"`
		TimeoutSeconds int    `json:"timeout_seconds"`
	}
	return kernelWorkspaceStageTool(toolCtx, kernelStageSpec{
		Name:        "build_triton_workspace",
		Description: "Compile a Triton workspace on a local or remote target and persist the build artifact.",
		Backend:     tritonws.BackendID,
		Stage:       "build",
		LoadName: func(path string) (string, error) {
			workspace, err := tritonws.Load(path)
			if err != nil {
				return "", err
			}
			return workspace.Name, nil
		},
		BuildCommand: func(req kernelStageRequest) string {
			return tritonws.BuildCommand(".", tritonws.BuildArgs{
				PythonBin: req.Python,
				GPUArch:   req.GPUArch,
				Size:      req.Size,
			})
		},
		InputSchema: objectSchema(
			map[string]any{
				"session":         stringSchema("optional optimization session id"),
				"candidate":       stringSchema("optional session candidate id"),
				"workspace":       stringSchema("workspace path"),
				"target":          stringSchema("optional configured target name"),
				"gpu_arch":        stringSchema("override GPU arch"),
				"python":          stringSchema("python interpreter"),
				"size":            intSchema("number of elements used for the JIT warmup"),
				"timeout_seconds": intSchema("command timeout in seconds"),
			},
			nil,
		),
	})
}

func verifyTritonWorkspaceTool(toolCtx ToolContext) Tool {
	return kernelWorkspaceStageTool(toolCtx, kernelStageSpec{
		Name:        "verify_triton_workspace",
		Description: "Run correctness verification for a Triton workspace and persist the verification artifact.",
		Backend:     tritonws.BackendID,
		Stage:       "verify",
		LoadName: func(path string) (string, error) {
			workspace, err := tritonws.Load(path)
			if err != nil {
				return "", err
			}
			return workspace.Name, nil
		},
		BuildCommand: func(req kernelStageRequest) string {
			return tritonws.VerifyCommand(".", tritonws.VerifyArgs{
				PythonBin: req.Python,
				GPUArch:   req.GPUArch,
				Size:      req.Size,
				ATol:      req.ATol,
				RTol:      req.RTol,
			})
		},
		InputSchema: objectSchema(
			map[string]any{
				"session":         stringSchema("optional optimization session id"),
				"candidate":       stringSchema("optional session candidate id"),
				"workspace":       stringSchema("workspace path"),
				"target":          stringSchema("optional configured target name"),
				"gpu_arch":        stringSchema("override GPU arch"),
				"python":          stringSchema("python interpreter"),
				"size":            intSchema("number of elements"),
				"atol":            numberSchema("absolute tolerance"),
				"rtol":            numberSchema("relative tolerance"),
				"timeout_seconds": intSchema("command timeout in seconds"),
			},
			nil,
		),
	})
}

func benchmarkTritonWorkspaceTool(toolCtx ToolContext) Tool {
	return kernelWorkspaceStageTool(toolCtx, kernelStageSpec{
		Name:        "benchmark_triton_workspace",
		Description: "Benchmark a Triton workspace and persist the benchmark artifact.",
		Backend:     tritonws.BackendID,
		Stage:       "benchmark",
		LoadName: func(path string) (string, error) {
			workspace, err := tritonws.Load(path)
			if err != nil {
				return "", err
			}
			return workspace.Name, nil
		},
		BuildCommand: func(req kernelStageRequest) string {
			return tritonws.BenchmarkCommand(".", tritonws.BenchmarkArgs{
				PythonBin: req.Python,
				GPUArch:   req.GPUArch,
				Size:      req.Size,
				Warmup:    req.Warmup,
				Repeats:   req.Repeats,
			})
		},
		InputSchema: objectSchema(
			map[string]any{
				"session":         stringSchema("optional optimization session id"),
				"candidate":       stringSchema("optional session candidate id"),
				"workspace":       stringSchema("workspace path"),
				"target":          stringSchema("optional configured target name"),
				"gpu_arch":        stringSchema("override GPU arch"),
				"python":          stringSchema("python interpreter"),
				"size":            intSchema("number of elements"),
				"warmup":          intSchema("warmup iterations"),
				"repeats":         intSchema("benchmark repetitions"),
				"timeout_seconds": intSchema("command timeout in seconds"),
			},
			nil,
		),
	})
}

func buildCUDAWorkspaceTool(toolCtx ToolContext) Tool {
	return kernelWorkspaceStageTool(toolCtx, kernelStageSpec{
		Name:        "build_cuda_workspace",
		Description: "Compile a CUDA workspace on a local or remote target and persist the build artifact.",
		Backend:     cudaworkspace.BackendID,
		Stage:       "build",
		LoadName: func(path string) (string, error) {
			workspace, err := cudaworkspace.Load(path)
			if err != nil {
				return "", err
			}
			return workspace.Name, nil
		},
		BuildCommand: func(req kernelStageRequest) string {
			return cudaworkspace.BuildCommand(".", cudaworkspace.BuildArgs{
				PythonBin: req.Python,
				GPUArch:   req.GPUArch,
				Output:    req.Binary,
			})
		},
		InputSchema: objectSchema(
			map[string]any{
				"session":         stringSchema("optional optimization session id"),
				"candidate":       stringSchema("optional session candidate id"),
				"workspace":       stringSchema("workspace path"),
				"target":          stringSchema("optional configured target name"),
				"gpu_arch":        stringSchema("override GPU arch"),
				"python":          stringSchema("python interpreter"),
				"binary":          stringSchema("output binary path inside the workspace"),
				"timeout_seconds": intSchema("command timeout in seconds"),
			},
			nil,
		),
	})
}

func verifyCUDAWorkspaceTool(toolCtx ToolContext) Tool {
	return kernelWorkspaceStageTool(toolCtx, kernelStageSpec{
		Name:        "verify_cuda_workspace",
		Description: "Run correctness verification for a CUDA workspace and persist the verification artifact.",
		Backend:     cudaworkspace.BackendID,
		Stage:       "verify",
		LoadName: func(path string) (string, error) {
			workspace, err := cudaworkspace.Load(path)
			if err != nil {
				return "", err
			}
			return workspace.Name, nil
		},
		BuildCommand: func(req kernelStageRequest) string {
			return cudaworkspace.VerifyCommand(".", cudaworkspace.VerifyArgs{
				PythonBin: req.Python,
				GPUArch:   req.GPUArch,
				Binary:    req.Binary,
				Size:      req.Size,
				ATol:      req.ATol,
				RTol:      req.RTol,
			})
		},
		InputSchema: objectSchema(
			map[string]any{
				"session":         stringSchema("optional optimization session id"),
				"candidate":       stringSchema("optional session candidate id"),
				"workspace":       stringSchema("workspace path"),
				"target":          stringSchema("optional configured target name"),
				"gpu_arch":        stringSchema("override GPU arch"),
				"python":          stringSchema("python interpreter"),
				"binary":          stringSchema("binary path inside the workspace"),
				"size":            intSchema("number of elements"),
				"atol":            numberSchema("absolute tolerance"),
				"rtol":            numberSchema("relative tolerance"),
				"timeout_seconds": intSchema("command timeout in seconds"),
			},
			nil,
		),
	})
}

func benchmarkCUDAWorkspaceTool(toolCtx ToolContext) Tool {
	return kernelWorkspaceStageTool(toolCtx, kernelStageSpec{
		Name:        "benchmark_cuda_workspace",
		Description: "Benchmark a CUDA workspace and persist the benchmark artifact.",
		Backend:     cudaworkspace.BackendID,
		Stage:       "benchmark",
		LoadName: func(path string) (string, error) {
			workspace, err := cudaworkspace.Load(path)
			if err != nil {
				return "", err
			}
			return workspace.Name, nil
		},
		BuildCommand: func(req kernelStageRequest) string {
			return cudaworkspace.BenchmarkCommand(".", cudaworkspace.BenchmarkArgs{
				PythonBin: req.Python,
				GPUArch:   req.GPUArch,
				Binary:    req.Binary,
				Size:      req.Size,
				Warmup:    req.Warmup,
				Repeats:   req.Repeats,
			})
		},
		InputSchema: objectSchema(
			map[string]any{
				"session":         stringSchema("optional optimization session id"),
				"candidate":       stringSchema("optional session candidate id"),
				"workspace":       stringSchema("workspace path"),
				"target":          stringSchema("optional configured target name"),
				"gpu_arch":        stringSchema("override GPU arch"),
				"python":          stringSchema("python interpreter"),
				"binary":          stringSchema("binary path inside the workspace"),
				"size":            intSchema("number of elements"),
				"warmup":          intSchema("warmup iterations"),
				"repeats":         intSchema("benchmark repetitions"),
				"timeout_seconds": intSchema("command timeout in seconds"),
			},
			nil,
		),
	})
}

func buildCuteWorkspaceSessionTool(toolCtx ToolContext) Tool {
	return kernelWorkspaceStageTool(toolCtx, kernelStageSpec{
		Name:        "build_cute_workspace",
		Description: "Compile a CuTe DSL workspace on a local or remote target and persist the build artifact.",
		Backend:     cutedsl.BackendID,
		Stage:       "build",
		LoadName: func(path string) (string, error) {
			workspace, err := cutedsl.Load(path)
			if err != nil {
				return "", err
			}
			return workspace.Name, nil
		},
		BuildCommand: func(req kernelStageRequest) string {
			return cutedsl.BuildCommand(".", cutedsl.BuildArgs{
				PythonBin: req.Python,
				GPUArch:   req.GPUArch,
				OptLevel:  req.OptLevel,
				ExportDir: req.ExportDir,
			})
		},
		InputSchema: objectSchema(
			map[string]any{
				"session":         stringSchema("optional optimization session id"),
				"candidate":       stringSchema("optional session candidate id"),
				"workspace":       stringSchema("workspace path"),
				"target":          stringSchema("optional configured target name"),
				"gpu_arch":        stringSchema("override GPU arch"),
				"python":          stringSchema("python interpreter"),
				"export_dir":      stringSchema("optional AOT export directory"),
				"opt_level":       intSchema("CuTe optimization level"),
				"timeout_seconds": intSchema("command timeout in seconds"),
			},
			nil,
		),
	})
}

func verifyCuteWorkspaceSessionTool(toolCtx ToolContext) Tool {
	return kernelWorkspaceStageTool(toolCtx, kernelStageSpec{
		Name:        "verify_cute_workspace",
		Description: "Run correctness verification for a CuTe DSL workspace and persist the verification artifact.",
		Backend:     cutedsl.BackendID,
		Stage:       "verify",
		LoadName: func(path string) (string, error) {
			workspace, err := cutedsl.Load(path)
			if err != nil {
				return "", err
			}
			return workspace.Name, nil
		},
		BuildCommand: func(req kernelStageRequest) string {
			return cutedsl.VerifyCommand(".", cutedsl.VerifyArgs{
				PythonBin: req.Python,
				GPUArch:   req.GPUArch,
				Size:      req.Size,
				ATol:      req.ATol,
				RTol:      req.RTol,
			})
		},
		InputSchema: objectSchema(
			map[string]any{
				"session":         stringSchema("optional optimization session id"),
				"candidate":       stringSchema("optional session candidate id"),
				"workspace":       stringSchema("workspace path"),
				"target":          stringSchema("optional configured target name"),
				"gpu_arch":        stringSchema("override GPU arch"),
				"python":          stringSchema("python interpreter"),
				"size":            intSchema("number of elements"),
				"atol":            numberSchema("absolute tolerance"),
				"rtol":            numberSchema("relative tolerance"),
				"timeout_seconds": intSchema("command timeout in seconds"),
			},
			nil,
		),
	})
}

func benchmarkCuteWorkspaceSessionTool(toolCtx ToolContext) Tool {
	return kernelWorkspaceStageTool(toolCtx, kernelStageSpec{
		Name:        "benchmark_cute_workspace",
		Description: "Benchmark a CuTe DSL workspace and persist the benchmark artifact.",
		Backend:     cutedsl.BackendID,
		Stage:       "benchmark",
		LoadName: func(path string) (string, error) {
			workspace, err := cutedsl.Load(path)
			if err != nil {
				return "", err
			}
			return workspace.Name, nil
		},
		BuildCommand: func(req kernelStageRequest) string {
			return cutedsl.BenchmarkCommand(".", cutedsl.BenchmarkArgs{
				PythonBin: req.Python,
				GPUArch:   req.GPUArch,
				Size:      req.Size,
				Warmup:    req.Warmup,
				Repeats:   req.Repeats,
			})
		},
		InputSchema: objectSchema(
			map[string]any{
				"session":         stringSchema("optional optimization session id"),
				"candidate":       stringSchema("optional session candidate id"),
				"workspace":       stringSchema("workspace path"),
				"target":          stringSchema("optional configured target name"),
				"gpu_arch":        stringSchema("override GPU arch"),
				"python":          stringSchema("python interpreter"),
				"size":            intSchema("number of elements"),
				"warmup":          intSchema("warmup iterations"),
				"repeats":         intSchema("benchmark repetitions"),
				"timeout_seconds": intSchema("command timeout in seconds"),
			},
			nil,
		),
	})
}

type kernelStageRequest struct {
	Session        string  `json:"session"`
	Candidate      string  `json:"candidate"`
	Workspace      string  `json:"workspace"`
	Target         string  `json:"target"`
	GPUArch        string  `json:"gpu_arch"`
	Python         string  `json:"python"`
	Binary         string  `json:"binary"`
	ExportDir      string  `json:"export_dir"`
	OptLevel       int     `json:"opt_level"`
	Size           int     `json:"size"`
	Warmup         int     `json:"warmup"`
	Repeats        int     `json:"repeats"`
	ATol           float64 `json:"atol"`
	RTol           float64 `json:"rtol"`
	TimeoutSeconds int     `json:"timeout_seconds"`
}

type kernelStageSpec struct {
	Name         string
	Description  string
	Backend      string
	Stage        string
	InputSchema  map[string]any
	LoadName     func(path string) (string, error)
	BuildCommand func(req kernelStageRequest) string
}

func kernelWorkspaceStageTool(toolCtx ToolContext, spec kernelStageSpec) Tool {
	return Tool{
		Definition: ToolDefinition{
			Name:        spec.Name,
			Description: spec.Description,
			InputSchema: spec.InputSchema,
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req kernelStageRequest
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}

			workspacePath, candidateID, sessionID, err := resolveSessionWorkspace(toolCtx.CWD, spec.Backend, req.Session, req.Candidate, req.Workspace)
			if err != nil {
				return "", err
			}
			workspaceName, err := spec.LoadName(workspacePath)
			if err != nil {
				return "", err
			}

			targetName := req.Target
			if strings.TrimSpace(targetName) == "" && strings.TrimSpace(sessionID) != "" {
				session, _, loadErr := loadOptimizationSession(sessionID)
				if loadErr == nil {
					targetName = session.Target
				}
			}
			target, _, err := resolveTarget(toolCtx, targetName)
			if err != nil {
				return "", err
			}
			command := spec.BuildCommand(req)
			executedCommand, runResult, err := executeKernelWorkspace(ctx, toolCtx, workspacePath, target, time.Duration(defaultInt(req.TimeoutSeconds, 1800))*time.Second, command)
			if err != nil && runResult.ExitCode == 0 {
				return "", err
			}

			artifactPath, metrics, artifactErr := saveKernelArtifact(spec.Backend, spec.Stage, workspaceName, workspacePath, target, executedCommand, runResult)
			if artifactErr != nil {
				return "", artifactErr
			}
			if stageErr := updateOptimizationSessionStage(sessionID, candidateID, spec.Backend, workspacePath, spec.Stage, artifactPath, executedCommand, runResult.ExitCode, metrics); stageErr != nil {
				return "", stageErr
			}

			payload := map[string]any{
				"workspace":     workspaceName,
				"path":          workspacePath,
				"artifact_path": artifactPath,
				"command":       executedCommand,
				"duration_ms":   runResult.DurationMS,
				"exit_code":     runResult.ExitCode,
				"stdout":        truncate(runResult.Stdout, 24000),
				"stderr":        truncate(runResult.Stderr, 12000),
				"warnings":      runResult.Warnings,
			}
			if len(metrics) > 0 {
				payload["metrics"] = metrics
			}
			if sessionID != "" {
				payload["session"] = sessionID
				payload["candidate"] = candidateID
			}
			if err != nil {
				payload["error"] = err.Error()
			}
			return marshalPretty(payload)
		},
	}
}

func resolveBackendOutput(cwd, sessionID, backend, name, output string) (string, *optimize.Session, *optimize.SessionStore, error) {
	output = strings.TrimSpace(output)
	if sessionID == "" {
		if output == "" {
			output = name
		}
		resolved, err := resolvePath(cwd, output)
		if err != nil {
			return "", nil, nil, err
		}
		return resolved, nil, nil, nil
	}

	session, store, err := loadOptimizationSession(sessionID)
	if err != nil {
		return "", nil, nil, err
	}
	if output == "" {
		output = filepath.Join(session.WorkspaceRoot, backend+"-"+sanitizeWorkspaceName(name))
	}
	resolved, err := resolvePath(cwd, output)
	if err != nil {
		return "", nil, nil, err
	}
	return resolved, session, store, nil
}

func sessionTarget(session *optimize.Session) string {
	if session == nil {
		return ""
	}
	return session.Target
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

func resolveSessionWorkspace(cwd, backend, sessionID, candidateID, workspacePath string) (string, string, string, error) {
	if strings.TrimSpace(workspacePath) == "" && strings.TrimSpace(sessionID) != "" && strings.TrimSpace(candidateID) != "" {
		session, _, err := loadOptimizationSession(sessionID)
		if err != nil {
			return "", "", "", err
		}
		candidate, ok := session.CandidateByID(candidateID)
		if !ok {
			return "", "", "", fmt.Errorf("candidate %q not found in session %s", candidateID, session.ID)
		}
		workspacePath = candidate.Workspace
	}
	if strings.TrimSpace(workspacePath) == "" {
		return "", "", "", fmt.Errorf("workspace is required")
	}
	resolved, err := resolvePath(cwd, workspacePath)
	if err != nil {
		return "", "", "", err
	}
	if sessionID != "" && candidateID == "" {
		session, _, err := loadOptimizationSession(sessionID)
		if err != nil {
			return "", "", "", err
		}
		if candidate, ok := session.CandidateByWorkspace(backend, resolved); ok {
			candidateID = candidate.ID
		}
	}
	return resolved, candidateID, sessionID, nil
}

func executeKernelWorkspace(ctx context.Context, toolCtx ToolContext, workspacePath string, target config.TargetConfig, timeout time.Duration, command string) (string, runner.Result, error) {
	executedCommand := formatDisplayedToolCommand(workspacePath, command)
	runReq := runner.Request{
		Target:  target,
		Command: command,
		WorkDir: workspacePath,
		Timeout: timeout,
	}
	if needsRemoteTargetWorkspace(target) {
		remoteWorkspace := remoteKernelWorkspace(target, workspacePath)
		copyTarget := target
		copyTarget.RemoteDir = ""
		if _, err := runner.Copy(runner.CopyRequest{
			Target:    copyTarget,
			Source:    workspacePath,
			Dest:      remoteWorkspace,
			Recursive: true,
			Timeout:   timeout,
		}); err != nil {
			return "", runner.Result{}, fmt.Errorf("copy workspace to target: %w", err)
		}
		runReq.WorkDir = remoteWorkspace
		executedCommand = formatDisplayedToolCommand(remoteWorkspace, command)
	}

	runReq.Target = target
	runResult, runErr := runner.Execute(runReq)
	return executedCommand, runResult, runErr
}

func saveKernelArtifact(backend, stage, workspaceName, workspacePath string, target config.TargetConfig, command string, runResult runner.Result) (string, map[string]float64, error) {
	metrics := artifacts.ParseMetrics(runResult.Stdout)
	store, err := artifacts.NewStore()
	if err != nil {
		return "", nil, err
	}
	artifact := artifacts.KernelRunResult{
		Name:          workspaceName + "-" + backend + "-" + stage,
		Backend:       backend,
		Stage:         stage,
		Workspace:     workspacePath,
		TargetName:    target.Name,
		TargetMode:    target.Mode,
		Command:       command,
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
	path, err := store.SaveKernelRun(artifact, "")
	if err != nil {
		return "", nil, err
	}
	return path, metrics, nil
}

func updateOptimizationSessionStage(sessionID, candidateID, backend, workspacePath, stage, artifactPath, command string, exitCode int, metrics map[string]float64) error {
	if strings.TrimSpace(sessionID) == "" {
		return nil
	}
	session, store, err := loadOptimizationSession(sessionID)
	if err != nil {
		return err
	}
	if strings.TrimSpace(candidateID) == "" {
		if candidate, ok := session.CandidateByWorkspace(backend, workspacePath); ok {
			candidateID = candidate.ID
		}
	}
	if candidateID == "" {
		return fmt.Errorf("no candidate was registered for backend %s at %s in session %s", backend, workspacePath, session.ID)
	}
	if err := session.RecordCandidateStage(candidateID, stage, artifactPath, command, exitCode, metrics); err != nil {
		return err
	}
	_, err = store.Save(session)
	return err
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

func needsRemoteTargetWorkspace(target config.TargetConfig) bool {
	mode := targets.Normalize(target.Mode)
	return mode == targets.ModeSSH || (mode == targets.ModeSim && strings.TrimSpace(target.Host) != "")
}

func remoteKernelWorkspace(target config.TargetConfig, workspacePath string) string {
	root := strings.TrimSpace(target.RemoteDir)
	if root == "" {
		root = "~/fusion"
	}
	return strings.TrimRight(root, "/") + "/" + filepath.Base(workspacePath)
}
