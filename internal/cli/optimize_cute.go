package cli

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/ModelsLab/fusion/internal/cutedsl"
	"github.com/ModelsLab/fusion/internal/optimize"
	"github.com/spf13/cobra"
)

func newOptimizeCuteCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "cute",
		Short: "Scaffold and run CuTe DSL kernel workspaces",
	}

	cmd.AddCommand(
		newOptimizeCuteInitCommand(),
		newOptimizeCuteBuildCommand(),
		newOptimizeCuteVerifyCommand(),
		newOptimizeCuteBenchmarkCommand(),
	)
	return cmd
}

func newOptimizeCuteInitCommand() *cobra.Command {
	var workspaceName string
	var outputDir string
	var templateName string
	var operation string
	var gpuArch string
	var targetName string
	var sessionID string
	var force bool

	cmd := &cobra.Command{
		Use:   "init",
		Short: "Create a CuTe DSL workspace with build, verify, and benchmark scaffolds",
		RunE: func(cmd *cobra.Command, args []string) error {
			if strings.TrimSpace(workspaceName) == "" {
				if strings.TrimSpace(outputDir) != "" {
					workspaceName = filepath.Base(outputDir)
				} else {
					return fmt.Errorf("workspace name is required")
				}
			}

			sessionTarget := targetName
			if strings.TrimSpace(sessionID) != "" {
				session, store, err := loadOptimizationSession(sessionID)
				if err != nil {
					return err
				}
				outputDir = resolveSessionWorkspaceRoot(session, cutedsl.BackendID, workspaceName, outputDir)
				if sessionTarget == "" {
					sessionTarget = session.Target
				}
				workspace, err := cutedsl.Init(cutedsl.InitRequest{
					Name:      workspaceName,
					OutputDir: outputDir,
					Template:  templateName,
					Operation: operation,
					GPUArch:   gpuArch,
					Target:    sessionTarget,
					Force:     force,
				})
				if err != nil {
					return err
				}
				absOutput, err := filepath.Abs(outputDir)
				if err != nil {
					return err
				}
				if err := attachSessionCandidate(session, store, optimize.Candidate{
					Name:      workspaceName,
					Backend:   cutedsl.BackendID,
					Template:  workspace.Template,
					Operation: workspace.Operation,
					GPUArch:   workspace.GPUArch,
					Workspace: absOutput,
				}); err != nil {
					return err
				}
				cmd.Printf("Created CuTe workspace: %s\n", absOutput)
				cmd.Printf("Session: %s\n", session.ID)
				return nil
			}

			workspace, err := cutedsl.Init(cutedsl.InitRequest{
				Name:      workspaceName,
				OutputDir: outputDir,
				Template:  templateName,
				Operation: operation,
				GPUArch:   gpuArch,
				Target:    sessionTarget,
				Force:     force,
			})
			if err != nil {
				return err
			}

			absOutput := outputDir
			if strings.TrimSpace(absOutput) == "" {
				absOutput = workspaceName
			}
			absOutput, err = filepath.Abs(absOutput)
			if err != nil {
				return err
			}

			cmd.Printf("Created CuTe workspace: %s\n", absOutput)
			cmd.Printf("Template: %s\n", workspace.Template)
			cmd.Printf("Default GPU arch: %s\n", workspace.GPUArch)
			cmd.Printf("Next: fusion optimize cute build --workspace %s --gpu-arch %s\n", absOutput, workspace.GPUArch)
			return nil
		},
	}

	cmd.Flags().StringVar(&workspaceName, "name", "", "workspace name; defaults to the output directory name")
	cmd.Flags().StringVar(&outputDir, "output", "", "directory to create; defaults to ./<name>")
	cmd.Flags().StringVar(&templateName, "template", cutedsl.DefaultTemplate, "CuTe workspace template")
	cmd.Flags().StringVar(&operation, "operation", "elementwise_add_one", "logical kernel operation label for the workspace metadata")
	cmd.Flags().StringVar(&gpuArch, "gpu-arch", "sm90", "default GPU arch passed to generated scripts, for example sm90 or sm100")
	cmd.Flags().StringVar(&targetName, "target", "", "optional target label recorded in workspace metadata")
	cmd.Flags().StringVar(&sessionID, "session", "", "optimization session id to attach this candidate to")
	cmd.Flags().BoolVar(&force, "force", false, "overwrite the scaffold into a non-empty output directory")
	return cmd
}

func newOptimizeCuteBuildCommand() *cobra.Command {
	var workspaceDir string
	var sessionID string
	var candidateID string
	var targetName string
	var runName string
	var gpuArch string
	var outputPath string
	var exportDir string
	var pythonBin string
	var timeout time.Duration
	var optLevel int

	cmd := &cobra.Command{
		Use:   "build",
		Short: "Compile a CuTe DSL workspace on a local or remote target",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}
			workspacePath, workspace, err := resolveCuteWorkspace(workspaceDir, sessionID, candidateID)
			if err != nil {
				return err
			}
			target, _, err := resolveTarget(runtimeState, targetNameFromSession(sessionID, targetName))
			if err != nil {
				return err
			}
			if runName == "" {
				runName = workspace.Name + "-cute-build"
			}

			runResult, executedCommand, err := executeWorkspaceRun(workspaceRunRequest{
				Target:       target,
				WorkspaceDir: workspacePath,
				Command: cutedsl.BuildCommand(".", cutedsl.BuildArgs{
					PythonBin: pythonBin,
					GPUArch:   valueOrFallback(gpuArch, workspace.GPUArch),
					OptLevel:  optLevel,
					ExportDir: exportDir,
				}),
				Timeout: timeout,
			})
			if err != nil && runResult.ExitCode == 0 {
				return err
			}

			artifactPath, metrics, saveErr := saveKernelStageArtifact(runName, cutedsl.BackendID, "build", workspacePath, target, executedCommand, runResult, outputPath)
			if saveErr != nil {
				return saveErr
			}
			if stageErr := recordCandidateStage(sessionID, candidateID, cutedsl.BackendID, workspacePath, "build", artifactPath, executedCommand, runResult.ExitCode, metrics); stageErr != nil {
				return stageErr
			}

			cmd.Printf("Saved CuTe build artifact: %s\n", artifactPath)
			cmd.Printf("Workspace: %s\n", workspacePath)
			printKernelMetrics(cmd, metrics)
			if err != nil {
				cmd.Printf("command exited with error: %v\n", err)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&workspaceDir, "workspace", "", "path to the CuTe workspace")
	cmd.Flags().StringVar(&sessionID, "session", "", "optimization session id")
	cmd.Flags().StringVar(&candidateID, "candidate", "", "session candidate id; used when --workspace is omitted or for stage updates")
	cmd.Flags().StringVar(&targetName, "target", "", "target name; defaults to the configured default target or implicit local")
	cmd.Flags().StringVar(&runName, "name", "", "artifact name")
	cmd.Flags().StringVar(&gpuArch, "gpu-arch", "", "override the workspace GPU arch for this build")
	cmd.Flags().StringVar(&pythonBin, "python", "python3", "Python interpreter to run on the target")
	cmd.Flags().StringVar(&exportDir, "export-dir", "", "optional output directory for CuTe AOT exports")
	cmd.Flags().StringVar(&outputPath, "output", "", "custom artifact output path")
	cmd.Flags().IntVar(&optLevel, "opt-level", 3, "CuTe optimization level passed to cute.compile")
	cmd.Flags().DurationVar(&timeout, "timeout", 30*time.Minute, "command timeout")
	return cmd
}

func newOptimizeCuteVerifyCommand() *cobra.Command {
	var workspaceDir string
	var sessionID string
	var candidateID string
	var targetName string
	var runName string
	var gpuArch string
	var outputPath string
	var pythonBin string
	var timeout time.Duration
	var size int
	var atol float64
	var rtol float64

	cmd := &cobra.Command{
		Use:   "verify",
		Short: "Run correctness checks for a CuTe DSL workspace on a local or remote target",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}
			workspacePath, workspace, err := resolveCuteWorkspace(workspaceDir, sessionID, candidateID)
			if err != nil {
				return err
			}
			target, _, err := resolveTarget(runtimeState, targetNameFromSession(sessionID, targetName))
			if err != nil {
				return err
			}
			if runName == "" {
				runName = workspace.Name + "-cute-verify"
			}

			runResult, executedCommand, err := executeWorkspaceRun(workspaceRunRequest{
				Target:       target,
				WorkspaceDir: workspacePath,
				Command: cutedsl.VerifyCommand(".", cutedsl.VerifyArgs{
					PythonBin: pythonBin,
					GPUArch:   valueOrFallback(gpuArch, workspace.GPUArch),
					Size:      size,
					ATol:      atol,
					RTol:      rtol,
				}),
				Timeout: timeout,
			})
			if err != nil && runResult.ExitCode == 0 {
				return err
			}

			artifactPath, metrics, saveErr := saveKernelStageArtifact(runName, cutedsl.BackendID, "verify", workspacePath, target, executedCommand, runResult, outputPath)
			if saveErr != nil {
				return saveErr
			}
			if stageErr := recordCandidateStage(sessionID, candidateID, cutedsl.BackendID, workspacePath, "verify", artifactPath, executedCommand, runResult.ExitCode, metrics); stageErr != nil {
				return stageErr
			}

			cmd.Printf("Saved CuTe verification artifact: %s\n", artifactPath)
			cmd.Printf("Workspace: %s\n", workspacePath)
			printKernelMetrics(cmd, metrics)
			if err != nil {
				cmd.Printf("command exited with error: %v\n", err)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&workspaceDir, "workspace", "", "path to the CuTe workspace")
	cmd.Flags().StringVar(&sessionID, "session", "", "optimization session id")
	cmd.Flags().StringVar(&candidateID, "candidate", "", "session candidate id; used when --workspace is omitted or for stage updates")
	cmd.Flags().StringVar(&targetName, "target", "", "target name; defaults to the configured default target or implicit local")
	cmd.Flags().StringVar(&runName, "name", "", "artifact name")
	cmd.Flags().StringVar(&gpuArch, "gpu-arch", "", "override the workspace GPU arch for this verification run")
	cmd.Flags().StringVar(&pythonBin, "python", "python3", "Python interpreter to run on the target")
	cmd.Flags().StringVar(&outputPath, "output", "", "custom artifact output path")
	cmd.Flags().DurationVar(&timeout, "timeout", 30*time.Minute, "command timeout")
	cmd.Flags().IntVar(&size, "size", 1<<20, "number of elements used for correctness checking")
	cmd.Flags().Float64Var(&atol, "atol", 1e-6, "absolute tolerance")
	cmd.Flags().Float64Var(&rtol, "rtol", 1e-5, "relative tolerance")
	return cmd
}

func newOptimizeCuteBenchmarkCommand() *cobra.Command {
	var workspaceDir string
	var sessionID string
	var candidateID string
	var targetName string
	var runName string
	var gpuArch string
	var outputPath string
	var pythonBin string
	var timeout time.Duration
	var size int
	var warmup int
	var repeats int

	cmd := &cobra.Command{
		Use:   "benchmark",
		Short: "Benchmark a CuTe DSL workspace on a local or remote target",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}
			workspacePath, workspace, err := resolveCuteWorkspace(workspaceDir, sessionID, candidateID)
			if err != nil {
				return err
			}
			target, _, err := resolveTarget(runtimeState, targetNameFromSession(sessionID, targetName))
			if err != nil {
				return err
			}
			if runName == "" {
				runName = workspace.Name + "-cute-benchmark"
			}

			runResult, executedCommand, err := executeWorkspaceRun(workspaceRunRequest{
				Target:       target,
				WorkspaceDir: workspacePath,
				Command: cutedsl.BenchmarkCommand(".", cutedsl.BenchmarkArgs{
					PythonBin: pythonBin,
					GPUArch:   valueOrFallback(gpuArch, workspace.GPUArch),
					Size:      size,
					Warmup:    warmup,
					Repeats:   repeats,
				}),
				Timeout: timeout,
			})
			if err != nil && runResult.ExitCode == 0 {
				return err
			}

			artifactPath, metrics, saveErr := saveKernelStageArtifact(runName, cutedsl.BackendID, "benchmark", workspacePath, target, executedCommand, runResult, outputPath)
			if saveErr != nil {
				return saveErr
			}
			if stageErr := recordCandidateStage(sessionID, candidateID, cutedsl.BackendID, workspacePath, "benchmark", artifactPath, executedCommand, runResult.ExitCode, metrics); stageErr != nil {
				return stageErr
			}

			cmd.Printf("Saved CuTe benchmark artifact: %s\n", artifactPath)
			cmd.Printf("Workspace: %s\n", workspacePath)
			printKernelMetrics(cmd, metrics)
			if err != nil {
				cmd.Printf("command exited with error: %v\n", err)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&workspaceDir, "workspace", "", "path to the CuTe workspace")
	cmd.Flags().StringVar(&sessionID, "session", "", "optimization session id")
	cmd.Flags().StringVar(&candidateID, "candidate", "", "session candidate id; used when --workspace is omitted or for stage updates")
	cmd.Flags().StringVar(&targetName, "target", "", "target name; defaults to the configured default target or implicit local")
	cmd.Flags().StringVar(&runName, "name", "", "artifact name")
	cmd.Flags().StringVar(&gpuArch, "gpu-arch", "", "override the workspace GPU arch for this benchmark run")
	cmd.Flags().StringVar(&pythonBin, "python", "python3", "Python interpreter to run on the target")
	cmd.Flags().StringVar(&outputPath, "output", "", "custom artifact output path")
	cmd.Flags().DurationVar(&timeout, "timeout", 30*time.Minute, "command timeout")
	cmd.Flags().IntVar(&size, "size", 1<<20, "number of elements used for benchmarking")
	cmd.Flags().IntVar(&warmup, "warmup", 25, "number of warmup iterations")
	cmd.Flags().IntVar(&repeats, "repeats", 100, "number of benchmark repetitions")
	return cmd
}

func resolveCuteWorkspace(path, sessionID, candidateID string) (string, cutedsl.Workspace, error) {
	if strings.TrimSpace(path) == "" && strings.TrimSpace(sessionID) != "" && strings.TrimSpace(candidateID) != "" {
		_, candidate, err := candidateFromSession(sessionID, candidateID)
		if err != nil {
			return "", cutedsl.Workspace{}, err
		}
		path = candidate.Workspace
	}
	absPath, err := filepath.Abs(strings.TrimSpace(path))
	if err != nil {
		return "", cutedsl.Workspace{}, fmt.Errorf("resolve workspace path: %w", err)
	}
	workspace, err := cutedsl.Load(absPath)
	if err != nil {
		return "", cutedsl.Workspace{}, err
	}
	return absPath, workspace, nil
}

func printKernelMetrics(cmd *cobra.Command, metrics map[string]float64) {
	if len(metrics) == 0 {
		return
	}
	cmd.Println("Metrics")
	for _, key := range sortedMetricKeys(metrics) {
		cmd.Printf("- %s: %.4f\n", key, metrics[key])
	}
}
