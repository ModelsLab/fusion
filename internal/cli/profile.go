package cli

import (
	"encoding/json"
	"os"
	"time"

	"github.com/ModelsLab/fusion/internal/artifacts"
	"github.com/ModelsLab/fusion/internal/optimize"
	"github.com/ModelsLab/fusion/internal/runner"
	"github.com/spf13/cobra"
)

func newProfileCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "profile",
		Short: "Run profiling commands and persist the raw artifacts",
	}

	cmd.AddCommand(
		newProfileRunCommand(),
		newProfileAnalyzeCommand(),
	)
	return cmd
}

func newProfileRunCommand() *cobra.Command {
	var targetName string
	var runName string
	var tool string
	var command string
	var outputPath string
	var timeout time.Duration

	cmd := &cobra.Command{
		Use:   "run",
		Short: "Execute a profiling command on a target and save the result",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			target, resolvedName, err := resolveTarget(runtimeState, targetName)
			if err != nil {
				return err
			}
			env, err := runtimeShellEnv(runtimeState)
			if err != nil {
				return err
			}
			if runName == "" {
				runName = resolvedName + "-profile"
			}

			runResult, err := runner.Execute(runner.Request{
				Target:  target,
				Command: command,
				Env:     env,
				Timeout: timeout,
			})
			if err != nil && runResult.ExitCode == 0 {
				return err
			}

			store, err := artifacts.NewStore()
			if err != nil {
				return err
			}

			artifact := artifacts.ProfileResult{
				Name:          runName,
				Tool:          tool,
				TargetName:    target.Name,
				TargetMode:    target.Mode,
				Command:       command,
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

			path, saveErr := store.SaveProfile(artifact, outputPath)
			if saveErr != nil {
				return saveErr
			}

			cmd.Printf("Saved profile artifact: %s\n", path)
			cmd.Printf("Tool: %s\n", valueOrFallback(tool, "custom"))
			cmd.Printf("Wall time: %d ms\n", artifact.DurationMS)
			for _, warning := range artifact.Warnings {
				cmd.Printf("warning: %s\n", warning)
			}
			if err != nil {
				cmd.Printf("command exited with error: %v\n", err)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&targetName, "target", "", "target name; defaults to the configured default target or implicit local")
	cmd.Flags().StringVar(&runName, "name", "", "profile run name")
	cmd.Flags().StringVar(&tool, "tool", "custom", "profile tool label, for example ncu or nsys")
	cmd.Flags().StringVar(&command, "command", "", "shell command to execute")
	cmd.Flags().StringVar(&outputPath, "output", "", "custom artifact output path")
	cmd.Flags().DurationVar(&timeout, "timeout", 30*time.Minute, "command timeout")
	cmd.MarkFlagRequired("command")
	return cmd
}

func newProfileAnalyzeCommand() *cobra.Command {
	var artifactPath string
	var tool string
	var outputPath string

	cmd := &cobra.Command{
		Use:   "analyze",
		Short: "Parse a saved profile artifact into stable Nsight metrics, a bottleneck report, and a prescription",
		RunE: func(cmd *cobra.Command, args []string) error {
			store, err := artifacts.NewStore()
			if err != nil {
				return err
			}
			artifact, err := store.LoadProfile(artifactPath)
			if err != nil {
				return err
			}

			resolvedTool := valueOrFallback(tool, artifact.Tool)
			profile := optimize.ParseNsightProfile(resolvedTool, artifact.Stdout, artifact.Stderr)
			report := optimize.AnalyzeRoofline(profile)
			prescription := optimize.PrescribeFromReport(report, optimize.Request{}, optimize.Candidate{
				Name:    artifact.Name,
				Backend: resolvedTool,
			})

			payload := map[string]any{
				"artifact":      artifactPath,
				"profile":       profile,
				"diagnosis":     report,
				"prescription":  prescription,
			}
			if outputPath != "" {
				data, err := json.MarshalIndent(payload, "", "  ")
				if err != nil {
					return err
				}
				data = append(data, '\n')
				if err := os.WriteFile(outputPath, data, 0o600); err != nil {
					return err
				}
				cmd.Printf("Saved profile analysis: %s\n", outputPath)
			}

			cmd.Printf("Tool: %s\n", profile.Tool)
			cmd.Printf("Category: %s\n", report.Category)
			cmd.Printf("Efficiency: %.2f%%\n", report.Efficiency*100)
			cmd.Printf("Confidence: %.2f\n", report.Confidence)
			cmd.Printf("Summary: %s\n", report.Summary)
			if len(report.RootCauses) > 0 {
				cmd.Println("Root causes")
				for _, cause := range report.RootCauses {
					cmd.Printf("- %s\n", cause)
				}
			}
			if len(prescription.Fixes) > 0 {
				cmd.Println("Fixes")
				for _, fix := range prescription.Fixes {
					cmd.Printf("- %s\n", fix.Action)
				}
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&artifactPath, "artifact", "", "path to a saved profile artifact JSON")
	cmd.Flags().StringVar(&tool, "tool", "", "override the profiler tool label, for example ncu or nsys")
	cmd.Flags().StringVar(&outputPath, "output", "", "optional output path for the normalized analysis JSON")
	cmd.MarkFlagRequired("artifact")
	return cmd
}
