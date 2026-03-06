package cli

import (
	"time"

	"github.com/ModelsLab/fusion/internal/artifacts"
	"github.com/ModelsLab/fusion/internal/runner"
	"github.com/spf13/cobra"
)

func newProfileCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "profile",
		Short: "Run profiling commands and persist the raw artifacts",
	}

	cmd.AddCommand(newProfileRunCommand())
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
			if runName == "" {
				runName = resolvedName + "-profile"
			}

			runResult, err := runner.Execute(runner.Request{
				Target:  target,
				Command: command,
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
