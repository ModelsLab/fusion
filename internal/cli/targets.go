package cli

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/ModelsLab/fusion/internal/config"
	"github.com/ModelsLab/fusion/internal/runner"
	"github.com/ModelsLab/fusion/internal/targets"
	"github.com/spf13/cobra"
)

func newTargetCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "target",
		Short: "Manage local, SSH, and proxy simulation targets",
	}

	cmd.AddCommand(
		newTargetAddCommand(),
		newTargetListCommand(),
		newTargetShowCommand(),
		newTargetRemoveCommand(),
		newTargetDefaultCommand(),
		newTargetExecCommand(),
		newTargetCopyCommand(),
	)

	return cmd
}

func newTargetAddCommand() *cobra.Command {
	var target config.TargetConfig
	var makeDefault bool

	cmd := &cobra.Command{
		Use:   "add",
		Short: "Register a local, SSH, or sim target",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			target.Mode = targets.Normalize(target.Mode)
			if target.Mode == targets.ModeLocal && target.Name == "" {
				target.Name = "local"
			}
			if target.Port == 0 && (target.Mode == targets.ModeSSH || target.Mode == targets.ModeSim) {
				target.Port = 22
			}

			validation := targets.Validate(target, runtimeState.KB)
			if len(validation.Errors) > 0 {
				return fmt.Errorf("%s", strings.Join(validation.Errors, "; "))
			}

			if err := runtimeState.Config.SetTarget(target); err != nil {
				return err
			}
			if makeDefault {
				if err := runtimeState.Config.SetDefaultTarget(target.Name); err != nil {
					return err
				}
			}

			cmd.Printf("Saved target %s (%s)\n", target.Name, target.Mode)
			cmd.Printf("Execution: %s\n", targets.ExecutionSummary(target))
			for _, warning := range validation.Warnings {
				cmd.Printf("Warning: %s\n", warning)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&target.Name, "name", "", "target name")
	cmd.Flags().StringVar(&target.Mode, "mode", "", "target mode: local, ssh, sim")
	cmd.Flags().StringVar(&target.Description, "description", "", "short description for this target")
	cmd.Flags().StringVar(&target.GPU, "gpu", "", "declared target GPU; for sim targets this is the intended GPU being approximated")
	cmd.Flags().StringVar(&target.ProxyGPU, "proxy-gpu", "", "actual proxy GPU used for sim mode")
	cmd.Flags().StringVar(&target.Host, "host", "", "SSH host for remote targets")
	cmd.Flags().StringVar(&target.User, "user", "", "SSH user")
	cmd.Flags().IntVar(&target.Port, "port", 22, "SSH port")
	cmd.Flags().StringVar(&target.IdentityFile, "key", "", "SSH identity file")
	cmd.Flags().StringVar(&target.RemoteDir, "remote-dir", "", "remote working directory for exec/copy")
	cmd.Flags().StringVar(&target.Shell, "shell", "", "override shell for local or remote command execution")
	cmd.Flags().BoolVar(&makeDefault, "default", false, "set this target as the default")
	cmd.MarkFlagRequired("mode")
	return cmd
}

func newTargetListCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List configured targets",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}
			cfg, err := runtimeState.Config.Load()
			if err != nil {
				return err
			}

			if len(cfg.Targets) == 0 {
				cmd.Println("No targets configured. Add one with `fusion target add`.")
				return nil
			}

			names := make([]string, 0, len(cfg.Targets))
			for name := range cfg.Targets {
				names = append(names, name)
			}
			sort.Strings(names)

			for _, name := range names {
				target := cfg.Targets[name]
				marker := ""
				if cfg.DefaultTarget == name {
					marker = " [default]"
				}
				cmd.Printf("%s%s\n", name, marker)
				cmd.Printf("  mode: %s\n", target.Mode)
				cmd.Printf("  execution: %s\n", targets.ExecutionSummary(target))
				if target.Description != "" {
					cmd.Printf("  description: %s\n", target.Description)
				}
			}

			return nil
		},
	}
}

func newTargetShowCommand() *cobra.Command {
	var name string

	cmd := &cobra.Command{
		Use:   "show",
		Short: "Show one configured target",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			target, _, err := resolveTarget(runtimeState, name)
			if err != nil {
				return err
			}

			cmd.Printf("%s\n", target.Name)
			cmd.Printf("mode: %s\n", target.Mode)
			cmd.Printf("execution: %s\n", targets.ExecutionSummary(target))
			if target.Description != "" {
				cmd.Printf("description: %s\n", target.Description)
			}
			if target.GPU != "" {
				cmd.Printf("gpu: %s\n", target.GPU)
			}
			if target.ProxyGPU != "" {
				cmd.Printf("proxy gpu: %s\n", target.ProxyGPU)
			}
			if target.Host != "" {
				cmd.Printf("host: %s\n", target.Host)
			}
			if target.User != "" {
				cmd.Printf("user: %s\n", target.User)
			}
			if target.Port != 0 {
				cmd.Printf("port: %d\n", target.Port)
			}
			if target.RemoteDir != "" {
				cmd.Printf("remote dir: %s\n", target.RemoteDir)
			}

			for _, warning := range targets.Warnings(target) {
				cmd.Printf("warning: %s\n", warning)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&name, "name", "", "target name; defaults to the configured default target")
	return cmd
}

func newTargetRemoveCommand() *cobra.Command {
	var name string

	cmd := &cobra.Command{
		Use:   "remove",
		Short: "Remove a configured target",
		RunE: func(cmd *cobra.Command, args []string) error {
			if strings.TrimSpace(name) == "" {
				return fmt.Errorf("target name is required")
			}
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			if err := runtimeState.Config.RemoveTarget(name); err != nil {
				return err
			}

			cmd.Printf("Removed target %s\n", name)
			return nil
		},
	}

	cmd.Flags().StringVar(&name, "name", "", "target name")
	cmd.MarkFlagRequired("name")
	return cmd
}

func newTargetDefaultCommand() *cobra.Command {
	var name string

	cmd := &cobra.Command{
		Use:   "default",
		Short: "Set the default target",
		RunE: func(cmd *cobra.Command, args []string) error {
			if strings.TrimSpace(name) == "" {
				return fmt.Errorf("target name is required")
			}
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			cfg, err := runtimeState.Config.Load()
			if err != nil {
				return err
			}
			if _, ok := cfg.Targets[name]; !ok {
				return fmt.Errorf("target %s is not configured", name)
			}

			if err := runtimeState.Config.SetDefaultTarget(name); err != nil {
				return err
			}

			cmd.Printf("%s is now the default target\n", name)
			return nil
		},
	}

	cmd.Flags().StringVar(&name, "name", "", "target name")
	cmd.MarkFlagRequired("name")
	return cmd
}

func newTargetExecCommand() *cobra.Command {
	var name string
	var command string
	var timeout time.Duration

	cmd := &cobra.Command{
		Use:   "exec",
		Short: "Run a shell command on a configured target",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			target, _, err := resolveTarget(runtimeState, name)
			if err != nil {
				return err
			}
			env, err := runtimeShellEnv(runtimeState)
			if err != nil {
				return err
			}

			result, err := runner.Execute(runner.Request{
				Target:  target,
				Command: command,
				Env:     env,
				Timeout: timeout,
			})
			if result.Stdout != "" {
				cmd.Printf("%s\n", result.Stdout)
			}
			if result.Stderr != "" {
				cmd.Printf("stderr:\n%s\n", result.Stderr)
			}
			for _, warning := range result.Warnings {
				cmd.Printf("warning: %s\n", warning)
			}
			if err != nil {
				return err
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&name, "name", "", "target name; defaults to the configured default target or implicit local")
	cmd.Flags().StringVar(&command, "command", "", "shell command to run")
	cmd.Flags().DurationVar(&timeout, "timeout", 30*time.Minute, "command timeout")
	cmd.MarkFlagRequired("command")
	return cmd
}

func newTargetCopyCommand() *cobra.Command {
	var name string
	var src string
	var dst string
	var recursive bool
	var timeout time.Duration

	cmd := &cobra.Command{
		Use:   "copy",
		Short: "Copy files to a configured target",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			target, _, err := resolveTarget(runtimeState, name)
			if err != nil {
				return err
			}

			result, err := runner.Copy(runner.CopyRequest{
				Target:    target,
				Source:    src,
				Dest:      dst,
				Recursive: recursive,
				Timeout:   timeout,
			})
			if err != nil {
				return err
			}

			cmd.Printf("Copied %s -> %s via %s\n", result.Source, result.Dest, result.ExecutionMode)
			for _, warning := range result.Warnings {
				cmd.Printf("warning: %s\n", warning)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&name, "name", "", "target name; defaults to the configured default target or implicit local")
	cmd.Flags().StringVar(&src, "src", "", "source path")
	cmd.Flags().StringVar(&dst, "dst", "", "destination path")
	cmd.Flags().BoolVar(&recursive, "recursive", false, "copy directories recursively")
	cmd.Flags().DurationVar(&timeout, "timeout", 15*time.Minute, "copy timeout")
	cmd.MarkFlagRequired("src")
	cmd.MarkFlagRequired("dst")
	return cmd
}
