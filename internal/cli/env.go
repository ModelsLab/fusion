package cli

import (
	"encoding/json"

	"github.com/ModelsLab/fusion/internal/system"
	"github.com/spf13/cobra"
)

func newEnvCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "env",
		Short: "Inspect the local host and toolchain",
	}

	cmd.AddCommand(&cobra.Command{
		Use:   "detect",
		Short: "Detect host OS, NVIDIA tools, and visible GPUs",
		Run: func(cmd *cobra.Command, args []string) {
			env := system.DetectEnvironment()
			capabilities := system.AssessCapabilities(env)
			cmd.Printf("Host: %s/%s\n", env.Host.OS, env.Host.Arch)

			cmd.Println("\nTools")
			for _, tool := range env.Tools {
				state := "missing"
				if tool.Available {
					state = "available"
				}
				cmd.Printf("- %s: %s", tool.Name, state)
				if tool.Version != "" {
					cmd.Printf(" (%s)", tool.Version)
				}
				if tool.Path != "" {
					cmd.Printf(" [%s]", tool.Path)
				}
				cmd.Printf("\n  %s\n", tool.Notes)
			}

			cmd.Println("\nGPUs")
			if len(env.GPUs) == 0 {
				cmd.Println("- no NVIDIA GPU detected on this host")
				cmd.Println("  use `fusion optimize plan --gpu <target>` to plan for a remote Ubuntu box")
			} else {
				for _, gpu := range env.GPUs {
					cmd.Printf("- %s\n", system.FormatGPU(gpu))
				}
			}

			cmd.Println("\nCapabilities")
			cmd.Printf("- planning: %t\n", capabilities.CanPlan)
			cmd.Printf("- target management: %t\n", capabilities.CanManageTargets)
			cmd.Printf("- local command execution: %t\n", capabilities.CanRunLocal)
			cmd.Printf("- local CUDA compilation: %t\n", capabilities.CanCompileCUDA)
			cmd.Printf("- local CUDA profiling: %t\n", capabilities.CanProfileCUDA)
			cmd.Printf("- SSH execution: %t\n", capabilities.CanUseSSH)

			if len(capabilities.Limitations) > 0 {
				cmd.Println("\nLimitations")
				for _, limitation := range capabilities.Limitations {
					cmd.Printf("- %s\n", limitation)
				}
			}

			if len(capabilities.NextSteps) > 0 {
				cmd.Println("\nNext Steps")
				for _, step := range capabilities.NextSteps {
					cmd.Printf("- %s\n", step)
				}
			}
		},
	})

	cmd.AddCommand(newEnvDoctorCommand())

	return cmd
}

func newEnvDoctorCommand() *cobra.Command {
	var backend string
	var jsonOutput bool
	var fixScript bool

	cmd := &cobra.Command{
		Use:   "doctor",
		Short: "Check whether this host is ready for Fusion chat, Triton, CuTe, CUDA, and profiling flows",
		RunE: func(cmd *cobra.Command, args []string) error {
			report := system.RunDoctor(backend)
			if jsonOutput {
				data, err := json.MarshalIndent(report, "", "  ")
				if err != nil {
					return err
				}
				cmd.Println(string(data))
				return nil
			}

			cmd.Printf("Backend: %s\n", report.Backend)
			cmd.Printf("Host: %s/%s\n", report.Host.OS, report.Host.Arch)
			cmd.Printf("Ready: %t\n", report.Ready)

			cmd.Println("\nChecks")
			for _, check := range report.Checks {
				state := "missing"
				if check.OK {
					state = "ok"
				}
				cmd.Printf("- %s: %s [%s]\n", check.Requirement, state, check.Severity)
				if check.Details != "" {
					cmd.Printf("  %s\n", check.Details)
				}
				if !check.OK && check.FixHint != "" {
					cmd.Printf("  fix: %s\n", check.FixHint)
				}
			}

			if len(report.MissingRequired) > 0 {
				cmd.Println("\nMissing Required")
				for _, value := range report.MissingRequired {
					cmd.Printf("- %s\n", value)
				}
			}

			if len(report.MissingRecommended) > 0 {
				cmd.Println("\nMissing Recommended")
				for _, value := range report.MissingRecommended {
					cmd.Printf("- %s\n", value)
				}
			}

			if fixScript {
				cmd.Println("\nFix Script")
				cmd.Println(report.RecommendedFixScript)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&backend, "backend", "all", "backend to validate: all, chat, cuda, triton, cute, or profile")
	cmd.Flags().BoolVar(&jsonOutput, "json", false, "emit the doctor report as JSON")
	cmd.Flags().BoolVar(&fixScript, "fix-script", false, "print a best-effort Ubuntu bootstrap script for missing requirements")
	return cmd
}
