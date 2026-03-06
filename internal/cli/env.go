package cli

import (
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
				return
			}

			for _, gpu := range env.GPUs {
				cmd.Printf("- %s\n", system.FormatGPU(gpu))
			}
		},
	})

	return cmd
}
