package cli

import (
	"github.com/spf13/cobra"
)

const version = "0.1.0"

func NewRootCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:           "fusion",
		Short:         "Fusion plans GPU inference optimizations and manages an embedded kernel knowledge base.",
		SilenceUsage:  true,
		SilenceErrors: true,
	}

	cmd.AddCommand(
		newAuthCommand(),
		newEnvCommand(),
		newGPUCommand(),
		newKnowledgeCommand(),
		newOptimizeCommand(),
		newVersionCommand(),
	)

	return cmd
}

func newVersionCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Print the Fusion CLI version",
		Run: func(cmd *cobra.Command, args []string) {
			cmd.Println(version)
		},
	}
}
