package cli

import (
	"github.com/ModelsLab/fusion/internal/sshkeys"
	"github.com/spf13/cobra"
)

func newGenerateCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "generate",
		Short: "Generate reusable Fusion assets such as SSH key material",
	}

	cmd.AddCommand(newGenerateKeychainCommand())
	return cmd
}

func newGenerateKeychainCommand() *cobra.Command {
	var name string
	var comment string
	var output string
	var overwrite bool
	var addToAgent bool

	cmd := &cobra.Command{
		Use:   "keychain",
		Short: "Generate an SSH keypair for GPU providers and optionally add it to the local ssh-agent/keychain",
		RunE: func(cmd *cobra.Command, args []string) error {
			key, err := sshkeys.GenerateEd25519(sshkeys.GenerateRequest{
				Name:      name,
				Comment:   comment,
				Output:    output,
				Overwrite: overwrite,
			})
			if err != nil {
				return err
			}

			cmd.Printf("Generated SSH keypair\n")
			cmd.Printf("Private key: %s\n", key.PrivatePath)
			cmd.Printf("Public key:  %s\n", key.PublicPath)
			cmd.Printf("Comment:     %s\n", key.Comment)

			if addToAgent {
				note, agentErr := sshkeys.AddToAgent(key.PrivatePath)
				if note != "" {
					cmd.Printf("Agent:       %s\n", note)
				}
				if agentErr != nil {
					cmd.Printf("warning: %v\n", agentErr)
				}
			}

			cmd.Println("\nPublic key to paste into your GPU provider")
			cmd.Println(key.PublicKey)
			cmd.Println("\nExample target registration")
			cmd.Printf("fusion target add --name my-gpu --mode ssh --host <host> --user <user> --key %s --gpu <gpu>\n", key.PrivatePath)
			return nil
		},
	}

	cmd.Flags().StringVar(&name, "name", "default", "short key name used in the generated filename")
	cmd.Flags().StringVar(&comment, "comment", "", "optional SSH public key comment")
	cmd.Flags().StringVar(&output, "output", "", "private key output path; defaults to ~/.ssh/fusion-<name>-ed25519")
	cmd.Flags().BoolVar(&overwrite, "overwrite", false, "replace existing key files if they already exist")
	cmd.Flags().BoolVar(&addToAgent, "agent", true, "attempt to add the private key to the local ssh-agent or platform keychain")
	return cmd
}
