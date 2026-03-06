package cli

import (
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/ModelsLab/fusion/internal/providers"
	"github.com/spf13/cobra"
)

func newAuthCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "auth",
		Short: "Manage provider tokens for OpenAI, Anthropic, and compatible APIs",
	}

	cmd.AddCommand(
		newAuthProvidersCommand(),
		newAuthListCommand(),
		newAuthSetCommand(),
		newAuthRemoveCommand(),
		newAuthDefaultCommand(),
	)

	return cmd
}

func newAuthProvidersCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "providers",
		Short: "List supported model providers",
		Run: func(cmd *cobra.Command, args []string) {
			for _, provider := range providers.All() {
				cmd.Printf("%s (%s)\n", provider.Name, provider.ID)
				cmd.Printf("  base URL: %s\n", provider.BaseURL)
				cmd.Printf("  token env: %s\n", provider.TokenEnvVar)
				cmd.Printf("  docs: %s\n", provider.DocsURL)
				cmd.Printf("  notes: %s\n\n", provider.Notes)
			}
		},
	}
}

func newAuthListCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "Show configured provider tokens",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			cfg, err := runtimeState.Config.Load()
			if err != nil {
				return err
			}

			if len(cfg.Providers) == 0 {
				cmd.Printf("No providers configured yet. Config path: %s\n", runtimeState.Config.Path())
				return nil
			}

			keys := make([]string, 0, len(cfg.Providers))
			for key := range cfg.Providers {
				keys = append(keys, key)
			}
			sort.Strings(keys)

			cmd.Printf("Config path: %s\n", runtimeState.Config.Path())
			for _, key := range keys {
				providerConfig := cfg.Providers[key]
				defaultMarker := ""
				if cfg.DefaultProvider == key {
					defaultMarker = " [default]"
				}
				cmd.Printf("%s%s\n", key, defaultMarker)
				cmd.Printf("  token: %s\n", maskToken(providerConfig.Token))
			}

			return nil
		},
	}
}

func newAuthSetCommand() *cobra.Command {
	var providerID string
	var token string
	var makeDefault bool

	cmd := &cobra.Command{
		Use:   "set",
		Short: "Store a provider token under the local Fusion config",
		RunE: func(cmd *cobra.Command, args []string) error {
			provider, ok := providers.Lookup(providerID)
			if !ok {
				return fmt.Errorf("unsupported provider %q", providerID)
			}

			if strings.TrimSpace(token) == "" {
				token = os.Getenv(provider.TokenEnvVar)
			}
			if strings.TrimSpace(token) == "" {
				return errors.New("token is required via --token or the provider's environment variable")
			}

			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			if err := runtimeState.Config.SetToken(provider.ID, token); err != nil {
				return err
			}
			if makeDefault {
				if err := runtimeState.Config.SetDefaultProvider(provider.ID); err != nil {
					return err
				}
			}

			cmd.Printf("Stored token for %s in %s\n", provider.ID, runtimeState.Config.Path())
			if makeDefault {
				cmd.Printf("%s is now the default provider\n", provider.ID)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&providerID, "provider", "", "provider id (openai, anthropic, openrouter, google, groq)")
	cmd.Flags().StringVar(&token, "token", "", "provider token; falls back to the provider env var when omitted")
	cmd.Flags().BoolVar(&makeDefault, "default", false, "set this provider as the default")
	cmd.MarkFlagRequired("provider")
	return cmd
}

func newAuthRemoveCommand() *cobra.Command {
	var providerID string

	cmd := &cobra.Command{
		Use:   "remove",
		Short: "Remove a configured provider token",
		RunE: func(cmd *cobra.Command, args []string) error {
			provider, ok := providers.Lookup(providerID)
			if !ok {
				return fmt.Errorf("unsupported provider %q", providerID)
			}

			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			if err := runtimeState.Config.RemoveProvider(provider.ID); err != nil {
				return err
			}

			cmd.Printf("Removed %s from %s\n", provider.ID, runtimeState.Config.Path())
			return nil
		},
	}

	cmd.Flags().StringVar(&providerID, "provider", "", "provider id")
	cmd.MarkFlagRequired("provider")
	return cmd
}

func newAuthDefaultCommand() *cobra.Command {
	var providerID string

	cmd := &cobra.Command{
		Use:   "default",
		Short: "Set the default provider used by future agent flows",
		RunE: func(cmd *cobra.Command, args []string) error {
			provider, ok := providers.Lookup(providerID)
			if !ok {
				return fmt.Errorf("unsupported provider %q", providerID)
			}

			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			cfg, err := runtimeState.Config.Load()
			if err != nil {
				return err
			}
			if _, exists := cfg.Providers[provider.ID]; !exists {
				return fmt.Errorf("provider %s is not configured yet; run `fusion auth set --provider %s` first", provider.ID, provider.ID)
			}

			if err := runtimeState.Config.SetDefaultProvider(provider.ID); err != nil {
				return err
			}

			cmd.Printf("%s is now the default provider\n", provider.ID)
			return nil
		},
	}

	cmd.Flags().StringVar(&providerID, "provider", "", "provider id")
	cmd.MarkFlagRequired("provider")
	return cmd
}
