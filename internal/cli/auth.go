package cli

import (
	"errors"
	"os"
	"strings"
	"time"

	"github.com/ModelsLab/fusion/internal/modelslab"
	"github.com/spf13/cobra"
)

func newAuthCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "auth",
		Short: "Manage ModelsLab authentication for Fusion",
	}

	cmd.AddCommand(
		newModelsLabLoginCommand("login", "Open the browser and connect Fusion to your ModelsLab account"),
		newAuthShowCommand(),
		newAuthSetCommand(),
		newAuthLogoutCommand(),
	)

	return cmd
}

func newAuthShowCommand() *cobra.Command {
	return &cobra.Command{
		Use:     "show",
		Aliases: []string{"list", "status"},
		Short:   "Show the current ModelsLab configuration",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			cfg, err := runtimeState.Config.Load()
			if err != nil {
				return err
			}

			token := strings.TrimSpace(cfg.ModelsLab.Token)
			if token == "" {
				token = strings.TrimSpace(os.Getenv(modelslab.TokenEnvVar))
			}

			model := strings.TrimSpace(cfg.ModelsLab.Model)
			if model == "" {
				model = strings.TrimSpace(os.Getenv(modelslab.ModelEnvVar))
			}
			if model == "" {
				model = modelslab.DefaultModelID
			}

			cmd.Printf("Config path: %s\n", runtimeState.Config.Path())
			cmd.Printf("Provider: %s\n", modelslab.Name)
			cmd.Printf("API URL: %s\n", modelslab.APIBaseURL())
			cmd.Printf("Web URL: %s\n", modelslab.WebBaseURL())
			cmd.Printf("Token: %s\n", valueOrFallback(maskToken(token), "unset"))
			cmd.Printf("Model: %s\n", model)
			return nil
		},
	}
}

func newAuthSetCommand() *cobra.Command {
	var token string
	var model string

	cmd := &cobra.Command{
		Use:   "set",
		Short: "Store a ModelsLab API key locally without using the browser login flow",
		RunE: func(cmd *cobra.Command, args []string) error {
			token = strings.TrimSpace(token)
			if token == "" {
				token = strings.TrimSpace(os.Getenv(modelslab.TokenEnvVar))
			}
			if token == "" {
				return errors.New("token is required via --token or MODELSLAB_API_KEY")
			}

			model = strings.TrimSpace(model)
			if model == "" {
				model = strings.TrimSpace(os.Getenv(modelslab.ModelEnvVar))
			}
			if model == "" {
				model = modelslab.DefaultModelID
			}

			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			if err := runtimeState.Config.SetModelsLabToken(token); err != nil {
				return err
			}
			if err := runtimeState.Config.SetModelsLabModel(model); err != nil {
				return err
			}

			cmd.Printf("Stored ModelsLab credentials in %s\n", runtimeState.Config.Path())
			cmd.Printf("Default model: %s\n", model)
			return nil
		},
	}

	cmd.Flags().StringVar(&token, "token", "", "ModelsLab API key; falls back to MODELSLAB_API_KEY")
	cmd.Flags().StringVar(&model, "model", "", "default model id; falls back to MODELSLAB_MODEL_ID and then openai-gpt-5.4-pro")
	return cmd
}

func newAuthLogoutCommand() *cobra.Command {
	return &cobra.Command{
		Use:     "logout",
		Aliases: []string{"remove", "clear"},
		Short:   "Remove the stored ModelsLab API key and default model from local config",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			if err := runtimeState.Config.ClearModelsLab(); err != nil {
				return err
			}

			cmd.Printf("Cleared ModelsLab credentials from %s\n", runtimeState.Config.Path())
			return nil
		},
	}
}

func newLoginCommand() *cobra.Command {
	return newModelsLabLoginCommand("login", "Open the browser and connect Fusion to your ModelsLab account")
}

func newModelsLabLoginCommand(use string, short string) *cobra.Command {
	var opts modelslabLoginOptions

	cmd := &cobra.Command{
		Use:   use,
		Short: short,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runModelsLabBrowserLogin(cmd, opts)
		},
	}

	cmd.Flags().StringVar(&opts.Model, "model", "", "model id to store after login; defaults to openai-gpt-5.4-pro")
	cmd.Flags().DurationVar(&opts.Timeout, "timeout", 3*time.Minute, "how long to wait for browser login to complete")
	cmd.Flags().StringVar(&opts.WebURL, "web-url", "", "override the ModelsLab web origin used for browser login")
	cmd.Flags().BoolVar(&opts.NoBrowser, "no-browser", false, "print the login URL instead of opening it automatically")
	return cmd
}
