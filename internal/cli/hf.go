package cli

import (
	"context"
	"encoding/json"
	"errors"
	"sort"
	"strings"
	"time"

	"github.com/ModelsLab/fusion/internal/huggingface"
	"github.com/spf13/cobra"
)

func newHuggingFaceCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "hf",
		Aliases: []string{"huggingface"},
		Short:   "Manage Hugging Face credentials for model download and upload workflows",
	}

	cmd.AddCommand(
		newHuggingFaceLoginCommand(),
		newHuggingFaceShowCommand(),
		newHuggingFaceWhoAmICommand(),
		newHuggingFaceLogoutCommand(),
	)
	return cmd
}

func newHuggingFaceLoginCommand() *cobra.Command {
	var token string

	cmd := &cobra.Command{
		Use:   "login",
		Short: "Store a Hugging Face token locally for Fusion shell tools and workflows",
		RunE: func(cmd *cobra.Command, args []string) error {
			token = strings.TrimSpace(token)
			if token == "" {
				token = huggingface.TokenFromEnv()
			}
			if token == "" {
				return errors.New("token is required via --token, HF_TOKEN, or HUGGING_FACE_HUB_TOKEN")
			}

			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}
			if err := runtimeState.Config.SetHuggingFaceToken(token); err != nil {
				return err
			}

			cmd.Printf("Stored Hugging Face token in %s\n", runtimeState.Config.Path())
			cmd.Printf("Token URL: %s\n", huggingface.TokensURL())
			return nil
		},
	}

	cmd.Flags().StringVar(&token, "token", "", "Hugging Face token; falls back to HF_TOKEN and HUGGING_FACE_HUB_TOKEN")
	return cmd
}

func newHuggingFaceShowCommand() *cobra.Command {
	return &cobra.Command{
		Use:     "show",
		Aliases: []string{"status"},
		Short:   "Show the current Hugging Face token status",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}
			token, err := resolveHuggingFaceToken(runtimeState)
			if err != nil {
				return err
			}

			cmd.Printf("Config path: %s\n", runtimeState.Config.Path())
			cmd.Printf("Provider: %s\n", huggingface.Name)
			cmd.Printf("Web URL: %s\n", huggingface.WebBaseURL())
			cmd.Printf("Token URL: %s\n", huggingface.TokensURL())
			cmd.Printf("Token: %s\n", valueOrFallback(maskToken(token), "unset"))
			cmd.Printf("Shell env vars: %s\n", strings.Join(sortedKeys(huggingface.ShellEnv("set")), ", "))
			return nil
		},
	}
}

func newHuggingFaceWhoAmICommand() *cobra.Command {
	var timeout time.Duration

	cmd := &cobra.Command{
		Use:   "whoami",
		Short: "Validate the stored Hugging Face token against the whoami API",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}
			token, err := resolveHuggingFaceToken(runtimeState)
			if err != nil {
				return err
			}
			if strings.TrimSpace(token) == "" {
				return errors.New("no Hugging Face token is configured; run `fusion hf login --token ...` first")
			}

			ctx, cancel := context.WithTimeout(context.Background(), timeout)
			defer cancel()

			payload, err := huggingface.WhoAmI(ctx, token)
			if err != nil {
				return err
			}
			output, err := json.MarshalIndent(payload, "", "  ")
			if err != nil {
				return err
			}
			cmd.Println(string(output))
			return nil
		},
	}

	cmd.Flags().DurationVar(&timeout, "timeout", 30*time.Second, "how long to wait for the Hugging Face whoami API")
	return cmd
}

func newHuggingFaceLogoutCommand() *cobra.Command {
	return &cobra.Command{
		Use:     "logout",
		Aliases: []string{"remove", "clear"},
		Short:   "Remove the stored Hugging Face token from local config",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}
			if err := runtimeState.Config.ClearHuggingFace(); err != nil {
				return err
			}
			cmd.Printf("Cleared Hugging Face token from %s\n", runtimeState.Config.Path())
			return nil
		},
	}
}

func sortedKeys(values map[string]string) []string {
	if len(values) == 0 {
		return nil
	}
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
