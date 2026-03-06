package cli

import (
	"context"
	"encoding/json"
	"errors"
	"sort"
	"strings"
	"time"

	"github.com/ModelsLab/fusion/internal/githubauth"
	"github.com/spf13/cobra"
)

func newGitHubCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "github",
		Aliases: []string{"gh"},
		Short:   "Manage GitHub credentials for private repo access and automation",
	}

	cmd.AddCommand(
		newGitHubLoginCommand(),
		newGitHubShowCommand(),
		newGitHubWhoAmICommand(),
		newGitHubLogoutCommand(),
	)
	return cmd
}

func newGitHubLoginCommand() *cobra.Command {
	var token string

	cmd := &cobra.Command{
		Use:   "login",
		Short: "Store a GitHub personal access token for Fusion shell tools and workflows",
		RunE: func(cmd *cobra.Command, args []string) error {
			token = strings.TrimSpace(token)
			if token == "" {
				token = githubauth.TokenFromEnv()
			}
			if token == "" {
				return errors.New("token is required via --token, GITHUB_TOKEN, or GH_TOKEN")
			}

			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}
			if err := runtimeState.Config.SetGitHubToken(token); err != nil {
				return err
			}

			cmd.Printf("Stored GitHub token in %s\n", runtimeState.Config.Path())
			return nil
		},
	}

	cmd.Flags().StringVar(&token, "token", "", "GitHub personal access token; falls back to GITHUB_TOKEN and GH_TOKEN")
	return cmd
}

func newGitHubShowCommand() *cobra.Command {
	return &cobra.Command{
		Use:     "show",
		Aliases: []string{"status"},
		Short:   "Show the current GitHub token status",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}
			token, err := resolveGitHubToken(runtimeState)
			if err != nil {
				return err
			}

			cmd.Printf("Config path: %s\n", runtimeState.Config.Path())
			cmd.Printf("Provider: %s\n", githubauth.Name)
			cmd.Printf("API URL: %s\n", githubauth.APIBaseURL())
			cmd.Printf("Token: %s\n", valueOrFallback(maskToken(token), "unset"))
			cmd.Printf("Shell env vars: %s\n", strings.Join(sortedShellKeys(githubauth.ShellEnv("set")), ", "))
			return nil
		},
	}
}

func newGitHubWhoAmICommand() *cobra.Command {
	var timeout time.Duration

	cmd := &cobra.Command{
		Use:   "whoami",
		Short: "Validate the stored GitHub token against the GitHub user API",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}
			token, err := resolveGitHubToken(runtimeState)
			if err != nil {
				return err
			}
			if strings.TrimSpace(token) == "" {
				return errors.New("no GitHub token is configured; run `fusion github login --token ...` first")
			}

			ctx, cancel := context.WithTimeout(context.Background(), timeout)
			defer cancel()

			payload, err := githubauth.WhoAmI(ctx, token)
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

	cmd.Flags().DurationVar(&timeout, "timeout", 30*time.Second, "how long to wait for the GitHub user API")
	return cmd
}

func newGitHubLogoutCommand() *cobra.Command {
	return &cobra.Command{
		Use:     "logout",
		Aliases: []string{"remove", "clear"},
		Short:   "Remove the stored GitHub token from local config",
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}
			if err := runtimeState.Config.ClearGitHub(); err != nil {
				return err
			}
			cmd.Printf("Cleared GitHub token from %s\n", runtimeState.Config.Path())
			return nil
		},
	}
}

func sortedShellKeys(values map[string]string) []string {
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
