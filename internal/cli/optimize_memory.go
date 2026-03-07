package cli

import (
	"strings"

	"github.com/ModelsLab/fusion/internal/artifacts"
	"github.com/ModelsLab/fusion/internal/optimize"
	"github.com/spf13/cobra"
)

func newOptimizeMemoryCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "memory",
		Short: "Write and refresh generic per-session markdown memory",
	}
	cmd.AddCommand(
		newOptimizeMemoryAddCommand(),
		newOptimizeMemoryRefreshCommand(),
	)
	return cmd
}

func newOptimizeMemoryAddCommand() *cobra.Command {
	var sessionID string
	var title string
	var category string
	var summary string
	var outcome string
	var candidateID string
	var lessons []string
	var nextSteps []string
	var files []string
	var metricsText string

	cmd := &cobra.Command{
		Use:   "add",
		Short: "Write a markdown memory entry for an optimization session",
		RunE: func(cmd *cobra.Command, args []string) error {
			session, _, err := loadOptimizationSession(sessionID)
			if err != nil {
				return err
			}
			path, err := optimize.SaveSessionMemoryEntry(session, optimize.SessionMemoryEntry{
				Title:       title,
				Category:    category,
				Summary:     summary,
				Outcome:     outcome,
				CandidateID: candidateID,
				Metrics:     artifacts.ParseMetrics(metricsText),
				Lessons:     append([]string{}, lessons...),
				NextSteps:   append([]string{}, nextSteps...),
				Files:       append([]string{}, files...),
			})
			if err != nil {
				return err
			}
			cmd.Printf("Saved session memory entry: %s\n", path)
			return nil
		},
	}

	cmd.Flags().StringVar(&sessionID, "id", "", "optimization session id")
	cmd.Flags().StringVar(&title, "title", "", "memory entry title")
	cmd.Flags().StringVar(&category, "category", "", "memory category such as benchmark, failure, winner, or environment")
	cmd.Flags().StringVar(&summary, "summary", "", "summary of what happened")
	cmd.Flags().StringVar(&outcome, "outcome", "", "outcome label such as passed, failed, regressed, or winner")
	cmd.Flags().StringVar(&candidateID, "candidate", "", "optional candidate id")
	cmd.Flags().StringSliceVar(&lessons, "lesson", nil, "lesson learned; repeat as needed")
	cmd.Flags().StringSliceVar(&nextSteps, "next-step", nil, "next step to try; repeat as needed")
	cmd.Flags().StringSliceVar(&files, "file", nil, "file path to mention; repeat as needed")
	cmd.Flags().StringVar(&metricsText, "metrics", "", "optional inline metrics like 'tokens_per_sec=42 x_real_time=1.8'")
	cmd.MarkFlagRequired("id")
	cmd.MarkFlagRequired("title")
	cmd.MarkFlagRequired("summary")
	return cmd
}

func newOptimizeMemoryRefreshCommand() *cobra.Command {
	var sessionID string

	cmd := &cobra.Command{
		Use:   "refresh",
		Short: "Refresh the markdown memory index for a session",
		RunE: func(cmd *cobra.Command, args []string) error {
			session, _, err := loadOptimizationSession(sessionID)
			if err != nil {
				return err
			}
			path, err := optimize.RefreshSessionMemoryIndex(session)
			if err != nil {
				return err
			}
			cmd.Printf("Refreshed session memory index: %s\n", path)
			if strings.TrimSpace(session.CurrentBestID) != "" {
				cmd.Printf("Current best candidate: %s\n", session.CurrentBestID)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&sessionID, "id", "", "optimization session id")
	cmd.MarkFlagRequired("id")
	return cmd
}
