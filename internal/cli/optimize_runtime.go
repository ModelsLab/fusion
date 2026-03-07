package cli

import (
	"encoding/json"
	"os"

	"github.com/ModelsLab/fusion/internal/optimize"
	"github.com/spf13/cobra"
)

func newOptimizeRuntimeCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "runtime",
		Short: "Inspect and patch runtime code through generic transaction-based tools",
	}
	cmd.AddCommand(
		newOptimizeRuntimeDetectCommand(),
		newOptimizeRuntimeApplyCommand(),
		newOptimizeRuntimeRevertCommand(),
	)
	return cmd
}

func newOptimizeRuntimeDetectCommand() *cobra.Command {
	var root string
	cmd := &cobra.Command{
		Use:   "detect",
		Short: "Detect likely runtime environments from the checked-out project",
		RunE: func(cmd *cobra.Command, args []string) error {
			if root == "" {
				var err error
				root, err = os.Getwd()
				if err != nil {
					return err
				}
			}
			detections, err := optimize.DetectRuntimeAdapters(root)
			if err != nil {
				return err
			}
			for _, detection := range detections {
				cmd.Printf("- %s: matched=%t confidence=%.2f\n", detection.Adapter, detection.Matched, detection.Confidence)
				for _, reason := range detection.Reasons {
					cmd.Printf("  reason: %s\n", reason)
				}
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&root, "root", "", "project root; defaults to the current working directory")
	return cmd
}

func newOptimizeRuntimeApplyCommand() *cobra.Command {
	var planPath string
	cmd := &cobra.Command{
		Use:   "apply",
		Short: "Apply a generic runtime patch transaction from a JSON plan",
		RunE: func(cmd *cobra.Command, args []string) error {
			var plan optimize.RuntimePatchPlan
			data, err := os.ReadFile(planPath)
			if err != nil {
				return err
			}
			if err := json.Unmarshal(data, &plan); err != nil {
				return err
			}
			state, statePath, err := optimize.ApplyRuntimePatch(plan)
			if err != nil {
				return err
			}
			cmd.Printf("Applied runtime patch state: %s\n", statePath)
			cmd.Printf("Adapter: %s\n", state.Adapter)
			cmd.Printf("Files touched: %d\n", len(state.Records))
			return nil
		},
	}
	cmd.Flags().StringVar(&planPath, "plan", "", "path to a runtime patch plan JSON")
	cmd.MarkFlagRequired("plan")
	return cmd
}

func newOptimizeRuntimeRevertCommand() *cobra.Command {
	var statePath string
	cmd := &cobra.Command{
		Use:   "revert",
		Short: "Revert a previously applied generic runtime patch transaction",
		RunE: func(cmd *cobra.Command, args []string) error {
			state, err := optimize.RevertRuntimePatch(statePath)
			if err != nil {
				return err
			}
			cmd.Printf("Reverted runtime patch state: %s\n", statePath)
			cmd.Printf("Adapter: %s\n", state.Adapter)
			cmd.Printf("Files restored: %d\n", len(state.Records))
			return nil
		},
	}
	cmd.Flags().StringVar(&statePath, "state", "", "path to a runtime patch state JSON")
	cmd.MarkFlagRequired("state")
	return cmd
}
