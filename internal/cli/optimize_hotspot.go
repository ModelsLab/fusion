package cli

import (
	"encoding/json"
	"os"

	"github.com/ModelsLab/fusion/internal/optimize"
	"github.com/spf13/cobra"
)

func newOptimizeHotspotCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "hotspot",
		Short: "Infer and persist generic hotspot attribution from kernel names and task context",
	}
	cmd.AddCommand(newOptimizeHotspotInferCommand())
	return cmd
}

func newOptimizeHotspotInferCommand() *cobra.Command {
	var sessionID string
	var candidateID string
	var round int
	var task string
	var runtimeName string
	var workload string
	var kernels []string
	var outputPath string

	cmd := &cobra.Command{
		Use:   "infer",
		Short: "Infer generic hotspot attribution from kernel names",
		RunE: func(cmd *cobra.Command, args []string) error {
			attribution := optimize.InferHotspotAttribution(task, runtimeName, workload, kernels)
			if sessionID != "" && candidateID != "" && round > 0 {
				session, _, err := loadOptimizationSession(sessionID)
				if err != nil {
					return err
				}
				path, err := optimize.SaveHotspotAttribution(session, candidateID, round, attribution)
				if err != nil {
					return err
				}
				cmd.Printf("Saved hotspot attribution: %s\n", path)
				return nil
			}
			data, err := json.MarshalIndent(attribution, "", "  ")
			if err != nil {
				return err
			}
			data = append(data, '\n')
			if outputPath != "" {
				if err := os.WriteFile(outputPath, data, 0o600); err != nil {
					return err
				}
				cmd.Printf("Saved hotspot attribution: %s\n", outputPath)
				return nil
			}
			cmd.Print(string(data))
			return nil
		},
	}

	cmd.Flags().StringVar(&sessionID, "session", "", "optional optimization session id")
	cmd.Flags().StringVar(&candidateID, "candidate", "", "optional candidate id")
	cmd.Flags().IntVar(&round, "round", 0, "optional round number when saving into a session")
	cmd.Flags().StringVar(&task, "task", "", "task family like text-generation, image-generation, image-editing, video-generation, or audio-generation")
	cmd.Flags().StringVar(&runtimeName, "runtime", "", "runtime label")
	cmd.Flags().StringVar(&workload, "workload", "", "workload label")
	cmd.Flags().StringSliceVar(&kernels, "kernel", nil, "kernel name to attribute; repeat as needed")
	cmd.Flags().StringVar(&outputPath, "output", "", "optional output path when not saving into a session")
	cmd.MarkFlagRequired("kernel")
	return cmd
}
