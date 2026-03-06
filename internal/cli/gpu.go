package cli

import (
	"strings"

	"github.com/ModelsLab/fusion/internal/system"
	"github.com/spf13/cobra"
)

func newGPUCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "gpu",
		Short: "Inspect detected NVIDIA GPUs",
	}

	cmd.AddCommand(&cobra.Command{
		Use:   "detect",
		Short: "Show NVIDIA GPUs visible to this machine",
		Run: func(cmd *cobra.Command, args []string) {
			gpus := system.DetectNVIDIAGPUs()
			if len(gpus) == 0 {
				cmd.Println("No NVIDIA GPUs detected.")
				cmd.Println("This is expected on the current macOS host; pass `--gpu` when planning for a Linux target.")
				return
			}

			for _, gpu := range gpus {
				cmd.Printf("%s\n", system.FormatGPU(gpu))
			}
		},
	})

	cmd.AddCommand(&cobra.Command{
		Use:   "normalize [gpu name]",
		Short: "Show the GPU name normalized for Fusion lookups",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			runtimeState, err := loadRuntime()
			if err != nil {
				return err
			}

			query := strings.Join(args, " ")
			gpu, ok := runtimeState.KB.GPUByID(query)
			if !ok {
				cmd.Printf("No curated GPU profile matched %q\n", query)
				return nil
			}

			cmd.Printf("%s -> %s (%s, cc %s)\n", query, gpu.ID, gpu.Family, gpu.ComputeCapability)
			return nil
		},
	})

	return cmd
}
