package cli

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ModelsLab/fusion/internal/kb"
	embedded "github.com/ModelsLab/fusion/knowledge"
	"github.com/spf13/cobra"
)

func newUpdateCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "update",
		Short: "Refresh generated Fusion assets such as the local knowledge base",
	}

	cmd.AddCommand(newUpdateKnowledgeCommand())
	return cmd
}

func newUpdateKnowledgeCommand() *cobra.Command {
	var sourceRoot string
	var outputRoot string
	var bootstrap bool

	cmd := &cobra.Command{
		Use:   "kb",
		Short: "Rebuild Fusion's knowledge SQLite index from Markdown knowledge docs",
		RunE: func(cmd *cobra.Command, args []string) error {
			sourceRoot = strings.TrimSpace(sourceRoot)
			if sourceRoot == "" {
				configDir, err := os.UserConfigDir()
				if err != nil {
					return fmt.Errorf("resolve user config dir: %w", err)
				}
				sourceRoot = filepath.Join(configDir, "fusion", "knowledgebase")
			}
			sourceRoot, err := filepath.Abs(sourceRoot)
			if err != nil {
				return fmt.Errorf("resolve markdown knowledge root: %w", err)
			}

			outputRoot = strings.TrimSpace(outputRoot)
			if outputRoot == "" {
				configDir, err := os.UserConfigDir()
				if err != nil {
					return fmt.Errorf("resolve user config dir: %w", err)
				}
				outputRoot = filepath.Join(configDir, "fusion", "knowledge")
			}
			outputRoot, err := filepath.Abs(outputRoot)
			if err != nil {
				return fmt.Errorf("resolve generated knowledge output root: %w", err)
			}

			hasMarkdown, err := kb.HasMarkdownFiles(sourceRoot)
			if err != nil {
				return err
			}
			if !hasMarkdown && !bootstrap {
				return fmt.Errorf("no markdown knowledge docs found in %s; create .md files there or rerun with --bootstrap", sourceRoot)
			}

			var fallback *kb.Store
			if !hasMarkdown {
				fallback, err = kb.LoadFromFS(embedded.Files)
				if err != nil {
					return err
				}
			}

			store, err := kb.BuildMarkdownBackedStore(sourceRoot, fallback)
			if err != nil {
				return err
			}

			dbPath := filepath.Join(outputRoot, "knowledge.db")
			if err := kb.WriteSQLiteIndex(store, dbPath); err != nil {
				return err
			}

			cmd.Printf("Markdown knowledge root: %s\n", sourceRoot)
			cmd.Printf("Generated knowledge root: %s\n", outputRoot)
			cmd.Printf("SQLite index: %s\n", dbPath)
			cmd.Printf("Counts: %d sources, %d GPUs, %d strategies, %d skills, %d examples, %d documents\n", len(store.Sources), len(store.GPUs), len(store.Strategies), len(store.Skills), len(store.Examples), len(store.Documents))
			cmd.Println("Future Fusion runs will prefer this rebuilt local knowledge base over the embedded default.")
			return nil
		},
	}

	cmd.Flags().StringVar(&sourceRoot, "source-root", "", "Markdown knowledge root; defaults to ~/.config/fusion/knowledgebase")
	cmd.Flags().StringVar(&outputRoot, "output-root", "", "generated SQLite output root; defaults to ~/.config/fusion/knowledge")
	cmd.Flags().BoolVar(&bootstrap, "bootstrap", true, "bootstrap the Markdown knowledge tree from the embedded curated knowledge when the source root is empty")
	return cmd
}
